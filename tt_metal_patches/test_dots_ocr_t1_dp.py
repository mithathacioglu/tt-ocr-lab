# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-worker dots.ocr runner for distributed N300 setup.

Each worker handles a slice of pages defined by env var WORKER_PAGES (comma list).
Each worker pins to a single N300 board via TT_METAL_PCI_BUS_IDS.
"""

import os
import time
import json

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline


DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
IMAGE_DIR = "/home/mlops/dll_project/ocr_lab"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 2000


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


DOTS_OCR_LOCAL_PATH = _resolve_model_path()


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 300000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # N300 = single board, TP=2
    indirect=True,
)
def test_dots_ocr_worker(mesh_device):
    """Process a worker-assigned slice of t1.pdf pages on one N300 board."""
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    worker_id = int(os.environ.get("WORKER_ID", 0))
    pages_csv = os.environ.get("WORKER_PAGES", "1,2,3")
    page_indices = [int(x) for x in pages_csv.split(",")]
    log_prefix = f"[W{worker_id}]"

    print(f"\n{log_prefix} pages={page_indices} starting...", flush=True)

    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
    )

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    image_paths = [f"{IMAGE_DIR}/t1_page{i}.png" for i in page_indices]

    # Warmup with first page
    print(f"{log_prefix} warmup...", flush=True)
    msgs_warm = [{"role": "user", "content": [
        {"type": "image", "image": image_paths[0]},
        {"type": "text", "text": PROMPT},
    ]}]
    text_w = processor.apply_chat_template(msgs_warm, tokenize=False, add_generation_prompt=True)
    img_w, vid_w = process_vision_info(msgs_warm)
    inp_w = processor(text=[text_w], images=img_w, videos=vid_w, padding=True, return_tensors="pt")
    pipeline.warmup(inp_w["input_ids"],
                    pixel_values=inp_w["pixel_values"].to(torch.bfloat16),
                    image_grid_thw=inp_w["image_grid_thw"])
    print(f"{log_prefix} warmup done", flush=True)

    results = []
    grand_start = time.time()
    for page_num, path in zip(page_indices, image_paths):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": path},
            {"type": "text", "text": PROMPT},
        ]}]
        text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")

        DispatchManager.clear_timings()
        t0 = time.time()
        generated_ids = pipeline.generate(
            inputs["input_ids"],
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            image_grid_thw=inputs["image_grid_thw"],
            max_new_tokens=MAX_NEW_TOKENS,
        )
        ttnn.synchronize_device(mesh_device)
        page_time = time.time() - t0
        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        n_tok = len(generated_ids)

        print(f"{log_prefix} PAGE {page_num}: {page_time:.2f}s, {n_tok} tok", flush=True)
        results.append({"page": page_num, "time_s": page_time, "tokens": n_tok, "text": decoded})

    worker_total = time.time() - grand_start
    print(f"\n{log_prefix} WORKER DONE: {worker_total:.2f}s total for {len(page_indices)} pages", flush=True)
    for r in results:
        print(f"\n{log_prefix} === PAGE {r['page']} OUTPUT ===")
        print(r['text'])
        print(f"{log_prefix} === PAGE {r['page']} END ===", flush=True)

    # Save per-worker JSON
    out = {"worker_id": worker_id, "total_s": worker_total, "results": results}
    out_path = f"/tmp/t1_dp_worker_{worker_id}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{log_prefix} saved {out_path}", flush=True)

    pipeline.release()
