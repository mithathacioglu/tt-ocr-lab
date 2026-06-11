# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run t1.pdf full 9-page OCR sequentially with unlimited tokens."""

import os
import time
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline


MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
}

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
IMAGE_DIR = "/home/mlops/dll_project/ocr_lab"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 2000  # large cap — let EOS terminate naturally


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
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_dots_ocr_t1_full(mesh_device):
    """OCR all 9 pages of t1.pdf sequentially on a single mesh."""
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

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

    image_paths = [f"{IMAGE_DIR}/t1_page{i}.png" for i in range(1, 10)]

    # Warmup once with page 1 (reuse trace across pages of same shape)
    print(f"\n{'='*60}\nWarmup on page 1...\n{'='*60}", flush=True)
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

    # Now run each page
    results = []
    grand_start = time.time()
    for i, path in enumerate(image_paths, 1):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": path},
            {"type": "text", "text": PROMPT},
        ]}]
        text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        image_grid_thw = inputs["image_grid_thw"]

        DispatchManager.clear_timings()
        t0 = time.time()
        generated_ids = pipeline.generate(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        ttnn.synchronize_device(mesh_device)
        page_time = time.time() - t0

        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        n_tok = len(generated_ids)
        ms_tok = page_time / n_tok * 1000 if n_tok else 0

        print(f"\n--- PAGE {i}/9 ({page_time:.2f}s, {n_tok} tok, {ms_tok:.1f} ms/tok) ---")
        print(decoded)
        results.append({"page": i, "time_s": page_time, "tokens": n_tok, "ms_per_tok": ms_tok, "text": decoded})

    total = time.time() - grand_start
    print(f"\n{'='*60}")
    print(f"9-PAGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time:           {total:.2f} s")
    print(f"Total tokens:         {sum(r['tokens'] for r in results)}")
    print(f"Avg time per page:    {total/9:.2f} s")
    print(f"Per-page times:       {[round(r['time_s'], 2) for r in results]}")
    print(f"Per-page tokens:      {[r['tokens'] for r in results]}")
    print(f"{'='*60}\n")

    pipeline.release()
    assert all(len(r["text"].strip()) > 0 for r in results), "Some pages returned empty"
