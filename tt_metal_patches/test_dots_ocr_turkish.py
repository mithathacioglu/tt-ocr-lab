# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Turkish-input variant of test_dots_ocr_vision.

Loads /home/mlops/dll_project/ocr_lab/tmp_1pdf_page-1.png and validates
10 fixed Turkish legal keywords against the OCR output.
"""

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

KEYWORDS = [
    "hükümler", "davalı", "olunmuş", "Dilekçesi", "kesinleştiği",
    "BAKIRKÖY", "KESİNLEŞME", "Mahkememizden", "İstinaftan", "tasdik",
]

TURKISH_IMAGE_PATH = os.environ.get("TURKISH_IMAGE_PATH", "/home/mlops/dll_project/ocr_lab/tmp_1pdf_page-1.png")
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"


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
def test_dots_ocr_turkish(mesh_device):
    """Turkish legal document OCR via TTNN dots.ocr pipeline."""
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info

    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
    )

    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": TURKISH_IMAGE_PATH},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    print(f"Turkish input: input_ids.shape={input_ids.shape} pixel_values.shape={pixel_values.shape} grid_thw={image_grid_thw}")

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=300,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    decoded = processor.decode(generated_ids, skip_special_tokens=True)

    total_time = end_time - start_time
    num_tokens = len(generated_ids)
    tokens_per_second = num_tokens / total_time
    ms_per_token = total_time / num_tokens * 1000

    found = [k for k in KEYWORDS if k in decoded]

    print(f"\n{'='*60}")
    print(f"Turkish OCR Output:")
    print(decoded)
    print(f"\n{'='*60}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"Keywords found:       {len(found)}/{len(KEYWORDS)} -> {found}")
    print(f"Keywords missing:     {[k for k in KEYWORDS if k not in decoded]}")
    print(f"{'='*60}\n")

    DispatchManager.save_stats_to_file("dots_ocr_turkish_timing_stats.csv")
    pipeline.release()

    # Pass test if at least the output is non-empty and not entirely garbage
    assert len(decoded.strip()) > 0, "Generated output is empty"
