#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_lab.dots_model import DEFAULT_MODEL_PATH, ensure_dots_ocr_processor_compat
from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import DEFAULT_A4_HEIGHT, DEFAULT_A4_WIDTH, fixed_page_message
DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "tt-mlir"
    / "third_party"
    / "tt-metal"
    / "src"
    / "tt-metal"
    / "models"
    / "sample_data"
    / "iam_ocr_image.jpg"
)

DEFAULT_PROMPT = """Please output the exact text in the image.

Return plain text only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fixed-page-width", type=int, default=DEFAULT_A4_WIDTH)
    parser.add_argument("--fixed-page-height", type=int, default=DEFAULT_A4_HEIGHT)
    parser.add_argument("--no-fixed-page", action="store_true")
    parser.add_argument("--json-out", default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_vision_hybrid_latest.json"))
    return parser.parse_args()


def build_inputs(
    model_path: str,
    image_path: Path,
    prompt: str,
    fixed_page_width: int,
    fixed_page_height: int,
    use_fixed_page: bool,
):
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    ensure_dots_ocr_processor_compat(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if use_fixed_page:
        image_message, fixed_page_meta = fixed_page_message(
            image_path,
            width=fixed_page_width,
            height=fixed_page_height,
        )
    else:
        image_message = {"type": "image", "image": str(image_path)}
        fixed_page_meta = {
            "enabled": False,
            "source_width": None,
            "source_height": None,
            "target_width": None,
            "target_height": None,
            "scale": None,
            "resized_width": None,
            "resized_height": None,
            "pad_left": None,
            "pad_top": None,
        }
    messages = [
        {
            "role": "user",
            "content": [
                image_message,
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return processor, inputs, fixed_page_meta


def inject_vision_embeddings(model, input_ids: torch.Tensor, img_mask: torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:
    inputs_embeds = model.get_input_embeddings()(input_ids)
    true_indices = torch.nonzero(img_mask).squeeze()
    if len(true_indices) > vision_embeddings.size(0):
        true_indices = true_indices[: vision_embeddings.size(0)]
        new_img_mask = torch.zeros_like(img_mask, device=img_mask.device)
        new_img_mask[true_indices[:, 0], true_indices[:, 1]] = True
    else:
        new_img_mask = img_mask

    if vision_embeddings.size(0) != new_img_mask.sum():
        raise RuntimeError(
            f"vision embedding count mismatch: {vision_embeddings.size(0)=} mask_sum={int(new_img_mask.sum())}"
        )

    return inputs_embeds.masked_scatter(
        new_img_mask.unsqueeze(-1).expand_as(inputs_embeds),
        vision_embeddings.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
    )


def build_generation_text(processor, inputs, generated_ids) -> str:
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from dots_tt_port.vision_tower_smoke import (
        TowerHybridCpuUnary,
        ensure_tt_env,
        load_vision_model,
        resolve_snapshot_dir,
    )
    import torch_xla.runtime as xr
    from transformers import AutoModelForCausalLM

    snapshot_dir = resolve_snapshot_dir(args.model_path)
    model_path = str(snapshot_dir)

    processor, inputs, fixed_page_meta = build_inputs(
        model_path,
        image_path,
        args.prompt,
        args.fixed_page_width,
        args.fixed_page_height,
        use_fixed_page=not args.no_fixed_page,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()

    original_forward = model.forward

    def patched_forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is None and inputs_embeds is not None:
            input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        return original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

    model.forward = types.MethodType(patched_forward, model)

    tt_vision_model, _ = load_vision_model(snapshot_dir, args.attn, limit_layers=0)
    tt_vision_model = TowerHybridCpuUnary(tt_vision_model).eval().to(dtype=torch.bfloat16)

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    ensure_tt_env(args.device_index)
    perf_runtime = configure_tt_runtime(
        cache_dir=None,
        enable_trace=False,
    )
    xr.set_device_type("TT")
    import torch_xla

    tt_device = torch_xla.device()
    pixel_values_tt = pixel_values.to(dtype=torch.bfloat16).to(tt_device)
    image_grid_thw_tt = image_grid_thw.to(dtype=torch.int32).to(tt_device)
    tt_vision_model = tt_vision_model.to(tt_device)

    img_mask = inputs.input_ids == model.config.image_token_id
    iterations = []
    output_text = ""

    with torch.no_grad():
        for idx in range(args.repeats):
            t0 = time.perf_counter()
            vision_embeddings_tt = tt_vision_model(pixel_values_tt, image_grid_thw_tt)
            torch_xla.sync(wait=True)
            vision_embeddings = vision_embeddings_tt.detach().cpu().float()
            t1 = time.perf_counter()

            inputs_embeds = inject_vision_embeddings(model, inputs.input_ids, img_mask, vision_embeddings)
            gen_inputs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "inputs_embeds": inputs_embeds,
                "max_new_tokens": args.max_new_tokens,
            }
            t2 = time.perf_counter()
            generated_ids = model.generate(**gen_inputs)
            t3 = time.perf_counter()
            output_text = build_generation_text(processor, inputs, generated_ids)
            iterations.append(
                {
                    "index": idx + 1,
                    "tt_vision_ms": round((t1 - t0) * 1000.0, 3),
                    "cpu_decoder_ms": round((t3 - t2) * 1000.0, 3),
                    "total_ms": round((t3 - t0) * 1000.0, 3),
                    "output_text": output_text,
                }
            )

    result = {
        "image": str(image_path),
        "model_path": args.model_path,
        "device_index": args.device_index,
        "prompt": args.prompt,
        "repeats": args.repeats,
        "fixed_page": fixed_page_meta,
        "tt_device": str(tt_device),
        "tt_runtime": perf_runtime,
        "iterations": iterations,
        "summary": {
            "first_total_ms": iterations[0]["total_ms"],
            "first_tt_vision_ms": iterations[0]["tt_vision_ms"],
            "first_cpu_decoder_ms": iterations[0]["cpu_decoder_ms"],
            "hot_total_avg_ms": round(sum(x["total_ms"] for x in iterations[1:]) / len(iterations[1:]), 3)
            if len(iterations) > 1
            else None,
            "hot_tt_vision_avg_ms": round(sum(x["tt_vision_ms"] for x in iterations[1:]) / len(iterations[1:]), 3)
            if len(iterations) > 1
            else None,
            "hot_cpu_decoder_avg_ms": round(
                sum(x["cpu_decoder_ms"] for x in iterations[1:]) / len(iterations[1:]), 3
            )
            if len(iterations) > 1
            else None,
        },
        "output_text": output_text,
    }

    out_path = Path(args.json_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
