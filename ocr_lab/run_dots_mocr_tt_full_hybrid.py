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
    parser.add_argument("--model-path", default="rednote-hilab/dots.mocr")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--mlp-mode",
        default="cpu_silu",
        choices=["cpu_silu", "cpu_silu_tt_mul", "tt_exact", "tt_poly2", "tt_poly7_datafit"],
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fixed-page-width", type=int, default=DEFAULT_A4_WIDTH)
    parser.add_argument("--fixed-page-height", type=int, default=DEFAULT_A4_HEIGHT)
    parser.add_argument(
        "--json-out",
        default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_full_hybrid_latest.json"),
    )
    return parser.parse_args()


def build_inputs(model_path: str, image_path: Path, prompt: str, fixed_page_width: int, fixed_page_height: int):
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    image_message, fixed_page_meta = fixed_page_message(
        image_path,
        width=fixed_page_width,
        height=fixed_page_height,
    )
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


def build_generation_text(processor, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
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

    from transformers import AutoModelForCausalLM
    import torch_xla.runtime as xr

    from dots_tt_port.vision_tower_smoke import (
        TowerHybridCpuUnary,
        ensure_tt_env,
        load_vision_model,
        resolve_snapshot_dir,
    )
    from ocr_lab.dots_tt_hybrid_decoder import (
        build_inputs_embeds_cpu,
        hybrid_greedy_generate,
    )

    snapshot_dir = resolve_snapshot_dir(args.model_path)
    model_path = str(snapshot_dir)

    processor, inputs, fixed_page_meta = build_inputs(
        model_path,
        image_path,
        args.prompt,
        args.fixed_page_width,
        args.fixed_page_height,
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
    embedding_weight_cpu = model.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        cpu_t0 = time.perf_counter()
        cpu_inputs_embeds = build_inputs_embeds_cpu(
            model,
            input_ids,
            tt_vision_model(pixel_values.to(dtype=torch.bfloat16), image_grid_thw.to(dtype=torch.int32)).detach().cpu(),
            embedding_weight_cpu=embedding_weight_cpu,
        )
        cpu_generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=cpu_inputs_embeds,
            max_new_tokens=args.max_new_tokens,
        )
        cpu_t1 = time.perf_counter()
        cpu_output_text = build_generation_text(processor, input_ids, cpu_generated_ids)

    ensure_tt_env(args.device_index)
    perf_runtime = configure_tt_runtime(
        cache_dir=None,
        enable_trace=False,
    )
    xr.set_device_type("TT")
    import torch_xla

    tt_device = torch_xla.device()

    tt_vision_model = tt_vision_model.to(tt_device)
    model = model.to(tt_device)
    pixel_values_tt = pixel_values.to(dtype=torch.bfloat16).to(tt_device)
    image_grid_thw_tt = image_grid_thw.to(dtype=torch.int32).to(tt_device)

    with torch.no_grad():
        t0 = time.perf_counter()
        vision_embeddings_tt = tt_vision_model(pixel_values_tt, image_grid_thw_tt)
        torch_xla.sync(wait=True)
        t1 = time.perf_counter()
        vision_embeddings = vision_embeddings_tt.detach().cpu().to(dtype=torch.bfloat16)

        hybrid = hybrid_greedy_generate(
            model,
            processor,
            input_ids,
            vision_embeddings,
            tt_device,
            max_new_tokens=args.max_new_tokens,
            sync_fn=lambda: torch_xla.sync(wait=True),
            eos_token_id=model.generation_config.eos_token_id,
            embedding_weight_cpu=embedding_weight_cpu,
            mlp_mode=args.mlp_mode,
        )
        t2 = time.perf_counter()

    result = {
        "image": str(image_path),
        "model_path": args.model_path,
        "device_index": args.device_index,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "mlp_mode": args.mlp_mode,
        "fixed_page": fixed_page_meta,
        "tt_device": str(tt_device),
        "tt_runtime": perf_runtime,
        "cpu_baseline": {
            "total_ms": round((cpu_t1 - cpu_t0) * 1000.0, 3),
            "output_text": cpu_output_text,
        },
        "tt_hybrid": {
            "tt_vision_ms": round((t1 - t0) * 1000.0, 3),
            "decoder_total_ms": hybrid["total_ms"],
            "total_ms": round((t2 - t0) * 1000.0, 3),
            "num_generated_tokens": hybrid["num_generated_tokens"],
            "token_timings_ms": hybrid["token_timings_ms"],
            "generated_token_ids": hybrid["generated_token_ids"],
            "output_text": hybrid["generated_text"],
            "output_text_raw": hybrid["generated_text_raw"],
        },
        "match_cpu": hybrid["generated_text"].strip() == cpu_output_text.strip(),
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
