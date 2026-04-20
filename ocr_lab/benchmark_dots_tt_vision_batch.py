#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGE = Path("/home/mlops/Desktop/dosyalar/rendered/1_page.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--images-glob", default="")
    parser.add_argument("--model-path", default="rednote-hilab/dots.mocr")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json-out", default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_vision_batch_latest.json"))
    return parser.parse_args()


def build_single_inputs(model_path: str, image_path: Path):
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "Read the page."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs["pixel_values"], inputs["image_grid_thw"]


def resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.images_glob:
        paths = [Path(p).resolve() for p in sorted(glob.glob(args.images_glob))]
        if not paths:
            raise FileNotFoundError(f"no images matched glob: {args.images_glob}")
        return paths
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    return [image_path]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    image_paths = resolve_image_paths(args)

    from dots_tt_port.vision_tower_smoke import (
        TowerHybridCpuUnary,
        ensure_tt_env,
        load_vision_model,
        resolve_snapshot_dir,
    )
    import torch_xla
    import torch_xla.runtime as xr

    snapshot_dir = resolve_snapshot_dir(args.model_path)
    model_path = str(snapshot_dir)
    pixel_values_parts = []
    image_grid_parts = []
    for image_path in image_paths:
        pixel_values_single, image_grid_thw_single = build_single_inputs(model_path, image_path)
        pixel_values_parts.append(pixel_values_single)
        image_grid_parts.append(image_grid_thw_single)
    pixel_values_base = torch.cat(pixel_values_parts, dim=0)
    image_grid_thw_base = torch.cat(image_grid_parts, dim=0)

    tt_vision_model, _ = load_vision_model(snapshot_dir, args.attn, limit_layers=0)
    tt_vision_model = TowerHybridCpuUnary(tt_vision_model).eval().to(dtype=torch.bfloat16)

    ensure_tt_env(args.device_index)
    xr.set_device_type("TT")
    device = torch_xla.device()
    tt_vision_model = tt_vision_model.to(device)

    results = []
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    with torch.no_grad():
        for batch_size in batch_sizes:
            pixel_values = torch.cat([pixel_values_base] * batch_size, dim=0)
            image_grid_thw = torch.cat([image_grid_thw_base] * batch_size, dim=0)
            pixel_values_tt = pixel_values.to(dtype=torch.bfloat16).to(device)
            image_grid_thw_tt = image_grid_thw.to(dtype=torch.int32).to(device)

            timings = []
            error = None
            output_shape = None
            try:
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    out = tt_vision_model(pixel_values_tt, image_grid_thw_tt)
                    torch_xla.sync(wait=True)
                    out_cpu = out.detach().cpu()
                    t1 = time.perf_counter()
                    timings.append((t1 - t0) * 1000.0)
                    output_shape = list(out_cpu.shape)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            item = {
                "base_image_count": len(image_paths),
                "batch_size": batch_size,
                "effective_image_count": len(image_paths) * batch_size,
                "pixel_values_shape": list(pixel_values.shape),
                "image_grid_thw_shape": list(image_grid_thw.shape),
                "output_shape": output_shape,
                "timings_ms": [round(x, 3) for x in timings],
                "first_ms": round(timings[0], 3) if timings else None,
                "hot_avg_ms": round(sum(timings[1:]) / len(timings[1:]), 3) if len(timings) > 1 else None,
                "error": error,
            }
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            if error is not None:
                break

    report = {
        "images": [str(p) for p in image_paths],
        "device": str(device),
        "repeats": args.repeats,
        "results": results,
    }
    out_path = Path(args.json_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
