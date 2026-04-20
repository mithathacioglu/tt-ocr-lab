#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_sizes(raw: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        w, h = item.lower().split("x", 1)
        sizes.append((int(w), int(h)))
    return sizes


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--sizes", default="672x952,560x792")
    parser.add_argument("--tokens", default="16,32,64")
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ocr_lab.worker_dots_mocr_tt_vision_hybrid import DotsHybridWorker

    image_path = Path(args.image).resolve()
    sizes = parse_sizes(args.sizes)
    token_list = parse_ints(args.tokens)

    all_results = []
    for width, height in sizes:
        worker_args = argparse.Namespace(
            model_path="rednote-hilab/dots.mocr",
            device_index=args.device_index,
            attn="sdpa",
            max_new_tokens=max(token_list),
            prompt="Please output the exact text in the image.\n\nReturn plain text only.\n",
            poll_sec=0.0,
            inbox="",
            done_dir="",
            failed_dir="",
            results_dir="",
            once=True,
            warmup_image="",
            fixed_page_width=width,
            fixed_page_height=height,
            no_fixed_page=False,
        )
        worker = DotsHybridWorker(worker_args)
        warmup = worker.infer(image_path)
        runs = []
        for max_tokens in token_list:
            worker.args.max_new_tokens = max_tokens
            result = worker.infer(image_path)
            runs.append(
                {
                    "max_new_tokens": max_tokens,
                    "total_ms": result["total_ms"],
                    "tt_vision_ms": result["tt_vision_ms"],
                    "cpu_decoder_ms": result["cpu_decoder_ms"],
                    "output_preview": result["output_text"][:200],
                    "output_text": result["output_text"],
                }
            )
        all_results.append(
            {
                "fixed_page_width": width,
                "fixed_page_height": height,
                "warmup_total_ms": warmup["total_ms"],
                "warmup_tt_vision_ms": warmup["tt_vision_ms"],
                "warmup_cpu_decoder_ms": warmup["cpu_decoder_ms"],
                "runs": runs,
            }
        )

    out = {
        "image": str(image_path),
        "device_index": args.device_index,
        "sizes": all_results,
    }
    out_path = Path(args.json_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
