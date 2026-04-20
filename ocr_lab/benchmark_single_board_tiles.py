#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--fixed-page-width", type=int, required=True)
    parser.add_argument("--fixed-page-height", type=int, required=True)
    parser.add_argument("--warmup-first-tile", action="store_true")
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from ocr_lab.worker_dots_mocr_tt_vision_hybrid import DotsHybridWorker

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiles = manifest["tiles"]

    worker_args = argparse.Namespace(
        model_path="rednote-hilab/dots.mocr",
        device_index=args.device_index,
        attn="sdpa",
        max_new_tokens=args.max_new_tokens,
        prompt="Please output the exact text in the image.\n\nReturn plain text only.\n",
        poll_sec=0.0,
        inbox="",
        done_dir="",
        failed_dir="",
        results_dir="",
        once=True,
        warmup_image="",
        fixed_page_width=args.fixed_page_width,
        fixed_page_height=args.fixed_page_height,
        no_fixed_page=False,
    )

    worker = DotsHybridWorker(worker_args)

    warmup = None
    if args.warmup_first_tile and tiles:
        warmup = worker.infer(Path(tiles[0]["path"]).resolve())

    rows = []
    for tile in tiles:
        result = worker.infer(Path(tile["path"]).resolve())
        rows.append(
            {
                "row": tile["row"],
                "col": tile["col"],
                "path": tile["path"],
                "bbox": tile["bbox"],
                "size": tile["size"],
                "total_ms": result["total_ms"],
                "tt_vision_ms": result["tt_vision_ms"],
                "cpu_decoder_ms": result["cpu_decoder_ms"],
                "output_text": result["output_text"],
                "fixed_page": result["fixed_page"],
            }
        )

    summary = {
        "manifest": str(manifest_path),
        "device_index": args.device_index,
        "fixed_page_width": args.fixed_page_width,
        "fixed_page_height": args.fixed_page_height,
        "max_new_tokens": args.max_new_tokens,
        "warmup_first_tile": args.warmup_first_tile,
        "warmup": warmup,
        "tile_count": len(rows),
        "tiles": rows,
        "sequential_total_ms": round(sum(r["total_ms"] for r in rows), 3) if rows else None,
        "parallel_max_ms": round(max(r["total_ms"] for r in rows), 3) if rows else None,
        "avg_tile_ms": round(sum(r["total_ms"] for r in rows) / len(rows), 3) if rows else None,
        "avg_tt_vision_ms": round(sum(r["tt_vision_ms"] for r in rows) / len(rows), 3) if rows else None,
        "avg_cpu_decoder_ms": round(sum(r["cpu_decoder_ms"] for r in rows) / len(rows), 3) if rows else None,
    }

    out_path = Path(args.json_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
