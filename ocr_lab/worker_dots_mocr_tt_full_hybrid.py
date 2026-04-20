#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import DEFAULT_A4_HEIGHT, DEFAULT_A4_WIDTH, fixed_page_message

OCR_LAB = PROJECT_ROOT / "ocr_lab"
DEFAULT_INBOX = OCR_LAB / "inbox_tt_full_hybrid"
DEFAULT_DONE = OCR_LAB / "done_tt_full_hybrid"
DEFAULT_FAILED = OCR_LAB / "failed_tt_full_hybrid"
DEFAULT_RESULTS = OCR_LAB / "results_tt_full_hybrid"

DEFAULT_PROMPT = """Please output the exact text in the image.

Return plain text only.
"""

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--inbox", default=str(DEFAULT_INBOX))
    parser.add_argument("--done-dir", default=str(DEFAULT_DONE))
    parser.add_argument("--failed-dir", default=str(DEFAULT_FAILED))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--warmup-image", default="")
    parser.add_argument("--fixed-page-width", type=int, default=DEFAULT_A4_WIDTH)
    parser.add_argument("--fixed-page-height", type=int, default=DEFAULT_A4_HEIGHT)
    return parser.parse_args()


def list_images(inbox: Path) -> list[Path]:
    return sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def write_result(results_dir: Path, image_path: Path, payload: dict) -> Path:
    out_path = results_dir / f"{image_path.stem}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path = results_dir / f"{image_path.stem}.txt"
    txt_path.write_text(payload.get("output_text", ""), encoding="utf-8")
    return out_path


def move_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    shutil.move(str(src), str(dst))
    return dst


class DotsFullHybridWorker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
        if shims_dir.exists():
            sys.path.insert(0, str(shims_dir))

        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch_xla.runtime as xr

        from dots_tt_port.vision_tower_smoke import (
            TowerHybridCpuUnary,
            ensure_tt_env,
            load_vision_model,
            resolve_snapshot_dir,
        )
        from ocr_lab.dots_tt_hybrid_decoder import hybrid_greedy_generate

        self.process_vision_info = process_vision_info
        self.hybrid_greedy_generate = hybrid_greedy_generate
        self.snapshot_dir = resolve_snapshot_dir(args.model_path)
        self.model_path = str(self.snapshot_dir)

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()

        tt_vision_model, _ = load_vision_model(self.snapshot_dir, args.attn, limit_layers=0)
        self.tt_vision_model = TowerHybridCpuUnary(tt_vision_model).eval().to(dtype=torch.bfloat16)

        ensure_tt_env(args.device_index)
        self.tt_runtime = configure_tt_runtime(
            cache_dir=None,
            enable_trace=False,
        )
        xr.set_device_type("TT")
        import torch_xla

        self.torch_xla = torch_xla
        self.tt_device = torch_xla.device()

        self.model = self.model.to(self.tt_device)
        self.tt_vision_model = self.tt_vision_model.to(self.tt_device)
        self.embedding_weight_cpu = self.model.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

    def build_inputs(self, image_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
        image_message, fixed_page_meta = fixed_page_message(
            image_path,
            width=self.args.fixed_page_width,
            height=self.args.fixed_page_height,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    image_message,
                    {"type": "text", "text": self.args.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        return (
            self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"),
            fixed_page_meta,
        )

    def infer(self, image_path: Path) -> dict:
        inputs, fixed_page_meta = self.build_inputs(image_path)
        pixel_values_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(self.tt_device)
        image_grid_thw_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(self.tt_device)

        t0 = time.perf_counter()
        vision_embeddings_tt = self.tt_vision_model(pixel_values_tt, image_grid_thw_tt)
        self.torch_xla.sync(wait=True)
        t1 = time.perf_counter()
        vision_embeddings = vision_embeddings_tt.detach().cpu().to(dtype=torch.bfloat16)

        hybrid = self.hybrid_greedy_generate(
            self.model,
            self.processor,
            inputs["input_ids"],
            vision_embeddings,
            self.tt_device,
            max_new_tokens=self.args.max_new_tokens,
            sync_fn=lambda: self.torch_xla.sync(wait=True),
            eos_token_id=self.model.generation_config.eos_token_id,
            embedding_weight_cpu=self.embedding_weight_cpu,
            mlp_mode=self.args.mlp_mode,
        )
        t2 = time.perf_counter()

        return {
            "image": str(image_path),
            "device_index": self.args.device_index,
            "tt_device": str(self.tt_device),
            "fixed_page": fixed_page_meta,
            "tt_runtime": self.tt_runtime,
            "mlp_mode": self.args.mlp_mode,
            "tt_vision_ms": round((t1 - t0) * 1000.0, 3),
            "decoder_total_ms": round(float(hybrid["total_ms"]), 3),
            "total_ms": round((t2 - t0) * 1000.0, 3),
            "num_generated_tokens": hybrid["num_generated_tokens"],
            "token_timings_ms": hybrid["token_timings_ms"],
            "generated_token_ids": hybrid["generated_token_ids"],
            "output_text": hybrid["generated_text"],
            "output_text_raw": hybrid["generated_text_raw"],
        }


def main() -> int:
    args = parse_args()
    inbox = Path(args.inbox).resolve()
    done_dir = Path(args.done_dir).resolve()
    failed_dir = Path(args.failed_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    inbox.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    worker = DotsFullHybridWorker(args)

    if args.warmup_image:
        warmup_path = Path(args.warmup_image).resolve()
        if warmup_path.exists():
            warmup_result = worker.infer(warmup_path)
            print(json.dumps({"warmup": warmup_result}, ensure_ascii=False), flush=True)

    while True:
        images = list_images(inbox)
        if not images:
            if args.once:
                break
            time.sleep(args.poll_sec)
            continue

        for image_path in images:
            started = time.time()
            try:
                result = worker.infer(image_path)
                result["started_at"] = started
                result["finished_at"] = time.time()
                result["status"] = "ok"
                result_path = write_result(results_dir, image_path, result)
                moved_to = move_file(image_path, done_dir)
                print(
                    json.dumps(
                        {
                            "status": "ok",
                            "image": str(image_path),
                            "result": str(result_path),
                            "moved_to": str(moved_to),
                            "total_ms": result["total_ms"],
                            "output_text": result["output_text"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                error_payload = {
                    "status": "error",
                    "image": str(image_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "started_at": started,
                    "finished_at": time.time(),
                }
                result_path = write_result(results_dir, image_path, error_payload)
                moved_to = move_file(image_path, failed_dir)
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "image": str(image_path),
                            "result": str(result_path),
                            "moved_to": str(moved_to),
                            "error": error_payload["error"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if args.once:
            break

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
