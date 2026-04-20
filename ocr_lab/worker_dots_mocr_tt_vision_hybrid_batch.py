#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
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

OCR_LAB = PROJECT_ROOT / "ocr_lab"
DEFAULT_INBOX = OCR_LAB / "inbox_tt_hybrid_batch"
DEFAULT_DONE = OCR_LAB / "done_tt_hybrid_batch"
DEFAULT_FAILED = OCR_LAB / "failed_tt_hybrid_batch"
DEFAULT_RESULTS = OCR_LAB / "results_tt_hybrid_batch"

DEFAULT_PROMPT = """Please output the exact text in the image.

Return plain text only.
"""

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--inbox", default=str(DEFAULT_INBOX))
    parser.add_argument("--done-dir", default=str(DEFAULT_DONE))
    parser.add_argument("--failed-dir", default=str(DEFAULT_FAILED))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--images-glob", default="")
    parser.add_argument("--warmup-glob", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--fixed-page-width", type=int, default=DEFAULT_A4_WIDTH)
    parser.add_argument("--fixed-page-height", type=int, default=DEFAULT_A4_HEIGHT)
    return parser.parse_args()


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


class DotsHybridBatchWorker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
        if shims_dir.exists():
            sys.path.insert(0, str(shims_dir))

        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForCausalLM, AutoProcessor
        from dots_tt_port.vision_tower_smoke import (
            TowerHybridCpuUnary,
            ensure_tt_env,
            load_vision_model,
            resolve_snapshot_dir,
        )
        import torch_xla.runtime as xr

        self.process_vision_info = process_vision_info
        self.snapshot_dir = resolve_snapshot_dir(args.model_path)
        self.model_path = str(self.snapshot_dir)
        ensure_dots_ocr_processor_compat(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()

        original_forward = self.model.forward

        def patched_forward(model_self, input_ids=None, inputs_embeds=None, **kwargs):
            if input_ids is None and inputs_embeds is not None:
                input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
            return original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

        self.model.forward = types.MethodType(patched_forward, self.model)

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
        self.tt_vision_model = self.tt_vision_model.to(self.tt_device)

    def build_batch_inputs(self, image_paths: list[Path]) -> tuple[dict[str, torch.Tensor], list[dict]]:
        conversations = []
        texts = []
        fixed_page_meta = []
        for image_path in image_paths:
            image_message, image_meta = fixed_page_message(
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
            conversations.append(messages)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            fixed_page_meta.append(image_meta)

        image_inputs, video_inputs = self.process_vision_info(conversations)
        return (
            self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"),
            fixed_page_meta,
        )

    def infer_batch(self, image_paths: list[Path]) -> dict:
        if not image_paths:
            raise ValueError("batch is empty")

        inputs, fixed_page_meta = self.build_batch_inputs(image_paths)
        pixel_values_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(self.tt_device)
        image_grid_thw_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(self.tt_device)

        t0 = time.perf_counter()
        vision_embeddings_tt = self.tt_vision_model(pixel_values_tt, image_grid_thw_tt)
        self.torch_xla.sync(wait=True)
        vision_embeddings = vision_embeddings_tt.detach().cpu().float()
        t1 = time.perf_counter()

        img_mask = inputs.input_ids == self.model.config.image_token_id
        inputs_embeds = inject_vision_embeddings(self.model, inputs.input_ids, img_mask, vision_embeddings)

        gen_inputs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "inputs_embeds": inputs_embeds,
            "max_new_tokens": self.args.max_new_tokens,
        }

        t2 = time.perf_counter()
        generated_ids = self.model.generate(**gen_inputs)
        t3 = time.perf_counter()

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        batch_result = {
            "image_count": len(image_paths),
            "images": [str(p) for p in image_paths],
            "device_index": self.args.device_index,
            "tt_device": str(self.tt_device),
            "fixed_pages": fixed_page_meta,
            "tt_runtime": self.tt_runtime,
            "tt_vision_ms": round((t1 - t0) * 1000.0, 3),
            "cpu_decoder_ms": round((t3 - t2) * 1000.0, 3),
            "total_ms": round((t3 - t0) * 1000.0, 3),
            "outputs": [
                {
                    "image": str(image_path),
                    "output_text": output_text,
                }
                for image_path, output_text in zip(image_paths, output_texts)
            ],
        }
        return batch_result


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


def resolve_image_glob(images_glob: str) -> list[Path]:
    if not images_glob:
        return []
    return [Path(p).resolve() for p in sorted(glob.glob(images_glob))]


def write_batch_results(results_dir: Path, batch_result: dict, started: float, finished: float) -> list[Path]:
    paths = []
    image_count = max(1, batch_result["image_count"])
    per_item_tt_ms = round(batch_result["tt_vision_ms"] / image_count, 3)
    per_item_cpu_ms = round(batch_result["cpu_decoder_ms"] / image_count, 3)
    per_item_total_ms = round(batch_result["total_ms"] / image_count, 3)

    for item in batch_result["outputs"]:
        image_path = Path(item["image"])
        payload = {
            "status": "ok",
            "image": item["image"],
            "batch_image_count": image_count,
            "device_index": batch_result["device_index"],
            "tt_device": batch_result["tt_device"],
            "batch_tt_vision_ms": batch_result["tt_vision_ms"],
            "batch_cpu_decoder_ms": batch_result["cpu_decoder_ms"],
            "batch_total_ms": batch_result["total_ms"],
            "estimated_item_tt_vision_ms": per_item_tt_ms,
            "estimated_item_cpu_decoder_ms": per_item_cpu_ms,
            "estimated_item_total_ms": per_item_total_ms,
            "output_text": item["output_text"],
            "started_at": started,
            "finished_at": finished,
        }
        paths.append(write_result(results_dir, image_path, payload))
    return paths


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

    worker = DotsHybridBatchWorker(args)

    warmup_paths = resolve_image_glob(args.warmup_glob)
    if warmup_paths:
        warmup_result = worker.infer_batch(warmup_paths[: args.batch_size])
        print(json.dumps({"warmup": warmup_result}, ensure_ascii=False), flush=True)

    direct_paths = resolve_image_glob(args.images_glob)
    if direct_paths:
        started = time.time()
        batch_result = worker.infer_batch(direct_paths[: args.batch_size])
        finished = time.time()
        report = {
            "status": "ok",
            "started_at": started,
            "finished_at": finished,
            "batch": batch_result,
        }
        out_path = Path(args.json_out).resolve() if args.json_out else results_dir / "batch_latest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0

    while True:
        images = list_images(inbox)
        if not images:
            if args.once:
                break
            time.sleep(args.poll_sec)
            continue

        batch_paths = images[: args.batch_size]
        started = time.time()
        try:
            batch_result = worker.infer_batch(batch_paths)
            finished = time.time()
            result_paths = write_batch_results(results_dir, batch_result, started, finished)
            moved_to = [str(move_file(image_path, done_dir)) for image_path in batch_paths]
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "images": [str(p) for p in batch_paths],
                        "results": [str(p) for p in result_paths],
                        "moved_to": moved_to,
                        "batch_total_ms": batch_result["total_ms"],
                        "image_count": batch_result["image_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            finished = time.time()
            error_payload = {
                "status": "error",
                "images": [str(p) for p in batch_paths],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "started_at": started,
                "finished_at": finished,
            }
            out_path = results_dir / f"batch_error_{int(started)}.json"
            out_path.write_text(json.dumps(error_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            moved_to = [str(move_file(image_path, failed_dir)) for image_path in batch_paths]
            print(
                json.dumps(
                    {
                        "status": "error",
                        "images": [str(p) for p in batch_paths],
                        "result": str(out_path),
                        "moved_to": moved_to,
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
