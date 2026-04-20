#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
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

DEFAULT_PROMPT = """Please output the exact text in the image.

Return plain text only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="rednote-hilab/dots.mocr")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--images-glob", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--direct-repeats", type=int, default=1)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--fixed-page-width", type=int, default=DEFAULT_A4_WIDTH)
    parser.add_argument("--fixed-page-height", type=int, default=DEFAULT_A4_HEIGHT)
    parser.add_argument(
        "--json-out",
        default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_vision_cpu_profile_latest.json"),
    )
    return parser.parse_args()


def resolve_image_paths(images_glob: str) -> list[Path]:
    paths = [Path(p).resolve() for p in sorted(glob.glob(images_glob))]
    if not paths:
        raise FileNotFoundError(f"no images matched glob: {images_glob}")
    return paths


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


class ProfileRunner:
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

    def build_batch_inputs(self, image_paths: list[Path]) -> tuple[dict[str, torch.Tensor], dict]:
        t0 = time.perf_counter()
        conversations = []
        texts = []
        fixed_pages = []
        for image_path in image_paths:
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
            conversations.append(messages)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            fixed_pages.append(fixed_page_meta)
        t1 = time.perf_counter()
        image_inputs, video_inputs = self.process_vision_info(conversations)
        t2 = time.perf_counter()
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        t3 = time.perf_counter()
        return inputs, {
            "chat_template_ms": round((t1 - t0) * 1000.0, 3),
            "vision_info_ms": round((t2 - t1) * 1000.0, 3),
            "processor_ms": round((t3 - t2) * 1000.0, 3),
            "build_inputs_ms": round((t3 - t0) * 1000.0, 3),
            "fixed_pages": fixed_pages,
        }

    def run_once(self, image_paths: list[Path], label: str) -> dict:
        overall_t0 = time.perf_counter()
        inputs, build_stats = self.build_batch_inputs(image_paths)

        t_h2d_0 = time.perf_counter()
        pixel_values_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(self.tt_device)
        image_grid_thw_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(self.tt_device)
        t_h2d_1 = time.perf_counter()

        t_tt_0 = time.perf_counter()
        vision_embeddings_tt = self.tt_vision_model(pixel_values_tt, image_grid_thw_tt)
        self.torch_xla.sync(wait=True)
        t_tt_1 = time.perf_counter()

        t_d2h_0 = time.perf_counter()
        vision_embeddings = vision_embeddings_tt.detach().cpu().float()
        t_d2h_1 = time.perf_counter()

        t_inject_0 = time.perf_counter()
        img_mask = inputs.input_ids == self.model.config.image_token_id
        inputs_embeds = inject_vision_embeddings(self.model, inputs.input_ids, img_mask, vision_embeddings)
        t_inject_1 = time.perf_counter()

        t_gen_0 = time.perf_counter()
        generated_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.args.max_new_tokens,
        )
        t_gen_1 = time.perf_counter()

        t_decode_0 = time.perf_counter()
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        t_decode_1 = time.perf_counter()
        overall_t1 = time.perf_counter()

        generated_token_counts = [int(ids.shape[0]) for ids in generated_ids_trimmed]
        return {
            "label": label,
            "image_count": len(image_paths),
            "images": [str(p) for p in image_paths],
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "image_grid_thw_shape": list(inputs["image_grid_thw"].shape),
            "generated_token_counts": generated_token_counts,
            "generated_token_total": int(sum(generated_token_counts)),
            "timings_ms": {
                **build_stats,
                "h2d_ms": round((t_h2d_1 - t_h2d_0) * 1000.0, 3),
                "tt_vision_ms": round((t_tt_1 - t_tt_0) * 1000.0, 3),
                "vision_d2h_ms": round((t_d2h_1 - t_d2h_0) * 1000.0, 3),
                "inject_ms": round((t_inject_1 - t_inject_0) * 1000.0, 3),
                "cpu_generate_ms": round((t_gen_1 - t_gen_0) * 1000.0, 3),
                "decode_ms": round((t_decode_1 - t_decode_0) * 1000.0, 3),
                "total_ms": round((overall_t1 - overall_t0) * 1000.0, 3),
            },
            "outputs": [
                {
                    "image": str(image_path),
                    "generated_tokens": token_count,
                    "output_text": output_text,
                }
                for image_path, token_count, output_text in zip(image_paths, generated_token_counts, output_texts)
            ],
        }


def main() -> int:
    args = parse_args()
    image_paths = resolve_image_paths(args.images_glob)
    runner = ProfileRunner(args)

    report = {
        "images": [str(p) for p in image_paths],
        "device_index": args.device_index,
        "max_new_tokens": args.max_new_tokens,
        "tt_device": str(runner.tt_device),
        "tt_runtime": runner.tt_runtime,
        "runs": [],
    }

    if not args.skip_warmup:
        report["runs"].append(runner.run_once(image_paths, "warmup"))

    for idx in range(args.direct_repeats):
        report["runs"].append(runner.run_once(image_paths, f"direct_{idx + 1}"))

    out_path = Path(args.json_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
