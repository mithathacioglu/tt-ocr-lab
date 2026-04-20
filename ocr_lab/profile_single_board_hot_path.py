#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from ocr_lab.dots_model import DEFAULT_MODEL_PATH, ensure_dots_ocr_processor_compat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dots_tt_port.vision_tower_smoke import apply_norm_cpu_module
from ocr_lab.worker_dots_mocr_tt_vision_hybrid import DotsHybridWorker, inject_vision_embeddings


DEFAULT_IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
DEFAULT_JSON = PROJECT_ROOT / "ocr_lab" / "single_board_hot_function_profile.json"
DEFAULT_TXT = PROJECT_ROOT / "ocr_lab" / "single_board_hot_function_profile.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Please output the exact text in the image.\n\nReturn plain text only.\n")
    parser.add_argument("--fixed-page-width", type=int, default=672)
    parser.add_argument("--fixed-page-height", type=int, default=952)
    parser.add_argument("--no-fixed-page", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--txt-out", default=str(DEFAULT_TXT))
    return parser.parse_args()


class TimerBook:
    def __init__(self) -> None:
        self._stats: dict[str, dict[str, float | int]] = defaultdict(lambda: {"calls": 0, "ms": 0.0})

    def add(self, name: str, elapsed_ms: float) -> None:
        self._stats[name]["calls"] += 1
        self._stats[name]["ms"] += float(elapsed_ms)

    def as_sorted_dict(self) -> dict[str, dict[str, float | int]]:
        ordered = {}
        for name, payload in sorted(self._stats.items(), key=lambda item: (-float(item[1]["ms"]), item[0])):
            ordered[name] = {
                "calls": int(payload["calls"]),
                "ms": round(float(payload["ms"]), 3),
                "avg_ms": round(float(payload["ms"]) / int(payload["calls"]), 3),
            }
        return ordered


class ForwardProfiler:
    def __init__(self) -> None:
        self.timers = TimerBook()
        self._originals: list[tuple[torch.nn.Module, object]] = []

    def wrap(self, name: str, module: torch.nn.Module | None) -> None:
        if module is None:
            return
        original_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            t0 = time.perf_counter()
            out = original_forward(*args, **kwargs)
            t1 = time.perf_counter()
            self.timers.add(name, (t1 - t0) * 1000.0)
            return out

        module.forward = wrapped_forward  # type: ignore[method-assign]
        self._originals.append((module, original_forward))

    def restore(self) -> None:
        while self._originals:
            module, original_forward = self._originals.pop()
            module.forward = original_forward  # type: ignore[method-assign]


def build_worker_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=args.model_path,
        device_index=args.device_index,
        attn=args.attn,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        poll_sec=1.0,
        inbox="",
        done_dir="",
        failed_dir="",
        results_dir="",
        once=True,
        warmup_image="",
        fixed_page_width=args.fixed_page_width,
        fixed_page_height=args.fixed_page_height,
        no_fixed_page=args.no_fixed_page,
    )


def profile_tt_vision(worker: DotsHybridWorker, pixel_values_tt: torch.Tensor, image_grid_thw_tt: torch.Tensor) -> tuple[torch.Tensor, dict]:
    model = worker.tt_vision_model
    timers = TimerBook()
    block_details = []

    def sync_record(name: str, t0: float) -> float:
        worker.torch_xla.sync(wait=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timers.add(name, elapsed_ms)
        return elapsed_ms

    overall_t0 = time.perf_counter()

    t0 = time.perf_counter()
    hidden_states = model.patch_embed(pixel_values_tt, image_grid_thw_tt)
    sync_record("vision.patch_embed_tt", t0)

    t0 = time.perf_counter()
    cos_cpu, sin_cpu = model._build_rotary_cache_cpu(image_grid_thw_tt)
    timers.add("vision.rotary_cache_cpu", (time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    cu_seqlens_cpu = model._build_cu_seqlens_cpu(image_grid_thw_tt)
    timers.add("vision.cu_seqlens_cpu", (time.perf_counter() - t0) * 1000.0)

    for block_idx, blk in enumerate(model.model.blocks):
        block_timer = {}

        t0 = time.perf_counter()
        norm1 = apply_norm_cpu_module(blk.norm1, hidden_states)
        block_timer["norm1_cpu_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        timers.add("vision.blocks.norm1_cpu", block_timer["norm1_cpu_ms"])

        t0 = time.perf_counter()
        qkv_tt = blk.attn.qkv(norm1)
        block_timer["qkv_tt_ms"] = round(sync_record("vision.blocks.qkv_tt", t0), 3)

        t0 = time.perf_counter()
        attn_ctx_cpu = model._cpu_attention_context(blk.attn, qkv_tt, cu_seqlens_cpu, cos_cpu, sin_cpu)
        block_timer["attn_cpu_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        timers.add("vision.blocks.attn_cpu", block_timer["attn_cpu_ms"])

        t0 = time.perf_counter()
        attn_ctx_tt = attn_ctx_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype)
        proj_tt = blk.attn.proj(attn_ctx_tt)
        hidden_states = hidden_states + proj_tt
        block_timer["proj_tt_residual_ms"] = round(sync_record("vision.blocks.proj_tt_residual", t0), 3)

        t0 = time.perf_counter()
        norm2 = apply_norm_cpu_module(blk.norm2, hidden_states)
        block_timer["norm2_cpu_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        timers.add("vision.blocks.norm2_cpu", block_timer["norm2_cpu_ms"])

        t0 = time.perf_counter()
        fc1_tt = blk.mlp.fc1(norm2)
        block_timer["fc1_tt_ms"] = round(sync_record("vision.blocks.fc1_tt", t0), 3)

        t0 = time.perf_counter()
        fc3_tt = blk.mlp.fc3(norm2)
        block_timer["fc3_tt_ms"] = round(sync_record("vision.blocks.fc3_tt", t0), 3)

        t0 = time.perf_counter()
        gated_tt = torch.nn.functional.silu(fc1_tt) * fc3_tt
        block_timer["gated_tt_ms"] = round(sync_record("vision.blocks.gated_tt", t0), 3)

        t0 = time.perf_counter()
        down_tt = blk.mlp.fc2(gated_tt)
        hidden_states = hidden_states + down_tt
        block_timer["fc2_tt_residual_ms"] = round(sync_record("vision.blocks.fc2_tt_residual", t0), 3)

        block_timer["total_ms"] = round(sum(float(v) for v in block_timer.values()), 3)
        block_details.append({"index": block_idx, **block_timer})

    if model.model.config.post_norm:
        t0 = time.perf_counter()
        hidden_states = apply_norm_cpu_module(model.model.post_trunk_norm, hidden_states)
        timers.add("vision.post_norm_cpu", (time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    out = model.cpu_merger(hidden_states)
    timers.add("vision.merger_cpu", (time.perf_counter() - t0) * 1000.0)

    overall_ms = (time.perf_counter() - overall_t0) * 1000.0
    report = {
        "segmented_total_ms": round(overall_ms, 3),
        "segments": timers.as_sorted_dict(),
        "blocks": block_details,
        "note": "TT segmentlerinde sync eklenmiş ayrıntılı profil; toplam normal hot latency ile birebir aynı olmayabilir.",
    }
    return out, report


def attach_decoder_probes(model) -> ForwardProfiler:
    profiler = ForwardProfiler()
    profiler.wrap("decoder.embed_tokens", model.model.embed_tokens)
    profiler.wrap("decoder.final_norm", model.model.norm)
    profiler.wrap("decoder.rotary_emb", model.model.rotary_emb)
    profiler.wrap("decoder.lm_head", model.lm_head)

    for idx, layer in enumerate(model.model.layers):
        prefix = f"decoder.layers.{idx}"
        profiler.wrap(f"{prefix}.input_layernorm", layer.input_layernorm)
        profiler.wrap(f"{prefix}.self_attn", layer.self_attn)
        profiler.wrap(f"{prefix}.self_attn.q_proj", layer.self_attn.q_proj)
        profiler.wrap(f"{prefix}.self_attn.k_proj", layer.self_attn.k_proj)
        profiler.wrap(f"{prefix}.self_attn.v_proj", layer.self_attn.v_proj)
        profiler.wrap(f"{prefix}.self_attn.o_proj", layer.self_attn.o_proj)
        profiler.wrap(f"{prefix}.post_attention_layernorm", layer.post_attention_layernorm)
        profiler.wrap(f"{prefix}.mlp", layer.mlp)
        profiler.wrap(f"{prefix}.mlp.gate_proj", layer.mlp.gate_proj)
        profiler.wrap(f"{prefix}.mlp.up_proj", layer.mlp.up_proj)
        profiler.wrap(f"{prefix}.mlp.down_proj", layer.mlp.down_proj)
    return profiler


def aggregate_decoder_profile(module_stats: dict[str, dict[str, float | int]]) -> dict[str, float]:
    def sum_prefix(prefix: str) -> float:
        return round(
            sum(float(payload["ms"]) for name, payload in module_stats.items() if name.endswith(prefix)),
            3,
        )

    def sum_exact_suffix(suffix: str) -> float:
        return round(
            sum(float(payload["ms"]) for name, payload in module_stats.items() if name.endswith(suffix)),
            3,
        )

    self_attn_ms = sum_exact_suffix(".self_attn")
    q_proj_ms = sum_exact_suffix(".self_attn.q_proj")
    k_proj_ms = sum_exact_suffix(".self_attn.k_proj")
    v_proj_ms = sum_exact_suffix(".self_attn.v_proj")
    o_proj_ms = sum_exact_suffix(".self_attn.o_proj")
    mlp_ms = sum_exact_suffix(".mlp")
    gate_proj_ms = sum_exact_suffix(".mlp.gate_proj")
    up_proj_ms = sum_exact_suffix(".mlp.up_proj")
    down_proj_ms = sum_exact_suffix(".mlp.down_proj")

    return {
        "embed_tokens_ms": round(float(module_stats.get("decoder.embed_tokens", {}).get("ms", 0.0)), 3),
        "rotary_emb_ms": round(float(module_stats.get("decoder.rotary_emb", {}).get("ms", 0.0)), 3),
        "input_layernorm_ms": sum_prefix(".input_layernorm"),
        "self_attn_total_ms": self_attn_ms,
        "q_proj_ms": q_proj_ms,
        "k_proj_ms": k_proj_ms,
        "v_proj_ms": v_proj_ms,
        "o_proj_ms": o_proj_ms,
        "attn_residual_ms": round(self_attn_ms - (q_proj_ms + k_proj_ms + v_proj_ms + o_proj_ms), 3),
        "post_attention_layernorm_ms": sum_prefix(".post_attention_layernorm"),
        "mlp_total_ms": mlp_ms,
        "gate_proj_ms": gate_proj_ms,
        "up_proj_ms": up_proj_ms,
        "down_proj_ms": down_proj_ms,
        "mlp_residual_ms": round(mlp_ms - (gate_proj_ms + up_proj_ms + down_proj_ms), 3),
        "final_norm_ms": round(float(module_stats.get("decoder.final_norm", {}).get("ms", 0.0)), 3),
        "lm_head_ms": round(float(module_stats.get("decoder.lm_head", {}).get("ms", 0.0)), 3),
    }


def top_items(mapping: dict[str, dict[str, float | int]], limit: int) -> list[dict]:
    items = []
    for name, payload in list(mapping.items())[:limit]:
        items.append({"name": name, **payload})
    return items


def render_text_summary(report: dict) -> str:
    lines = []
    lines.append(f"image: {report['image']}")
    lines.append(f"warmup_runs: {report['warmup_runs']}")
    lines.append(f"hot_total_ms: {report['hot_run']['timings_ms']['total_ms']}")
    lines.append(f"build_inputs_ms: {report['hot_run']['timings_ms']['build_inputs_ms']}")
    lines.append(f"tt_vision_ms: {report['hot_run']['timings_ms']['tt_vision_ms']}")
    lines.append(f"inject_ms: {report['hot_run']['timings_ms']['inject_ms']}")
    lines.append(f"cpu_generate_ms: {report['hot_run']['timings_ms']['cpu_generate_ms']}")
    lines.append(f"decode_ms: {report['hot_run']['timings_ms']['decode_ms']}")
    lines.append("")
    lines.append("decoder_top_modules:")
    for item in report["hot_run"]["decoder_profile"]["top_modules"]:
        lines.append(f"- {item['name']}: {item['ms']} ms ({item['calls']} call)")
    lines.append("")
    lines.append("decoder_aggregates:")
    for key, value in report["hot_run"]["decoder_profile"]["aggregates"].items():
        lines.append(f"- {key}: {value} ms")
    lines.append("")
    lines.append("vision_top_segments:")
    for name, payload in list(report["hot_run"]["tt_vision_profile"]["segments"].items())[:12]:
        lines.append(f"- {name}: {payload['ms']} ms ({payload['calls']} call)")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    worker = DotsHybridWorker(build_worker_args(args))

    warmups = []
    for idx in range(args.warmup_runs):
        warmups.append(worker.infer(image_path) | {"index": idx + 1})

    overall_t0 = time.perf_counter()

    t0 = time.perf_counter()
    inputs, fixed_page_meta = worker.build_inputs(image_path)
    build_inputs_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    pixel_values_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(worker.tt_device)
    image_grid_thw_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(worker.tt_device)
    h2d_ms = (time.perf_counter() - t0) * 1000.0

    vision_t0 = time.perf_counter()
    vision_embeddings_bf16, tt_vision_profile = profile_tt_vision(worker, pixel_values_tt, image_grid_thw_tt)
    tt_vision_ms = (time.perf_counter() - vision_t0) * 1000.0

    t0 = time.perf_counter()
    vision_embeddings = vision_embeddings_bf16.detach().cpu().float()
    vision_d2h_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    img_mask = inputs.input_ids == worker.model.config.image_token_id
    inputs_embeds = inject_vision_embeddings(worker.model, inputs.input_ids, img_mask, vision_embeddings)
    inject_ms = (time.perf_counter() - t0) * 1000.0

    decoder_profiler = attach_decoder_probes(worker.model)
    try:
        t0 = time.perf_counter()
        generated_ids = worker.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=args.max_new_tokens,
        )
        cpu_generate_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        decoder_profiler.restore()

    decoder_module_stats = decoder_profiler.timers.as_sorted_dict()
    decoder_profile = {
        "module_totals": decoder_module_stats,
        "top_modules": top_items(decoder_module_stats, 20),
        "aggregates": aggregate_decoder_profile(decoder_module_stats),
    }

    t0 = time.perf_counter()
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = worker.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    decode_ms = (time.perf_counter() - t0) * 1000.0

    total_ms = (time.perf_counter() - overall_t0) * 1000.0

    report = {
        "image": str(image_path),
        "warmup_runs": args.warmup_runs,
        "fixed_page": fixed_page_meta,
        "tt_device": str(worker.tt_device),
        "tt_runtime": worker.tt_runtime,
        "warmups": warmups,
        "hot_run": {
            "timings_ms": {
                "build_inputs_ms": round(build_inputs_ms, 3),
                "h2d_ms": round(h2d_ms, 3),
                "tt_vision_ms": round(tt_vision_ms, 3),
                "vision_d2h_ms": round(vision_d2h_ms, 3),
                "inject_ms": round(inject_ms, 3),
                "cpu_generate_ms": round(cpu_generate_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "generated_token_count": int(generated_ids_trimmed[0].shape[0]),
            "output_text": output_text,
            "tt_vision_profile": tt_vision_profile,
            "decoder_profile": decoder_profile,
        },
    }

    json_out = Path(args.json_out).resolve()
    txt_out = Path(args.txt_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_out.write_text(render_text_summary(report), encoding="utf-8")
    print(render_text_summary(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
