#!/usr/bin/env python3
"""A/B test: TowerHybridCpuUnary vs TowerHybridOptimized."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dots_tt_port.vision_tower_smoke import (
    TowerHybridCpuUnary, ensure_tt_env, load_vision_model, resolve_snapshot_dir,
)
from ocr_lab.vision_optimized import TowerHybridOptimized
from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import fixed_page_message

SNAPSHOT_DIR = resolve_snapshot_dir("rednote-hilab/dots.mocr")
IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"


def bench_vision(name, model_class, tt_device, pv_tt, gt_tt, torch_xla, runs=3):
    vm, _ = load_vision_model(SNAPSHOT_DIR, "sdpa", limit_layers=0)
    model = model_class(vm).eval().to(dtype=torch.bfloat16).to(tt_device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(pv_tt, gt_tt)
            torch_xla.sync(wait=True)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            out = model(pv_tt, gt_tt)
            torch_xla.sync(wait=True)
            times.append((time.perf_counter() - t0) * 1000)

    out_cpu = out.detach().cpu().float()
    avg = sum(times) / len(times)
    print(f"  {name}: avg={avg:.0f}ms  runs={[f'{t:.0f}' for t in times]}  shape={out_cpu.shape}")
    return avg, out_cpu


def main():
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    ensure_tt_env("0")
    configure_tt_runtime(cache_dir=None, enable_trace=False, optimization_level=2)
    import torch_xla.runtime as xr
    xr.set_device_type("TT")
    import torch_xla
    tt_device = torch_xla.device()

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(str(SNAPSHOT_DIR), trust_remote_code=True)
    img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\nReturn plain text only.\n"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    pv_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
    gt_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(tt_device)

    print("=== A: TowerHybridCpuUnary (baseline) ===")
    t_a, out_a = bench_vision("baseline", TowerHybridCpuUnary, tt_device, pv_tt, gt_tt, torch_xla)

    print("\n=== B: TowerHybridOptimized (compiled CPU attn + 32 threads) ===")
    t_b, out_b = bench_vision("optimized", TowerHybridOptimized, tt_device, pv_tt, gt_tt, torch_xla)

    # Compare outputs
    if out_a.shape == out_b.shape:
        diff = (out_a - out_b).abs()
        cos_sim = torch.nn.functional.cosine_similarity(out_a.flatten().unsqueeze(0), out_b.flatten().unsqueeze(0))
        print(f"\n  Max diff: {diff.max():.6f}  Mean diff: {diff.mean():.6f}")
        print(f"  Cosine similarity: {cos_sim.item():.6f}")

    speedup = t_a / t_b if t_b > 0 else 0
    saved = t_a - t_b
    print(f"\n=== RESULT: {t_a:.0f}ms -> {t_b:.0f}ms ({speedup:.2f}x, saved {saved:.0f}ms) ===")


if __name__ == "__main__":
    main()
