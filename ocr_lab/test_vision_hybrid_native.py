#!/usr/bin/env python3
"""Test hybrid vision (TT matmul + CPU SDPA) at native resolution.

Compare output with CPU reference to verify accuracy.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL = PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
sys.path.insert(0, str(TT_METAL))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"

from dots_tt_port.vision_tower_smoke import load_vision_model, resolve_snapshot_dir

SNAPSHOT = resolve_snapshot_dir("rednote-hilab/dots.mocr")
IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

# --- Build inputs at NATIVE resolution ---
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

proc = AutoProcessor.from_pretrained(str(SNAPSHOT), trust_remote_code=True)
msgs = [{"role": "user", "content": [
    {"type": "image", "image": str(IMAGE)},
    {"type": "text", "text": "test"},
]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
pv = inputs["pixel_values"]
gt = inputs["image_grid_thw"]
print(f"Native: pixel_values={pv.shape}, grid_thw={gt}")

# --- CPU reference ---
print("\n=== CPU REFERENCE ===")
vm, _ = load_vision_model(SNAPSHOT, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
t0 = time.perf_counter()
with torch.no_grad():
    ref = vm(pv.to(torch.bfloat16), gt.to(torch.int32))
cpu_ms = (time.perf_counter() - t0) * 1000
ref_cpu = ref.float()
print(f"  CPU vision: {cpu_ms:.0f}ms ({cpu_ms/1000:.1f}s)")
print(f"  Output: {ref_cpu.shape}")

# --- TT Hybrid ---
print("\n=== TT HYBRID (TT matmul + CPU SDPA) ===")
import ttnn
from ocr_lab.ttnn_vision_hybrid import load_vision_weights, vision_forward_hybrid

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()

print("  Loading vision weights to TT...", flush=True)
t0 = time.perf_counter()
blocks_tt, final_tt = load_vision_weights(vm, mesh)
load_ms = (time.perf_counter() - t0) * 1000
print(f"  Weight load: {load_ms:.0f}ms")

# Cold run
print("  Cold run...", flush=True)
t0 = time.perf_counter()
with torch.no_grad():
    tt_out = vision_forward_hybrid(pv, gt.to(torch.int32), vm, blocks_tt, final_tt, mesh)
cold_ms = (time.perf_counter() - t0) * 1000
print(f"  Cold: {cold_ms:.0f}ms ({cold_ms/1000:.1f}s)")

# Warm run
print("  Warm run...", flush=True)
t0 = time.perf_counter()
with torch.no_grad():
    tt_out2 = vision_forward_hybrid(pv, gt.to(torch.int32), vm, blocks_tt, final_tt, mesh)
warm_ms = (time.perf_counter() - t0) * 1000
tt_cpu = tt_out2.float()
print(f"  Warm: {warm_ms:.0f}ms ({warm_ms/1000:.1f}s)")

# --- Compare ---
print(f"\n=== COMPARISON ===")
print(f"  CPU shape: {ref_cpu.shape}")
print(f"  TT  shape: {tt_cpu.shape}")

if ref_cpu.shape == tt_cpu.shape:
    diff = (ref_cpu - tt_cpu).abs()
    cos_sim = F.cosine_similarity(ref_cpu.flatten().unsqueeze(0), tt_cpu.flatten().unsqueeze(0))
    print(f"  Max diff:   {diff.max():.6f}")
    print(f"  Mean diff:  {diff.mean():.6f}")
    print(f"  Cosine sim: {cos_sim.item():.6f}")
    print(f"  Exact match (< 0.01): {diff.max().item() < 0.01}")

print(f"\n=== TIMING ===")
print(f"  CPU:    {cpu_ms:.0f}ms ({cpu_ms/1000:.1f}s)")
print(f"  TT:     {warm_ms:.0f}ms ({warm_ms/1000:.1f}s)")
print(f"  Speedup: {cpu_ms/warm_ms:.1f}x")

ttnn.close_mesh_device(mesh)
