#!/usr/bin/env python3
"""Compare DropIn TT vision embeddings with CPU reference.

Goal: find exactly where DropIn diverges so we can fix it.
Tests at 476x674 (small, fast) first.
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

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message
from dots_tt_port.vision_tower_smoke import load_vision_model

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

# Build inputs at 476x674 (fast test)
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "test"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"]
print(f"pixel_values: {pv.shape}, grid_thw: {gt}")

# === CPU reference ===
print("\n=== CPU REFERENCE ===")
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    t0 = time.perf_counter()
    ref = vm(pv, gt.to(torch.int32))
    cpu_ms = (time.perf_counter() - t0) * 1000
ref_cpu = ref.float()
print(f"  CPU: {cpu_ms:.0f}ms, shape={ref_cpu.shape}")
print(f"  Stats: min={ref_cpu.min():.4f} max={ref_cpu.max():.4f} mean={ref_cpu.mean():.4f}")

# === DropIn TT vision ===
print("\n=== DROPIN TT VISION ===")
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()

print("Creating VisionModelArgs...", flush=True)
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
ref_model = args.reference_vision_model()

print("Creating DropInVisionTransformer...", flush=True)
tt_vision = DropInVisionTransformer(ref_model, args, debug=False)

# Cold run
print("Cold run...", flush=True)
with torch.no_grad():
    t0 = time.perf_counter()
    tt_out = tt_vision(pv, gt)
    cold_ms = (time.perf_counter() - t0) * 1000
print(f"  Cold: {cold_ms:.0f}ms, shape={tt_out.shape}")

# Warm run
print("Warm run...", flush=True)
with torch.no_grad():
    t0 = time.perf_counter()
    tt_out2 = tt_vision(pv, gt)
    warm_ms = (time.perf_counter() - t0) * 1000
tt_cpu = tt_out2.float()
print(f"  Warm: {warm_ms:.0f}ms, shape={tt_cpu.shape}")
print(f"  Stats: min={tt_cpu.min():.4f} max={tt_cpu.max():.4f} mean={tt_cpu.mean():.4f}")

# === Compare ===
print(f"\n=== COMPARISON ===")
if ref_cpu.shape == tt_cpu.shape:
    diff = (ref_cpu - tt_cpu).abs()
    cos_sim = F.cosine_similarity(ref_cpu.flatten().unsqueeze(0), tt_cpu.flatten().unsqueeze(0))

    # Per-token cosine similarity
    per_tok_cos = F.cosine_similarity(ref_cpu, tt_cpu, dim=-1)

    print(f"  Shape match: True")
    print(f"  Max diff:       {diff.max():.4f}")
    print(f"  Mean diff:      {diff.mean():.4f}")
    print(f"  Global cosine:  {cos_sim.item():.6f}")
    print(f"  Per-token cos:  min={per_tok_cos.min():.4f} mean={per_tok_cos.mean():.4f}")

    if cos_sim.item() > 0.99:
        print("  >>> Good accuracy — might work for OCR")
    elif cos_sim.item() > 0.95:
        print("  >>> Moderate accuracy — likely degraded OCR")
    else:
        print("  >>> Poor accuracy — OCR will be wrong")
else:
    print(f"  Shape MISMATCH: CPU={ref_cpu.shape} TT={tt_cpu.shape}")

print(f"\n=== TIMING ===")
print(f"  CPU:       {cpu_ms:.0f}ms")
print(f"  TT cold:   {cold_ms:.0f}ms")
print(f"  TT warm:   {warm_ms:.0f}ms")
print(f"  Speedup:   {cpu_ms/warm_ms:.1f}x")

ttnn.close_mesh_device(mesh)
