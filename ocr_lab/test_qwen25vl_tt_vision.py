#!/usr/bin/env python3
"""Test Qwen2.5-VL-3B vision on TT using the native DropIn pipeline.

This uses TT-metal's own Qwen2.5-VL implementation — no adapter hacks needed.
"""
import os, sys, time, torch, torch.nn.functional as F
os.environ['TT_METAL_LOGGER_LEVEL'] = 'FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TT = os.path.join(P, "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, TT); sys.path.insert(0, P)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE = os.path.join(P, "ocr_lab/tmp_1pdf_page-1.png")

# CPU reference first
print("=== CPU REFERENCE ===", flush=True)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.set_num_threads(32)
proc = AutoProcessor.from_pretrained(MODEL)
hf = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
print(f"  vision blocks: {len(hf.visual.blocks)}", flush=True)

PROMPT = "Extract the text content from this image."
msgs = [{"role": "user", "content": [
    {"type": "image", "image": IMAGE},
    {"type": "text", "text": PROMPT},
]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
pv = inputs["pixel_values"]
gt = inputs["image_grid_thw"]
print(f"  pv={pv.shape} gt={gt}", flush=True)

# Get CPU vision embeddings
t0 = time.perf_counter()
with torch.no_grad():
    cpu_vis = hf.visual(pv.float(), gt).float()
cpu_vis_ms = (time.perf_counter() - t0) * 1000
print(f"  CPU vision: {cpu_vis_ms:.0f}ms shape={cpu_vis.shape}", flush=True)
del hf

# === TT DropIn Vision ===
print("\n=== TT DROPIN VISION ===", flush=True)
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()

# This time use Qwen2.5-VL's own model — no dots.mocr adapter
os.environ["HF_MODEL"] = MODEL
args = VisionModelArgs(mesh, instruct=True, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
ref = args.reference_vision_model()
print(f"  ref blocks: {len(ref.blocks)}", flush=True)

tt_vis = DropInVisionTransformer(ref, args, debug=False)

# Cold
print("  Cold run...", flush=True)
with torch.no_grad():
    t0 = time.perf_counter()
    tt_out = tt_vis(pv.to(torch.bfloat16), gt)
    cold_ms = (time.perf_counter() - t0) * 1000
print(f"  Cold: {cold_ms:.0f}ms shape={tt_out.shape}", flush=True)

# Warm
print("  Warm run...", flush=True)
with torch.no_grad():
    t0 = time.perf_counter()
    tt_out2 = tt_vis(pv.to(torch.bfloat16), gt)
    warm_ms = (time.perf_counter() - t0) * 1000
tt_cpu = tt_out2.float()
print(f"  Warm: {warm_ms:.0f}ms shape={tt_cpu.shape}", flush=True)

# Compare
cos_sim = F.cosine_similarity(cpu_vis.flatten().unsqueeze(0), tt_cpu.flatten().unsqueeze(0))
diff = (cpu_vis - tt_cpu).abs()
print(f"\n=== COMPARISON ===")
print(f"  Cosine:   {cos_sim.item():.6f}")
print(f"  Max diff: {diff.max():.4f}")
print(f"  CPU vis:  [{cpu_vis.min():.1f}, {cpu_vis.max():.1f}]")
print(f"  TT vis:   [{tt_cpu.min():.1f}, {tt_cpu.max():.1f}]")
print(f"\n=== TIMING ===")
print(f"  CPU:     {cpu_vis_ms:.0f}ms ({cpu_vis_ms/1000:.1f}s)")
print(f"  TT warm: {warm_ms:.0f}ms ({warm_ms/1000:.1f}s)")
print(f"  Speedup: {cpu_vis_ms/warm_ms:.1f}x")

if cos_sim.item() > 0.95:
    print("  >>> GOOD — worth testing E2E OCR")
elif cos_sim.item() > 0.8:
    print("  >>> MODERATE — may work for OCR")
else:
    print("  >>> POOR — same drift problem")

ttnn.close_mesh_device(mesh)
