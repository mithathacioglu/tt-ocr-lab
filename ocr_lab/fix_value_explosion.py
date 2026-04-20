#!/usr/bin/env python3
"""
Fix dots.mocr value explosion by scaling norm weights.

Analysis: Values grow from [-5,5] at block 1 to [-2336,1680] at block 42.
This 400x growth happens because residual connections accumulate without bound.

Fix: Scale down the RMSNorm weights in deeper layers. This reduces the
magnitude of attention/MLP outputs, keeping the residual stream bounded.

No training needed — just analytical weight modification.
"""
import os, sys, time, json, torch
import torch.nn.functional as F
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
sys.path.insert(0, 'ocr_lab/shims')

from dots_tt_port.vision_tower_smoke import load_vision_model, resolve_snapshot_dir

torch.set_num_threads(32)
sd = resolve_snapshot_dir("rednote-hilab/dots.mocr")

# Load model
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)

# Load test image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message

proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
img_msg, _ = fixed_page_message("ocr_lab/tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "t"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"].to(torch.int32)

# Original: measure per-block value range
print("=== ORIGINAL model ===", flush=True)
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary = vm.rot_pos_emb(gt)
    cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)

    for i in range(42):
        h = vm.blocks[i](h, cu_seqlens=cu, rotary_pos_emb=rotary)
        if i in [0, 5, 10, 15, 20, 25, 30, 35, 41]:
            print(f"  Block {i+1:>2}: [{h.min():.0f}, {h.max():.0f}] std={h.std():.2f}", flush=True)

    # Get reference output
    if hasattr(vm, 'post_trunk_norm'):
        h_normed = vm.post_trunk_norm(h)
    cpu_orig = vm.merger(h_normed).float()

# Strategy: scale norm weights to keep values bounded
# After block N, values should stay within [-50, 50]
# Current growth: ~1.15x per block after block 10
# Target: ~1.0x per block (no growth)
# Fix: multiply norm1.weight and norm2.weight by decay factor for deep layers

print("\n=== Applying norm weight scaling ===", flush=True)
vm2, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm2 = vm2.eval().to(dtype=torch.bfloat16)

# Scale factors: no change for first 10 blocks, then increasingly aggressive
for i in range(42):
    if i < 10:
        scale = 1.0
    elif i < 20:
        scale = 0.95 ** (i - 10)  # 0.95^1 to 0.95^10
    elif i < 30:
        scale = 0.95 ** 10 * 0.90 ** (i - 20)  # more aggressive
    else:
        scale = 0.95 ** 10 * 0.90 ** 10 * 0.85 ** (i - 30)  # very aggressive

    vm2.blocks[i].norm1.weight.data *= scale
    vm2.blocks[i].norm2.weight.data *= scale

    if i in [0, 10, 20, 30, 41]:
        print(f"  Block {i}: scale={scale:.4f}", flush=True)

# Test scaled model
print("\n=== SCALED model ===", flush=True)
with torch.no_grad():
    h2 = vm2.patch_embed(pv, gt)
    rotary2 = vm2.rot_pos_emb(gt)

    for i in range(42):
        h2 = vm2.blocks[i](h2, cu_seqlens=cu, rotary_pos_emb=rotary2)
        if i in [0, 5, 10, 15, 20, 25, 30, 35, 41]:
            print(f"  Block {i+1:>2}: [{h2.min():.0f}, {h2.max():.0f}] std={h2.std():.2f}", flush=True)

    if hasattr(vm2, 'post_trunk_norm'):
        h2_normed = vm2.post_trunk_norm(h2)
    cpu_scaled = vm2.merger(h2_normed).float()

# Compare
cos = F.cosine_similarity(cpu_orig.flatten().unsqueeze(0), cpu_scaled.flatten().unsqueeze(0)).item()
print(f"\nScaled vs Original CPU cosine: {cos:.6f}")
print(f"Original range:  [{cpu_orig.min():.1f}, {cpu_orig.max():.1f}]")
print(f"Scaled range:    [{cpu_scaled.min():.1f}, {cpu_scaled.max():.1f}]")

# Save scaled weights
torch.save(vm2.state_dict(), "ocr_lab/qat_weights/vision_scaled_norms.pt")
print(f"\nSaved scaled weights")
