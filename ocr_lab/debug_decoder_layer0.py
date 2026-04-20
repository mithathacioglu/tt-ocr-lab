#!/usr/bin/env python3
"""Debug: Compare decoder layer 0 output CPU vs TT to find divergence source."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL = PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
sys.path.insert(0, str(TT_METAL))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()

# --- CPU Reference: get layer 0 input/output ---
print("=== CPU Reference ===", flush=True)
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message
from dots_tt_port.vision_tower_smoke import resolve_snapshot_dir, load_vision_model

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
img_msg, _ = fixed_page_message(PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

# Vision embeddings (CPU, proven correct)
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    vision_emb = vm(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"].to(torch.int32)).to(torch.bfloat16)
del vm

# HF model
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_mask = inputs["input_ids"] == hf.config.image_token_id
cpu_embeds = F.embedding(inputs["input_ids"], emb_w)
cpu_embeds = cpu_embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(cpu_embeds), vision_emb.to(cpu_embeds.dtype))
print(f"  inputs_embeds: {cpu_embeds.shape}", flush=True)

# Hook layer 0 to capture input/output
layer0_input = []
layer0_output = []

def hook_pre(module, args, kwargs):
    # Capture hidden_states input to layer 0
    if args:
        layer0_input.append(args[0].detach().clone())
    return None

def hook_post(module, args, kwargs, output):
    if isinstance(output, tuple):
        layer0_output.append(output[0].detach().clone())
    else:
        layer0_output.append(output.detach().clone())
    return None

hf.model.layers[0].register_forward_pre_hook(hook_pre, with_kwargs=True)
hf.model.layers[0].register_forward_hook(hook_post, with_kwargs=True)

import types
orig_fwd = hf.forward
def pfwd(self, input_ids=None, inputs_embeds=None, **kw):
    if input_ids is None and inputs_embeds is not None:
        input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    return orig_fwd(input_ids=input_ids, inputs_embeds=inputs_embeds, **kw)
hf.forward = types.MethodType(pfwd, hf)

with torch.no_grad():
    cpu_logits = hf(input_ids=inputs["input_ids"], inputs_embeds=cpu_embeds).logits

cpu_first_token = cpu_logits[0, -1, :].float().argmax(-1).item()
tokenizer = getattr(proc, "tokenizer", proc)
print(f"  CPU first token: {cpu_first_token} = {repr(tokenizer.decode([cpu_first_token]))}")
print(f"  Layer 0 input shape: {layer0_input[0].shape}")
print(f"  Layer 0 output shape: {layer0_output[0].shape}")
print(f"  Layer 0 input stats: min={layer0_input[0].float().min():.4f} max={layer0_input[0].float().max():.4f}")
print(f"  Layer 0 output stats: min={layer0_output[0].float().min():.4f} max={layer0_output[0].float().max():.4f}")

del hf

# --- TT Decoder: get state dict and check weight format ---
print("\n=== TT State Dict Check ===", flush=True)
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys, convert_hf_to_meta, convert_hf_to_meta_no_qkv_permute

dev = ttnn.open_device(device_id=0)
args = ModelArgs(dev, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)

# Load raw HF state dict (before conversion)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict
raw_sd = load_hf_state_dict(args.CKPT_DIR)

# Strip model. prefix
stripped = {}
for k, v in raw_sd.items():
    stripped[k[len("model."):] if k.startswith("model.") else k] = v

# Check what keys exist for layer 0
l0_keys = sorted([k for k in stripped if k.startswith("layers.0.")])
print(f"  Raw HF layer 0 keys: {l0_keys}")

# Try both conversions and compare QKV weights
std = standardize_hf_keys(stripped)
meta_permuted = convert_hf_to_meta(std.copy(), args.head_dim)
meta_noperm = convert_hf_to_meta_no_qkv_permute(std.copy(), args.head_dim)

wq_perm = meta_permuted.get("layers.0.attention.wq.weight")
wq_noperm = meta_noperm.get("layers.0.attention.wq.weight")

if wq_perm is not None and wq_noperm is not None:
    diff = (wq_perm.float() - wq_noperm.float()).abs()
    print(f"\n  QKV weight difference (permuted vs no-permute):")
    print(f"    wq max_diff: {diff.max():.6f}")
    print(f"    Same: {torch.equal(wq_perm, wq_noperm)}")

# Check: does the original HF model use fused qkv?
qkv_key = [k for k in stripped if "q_proj" in k and "layers.0" in k]
fused_qkv = [k for k in stripped if "qkv" in k and "layers.0" in k]
print(f"\n  HF uses separate q/k/v: {len(qkv_key) > 0}")
print(f"  HF uses fused qkv: {len(fused_qkv) > 0}")
if qkv_key:
    print(f"    Keys: {qkv_key}")
if fused_qkv:
    print(f"    Keys: {fused_qkv}")

# Compare embedding input to TT
# The TT model gets the same inputs_embeds, so layer 0 input should be the same
# unless embedding or vision injection differs
print(f"\n  CPU layer 0 input [0,0,:5]: {layer0_input[0][0,0,:5].float()}")

ttnn.close_device(dev)
print("\nDone. Use these results to determine correct weight format.")
