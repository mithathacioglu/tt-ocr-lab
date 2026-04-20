#!/usr/bin/env python3
"""Test tt-metal native decoder with dots.mocr weights on 1.pdf vision embeddings."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL = PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
sys.path.insert(0, str(TT_METAL))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()

import ttnn
from models.demos.qwen25_vl.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

# --- Build vision embeddings using standalone ttnn (already proven correct) ---
from dots_tt_port.vision_tower_smoke import (
    resolve_snapshot_dir, load_vision_model, load_vision_state,
    apply_rotary_pos_emb_cached_cpu, CpuPatchMerger,
)
from ocr_lab.fixed_page import fixed_page_message
import torch.nn.functional as F

DIM, N_HEADS, HEAD_DIM, EPS, N_BLOCKS = 1536, 12, 128, 1e-5, 42
IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 16  # Start small for testing
REF = "T.C.\nBAKIRKÖY\n2. AİLE MA"  # First 16 tokens of reference

shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
sys.path.insert(0, str(shims_dir))
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

pixel_values = inputs["pixel_values"].to(torch.bfloat16)
grid_thw = inputs["image_grid_thw"]
input_ids = inputs["input_ids"]

# Get vision embeddings using CPU model (proven correct)
print("Getting CPU vision embeddings...", flush=True)
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    vision_emb = vm(pixel_values, grid_thw.to(torch.int32)).to(torch.bfloat16)
print(f"  Vision embeddings: {vision_emb.shape}", flush=True)

# Build inputs_embeds on CPU
emb_weight = torch.load(f"{snapshot_dir}/model-00001-of-00002.safetensors", weights_only=True) if False else None

# Use HF model's embedding
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
emb_w = hf_model.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_token_id = hf_model.config.image_token_id

img_mask = input_ids == img_token_id
inputs_embeds = F.embedding(input_ids, emb_w)
inputs_embeds = inputs_embeds.masked_scatter(
    img_mask.unsqueeze(-1).expand_as(inputs_embeds), vision_emb.to(inputs_embeds.dtype)
)
print(f"  inputs_embeds: {inputs_embeds.shape}", flush=True)  # [1, 427, 1536]

# Get rotary embeddings for decoder (Qwen2.5-VL style from HF)
# For simplicity, use position_ids = 0..seq_len
seq_len = inputs_embeds.shape[1]

del hf_model, vm  # Free memory

# --- Create native TT decoder ---
print("Opening TT device...", flush=True)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh_device.enable_program_cache()

print("Creating decoder ModelArgs...", flush=True)
args = ModelArgs(mesh_device, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
print(f"  {args.model_name}: dim={args.dim} layers={args.n_layers}")

print("Loading decoder state dict...", flush=True)
sd = args.load_state_dict()
print(f"  {len(sd)} keys")

print("Creating Transformer...", flush=True)
t0 = time.perf_counter()
model = Transformer(
    args=args, mesh_device=mesh_device, dtype=ttnn.bfloat8_b,
    state_dict=sd, weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
)
create_ms = (time.perf_counter() - t0) * 1000
print(f"  OK! {len(model.layers)} layers in {create_ms:.0f}ms", flush=True)

# --- Prefill ---
print(f"\nPrefill ({seq_len} tokens)...", flush=True)

# Pad seq_len to 128 boundary
padded_seq_len = ((seq_len + 127) // 128) * 128  # 512
inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padded_seq_len - seq_len))
print(f"  Padded: {seq_len} -> {padded_seq_len}", flush=True)
seq_len = padded_seq_len

# Compute rotary cos/sin for decoder
# _prepare_cos_sin expects [batch, 1, seq, head_dim] torch tensors
# Use HF-style RoPE computation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
import json

config_dict = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = config_dict.get("rope_theta", 1000000.0)
head_dim = args.head_dim

# Compute cos/sin manually
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
t = torch.arange(seq_len, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos_prefill = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
sin_prefill = emb.sin().unsqueeze(0).unsqueeze(0)
rot_mats = [cos_prefill, sin_prefill]

# Prepare inputs
tt_embeds, tt_rot_mats, tt_page_table, _ = model.prepare_inputs_prefill(
    inputs_embeds,  # [1, seq, dim]
    rot_mats,
)

from models.tt_transformers.tt.common import Mode

# get_last_token: index of last real token (before padding)
# Original seq was 427, padded to 512. Last real token is at index 426.
last_real_token = 427 - 1  # 426
# Round down to nearest 32 boundary (model slices at 32-aligned positions)
last_token_idx = (last_real_token // 32) * 32  # 416

t0 = time.perf_counter()
tt_logits = model.forward(
    tt_embeds,
    current_pos=None,
    rot_mats_global=tt_rot_mats,
    mode=Mode.PREFILL,
    get_last_token=last_token_idx,
)
prefill_ms = (time.perf_counter() - t0) * 1000
logits_cpu = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
print(f"  Prefill: {prefill_ms:.0f}ms, logits shape={logits_cpu.shape}", flush=True)

# Get first token — logits are [1, 1, 32, vocab_size], real token at offset (426 - 416) = 10
token_offset = last_real_token - last_token_idx  # 10
next_token = logits_cpu[0, 0, token_offset, :args.vocab_size].float().argmax(-1).item()
print(f"  First token: {next_token} = {repr(tokenizer.decode([next_token]))}", flush=True)

# === Warm prefill ===
print("Warm prefill...", flush=True)
# Need fresh KV cache — recreate model or reset cache
# For now just run forward again (KV cache will be wrong but timing is valid)
tt_embeds2, tt_rot_mats2, _, _ = model.prepare_inputs_prefill(inputs_embeds, rot_mats)
t0 = time.perf_counter()
tt_logits2 = model.forward(tt_embeds2, current_pos=None, rot_mats_global=tt_rot_mats2,
                            mode=Mode.PREFILL, get_last_token=last_token_idx)
warm_pf_ms = (time.perf_counter() - t0) * 1000
logits2 = ttnn.to_torch(tt_logits2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
warm_token = logits2[0, 0, token_offset, :args.vocab_size].float().argmax(-1).item()
print(f"  Warm prefill: {warm_pf_ms:.0f}ms, token={warm_token}={repr(tokenizer.decode([warm_token]))}", flush=True)

# CPU reference first token for comparison
print("CPU reference first token...", flush=True)
import types
hf_dec = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
orig_fwd = hf_dec.forward
def pfwd(self, input_ids=None, inputs_embeds=None, **kw):
    if input_ids is None and inputs_embeds is not None:
        input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    return orig_fwd(input_ids=input_ids, inputs_embeds=inputs_embeds, **kw)
hf_dec.forward = types.MethodType(pfwd, hf_dec)

# Re-build unpadded embeds for CPU
img_mask2 = inputs["input_ids"] == img_token_id
cpu_embeds = F.embedding(inputs["input_ids"], emb_w)
cpu_embeds = cpu_embeds.masked_scatter(img_mask2.unsqueeze(-1).expand_as(cpu_embeds), vision_emb.to(cpu_embeds.dtype))

with torch.no_grad():
    cpu_logits = hf_dec(input_ids=inputs["input_ids"], inputs_embeds=cpu_embeds).logits
cpu_first = cpu_logits[0, -1, :].float().argmax(-1).item()
print(f"  CPU ref first token: {cpu_first}={repr(tokenizer.decode([cpu_first]))}", flush=True)

print(f"\n{'='*60}")
print(f"  Cold prefill:  {prefill_ms:.0f}ms  token={next_token}={repr(tokenizer.decode([next_token]))}")
print(f"  Warm prefill:  {warm_pf_ms:.0f}ms  token={warm_token}={repr(tokenizer.decode([warm_token]))}")
print(f"  CPU reference:             token={cpu_first}={repr(tokenizer.decode([cpu_first]))}")
print(f"  Match: {warm_token == cpu_first}")
print(f"{'='*60}")

ttnn.close_mesh_device(mesh_device)
