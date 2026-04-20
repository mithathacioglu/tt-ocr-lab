#!/usr/bin/env python3
"""Use base TTTransformer (not Qwen2.5-VL subclass) with dots.mocr weights.
This avoids the VL-specific rope and uses standard HfRotarySetup."""
from __future__ import annotations
import os, sys, time, json, types
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

# --- Build vision embeddings + inputs_embeds on CPU ---
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message
from dots_tt_port.vision_tower_smoke import load_vision_model

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

print("Getting vision embeddings...", flush=True)
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    vision_emb = vm(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"].to(torch.int32)).to(torch.bfloat16)
del vm

hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_mask = inputs["input_ids"] == hf.config.image_token_id
cpu_embeds = F.embedding(inputs["input_ids"], emb_w)
cpu_embeds = cpu_embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(cpu_embeds), vision_emb.to(cpu_embeds.dtype))
seq_len = cpu_embeds.shape[1]  # 427
del hf

# Pad
padded = ((seq_len + 127) // 128) * 128
cpu_embeds_padded = F.pad(cpu_embeds, (0, 0, 0, padded - seq_len))

# --- TT decoder with BASE Transformer (not Qwen VL subclass) ---
print("Opening TT device...", flush=True)
import ttnn
from models.tt_transformers.tt.model import Transformer as TTTransformer
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.rope import HfRotarySetup
from models.tt_transformers.tt.common import Mode

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh_device.enable_program_cache()

args = ModelArgs(mesh_device, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
print(f"  {args.model_name}: dim={args.dim} layers={args.n_layers} use_hf_rope={args.use_hf_rope}")

sd = args.load_state_dict()
print(f"  State dict: {len(sd)} keys")

print("Creating base TTTransformer with HfRotarySetup...", flush=True)
t0 = time.perf_counter()
model = TTTransformer(
    args=args,
    mesh_device=mesh_device,
    dtype=ttnn.bfloat8_b,
    state_dict=sd,
    weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
    attention_class=Attention,
    rope_setup_class=HfRotarySetup,
)
print(f"  OK! {len(model.layers)} layers in {(time.perf_counter()-t0)*1000:.0f}ms")

# --- Prefill ---
print(f"\nPrefill ({padded} tokens, real={seq_len})...", flush=True)

# Get cos/sin for standard positions
last_real = seq_len - 1
last_token_idx = (last_real // 32) * 32

# Prepare input embedding as ttnn tensor
tt_embeds = ttnn.from_torch(
    cpu_embeds_padded.unsqueeze(1),  # [1, 1, padded, 1536]
    device=mesh_device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
)

# Build cos/sin for HF-style RoPE prefill
# HfRotarySetup stores cos/sin as ttnn embedding tables
# For prefill, attention._hf_rope_prefill slices from rot_mats
# rot_mats format: [cos_ttnn, sin_ttnn] where each is [1, 1, seq, head_dim]
config_dict = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = config_dict.get("rope_theta", 1000000.0)
head_dim = args.head_dim

inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
t = torch.arange(padded, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos_pt = emb.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1, 1, padded, head_dim]
sin_pt = emb.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

cos_tt = ttnn.from_torch(cos_pt, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
sin_tt = ttnn.from_torch(sin_pt, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
rot_mats_prefill = [cos_tt, sin_tt]

print("Forward...", flush=True)
t0 = time.perf_counter()
tt_logits = model.forward(
    tt_embeds,
    current_pos=None,
    rot_mats_global=rot_mats_prefill,
    mode=Mode.PREFILL,
    get_last_token=last_token_idx,
)
fwd_ms = (time.perf_counter() - t0) * 1000
logits = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
print(f"  Forward: {fwd_ms:.0f}ms, logits={logits.shape}")

token_offset = last_real - last_token_idx
tt_token = logits[0, 0, token_offset, :args.vocab_size].float().argmax(-1).item()
print(f"  TT token: {tt_token} = {repr(tokenizer.decode([tt_token]))}")
print(f"  Expected: 51 = {repr(tokenizer.decode([51]))}")
print(f"  Match: {tt_token == 51}")

# Warm
print("\nWarm prefill...", flush=True)
tt_embeds2 = ttnn.from_torch(
    cpu_embeds_padded.unsqueeze(1),
    device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
)
t0 = time.perf_counter()
tt_logits2 = model.forward(tt_embeds2, current_pos=None, rot_mats_global=rot_mats_prefill,
                            mode=Mode.PREFILL, get_last_token=last_token_idx)
warm_ms = (time.perf_counter() - t0) * 1000
logits2 = ttnn.to_torch(tt_logits2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
warm_token = logits2[0, 0, token_offset, :args.vocab_size].float().argmax(-1).item()
print(f"  Warm: {warm_ms:.0f}ms, token={warm_token}={repr(tokenizer.decode([warm_token]))}")

print(f"\n{'='*50}")
print(f"  Cold: {fwd_ms:.0f}ms  Warm: {warm_ms:.0f}ms")
print(f"  Token: {warm_token} = {repr(tokenizer.decode([warm_token]))}")
print(f"  Expected: 51 = 'T'")
print(f"  Match: {warm_token == 51}")
print(f"{'='*50}")

ttnn.close_mesh_device(mesh_device)
