#!/usr/bin/env python3
"""Full pipeline: ttnn vision (standalone) + ttnn decoder (hybrid) + generate text."""
from __future__ import annotations
import os, sys, time, json
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

from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from dots_tt_port.vision_tower_smoke import (
    resolve_snapshot_dir, load_vision_model, load_vision_state,
    apply_rotary_pos_emb_cached_cpu, CpuPatchMerger,
)
from ocr_lab.fixed_page import fixed_page_message

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_TOKENS = 64
REF = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli"

# --- Build inputs ---
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

# --- Vision (standalone ttnn — proven correct) ---
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

DIM_V, N_HEADS_V, HEAD_DIM_V, EPS_V, N_BLOCKS_V = 1536, 12, 128, 1e-5, 42

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

# Load vision
print("Loading vision...", flush=True)
vision_model, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vision_model = vision_model.eval().to(dtype=torch.bfloat16)
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"]

# Patch embed + rotary (CPU)
with torch.no_grad():
    hidden_v = vision_model.patch_embed(pv, gt.to(torch.int32))
seq_v = hidden_v.shape[0]
pos_ids = vision_model.get_pos_ids_by_grid(gt.cpu())
pos_ids = torch.cat(pos_ids, dim=0)
rotary_full = vision_model.rotary_pos_emb(int(gt.cpu()[:, 1:].max().item())).cpu()
rotary = rotary_full[pos_ids].flatten(1).float()
cos_v = rotary.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
sin_v = rotary.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
post_norm_w = vision_model.post_trunk_norm.weight.detach().cpu()
cpu_merger = CpuPatchMerger(vision_model.merger)

# Vision weights
vs = load_vision_state(snapshot_dir, limit_layers=0)

def make_vw(t): return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
vblocks = []
for i in range(N_BLOCKS_V):
    p = f"blocks.{i}."
    w = {}
    w["norm1"] = ttnn.from_torch(vs[f"{p}norm1.weight"].unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w["norm2"] = ttnn.from_torch(vs[f"{p}norm2.weight"].unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w["qkv_w"] = make_vw(vs[f"{p}attn.qkv.weight"].T)
    w["proj_w"] = make_vw(vs[f"{p}attn.proj.weight"].T)
    w["fc1_w"] = make_vw(vs[f"{p}mlp.fc1.weight"].T)
    w["fc2_w"] = make_vw(vs[f"{p}mlp.fc2.weight"].T)
    w["fc3_w"] = make_vw(vs[f"{p}mlp.fc3.weight"].T)
    vblocks.append(w)
post_norm_tt = ttnn.from_torch(post_norm_w.unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Vision forward (from test_ttnn_vision_e2e_1pdf.py pattern)
def vision_block(x, w, dev, seq, cos, sin):
    norm1 = ttnn.rms_norm(x, epsilon=EPS_V, weight=w["norm1"])
    qkv = ttnn.linear(norm1, w["qkv_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(norm1)
    qkv_cpu = ttnn.to_torch(qkv).float(); ttnn.deallocate(qkv)
    q = qkv_cpu[..., :DIM_V].reshape(seq, N_HEADS_V, HEAD_DIM_V)
    k = qkv_cpu[..., DIM_V:2*DIM_V].reshape(seq, N_HEADS_V, HEAD_DIM_V)
    v = qkv_cpu[..., 2*DIM_V:].reshape(seq, N_HEADS_V, HEAD_DIM_V)
    q = apply_rotary_pos_emb_cached_cpu(q.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
    k = apply_rotary_pos_emb_cached_cpu(k.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
    attn = F.scaled_dot_product_attention(q.transpose(0,1), k.transpose(0,1), v.float().transpose(0,1), dropout_p=0.0)
    attn = attn.transpose(0,1).reshape(1,1,seq,DIM_V).to(torch.bfloat16)
    attn_tt = ttnn.from_torch(attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    proj = ttnn.linear(attn_tt, w["proj_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(attn_tt)
    h = ttnn.add(x, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(proj); ttnn.deallocate(x)
    norm2 = ttnn.rms_norm(h, epsilon=EPS_V, weight=w["norm2"])
    fc1 = ttnn.linear(norm2, w["fc1_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fc3 = ttnn.linear(norm2, w["fc3_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(norm2)
    gated = ttnn.mul(fc1, fc3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(fc1); ttnn.deallocate(fc3)
    fc2 = ttnn.linear(gated, w["fc2_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(gated)
    out = ttnn.add(h, fc2, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(h); ttnn.deallocate(fc2)
    return out

# Run vision
print("Vision forward...", flush=True)
tt_h = ttnn.from_torch(hidden_v.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Cold
t0 = time.perf_counter()
for bw in vblocks:
    tt_h = vision_block(tt_h, bw, device, seq_v, cos_v, sin_v)
tt_h = ttnn.rms_norm(tt_h, epsilon=EPS_V, weight=post_norm_tt)
vision_cold = (time.perf_counter() - t0) * 1000
vo = ttnn.to_torch(tt_h).squeeze(0).squeeze(0)[:seq_v].float()
vision_emb = cpu_merger(vo.to(torch.bfloat16))

# Warm vision
tt_h2 = ttnn.from_torch(hidden_v.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
t0 = time.perf_counter()
for bw in vblocks:
    tt_h2 = vision_block(tt_h2, bw, device, seq_v, cos_v, sin_v)
tt_h2 = ttnn.rms_norm(tt_h2, epsilon=EPS_V, weight=post_norm_tt)
vision_warm = (time.perf_counter() - t0) * 1000
vo2 = ttnn.to_torch(tt_h2).squeeze(0).squeeze(0)[:seq_v].float()
vision_emb = cpu_merger(vo2.to(torch.bfloat16))
print(f"  Cold: {vision_cold:.0f}ms  Warm: {vision_warm:.0f}ms  shape={vision_emb.shape}", flush=True)

# --- Build decoder input ---
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
del hf
img_mask = inputs["input_ids"] == 151665  # image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_emb.to(embeds.dtype))
seq_d = embeds.shape[1]

# Decoder rotary
cfg = json.loads((Path(str(snapshot_dir)) / 'config.json').read_text())
inv_freq = 1.0 / (cfg.get('rope_theta',1e6) ** (torch.arange(0,128,2,dtype=torch.float32)/128))
freqs = torch.outer(torch.arange(seq_d,dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_d = emb_rope.cos().unsqueeze(1).unsqueeze(0)
sin_d = emb_rope.sin().unsqueeze(1).unsqueeze(0)

# --- Decoder (standalone ttnn hybrid) ---
# Close vision device, reopen as mesh for decoder
ttnn.close_device(device)
print("Loading decoder...", flush=True)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
sd = args.load_state_dict()
from ocr_lab.ttnn_decoder_hybrid import load_decoder_weights, prefill
dec_layers, dec_final = load_decoder_weights(sd, mesh)

# Cold prefill
embeds_tt = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print("Decoder prefill (cold)...", flush=True)
t0 = time.perf_counter()
logits_tt = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_d, sin_d, seq_d)
dec_cold = (time.perf_counter() - t0) * 1000
logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq_d-1, :cfg.get('vocab_size',151936)].float()
tok = logits_cpu.argmax(-1).item()
print(f"  Cold: {dec_cold:.0f}ms  token={tok}={repr(tokenizer.decode([tok]))}", flush=True)

# Warm prefill
embeds_tt2 = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print("Decoder prefill (warm)...", flush=True)
t0 = time.perf_counter()
logits_tt2 = prefill(embeds_tt2, dec_layers, dec_final, mesh, cos_d, sin_d, seq_d)
dec_warm = (time.perf_counter() - t0) * 1000
logits2 = ttnn.to_torch(logits_tt2)[0, 0, seq_d-1, :cfg.get('vocab_size',151936)].float()
tok2 = logits2.argmax(-1).item()
print(f"  Warm: {dec_warm:.0f}ms  token={tok2}={repr(tokenizer.decode([tok2]))}", flush=True)

# --- Results ---
total = vision_warm + dec_warm
print(f"\n{'='*60}")
print(f"  Vision (warm):   {vision_warm:.0f}ms")
print(f"  Decoder (warm):  {dec_warm:.0f}ms")
print(f"  TOTAL:           {total:.0f}ms")
print(f"  First token:     {tok2}={repr(tokenizer.decode([tok2]))}")
print(f"  Match:           {tok2 == 51}")
print(f"{'='*60}")

ttnn.close_mesh_device(mesh)
