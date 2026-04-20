#!/usr/bin/env python3
"""
Diagnostic probe: compare HF vs TT decoder logits at steps 0, 5, 10 after a
native-resolution vision prefill, to localize where TT decode diverges at
long sequence lengths (~2791 tokens).

Env knobs (mirror run_hybrid_fast.py):
  NATIVE=1       use native image resolution
  HF_DECODE=1    (unused here — we always run BOTH HF and TT for comparison)
  N_TT=10        number of TT vision blocks (rest on CPU)
  MESH_SHAPE=1,1
"""
import os, sys, time, json, torch, torch.nn.functional as F
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TT_METAL_LOGGER_LEVEL', 'FATAL')
os.environ.setdefault('NATIVE', '1')
os.environ.setdefault('N_TT', '10')

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P + "/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, P)
sys.path.insert(0, P + "/ocr_lab/shims")
torch.set_num_threads(32)

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
sd = patch_dots_mocr_config_for_tt_metal()
from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.dots_model import ensure_dots_ocr_processor_compat
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message
from ocr_lab.inference_config import *

ensure_dots_ocr_processor_compat(str(sd))
proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
tokenizer = getattr(proc, 'tokenizer', proc)

W = int(os.environ.get("IMG_W", 672)); H = int(os.environ.get("IMG_H", 952))
if os.environ.get("NATIVE") == "1":
    img_msg = {"type": "image", "image": P + '/ocr_lab/tmp_1pdf_page-1.png'}
else:
    img_msg, _ = fixed_page_message(P + '/ocr_lab/tmp_1pdf_page-1.png', width=W, height=H)
msgs = [{"role": "user", "content": [img_msg, {"type": "text",
    "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16); gt = inputs["image_grid_thw"].to(torch.int32)
seq = inputs["input_ids"].shape[1]
print(f"{W}x{H}: pv={pv.shape} seq={seq}", flush=True)

# ========== TT Setup ==========
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, decode_step_tt, precompute_attn_masks,
    init_tt_kv_cache, fill_tt_kv_cache,
    DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS)

MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH_SHAPE", "1,1").split(","))
print(f"Opening mesh {MESH_SHAPE}", flush=True)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE)); mesh.enable_program_cache()

_orig_to_torch = ttnn.to_torch
def _patched_to_torch(tensor, *args, **kwargs):
    if "mesh_composer" not in kwargs and "device" not in kwargs:
        try:
            return _orig_to_torch(tensor, *args, **kwargs)
        except RuntimeError as e:
            if "mesh composer" in str(e) or "buffers.size() == 1" in str(e):
                shards = ttnn.get_device_tensors(tensor)
                return _orig_to_torch(shards[0], *args, **kwargs)
            raise
    return _orig_to_torch(tensor, *args, **kwargs)
ttnn.to_torch = _patched_to_torch

# Vision TT setup
vis_args = VisionModelArgs(mesh, instruct=False, dummy_weights=False,
                           max_batch_size=1, max_seq_len=int(os.environ.get("MAX_SEQ", 2048)))
ref = vis_args.reference_vision_model()
tt_vis = DropInVisionTransformer(ref, vis_args, debug=False)

unp = (gt[:, 1] * gt[:, 2]).sum().item(); sl = ((unp // 2048) + 1) * 2048; dim = 1536
pos_ids = ref.get_pos_ids_by_grid(gt.cpu()); pos_ids = torch.cat(pos_ids, dim=0)
mg = int(gt.cpu()[:, 1:].max().item()); rf = ref.rotary_pos_emb(mg).cpu()
rot = rf[pos_ids].flatten(1).float()
ch = rot.cos().unsqueeze(1).repeat(1, 1, 2); sh = rot.sin().unsqueeze(1).repeat(1, 1, 2)
cm, sm = convert_rope_style_hf_to_meta(ch, sh)
cp = F.pad(cm, (0, 0, 0, sl - unp), value=1).unsqueeze(0).unsqueeze(0)
sp = F.pad(sm, (0, 0, 0, sl - unp), value=0).unsqueeze(0).unsqueeze(0)
_rot_mapper = ttnn.ReplicateTensorToMesh(mesh) if MESH_SHAPE != (1, 1) else ttnn.ShardTensorToMesh(mesh, dim=0)
cos_tt = ttnn.from_torch(cp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=_rot_mapper)
sin_tt = ttnn.from_torch(sp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=_rot_mapper)
cs, cws, _, wi = qwen2_5_vision_transformer_preprocess(seq_len=unp, grid_thw=gt,
    head_dim=vis_args.vision_head_dim,
    spatial_merge_size=vis_args.hf_config.vision_config.spatial_merge_size,
    window_size=vis_args.hf_config.vision_config.window_size,
    patch_size=vis_args.hf_config.vision_config.patch_size)
cu_tt = ttnn.from_torch(cs, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

# Decoder TT setup
dec_args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
dec_sd = dec_args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(dec_sd, mesh)

cfg = json.loads(__import__('pathlib').Path(str(sd), 'config.json').read_text())
vocab_size = cfg.get('vocab_size', 151936)
rope_theta = cfg.get("rope_theta", 1e6)
MAX_NEW = int(os.environ.get("MAX_NEW", 180))
max_pos = seq + MAX_NEW + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)
cos_full_tt = ttnn.from_torch(cos_full.unsqueeze(0).unsqueeze(0), device=mesh,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
sin_full_tt = ttnn.from_torch(sin_full.unsqueeze(0).unsqueeze(0), device=mesh,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# CPU vision model (fp32)
vm, _ = load_vision_model(sd, 'sdpa', limit_layers=0); vm = vm.eval().to(torch.float32)
rotary_cpu = vm.rot_pos_emb(gt)
cu_cpu = F.pad(torch.repeat_interleave(gt[:, 1] * gt[:, 2], gt[:, 0]).cumsum(0, dtype=torch.int32),
               (1, 0), value=0)

# HF model for prefill + reference decode
hf = AutoModelForCausalLM.from_pretrained(str(sd), trust_remote_code=True,
                                          torch_dtype=torch.float32, attn_implementation='sdpa').eval()
hf.vision_tower = hf.vision_tower.to(torch.float32)
emb_w = hf.get_input_embeddings().weight.detach().to(torch.bfloat16)

# ========== 1. TT vision (N_TT blocks) ==========
N_TT = int(os.environ.get("N_TT", 10))
t0 = time.perf_counter()
if N_TT > 0:
    pe = ref.patch_embed(pv); x = tt_vis.tt_model.prepare_input(pe, wi, sl)
    for i in range(N_TT):
        x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
    raw = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    h = raw[:, 0:1, :unp, :dim].squeeze(0).squeeze(0).to(torch.float32)
    ttnn.deallocate(x)
else:
    h = vm.patch_embed(pv.to(torch.float32), gt).to(torch.float32)
print(f"TT vision ({N_TT} blk): {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

# ========== 2. CPU vision (remaining blocks) ==========
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(N_TT, 42):
        h = vm.blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
    if hasattr(vm, 'post_trunk_norm'):
        h = vm.post_trunk_norm(h)
    vision_out = vm.merger(h)
print(f"CPU vision: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

# ========== 3. HF prefill ==========
emb_w_fp32 = hf.get_input_embeddings().weight.detach()
img_mask = inputs["input_ids"] == hf.config.image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w_fp32)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds),
                               vision_out.to(embeds.dtype))
t0 = time.perf_counter()
with torch.no_grad():
    outputs = hf(input_ids=inputs["input_ids"], inputs_embeds=embeds, use_cache=True)
print(f"HF prefill: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
first_tok = outputs.logits[0, -1, :vocab_size].float().argmax(-1).item()
past_kv = outputs.past_key_values
print(f"First predicted token: {first_tok} ({tokenizer.decode([first_tok])!r})", flush=True)

# ========== 4. Fill TT KV cache from HF past_kv ==========
max_cache = seq + MAX_NEW + 32
max_cache = ((max_cache + 31) // 32) * 32
tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)
for i in range(N_LAYERS):
    k_cpu = past_kv[i][0].to(torch.bfloat16)
    v_cpu = past_kv[i][1].to(torch.bfloat16)
    k_tt_i = ttnn.from_torch(k_cpu, device=mesh, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_tt_i = ttnn.from_torch(v_cpu, device=mesh, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.fill_cache(tt_cache[i][0], k_tt_i, 0)
    ttnn.fill_cache(tt_cache[i][1], v_tt_i, 0)
    ttnn.deallocate(k_tt_i); ttnn.deallocate(v_tt_i)

decode_masks = precompute_attn_masks(start_pos=seq, num_steps=MAX_NEW,
                                     max_cache_seq=max_cache, device=mesh)

# ========== 5. Step-by-step HF vs TT comparison ==========
def compare_logits(step_idx, hf_logits, tt_logits, tokenizer, topk=5):
    hf_f = hf_logits.float().cpu()
    tt_f = tt_logits.float().cpu()
    cos = F.cosine_similarity(hf_f.unsqueeze(0), tt_f.unsqueeze(0)).item()
    max_abs = (hf_f - tt_f).abs().max().item()
    hf_arg = int(hf_f.argmax().item()); tt_arg = int(tt_f.argmax().item())
    hf_top = torch.topk(hf_f, topk); tt_top = torch.topk(tt_f, topk)
    print(f"\n--- Step {step_idx} comparison ---")
    print(f"  cosine_sim:  {cos:.6f}")
    print(f"  max_abs_diff:{max_abs:.4f}")
    print(f"  argmax HF={hf_arg} ({tokenizer.decode([hf_arg])!r})  "
          f"TT={tt_arg} ({tokenizer.decode([tt_arg])!r})  match={hf_arg==tt_arg}")
    print(f"  HF top-{topk}:")
    for v, i in zip(hf_top.values.tolist(), hf_top.indices.tolist()):
        print(f"    {i:>7}  {v:>10.3f}  {tokenizer.decode([i])!r}")
    print(f"  TT top-{topk}:")
    for v, i in zip(tt_top.values.tolist(), tt_top.indices.tolist()):
        print(f"    {i:>7}  {v:>10.3f}  {tokenizer.decode([i])!r}")
    return cos, max_abs, hf_arg, tt_arg

def tt_step(tok_id, cur_pos, mask_idx):
    token_embed = emb_w[tok_id].reshape(1, 1, 1, DIM)
    tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_logits = decode_step_tt(tt_tok, dec_layers, dec_final, mesh,
                                cos_full_tt, sin_full_tt, tt_cache, cur_pos,
                                attn_mask_tt=decode_masks[mask_idx])
    return ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()

def hf_step(tok_id, past):
    with torch.no_grad():
        out = hf(input_ids=torch.tensor([[tok_id]]), past_key_values=past, use_cache=True)
    return out.logits[0, -1, :vocab_size].float(), out.past_key_values

# We want to compare at steps 0, 5, 10.
# "Step k" = logits produced when feeding the k-th decode input token.
# Step 0 input = first_tok (the prefill's predicted token)
COMPARE_STEPS = {0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 178}
TRACK_ALL_ARGMAX = True  # also record argmax at EVERY step to find first divergence
MAX_STEP = max(COMPARE_STEPS)

cur_tok = first_tok
cur_pos = seq
results = {}
diverged_at = None
all_argmax = []
for step in range(MAX_STEP + 1):
    hf_logits, past_kv = hf_step(cur_tok, past_kv)
    tt_logits = tt_step(cur_tok, cur_pos, step)
    if step in COMPARE_STEPS:
        results[step] = compare_logits(step, hf_logits, tt_logits, tokenizer)
    hf_a = int(hf_logits.argmax().item()); tt_a = int(tt_logits.argmax().item())
    if TRACK_ALL_ARGMAX:
        all_argmax.append((step, hf_a, tt_a, hf_a == tt_a))
        if hf_a != tt_a:
            if diverged_at is None:
                diverged_at = step
            print(f"[DIVERGENCE] step {step}: HF={hf_a} ({tokenizer.decode([hf_a])!r}) vs TT={tt_a} ({tokenizer.decode([tt_a])!r})", flush=True)
    cur_tok = hf_a
    cur_pos += 1
print(f"\n[ARGMAX SUMMARY] divergences: {sum(1 for _,_,_,m in all_argmax if not m)}/{len(all_argmax)}, first_diverge={diverged_at}", flush=True)

print(f"\n{'='*60}")
print("Summary (cosine / max_abs / hf_arg / tt_arg):")
for step in sorted(results):
    cos, ma, ha, ta = results[step]
    match = "OK " if ha == ta else "MISMATCH"
    print(f"  step {step:>2}: cos={cos:.4f}  max|d|={ma:7.3f}  "
          f"hf={ha:>6}  tt={ta:>6}  {match}")
print('='*60)

for mask in decode_masks:
    ttnn.deallocate(mask)
ttnn.close_mesh_device(mesh)
