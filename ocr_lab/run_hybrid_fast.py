#!/usr/bin/env python3
"""
Fast hybrid: 5 TT vision + 37 CPU vision + fp32 prefill + TT decode.
Target: <20s per page at 672x952.
"""
import os, sys, time, json, torch, torch.nn.functional as F
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TT_METAL_LOGGER_LEVEL']='FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P+"/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, P); sys.path.insert(0, P+"/ocr_lab/shims")
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
    img_msg = {"type": "image", "image": P+'/ocr_lab/tmp_1pdf_page-1.png'}
else:
    img_msg, _ = fixed_page_message(P+'/ocr_lab/tmp_1pdf_page-1.png', width=W, height=H)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
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
    load_decoder_weights, decode_step_tt, decode_step_flash, precompute_attn_masks,
    init_tt_kv_cache, fill_tt_kv_cache,
    DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS)

MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH_SHAPE", "1,1").split(","))
print(f"Opening mesh {MESH_SHAPE}", flush=True)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE)); mesh.enable_program_cache()

# Patch ttnn.to_torch to auto-handle replicated multi-device tensors (extract shard 0)
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
vis_args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=int(os.environ.get("MAX_SEQ", 2048)))
ref = vis_args.reference_vision_model()
tt_vis = DropInVisionTransformer(ref, vis_args, debug=False)

unp = (gt[:,1]*gt[:,2]).sum().item(); sl = ((unp//2048)+1)*2048; dim = 1536
pos_ids = ref.get_pos_ids_by_grid(gt.cpu()); pos_ids = torch.cat(pos_ids, dim=0)
mg = int(gt.cpu()[:,1:].max().item()); rf = ref.rotary_pos_emb(mg).cpu()
rot = rf[pos_ids].flatten(1).float()
ch = rot.cos().unsqueeze(1).repeat(1,1,2); sh = rot.sin().unsqueeze(1).repeat(1,1,2)
cm, sm = convert_rope_style_hf_to_meta(ch, sh)
cp = F.pad(cm,(0,0,0,sl-unp),value=1).unsqueeze(0).unsqueeze(0)
sp = F.pad(sm,(0,0,0,sl-unp),value=0).unsqueeze(0).unsqueeze(0)
_rot_mapper = ttnn.ReplicateTensorToMesh(mesh) if MESH_SHAPE != (1, 1) else ttnn.ShardTensorToMesh(mesh, dim=0)
cos_tt = ttnn.from_torch(cp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=_rot_mapper)
sin_tt = ttnn.from_torch(sp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=_rot_mapper)
cs, cws, _, wi = qwen2_5_vision_transformer_preprocess(seq_len=unp, grid_thw=gt,
    head_dim=vis_args.vision_head_dim, spatial_merge_size=vis_args.hf_config.vision_config.spatial_merge_size,
    window_size=vis_args.hf_config.vision_config.window_size, patch_size=vis_args.hf_config.vision_config.patch_size)
cu_tt = ttnn.from_torch(cs, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

# Decoder TT setup
dec_args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
dec_sd = dec_args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(dec_sd, mesh)

cfg = json.loads(__import__('pathlib').Path(str(sd), 'config.json').read_text())
vocab_size = cfg.get('vocab_size', 151936)
rope_theta = cfg.get("rope_theta", 1e6)
MAX_NEW = 300  # Limit for decode
max_pos = seq + MAX_NEW + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)
# TT-side full cos/sin tables (pre-loaded once)
cos_full_tt = ttnn.from_torch(cos_full.unsqueeze(0).unsqueeze(0),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
sin_full_tt = ttnn.from_torch(sin_full.unsqueeze(0).unsqueeze(0),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)

logit_bias_tensor = torch.zeros(vocab_size)
for tok_id, bias in LOGIT_BIAS.items():
    if tok_id < vocab_size: logit_bias_tensor[tok_id] = bias

# CPU vision model (fp32)
vm, _ = load_vision_model(sd, 'sdpa', limit_layers=0); vm = vm.eval().to(torch.float32)
rotary_cpu = vm.rot_pos_emb(gt)
cu_cpu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)

# HF model for prefill
hf = AutoModelForCausalLM.from_pretrained(str(sd), trust_remote_code=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
hf.vision_tower = hf.vision_tower.to(torch.float32)
emb_w = hf.get_input_embeddings().weight.detach().to(torch.bfloat16)

# ========== Cold run ==========
N_TT = int(os.environ.get("N_TT", 5))
if N_TT > 0:
    pe = ref.patch_embed(pv); x = tt_vis.tt_model.prepare_input(pe, wi, sl)
    for i in range(N_TT): x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
    ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)); ttnn.deallocate(x)
print("Cold done", flush=True)

# ========== Warm run ==========
t_total = time.perf_counter()

# 1. TT vision (or CPU-only if N_TT=0)
t0 = time.perf_counter()
if N_TT > 0:
    pe = ref.patch_embed(pv); x = tt_vis.tt_model.prepare_input(pe, wi, sl)
    for i in range(N_TT): x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
    raw = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    h = raw[:,0:1,:unp,:dim].squeeze(0).squeeze(0).to(torch.float32)
    ttnn.deallocate(x)
else:
    h = vm.patch_embed(pv.to(torch.float32), gt).to(torch.float32)
tt_vis_ms = (time.perf_counter()-t0)*1000

# 2. CPU vision (remaining blocks, fp32)
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(N_TT, 42): h = vm.blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
    if hasattr(vm, 'post_trunk_norm'): h = vm.post_trunk_norm(h)
    vision_out = vm.merger(h)
cpu_vis_ms = (time.perf_counter()-t0)*1000

# 3. Decoder prefill (fp32)
emb_w_fp32 = hf.get_input_embeddings().weight.detach()
img_mask = inputs["input_ids"] == hf.config.image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w_fp32)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_out.to(embeds.dtype))

t0 = time.perf_counter()
with torch.no_grad():
    outputs = hf(input_ids=inputs["input_ids"], inputs_embeds=embeds, use_cache=True)
prefill_ms = (time.perf_counter()-t0)*1000
first_tok = outputs.logits[0, -1, :vocab_size].float().argmax(-1).item()
past_kv = outputs.past_key_values

# 4. TT Decode — fully on-device with TT KV cache
max_cache = seq + MAX_NEW + 32
max_cache = ((max_cache + 31) // 32) * 32
tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)

# Fill TT KV cache from HF prefill output
_kv_fp32 = os.environ.get("KV_FP32") == "1"
_kv_ttnn_dtype = ttnn.float32 if _kv_fp32 else ttnn.bfloat16
_kv_torch_dtype = torch.float32 if _kv_fp32 else torch.bfloat16
for i in range(N_LAYERS):
    k_cpu = past_kv[i][0].to(_kv_torch_dtype)
    v_cpu = past_kv[i][1].to(_kv_torch_dtype)
    k_tt_i = ttnn.from_torch(k_cpu, device=mesh, dtype=_kv_ttnn_dtype,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_tt_i = ttnn.from_torch(v_cpu, device=mesh, dtype=_kv_ttnn_dtype,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.fill_cache(tt_cache[i][0], k_tt_i, 0)
    ttnn.fill_cache(tt_cache[i][1], v_tt_i, 0)
    ttnn.deallocate(k_tt_i); ttnn.deallocate(v_tt_i)
_use_hf_decode = os.environ.get("HF_DECODE") == "1"
if not _use_hf_decode:
    del past_kv, hf

# Pre-build attention masks
decode_masks = precompute_attn_masks(start_pos=seq, num_steps=MAX_NEW-1, max_cache_seq=max_cache, device=mesh)

generated = [first_tok]; cur_pos = seq; next_token = first_tok
token_counts = {}
t0 = time.perf_counter()
for step in range(1, MAX_NEW):
    if _use_hf_decode:
        with torch.no_grad():
            out = hf(input_ids=torch.tensor([[next_token]]), past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits_d = out.logits[0, -1, :vocab_size].float()
    else:
        token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
        tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if os.environ.get("FLASH_DECODE") == "1":
            tt_logits = decode_step_flash(tt_tok, dec_layers, dec_final, mesh,
                                          cos_full_tt, sin_full_tt, tt_cache, cur_pos)
        else:
            tt_logits = decode_step_tt(tt_tok, dec_layers, dec_final, mesh,
                                        cos_full_tt, sin_full_tt, tt_cache, cur_pos,
                                        attn_mask_tt=decode_masks[step-1])
        logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
    logits_d += logit_bias_tensor
    if FREQUENCY_PENALTY > 0:
        for tid, cnt in token_counts.items():
            if tid < vocab_size: logits_d[tid] -= FREQUENCY_PENALTY * cnt
    if os.environ.get("GREEDY") == "1":
        next_token = int(logits_d.argmax().item())
    else:
        logits_d /= TEMPERATURE
        sorted_l, sorted_i = torch.sort(logits_d, descending=True)
        cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        rm = cum_p > TOP_P; rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = 0
        logits_d[sorted_i[rm]] = float('-inf')
        next_token = torch.multinomial(F.softmax(logits_d, dim=-1), 1).item()
    token_counts[next_token] = token_counts.get(next_token, 0) + 1
    generated.append(next_token); cur_pos += 1
    if next_token in EOS_TOKEN_IDS: break
decode_ms = (time.perf_counter()-t0)*1000
total_ms = (time.perf_counter()-t_total)*1000

gen_text = tokenizer.decode(generated, skip_special_tokens=True)
kws = ['hükümler', 'davalı', 'olunmuş', 'Dilekçesi', 'kesinleştiği']
found = [k for k in kws if k in gen_text]

print(f"\n{'='*60}")
print(f"  TT vision (5 blk): {tt_vis_ms:.0f}ms")
print(f"  CPU vision (37):   {cpu_vis_ms:.0f}ms ({cpu_vis_ms/1000:.1f}s)")
print(f"  Decoder prefill:   {prefill_ms:.0f}ms ({prefill_ms/1000:.1f}s)")
print(f"  TT decode:         {decode_ms:.0f}ms ({len(generated)} tok, {decode_ms/len(generated):.0f}ms/tok)")
print(f"  TOTAL:             {total_ms:.0f}ms ({total_ms/1000:.1f}s)")
print(f"  Keywords: {len(found)}/{len(kws)} {found}")
print(f"  Text (full): {repr(gen_text)}")
print(f"{'='*60}")

for mask in decode_masks:
    ttnn.deallocate(mask)
ttnn.close_mesh_device(mesh)
