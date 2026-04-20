#!/usr/bin/env python3
"""Quick: 3 warm runs of standalone ttnn vision + decoder to get stable timing."""
from __future__ import annotations
import os, sys, time, types
from pathlib import Path
import torch, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import ttnn
from dots_tt_port.vision_tower_smoke import (
    resolve_snapshot_dir, load_vision_model, load_vision_state,
    apply_rotary_pos_emb_cached_cpu, CpuPatchMerger,
)
from ocr_lab.fixed_page import fixed_page_message

DIM, N_HEADS, HEAD_DIM, EPS, N_BLOCKS = 1536, 12, 128, 1e-5, 42
snapshot_dir = resolve_snapshot_dir("rednote-hilab/dots.mocr")
IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

def make_w(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def load_blocks(sd, dev):
    blocks = []
    for i in range(N_BLOCKS):
        p = f"blocks.{i}."
        w = {}
        w["norm1"] = ttnn.from_torch(sd[f"{p}norm1.weight"].unsqueeze(0).view(1,1,DIM//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["norm2"] = ttnn.from_torch(sd[f"{p}norm2.weight"].unsqueeze(0).view(1,1,DIM//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["qkv_w"] = make_w(sd[f"{p}attn.qkv.weight"].T, dev)
        w["proj_w"] = make_w(sd[f"{p}attn.proj.weight"].T, dev)
        w["fc1_w"] = make_w(sd[f"{p}mlp.fc1.weight"].T, dev)
        w["fc2_w"] = make_w(sd[f"{p}mlp.fc2.weight"].T, dev)
        w["fc3_w"] = make_w(sd[f"{p}mlp.fc3.weight"].T, dev)
        blocks.append(w)
    return blocks

def vision_block(x, w, dev, seq, cos, sin):
    norm1 = ttnn.rms_norm(x, epsilon=EPS, weight=w["norm1"])
    qkv = ttnn.linear(norm1, w["qkv_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm1)
    qkv_cpu = ttnn.to_torch(qkv).float()
    ttnn.deallocate(qkv)
    q = qkv_cpu[..., :DIM].reshape(seq, N_HEADS, HEAD_DIM)
    k = qkv_cpu[..., DIM:2*DIM].reshape(seq, N_HEADS, HEAD_DIM)
    v = qkv_cpu[..., 2*DIM:].reshape(seq, N_HEADS, HEAD_DIM)
    q = apply_rotary_pos_emb_cached_cpu(q.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
    k = apply_rotary_pos_emb_cached_cpu(k.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
    attn = F.scaled_dot_product_attention(q.transpose(0,1), k.transpose(0,1), v.float().transpose(0,1), dropout_p=0.0)
    attn = attn.transpose(0,1).reshape(1,1,seq,DIM).to(torch.bfloat16)
    attn_tt = ttnn.from_torch(attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    proj = ttnn.linear(attn_tt, w["proj_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_tt)
    h = ttnn.add(x, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(proj); ttnn.deallocate(x)
    norm2 = ttnn.rms_norm(h, epsilon=EPS, weight=w["norm2"])
    fc1 = ttnn.linear(norm2, w["fc1_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fc3 = ttnn.linear(norm2, w["fc3_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(norm2)
    gated = ttnn.mul(fc1, fc3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(fc1); ttnn.deallocate(fc3)
    fc2 = ttnn.linear(gated, w["fc2_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(gated)
    out = ttnn.add(h, fc2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h); ttnn.deallocate(fc2)
    return out

def run_vision(blocks, post_norm_tt, hidden, seq, cos, sin, dev):
    tt_h = ttnn.from_torch(hidden.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for bw in blocks:
        tt_h = vision_block(tt_h, bw, dev, seq, cos, sin)
    tt_h = ttnn.rms_norm(tt_h, epsilon=EPS, weight=post_norm_tt)
    return ttnn.to_torch(tt_h).squeeze(0).squeeze(0)[:seq].float()

# --- Main ---
shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
sys.path.insert(0, str(shims_dir))
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

# Decoder
print("Loading decoder...", flush=True)
decoder = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
orig_fwd = decoder.forward
def pfwd(self, input_ids=None, inputs_embeds=None, **kw):
    if input_ids is None and inputs_embeds is not None:
        input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    return orig_fwd(input_ids=input_ids, inputs_embeds=inputs_embeds, **kw)
decoder.forward = types.MethodType(pfwd, decoder)
emb_w = decoder.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

# Vision model for patch_embed + rotary
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"]
with torch.no_grad():
    hidden = vm.patch_embed(pv, gt.to(torch.int32))
seq = hidden.shape[0]
pos_ids = vm.get_pos_ids_by_grid(gt.cpu())
pos_ids = torch.cat(pos_ids, dim=0)
rotary = vm.rotary_pos_emb(int(gt.cpu()[:, 1:].max().item())).cpu()[pos_ids].flatten(1).float()
cos_cpu = rotary.cos().unsqueeze(1).repeat(1,1,2).unsqueeze(0)
sin_cpu = rotary.sin().unsqueeze(1).repeat(1,1,2).unsqueeze(0)
post_norm_w = vm.post_trunk_norm.weight.detach().cpu()
cpu_merger = CpuPatchMerger(vm.merger)
del vm

# TT device
print("Opening TT...", flush=True)
dev = ttnn.open_device(device_id=0)
dev.enable_program_cache()
sd = load_vision_state(snapshot_dir, limit_layers=0)
blocks = load_blocks(sd, dev)
post_norm_tt = ttnn.from_torch(post_norm_w.unsqueeze(0).view(1,1,DIM//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# 3 warm runs
REF = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli"
for run in range(3):
    t0 = time.perf_counter()
    vo = run_vision(blocks, post_norm_tt, hidden, seq, cos_cpu, sin_cpu, dev)
    vis_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    ve = cpu_merger(vo.to(torch.bfloat16))
    merger_ms = (time.perf_counter() - t0) * 1000

    # Decoder
    input_ids = inputs["input_ids"]
    img_mask = input_ids == decoder.config.image_token_id
    ie = F.embedding(input_ids, emb_w)
    ie = ie.masked_scatter(img_mask.unsqueeze(-1).expand_as(ie), ve.to(ie.dtype))
    with torch.no_grad():
        t0 = time.perf_counter()
        gids = decoder.generate(input_ids=input_ids, attention_mask=inputs["attention_mask"], inputs_embeds=ie, max_new_tokens=64)
        dec_ms = (time.perf_counter() - t0) * 1000
    gt_text = tokenizer.decode(gids[0, input_ids.shape[1]:], skip_special_tokens=True)
    match = gt_text.strip() == REF.strip()
    print(f"Run {run+1}: vis={vis_ms:.0f}ms merger={merger_ms:.0f}ms dec={dec_ms:.0f}ms total={vis_ms+merger_ms+dec_ms:.0f}ms match={match}", flush=True)
    if run == 0:
        print(f"  Text: {repr(gt_text[:100])}", flush=True)

ttnn.close_device(dev)
