#!/usr/bin/env python3
"""Deep debug of block 18: attention vs MLP sub-components."""
import os, sys, torch, torch.nn.functional as F
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TT_METAL_LOGGER_LEVEL']='FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P+"/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, P); sys.path.insert(0, P+"/ocr_lab/shims")

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
sd = patch_dots_mocr_config_for_tt_metal()
from dots_tt_port.vision_tower_smoke import load_vision_model
from transformers import AutoProcessor; from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message

proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
img_msg, _ = fixed_page_message(P+"/ocr_lab/tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role":"user","content":[img_msg,{"type":"text","text":"t"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16); gt = inputs["image_grid_thw"].to(torch.int32)

# CPU: run to block 17, save state
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0); vm = vm.eval().to(torch.bfloat16)
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary_cpu = vm.rot_pos_emb(gt)
    cu_cpu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)
    for i in range(17):  # 0..16
        h = vm.blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)

# State BEFORE block 17 (index 17 in pytorch, block 18 in 1-indexed)
block18_in = h.clone()
print(f"Block 18 input: {block18_in.shape} range [{block18_in.min():.2f}, {block18_in.max():.2f}]")

# Manually run block 18 components on CPU
blk = vm.blocks[17]  # block 18 (0-indexed 17)
with torch.no_grad():
    # attention path
    n1 = blk.norm1(block18_in)
    print(f"After norm1: [{n1.min():.2f}, {n1.max():.2f}]")
    attn_out = blk.attn(n1, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
    print(f"Attention output: [{attn_out.min():.2f}, {attn_out.max():.2f}]")
    h_after_attn = block18_in + attn_out
    print(f"After attn residual: [{h_after_attn.min():.2f}, {h_after_attn.max():.2f}]")
    # MLP path
    n2 = blk.norm2(h_after_attn)
    print(f"After norm2: [{n2.min():.2f}, {n2.max():.2f}]")
    mlp_out = blk.mlp(n2)
    print(f"MLP output: [{mlp_out.min():.2f}, {mlp_out.max():.2f}]")
    h_final = h_after_attn + mlp_out
    print(f"Block 18 CPU output: [{h_final.min():.2f}, {h_final.max():.2f}]")

# Check: which component explodes?
print(f"\nAttention output max abs: {attn_out.abs().max():.2f}")
print(f"MLP output max abs: {mlp_out.abs().max():.2f}")

# Check attention internals
with torch.no_grad():
    seq_len = n1.shape[0]
    q, k, v = blk.attn.qkv(n1).reshape(seq_len, 3, blk.attn.num_heads, -1).permute(1,0,2,3).unbind(0)
    print(f"\nQKV output: Q=[{q.min():.2f},{q.max():.2f}] K=[{k.min():.2f},{k.max():.2f}] V=[{v.min():.2f},{v.max():.2f}]")
    from transformers_modules.rednote_hyphen_hilab.dots_dot_mocr import modeling_dots_vision as mdv
    q_rot = mdv.apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_cpu).squeeze(0)
    k_rot = mdv.apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_cpu).squeeze(0)
    print(f"After rotary: Q=[{q_rot.min():.2f},{q_rot.max():.2f}] K=[{k_rot.min():.2f},{k_rot.max():.2f}]")
    # SDPA
    attention_mask = torch.zeros([1, seq_len, seq_len], device=q.device, dtype=torch.bool)
    attention_mask[..., :seq_len, :seq_len] = True
    q_t = q_rot.transpose(0,1); k_t = k_rot.transpose(0,1); v_t = v.transpose(0,1)
    attn_out_sdpa = F.scaled_dot_product_attention(q_t, k_t, v_t, attention_mask)
    print(f"SDPA output: [{attn_out_sdpa.min():.2f}, {attn_out_sdpa.max():.2f}]")
    # After proj
    attn_proj_in = attn_out_sdpa.transpose(0,1).reshape(seq_len, -1)
    print(f"Pre-proj: [{attn_proj_in.min():.2f}, {attn_proj_in.max():.2f}]")
    proj_out = blk.attn.proj(attn_proj_in)
    print(f"After proj: [{proj_out.min():.2f}, {proj_out.max():.2f}]")
    print(f"  proj weight range: [{blk.attn.proj.weight.min():.2f}, {blk.attn.proj.weight.max():.2f}]")
