#!/usr/bin/env python3
"""Test MLP layer only: CPU block 18 MLP vs TT MLP with various dtypes."""
import os, sys, torch, torch.nn.functional as F
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TT_METAL_LOGGER_LEVEL']='FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P+"/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, P); sys.path.insert(0, P+"/ocr_lab/shims")

from dots_tt_port.vision_tower_smoke import load_vision_model, resolve_snapshot_dir
from transformers import AutoProcessor; from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message

sd = resolve_snapshot_dir("rednote-hilab/dots.mocr")
proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
img_msg, _ = fixed_page_message(P+"/ocr_lab/tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role":"user","content":[img_msg,{"type":"text","text":"t"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16); gt = inputs["image_grid_thw"].to(torch.int32)

# Get state before block 18 MLP (after attention residual)
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0); vm = vm.eval().to(torch.bfloat16)
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary = vm.rot_pos_emb(gt)
    cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)
    for i in range(17):
        h = vm.blocks[i](h, cu_seqlens=cu, rotary_pos_emb=rotary)
    # Block 18 attention part
    blk = vm.blocks[17]
    n1 = blk.norm1(h)
    attn_out = blk.attn(n1, cu_seqlens=cu, rotary_pos_emb=rotary)
    h_after_attn = h + attn_out
    # Input to MLP
    mlp_in_pre_norm = h_after_attn
    n2 = blk.norm2(h_after_attn)
    # CPU MLP
    w1_out = blk.mlp.fc1(n2)
    w3_out = blk.mlp.fc3(n2)
    silu_gate = F.silu(w1_out) * w3_out
    cpu_mlp_out = blk.mlp.fc2(silu_gate)
    print(f"CPU MLP input (n2): [{n2.min():.2f}, {n2.max():.2f}]")
    print(f"CPU w1 out: [{w1_out.min():.2f}, {w1_out.max():.2f}]")
    print(f"CPU w3 out: [{w3_out.min():.2f}, {w3_out.max():.2f}]")
    print(f"CPU silu*w3: [{silu_gate.min():.2f}, {silu_gate.max():.2f}]")
    print(f"CPU w2 out (MLP final): [{cpu_mlp_out.min():.2f}, {cpu_mlp_out.max():.2f}]")

# Get MLP weights
w1_w = blk.mlp.fc1.weight.data  # [4224, 1536]
w1_b = blk.mlp.fc1.bias.data if blk.mlp.fc1.bias is not None else None
w2_w = blk.mlp.fc2.weight.data
w2_b = blk.mlp.fc2.bias.data if blk.mlp.fc2.bias is not None else None
w3_w = blk.mlp.fc3.weight.data
w3_b = blk.mlp.fc3.bias.data if blk.mlp.fc3.bias is not None else None

# TT MLP test
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()

def test_dtype(name, wdt):
    # Load weights in specified dtype
    x_tt = ttnn.from_torch(n2.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w1_tt = ttnn.from_torch(w1_w.T.contiguous().unsqueeze(0).unsqueeze(0), dtype=wdt,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w3_tt = ttnn.from_torch(w3_w.T.contiguous().unsqueeze(0).unsqueeze(0), dtype=wdt,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w2_tt = ttnn.from_torch(w2_w.T.contiguous().unsqueeze(0).unsqueeze(0), dtype=wdt,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    w1_out_tt = ttnn.linear(x_tt, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    w3_out_tt = ttnn.linear(x_tt, w3_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    # Read intermediates
    w1_cpu = ttnn.to_torch(w1_out_tt).squeeze()
    w3_cpu = ttnn.to_torch(w3_out_tt).squeeze()
    # SiLU mul
    gated = ttnn.mul(w1_out_tt, w3_out_tt, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                     dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gated_cpu = ttnn.to_torch(gated).squeeze()
    # w2
    w2_out_tt = ttnn.linear(gated, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    tt_out = ttnn.to_torch(w2_out_tt).squeeze()

    cos = F.cosine_similarity(cpu_mlp_out.float().flatten().unsqueeze(0), tt_out.float().flatten().unsqueeze(0)).item()
    print(f"\n[{name}]")
    print(f"  TT w1: [{w1_cpu.min():.2f}, {w1_cpu.max():.2f}]")
    print(f"  TT w3: [{w3_cpu.min():.2f}, {w3_cpu.max():.2f}]")
    print(f"  TT gated: [{gated_cpu.min():.2f}, {gated_cpu.max():.2f}]")
    print(f"  TT w2 out: [{tt_out.min():.2f}, {tt_out.max():.2f}]")
    print(f"  Cosine vs CPU: {cos:.6f}")
    ttnn.deallocate(x_tt); ttnn.deallocate(w1_tt); ttnn.deallocate(w3_tt); ttnn.deallocate(w2_tt)
    ttnn.deallocate(w1_out_tt); ttnn.deallocate(w3_out_tt); ttnn.deallocate(gated); ttnn.deallocate(w2_out_tt)

test_dtype("bfloat8_b", ttnn.bfloat8_b)
test_dtype("bfloat16", ttnn.bfloat16)

ttnn.close_mesh_device(mesh)
