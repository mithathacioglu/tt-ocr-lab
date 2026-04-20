#!/usr/bin/env python3
"""Measure exact per-block cosine decay: TT vs CPU, same input, sequential."""
import os, sys, time, torch, torch.nn.functional as F
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

# CPU all blocks
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
cpu_states = []
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary = vm.rot_pos_emb(gt)
    cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2],gt[:,0]).cumsum(0,dtype=torch.int32),(1,0),value=0)
    cpu_states.append(h.float().clone())
    for i in range(42):
        h = vm.blocks[i](h, cu_seqlens=cu, rotary_pos_emb=rotary)
        cpu_states.append(h.float().clone())

# TT: manual block-by-block loop
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1)); mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
ref = args.reference_vision_model()
tt_vis = DropInVisionTransformer(ref, args, debug=False)

unp = (gt[:,1]*gt[:,2]).sum().item(); sl = ((unp//2048)+1)*2048; dim = 1536
pos_ids = ref.get_pos_ids_by_grid(gt.cpu()); pos_ids = torch.cat(pos_ids, dim=0)
mg = int(gt.cpu()[:,1:].max().item()); rf = ref.rotary_pos_emb(mg).cpu()
rot = rf[pos_ids].flatten(1).float()
ch = rot.cos().unsqueeze(1).repeat(1,1,2); sh = rot.sin().unsqueeze(1).repeat(1,1,2)
cm, sm = convert_rope_style_hf_to_meta(ch, sh)
cp = F.pad(cm,(0,0,0,sl-unp),value=1).unsqueeze(0).unsqueeze(0)
sp = F.pad(sm,(0,0,0,sl-unp),value=0).unsqueeze(0).unsqueeze(0)
cos_tt = ttnn.from_torch(cp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0))
sin_tt = ttnn.from_torch(sp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0))
cs, cws, _, wi = qwen2_5_vision_transformer_preprocess(seq_len=unp, grid_thw=gt, head_dim=args.vision_head_dim,
    spatial_merge_size=args.hf_config.vision_config.spatial_merge_size,
    window_size=args.hf_config.vision_config.window_size, patch_size=args.hf_config.vision_config.patch_size)
cu_tt = ttnn.from_torch(cs, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

# Cold run
pe = ref.patch_embed(pv)
x = tt_vis.tt_model.prepare_input(pe, wi, sl)
for i in range(42):
    x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
ttnn.deallocate(x)
print("Cold done", flush=True)

# Warm run with measurements at EVERY block
pe = ref.patch_embed(pv)
x = tt_vis.tt_model.prepare_input(pe, wi, sl)

# Run all 42 blocks, then read final output
print("Running 42 blocks...", flush=True)
for i in range(42):
    x = tt_vis.tt_model.blocks[i](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])

raw = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
tt_42 = raw[:,0:1,:unp,:dim].squeeze(0).squeeze(0).float()
cpu_42 = cpu_states[42]
cos_42 = F.cosine_similarity(cpu_42.flatten().unsqueeze(0), tt_42.flatten().unsqueeze(0)).item()
print(f"Block 42: cosine={cos_42:.6f} cpu=[{cpu_42.min():.0f},{cpu_42.max():.0f}] tt=[{tt_42.min():.0f},{tt_42.max():.0f}]", flush=True)

# Now test: run with CPU correction every N blocks
for N in [5, 10, 21]:
    pe2 = ref.patch_embed(pv)
    x2 = tt_vis.tt_model.prepare_input(pe2, wi, sl)
    h_cpu = cpu_states[0].to(torch.bfloat16)  # CPU shadow starts from same patch_embed

    for i in range(42):
        # Always maintain CPU shadow
        with torch.no_grad():
            h_cpu = vm.blocks[i](h_cpu, cu_seqlens=cu, rotary_pos_emb=rotary)

        if (i+1) % N == 0 and (i+1) < 42:
            # INJECT CPU state into TT — replace TT state with correct CPU state
            ttnn.deallocate(x2)
            h_padded = F.pad(h_cpu.to(torch.bfloat16), (0,0,0,sl-unp)).unsqueeze(0)
            x2 = args.prepare_residual_tensor_prefill(h_padded, force_replicated=True)
        else:
            x2 = tt_vis.tt_model.blocks[i](x2, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])

    raw2 = ttnn.to_torch(x2, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    tt_corr = raw2[:,0:1,:unp,:dim].squeeze(0).squeeze(0).float()
    cos_corr = F.cosine_similarity(cpu_42.flatten().unsqueeze(0), tt_corr.flatten().unsqueeze(0)).item()
    print(f"Corrected N={N}: cosine={cos_corr:.6f} tt=[{tt_corr.min():.0f},{tt_corr.max():.0f}]", flush=True)
    ttnn.deallocate(x2)

ttnn.close_mesh_device(mesh)
