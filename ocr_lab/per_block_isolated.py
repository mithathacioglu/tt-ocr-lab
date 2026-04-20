#!/usr/bin/env python3
"""
Per-block isolated test: for EACH block, feed CPU input → run on TT → compare.
This isolates each block's intrinsic TT error from accumulation drift.
"""
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

# CPU: run all blocks, save state BEFORE each block
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(torch.bfloat16)
cpu_states = []  # state[i] = hidden state BEFORE block i
cpu_outputs = []  # output[i] = hidden state AFTER block i
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary_cpu = vm.rot_pos_emb(gt)
    cu_cpu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)
    for i in range(42):
        cpu_states.append(h.clone())
        h = vm.blocks[i](h, cu_seqlens=cu_cpu, rotary_pos_emb=rotary_cpu)
        cpu_outputs.append(h.clone())
print("CPU states saved", flush=True)

# TT: DropIn setup
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
cs, cws, _, wi = qwen2_5_vision_transformer_preprocess(seq_len=unp, grid_thw=gt,
    head_dim=args.vision_head_dim, spatial_merge_size=args.hf_config.vision_config.spatial_merge_size,
    window_size=args.hf_config.vision_config.window_size, patch_size=args.hf_config.vision_config.patch_size)
cu_tt = ttnn.from_torch(cs, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

def send_cpu_to_tt(h_cpu):
    """CPU state → TT format (padded, prepared as residual)."""
    return tt_vis.tt_model.prepare_input(h_cpu, wi, sl)

# Cold: run 1 TT block to warm up program cache
pe = ref.patch_embed(pv); x = tt_vis.tt_model.prepare_input(pe, wi, sl)
x = tt_vis.tt_model.blocks[0](x, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
ttnn.deallocate(x)

# Per-block isolated test
print(f"\n{'Block':>6} {'TT cos':>10} {'Input range':>20} {'CPU out':>20} {'TT out':>20}")
print("-" * 85)
for block_idx in range(42):
    # Load CPU state BEFORE this block
    cpu_in = cpu_states[block_idx]
    cpu_out = cpu_outputs[block_idx]

    # Send CPU input to TT
    x_tt = send_cpu_to_tt(cpu_in)
    # Run ONE block on TT
    x_tt = tt_vis.tt_model.blocks[block_idx](x_tt, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])
    # Read output
    raw = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    tt_out = raw[:,0:1,:unp,:dim].squeeze(0).squeeze(0)
    ttnn.deallocate(x_tt)

    cos = F.cosine_similarity(cpu_out.float().flatten().unsqueeze(0), tt_out.float().flatten().unsqueeze(0)).item()
    in_rng = f"[{cpu_in.min():.0f},{cpu_in.max():.0f}]"
    cpu_rng = f"[{cpu_out.min():.0f},{cpu_out.max():.0f}]"
    tt_rng = f"[{tt_out.min():.0f},{tt_out.max():.0f}]"
    print(f"{block_idx+1:>6} {cos:>10.6f} {in_rng:>20} {cpu_rng:>20} {tt_rng:>20}", flush=True)

ttnn.close_mesh_device(mesh)
