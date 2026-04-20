#!/usr/bin/env python3
"""Find optimal TT/CPU cutoff for hybrid vision.

Run first N blocks on TT (DropIn), remaining on CPU.
Measure final accuracy at different cutoff points.
"""
import os, sys, time, torch, torch.nn.functional as F
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TT_METAL_LOGGER_LEVEL']='FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P+"/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, P); sys.path.insert(0, P+"/ocr_lab/shims")

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
sd = patch_dots_mocr_config_for_tt_metal()
from dots_tt_port.vision_tower_smoke import load_vision_model
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message

proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
img_msg, _ = fixed_page_message(P+"/ocr_lab/tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "test"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"].to(torch.int32)

# CPU reference: full 42 blocks + post_norm + merger
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    cpu_full = vm(pv, gt)
cpu_ref = cpu_full.float()
print(f"CPU ref: {cpu_ref.shape}")

# Also get intermediate states after each block
cpu_block_outputs = []
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary = vm.rot_pos_emb(gt)
    cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)
    cpu_block_outputs.append(h.clone())  # after patch_embed = index 0
    for i in range(42):
        h = vm.blocks[i](h, cu_seqlens=cu, rotary_pos_emb=rotary)
        cpu_block_outputs.append(h.clone())  # after block i = index i+1

# TT DropIn setup
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
ref = args.reference_vision_model()
tt_vision = DropInVisionTransformer(ref, args, debug=False)

# Precompute TT inputs
unpadded_seq = (gt[:,1] * gt[:,2]).sum().item()
seq_len = ((unpadded_seq // 2048) + 1) * 2048
out_dim = args.hf_config.vision_config.out_hidden_size

pos_ids = ref.get_pos_ids_by_grid(gt.cpu())
pos_ids = torch.cat(pos_ids, dim=0)
max_grid = int(gt.cpu()[:,1:].max().item())
rotary_full = ref.rotary_pos_emb(max_grid).cpu()
rotary_tt = rotary_full[pos_ids].flatten(1).float()
cos_orig = rotary_tt.cos().unsqueeze(1).repeat(1, 1, 2)
sin_orig = rotary_tt.sin().unsqueeze(1).repeat(1, 1, 2)
cos_meta, sin_meta = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
cos_padded = F.pad(cos_meta, (0, 0, 0, seq_len - unpadded_seq), value=1).unsqueeze(0).unsqueeze(0)
sin_padded = F.pad(sin_meta, (0, 0, 0, seq_len - unpadded_seq), value=0).unsqueeze(0).unsqueeze(0)
cos_tt = ttnn.from_torch(cos_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
sin_tt = ttnn.from_torch(sin_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
cu_seqlens_cpu, cu_window_seqlens, _, window_index = qwen2_5_vision_transformer_preprocess(
    seq_len=unpadded_seq, grid_thw=gt, head_dim=args.vision_head_dim,
    spatial_merge_size=args.hf_config.vision_config.spatial_merge_size,
    window_size=args.hf_config.vision_config.window_size,
    patch_size=args.hf_config.vision_config.patch_size)
cu_tt = ttnn.from_torch(cu_seqlens_cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

# Test different cutoff points
cutoffs = [5, 10, 15, 20, 25, 30, 35, 42]
print(f"\n{'TT blocks':>10} {'CPU blocks':>10} {'Cosine':>10} {'MaxDiff':>10} {'TT ms':>8}")
print("-" * 55)

for n_tt in cutoffs:
    # Run first n_tt blocks on TT
    patch_embed = ref.patch_embed(pv)
    tt_input = tt_vision.tt_model.prepare_input(patch_embed, window_index, seq_len)

    t0 = time.perf_counter()
    x_tt = tt_input
    for i in range(n_tt):
        x_tt = tt_vision.tt_model.blocks[i](x_tt, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])

    # Read TT output back to CPU
    tt_cpu_raw = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    tt_ms = (time.perf_counter() - t0) * 1000
    tt_hidden = tt_cpu_raw[:, 0:1, :unpadded_seq, :out_dim].squeeze(0).squeeze(0).to(torch.bfloat16)
    ttnn.deallocate(x_tt)

    # Run remaining blocks on CPU
    h = tt_hidden
    with torch.no_grad():
        for i in range(n_tt, 42):
            h = vm.blocks[i](h, cu_seqlens=cu, rotary_pos_emb=rotary)
        # Post-norm + merger on CPU
        if hasattr(vm, 'post_trunk_norm'):
            h = vm.post_trunk_norm(h)
        hybrid_out = vm.merger(h.to(vm.merger.ln_q.weight.dtype))

    # Compare with CPU reference
    cos_sim = F.cosine_similarity(cpu_ref.flatten().unsqueeze(0), hybrid_out.float().flatten().unsqueeze(0))
    diff = (cpu_ref - hybrid_out.float()).abs()
    print(f"{n_tt:>10} {42-n_tt:>10} {cos_sim.item():>10.6f} {diff.max().item():>10.4f} {tt_ms:>8.0f}")

ttnn.close_mesh_device(mesh)
