#!/usr/bin/env python3
"""Measure cosine decay per block: run DropIn N blocks, compare with CPU N blocks."""
import os, sys, torch, torch.nn.functional as F
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

# CPU: run all 42 blocks, save intermediate states
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
cpu_states = {}
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary = vm.rot_pos_emb(gt)
    cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)
    cpu_states[0] = h.clone()
    for i in range(42):
        h = vm.blocks[i](h, cu_seqlens=cu, rotary_pos_emb=rotary)
        cpu_states[i+1] = h.clone()
    print(f"CPU: 42 blocks done")

# TT: run DropIn, but hook every block to capture output
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
ref = args.reference_vision_model()
tt_vision = DropInVisionTransformer(ref, args, debug=False)

# Run full DropIn forward once (cold + warm)
with torch.no_grad():
    _ = tt_vision(pv, gt)  # cold
    tt_out = tt_vision(pv, gt)  # warm

# Now we need to run block-by-block on TT and capture intermediates
# Simplest: monkeypatch the VisionTransformer.forward to save intermediates
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess

unpadded_seq = (gt[:,1] * gt[:,2]).sum().item()
seq_len = ((unpadded_seq // 2048) + 1) * 2048

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
    seq_len=unpadded_seq, grid_thw=gt,
    head_dim=args.vision_head_dim,
    spatial_merge_size=args.hf_config.vision_config.spatial_merge_size,
    window_size=args.hf_config.vision_config.window_size,
    patch_size=args.hf_config.vision_config.patch_size,
)
cu_tt = ttnn.from_torch(cu_seqlens_cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

# Prepare input
patch_embed = ref.patch_embed(pv)
tt_input = tt_vision.tt_model.prepare_input(patch_embed, window_index, seq_len)

# Run blocks one by one, measure cosine at checkpoints
out_dim = args.hf_config.vision_config.out_hidden_size
x_tt = tt_input
print(f"\n{'Block':>6} {'Cosine':>10} {'MaxDiff':>10} {'MeanDiff':>10}")
print("-" * 45)

checkpoints = [0, 4, 9, 14, 19, 24, 29, 34, 39, 41]
for i in range(42):
    x_tt = tt_vision.tt_model.blocks[i](x_tt, cu_seqlens=cu_tt, rot_mats=[cos_tt, sin_tt])

    if i in checkpoints:
        try:
            # Clone to avoid deallocation issues
            x_copy = ttnn.to_memory_config(x_tt, ttnn.DRAM_MEMORY_CONFIG)
            tt_cpu = ttnn.to_torch(x_copy, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
            ttnn.deallocate(x_copy)
            tt_trimmed = tt_cpu[:, 0:1, :unpadded_seq, :out_dim].squeeze(0).squeeze(0).float()
            cpu_ref = cpu_states[i+1].float()
            cos_sim = F.cosine_similarity(cpu_ref.flatten().unsqueeze(0), tt_trimmed.flatten().unsqueeze(0))
            diff = (cpu_ref - tt_trimmed).abs()
            print(f"{i+1:>6} {cos_sim.item():>10.6f} {diff.max().item():>10.4f} {diff.mean().item():>10.6f}", flush=True)
        except Exception as e:
            print(f"{i+1:>6} ERROR: {e}", flush=True)

ttnn.close_mesh_device(mesh)
