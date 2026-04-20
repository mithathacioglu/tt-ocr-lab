#!/usr/bin/env python3
"""Debug: run JUST block 0 of DropIn and compare with CPU block 0."""
import os, sys, time, torch, torch.nn.functional as F
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TT_METAL_LOGGER_LEVEL']='FATAL'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TT_METAL = os.path.join(PROJECT_ROOT, "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, TT_METAL); sys.path.insert(0, PROJECT_ROOT); sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr_lab/shims"))

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
sd = patch_dots_mocr_config_for_tt_metal()

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message
from dots_tt_port.vision_tower_smoke import load_vision_model

proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
img_msg, _ = fixed_page_message(os.path.join(PROJECT_ROOT, "ocr_lab/tmp_1pdf_page-1.png"), width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "test"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"].to(torch.int32)

# CPU reference: run block 0 only
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)

with torch.no_grad():
    h_cpu = vm.patch_embed(pv, gt)
    rotary = vm.rot_pos_emb(gt)
    cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)

    # Save pre-block state
    pre_block = h_cpu.clone()

    # Run block 0
    h_after_0 = vm.blocks[0](h_cpu, cu_seqlens=cu, rotary_pos_emb=rotary)
    print(f"CPU block 0: shape={h_after_0.shape} mean={h_after_0.mean():.6f} std={h_after_0.std():.6f}")

# TT: create DropIn, run only block 0
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
ref = args.reference_vision_model()
tt_vision = DropInVisionTransformer(ref, args, debug=False)

# Get the same preprocessing the adapter does
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess

grid_thw = gt
unpadded_seq = (grid_thw[:,1] * grid_thw[:,2]).sum().item()
seq_len = ((unpadded_seq // 2048) + 1) * 2048

# Rotary (same as adapter's forward)
pos_ids = ref.get_pos_ids_by_grid(grid_thw.cpu())
pos_ids = torch.cat(pos_ids, dim=0)
max_grid = int(grid_thw.cpu()[:,1:].max().item())
rotary_full = ref.rotary_pos_emb(max_grid).cpu()
rotary_tt = rotary_full[pos_ids].flatten(1).float()
cos_orig = rotary_tt.cos().unsqueeze(1).repeat(1, 1, 2)
sin_orig = rotary_tt.sin().unsqueeze(1).repeat(1, 1, 2)
cos_meta, sin_meta = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
cos_padded = F.pad(cos_meta, (0, 0, 0, seq_len - unpadded_seq), value=1).unsqueeze(0).unsqueeze(0)
sin_padded = F.pad(sin_meta, (0, 0, 0, seq_len - unpadded_seq), value=0).unsqueeze(0).unsqueeze(0)

cos_tt = ttnn.from_torch(cos_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
sin_tt = ttnn.from_torch(sin_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))

cu_seqlens, cu_window_seqlens, _, window_index = qwen2_5_vision_transformer_preprocess(
    seq_len=unpadded_seq, grid_thw=grid_thw,
    head_dim=args.vision_head_dim,
    spatial_merge_size=args.hf_config.vision_config.spatial_merge_size,
    window_size=args.hf_config.vision_config.window_size,
    patch_size=args.hf_config.vision_config.patch_size,
)

# Prepare input (same as adapter)
tt_input = tt_vision.tt_model.prepare_input(pre_block, window_index, seq_len)

# Run ONLY block 0
cu_tt = ttnn.from_torch(cu_seqlens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
rot_mats = [cos_tt, sin_tt]

tt_block0_out = tt_vision.tt_model.blocks[0](tt_input, cu_seqlens=cu_tt, rot_mats=rot_mats)

# Convert to CPU
tt_out_cpu = ttnn.to_torch(tt_block0_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
tt_out_trimmed = tt_out_cpu[:, 0:1, :unpadded_seq, :1536].squeeze(0).squeeze(0).float()

# Reverse window index
reverse_idx = torch.argsort(window_index)
# window_index was applied to groups of 4 in prepare_input, so we need to reverse at patch level
# Actually window_index is identity for dots.mocr, so no reordering needed

cpu_out = h_after_0.float()

print(f"\nTT block 0: shape={tt_out_trimmed.shape} mean={tt_out_trimmed.mean():.6f} std={tt_out_trimmed.std():.6f}")
print(f"CPU block 0: shape={cpu_out.shape} mean={cpu_out.mean():.6f} std={cpu_out.std():.6f}")

cos_sim = F.cosine_similarity(cpu_out.flatten().unsqueeze(0), tt_out_trimmed.flatten().unsqueeze(0))
diff = (cpu_out - tt_out_trimmed).abs()
print(f"\nBlock 0 comparison:")
print(f"  Cosine sim: {cos_sim.item():.6f}")
print(f"  Max diff:   {diff.max():.4f}")
print(f"  Mean diff:  {diff.mean():.4f}")

ttnn.close_mesh_device(mesh)
