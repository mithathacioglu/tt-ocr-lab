#!/usr/bin/env python3
"""Run DropIn with N blocks, compare final output. Quick decay measurement."""
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

# CPU reference: all 42 blocks pre-merger outputs
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
cpu_refs = {}
with torch.no_grad():
    h = vm.patch_embed(pv, gt)
    rotary = vm.rot_pos_emb(gt)
    cu = F.pad(torch.repeat_interleave(gt[:,1]*gt[:,2], gt[:,0]).cumsum(0, dtype=torch.int32), (1,0), value=0)
    for i in range(42):
        h = vm.blocks[i](h, cu_seqlens=cu, rotary_pos_emb=rotary)
    # Post-norm
    h_f32 = h.float()
    var = h_f32.pow(2).mean(-1, keepdim=True)
    h_normed = (h_f32 * torch.rsqrt(var + 1e-6) * vm.post_trunk_norm.weight.float()).to(torch.bfloat16)
    cpu_pre_merger = h_normed
    cpu_post_merger = vm.merger(h_normed.to(vm.merger.ln_q.weight.dtype))
print(f"CPU done: pre_merger={cpu_pre_merger.shape}, post_merger={cpu_post_merger.shape}")

# TT DropIn
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
ref = args.reference_vision_model()
tt_vision = DropInVisionTransformer(ref, args, debug=False)

# Full forward
with torch.no_grad():
    _ = tt_vision(pv, gt)  # cold
    tt_out = tt_vision(pv, gt)  # warm

tt_cpu = tt_out.float()
cpu_ref = cpu_post_merger.float()

cos_sim = F.cosine_similarity(cpu_ref.flatten().unsqueeze(0), tt_cpu.flatten().unsqueeze(0))
diff = (cpu_ref - tt_cpu).abs()
print(f"\n42 blocks + merger:")
print(f"  Cosine: {cos_sim.item():.6f}")
print(f"  Max diff: {diff.max():.4f}")
print(f"  TT  stats: min={tt_cpu.min():.2f} max={tt_cpu.max():.2f} mean={tt_cpu.mean():.4f}")
print(f"  CPU stats: min={cpu_ref.min():.2f} max={cpu_ref.max():.2f} mean={cpu_ref.mean():.4f}")

# Now test: what if we feed the CORRECT (CPU) pre-merger output to TT merger?
# This isolates: is the drift in blocks or in merger?
# Actually, the DropIn includes merger inside VisionTransformer. Let's check the final shape.
print(f"\n  TT shape: {tt_cpu.shape}, CPU shape: {cpu_ref.shape}")

ttnn.close_mesh_device(mesh)
