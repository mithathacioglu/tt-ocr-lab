#!/usr/bin/env python3
"""
Run dots.mocr vision tower via TT-Symbiote with DPL_NO_ERROR_PROP mode.
This runs both TTNN and PyTorch separately, compares outputs with PCC,
and feeds PyTorch tensors to TTNN to avoid error propagation.
"""
import os, sys, time, torch
os.environ['TT_METAL_LOGGER_LEVEL'] = 'FATAL'
os.environ['TT_SYMBIOTE_RUN_MODE'] = os.environ.get('SYMBIOTE_MODE', 'DPL_NO_ERROR_PROP')
os.environ['TT_SYMBIOTE_DISPATCHER'] = 'DEFAULT'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TT = os.path.join(P, "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, TT)
sys.path.insert(0, P)
sys.path.insert(0, P + "/ocr_lab/shims")

torch.set_num_threads(32)

from torch import nn
from dots_tt_port.vision_tower_smoke import load_vision_model, resolve_snapshot_dir
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message

from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm, TTNNRMSNorm
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.run_config import DispatchManager

# Load dots.mocr
sd = resolve_snapshot_dir("rednote-hilab/dots.mocr")
vm, _ = load_vision_model(sd, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
print(f"Loaded dots.mocr: {len(vm.blocks)} blocks", flush=True)

# Build input
proc = AutoProcessor.from_pretrained(str(sd), trust_remote_code=True)
img_msg, _ = fixed_page_message(P + "/ocr_lab/tmp_1pdf_page-1.png", width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": "t"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
pv = inputs["pixel_values"].to(torch.bfloat16)
gt = inputs["image_grid_thw"].to(torch.int32)

# Set TT device
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
set_device(vm, mesh)

# Find dots.mocr's RMSNorm class
RMSNormClass = type(vm.blocks[0].norm1)
print(f"RMSNorm class: {RMSNormClass}", flush=True)

# Replace PyTorch modules with TTNN equivalents
# Only Linear — RMSNorm on CPU (less transfer overhead)
nn_to_ttnn = {
    nn.Linear: TTNNLinear,
}

print("Replacing modules with TTNN...", flush=True)
register_module_replacement_dict(vm, nn_to_ttnn, model_config=None)

# Must preprocess + move weights to device for each TTNNModule
from models.experimental.tt_symbiote.core.module import TTNNModule
set_device(vm, mesh)
for name, mod in vm.named_modules():
    if isinstance(mod, TTNNModule):
        mod.preprocess_weights()
        mod.move_weights_to_device()
        print(f"  {name}: device={mod.device}", flush=True)
print("Done — weights on device", flush=True)

# Run with DPL_NO_ERROR_PROP — will print PCC for each layer
print(f"\nRunning in {os.environ['TT_SYMBIOTE_RUN_MODE']} mode...", flush=True)
with torch.no_grad():
    t0 = time.perf_counter()
    out = vm(pv, gt)
    elapsed = (time.perf_counter() - t0) * 1000

print(f"\nTime: {elapsed:.0f}ms")
print(f"Output shape: {out.shape}")
out_t = out.data if hasattr(out, 'data') else out
print(f"Output range: [{float(out_t.min()):.2f}, {float(out_t.max()):.2f}]")

ttnn.close_mesh_device(mesh)
