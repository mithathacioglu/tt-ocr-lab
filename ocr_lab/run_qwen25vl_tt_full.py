#!/usr/bin/env python3
"""Full TT pipeline: Qwen2.5-VL vision + decoder on TT hardware.
Adapts demo.py for OCR use case.
"""
import os, sys, time, json, torch
os.environ['TT_METAL_LOGGER_LEVEL'] = 'FATAL'
TT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, TT)

MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
DEVICE = os.environ.get("MESH_DEVICE", "N300")
IMAGE = os.environ.get("IMAGE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_1pdf_page-1.png"))
PROMPT = os.environ.get("PROMPT", "Extract the text content from this image.")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

os.environ["HF_MODEL"] = MODEL
os.environ["MESH_DEVICE"] = DEVICE

print(f"Model: {MODEL}")
print(f"Device: {DEVICE}")
print(f"Image: {IMAGE}")

import ttnn
from models.demos.qwen25_vl.demo.demo import create_tt_model
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.tt.common import merge_vision_tokens, multimodal_rope_from_hf, preprocess_inputs_prefill
from models.common.sampling import SamplingParams
from models.tt_transformers.tt.model_config import ModelArgs
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

torch.set_num_threads(32)
proc = AutoProcessor.from_pretrained(MODEL)

# Build prompt
msgs = [{"role": "user", "content": [
    {"type": "image", "image": IMAGE},
    {"type": "text", "text": PROMPT},
]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
print(f"pixel_values: {inputs['pixel_values'].shape}, input_ids: {inputs['input_ids'].shape}")

# Open TT device
mesh_shape = {"N150": (1,1), "N300": (1,2)}[DEVICE]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
mesh.enable_program_cache()

# === Vision (TT) ===
print("\n=== TT Vision ===", flush=True)
vis_args = VisionModelArgs(mesh, instruct=True, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
ref_vis = vis_args.reference_vision_model()
tt_vis = DropInVisionTransformer(ref_vis, vis_args)

t0 = time.perf_counter()
with torch.no_grad():
    vision_out = tt_vis(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"])
vis_cold = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
with torch.no_grad():
    vision_out = tt_vis(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"])
vis_warm = (time.perf_counter() - t0) * 1000
print(f"  Vision cold: {vis_cold:.0f}ms, warm: {vis_warm:.0f}ms, shape: {vision_out.shape}")

# Merge vision tokens into input
merged_ids, merged_embeds, position_ids, mrope_position_delta = merge_vision_tokens(
    proc, inputs["input_ids"], vision_out, inputs["image_grid_thw"])

# === Decoder (TT) ===
print("\n=== TT Decoder ===", flush=True)
tt_model_args, tt_model, paged_attention_config, tt_kv_cache = create_tt_model(
    mesh, instruct=True, max_batch_size=1, optimizations=None,
    max_seq_len=4096,
    page_params={"page_block_size": 32, "page_max_num_blocks": 1024},
    use_paged_kv_cache=True,
)

# Prefill
rot_mats = multimodal_rope_from_hf(proc, merged_ids, inputs["image_grid_thw"], tt_model_args)
prefill_input = preprocess_inputs_prefill(tt_model, tt_model_args, merged_embeds, rot_mats, paged_attention_config)

t0 = time.perf_counter()
tt_logits = tt_model.forward(*prefill_input)
prefill_ms = (time.perf_counter() - t0) * 1000
print(f"  Prefill: {prefill_ms:.0f}ms")

# Decode
sampling = SamplingParams(temperature=0, top_p=0.08)
generated = []
for step in range(MAX_TOKENS):
    # Sample
    token = tt_logits.argmax(-1).item() if not isinstance(tt_logits, ttnn.Tensor) else 0
    # TODO: proper decode loop with TT
    break

print(f"\n  Total: vision {vis_warm:.0f}ms + prefill {prefill_ms:.0f}ms")

ttnn.close_mesh_device(mesh)
