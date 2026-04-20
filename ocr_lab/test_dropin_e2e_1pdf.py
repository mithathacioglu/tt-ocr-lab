#!/usr/bin/env python3
"""
End-to-end: tt-metal DropInVisionTransformer + CPU decoder on 1.pdf.
Verify text accuracy against known reference.
"""
from __future__ import annotations
import os, sys, time, types
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL_ROOT = PROJECT_ROOT / "tt-xla" / "third_party" / "tt-mlir" / "src" / "tt-mlir" / "third_party" / "tt-metal" / "src" / "tt-metal"
sys.path.insert(0, str(TT_METAL_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()

import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 64
REF_TEXT = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli"

# Build inputs
shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
sys.path.insert(0, str(shims_dir))
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)

img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

pixel_values = inputs["pixel_values"].to(torch.bfloat16)
grid_thw = inputs["image_grid_thw"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Load decoder (CPU)
print("Loading decoder...", flush=True)
decoder = AutoModelForCausalLM.from_pretrained(
    str(snapshot_dir), trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
).eval()

original_forward = decoder.forward
def patched_forward(self, input_ids=None, inputs_embeds=None, **kwargs):
    if input_ids is None and inputs_embeds is not None:
        input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    return original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
decoder.forward = types.MethodType(patched_forward, decoder)

emb_weight = decoder.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

# Open TT device
print("Opening TT device...", flush=True)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh_device.enable_program_cache()

# Create tt-metal native vision
print("Creating VisionModelArgs...", flush=True)
args = VisionModelArgs(mesh_device, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
ref_model = args.reference_vision_model()
print(f"  blocks={len(ref_model.blocks)}", flush=True)

print("Creating DropInVisionTransformer...", flush=True)
tt_vision = DropInVisionTransformer(ref_model, args, debug=False)
print("  OK!", flush=True)

# === Vision forward ===
print("Vision cold...", flush=True)
with torch.no_grad():
    t0 = time.perf_counter()
    vision_out = tt_vision(pixel_values, grid_thw)
    cold_ms = (time.perf_counter() - t0) * 1000
print(f"  Cold: {cold_ms:.0f}ms, shape={vision_out.shape}", flush=True)

print("Vision warm...", flush=True)
with torch.no_grad():
    t0 = time.perf_counter()
    vision_out2 = tt_vision(pixel_values, grid_thw)
    warm_ms = (time.perf_counter() - t0) * 1000
print(f"  Warm: {warm_ms:.0f}ms, shape={vision_out2.shape}", flush=True)

# Close TT device (decoder runs on CPU)
ttnn.close_mesh_device(mesh_device)

# === Inject vision embeddings into decoder ===
print("Generating text...", flush=True)
vision_embeddings = vision_out2.to(dtype=torch.bfloat16)

img_mask = input_ids == decoder.config.image_token_id
inputs_embeds = F.embedding(input_ids, emb_weight)
inputs_embeds = inputs_embeds.masked_scatter(
    img_mask.unsqueeze(-1).expand_as(inputs_embeds),
    vision_embeddings.to(inputs_embeds.dtype),
)

with torch.no_grad():
    t0 = time.perf_counter()
    gen_ids = decoder.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    gen_ms = (time.perf_counter() - t0) * 1000

gen_trimmed = gen_ids[0, input_ids.shape[1]:]
gen_text = tokenizer.decode(gen_trimmed, skip_special_tokens=True)
print(f"  Generate: {gen_ms:.0f}ms", flush=True)

# === Results ===
match = gen_text.strip() == REF_TEXT.strip()
total_ms = warm_ms + gen_ms

print(f"\n{'='*60}")
print(f"  Vision (warm):  {warm_ms:.0f}ms")
print(f"  Decoder (CPU):  {gen_ms:.0f}ms")
print(f"  TOTAL:          {total_ms:.0f}ms")
print(f"  Text match:     {match}")
print(f"  Reference:      {repr(REF_TEXT[:80])}")
print(f"  Output:         {repr(gen_text[:80])}")
print(f"{'='*60}")
