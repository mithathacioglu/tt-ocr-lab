#!/usr/bin/env python3
"""E2E: Qwen2.5-VL-3B TT vision + CPU decoder → OCR text."""
import os, sys, time, types, torch, torch.nn.functional as F
os.environ['TT_METAL_LOGGER_LEVEL'] = 'FATAL'
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TT = os.path.join(P, "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
sys.path.insert(0, TT); sys.path.insert(0, P)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE = os.path.join(P, "ocr_lab/tmp_1pdf_page-1.png")
PROMPT = "Extract the text content from this image."

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs

torch.set_num_threads(32)
proc = AutoProcessor.from_pretrained(MODEL)

# === TT Vision ===
print("=== TT Vision ===", flush=True)
os.environ["HF_MODEL"] = MODEL
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = VisionModelArgs(mesh, instruct=True, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
ref = args.reference_vision_model()
tt_vis = DropInVisionTransformer(ref, args)

msgs = [{"role": "user", "content": [
    {"type": "image", "image": IMAGE},
    {"type": "text", "text": PROMPT},
]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ii, vi = process_vision_info(msgs)
inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")

# Cold + warm
with torch.no_grad():
    _ = tt_vis(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"])
    t0 = time.perf_counter()
    vision_out = tt_vis(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"])
    vis_ms = (time.perf_counter() - t0) * 1000
print(f"  TT vision warm: {vis_ms:.0f}ms, shape={vision_out.shape}", flush=True)

ttnn.close_mesh_device(mesh)

# === CPU Decoder ===
print("\n=== CPU Decoder ===", flush=True)
hf = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation='sdpa').eval()

# Inject TT vision embeddings into decoder
emb_w = hf.get_input_embeddings().weight.detach()
img_token_id = hf.config.image_token_id
img_mask = inputs["input_ids"] == img_token_id
embeds = F.embedding(inputs["input_ids"], emb_w)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_out.to(embeds.dtype))

t0 = time.perf_counter()
with torch.no_grad():
    gen = hf.generate(
        input_ids=inputs["input_ids"],
        inputs_embeds=embeds,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
    )
dec_ms = (time.perf_counter() - t0) * 1000

trimmed = gen[0][inputs["input_ids"].shape[1]:]
out = proc.batch_decode([trimmed], skip_special_tokens=True)[0]

total_ms = vis_ms + dec_ms
print(f"  Decoder: {dec_ms:.0f}ms ({dec_ms/1000:.1f}s)")
print(f"\n{'='*60}")
print(f"  TT Vision:  {vis_ms:.0f}ms ({vis_ms/1000:.1f}s)")
print(f"  CPU Decoder: {dec_ms:.0f}ms ({dec_ms/1000:.1f}s)")
print(f"  Total:       {total_ms:.0f}ms ({total_ms/1000:.1f}s)")
print(f"\n  OCR Output:")
print(out[:800])
print(f"{'='*60}")
