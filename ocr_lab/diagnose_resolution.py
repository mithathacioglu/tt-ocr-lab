#!/usr/bin/env python3
"""Test: does the 476x674 resolution cause the OCR errors, or is it the TT compute?

Runs pure CPU HF generate with the SAME fixed_page resized image that TT uses.
If CPU+476x674 is also broken → resolution is the issue
If CPU+476x674 is correct → TT matmuls/prefill is the issue
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from ocr_lab.fixed_page import fixed_page_message
from ocr_lab.dots_model import DEFAULT_MODEL_PATH, ensure_dots_ocr_processor_compat

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"

ensure_dots_ocr_processor_compat(DEFAULT_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL_PATH, trust_remote_code=True,
    torch_dtype=torch.bfloat16, attn_implementation='sdpa')
processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)

# Patch vision tower attention to use SDPA (no flash_attn on CPU)
vision_module = sys.modules.get(model.vision_tower.__class__.__module__)
if vision_module is not None and hasattr(vision_module, "VisionSdpaAttention"):
    for blk in model.vision_tower.blocks:
        old_attn = blk.attn
        new_attn = vision_module.VisionSdpaAttention(
            model.config.vision_config,
            old_attn.proj.in_features,
            num_heads=old_attn.num_heads,
            bias=old_attn.qkv.bias is not None,
        )
        new_attn.load_state_dict(old_attn.state_dict())
        new_attn = new_attn.to(device=old_attn.qkv.weight.device, dtype=old_attn.qkv.weight.dtype)
        blk.attn = new_attn

results = {}

for label, width, height in [("476x674", 476, 674), ("672x952", 672, 952)]:
    print(f"\n{'='*60}")
    print(f"CPU generate with fixed_page {label}")
    print(f"{'='*60}")

    img_msg, meta = fixed_page_message(IMAGE, width=width, height=height)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"  input_ids length: {inputs['input_ids'].shape[1]}")

    t0 = time.perf_counter()
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=256)
    elapsed = time.perf_counter() - t0
    trimmed = gen_ids[0][inputs['input_ids'].shape[1]:]
    out_text = processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Text: {repr(out_text[:300])}")
    results[label] = out_text

# Also run with original image (no fixed_page)
print(f"\n{'='*60}")
print(f"CPU generate with ORIGINAL image (no fixed_page)")
print(f"{'='*60}")
msgs_orig = [{"role": "user", "content": [
    {"type": "image", "image": str(IMAGE)},
    {"type": "text", "text": PROMPT}
]}]
text_orig = processor.apply_chat_template(msgs_orig, tokenize=False, add_generation_prompt=True)
img_in_orig, vid_in_orig = process_vision_info(msgs_orig)
inputs_orig = processor(text=[text_orig], images=img_in_orig, videos=vid_in_orig, padding=True, return_tensors="pt")
print(f"  pixel_values shape: {inputs_orig['pixel_values'].shape}")
print(f"  input_ids length: {inputs_orig['input_ids'].shape[1]}")

t0 = time.perf_counter()
with torch.no_grad():
    gen_ids_orig = model.generate(**inputs_orig, max_new_tokens=256)
elapsed = time.perf_counter() - t0
trimmed_orig = gen_ids_orig[0][inputs_orig['input_ids'].shape[1]:]
out_orig = processor.batch_decode([trimmed_orig], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"  Time: {elapsed:.1f}s")
print(f"  Text: {repr(out_orig[:300])}")
results["original"] = out_orig

# Compare
REF_KEYWORDS = ["hükümler", "davalı", "olunmuş"]
print(f"\n{'='*60}")
print("KEYWORD CHECK")
print(f"{'='*60}")
for label, text in results.items():
    found = [kw for kw in REF_KEYWORDS if kw in text]
    missing = [kw for kw in REF_KEYWORDS if kw not in text]
    print(f"\n  {label}:")
    print(f"    Found:   {found}")
    print(f"    Missing: {missing}")
    if missing:
        # Show what's there instead
        for kw in missing:
            idx = text.find(kw[:3])
            if idx >= 0:
                print(f"    Near '{kw[:3]}': ...{repr(text[max(0,idx-5):idx+20])}...")

out_path = PROJECT_ROOT / "ocr_lab" / "diagnose_resolution_results.json"
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nSaved: {out_path}")
