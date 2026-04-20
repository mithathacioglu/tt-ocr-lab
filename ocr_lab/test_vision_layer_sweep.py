#!/usr/bin/env python3
"""Test vision tower with reduced layers to find speed/accuracy tradeoff."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dots_tt_port.vision_tower_smoke import (
    TowerHybridCpuUnary, ensure_tt_env, load_vision_model, resolve_snapshot_dir,
)
from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import fixed_page_message

SNAPSHOT_DIR = resolve_snapshot_dir("rednote-hilab/dots.mocr")
IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

# Build inputs once
shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
if shims_dir.exists():
    sys.path.insert(0, str(shims_dir))
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained(str(SNAPSHOT_DIR), trust_remote_code=True)
image_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
messages = [{"role": "user", "content": [image_msg, {"type": "text", "text": "Please output the exact text in the image.\n\nReturn plain text only.\n"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

# Load decoder model (CPU) for full generate
print("Loading decoder model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    str(SNAPSHOT_DIR), trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
).eval()

# Setup TT
ensure_tt_env("0")
configure_tt_runtime(cache_dir=None, enable_trace=False, optimization_level=2)
import torch_xla.runtime as xr
xr.set_device_type("TT")
import torch_xla
tt_device = torch_xla.device()

pixel_values = inputs["pixel_values"]
image_grid_thw = inputs["image_grid_thw"]
input_ids = inputs["input_ids"]

LAYER_COUNTS = [42, 32, 24, 21]
results = []

for n_layers in LAYER_COUNTS:
    print(f"\n=== Testing {n_layers} layers ===", flush=True)

    vision_model, _ = load_vision_model(SNAPSHOT_DIR, "sdpa", limit_layers=n_layers)
    hybrid = TowerHybridCpuUnary(vision_model).eval().to(dtype=torch.bfloat16).to(tt_device)

    pv_tt = pixel_values.to(dtype=torch.bfloat16).to(tt_device)
    gt_tt = image_grid_thw.to(dtype=torch.int32).to(tt_device)

    # Cold run
    with torch.no_grad():
        t0 = time.perf_counter()
        emb_tt = hybrid(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        cold_ms = (time.perf_counter() - t0) * 1000
        emb = emb_tt.detach().cpu().to(dtype=torch.bfloat16)

    # Warm run
    with torch.no_grad():
        t0 = time.perf_counter()
        emb_tt2 = hybrid(pv_tt, gt_tt)
        torch_xla.sync(wait=True)
        warm_ms = (time.perf_counter() - t0) * 1000
        emb2 = emb_tt2.detach().cpu().to(dtype=torch.bfloat16)

    # Generate with this embedding
    img_mask = input_ids == model.config.image_token_id
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.masked_scatter(
        img_mask.unsqueeze(-1).expand_as(inputs_embeds),
        emb2.to(inputs_embeds.dtype),
    )

    import types
    original_forward = model.forward
    def patched_forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is None and inputs_embeds is not None:
            input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        return original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
    model.forward = types.MethodType(patched_forward, model)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            inputs_embeds=inputs_embeds,
            max_new_tokens=64,
        )
    # Restore original forward
    model.forward = original_forward
    gen_trimmed = gen_ids[0, input_ids.shape[1]:]
    gen_text = processor.tokenizer.decode(gen_trimmed, skip_special_tokens=True)

    r = {
        "layers": n_layers,
        "cold_ms": round(cold_ms, 1),
        "warm_ms": round(warm_ms, 1),
        "text": gen_text,
    }
    results.append(r)
    print(f"  cold={cold_ms:.0f}ms  warm={warm_ms:.0f}ms", flush=True)
    print(f"  text: {repr(gen_text[:150])}", flush=True)

    # Cleanup
    del hybrid, vision_model, emb_tt, emb_tt2
    torch_xla.sync(wait=True)

# Summary
ref = results[0]["text"]
print("\n\n=== SUMMARY ===")
print(f"{'Layers':<8} {'Cold':>8} {'Warm':>8} {'Match':>6}  Text preview")
for r in results:
    match = "YES" if r["text"].strip() == ref.strip() else "NO"
    print(f"{r['layers']:<8} {r['cold_ms']:>7.0f}ms {r['warm_ms']:>7.0f}ms {match:>6}  {repr(r['text'][:80])}")

Path(PROJECT_ROOT / "ocr_lab" / "vision_layer_sweep_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False))
