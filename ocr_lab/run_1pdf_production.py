#!/usr/bin/env python3
"""
Production OCR: 1.pdf full extraction with nm-dots-ocr-service parameters.

Pipeline: HF model (fp32) → vision + prefill on CPU → decode on TT
Parameters: temperature=0.1, top_p=0.9, frequency_penalty=0.03, logit_bias
Prompt: prompt_ocr ("Extract the text content from this image.")
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import torch, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TT_METAL = PROJECT_ROOT / "tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
sys.path.insert(0, str(TT_METAL))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ocr_lab" / "shims"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"

from ocr_lab.dots_mocr_tt_metal_adapter import patch_dots_mocr_config_for_tt_metal
snapshot_dir = patch_dots_mocr_config_for_tt_metal()

from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from ocr_lab.dots_model import ensure_dots_ocr_processor_compat

# ============ Production config (from nm-dots-ocr-service) ============
PDF_PATH = os.environ.get("PDF_PATH", "/home/mlops/Desktop/dosyalar/1.pdf")
PDF_DPI = 200
PROMPT = "Extract the text content from this image."  # prompt_ocr
MAX_TOKENS = 8192
TEMPERATURE = 0.1
TOP_P = 0.9
FREQUENCY_PENALTY = 0.03
LOGIT_BIAS = {42269:-8, 3513:-8, 46932:-8, 4077:-8, 8152:-8, 9822:-8,
              13067:-8, 79518:-8, 68700:-8, 11304:-8, 1177:-8, 481:-8}
EOS_IDS = {151643, 151645}

OUTPUT_DIR = PROJECT_ROOT / "ocr_lab" / "1pdf_production_output"
OUTPUT_DIR.mkdir(exist_ok=True)

total_start = time.perf_counter()

# ============ Extract pages from PDF ============
print(f"=== PDF: {PDF_PATH} ===", flush=True)
import fitz
doc = fitz.open(PDF_PATH)
n_pages = len(doc)
page_images = []
for i in range(n_pages):
    pix = doc[i].get_pixmap(dpi=PDF_DPI)
    img_path = OUTPUT_DIR / f"page_{i+1}.png"
    pix.save(str(img_path))
    page_images.append(str(img_path))
    print(f"  Page {i+1}: {pix.width}x{pix.height}", flush=True)
doc.close()

# ============ Load model (fp32 for CPU speed) ============
print("\nLoading model (fp32)...", flush=True)
t_load = time.perf_counter()
ensure_dots_ocr_processor_compat(str(snapshot_dir))
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.float32, attn_implementation='sdpa').eval()
# Patch vision SDPA
vision_module = sys.modules.get(hf.vision_tower.__class__.__module__)
if vision_module and hasattr(vision_module, "VisionSdpaAttention"):
    for blk in hf.vision_tower.blocks:
        old = blk.attn
        new = vision_module.VisionSdpaAttention(hf.config.vision_config, old.proj.in_features,
                                                 num_heads=old.num_heads, bias=old.qkv.bias is not None)
        new.load_state_dict(old.state_dict())
        new = new.to(device=old.qkv.weight.device, dtype=old.qkv.weight.dtype)
        blk.attn = new
load_ms = (time.perf_counter() - t_load) * 1000
print(f"  Model loaded: {load_ms:.0f}ms", flush=True)

# ============ TT setup for decode ============
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, decode_step,
    DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
sd = args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(sd, mesh)

cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
vocab_size = cfg.get('vocab_size', 151936)
rope_theta = cfg.get("rope_theta", 1e6)

# Pre-build logit bias tensor
logit_bias_tensor = torch.zeros(vocab_size)
for tok_id, bias in LOGIT_BIAS.items():
    if tok_id < vocab_size:
        logit_bias_tensor[tok_id] = bias

emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)

# ============ Process each page ============
all_results = []
for page_idx, img_path in enumerate(page_images):
    page_start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"  PAGE {page_idx+1}/{n_pages}: {img_path}")
    print(f"{'='*60}", flush=True)

    # Build inputs (native resolution)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": PROMPT},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    seq = inputs["input_ids"].shape[1]
    print(f"  pixel_values: {inputs['pixel_values'].shape}, seq: {seq}", flush=True)

    # --- Vision (fp32) ---
    t_vis = time.perf_counter()
    with torch.no_grad():
        vision_out = hf.vision_tower(inputs["pixel_values"].to(torch.float32),
                                      inputs["image_grid_thw"], bf16=False)
    vision_ms = (time.perf_counter() - t_vis) * 1000
    print(f"  Vision: {vision_ms:.0f}ms ({vision_ms/1000:.1f}s)", flush=True)

    # --- Build embeds + decoder prefill (fp32) ---
    emb_w_fp32 = hf.get_input_embeddings().weight.detach()
    img_mask = inputs["input_ids"] == hf.config.image_token_id
    embeds = F.embedding(inputs["input_ids"], emb_w_fp32)
    embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_out.to(embeds.dtype))

    t_pf = time.perf_counter()
    with torch.no_grad():
        outputs = hf(input_ids=inputs["input_ids"], inputs_embeds=embeds, use_cache=True)
    prefill_ms = (time.perf_counter() - t_pf) * 1000
    print(f"  Prefill: {prefill_ms:.0f}ms ({prefill_ms/1000:.1f}s)", flush=True)

    logits_last = outputs.logits[0, -1, :vocab_size].float()
    first_tok = logits_last.argmax(-1).item()
    past_kv = outputs.past_key_values

    # --- Convert KV cache for TT decode ---
    kv_cache_k = {}
    kv_cache_v = {}
    for i in range(N_LAYERS):
        kv_cache_k[i] = past_kv[i][0].squeeze(0).float()
        kv_cache_v[i] = past_kv[i][1].squeeze(0).float()
    del past_kv

    # Rotary
    max_pos = seq + MAX_TOKENS + 32
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    emb_rope = torch.cat((freqs, freqs), dim=-1)
    cos_cpu = emb_rope.cos().float().unsqueeze(1).unsqueeze(0)
    sin_cpu = emb_rope.sin().float().unsqueeze(1).unsqueeze(0)

    # --- TT Decode with production sampling ---
    generated = [first_tok]
    cur_pos = seq
    next_token = first_tok
    token_counts = {}
    t_dec = time.perf_counter()

    for step in range(1, MAX_TOKENS):
        token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
        tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        cos_step = cos_cpu[:, cur_pos:cur_pos+1]
        sin_step = sin_cpu[:, cur_pos:cur_pos+1]

        tt_logits = decode_step(tt_tok, dec_layers, dec_final, mesh,
                                 cos_step, sin_step, kv_cache_k, kv_cache_v)
        logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()

        # Apply logit bias
        logits_d = logits_d + logit_bias_tensor

        # Frequency penalty
        if FREQUENCY_PENALTY > 0 and token_counts:
            for tok_id, count in token_counts.items():
                if tok_id < vocab_size:
                    logits_d[tok_id] -= FREQUENCY_PENALTY * count

        # Temperature + top-p sampling
        logits_d = logits_d / TEMPERATURE
        sorted_logits, sorted_indices = torch.sort(logits_d, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > TOP_P
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits_d[indices_to_remove] = float('-inf')
        probs = F.softmax(logits_d, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        token_counts[next_token] = token_counts.get(next_token, 0) + 1
        generated.append(next_token)
        cur_pos += 1

        if next_token in EOS_IDS:
            break

    decode_ms = (time.perf_counter() - t_dec) * 1000
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    page_ms = (time.perf_counter() - page_start) * 1000

    n_tokens = len(generated)
    avg_tok_ms = decode_ms / n_tokens if n_tokens > 0 else 0

    print(f"  Decode: {decode_ms:.0f}ms ({n_tokens} tokens, {avg_tok_ms:.0f}ms/tok)", flush=True)
    print(f"  Page total: {page_ms:.0f}ms ({page_ms/1000:.1f}s)", flush=True)
    print(f"  Text preview: {repr(gen_text[:200])}", flush=True)

    # Save page output
    page_out = OUTPUT_DIR / f"page_{page_idx+1}_text.txt"
    page_out.write_text(gen_text, encoding="utf-8")

    all_results.append({
        "page": page_idx + 1,
        "seq": seq,
        "vision_ms": vision_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "tokens": n_tokens,
        "page_total_ms": page_ms,
        "text": gen_text,
    })

    # Free KV cache
    del kv_cache_k, kv_cache_v

# ============ Summary ============
total_ms = (time.perf_counter() - total_start) * 1000

print(f"\n{'='*60}")
print(f"  TOTAL E2E: {total_ms:.0f}ms ({total_ms/1000:.1f}s)")
print(f"  Pages: {n_pages}")
print(f"  Pipeline: fp32 CPU vision+prefill → TT matmul+CPU SDPA decode")
print(f"  Config: temp={TEMPERATURE} top_p={TOP_P} freq_pen={FREQUENCY_PENALTY}")
print(f"  Logit bias: {len(LOGIT_BIAS)} suppressed tokens")
for r in all_results:
    print(f"  Page {r['page']}: vision={r['vision_ms']:.0f}ms prefill={r['prefill_ms']:.0f}ms decode={r['decode_ms']:.0f}ms tokens={r['tokens']} total={r['page_total_ms']:.0f}ms")
print(f"{'='*60}")

# Save full results
full_out = OUTPUT_DIR / "results.json"
full_out.write_text(json.dumps({
    "pdf": PDF_PATH,
    "pages": n_pages,
    "total_ms": total_ms,
    "config": {
        "temperature": TEMPERATURE, "top_p": TOP_P,
        "frequency_penalty": FREQUENCY_PENALTY,
        "logit_bias_count": len(LOGIT_BIAS),
        "pdf_dpi": PDF_DPI, "prompt": PROMPT,
    },
    "page_results": all_results,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n  Full text saved: {OUTPUT_DIR}")

# Print full OCR text
print(f"\n{'='*60}")
print("  FULL OCR OUTPUT")
print(f"{'='*60}")
for r in all_results:
    print(f"\n--- Page {r['page']} ---")
    print(r['text'])

ttnn.close_mesh_device(mesh)
