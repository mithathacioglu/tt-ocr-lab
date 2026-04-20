#!/usr/bin/env python3
"""Batch test: 4 pages from '3.1-3.2 Birleşik.pdf' — sequential decode with optimized pipeline."""
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
from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.fixed_page import fixed_page_message

PAGES = [
    PROJECT_ROOT / "ocr_lab" / "tmp_batch4" / f"page_{i}.png"
    for i in range(1, 5)
]
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 256  # full page content

# --- Load models once ---
print("Loading processor + vision + embeddings...", flush=True)
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)

vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)

hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
image_token_id = hf.config.image_token_id
vocab_size = hf.config.vocab_size
del hf

# --- TT setup ---
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, prefill, decode_step_tt,
    init_tt_kv_cache, fill_tt_kv_cache,
    DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
sd = args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(sd, mesh)

# --- Rotary tables ---
cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = cfg.get("rope_theta", 1e6)
max_pos = 2048
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)
cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)
sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)
cos_full_tt = ttnn.from_torch(cos_full.unsqueeze(0).unsqueeze(0), device=mesh,
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
sin_full_tt = ttnn.from_torch(sin_full.unsqueeze(0).unsqueeze(0), device=mesh,
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print("Setup done.", flush=True)


def process_page(page_path, page_idx):
    """Process one page: vision + prefill + decode. Returns (text, timings)."""
    t_total = time.perf_counter()

    # --- Build inputs ---
    img_msg, _ = fixed_page_message(page_path, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    # --- Vision (CPU) ---
    t0 = time.perf_counter()
    with torch.no_grad():
        ve = vm(inputs["pixel_values"].to(torch.bfloat16),
                inputs["image_grid_thw"].to(torch.int32)).to(torch.bfloat16)
    vision_ms = (time.perf_counter() - t0) * 1000

    # --- Build embeds ---
    img_mask = inputs["input_ids"] == image_token_id
    embeds = F.embedding(inputs["input_ids"], emb_w)
    embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), ve.to(embeds.dtype))
    seq = embeds.shape[1]

    # --- Prefill ---
    t0 = time.perf_counter()
    embeds_tt = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logits_tt, kv_k, kv_v = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_cpu, sin_cpu, seq)
    prefill_ms = (time.perf_counter() - t0) * 1000
    logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :vocab_size].float()
    first_tok = logits_cpu.argmax(-1).item()

    # --- Init KV cache ---
    tt_cache = init_tt_kv_cache(mesh, max_seq=seq + MAX_NEW_TOKENS + 32)
    fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)
    del kv_k, kv_v

    # --- Decode ---
    generated = [first_tok]
    cur_pos = seq
    next_token = first_tok
    t0 = time.perf_counter()
    for step in range(1, MAX_NEW_TOKENS):
        token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
        tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_logits = decode_step_tt(tt_tok, dec_layers, dec_final, mesh,
                                    cos_full_tt, sin_full_tt, tt_cache, cur_pos)
        logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
        next_token = logits_d.argmax(-1).item()
        generated.append(next_token)
        cur_pos += 1
        # Stop at EOS
        if next_token in (151643, 151645):  # <|endoftext|>, <|im_end|>
            break
    decode_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000

    # Cleanup KV cache
    for k, v in tt_cache:
        ttnn.deallocate(k); ttnn.deallocate(v)

    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    n_tokens = len(generated)
    ms_per_tok = decode_ms / max(n_tokens - 1, 1)

    return gen_text, {
        "page": page_idx + 1,
        "seq": seq,
        "tokens": n_tokens,
        "vision_ms": vision_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "ms_per_tok": ms_per_tok,
        "total_ms": total_ms,
    }


# === Process all 4 pages ===
print(f"\n{'='*60}")
print(f"  Processing 4 pages from 3.1-3.2 Birleşik.pdf")
print(f"{'='*60}\n")

all_timings = []
all_texts = []
grand_start = time.perf_counter()

for i, page_path in enumerate(PAGES):
    print(f"--- Page {i+1}/{len(PAGES)} ---", flush=True)
    gen_text, timings = process_page(page_path, i)
    all_timings.append(timings)
    all_texts.append(gen_text)
    print(f"  Vision: {timings['vision_ms']:.0f}ms | Prefill: {timings['prefill_ms']:.0f}ms | "
          f"Decode: {timings['decode_ms']:.0f}ms ({timings['tokens']} tok, {timings['ms_per_tok']:.0f}ms/tok) | "
          f"Total: {timings['total_ms']:.0f}ms", flush=True)
    print(f"  Text: {repr(gen_text[:120])}...\n", flush=True)

grand_total = (time.perf_counter() - grand_start) * 1000

# === Summary ===
total_tokens = sum(t["tokens"] for t in all_timings)
avg_page_ms = sum(t["total_ms"] for t in all_timings) / len(all_timings)
pages_per_sec = len(PAGES) / (grand_total / 1000)

print(f"\n{'='*60}")
print(f"  BATCH SUMMARY — 4 pages")
print(f"{'='*60}")
for t in all_timings:
    print(f"  Page {t['page']}: {t['total_ms']:.0f}ms ({t['tokens']} tokens)")
print(f"  ─────────────────────────────")
print(f"  Total time:     {grand_total:.0f}ms ({grand_total/1000:.1f}s)")
print(f"  Avg per page:   {avg_page_ms:.0f}ms ({avg_page_ms/1000:.1f}s)")
print(f"  Pages/second:   {pages_per_sec:.2f}")
print(f"  Total tokens:   {total_tokens}")
print(f"{'='*60}")

ttnn.close_mesh_device(mesh)
