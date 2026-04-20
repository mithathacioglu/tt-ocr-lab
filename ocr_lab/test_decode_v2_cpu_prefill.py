#!/usr/bin/env python3
"""Hybrid: CPU prefill (exact accuracy) + TT decode (fast).

CPU prefill ensures bit-exact KV cache, TT decode gives speed.
Native resolution — no fixed_page.
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

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

OUTPUT_TXT = PROJECT_ROOT / "ocr_lab" / "1pdf_ttnn_cpuprefill_output.txt"
OUTPUT_JSON = PROJECT_ROOT / "ocr_lab" / "1pdf_ttnn_cpuprefill_output_meta.json"

# --- Build inputs — native resolution ---
ensure_dots_ocr_processor_compat(str(snapshot_dir))
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
msgs = [{"role": "user", "content": [
    {"type": "image", "image": str(IMAGE)},
    {"type": "text", "text": PROMPT},
]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
print(f"Native resolution")
print(f"pixel_values: {inputs['pixel_values'].shape}")
print(f"input_ids: {inputs['input_ids'].shape}")

# --- Load HF model for CPU prefill ---
print("Loading HF model for CPU prefill...", flush=True)
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()

# Patch vision tower to use SDPA
vision_module = sys.modules.get(hf.vision_tower.__class__.__module__)
if vision_module is not None and hasattr(vision_module, "VisionSdpaAttention"):
    for blk in hf.vision_tower.blocks:
        old_attn = blk.attn
        new_attn = vision_module.VisionSdpaAttention(
            hf.config.vision_config,
            old_attn.proj.in_features,
            num_heads=old_attn.num_heads,
            bias=old_attn.qkv.bias is not None,
        )
        new_attn.load_state_dict(old_attn.state_dict())
        new_attn = new_attn.to(device=old_attn.qkv.weight.device, dtype=old_attn.qkv.weight.dtype)
        blk.attn = new_attn

# --- CPU Prefill: vision + full forward pass to get KV cache ---
print("CPU prefill (vision + decoder)...", flush=True)
t_prefill_start = time.perf_counter()

with torch.no_grad():
    # Single forward pass — returns logits + past_key_values
    outputs = hf(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
        image_grid_thw=inputs["image_grid_thw"],
        use_cache=True,
    )

cpu_prefill_ms = (time.perf_counter() - t_prefill_start) * 1000
seq = inputs["input_ids"].shape[1]
cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
vocab_size = cfg.get('vocab_size', 151936)

logits_last = outputs.logits[0, -1, :vocab_size].float()
first_tok = logits_last.argmax(-1).item()
past_kv = outputs.past_key_values
print(f"  CPU prefill: {cpu_prefill_ms:.0f}ms ({cpu_prefill_ms/1000:.1f}s)")
print(f"  seq={seq}, first token: {first_tok} = {repr(tokenizer.decode([first_tok]))}")

# Check KV cache shape
k0 = past_kv[0][0]  # layer 0, key
print(f"  KV cache shape: {k0.shape}")  # expect [1, n_kv_heads, seq, head_dim]

# --- CPU decode loop using HF model with past_key_values ---
print(f"\nCPU decode (HF model, {MAX_NEW_TOKENS-1} tokens)...", flush=True)
generated = [first_tok]
next_token = first_tok
decode_times = []

for step in range(1, MAX_NEW_TOKENS):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = hf(
            input_ids=torch.tensor([[next_token]]),
            past_key_values=past_kv,
            use_cache=True,
        )
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_times.append(decode_ms)
    past_kv = out.past_key_values

    logits_d = out.logits[0, -1, :vocab_size].float()
    next_token = logits_d.argmax(-1).item()
    generated.append(next_token)

    if next_token in (151643, 151645):
        print(f"  Step {step}: EOS, stopping.", flush=True)
        break

    if step <= 3 or step % 20 == 0:
        print(f"  Step {step}: {decode_ms:.0f}ms, tok={repr(tokenizer.decode([next_token]))}", flush=True)

gen_text_cpu = tokenizer.decode(generated, skip_special_tokens=True)
avg_cpu_decode = sum(decode_times) / len(decode_times) if decode_times else 0
print(f"  CPU decode: avg {avg_cpu_decode:.0f}ms/tok, {len(generated)} tokens")
print(f"  CPU text: {repr(gen_text_cpu[:300])}")

# ============================================================
# Now: TT decode from the same prefill KV cache
# ============================================================
print(f"\n{'='*60}")
print("TT decode from CPU prefill KV cache")
print(f"{'='*60}")

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, decode_step_tt, precompute_attn_masks,
    init_tt_kv_cache, DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
sd = args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(sd, mesh)

emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
del hf  # Free memory

# Rotary tables
rope_theta = cfg.get("rope_theta", 1e6)
max_pos = seq + MAX_NEW_TOKENS + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)

cos_full_tt = ttnn.from_torch(cos_full.unsqueeze(0).unsqueeze(0),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
sin_full_tt = ttnn.from_torch(sin_full.unsqueeze(0).unsqueeze(0),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Init TT KV cache from CPU past_key_values
print("Filling TT KV cache from CPU prefill...", flush=True)
max_cache = seq + MAX_NEW_TOKENS + 32
max_cache = ((max_cache + 31) // 32) * 32
tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)

for layer_idx in range(N_LAYERS):
    # HF past_kv[layer] = (key, value), shape [1, n_kv_heads, seq, head_dim]
    k_cpu = past_kv[layer_idx][0].to(torch.bfloat16)  # [1, 2, seq, 128]
    v_cpu = past_kv[layer_idx][1].to(torch.bfloat16)

    k_tt = ttnn.from_torch(k_cpu, device=mesh, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_tt = ttnn.from_torch(v_cpu, device=mesh, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.fill_cache(tt_cache[layer_idx][0], k_tt, 0)
    ttnn.fill_cache(tt_cache[layer_idx][1], v_tt, 0)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)

del past_kv

decode_masks = precompute_attn_masks(
    start_pos=seq, num_steps=MAX_NEW_TOKENS - 1,
    max_cache_seq=max_cache, device=mesh)

# TT decode loop
tt_generated = [first_tok]
cur_pos = seq
next_token = first_tok
tt_decode_times = []

print(f"TT decode up to {MAX_NEW_TOKENS - 1} tokens...", flush=True)
for step in range(1, MAX_NEW_TOKENS):
    token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
    tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t0 = time.perf_counter()
    tt_logits = decode_step_tt(tt_tok, dec_layers, dec_final, mesh,
                                cos_full_tt, sin_full_tt, tt_cache, cur_pos,
                                attn_mask_tt=decode_masks[step - 1])
    decode_ms = (time.perf_counter() - t0) * 1000
    tt_decode_times.append(decode_ms)

    logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
    next_token = logits_d.argmax(-1).item()
    tt_generated.append(next_token)
    cur_pos += 1

    if next_token in (151643, 151645):
        print(f"  Step {step}: EOS, stopping.", flush=True)
        break

    if step <= 3 or step % 20 == 0:
        print(f"  Step {step}: {decode_ms:.0f}ms, tok={repr(tokenizer.decode([next_token]))}", flush=True)

tt_gen_text = tokenizer.decode(tt_generated, skip_special_tokens=True)
avg_tt_decode = sum(tt_decode_times) / len(tt_decode_times) if tt_decode_times else 0
avg_tt_warm = sum(tt_decode_times[2:]) / len(tt_decode_times[2:]) if len(tt_decode_times) > 2 else avg_tt_decode

# --- Compare ---
REF_CPU = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli hükümler, Davacı 'e ve davalı 'e 18/01/2018 tarihinde tebliğ olunmuş, tarafların 18/01/2018 tarihinde vermiş olduğu \"İstinaftan Feragat Dilekçesi\" ile hükmün, 18/01/2018 tarihinde kesinleştiği tasdik olunur. 18/01/2018"
REF_KEYWORDS = ["hükümler", "davalı", "olunmuş", "Dilekçesi", "kesinleştiği"]

cpu_found = [kw for kw in REF_KEYWORDS if kw in gen_text_cpu]
tt_found = [kw for kw in REF_KEYWORDS if kw in tt_gen_text]

token_match = sum(1 for a, b in zip(generated, tt_generated) if a == b)

OUTPUT_TXT.write_text(tt_gen_text, encoding="utf-8")
meta = {
    "resolution": "native",
    "seq": seq,
    "cpu_prefill_ms": cpu_prefill_ms,
    "cpu_decode_avg_ms": avg_cpu_decode,
    "tt_decode_avg_ms": avg_tt_decode,
    "tt_decode_warm_ms": avg_tt_warm,
    "tt_total_decode_ms": sum(tt_decode_times),
    "tt_generated_tokens": len(tt_generated),
    "cpu_generated_tokens": len(generated),
    "token_match": f"{token_match}/{min(len(generated), len(tt_generated))}",
    "cpu_text_preview": gen_text_cpu[:500],
    "tt_text_preview": tt_gen_text[:500],
}
OUTPUT_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  CPU prefill:       {cpu_prefill_ms:.0f}ms ({cpu_prefill_ms/1000:.1f}s)")
print(f"  CPU decode:        {avg_cpu_decode:.0f}ms/tok")
print(f"  TT decode:         {avg_tt_decode:.1f}ms/tok (warm: {avg_tt_warm:.1f}ms)")
print(f"  Speedup decode:    {avg_cpu_decode/avg_tt_warm:.1f}x")
print(f"  Token match:       {token_match}/{min(len(generated), len(tt_generated))}")
print(f"  CPU keywords:      {cpu_found}")
print(f"  TT keywords:       {tt_found}")
print(f"  CPU text == ref:   {gen_text_cpu.strip().startswith(REF_CPU.strip())}")
print(f"  TT text == CPU:    {tt_gen_text == gen_text_cpu}")
print(f"\n  CPU: {repr(gen_text_cpu[:300])}")
print(f"\n  TT:  {repr(tt_gen_text[:300])}")
print(f"\n  Saved: {OUTPUT_TXT}")
print(f"{'='*60}")

for mask in decode_masks:
    ttnn.deallocate(mask)
ttnn.close_mesh_device(mesh)
