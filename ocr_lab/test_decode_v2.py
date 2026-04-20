#!/usr/bin/env python3
"""Test optimized decode: fused QKV + on-device head split/concat, zero CPU transfers."""
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

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
OUTPUT_TXT = PROJECT_ROOT / "ocr_lab" / "1pdf_ttnn_full_output.txt"
OUTPUT_JSON = PROJECT_ROOT / "ocr_lab" / "1pdf_ttnn_full_output_meta.json"

# --- Build inputs ---
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

# --- CPU vision (reference, correct) ---
print("Vision (CPU reference)...", flush=True)
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    ve = vm(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"].to(torch.int32)).to(torch.bfloat16)
del vm

# --- Build embeds ---
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_mask = inputs["input_ids"] == hf.config.image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), ve.to(embeds.dtype))
seq = embeds.shape[1]
del hf
print(f"  seq={seq}, embeds={embeds.shape}", flush=True)

# --- TT setup ---
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
sd = args.load_state_dict()

from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, prefill, decode_step_tt, precompute_attn_masks,
    init_tt_kv_cache, fill_tt_kv_cache,
    DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
)

dec_layers, dec_final = load_decoder_weights(sd, mesh)

# --- Rotary tables (pre-loaded on device, shared across all decode steps) ---
cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = cfg.get("rope_theta", 1e6)
max_pos = seq + MAX_NEW_TOKENS + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)

# CPU cos/sin for prefill
cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)  # [1, max_pos, 1, 128]
sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)

# TT cos/sin tables for decode (pre-loaded ONCE, reused every step)
cos_full_tt = ttnn.from_torch(
    cos_full.unsqueeze(0).unsqueeze(0),  # [1, 1, max_pos, 128]
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
sin_full_tt = ttnn.from_torch(
    sin_full.unsqueeze(0).unsqueeze(0),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)

# --- Prefill (CPU SDPA, proven correct) ---
print(f"Prefill ({seq} tokens)...", flush=True)
embeds_tt = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
t0 = time.perf_counter()
logits_tt, kv_k, kv_v = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_cpu, sin_cpu, seq)
prefill_ms = (time.perf_counter() - t0) * 1000
logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :cfg.get('vocab_size', 151936)].float()
first_tok = logits_cpu.argmax(-1).item()
print(f"  Prefill: {prefill_ms:.0f}ms, first token: {first_tok} = {repr(tokenizer.decode([first_tok]))}", flush=True)

# --- Init TT KV cache + fill from prefill ---
print("Init TT KV cache...", flush=True)
max_cache = seq + MAX_NEW_TOKENS + 32
# Pad to tile boundary
max_cache = ((max_cache + 31) // 32) * 32
tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)
fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)
del kv_k, kv_v

decode_masks = precompute_attn_masks(
    start_pos=seq,
    num_steps=MAX_NEW_TOKENS - 1,
    max_cache_seq=max_cache,
    device=mesh,
)

# --- Decode loop (optimized: zero CPU transfers + attn mask) ---
generated = [first_tok]
cur_pos = seq
next_token = first_tok
decode_times = []

print(f"Decode {MAX_NEW_TOKENS - 1} tokens (optimized v2 + attn mask)...", flush=True)
for step in range(1, MAX_NEW_TOKENS):
    token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
    tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    t0 = time.perf_counter()
    tt_logits = decode_step_tt(tt_tok, dec_layers, dec_final, mesh,
                                cos_full_tt, sin_full_tt, tt_cache, cur_pos,
                                attn_mask_tt=decode_masks[step - 1])
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_times.append(decode_ms)

    logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :cfg.get('vocab_size', 151936)].float()
    next_token = logits_d.argmax(-1).item()
    generated.append(next_token)
    cur_pos += 1

    if step <= 10 or step % 10 == 0 or decode_ms > 200:
        print(f"  Step {step}: {decode_ms:.0f}ms, token={next_token}={repr(tokenizer.decode([next_token]))}", flush=True)

# --- Results ---
gen_text = tokenizer.decode(generated, skip_special_tokens=True)
REF = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli"
avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
avg_warm = sum(decode_times[2:]) / len(decode_times[2:]) if len(decode_times) > 2 else avg_decode

OUTPUT_TXT.write_text(gen_text, encoding="utf-8")
OUTPUT_JSON.write_text(
    json.dumps(
        {
            "image": str(IMAGE),
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt": PROMPT,
            "seq": seq,
            "prefill_ms": prefill_ms,
            "avg_decode_ms": avg_decode,
            "avg_warm_ms": avg_warm,
            "total_decode_ms": sum(decode_times),
            "generated_tokens": len(generated),
            "output_txt": str(OUTPUT_TXT),
            "output_preview": gen_text[:400],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"\n{'='*60}")
print(f"  Prefill:       {prefill_ms:.0f}ms ({seq} tokens)")
print(f"  Avg decode:    {avg_decode:.0f}ms/token (all)")
print(f"  Avg warm:      {avg_warm:.0f}ms/token (skip first 2)")
print(f"  Total decode:  {sum(decode_times):.0f}ms ({len(generated)} tokens)")
print(f"  Generated:     {repr(gen_text[:200])}")
print(f"  Reference:     {repr(REF[:200])}")
print(f"  Match:         {gen_text.strip().startswith(REF.strip())}")
print(f"  Saved text:    {OUTPUT_TXT}")
print(f"  Saved meta:    {OUTPUT_JSON}")
print(f"{'='*60}")

for mask in decode_masks:
    ttnn.deallocate(mask)
ttnn.close_mesh_device(mesh)
