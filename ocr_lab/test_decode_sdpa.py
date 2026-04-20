#!/usr/bin/env python3
"""Test decode-kernel path: tensor current_pos + decode SDPA kernel."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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

from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.fixed_page import fixed_page_message
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 64

proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

print("Vision (CPU reference)...", flush=True)
vm, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vm = vm.eval().to(dtype=torch.bfloat16)
with torch.no_grad():
    ve = vm(inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"].to(torch.int32)).to(torch.bfloat16)
del vm

hf = AutoModelForCausalLM.from_pretrained(
    str(snapshot_dir),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
img_mask = inputs["input_ids"] == hf.config.image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), ve.to(embeds.dtype))
seq = embeds.shape[1]
vocab_size = hf.config.vocab_size
del hf
print(f"  seq={seq}, embeds={embeds.shape}", flush=True)

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

from ocr_lab.ttnn_decoder_hybrid import (
    DIM,
    align_decode_cache_seq,
    decode_step_sdpa_tt,
    fill_tt_kv_cache,
    init_tt_kv_cache,
    load_decoder_weights,
    make_current_pos_tensor,
    make_decode_compute_kernel_config,
    make_sdpa_decode_program_config,
    prefill,
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
sd = args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(sd, mesh)

cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = cfg.get("rope_theta", 1e6)
max_pos = seq + MAX_NEW_TOKENS + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)
cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)
sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)
cos_full_tt = ttnn.from_torch(
    cos_full.unsqueeze(0).unsqueeze(0),
    device=mesh,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
sin_full_tt = ttnn.from_torch(
    sin_full.unsqueeze(0).unsqueeze(0),
    device=mesh,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

print(f"Prefill ({seq} tokens)...", flush=True)
embeds_tt = ttnn.from_torch(
    embeds.unsqueeze(1),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
t0 = time.perf_counter()
logits_tt, kv_k, kv_v = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_cpu, sin_cpu, seq)
prefill_ms = (time.perf_counter() - t0) * 1000
logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :vocab_size].float()
first_tok = logits_cpu.argmax(-1).item()
print(f"  Prefill: {prefill_ms:.0f}ms, first token: {first_tok} = {repr(tokenizer.decode([first_tok]))}", flush=True)

print("Init TT KV cache...", flush=True)
max_cache = align_decode_cache_seq(seq + MAX_NEW_TOKENS + 32)
tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)
fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)
del kv_k, kv_v

decode_pos_tensors = [
    make_current_pos_tensor(cur_seq, mesh)
    for cur_seq in range(seq, seq + MAX_NEW_TOKENS - 1)
]
sdpa_program_config = make_sdpa_decode_program_config()
sdpa_compute_kernel_config = make_decode_compute_kernel_config()

generated = [first_tok]
next_token = first_tok
decode_times = []

print(f"Decode {MAX_NEW_TOKENS - 1} tokens (sdpa_decode kernel path)...", flush=True)
for step in range(1, MAX_NEW_TOKENS):
    token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
    tt_tok = ttnn.from_torch(
        token_embed,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    current_pos_tt = decode_pos_tensors[step - 1]
    current_pos = seq + step - 1

    t0 = time.perf_counter()
    tt_logits = decode_step_sdpa_tt(
        tt_tok,
        dec_layers,
        dec_final,
        mesh,
        cos_full_tt,
        sin_full_tt,
        tt_cache,
        current_pos_tt,
        current_pos,
        sdpa_program_config=sdpa_program_config,
        sdpa_compute_kernel_config=sdpa_compute_kernel_config,
    )
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_times.append(decode_ms)

    logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
    next_token = logits_d.argmax(-1).item()
    generated.append(next_token)

    if step <= 10 or step % 10 == 0 or decode_ms > 200:
        print(f"  Step {step}: {decode_ms:.0f}ms, token={next_token}={repr(tokenizer.decode([next_token]))}", flush=True)

gen_text = tokenizer.decode(generated, skip_special_tokens=True)
ref = (
    "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\n"
    "Mahkememizden verilen işbu 28/12/2017 tarihli"
)
avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
avg_warm = sum(decode_times[2:]) / len(decode_times[2:]) if len(decode_times) > 2 else avg_decode

print(f"\n{'=' * 60}")
print(f"  Prefill:       {prefill_ms:.0f}ms ({seq} tokens)")
print(f"  Avg decode:    {avg_decode:.0f}ms/token (all)")
print(f"  Avg warm:      {avg_warm:.0f}ms/token (skip first 2)")
print(f"  Total decode:  {sum(decode_times):.0f}ms ({len(generated)} tokens)")
print(f"  Generated:     {repr(gen_text[:200])}")
print(f"  Reference:     {repr(ref[:200])}")
print(f"  Match:         {gen_text.strip().startswith(ref.strip())}")
print(f"{'=' * 60}")

for current_pos_tt in decode_pos_tensors:
    ttnn.deallocate(current_pos_tt)
ttnn.close_mesh_device(mesh)
