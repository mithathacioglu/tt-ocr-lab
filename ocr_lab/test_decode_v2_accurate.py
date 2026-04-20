#!/usr/bin/env python3
"""Accurate pipeline: HF CPU prefill + CPU-SDPA decode (TT matmuls only).

HF model for prefill (exact), then decode_step (TT matmuls + CPU SDPA).
This avoids TT SDPA precision issues at long sequences.
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
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8192"))

OUTPUT_TXT = PROJECT_ROOT / "ocr_lab" / "1pdf_accurate_output.txt"
OUTPUT_JSON = PROJECT_ROOT / "ocr_lab" / "1pdf_accurate_output_meta.json"

# --- Production inference parameters (from nm-dots-ocr-service) ---
TEMPERATURE = 0.1
TOP_P = 0.9
FREQUENCY_PENALTY = 0.03
# Logit bias: suppress garbage/filler tokens
LOGIT_BIAS = {
    42269: -8,  # dots (.)
    3513: -8,   # dashes (-)
    46932: -8,  # underscore (_)
    4077: -8,   # asterisk (*)
    8152: -8,   # equals (=)
    9822: -8,   # slash (/)
    13067: -8,  # hash (#)
    79518: -8,  # percent (%)
    68700: -8,  # dollar ($I)
    11304: -8,
    1177: -8,
    481: -8,
}

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
print(f"Native resolution — pixel_values: {inputs['pixel_values'].shape}, input_ids: {inputs['input_ids'].shape}")

# --- Load HF model ---
# fp32 matmul is 2.8x faster on this CPU (no AVX512-BF16/AMX)
print("Loading HF model (fp32)...", flush=True)
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.float32, attn_implementation='sdpa').eval()

# Patch vision tower for CPU SDPA
vision_module = sys.modules.get(hf.vision_tower.__class__.__module__)
if vision_module is not None and hasattr(vision_module, "VisionSdpaAttention"):
    for blk in hf.vision_tower.blocks:
        old_attn = blk.attn
        new_attn = vision_module.VisionSdpaAttention(
            hf.config.vision_config, old_attn.proj.in_features,
            num_heads=old_attn.num_heads, bias=old_attn.qkv.bias is not None)
        new_attn.load_state_dict(old_attn.state_dict())
        new_attn = new_attn.to(device=old_attn.qkv.weight.device, dtype=old_attn.qkv.weight.dtype)
        blk.attn = new_attn

# --- CPU Prefill ---
print("CPU prefill (HF model)...", flush=True)
# --- Run vision in fp32 (2.8x faster on this CPU without AVX512-BF16) ---
print("Running vision tower (fp32)...", flush=True)
t_vis = time.perf_counter()
with torch.no_grad():
    vision_out = hf.vision_tower(inputs["pixel_values"].to(torch.float32),
                                  inputs["image_grid_thw"], bf16=False)
vision_ms = (time.perf_counter() - t_vis) * 1000
print(f"  Vision: {vision_ms:.0f}ms ({vision_ms/1000:.1f}s)")

# --- Build embeds with vision output, then run decoder prefill (all fp32) ---
emb_w_fp32 = hf.get_input_embeddings().weight.detach()  # fp32
img_mask = inputs["input_ids"] == hf.config.image_token_id
embeds = F.embedding(inputs["input_ids"], emb_w_fp32)
embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_out.to(embeds.dtype))
seq = embeds.shape[1]
# Save bf16 copy for TT decode later
emb_w = emb_w_fp32.to(torch.bfloat16)

print(f"Running decoder prefill (fp32)...", flush=True)
t_pf = time.perf_counter()
with torch.no_grad():
    outputs = hf(input_ids=inputs["input_ids"], inputs_embeds=embeds.unsqueeze(0) if embeds.dim() == 2 else embeds, use_cache=True)
prefill_ms = (time.perf_counter() - t_pf) * 1000

cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
vocab_size = cfg.get('vocab_size', 151936)
cpu_prefill_ms = vision_ms + prefill_ms

logits_last = outputs.logits[0, -1, :vocab_size].float()
first_tok = logits_last.argmax(-1).item()
past_kv = outputs.past_key_values
del hf
print(f"  Decoder prefill: {prefill_ms:.0f}ms ({prefill_ms/1000:.1f}s)")
print(f"  Total prefill: {cpu_prefill_ms:.0f}ms ({cpu_prefill_ms/1000:.1f}s), seq={seq}")
print(f"  First token: {first_tok} = {repr(tokenizer.decode([first_tok]))}")

# --- TT setup for decode ---
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, decode_step,
    DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
    apply_rotary,
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=4096)
sd = args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(sd, mesh)

# Rotary tables (CPU, for decode_step)
rope_theta = cfg.get("rope_theta", 1e6)
max_pos = seq + MAX_NEW_TOKENS + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)
cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)
sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)

# --- Convert HF KV cache to CPU decode format ---
# HF cache: [1, n_kv_heads, seq, head_dim] with post-rotary K
# CPU decode_step expects kv_cache_k[layer] = [n_kv_heads, seq, head_dim]
print("Converting HF KV cache to CPU decode format...", flush=True)
kv_cache_k = {}
kv_cache_v = {}
for i in range(N_LAYERS):
    kv_cache_k[i] = past_kv[i][0].squeeze(0).float()  # always float32 for CPU SDPA
    kv_cache_v[i] = past_kv[i][1].squeeze(0).float()
del past_kv

# --- Decode loop: TT matmuls + CPU SDPA + production sampling ---
generated = [first_tok]
cur_pos = seq
next_token = first_tok
decode_times = []
token_counts = {}  # for frequency penalty

# Pre-build logit bias tensor
logit_bias_tensor = torch.zeros(vocab_size)
for tok_id, bias in LOGIT_BIAS.items():
    if tok_id < vocab_size:
        logit_bias_tensor[tok_id] = bias

print(f"Decode (TT matmul + CPU SDPA, temp={TEMPERATURE}, freq_pen={FREQUENCY_PENALTY})...", flush=True)
for step in range(1, MAX_NEW_TOKENS):
    token_embed = emb_w[next_token].to(torch.bfloat16).reshape(1, 1, 1, DIM)
    tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    cos_step = cos_cpu[:, cur_pos:cur_pos+1]
    sin_step = sin_cpu[:, cur_pos:cur_pos+1]

    t0 = time.perf_counter()
    tt_logits = decode_step(tt_tok, dec_layers, dec_final, mesh,
                             cos_step, sin_step, kv_cache_k, kv_cache_v)
    decode_ms = (time.perf_counter() - t0) * 1000
    decode_times.append(decode_ms)

    logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()

    # Apply logit bias (suppress garbage tokens)
    logits_d = logits_d + logit_bias_tensor

    # Apply frequency penalty
    if FREQUENCY_PENALTY > 0 and token_counts:
        for tok_id, count in token_counts.items():
            if tok_id < vocab_size:
                logits_d[tok_id] -= FREQUENCY_PENALTY * count

    # Temperature sampling
    if TEMPERATURE > 0:
        logits_d = logits_d / TEMPERATURE
        # Top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits_d, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > TOP_P
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits_d[indices_to_remove] = float('-inf')
        # Sample
        probs = F.softmax(logits_d, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
    else:
        next_token = logits_d.argmax(-1).item()

    # Update frequency count
    token_counts[next_token] = token_counts.get(next_token, 0) + 1

    generated.append(next_token)
    cur_pos += 1

    if next_token in (151643, 151645):
        print(f"  Step {step}: EOS, stopping.", flush=True)
        break

    if step <= 3 or step % 50 == 0:
        print(f"  Step {step}: {decode_ms:.0f}ms, tok={repr(tokenizer.decode([next_token]))}", flush=True)

gen_text = tokenizer.decode(generated, skip_special_tokens=True)
avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
avg_warm = sum(decode_times[2:]) / len(decode_times[2:]) if len(decode_times) > 2 else avg_decode
total_e2e = cpu_prefill_ms + sum(decode_times)

OUTPUT_TXT.write_text(gen_text, encoding="utf-8")

REF_CPU = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli hükümler, Davacı 'e ve davalı 'e 18/01/2018 tarihinde tebliğ olunmuş, tarafların 18/01/2018 tarihinde vermiş olduğu \"İstinaftan Feragat Dilekçesi\" ile hükmün, 18/01/2018 tarihinde kesinleştiği tasdik olunur. 18/01/2018"
REF_KEYWORDS = ["hükümler", "davalı", "olunmuş", "Dilekçesi", "kesinleştiği"]
found = [kw for kw in REF_KEYWORDS if kw in gen_text]

meta = {
    "resolution": "native",
    "seq": seq,
    "cpu_prefill_ms": cpu_prefill_ms,
    "decode_path": "TT_matmul + CPU_SDPA",
    "avg_decode_ms": avg_decode,
    "avg_warm_ms": avg_warm,
    "total_decode_ms": sum(decode_times),
    "total_e2e_ms": total_e2e,
    "generated_tokens": len(generated),
    "match_ref": gen_text.strip().startswith(REF_CPU.strip()),
    "keywords_found": found,
    "output_preview": gen_text[:500],
}
OUTPUT_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

print(f"\n{'='*60}")
print(f"  CPU prefill:   {cpu_prefill_ms:.0f}ms ({cpu_prefill_ms/1000:.1f}s)")
print(f"  Decode:        {avg_warm:.1f}ms/tok (TT matmul + CPU SDPA)")
print(f"  Total decode:  {sum(decode_times):.0f}ms ({len(generated)} tokens)")
print(f"  Total E2E:     {total_e2e:.0f}ms ({total_e2e/1000:.1f}s)")
print(f"  Keywords:      {found} / {REF_KEYWORDS}")
print(f"  Match ref:     {gen_text.strip().startswith(REF_CPU.strip())}")
print(f"  Text: {repr(gen_text[:400])}")
print(f"{'='*60}")

ttnn.close_mesh_device(mesh)
