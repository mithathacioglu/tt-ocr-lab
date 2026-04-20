#!/usr/bin/env python3
"""Diagnose decode accuracy drift: CPU-correct path vs TT-optimized path.

Runs both paths side-by-side from the same prefill state, compares tokens
and logits at every step to find exact divergence point and root cause.

Also tests rotary embedding format: CPU rotate_half vs ttnn.experimental.rotary_embedding.
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
from dots_tt_port.vision_tower_smoke import load_vision_model
from ocr_lab.fixed_page import fixed_page_message

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_DIAG_TOKENS = int(os.environ.get("MAX_DIAG_TOKENS", "100"))

# --- Build inputs (same as test_decode_v2.py) ---
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

# --- CPU vision (reference) ---
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
    load_decoder_weights, prefill, decode_step, decode_step_tt,
    precompute_attn_masks, init_tt_kv_cache, fill_tt_kv_cache,
    apply_rotary, DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
)

dec_layers, dec_final = load_decoder_weights(sd, mesh)

# --- Rotary tables ---
cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = cfg.get("rope_theta", 1e6)
max_pos = seq + MAX_DIAG_TOKENS + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)

# CPU cos/sin for prefill + CPU decode path
cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)
sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)

# TT cos/sin tables for TT decode path
cos_full_tt = ttnn.from_torch(
    cos_full.unsqueeze(0).unsqueeze(0),
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
logits_tt, kv_k, kv_v = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_cpu, sin_cpu, seq)
vocab_size = cfg.get('vocab_size', 151936)
logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :vocab_size].float()
first_tok = logits_cpu.argmax(-1).item()
print(f"  First token: {first_tok} = {repr(tokenizer.decode([first_tok]))}", flush=True)

# ============================================================
# TEST 1: Rotary format comparison
# ============================================================
print(f"\n{'='*60}")
print("TEST 1: Rotary embedding format — CPU rotate_half vs TT kernel")
print(f"{'='*60}")

# Pick a position in the middle of decode range
test_pos = seq + 5
# Random Q/K vectors (single token)
torch.manual_seed(42)
q_test = torch.randn(1, 1, N_HEADS, HEAD_DIM, dtype=torch.bfloat16)
k_test = torch.randn(1, 1, N_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)

# CPU rotary (known correct)
cos_step = cos_cpu[:, test_pos:test_pos+1]
sin_step = sin_cpu[:, test_pos:test_pos+1]
q_cpu_rot, k_cpu_rot = apply_rotary(
    q_test.reshape(1, 1, N_HEADS, HEAD_DIM),
    k_test.reshape(1, 1, N_KV_HEADS, HEAD_DIM),
    cos_step, sin_step
)

# TT rotary — Q needs [1, N_HEADS, 1, HEAD_DIM] format for rotary_embedding
q_tt_in = ttnn.from_torch(
    q_test.reshape(1, N_HEADS, 1, HEAD_DIM),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
k_tt_in = ttnn.from_torch(
    k_test.reshape(1, N_KV_HEADS, 1, HEAD_DIM),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)

q_tt_rot = ttnn.experimental.rotary_embedding(q_tt_in, cos_full_tt, sin_full_tt, test_pos)
k_tt_rot = ttnn.experimental.rotary_embedding(k_tt_in, cos_full_tt, sin_full_tt, test_pos)

# TT output may be padded to tile boundary (32) on seq dim
q_raw = ttnn.to_torch(q_tt_rot)
k_raw = ttnn.to_torch(k_tt_rot)
print(f"  TT rotary raw shapes: Q={list(q_raw.shape)}, K={list(k_raw.shape)}")
# Input was [1, N_HEADS, 1, HEAD_DIM], output is [1, N_HEADS, padded, HEAD_DIM]
q_tt_out = q_raw[:, :, :1, :HEAD_DIM].reshape(1, 1, N_HEADS, HEAD_DIM).float()
k_tt_out = k_raw[:, :, :1, :HEAD_DIM].reshape(1, 1, N_KV_HEADS, HEAD_DIM).float()

q_diff = (q_cpu_rot.float() - q_tt_out).abs()
k_diff = (k_cpu_rot.float() - k_tt_out).abs()

print(f"  Q rotary max diff: {q_diff.max().item():.6f}")
print(f"  Q rotary mean diff: {q_diff.mean().item():.6f}")
print(f"  K rotary max diff: {k_diff.max().item():.6f}")
print(f"  K rotary mean diff: {k_diff.mean().item():.6f}")

if q_diff.max().item() > 0.1:
    print("  *** ROTARY FORMAT MISMATCH DETECTED ***")
    # Show first few values to diagnose
    print(f"  CPU Q[:8]: {q_cpu_rot[0,0,0,:8].tolist()}")
    print(f"  TT  Q[:8]: {q_tt_out[0,0,0,:8].tolist()}")
    print(f"  CPU Q[64:72]: {q_cpu_rot[0,0,0,64:72].tolist()}")
    print(f"  TT  Q[64:72]: {q_tt_out[0,0,0,64:72].tolist()}")

    # Test interleaved format cos/sin
    print("\n  Trying INTERLEAVED rotary format...")
    freqs_interleaved = torch.stack([freqs, freqs], dim=-1).reshape(max_pos, -1)
    cos_interleaved = freqs_interleaved.cos().to(torch.bfloat16)
    sin_interleaved = freqs_interleaved.sin().to(torch.bfloat16)

    cos_il_tt = ttnn.from_torch(
        cos_interleaved.unsqueeze(0).unsqueeze(0),
        device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_il_tt = ttnn.from_torch(
        sin_interleaved.unsqueeze(0).unsqueeze(0),
        device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)

    q_tt_in2 = ttnn.from_torch(
        q_test.reshape(1, N_HEADS, 1, HEAD_DIM),
        device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q_il_rot = ttnn.experimental.rotary_embedding(q_tt_in2, cos_il_tt, sin_il_tt, test_pos)
    q_il_raw = ttnn.to_torch(q_il_rot)
    q_il_out = q_il_raw[:, :, :1, :HEAD_DIM].reshape(1, 1, N_HEADS, HEAD_DIM).float()
    q_il_diff = (q_cpu_rot.float() - q_il_out).abs()
    print(f"  Interleaved Q max diff: {q_il_diff.max().item():.6f}")
    if q_il_diff.max().item() < 0.01:
        print("  *** INTERLEAVED FORMAT FIXES THE MISMATCH! ***")
else:
    print("  Rotary format looks OK (diff < 0.1)")

ttnn.deallocate(q_tt_in); ttnn.deallocate(k_tt_in)
ttnn.deallocate(q_tt_rot); ttnn.deallocate(k_tt_rot)

# ============================================================
# TEST 2: Side-by-side decode — CPU vs TT token comparison
# ============================================================
print(f"\n{'='*60}")
print(f"TEST 2: Side-by-side decode ({MAX_DIAG_TOKENS} tokens) — CPU vs TT")
print(f"{'='*60}")

# Clone KV caches for both paths
import copy
kv_k_cpu = copy.deepcopy(kv_k)
kv_v_cpu = copy.deepcopy(kv_v)

# TT KV cache
max_cache = seq + MAX_DIAG_TOKENS + 32
max_cache = ((max_cache + 31) // 32) * 32
tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)
fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)

decode_masks = precompute_attn_masks(
    start_pos=seq,
    num_steps=MAX_DIAG_TOKENS,
    max_cache_seq=max_cache,
    device=mesh,
)

# Both paths start from same first token
cpu_tokens = [first_tok]
tt_tokens = [first_tok]
cpu_next = first_tok
tt_next = first_tok
cur_pos_cpu = seq
cur_pos_tt = seq

diverged_at = None
logit_diffs = []

for step in range(1, MAX_DIAG_TOKENS):
    # --- CPU path ---
    cpu_embed = emb_w[cpu_next].reshape(1, 1, 1, DIM)
    cpu_embed_tt = ttnn.from_torch(cpu_embed, device=mesh, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cos_step = cos_cpu[:, cur_pos_cpu:cur_pos_cpu+1]
    sin_step = sin_cpu[:, cur_pos_cpu:cur_pos_cpu+1]
    cpu_logits = decode_step(cpu_embed_tt, dec_layers, dec_final, mesh, cos_step, sin_step, kv_k_cpu, kv_v_cpu)
    cpu_logits_np = ttnn.to_torch(cpu_logits)[0, 0, 0, :vocab_size].float()
    cpu_next = cpu_logits_np.argmax(-1).item()
    cpu_tokens.append(cpu_next)
    cur_pos_cpu += 1

    # --- TT path ---
    tt_embed = emb_w[tt_next].reshape(1, 1, 1, DIM)
    tt_embed_tt = ttnn.from_torch(tt_embed, device=mesh, dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_logits = decode_step_tt(tt_embed_tt, dec_layers, dec_final, mesh,
                                cos_full_tt, sin_full_tt, tt_cache, cur_pos_tt,
                                attn_mask_tt=decode_masks[step - 1])
    tt_logits_np = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
    tt_next = tt_logits_np.argmax(-1).item()
    tt_tokens.append(tt_next)
    cur_pos_tt += 1

    # Compare
    logit_diff = (cpu_logits_np - tt_logits_np).abs().max().item()
    logit_diffs.append(logit_diff)

    if cpu_next != tt_next and diverged_at is None:
        diverged_at = step
        print(f"\n  *** FIRST DIVERGENCE at step {step} ***")
        print(f"  CPU token: {cpu_next} = {repr(tokenizer.decode([cpu_next]))}")
        print(f"  TT  token: {tt_next} = {repr(tokenizer.decode([tt_next]))}")
        print(f"  Max logit diff: {logit_diff:.4f}")
        # Show top-5 for both
        cpu_top5 = cpu_logits_np.topk(5)
        tt_top5 = tt_logits_np.topk(5)
        print(f"  CPU top5: {[(tokenizer.decode([t.item()]), f'{v.item():.2f}') for t, v in zip(cpu_top5.indices, cpu_top5.values)]}")
        print(f"  TT  top5: {[(tokenizer.decode([t.item()]), f'{v.item():.2f}') for t, v in zip(tt_top5.indices, tt_top5.values)]}")

    if step <= 5 or step % 10 == 0 or (diverged_at and step == diverged_at + 1):
        match = "OK" if cpu_next == tt_next else "DIFF"
        print(f"  Step {step:3d}: CPU={cpu_next:6d} TT={tt_next:6d} logit_diff={logit_diff:.4f} [{match}]"
              f"  cpu={repr(tokenizer.decode([cpu_next]))} tt={repr(tokenizer.decode([tt_next]))}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
cpu_text = tokenizer.decode(cpu_tokens, skip_special_tokens=True)
tt_text = tokenizer.decode(tt_tokens, skip_special_tokens=True)
print(f"  Diverged at step: {diverged_at or 'never'}")
print(f"  Max logit diff:   {max(logit_diffs):.4f}")
print(f"  Mean logit diff:  {sum(logit_diffs)/len(logit_diffs):.4f}")
print(f"  Token match rate: {sum(1 for a,b in zip(cpu_tokens, tt_tokens) if a==b)}/{len(cpu_tokens)}")
print(f"\n  CPU text: {repr(cpu_text[:300])}")
print(f"\n  TT  text: {repr(tt_text[:300])}")

# Save detailed results
results = {
    "diverged_at_step": diverged_at,
    "max_logit_diff": max(logit_diffs),
    "mean_logit_diff": sum(logit_diffs)/len(logit_diffs),
    "token_match_rate": f"{sum(1 for a,b in zip(cpu_tokens, tt_tokens) if a==b)}/{len(cpu_tokens)}",
    "logit_diffs_per_step": logit_diffs,
    "cpu_tokens": cpu_tokens,
    "tt_tokens": tt_tokens,
    "cpu_text": cpu_text,
    "tt_text": tt_text,
}
out_path = PROJECT_ROOT / "ocr_lab" / "diagnose_decode_drift_results.json"
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n  Saved: {out_path}")

# Cleanup
for mask in decode_masks:
    ttnn.deallocate(mask)
ttnn.close_mesh_device(mesh)
