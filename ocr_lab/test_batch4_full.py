#!/usr/bin/env python3
"""Full pipeline batch: 4 pages, ttnn vision + ttnn decoder."""
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
from dots_tt_port.vision_tower_smoke import (
    resolve_snapshot_dir, load_vision_model, load_vision_state,
    apply_rotary_pos_emb_cached_cpu, CpuPatchMerger,
)
from ocr_lab.fixed_page import fixed_page_message
import ttnn

PAGES = [PROJECT_ROOT / "ocr_lab" / "tmp_batch4" / f"page_{i}.png" for i in range(1, 5)]
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 256

# --- Constants ---
DIM_V, N_HEADS_V, HEAD_DIM_V, EPS_V, N_BLOCKS_V = 1536, 12, 128, 1e-5, 42

# --- Load shared resources ---
print("Loading processor + embeddings...", flush=True)
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
hf = AutoModelForCausalLM.from_pretrained(str(snapshot_dir), trust_remote_code=True,
                                           torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
emb_w = hf.get_input_embeddings().weight.detach().cpu().to(torch.bfloat16)
image_token_id = hf.config.image_token_id
vocab_size = hf.config.vocab_size
del hf

# --- Prepare all page inputs ---
print("Preparing page inputs...", flush=True)
page_inputs = []
for p in PAGES:
    img_msg, _ = fixed_page_message(p, width=476, height=674)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    page_inputs.append(inputs)

# ====================================================================
# PHASE 1: TTNN VISION (all 4 pages)
# ====================================================================
print("\n=== PHASE 1: TTNN Vision ===", flush=True)
device = ttnn.open_device(device_id=0)
device.enable_program_cache()

# Load vision model + weights (once)
vision_model, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
vision_model = vision_model.eval().to(dtype=torch.bfloat16)
vs = load_vision_state(snapshot_dir, limit_layers=0)
post_norm_w = vision_model.post_trunk_norm.weight.detach().cpu()
cpu_merger = CpuPatchMerger(vision_model.merger)

def make_vw(t): return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

vblocks = []
for i in range(N_BLOCKS_V):
    p = f"blocks.{i}."
    w = {}
    w["norm1"] = ttnn.from_torch(vs[f"{p}norm1.weight"].unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w["norm2"] = ttnn.from_torch(vs[f"{p}norm2.weight"].unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w["qkv_w"] = make_vw(vs[f"{p}attn.qkv.weight"].T)
    w["proj_w"] = make_vw(vs[f"{p}attn.proj.weight"].T)
    w["fc1_w"] = make_vw(vs[f"{p}mlp.fc1.weight"].T)
    w["fc2_w"] = make_vw(vs[f"{p}mlp.fc2.weight"].T)
    w["fc3_w"] = make_vw(vs[f"{p}mlp.fc3.weight"].T)
    vblocks.append(w)
post_norm_tt = ttnn.from_torch(post_norm_w.unsqueeze(0).view(1,1,DIM_V//32,32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def vision_block(x, w, dev, seq, cos, sin):
    norm1 = ttnn.rms_norm(x, epsilon=EPS_V, weight=w["norm1"])
    qkv = ttnn.linear(norm1, w["qkv_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(norm1)
    qkv_cpu = ttnn.to_torch(qkv).float(); ttnn.deallocate(qkv)
    q = qkv_cpu[..., :DIM_V].reshape(seq, N_HEADS_V, HEAD_DIM_V)
    k = qkv_cpu[..., DIM_V:2*DIM_V].reshape(seq, N_HEADS_V, HEAD_DIM_V)
    v = qkv_cpu[..., 2*DIM_V:].reshape(seq, N_HEADS_V, HEAD_DIM_V)
    q = apply_rotary_pos_emb_cached_cpu(q.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
    k = apply_rotary_pos_emb_cached_cpu(k.unsqueeze(0), cos[:,:seq], sin[:,:seq]).squeeze(0)
    attn = F.scaled_dot_product_attention(q.transpose(0,1), k.transpose(0,1), v.float().transpose(0,1), dropout_p=0.0)
    attn = attn.transpose(0,1).reshape(1,1,seq,DIM_V).to(torch.bfloat16)
    attn_tt = ttnn.from_torch(attn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    proj = ttnn.linear(attn_tt, w["proj_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(attn_tt)
    h = ttnn.add(x, proj, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(proj); ttnn.deallocate(x)
    norm2 = ttnn.rms_norm(h, epsilon=EPS_V, weight=w["norm2"])
    fc1 = ttnn.linear(norm2, w["fc1_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fc3 = ttnn.linear(norm2, w["fc3_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(norm2)
    gated = ttnn.mul(fc1, fc3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(fc1); ttnn.deallocate(fc3)
    fc2 = ttnn.linear(gated, w["fc2_w"], memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(gated)
    out = ttnn.add(h, fc2, memory_config=ttnn.DRAM_MEMORY_CONFIG); ttnn.deallocate(h); ttnn.deallocate(fc2)
    return out

def run_vision(inputs_page, page_idx):
    pv = inputs_page["pixel_values"].to(torch.bfloat16)
    gt = inputs_page["image_grid_thw"]
    with torch.no_grad():
        hidden_v = vision_model.patch_embed(pv, gt.to(torch.int32))
    seq_v = hidden_v.shape[0]
    pos_ids = vision_model.get_pos_ids_by_grid(gt.cpu())
    pos_ids = torch.cat(pos_ids, dim=0)
    rotary_full = vision_model.rotary_pos_emb(int(gt.cpu()[:, 1:].max().item())).cpu()
    rotary = rotary_full[pos_ids].flatten(1).float()
    cos_v = rotary.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
    sin_v = rotary.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)

    tt_h = ttnn.from_torch(hidden_v.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for bw in vblocks:
        tt_h = vision_block(tt_h, bw, device, seq_v, cos_v, sin_v)
    tt_h = ttnn.rms_norm(tt_h, epsilon=EPS_V, weight=post_norm_tt)
    vo = ttnn.to_torch(tt_h).squeeze(0).squeeze(0)[:seq_v].float()
    return cpu_merger(vo.to(torch.bfloat16))

# Run vision for all pages
vision_embs = []
vision_times = []
for i, inputs in enumerate(page_inputs):
    t0 = time.perf_counter()
    emb = run_vision(inputs, i)
    ms = (time.perf_counter() - t0) * 1000
    vision_embs.append(emb)
    vision_times.append(ms)
    print(f"  Page {i+1}: vision {ms:.0f}ms, emb={emb.shape}", flush=True)

ttnn.close_device(device)
del vs, vblocks, post_norm_tt

# ====================================================================
# PHASE 2: TTNN DECODER (all 4 pages)
# ====================================================================
print("\n=== PHASE 2: TTNN Decoder ===", flush=True)
from models.tt_transformers.tt.model_config import ModelArgs
from ocr_lab.ttnn_decoder_hybrid import (
    load_decoder_weights, prefill, decode_step_tt, precompute_attn_masks,
    init_tt_kv_cache, fill_tt_kv_cache,
    DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS,
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
sd = args.load_state_dict()
dec_layers, dec_final = load_decoder_weights(sd, mesh)

# Rotary tables
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

def process_page_decoder(inputs, vision_emb, page_idx):
    t_start = time.perf_counter()
    # Build embeds
    img_mask = inputs["input_ids"] == image_token_id
    embeds = F.embedding(inputs["input_ids"], emb_w)
    embeds = embeds.masked_scatter(img_mask.unsqueeze(-1).expand_as(embeds), vision_emb.to(embeds.dtype))
    seq = embeds.shape[1]

    # Prefill
    t0 = time.perf_counter()
    embeds_tt = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logits_tt, kv_k, kv_v = prefill(embeds_tt, dec_layers, dec_final, mesh, cos_cpu, sin_cpu, seq)
    prefill_ms = (time.perf_counter() - t0) * 1000
    logits_cpu = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :vocab_size].float()
    first_tok = logits_cpu.argmax(-1).item()

    # KV cache (tile-padded)
    max_cache = ((seq + MAX_NEW_TOKENS + 32 + 31) // 32) * 32
    tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)
    fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)
    del kv_k, kv_v
    decode_masks = precompute_attn_masks(
        start_pos=seq,
        num_steps=MAX_NEW_TOKENS - 1,
        max_cache_seq=max_cache,
        device=mesh,
    )

    # Decode
    generated = [first_tok]
    cur_pos = seq
    next_token = first_tok
    t0 = time.perf_counter()
    for step in range(1, MAX_NEW_TOKENS):
        token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
        tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_logits = decode_step_tt(tt_tok, dec_layers, dec_final, mesh,
                                    cos_full_tt, sin_full_tt, tt_cache, cur_pos,
                                    attn_mask_tt=decode_masks[step - 1])
        logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
        next_token = logits_d.argmax(-1).item()
        generated.append(next_token)
        cur_pos += 1
        if next_token in (151643, 151645):
            break
    decode_ms = (time.perf_counter() - t0) * 1000

    # Cleanup
    for k, v in tt_cache:
        ttnn.deallocate(k); ttnn.deallocate(v)
    for mask in decode_masks:
        ttnn.deallocate(mask)

    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    total_ms = (time.perf_counter() - t_start) * 1000
    n_tokens = len(generated)
    return gen_text, {
        "seq": seq, "tokens": n_tokens,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "ms_per_tok": decode_ms / max(n_tokens - 1, 1),
        "total_ms": total_ms,
    }

# Run decoder for all pages
print("Decoding all pages...", flush=True)
decoder_results = []
for i in range(len(PAGES)):
    print(f"  Page {i+1}...", flush=True)
    gen_text, timings = process_page_decoder(page_inputs[i], vision_embs[i], i)
    timings["vision_ms"] = vision_times[i]
    decoder_results.append((gen_text, timings))
    print(f"    Prefill: {timings['prefill_ms']:.0f}ms | Decode: {timings['decode_ms']:.0f}ms "
          f"({timings['tokens']} tok, {timings['ms_per_tok']:.0f}ms/tok)", flush=True)

# ====================================================================
# SUMMARY
# ====================================================================
total_vision = sum(vision_times)
total_all = total_vision + sum(t["total_ms"] for _, t in decoder_results)
total_tokens = sum(t["tokens"] for _, t in decoder_results)

print(f"\n{'='*60}")
print(f"  FULL PIPELINE — 4 pages (ttnn vision + ttnn decoder)")
print(f"{'='*60}")
for i, (text, t) in enumerate(decoder_results):
    page_total = t["vision_ms"] + t["total_ms"]
    print(f"  Page {i+1}: V={t['vision_ms']:.0f}ms P={t['prefill_ms']:.0f}ms "
          f"D={t['decode_ms']:.0f}ms ({t['tokens']}tok) = {page_total:.0f}ms")
    print(f"    {repr(text[:100])}")
print(f"  ─────────────────────────────")
print(f"  Vision total:   {total_vision:.0f}ms ({total_vision/1000:.1f}s)")
print(f"  Decoder total:  {sum(t['total_ms'] for _,t in decoder_results):.0f}ms")
print(f"  GRAND TOTAL:    {total_all:.0f}ms ({total_all/1000:.1f}s)")
print(f"  Avg per page:   {total_all/4:.0f}ms ({total_all/4000:.1f}s)")
print(f"  Pages/second:   {4/(total_all/1000):.3f}")
print(f"  Total tokens:   {total_tokens}")
print(f"{'='*60}")

ttnn.close_mesh_device(mesh)
