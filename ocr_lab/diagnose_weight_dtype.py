#!/usr/bin/env python3
"""Diagnose: does bfloat8_b weight dtype cause the accuracy loss?

Compare prefill output (first generated token + text) between:
- bfloat8_b weights (current default, faster)
- bfloat16 weights (full precision, slower)

If bfloat16 fixes the text accuracy, the root cause is weight quantization.
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
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "80"))

# --- Build inputs ---
proc = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
tokenizer = getattr(proc, "tokenizer", proc)
img_msg, _ = fixed_page_message(IMAGE, width=476, height=674)
msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

# --- CPU vision ---
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
print(f"  seq={seq}", flush=True)

# --- TT setup ---
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
mesh.enable_program_cache()
args = ModelArgs(mesh, instruct=False, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
sd = args.load_state_dict()

cfg = json.loads((Path(str(snapshot_dir)) / "config.json").read_text())
rope_theta = cfg.get("rope_theta", 1e6)
vocab_size = cfg.get('vocab_size', 151936)
max_pos = seq + MAX_NEW_TOKENS + 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
emb_rope = torch.cat((freqs, freqs), dim=-1)
cos_full = emb_rope.cos().to(torch.bfloat16)
sin_full = emb_rope.sin().to(torch.bfloat16)
cos_cpu = cos_full.float().unsqueeze(1).unsqueeze(0)
sin_cpu = sin_full.float().unsqueeze(1).unsqueeze(0)

# TT cos/sin for decode
cos_full_tt = ttnn.from_torch(cos_full.unsqueeze(0).unsqueeze(0),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
sin_full_tt = ttnn.from_torch(sin_full.unsqueeze(0).unsqueeze(0),
    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)


def run_with_weight_dtype(weight_dtype, dtype_name):
    """Load weights with given dtype, run prefill + decode, return text."""
    from ocr_lab.ttnn_decoder_hybrid import (
        DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS, EPS,
        apply_rotary, GQA_GROUPS,
        init_tt_kv_cache, fill_tt_kv_cache,
        precompute_attn_masks, decode_step_tt,
    )

    print(f"\n{'='*60}")
    print(f"  Loading weights as {dtype_name}...")
    print(f"{'='*60}")

    def make_weight_custom(t, dev, dtype=weight_dtype):
        return ttnn.from_torch(t.T.contiguous().unsqueeze(0).unsqueeze(0),
                               dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    layers = []
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        w = {}
        w["attn_norm"] = ttnn.from_torch(
            sd[f"{p}attention_norm.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["ffn_norm"] = ttnn.from_torch(
            sd[f"{p}ffn_norm.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w["wq"] = make_weight_custom(sd[f"{p}attention.wq.weight"], mesh)
        w["wk"] = make_weight_custom(sd[f"{p}attention.wk.weight"], mesh)
        w["wv"] = make_weight_custom(sd[f"{p}attention.wv.weight"], mesh)
        w["wo"] = make_weight_custom(sd[f"{p}attention.wo.weight"], mesh)
        w["qb"] = sd.get(f"{p}attention.wq.bias")
        w["kb"] = sd.get(f"{p}attention.wk.bias")
        w["vb"] = sd.get(f"{p}attention.wv.bias")
        # TT bias
        qb = sd.get(f"{p}attention.wq.bias")
        kb = sd.get(f"{p}attention.wk.bias")
        vb = sd.get(f"{p}attention.wv.bias")
        for name, bias in [("qb_tt", qb), ("kb_tt", kb), ("vb_tt", vb)]:
            if bias is not None:
                w[name] = ttnn.from_torch(bias.reshape(1,1,1,-1).to(torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                w[name] = None
        w["w1"] = make_weight_custom(sd[f"{p}feed_forward.w1.weight"], mesh)
        w["w2"] = make_weight_custom(sd[f"{p}feed_forward.w2.weight"], mesh)
        w["w3"] = make_weight_custom(sd[f"{p}feed_forward.w3.weight"], mesh)
        # Fused QKV
        wq_raw = sd[f"{p}attention.wq.weight"]
        wk_raw = sd[f"{p}attention.wk.weight"]
        wv_raw = sd[f"{p}attention.wv.weight"]
        w["wqkv"] = make_weight_custom(torch.cat([wq_raw, wk_raw, wv_raw], dim=0), mesh)
        if qb is not None and kb is not None and vb is not None:
            qkvb_fused = torch.cat([qb, kb, vb], dim=0)
            w["qkvb_fused_tt"] = ttnn.from_torch(
                qkvb_fused.reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            w["qkvb_fused_tt"] = None
        layers.append(w)

    final = {}
    final["norm"] = ttnn.from_torch(
        sd["norm.weight"].unsqueeze(0).view(1, 1, DIM // 32, 32),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    final["lm_head"] = make_weight_custom(sd["output.weight"], mesh) if "output.weight" in sd else make_weight_custom(sd["tok_embeddings.weight"], mesh)

    # Prefill
    from ocr_lab.ttnn_decoder_hybrid import prefill
    print(f"  Prefill ({seq} tokens)...", flush=True)
    embeds_tt = ttnn.from_torch(embeds.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t0 = time.perf_counter()
    logits_tt, kv_k, kv_v = prefill(embeds_tt, layers, final, mesh, cos_cpu, sin_cpu, seq)
    prefill_ms = (time.perf_counter() - t0) * 1000
    logits_cpu_arr = ttnn.to_torch(logits_tt)[0, 0, seq - 1, :vocab_size].float()
    first_tok = logits_cpu_arr.argmax(-1).item()
    print(f"  Prefill: {prefill_ms:.0f}ms, first token: {first_tok} = {repr(tokenizer.decode([first_tok]))}")

    # Decode
    max_cache = seq + MAX_NEW_TOKENS + 32
    max_cache = ((max_cache + 31) // 32) * 32
    tt_cache = init_tt_kv_cache(mesh, max_seq=max_cache)
    fill_tt_kv_cache(tt_cache, kv_k, kv_v, mesh)
    del kv_k, kv_v

    decode_masks = precompute_attn_masks(start_pos=seq, num_steps=MAX_NEW_TOKENS,
                                          max_cache_seq=max_cache, device=mesh)

    generated = [first_tok]
    cur_pos = seq
    next_token = first_tok
    for step in range(1, MAX_NEW_TOKENS):
        token_embed = emb_w[next_token].reshape(1, 1, 1, DIM)
        tt_tok = ttnn.from_torch(token_embed, device=mesh, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_logits = decode_step_tt(tt_tok, layers, final, mesh,
                                    cos_full_tt, sin_full_tt, tt_cache, cur_pos,
                                    attn_mask_tt=decode_masks[step - 1])
        logits_d = ttnn.to_torch(tt_logits)[0, 0, 0, :vocab_size].float()
        next_token = logits_d.argmax(-1).item()
        generated.append(next_token)
        cur_pos += 1

    for mask in decode_masks:
        ttnn.deallocate(mask)

    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  Generated {len(generated)} tokens")
    print(f"  Text: {repr(gen_text[:300])}")
    return gen_text


# --- Run both ---
print("\n" + "="*60)
print("TEST A: bfloat8_b weights (current default)")
print("="*60)
text_bf8 = run_with_weight_dtype(ttnn.bfloat8_b, "bfloat8_b")

print("\n" + "="*60)
print("TEST B: bfloat16 weights (full precision)")
print("="*60)
text_bf16 = run_with_weight_dtype(ttnn.bfloat16, "bfloat16")

# --- Compare ---
REF_CPU = "T.C.\nBAKIRKÖY\n2. AİLE MAHKEMESİ\n\nEsas No :\n\nKarar No :\n\n- KESİNLEŞME ŞERHİ -\n\nMahkememizden verilen işbu 28/12/2017 tarihli hükümler, Davacı 'e ve davalı 'e 18/01/2018 tarihinde tebliğ olunmuş, tarafların 18/01/2018 tarihinde vermiş olduğu \"İstinaftan Feragat Dilekçesi\" ile hükmün, 18/01/2018 tarihinde kesinleştiği tasdik olunur. 18/01/2018"

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"\nCPU reference:\n  {repr(REF_CPU[:200])}")
print(f"\nbfloat8_b:\n  {repr(text_bf8[:200])}")
print(f"\nbfloat16:\n  {repr(text_bf16[:200])}")

# Character match analysis
def char_match(ref, test):
    matches = sum(1 for a, b in zip(ref, test) if a == b)
    return matches, max(len(ref), 1)

m8, t8 = char_match(REF_CPU, text_bf8)
m16, t16 = char_match(REF_CPU, text_bf16)
print(f"\nbfloat8_b  vs CPU ref: {m8}/{t8} chars match ({m8/t8*100:.1f}%)")
print(f"bfloat16   vs CPU ref: {m16}/{t16} chars match ({m16/t16*100:.1f}%)")

if m16 > m8:
    print("\n*** bfloat16 is more accurate — weight quantization is the root cause ***")
elif m16 == m8:
    print("\n*** Same accuracy — weight dtype is NOT the issue ***")
else:
    print("\n*** bfloat8_b is more accurate (unexpected) ***")

# Save
results = {
    "ref_cpu": REF_CPU,
    "text_bf8": text_bf8,
    "text_bf16": text_bf16,
    "bf8_match_pct": m8/t8*100,
    "bf16_match_pct": m16/t16*100,
}
out = PROJECT_ROOT / "ocr_lab" / "diagnose_weight_dtype_results.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nSaved: {out}")

ttnn.close_mesh_device(mesh)
