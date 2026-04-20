#!/usr/bin/env python3
"""3.1.pdf: batch=1 baseline for 3 pages, then batch=2 for pages 1+2."""
from __future__ import annotations
import json, os, sys, time, types
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.dots_model import DEFAULT_MODEL_PATH, ensure_dots_ocr_processor_compat
from ocr_lab.fixed_page import fixed_page_message

PAGES = [
    PROJECT_ROOT / "ocr_lab" / "tmp_3.1pdf_page-1.png",
    PROJECT_ROOT / "ocr_lab" / "tmp_3.1pdf_page-2.png",
    PROJECT_ROOT / "ocr_lab" / "tmp_3.1pdf_page-3.png",
]
W, H = 468, 662
MAX_TOK = 64
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"

shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
if shims_dir.exists():
    sys.path.insert(0, str(shims_dir))

from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.cache_utils import StaticCache
from qwen_vl_utils import process_vision_info
from dots_tt_port.vision_tower_smoke import (
    TowerHybridCpuUnary, ensure_tt_env, load_vision_model, resolve_snapshot_dir,
)
from ocr_lab.dots_tt_hybrid_decoder import build_inputs_embeds_cpu


def build_page_inputs(processor, image_path):
    img_msg, meta = fixed_page_message(image_path, width=W, height=H)
    msgs = [{"role": "user", "content": [img_msg, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    return inputs, meta


def generate_batch(compiled_model, dummy_ids, embeds_tt, cache, cache_pos, mask_tt, tt_device, eos_ids, max_tok, batch_size):
    import torch_xla
    tids = [[] for _ in range(batch_size)]
    timings = []

    with torch.no_grad():
        t0 = time.perf_counter()
        out = compiled_model(input_ids=dummy_ids, inputs_embeds=embeds_tt, past_key_values=cache,
                             cache_position=cache_pos, use_cache=True, attention_mask=mask_tt)
        logits = out.logits.to("cpu")
        torch_xla.sync(wait=True)
        timings.append(round((time.perf_counter() - t0) * 1000, 3))

        nxt = logits[:, -1, :].argmax(-1)
        for b in range(batch_size):
            tids[b].append(int(nxt[b].item()))

        pos = dummy_ids.shape[1]
        for step in range(1, max_tok):
            if all(tids[b][-1] in eos_ids for b in range(batch_size)):
                break
            t0 = time.perf_counter()
            inp = nxt.unsqueeze(1).to(tt_device)
            sp = torch.tensor([pos]).to(tt_device)
            out = compiled_model(input_ids=inp, past_key_values=cache, cache_position=sp,
                                 use_cache=True, attention_mask=mask_tt)
            logits = out.logits.to("cpu")
            torch_xla.sync(wait=True)
            timings.append(round((time.perf_counter() - t0) * 1000, 3))
            nxt = logits[:, -1, :].argmax(-1)
            for b in range(batch_size):
                tids[b].append(int(nxt[b].item()))
            pos += 1
    return tids, timings


def make_cache(config, batch, max_cache_len, kv_heads, head_dim, tt_device):
    c = StaticCache(config=config, max_batch_size=batch, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    c.early_initialization(batch_size=batch, num_heads=kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    for lc in c.layers:
        lc.keys = lc.keys.to(tt_device)
        lc.values = lc.values.to(tt_device)
    return c


def main():
    ensure_dots_ocr_processor_compat(DEFAULT_MODEL_PATH)
    snapshot_dir = resolve_snapshot_dir(DEFAULT_MODEL_PATH)
    model_path = str(snapshot_dir)

    ensure_tt_env("0")
    configure_tt_runtime(cache_dir=None, enable_trace=False, optimization_level=2)
    import torch_xla.runtime as xr
    xr.set_device_type("TT")
    import torch_xla
    tt_device = torch_xla.device()

    print("Loading models...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", use_cache=True,
    ).eval()

    tt_vis, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
    tt_vis = TowerHybridCpuUnary(tt_vis).eval().to(dtype=torch.bfloat16).to(tt_device)
    emb_w = model.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

    eos_id = model.config.eos_token_id
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id) if eos_id else set()

    kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Build inputs for all 3 pages
    print("Building inputs for 3 pages...", flush=True)
    all_inputs = []
    for p in PAGES:
        inp, meta = build_page_inputs(processor, p)
        all_inputs.append(inp)

    seq_len = all_inputs[0]["input_ids"].shape[1]
    max_cache_len = seq_len + MAX_TOK + 8

    # Vision warmup
    pv0 = all_inputs[0]["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
    gt0 = all_inputs[0]["image_grid_thw"].to(dtype=torch.int32).to(tt_device)
    with torch.no_grad():
        _ = tt_vis(pv0, gt0); torch_xla.sync(wait=True)

    # Vision for all 3 pages (warm)
    page_ve = []
    page_vis_ms = []
    for i, inp in enumerate(all_inputs):
        pv = inp["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
        gt = inp["image_grid_thw"].to(dtype=torch.int32).to(tt_device)
        with torch.no_grad():
            t0 = time.perf_counter()
            ve = tt_vis(pv, gt); torch_xla.sync(wait=True)
            ms = (time.perf_counter() - t0) * 1000
        page_ve.append(ve.detach().cpu().to(dtype=torch.bfloat16))
        page_vis_ms.append(round(ms, 1))
        print(f"  Page {i+1} vision: {ms:.0f}ms", flush=True)

    # Move decoder to TT
    model = model.to(tt_device)
    compiled = torch.compile(model, backend="tt")

    input_ids = all_inputs[0]["input_ids"]
    cache_pos = torch.arange(0, seq_len).to(tt_device)

    # ==========================================
    # BATCH=1: compile warmup
    # ==========================================
    print("\n=== BATCH=1 compile warmup ===", flush=True)
    e0 = build_inputs_embeds_cpu(model, input_ids, page_ve[0], embedding_weight_cpu=emb_w)
    e0_tt = e0.to(dtype=torch.bfloat16).to(tt_device)
    d1 = torch.zeros((1, seq_len), dtype=torch.long).to(tt_device)
    m1 = torch.ones((1, max_cache_len), dtype=torch.long).to(tt_device)
    c1 = make_cache(model.config, 1, max_cache_len, kv_heads, head_dim, tt_device)
    _ = generate_batch(compiled, d1, e0_tt, c1, cache_pos, m1, tt_device, eos_ids, MAX_TOK, 1)
    print("  Compile done.", flush=True)

    # BATCH=1: warm run for each page
    print("\n=== BATCH=1 warm runs ===", flush=True)
    b1_results = []
    for i in range(3):
        ei = build_inputs_embeds_cpu(model, input_ids, page_ve[i], embedding_weight_cpu=emb_w)
        ei_tt = ei.to(dtype=torch.bfloat16).to(tt_device)
        ci = make_cache(model.config, 1, max_cache_len, kv_heads, head_dim, tt_device)
        t0 = time.perf_counter()
        tids, timings = generate_batch(compiled, d1, ei_tt, ci, cache_pos, m1, tt_device, eos_ids, MAX_TOK, 1)
        dec_ms = (time.perf_counter() - t0) * 1000
        txt = tokenizer.decode(tids[0], skip_special_tokens=True)
        pf = timings[0]
        dk = sum(timings[1:])
        total = page_vis_ms[i] + dec_ms
        b1_results.append({"page": i+1, "vision_ms": page_vis_ms[i], "prefill_ms": round(pf,1),
                           "decode_ms": round(dk,1), "decoder_total_ms": round(dec_ms,1),
                           "total_ms": round(total,1), "text": txt})
        print(f"  Page {i+1}: vis={page_vis_ms[i]:.0f} pf={pf:.0f} dec={dk:.0f} total={total:.0f}ms", flush=True)
        print(f"    {repr(txt[:100])}", flush=True)

    # ==========================================
    # BATCH=2: pages 1+2 together
    # ==========================================
    batch2_result = None
    batch2_error = None
    try:
        print("\n=== BATCH=2 compile (pages 1+2) ===", flush=True)
        e12 = torch.cat([
            build_inputs_embeds_cpu(model, input_ids, page_ve[0], embedding_weight_cpu=emb_w),
            build_inputs_embeds_cpu(model, input_ids, page_ve[1], embedding_weight_cpu=emb_w),
        ], dim=0)
        e12_tt = e12.to(dtype=torch.bfloat16).to(tt_device)
        d2 = torch.zeros((2, seq_len), dtype=torch.long).to(tt_device)
        m2 = torch.ones((2, max_cache_len), dtype=torch.long).to(tt_device)
        c2_cold = make_cache(model.config, 2, max_cache_len, kv_heads, head_dim, tt_device)
        _ = generate_batch(compiled, d2, e12_tt, c2_cold, cache_pos, m2, tt_device, eos_ids, MAX_TOK, 2)
        print("  Compile done.", flush=True)

        print("\n=== BATCH=2 warm (pages 1+2) ===", flush=True)
        c2w = make_cache(model.config, 2, max_cache_len, kv_heads, head_dim, tt_device)
        t0 = time.perf_counter()
        tids2, timings2 = generate_batch(compiled, d2, e12_tt, c2w, cache_pos, m2, tt_device, eos_ids, MAX_TOK, 2)
        b2_dec_ms = (time.perf_counter() - t0) * 1000
        txt2_0 = tokenizer.decode(tids2[0], skip_special_tokens=True)
        txt2_1 = tokenizer.decode(tids2[1], skip_special_tokens=True)
        b2_vis = page_vis_ms[0] + page_vis_ms[1]
        b2_total = b2_vis + b2_dec_ms
        print(f"  Vision (p1+p2): {b2_vis:.0f}ms  Decoder: {b2_dec_ms:.0f}ms  Total: {b2_total:.0f}ms", flush=True)
        print(f"  Per page: {b2_total/2:.0f}ms", flush=True)
        print(f"  P1 match: {txt2_0.strip() == b1_results[0]['text'].strip()}", flush=True)
        print(f"  P2 match: {txt2_1.strip() == b1_results[1]['text'].strip()}", flush=True)
        print(f"  P1: {repr(txt2_0[:100])}", flush=True)
        print(f"  P2: {repr(txt2_1[:100])}", flush=True)
        batch2_result = {
            "vision_ms": round(b2_vis, 1),
            "decoder_ms": round(b2_dec_ms, 1),
            "total_ms": round(b2_total, 1),
            "per_page_ms": round(b2_total / 2, 1),
            "texts": [txt2_0, txt2_1],
            "match": [
                txt2_0.strip() == b1_results[0]["text"].strip(),
                txt2_1.strip() == b1_results[1]["text"].strip(),
            ],
            "timings": timings2,
        }
    except Exception as exc:
        batch2_error = f"{type(exc).__name__}: {exc}"
        print(f"\n=== BATCH=2 failed ===\n  {batch2_error}", flush=True)

    # ==========================================
    # SUMMARY
    # ==========================================
    b1_avg = sum(r["total_ms"] for r in b1_results) / 3
    b1_sum = sum(r["total_ms"] for r in b1_results)

    print(f"\n{'='*60}")
    print(f"=== SUMMARY: 3.1.pdf 3 pages ===")
    print(f"{'='*60}")
    print(f"Batch=1 sequential: {b1_sum:.0f}ms total, {b1_avg:.0f}ms/page")
    if batch2_result is not None:
        pipeline_total = batch2_result["total_ms"] + b1_results[2]["total_ms"]
        pipeline_per_page = pipeline_total / 3
        print(f"Batch=2(p1+p2)+Batch=1(p3): {pipeline_total:.0f}ms total, {pipeline_per_page:.0f}ms/page")
        print(f"Throughput improvement: {b1_avg/pipeline_per_page:.2f}x")
    else:
        pipeline_total = None
        pipeline_per_page = None
        print("Batch=2(p1+p2)+Batch=1(p3): failed")

    results = {
        "batch1": b1_results,
        "model_path": model_path,
        "batch2": batch2_result,
        "batch2_error": batch2_error,
        "summary": {
            "b1_total_ms": round(b1_sum, 1),
            "b1_per_page_ms": round(b1_avg, 1),
            "pipeline_total_ms": round(pipeline_total, 1) if pipeline_total is not None else None,
            "pipeline_per_page_ms": round(pipeline_per_page, 1) if pipeline_per_page is not None else None,
            "throughput_x": round(b1_avg / pipeline_per_page, 3) if pipeline_per_page is not None else None,
        },
    }
    Path(PROJECT_ROOT / "ocr_lab" / "batch_3_1pdf_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
