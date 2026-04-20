#!/usr/bin/env python3
"""
Test batch=2 decoder on single TT chip.
Same image duplicated to measure throughput gain.
"""
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
from ocr_lab.fixed_page import fixed_page_message

IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"
MAX_NEW_TOKENS = 64

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


def build_inputs(model_path, image_path, prompt, w, h):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    img_msg, meta = fixed_page_message(image_path, width=w, height=h)
    messages = [{"role": "user", "content": [img_msg, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    return processor, inputs, meta


def run_generate(compiled_model, input_ids_tt, inputs_embeds_tt, static_cache, cache_position_tt, attn_mask_tt, tt_device, eos_ids, max_tokens, batch_size):
    """Run greedy generate for batch, return per-token timings."""
    import torch_xla
    token_ids = [[] for _ in range(batch_size)]
    timings = []
    seq_len = input_ids_tt.shape[1]

    with torch.no_grad():
        # Prefill
        t0 = time.perf_counter()
        output = compiled_model(
            input_ids=input_ids_tt,
            inputs_embeds=inputs_embeds_tt,
            past_key_values=static_cache,
            cache_position=cache_position_tt,
            use_cache=True,
            attention_mask=attn_mask_tt,
        )
        logits = output.logits.to("cpu")
        torch_xla.sync(wait=True)
        t1 = time.perf_counter()
        timings.append(round((t1 - t0) * 1000, 3))

        next_ids = logits[:, -1, :].argmax(-1)  # [batch]
        for b in range(batch_size):
            token_ids[b].append(int(next_ids[b].item()))

        # Decode
        cur_pos = seq_len
        for step in range(1, max_tokens):
            # Check EOS for all
            all_eos = all(token_ids[b][-1] in eos_ids for b in range(batch_size))
            if all_eos:
                break

            t0 = time.perf_counter()
            next_input = next_ids.unsqueeze(1).to(tt_device)  # [batch, 1]
            step_pos = torch.tensor([cur_pos]).to(tt_device)

            output = compiled_model(
                input_ids=next_input,
                past_key_values=static_cache,
                cache_position=step_pos,
                use_cache=True,
                attention_mask=attn_mask_tt,
            )
            logits = output.logits.to("cpu")
            torch_xla.sync(wait=True)
            t1 = time.perf_counter()
            timings.append(round((t1 - t0) * 1000, 3))

            next_ids = logits[:, -1, :].argmax(-1)
            for b in range(batch_size):
                token_ids[b].append(int(next_ids[b].item()))
            cur_pos += 1

    return token_ids, timings


def main():
    snapshot_dir = resolve_snapshot_dir("rednote-hilab/dots.mocr")
    model_path = str(snapshot_dir)
    prompt = "Please output the exact text in the image.\n\nReturn plain text only.\n"

    ensure_tt_env("0")
    configure_tt_runtime(cache_dir=None, enable_trace=False, optimization_level=2)
    import torch_xla.runtime as xr
    xr.set_device_type("TT")
    import torch_xla
    tt_device = torch_xla.device()

    # Build inputs (single image)
    print("Building inputs...", flush=True)
    processor, inputs1, _ = build_inputs(model_path, IMAGE, prompt, 476, 674)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Load models
    print("Loading models...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", use_cache=True,
    ).eval()

    tt_vision, _ = load_vision_model(snapshot_dir, "sdpa", limit_layers=0)
    tt_vision = TowerHybridCpuUnary(tt_vision).eval().to(dtype=torch.bfloat16).to(tt_device)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

    pv_tt = inputs1["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
    gt_tt = inputs1["image_grid_thw"].to(dtype=torch.int32).to(tt_device)

    eos_id = model.config.eos_token_id
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id) if eos_id else set()

    input_ids_1 = inputs1["input_ids"]       # [1, seq]
    attn_mask_1 = inputs1["attention_mask"]   # [1, seq]
    seq_len = input_ids_1.shape[1]
    max_cache_len = seq_len + MAX_NEW_TOKENS + 8

    # ==========================================
    # BATCH=1 baseline
    # ==========================================
    print("\n=== BATCH=1 ===", flush=True)

    # Vision
    with torch.no_grad():
        _ = tt_vision(pv_tt, gt_tt); torch_xla.sync(wait=True)
        t0 = time.perf_counter()
        ve1 = tt_vision(pv_tt, gt_tt); torch_xla.sync(wait=True)
        v1_ms = (time.perf_counter() - t0) * 1000
        ve1_cpu = ve1.detach().cpu().to(dtype=torch.bfloat16)

    embeds1 = build_inputs_embeds_cpu(model, input_ids_1, ve1_cpu, embedding_weight_cpu=emb_weight)

    # Move model to TT
    model = model.to(tt_device)

    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    cache1 = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    cache1.early_initialization(batch_size=1, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    for lc in cache1.layers:
        lc.keys = lc.keys.to(tt_device)
        lc.values = lc.values.to(tt_device)

    embeds1_tt = embeds1.to(dtype=torch.bfloat16).to(tt_device)
    dummy1 = torch.zeros((1, seq_len), dtype=torch.long).to(tt_device)
    cpos1 = torch.arange(0, seq_len).to(tt_device)
    full_mask1 = torch.ones((1, max_cache_len), dtype=torch.long).to(tt_device)

    compiled_model = torch.compile(model, backend="tt")

    # Cold run (compile)
    print("  Cold compile...", flush=True)
    tids1_cold, timings1_cold = run_generate(compiled_model, dummy1, embeds1_tt, cache1, cpos1, full_mask1, tt_device, eos_ids, MAX_NEW_TOKENS, 1)

    # Warm run
    cache1w = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    cache1w.early_initialization(batch_size=1, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    for lc in cache1w.layers:
        lc.keys = lc.keys.to(tt_device)
        lc.values = lc.values.to(tt_device)

    print("  Warm run...", flush=True)
    t0_w = time.perf_counter()
    tids1_warm, timings1_warm = run_generate(compiled_model, dummy1, embeds1_tt, cache1w, cpos1, full_mask1, tt_device, eos_ids, MAX_NEW_TOKENS, 1)
    b1_total = (time.perf_counter() - t0_w) * 1000

    text1 = tokenizer.decode(tids1_warm[0], skip_special_tokens=True)
    b1_prefill = timings1_warm[0]
    b1_decode = sum(timings1_warm[1:])
    print(f"  Vision: {v1_ms:.0f}ms  Prefill: {b1_prefill:.0f}ms  Decode: {b1_decode:.0f}ms  Total: {v1_ms+b1_total:.0f}ms", flush=True)
    print(f"  Text: {repr(text1[:120])}", flush=True)

    # ==========================================
    # BATCH=2
    # ==========================================
    print("\n=== BATCH=2 (same image duplicated) ===", flush=True)

    # Duplicate inputs for batch=2
    input_ids_2 = input_ids_1.repeat(2, 1)     # [2, seq]
    embeds2_cpu = build_inputs_embeds_cpu(model, input_ids_1, ve1_cpu, embedding_weight_cpu=emb_weight)
    embeds2_cpu = embeds2_cpu.repeat(2, 1, 1)   # [2, seq, dim]

    embeds2_tt = embeds2_cpu.to(dtype=torch.bfloat16).to(tt_device)
    dummy2 = torch.zeros((2, seq_len), dtype=torch.long).to(tt_device)
    full_mask2 = torch.ones((2, max_cache_len), dtype=torch.long).to(tt_device)

    cache2 = StaticCache(config=model.config, max_batch_size=2, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    cache2.early_initialization(batch_size=2, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    for lc in cache2.layers:
        lc.keys = lc.keys.to(tt_device)
        lc.values = lc.values.to(tt_device)

    # Cold (will recompile for new batch size)
    print("  Cold compile batch=2...", flush=True)
    tids2_cold, timings2_cold = run_generate(compiled_model, dummy2, embeds2_tt, cache2, cpos1, full_mask2, tt_device, eos_ids, MAX_NEW_TOKENS, 2)

    # Warm
    cache2w = StaticCache(config=model.config, max_batch_size=2, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    cache2w.early_initialization(batch_size=2, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    for lc in cache2w.layers:
        lc.keys = lc.keys.to(tt_device)
        lc.values = lc.values.to(tt_device)

    print("  Warm run batch=2...", flush=True)
    t0_w2 = time.perf_counter()
    tids2_warm, timings2_warm = run_generate(compiled_model, dummy2, embeds2_tt, cache2w, cpos1, full_mask2, tt_device, eos_ids, MAX_NEW_TOKENS, 2)
    b2_total = (time.perf_counter() - t0_w2) * 1000

    text2_0 = tokenizer.decode(tids2_warm[0], skip_special_tokens=True)
    text2_1 = tokenizer.decode(tids2_warm[1], skip_special_tokens=True)
    b2_prefill = timings2_warm[0]
    b2_decode = sum(timings2_warm[1:])

    print(f"  Prefill: {b2_prefill:.0f}ms  Decode: {b2_decode:.0f}ms  Total decoder: {b2_total:.0f}ms", flush=True)
    print(f"  Per-page: {b2_total/2:.0f}ms decoder", flush=True)
    print(f"  Text[0]: {repr(text2_0[:120])}", flush=True)
    print(f"  Text[1]: {repr(text2_1[:120])}", flush=True)
    print(f"  Match: {text2_0.strip() == text1.strip() and text2_1.strip() == text1.strip()}", flush=True)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Batch=1: vision={v1_ms:.0f} + decoder={b1_total:.0f} = {v1_ms+b1_total:.0f}ms per page")
    print(f"Batch=2: vision={v1_ms:.0f} + decoder={b2_total:.0f} = {v1_ms+b2_total:.0f}ms for 2 pages")
    print(f"Batch=2 per page: {(v1_ms+b2_total)/2:.0f}ms")
    print(f"Speedup: {(v1_ms+b1_total)/((v1_ms+b2_total)/2):.2f}x throughput")

    results = {
        "batch1": {"vision_ms": round(v1_ms, 1), "decoder_ms": round(b1_total, 1), "text": text1, "token_timings": timings1_warm},
        "batch2": {"vision_ms": round(v1_ms, 1), "decoder_ms": round(b2_total, 1), "text_0": text2_0, "text_1": text2_1, "match": text2_0.strip() == text1.strip(), "token_timings": timings2_warm},
    }
    Path(PROJECT_ROOT / "ocr_lab" / "batch2_test_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
