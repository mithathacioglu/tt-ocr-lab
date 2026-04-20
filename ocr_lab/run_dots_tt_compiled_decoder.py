#!/usr/bin/env python3
"""
DOTS mOCR with TT-compiled vision + decoder.

Vision: torch.compile(backend="tt") on original DotsVisionTransformer
Decoder: torch.compile(backend="tt") with StaticCache

Based on tt-xla/examples/pytorch/llama.py pattern.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import DEFAULT_A4_HEIGHT, DEFAULT_A4_WIDTH, fixed_page_message

DEFAULT_IMAGE = PROJECT_ROOT / "ocr_lab" / "tmp_1pdf_page-1.png"

DEFAULT_PROMPT = """Please output the exact text in the image.

Return plain text only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--model-path", default="rednote-hilab/dots.mocr")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fixed-page-width", type=int, default=476)
    parser.add_argument("--fixed-page-height", type=int, default=674)
    parser.add_argument("--optimization-level", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--bfp8", action="store_true", help="Use bfp8 weight quantization")
    parser.add_argument(
        "--json-out",
        default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_compiled_decoder_latest.json"),
    )
    return parser.parse_args()


def build_inputs(model_path: str, image_path: Path, prompt: str, fixed_page_width: int, fixed_page_height: int):
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    image_message, fixed_page_meta = fixed_page_message(
        image_path,
        width=fixed_page_width,
        height=fixed_page_height,
    )
    messages = [
        {
            "role": "user",
            "content": [
                image_message,
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return processor, inputs, fixed_page_meta


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import StaticCache
    import torch_xla.runtime as xr

    from dots_tt_port.vision_tower_smoke import (
        TowerHybridCpuUnary,
        ensure_tt_env,
        load_vision_model,
        resolve_snapshot_dir,
    )
    from ocr_lab.vision_optimized import TowerHybridOptimized
    from ocr_lab.dots_tt_hybrid_decoder import build_inputs_embeds_cpu

    # --- Setup TT ---
    snapshot_dir = resolve_snapshot_dir(args.model_path)
    model_path = str(snapshot_dir)

    ensure_tt_env(args.device_index)
    from ocr_lab.tt_perf import default_cache_dir
    cache_dir = default_cache_dir("compiled_decoder", args.device_index)
    perf_runtime = configure_tt_runtime(
        cache_dir=str(cache_dir),
        optimization_level=args.optimization_level,
        enable_trace=False,
    )
    xr.set_device_type("TT")
    import torch_xla

    tt_device = torch_xla.device()

    # --- Build inputs ---
    print("Building inputs...", flush=True)
    processor, inputs, fixed_page_meta = build_inputs(
        model_path, image_path, args.prompt,
        args.fixed_page_width, args.fixed_page_height,
    )

    # --- Load models ---
    print("Loading language model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=True,
    ).eval()

    print("Loading TT vision model...", flush=True)
    tt_vision_model, _ = load_vision_model(snapshot_dir, args.attn, limit_layers=0)
    tt_vision_model = TowerHybridCpuUnary(tt_vision_model).eval().to(dtype=torch.bfloat16)
    embedding_weight_cpu = model.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_len = input_ids.shape[1]

    # --- TT Hybrid Vision forward ---
    print("Running TT vision...", flush=True)
    tt_vision_model = tt_vision_model.to(tt_device)
    pixel_values_tt = pixel_values.to(dtype=torch.bfloat16).to(tt_device)
    image_grid_thw_tt = image_grid_thw.to(dtype=torch.int32).to(tt_device)

    with torch.no_grad():
        vision_t0 = time.perf_counter()
        vision_embeddings_tt = tt_vision_model(pixel_values_tt, image_grid_thw_tt)
        torch_xla.sync(wait=True)
        vision_t1 = time.perf_counter()
        vision_embeddings = vision_embeddings_tt.detach().cpu().to(dtype=torch.bfloat16)

    vision_cold_ms = (vision_t1 - vision_t0) * 1000.0
    print(f"  Vision (cold): {vision_cold_ms:.0f}ms", flush=True)

    # --- Build inputs_embeds on CPU ---
    inputs_embeds = build_inputs_embeds_cpu(
        model, input_ids, vision_embeddings,
        embedding_weight_cpu=embedding_weight_cpu,
    )

    # --- Setup StaticCache ---
    max_cache_len = seq_len + args.max_new_tokens + 8
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    static_cache.early_initialization(
        batch_size=1,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    # --- Transfer decoder model + inputs to TT ---
    print("Transferring decoder to TT device...", flush=True)
    transfer_t0 = time.perf_counter()
    model = model.to(tt_device)

    # Transfer cache to TT
    for layer_cache in static_cache.layers:
        layer_cache.keys = layer_cache.keys.to(tt_device)
        layer_cache.values = layer_cache.values.to(tt_device)

    inputs_embeds_tt = inputs_embeds.to(dtype=torch.bfloat16).to(tt_device)
    attention_mask_tt = attention_mask.to(tt_device)
    cache_position = torch.arange(0, seq_len).to(tt_device)

    # Expand attention mask to max_cache_len
    full_attention_mask = torch.ones((1, max_cache_len), dtype=attention_mask.dtype)
    full_attention_mask[:, :seq_len] = attention_mask
    full_attention_mask_tt = full_attention_mask.to(tt_device)

    torch_xla.sync(wait=True)
    transfer_ms = (time.perf_counter() - transfer_t0) * 1000.0
    print(f"  Transfer to TT: {transfer_ms:.0f}ms", flush=True)

    # --- Compile decoder ---
    compile_options = {}
    if args.bfp8:
        compile_options["experimental_weight_dtype"] = "bfp8"
        print("Compiling decoder with torch.compile(backend='tt', bfp8=True)...", flush=True)
    else:
        print("Compiling decoder with torch.compile(backend='tt')...", flush=True)
    compiled_model = torch.compile(model, backend="tt", options=compile_options if compile_options else None)

    # --- Generate tokens ---
    print(f"Generating {args.max_new_tokens} tokens...", flush=True)
    generated_token_ids = []
    token_timings_ms = []
    eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    elif eos_token_id is not None:
        eos_ids = set(eos_token_id)
    else:
        eos_ids = set()

    # We need a dummy input_ids for the model (it expects it even with inputs_embeds)
    dummy_input_ids = torch.zeros((1, seq_len), dtype=torch.long).to(tt_device)

    with torch.no_grad():
        # --- Prefill ---
        prefill_t0 = time.perf_counter()
        output = compiled_model(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds_tt,
            past_key_values=static_cache,
            cache_position=cache_position,
            use_cache=True,
            attention_mask=full_attention_mask_tt,
        )
        logits_cpu = output.logits.to("cpu")
        torch_xla.sync(wait=True)
        prefill_t1 = time.perf_counter()

        next_token_id = int(logits_cpu[:, -1, :].argmax(-1).item())
        generated_token_ids.append(next_token_id)
        token_timings_ms.append(round((prefill_t1 - prefill_t0) * 1000.0, 3))
        print(f"  Prefill: {token_timings_ms[0]:.0f}ms (token: {next_token_id})", flush=True)

        # --- Decode loop ---
        cur_pos = seq_len
        for step in range(1, args.max_new_tokens):
            if next_token_id in eos_ids:
                break

            step_t0 = time.perf_counter()
            next_input_ids = torch.tensor([[next_token_id]], dtype=torch.long).to(tt_device)
            step_cache_position = torch.tensor([cur_pos]).to(tt_device)

            output = compiled_model(
                input_ids=next_input_ids,
                past_key_values=static_cache,
                cache_position=step_cache_position,
                use_cache=True,
                attention_mask=full_attention_mask_tt,
            )
            logits_cpu = output.logits.to("cpu")
            torch_xla.sync(wait=True)
            step_t1 = time.perf_counter()

            next_token_id = int(logits_cpu[:, -1, :].argmax(-1).item())
            generated_token_ids.append(next_token_id)
            elapsed = round((step_t1 - step_t0) * 1000.0, 3)
            token_timings_ms.append(elapsed)
            cur_pos += 1

            if step <= 3 or step % 10 == 0:
                print(f"  Step {step}: {elapsed:.0f}ms (token: {next_token_id})", flush=True)

    # --- Decode text ---
    tokenizer = getattr(processor, "tokenizer", processor)
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # =========================================================
    # WARM RUN 2: Same process, dynamo cache should be hot
    # =========================================================
    print("\n=== WARM RUN 2 (in-process, dynamo cached) ===", flush=True)

    # Warm vision
    with torch.no_grad():
        warm_v_t0 = time.perf_counter()
        vision_embeddings_tt2 = tt_vision_model(pixel_values_tt, image_grid_thw_tt)
        torch_xla.sync(wait=True)
        warm_v_t1 = time.perf_counter()
        vision_embeddings2 = vision_embeddings_tt2.detach().cpu().to(dtype=torch.bfloat16)
    warm_vision_ms = (warm_v_t1 - warm_v_t0) * 1000.0
    print(f"  Vision (warm): {warm_vision_ms:.0f}ms", flush=True)

    # Rebuild inputs_embeds and cache for run 2
    inputs_embeds2 = build_inputs_embeds_cpu(
        model, input_ids, vision_embeddings2,
        embedding_weight_cpu=embedding_weight_cpu,
    )
    static_cache2 = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    static_cache2.early_initialization(
        batch_size=1, num_heads=num_kv_heads, head_dim=head_dim,
        dtype=torch.bfloat16, device="cpu",
    )
    for lc in static_cache2.layers:
        lc.keys = lc.keys.to(tt_device)
        lc.values = lc.values.to(tt_device)

    inputs_embeds_tt2 = inputs_embeds2.to(dtype=torch.bfloat16).to(tt_device)
    cache_position2 = torch.arange(0, seq_len).to(tt_device)
    dummy_input_ids2 = torch.zeros((1, seq_len), dtype=torch.long).to(tt_device)

    warm_token_ids = []
    warm_timings = []
    with torch.no_grad():
        # Warm prefill
        wp_t0 = time.perf_counter()
        output2 = compiled_model(
            input_ids=dummy_input_ids2,
            inputs_embeds=inputs_embeds_tt2,
            past_key_values=static_cache2,
            cache_position=cache_position2,
            use_cache=True,
            attention_mask=full_attention_mask_tt,
        )
        logits2 = output2.logits.to("cpu")
        torch_xla.sync(wait=True)
        wp_t1 = time.perf_counter()
        next_id = int(logits2[:, -1, :].argmax(-1).item())
        warm_token_ids.append(next_id)
        warm_timings.append(round((wp_t1 - wp_t0) * 1000.0, 3))
        print(f"  Prefill (warm): {warm_timings[0]:.0f}ms", flush=True)

        # Warm decode
        cur_pos2 = seq_len
        for step in range(1, args.max_new_tokens):
            if next_id in eos_ids:
                break
            st0 = time.perf_counter()
            nids = torch.tensor([[next_id]], dtype=torch.long).to(tt_device)
            scp = torch.tensor([cur_pos2]).to(tt_device)
            output2 = compiled_model(
                input_ids=nids,
                past_key_values=static_cache2,
                cache_position=scp,
                use_cache=True,
                attention_mask=full_attention_mask_tt,
            )
            logits2 = output2.logits.to("cpu")
            torch_xla.sync(wait=True)
            st1 = time.perf_counter()
            next_id = int(logits2[:, -1, :].argmax(-1).item())
            warm_token_ids.append(next_id)
            elapsed_w = round((st1 - st0) * 1000.0, 3)
            warm_timings.append(elapsed_w)
            cur_pos2 += 1
            if step <= 3 or step % 10 == 0:
                print(f"  Step {step}: {elapsed_w:.0f}ms", flush=True)

    warm_text = tokenizer.decode(warm_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    warm_decode_total = sum(warm_timings)
    warm_avg = warm_decode_total / len(warm_timings) if warm_timings else 0
    warm_decode_no_prefill = sum(warm_timings[1:])

    print(f"\n  Warm run 2 total: vision={warm_vision_ms:.0f}ms + prefill={warm_timings[0]:.0f}ms + decode={warm_decode_no_prefill:.0f}ms")
    print(f"  Warm avg token: {warm_avg:.1f}ms")
    print(f"  Warm text: {repr(warm_text[:200])}")

    total_decode_ms = sum(token_timings_ms)
    avg_token_ms = total_decode_ms / len(token_timings_ms) if token_timings_ms else 0

    print(f"\n=== COLD RUN RESULTS ===")
    print(f"Vision (cold):  {vision_cold_ms:.0f}ms")
    print(f"Prefill:        {token_timings_ms[0]:.0f}ms")
    print(f"Decode total:   {total_decode_ms:.0f}ms ({len(generated_token_ids)} tokens)")
    print(f"Text: {repr(generated_text[:200])}")

    print(f"\n=== WARM RUN RESULTS ===")
    print(f"Vision (warm):  {warm_vision_ms:.0f}ms")
    print(f"Prefill (warm): {warm_timings[0]:.0f}ms")
    print(f"Decode (warm):  {warm_decode_no_prefill:.0f}ms ({len(warm_token_ids)-1} tokens)")
    print(f"TOTAL WARM:     {warm_vision_ms + warm_decode_total:.0f}ms")
    print(f"Text: {repr(warm_text[:200])}")

    result = {
        "image": str(image_path),
        "model_path": args.model_path,
        "device_index": args.device_index,
        "max_new_tokens": args.max_new_tokens,
        "optimization_level": args.optimization_level,
        "fixed_page": fixed_page_meta,
        "tt_device": str(tt_device),
        "tt_runtime": perf_runtime,
        "cold_run": {
            "vision_ms": round(vision_cold_ms, 3),
            "transfer_ms": round(transfer_ms, 3),
            "prefill_ms": token_timings_ms[0] if token_timings_ms else 0,
            "decode_total_ms": round(total_decode_ms, 3),
            "avg_token_ms": round(avg_token_ms, 3),
            "token_timings_ms": token_timings_ms,
            "generated_text": generated_text,
        },
        "warm_run": {
            "vision_ms": round(warm_vision_ms, 3),
            "prefill_ms": warm_timings[0] if warm_timings else 0,
            "decode_no_prefill_ms": round(warm_decode_no_prefill, 3),
            "total_ms": round(warm_vision_ms + warm_decode_total, 3),
            "avg_token_ms": round(warm_avg, 3),
            "token_timings_ms": warm_timings,
            "generated_text": warm_text,
            "match_cold": warm_text.strip() == generated_text.strip(),
        },
    }

    out_path = Path(args.json_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
