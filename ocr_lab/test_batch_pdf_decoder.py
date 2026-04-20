#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ocr_lab.tt_perf import configure_tt_runtime
from ocr_lab.fixed_page import fixed_page_message


DEFAULT_PDF = Path("/home/mlops/Desktop/dosyalar/3.1.pdf")
DEFAULT_RENDER_DIR = PROJECT_ROOT / "ocr_lab" / "tmp_pdf_batch_render"
DEFAULT_JSON = PROJECT_ROOT / "ocr_lab" / "batch3_pdf_test_results.json"
DEFAULT_PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--pages", default="1,2,3")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--fixed-page-width", type=int, default=468)
    parser.add_argument("--fixed-page-height", type=int, default=662)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--render-dir", default=str(DEFAULT_RENDER_DIR))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--pad-prefix-multiple", type=int, default=0)
    return parser.parse_args()


def parse_pages(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def render_page(pdf_path: Path, page_num: int, render_dir: Path) -> Path:
    render_dir.mkdir(parents=True, exist_ok=True)
    prefix = render_dir / f"{pdf_path.stem}_page_{page_num}"
    out_path = prefix.parent / f"{prefix.name}.png"
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not out_path.exists():
        raise FileNotFoundError(out_path)
    return out_path


def build_inputs(processor, process_vision_info, image_path: Path, prompt: str, width: int, height: int):
    img_msg, meta = fixed_page_message(image_path, width=width, height=height)
    messages = [{"role": "user", "content": [img_msg, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    return inputs, meta


def run_generate(
    compiled_model,
    input_ids_tt: torch.Tensor,
    inputs_embeds_tt: torch.Tensor,
    static_cache,
    cache_position_tt: torch.Tensor,
    attn_mask_tt: torch.Tensor,
    tt_device,
    eos_ids: set[int],
    max_tokens: int,
    batch_size: int,
):
    import torch_xla

    token_ids = [[] for _ in range(batch_size)]
    timings = []
    seq_len = input_ids_tt.shape[1]

    with torch.no_grad():
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
        timings.append(round((t1 - t0) * 1000.0, 3))

        next_ids = logits[:, -1, :].argmax(-1)
        for b in range(batch_size):
            token_ids[b].append(int(next_ids[b].item()))

        cur_pos = seq_len
        for _step in range(1, max_tokens):
            if all(token_ids[b][-1] in eos_ids for b in range(batch_size)):
                break

            t0 = time.perf_counter()
            next_input = next_ids.unsqueeze(1).to(tt_device)
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
            timings.append(round((t1 - t0) * 1000.0, 3))

            next_ids = logits[:, -1, :].argmax(-1)
            for b in range(batch_size):
                token_ids[b].append(int(next_ids[b].item()))
            cur_pos += 1

    return token_ids, timings


def move_cache_to_device(cache, tt_device) -> None:
    for layer_cache in cache.layers:
        layer_cache.keys = layer_cache.keys.to(tt_device)
        layer_cache.values = layer_cache.values.to(tt_device)


def pad_prefix_tensors(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embeds_cpu: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if multiple <= 0:
        return input_ids, attention_mask, embeds_cpu

    seq_len = input_ids.shape[1]
    padded_len = ((seq_len + multiple - 1) // multiple) * multiple
    pad_len = padded_len - seq_len
    if pad_len <= 0:
        return input_ids, attention_mask, embeds_cpu

    input_ids_padded = F.pad(input_ids, (0, pad_len), value=0)
    attention_mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
    embed_pad = torch.zeros(
        (embeds_cpu.shape[0], pad_len, embeds_cpu.shape[2]),
        dtype=embeds_cpu.dtype,
    )
    embeds_padded = torch.cat([embeds_cpu, embed_pad], dim=1)
    return input_ids_padded, attention_mask_padded, embeds_padded


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages = parse_pages(args.pages)
    if len(pages) != args.batch_size:
        raise ValueError(f"pages count {len(pages)} must equal batch-size {args.batch_size}")

    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.cache_utils import StaticCache
    from qwen_vl_utils import process_vision_info
    from dots_tt_port.vision_tower_smoke import (
        TowerHybridCpuUnary,
        ensure_tt_env,
        load_vision_model,
        resolve_snapshot_dir,
    )
    from ocr_lab.dots_tt_hybrid_decoder import build_inputs_embeds_cpu
    import torch_xla.runtime as xr
    import torch_xla

    render_dir = Path(args.render_dir).resolve() / pdf_path.stem
    rendered_pages = [render_page(pdf_path, page, render_dir) for page in pages]

    snapshot_dir = resolve_snapshot_dir("rednote-hilab/dots.mocr")
    model_path = str(snapshot_dir)

    ensure_tt_env(args.device_index)
    configure_tt_runtime(cache_dir=None, enable_trace=False, optimization_level=2)
    xr.set_device_type("TT")
    tt_device = torch_xla.device()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    page_inputs = []
    fixed_page_meta = []
    for image_path in rendered_pages:
        inputs, meta = build_inputs(
            processor,
            process_vision_info,
            image_path,
            args.prompt,
            args.fixed_page_width,
            args.fixed_page_height,
        )
        page_inputs.append(inputs)
        fixed_page_meta.append(meta)

    input_ids_ref = page_inputs[0]["input_ids"]
    attn_mask_ref = page_inputs[0]["attention_mask"]
    for idx, inputs in enumerate(page_inputs[1:], start=2):
        if not torch.equal(inputs["input_ids"], input_ids_ref):
            raise RuntimeError(f"page {idx} input_ids differ; batch decode path expects fixed prompt/token layout")
        if not torch.equal(inputs["attention_mask"], attn_mask_ref):
            raise RuntimeError(f"page {idx} attention mask differs")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=True,
    ).eval()
    tt_vision, _ = load_vision_model(snapshot_dir, args.attn, limit_layers=0)
    tt_vision = TowerHybridCpuUnary(tt_vision).eval().to(dtype=torch.bfloat16).to(tt_device)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)
    eos_id = model.config.eos_token_id
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id) if eos_id else set()

    # Vision warmup once for fixed shape
    warmup_pv = page_inputs[0]["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
    warmup_gt = page_inputs[0]["image_grid_thw"].to(dtype=torch.int32).to(tt_device)
    with torch.no_grad():
        _ = tt_vision(warmup_pv, warmup_gt)
        torch_xla.sync(wait=True)

    # Measure vision per page, hot
    page_vision_embeddings_cpu = []
    page_vision_ms = []
    for inputs in page_inputs:
        pv_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
        gt_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(tt_device)
        with torch.no_grad():
            t0 = time.perf_counter()
            ve_tt = tt_vision(pv_tt, gt_tt)
            torch_xla.sync(wait=True)
            t1 = time.perf_counter()
        page_vision_ms.append(round((t1 - t0) * 1000.0, 3))
        page_vision_embeddings_cpu.append(ve_tt.detach().cpu().to(dtype=torch.bfloat16))

    model = model.to(tt_device)
    compiled_model = torch.compile(model, backend="tt")

    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Batch=1 warmup compile
    embeds1_warm = build_inputs_embeds_cpu(model, input_ids_ref, page_vision_embeddings_cpu[0], embedding_weight_cpu=emb_weight)
    input_ids_single, attn_mask_single, embeds1_warm = pad_prefix_tensors(
        input_ids_ref,
        attn_mask_ref,
        embeds1_warm,
        args.pad_prefix_multiple,
    )
    seq_len = input_ids_single.shape[1]
    max_cache_len = seq_len + args.max_new_tokens + 8
    cache_position = torch.arange(0, seq_len).to(tt_device)
    embeds1_warm_tt = embeds1_warm.to(dtype=torch.bfloat16).to(tt_device)
    dummy1 = torch.zeros((1, seq_len), dtype=torch.long).to(tt_device)
    full_mask1_cpu = torch.ones((1, max_cache_len), dtype=torch.long)
    full_mask1_cpu[:, : attn_mask_single.shape[1]] = 0
    full_mask1_cpu[:, : attn_mask_single.shape[1]] = attn_mask_single
    full_mask1 = full_mask1_cpu.to(tt_device)
    cache1 = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    cache1.early_initialization(batch_size=1, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    move_cache_to_device(cache1, tt_device)
    _ = run_generate(compiled_model, dummy1, embeds1_warm_tt, cache1, cache_position, full_mask1, tt_device, eos_ids, args.max_new_tokens, 1)

    single_page_results = []
    for image_path, inputs, ve_cpu, vision_ms in zip(rendered_pages, page_inputs, page_vision_embeddings_cpu, page_vision_ms):
        embeds1_cpu = build_inputs_embeds_cpu(model, input_ids_ref, ve_cpu, embedding_weight_cpu=emb_weight)
        _, _, embeds1_cpu = pad_prefix_tensors(
            input_ids_ref,
            attn_mask_ref,
            embeds1_cpu,
            args.pad_prefix_multiple,
        )
        embeds1_tt = embeds1_cpu.to(dtype=torch.bfloat16).to(tt_device)

        cache1w = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
        cache1w.early_initialization(batch_size=1, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
        move_cache_to_device(cache1w, tt_device)

        t0 = time.perf_counter()
        token_ids, token_timings = run_generate(
            compiled_model,
            dummy1,
            embeds1_tt,
            cache1w,
            cache_position,
            full_mask1,
            tt_device,
            eos_ids,
            args.max_new_tokens,
            1,
        )
        decoder_ms = (time.perf_counter() - t0) * 1000.0
        text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        single_page_results.append(
            {
                "image": str(image_path),
                "vision_ms": vision_ms,
                "decoder_ms": round(decoder_ms, 3),
                "total_ms": round(vision_ms + decoder_ms, 3),
                "token_timings_ms": token_timings,
                "text": text,
            }
        )

    # Batch=3 warmup compile
    embeds3_cpu = torch.cat(
        [
            pad_prefix_tensors(
                input_ids_ref,
                attn_mask_ref,
                build_inputs_embeds_cpu(model, input_ids_ref, ve_cpu, embedding_weight_cpu=emb_weight),
                args.pad_prefix_multiple,
            )[2]
            for ve_cpu in page_vision_embeddings_cpu
        ],
        dim=0,
    )
    embeds3_tt = embeds3_cpu.to(dtype=torch.bfloat16).to(tt_device)
    dummy3 = torch.zeros((args.batch_size, seq_len), dtype=torch.long).to(tt_device)
    full_mask3_cpu = torch.ones((args.batch_size, max_cache_len), dtype=torch.long)
    full_mask3_cpu[:, : attn_mask_single.shape[1]] = 0
    full_mask3_cpu[:, : attn_mask_single.shape[1]] = attn_mask_single.repeat(args.batch_size, 1)
    full_mask3 = full_mask3_cpu.to(tt_device)
    cache3 = StaticCache(config=model.config, max_batch_size=args.batch_size, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    cache3.early_initialization(batch_size=args.batch_size, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    move_cache_to_device(cache3, tt_device)
    _ = run_generate(compiled_model, dummy3, embeds3_tt, cache3, cache_position, full_mask3, tt_device, eos_ids, args.max_new_tokens, args.batch_size)

    cache3w = StaticCache(config=model.config, max_batch_size=args.batch_size, max_cache_len=max_cache_len, device="cpu", dtype=torch.bfloat16)
    cache3w.early_initialization(batch_size=args.batch_size, num_heads=num_kv_heads, head_dim=head_dim, dtype=torch.bfloat16, device="cpu")
    move_cache_to_device(cache3w, tt_device)

    t0 = time.perf_counter()
    token_ids3, token_timings3 = run_generate(
        compiled_model,
        dummy3,
        embeds3_tt,
        cache3w,
        cache_position,
        full_mask3,
        tt_device,
        eos_ids,
        args.max_new_tokens,
        args.batch_size,
    )
    batch3_decoder_ms = (time.perf_counter() - t0) * 1000.0
    batch3_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids3]

    single_total_ms = round(sum(item["total_ms"] for item in single_page_results), 3)
    batch3_vision_total_ms = round(sum(page_vision_ms), 3)
    batch3_total_ms = round(batch3_vision_total_ms + batch3_decoder_ms, 3)
    per_page_batch3_ms = round(batch3_total_ms / args.batch_size, 3)

    result = {
        "pdf": str(pdf_path),
        "pages": pages,
        "rendered_pages": [str(p) for p in rendered_pages],
        "batch_size": args.batch_size,
        "fixed_page_width": args.fixed_page_width,
        "fixed_page_height": args.fixed_page_height,
        "max_new_tokens": args.max_new_tokens,
        "device_index": args.device_index,
        "pad_prefix_multiple": args.pad_prefix_multiple,
        "padded_seq_len": seq_len,
        "single_page": single_page_results,
        "batch": {
            "vision_total_ms": batch3_vision_total_ms,
            "decoder_ms": round(batch3_decoder_ms, 3),
            "total_ms": batch3_total_ms,
            "per_page_ms": per_page_batch3_ms,
            "token_timings_ms": token_timings3,
            "texts": batch3_texts,
            "match_single_page": [batch3_texts[i].strip() == single_page_results[i]["text"].strip() for i in range(args.batch_size)],
        },
        "summary": {
            "single_total_ms": single_total_ms,
            "single_per_page_avg_ms": round(single_total_ms / args.batch_size, 3),
            "batch_total_ms": batch3_total_ms,
            "batch_per_page_ms": per_page_batch3_ms,
            "throughput_speedup_x": round((single_total_ms / args.batch_size) / per_page_batch3_ms, 3),
            "decoder_speedup_x": round(
                (sum(item["decoder_ms"] for item in single_page_results) / args.batch_size) / (batch3_decoder_ms / args.batch_size),
                3,
            ),
        },
    }

    out_path = Path(args.json_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
