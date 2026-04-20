#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DUMP = PROJECT_ROOT / "ocr_lab" / "dots_tt_decoder_dump_iam.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="rednote-hilab/dots.mocr")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--dump-path", default=str(DEFAULT_DUMP))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", default="prefill", choices=["prefill", "generate"])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--mlp-mode",
        default="cpu_silu",
        choices=["cpu_silu", "cpu_silu_tt_mul", "tt_exact", "tt_poly2", "tt_poly7_datafit"],
    )
    parser.add_argument(
        "--json-out",
        default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_decoder_prefill_hybrid_latest.json"),
    )
    return parser.parse_args()


def load_dump(dump_path: Path) -> dict:
    dump = torch.load(dump_path, map_location="cpu")
    expected = {"input_ids", "attention_mask", "vision_embeddings"}
    missing = expected.difference(dump.keys())
    if missing:
        raise KeyError(f"decoder dump missing keys: {sorted(missing)}")
    return dump


def resolve_snapshot_dir(model_path: str) -> Path:
    candidate = Path(model_path)
    if candidate.exists():
        return candidate.resolve()

    repo_dir = model_path.replace("/", "--")
    if not repo_dir.startswith("models--"):
        repo_dir = f"models--{repo_dir}"

    roots = [
        PROJECT_ROOT / ".container-home" / ".cache" / "huggingface" / "hub",
        Path.home() / ".container-home" / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        snap_base = root / repo_dir / "snapshots"
        if not snap_base.exists():
            continue
        snaps = sorted(p for p in snap_base.iterdir() if p.is_dir())
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(f"no snapshots found for {model_path}")


def ensure_tt_env(device_index: str) -> None:
    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")
    os.environ.setdefault("TT_VISIBLE_DEVICES", device_index)
    mesh_desc = (
        PROJECT_ROOT
        / ".venv-tt-xla"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "pjrt_plugin_tt"
        / "tt-metal"
        / "tt_metal"
        / "fabric"
        / "mesh_graph_descriptors"
        / "n300_mesh_graph_descriptor.textproto"
    )
    if mesh_desc.exists():
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", str(mesh_desc))


def inject_vision_embeddings(model, input_ids: torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:
    img_mask = input_ids == model.config.image_token_id
    inputs_embeds = model.get_input_embeddings()(input_ids)
    return inputs_embeds.masked_scatter(
        img_mask.unsqueeze(-1).expand_as(inputs_embeds),
        vision_embeddings.to(inputs_embeds.dtype),
    )


def build_rotary_cache_cpu(rotary_emb: torch.nn.Module, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    inv_freq = rotary_emb.inv_freq.detach().cpu().float()
    pos_cpu = position_ids.float()
    freqs = (inv_freq[None, :, None].expand(pos_cpu.shape[0], -1, 1) @ pos_cpu[:, None, :]).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    scaling = float(rotary_emb.attention_scaling)
    cos_cpu = (emb.cos() * scaling).to(torch.float32)
    sin_cpu = (emb.sin() * scaling).to(torch.float32)
    return cos_cpu, sin_cpu


def main() -> int:
    args = parse_args()
    dump_path = Path(args.dump_path).resolve()
    snapshot_dir = resolve_snapshot_dir(args.model_path)
    dump = load_dump(dump_path)

    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch_xla
    import torch_xla.runtime as xr
    from ocr_lab.dots_tt_hybrid_decoder import (
        build_rotary_cache_cpu,
        hybrid_greedy_generate,
    )

    ensure_tt_env(args.device_index)

    processor = AutoProcessor.from_pretrained(str(snapshot_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to(dtype=torch.bfloat16)

    input_ids = dump["input_ids"]
    attention_mask = dump["attention_mask"]
    vision_embeddings = dump["vision_embeddings"].to(dtype=torch.bfloat16)
    inputs_embeds = inject_vision_embeddings(model, input_ids, vision_embeddings)

    with torch.no_grad():
        cpu_t0 = time.perf_counter()
        cpu_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        cpu_t1 = time.perf_counter()
        cpu_logits = cpu_out.logits.detach().cpu().float()
        cpu_next = cpu_logits[:, -1, :].argmax(-1)

    xr.set_device_type("TT")
    device = torch_xla.device()
    model = model.to(device)

    iterations = []
    tt_logits = None
    with torch.no_grad():
        if args.mode == "prefill":
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

            num_heads = model.config.num_attention_heads
            num_kv_heads = model.config.num_key_value_heads
            head_dim = model.config.hidden_size // num_heads
            num_kv_groups = num_heads // num_kv_heads
            seq_len = input_ids.shape[1]
            cos_cpu, sin_cpu = build_rotary_cache_cpu(model.model.rotary_emb, seq_len)

            for rep in range(args.repeats):
                hidden_tt = inputs_embeds.to(device=device, dtype=torch.bfloat16)
                t0 = time.perf_counter()

                for layer in model.model.layers:
                    residual_tt = hidden_tt
                    norm1_tt = layer.input_layernorm(hidden_tt)
                    q_tt = layer.self_attn.q_proj(norm1_tt)
                    k_tt = layer.self_attn.k_proj(norm1_tt)
                    v_tt = layer.self_attn.v_proj(norm1_tt)

                    q_cpu = q_tt.detach().cpu().float().view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                    k_cpu = k_tt.detach().cpu().float().view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                    v_cpu = v_tt.detach().cpu().float().view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                    q_cpu, k_cpu = apply_rotary_pos_emb(q_cpu, k_cpu, cos_cpu, sin_cpu)
                    k_cpu = repeat_kv(k_cpu, num_kv_groups)
                    v_cpu = repeat_kv(v_cpu, num_kv_groups)
                    attn_out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, dropout_p=0.0, is_causal=True)
                    attn_out_cpu = attn_out_cpu.transpose(1, 2).reshape(1, seq_len, -1).to(dtype=torch.bfloat16)
                    attn_out_tt = layer.self_attn.o_proj(attn_out_cpu.to(device))
                    hidden_tt = residual_tt + attn_out_tt

                    residual_tt = hidden_tt
                    norm2_tt = layer.post_attention_layernorm(hidden_tt)
                    gate_tt = layer.mlp.gate_proj(norm2_tt)
                    up_tt = layer.mlp.up_proj(norm2_tt)
                    gated_cpu = F.silu(gate_tt.detach().cpu().float()) * up_tt.detach().cpu().float()
                    down_tt = layer.mlp.down_proj(gated_cpu.to(device=device, dtype=torch.bfloat16))
                    hidden_tt = residual_tt + down_tt

                hidden_tt = model.model.norm(hidden_tt)
                logits_tt = model.lm_head(hidden_tt)
                torch_xla.sync(wait=True)
                t1 = time.perf_counter()

                tt_logits = logits_tt.detach().cpu().float()
                tt_next = tt_logits[:, -1, :].argmax(-1)
                iterations.append(
                    {
                        "index": rep + 1,
                        "tt_hybrid_ms": round((t1 - t0) * 1000.0, 3),
                        "tt_next_token_id": int(tt_next.item()),
                        "tt_next_text": processor.batch_decode(tt_next.unsqueeze(0), skip_special_tokens=False)[0],
                    }
                )
        else:
            embedding_weight_cpu = model.get_input_embeddings().weight.detach().cpu().to(dtype=torch.bfloat16)
            for rep in range(args.repeats):
                t0 = time.perf_counter()
                generate_out = hybrid_greedy_generate(
                    model,
                    processor,
                    input_ids,
                    vision_embeddings,
                    device,
                    max_new_tokens=args.max_new_tokens,
                    sync_fn=lambda: torch_xla.sync(wait=True),
                    eos_token_id=model.generation_config.eos_token_id,
                    embedding_weight_cpu=embedding_weight_cpu,
                    mlp_mode=args.mlp_mode,
                )
                t1 = time.perf_counter()
                iterations.append(
                    {
                        "index": rep + 1,
                        "tt_hybrid_ms": round((t1 - t0) * 1000.0, 3),
                        "num_generated_tokens": generate_out["num_generated_tokens"],
                        "token_timings_ms": generate_out["token_timings_ms"],
                        "generated_token_ids": generate_out["generated_token_ids"],
                        "generated_text": generate_out["generated_text"],
                        "generated_text_raw": generate_out["generated_text_raw"],
                    }
                )

    if args.mode == "generate":
        result = {
            "dump_path": str(dump_path),
            "device_index": args.device_index,
            "tt_device": str(device),
            "mode": args.mode,
            "mlp_mode": args.mlp_mode,
            "cpu_ms": round((cpu_t1 - cpu_t0) * 1000.0, 3),
            "cpu_next_token_id": int(cpu_next.item()),
            "cpu_next_text": processor.batch_decode(cpu_next.unsqueeze(0), skip_special_tokens=False)[0],
            "iterations": iterations,
            "summary": {
                "first_ms": iterations[0]["tt_hybrid_ms"],
                "hot_avg_ms": round(
                    sum(x["tt_hybrid_ms"] for x in iterations[1:]) / max(1, len(iterations) - 1),
                    3,
                ),
            },
        }
    else:
        if tt_logits is None:
            raise RuntimeError("no TT logits produced")

        flat_cpu = cpu_logits.flatten()
        flat_tt = tt_logits.flatten()
        diff = (flat_cpu - flat_tt).abs()
        pcc = torch.corrcoef(torch.stack([flat_cpu, flat_tt]))[0, 1].item()

        result = {
            "dump_path": str(dump_path),
            "device_index": args.device_index,
            "tt_device": str(device),
            "mode": args.mode,
            "mlp_mode": args.mlp_mode,
            "cpu_ms": round((cpu_t1 - cpu_t0) * 1000.0, 3),
            "iterations": iterations,
            "summary": {
                "first_ms": iterations[0]["tt_hybrid_ms"],
                "hot_avg_ms": round(
                    sum(x["tt_hybrid_ms"] for x in iterations[1:]) / max(1, len(iterations) - 1),
                    3,
                ),
                "cpu_next_token_id": int(cpu_next.item()),
                "cpu_next_text": processor.batch_decode(cpu_next.unsqueeze(0), skip_special_tokens=False)[0],
                "pcc_vs_cpu_last": round(float(pcc), 6),
                "max_abs_diff_last": round(float(diff.max()), 6),
                "mean_abs_diff_last": round(float(diff.mean()), 6),
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
