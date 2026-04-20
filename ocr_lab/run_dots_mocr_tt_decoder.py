#!/usr/bin/env python3
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

DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "tt-mlir"
    / "third_party"
    / "tt-metal"
    / "src"
    / "tt-metal"
    / "models"
    / "sample_data"
    / "iam_ocr_image.jpg"
)

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
    parser.add_argument("--dump-path", default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_decoder_dump_iam.pt"))
    parser.add_argument("--mode", default="forward", choices=["forward", "generate"])
    parser.add_argument("--json-out", default=str(PROJECT_ROOT / "ocr_lab" / "dots_tt_decoder_latest.json"))
    return parser.parse_args()


def build_inputs(model_path: str, image_path: Path, prompt: str):
    shims_dir = PROJECT_ROOT / "ocr_lab" / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return processor, inputs


def inject_vision_embeddings(model, input_ids: torch.Tensor, img_mask: torch.Tensor, vision_embeddings: torch.Tensor) -> torch.Tensor:
    inputs_embeds = model.get_input_embeddings()(input_ids)
    true_indices = torch.nonzero(img_mask).squeeze()
    if len(true_indices) > vision_embeddings.size(0):
        true_indices = true_indices[: vision_embeddings.size(0)]
        new_img_mask = torch.zeros_like(img_mask, device=img_mask.device)
        new_img_mask[true_indices[:, 0], true_indices[:, 1]] = True
    else:
        new_img_mask = img_mask

    if vision_embeddings.size(0) != new_img_mask.sum():
        raise RuntimeError(
            f"vision embedding count mismatch: {vision_embeddings.size(0)=} mask_sum={int(new_img_mask.sum())}"
        )

    return inputs_embeds.masked_scatter(
        new_img_mask.unsqueeze(-1).expand_as(inputs_embeds),
        vision_embeddings.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
    )


def load_dump(dump_path: Path) -> dict:
    dump = torch.load(dump_path, map_location="cpu")
    expected = {"input_ids", "attention_mask", "vision_embeddings"}
    missing = expected.difference(dump.keys())
    if missing:
        raise KeyError(f"decoder dump missing keys: {sorted(missing)}")
    return dump


def patch_decoder_rotary_cpu(rotary_emb: torch.nn.Module) -> None:
    inv_freq_cpu = rotary_emb.inv_freq.detach().cpu().float()
    attention_scaling = float(rotary_emb.attention_scaling)

    @torch.no_grad()
    def cpu_rotary_forward(self, x, position_ids):
        position_ids_cpu = position_ids.detach().cpu().float()
        inv_freq_expanded = inv_freq_cpu[None, :, None].expand(position_ids_cpu.shape[0], -1, 1)
        freqs = (inv_freq_expanded @ position_ids_cpu[:, None, :]).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * attention_scaling).to(device=x.device, dtype=x.dtype)
        sin = (emb.sin() * attention_scaling).to(device=x.device, dtype=x.dtype)
        return cos, sin

    rotary_emb.forward = types.MethodType(cpu_rotary_forward, rotary_emb)


def patch_decoder_mlp_cpu_silu(model: torch.nn.Module) -> None:
    def cpu_silu_mlp_forward(self, x):
        gate_tt = self.gate_proj(x)
        up_tt = self.up_proj(x)
        gated_cpu = torch.nn.functional.silu(gate_tt.detach().cpu().float()) * up_tt.detach().cpu().float()
        gated_tt = gated_cpu.to(device=x.device, dtype=x.dtype)
        return self.down_proj(gated_tt)

    for layer in model.model.layers:
        layer.mlp.forward = types.MethodType(cpu_silu_mlp_forward, layer.mlp)


def build_generation_text(processor, inputs, generated_ids: torch.Tensor) -> str:
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def build_generation_text_from_ids(processor, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


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

    from dots_tt_port.vision_tower_smoke import (
        TowerHybridCpuUnary,
        ensure_tt_env,
        load_vision_model,
        resolve_snapshot_dir,
    )
    import torch_xla
    import torch_xla.runtime as xr
    from transformers import AutoModelForCausalLM

    snapshot_dir = resolve_snapshot_dir(args.model_path)
    model_path = str(snapshot_dir)

    processor = None
    inputs = None
    dump_path = Path(args.dump_path).resolve() if args.dump_path else None

    ensure_tt_env(args.device_index)
    xr.set_device_type("TT")
    tt_device = torch_xla.device()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to(dtype=torch.bfloat16)

    patch_decoder_rotary_cpu(model.model.rotary_emb)
    patch_decoder_mlp_cpu_silu(model)

    if dump_path and dump_path.exists():
        processor = build_inputs(model_path, image_path, args.prompt)[0]
        dump = load_dump(dump_path)
        input_ids = dump["input_ids"]
        attention_mask = dump["attention_mask"]
        vision_embeddings = dump["vision_embeddings"].to(dtype=torch.bfloat16)
        img_mask = input_ids == model.config.image_token_id
        inputs_embeds = inject_vision_embeddings(model, input_ids, img_mask, vision_embeddings)
    else:
        processor, inputs = build_inputs(model_path, image_path, args.prompt)
        tt_vision_model, _ = load_vision_model(snapshot_dir, args.attn, limit_layers=0)
        tt_vision_model = TowerHybridCpuUnary(tt_vision_model).eval().to(dtype=torch.bfloat16).to(tt_device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        img_mask = inputs.input_ids == model.config.image_token_id
        pixel_values_tt = inputs["pixel_values"].to(dtype=torch.bfloat16).to(tt_device)
        image_grid_thw_tt = inputs["image_grid_thw"].to(dtype=torch.int32).to(tt_device)

        with torch.no_grad():
            vision_embeddings_tt = tt_vision_model(pixel_values_tt, image_grid_thw_tt)
            torch_xla.sync(wait=True)
            vision_embeddings = vision_embeddings_tt.detach().cpu().to(dtype=torch.bfloat16)
        inputs_embeds = inject_vision_embeddings(model, input_ids, img_mask, vision_embeddings)

    model = model.to(tt_device)
    input_ids_tt = input_ids.to(tt_device)
    attention_mask_tt = attention_mask.to(tt_device)
    inputs_embeds_tt = inputs_embeds.to(device=tt_device, dtype=torch.bfloat16)

    with torch.no_grad():
        cpu_t0 = time.perf_counter()
        cpu_out = model.cpu()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        cpu_t1 = time.perf_counter()
        cpu_logits = cpu_out.logits.detach().cpu().float()
        cpu_next_token = cpu_logits[:, -1, :].argmax(-1)

        model = model.to(tt_device)
        tt_t0 = time.perf_counter()
        if args.mode == "forward":
            tt_out = model(
                input_ids=input_ids_tt,
                attention_mask=attention_mask_tt,
                inputs_embeds=inputs_embeds_tt,
                use_cache=False,
                return_dict=True,
            )
            torch_xla.sync(wait=True)
            tt_t1 = time.perf_counter()
            tt_logits = tt_out.logits.detach().cpu().float()
            tt_next_token = tt_logits[:, -1, :].argmax(-1)
            generated_ids = None
            output_text = processor.batch_decode(tt_next_token.unsqueeze(0), skip_special_tokens=False)[0]
        else:
            generated_ids_tt = model.generate(
                input_ids=input_ids_tt,
                attention_mask=attention_mask_tt,
                inputs_embeds=inputs_embeds_tt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            torch_xla.sync(wait=True)
            tt_t1 = time.perf_counter()
            generated_ids = generated_ids_tt.detach().cpu()
            tt_logits = None
            tt_next_token = None
            output_text = build_generation_text_from_ids(processor, input_ids, generated_ids)

    result = {
        "image": str(image_path),
        "model_path": args.model_path,
        "dump_path": str(dump_path) if dump_path else None,
        "device_index": args.device_index,
        "tt_device": str(tt_device),
        "mode": args.mode,
        "cpu_prefill_ms": round((cpu_t1 - cpu_t0) * 1000.0, 3),
        "tt_ms": round((tt_t1 - tt_t0) * 1000.0, 3),
        "input_ids_shape": list(input_ids.shape),
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "output_text": output_text,
    }
    if args.mode == "forward":
        flat_cpu = cpu_logits.flatten()
        flat_tt = tt_logits.flatten()
        diff = (flat_cpu - flat_tt).abs()
        pcc = torch.corrcoef(torch.stack([flat_cpu, flat_tt]))[0, 1].item()
        result.update(
            {
                "cpu_logits_shape": list(cpu_logits.shape),
                "tt_logits_shape": list(tt_logits.shape),
                "cpu_next_token_id": int(cpu_next_token.item()),
                "tt_next_token_id": int(tt_next_token.item()),
                "cpu_next_token_text": processor.batch_decode(cpu_next_token.unsqueeze(0), skip_special_tokens=False)[0],
                "tt_next_token_text": output_text,
                "pcc": round(float(pcc), 6),
                "max_abs_diff": round(float(diff.max()), 6),
                "mean_abs_diff": round(float(diff.mean()), 6),
            }
        )
    else:
        result["generated_ids_shape"] = list(generated_ids.shape)

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
