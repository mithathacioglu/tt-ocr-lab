#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from ocr_lab.dots_model import DEFAULT_MODEL_PATH, ensure_dots_ocr_processor_compat

DEFAULT_PROMPT = """Please output the layout information from the document image, including each layout element's bbox, category, and text.

1. Bbox format: [x1, y1, x2, y2]
2. Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
3. Format formulas as LaTeX, tables as HTML, and all other text as Markdown.
4. Keep the original language and preserve human reading order.
5. Return a single JSON object.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    shims_dir = Path(__file__).resolve().parent / "shims"
    if shims_dir.exists():
        sys.path.insert(0, str(shims_dir))

    if args.device.lower() == "tt":
        raise RuntimeError("dots OCR icin bu script CPU/CUDA referans yoludur; TT icin hibrit ya da compiled path kullan.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        raise RuntimeError("transformers ve torch gerekli") from exc

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError("qwen_vl_utils gerekli") from exc

    use_cuda = args.device.lower() in {"auto", "cuda"} and torch.cuda.is_available()
    load_kwargs = {
        "trust_remote_code": True,
    }
    if use_cuda:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
        load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["attn_implementation"] = "sdpa"

    ensure_dots_ocr_processor_compat(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    if not use_cuda:
        vision_module = sys.modules.get(model.vision_tower.__class__.__module__)
        if vision_module is not None and hasattr(vision_module, "VisionSdpaAttention"):
            for blk in model.vision_tower.blocks:
                old_attn = blk.attn
                new_attn = vision_module.VisionSdpaAttention(
                    model.config.vision_config,
                    old_attn.proj.in_features,
                    num_heads=old_attn.num_heads,
                    bias=old_attn.qkv.bias is not None,
                )
                new_attn.load_state_dict(old_attn.state_dict())
                new_attn = new_attn.to(device=old_attn.qkv.weight.device, dtype=old_attn.qkv.weight.dtype)
                blk.attn = new_attn

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if use_cuda:
        inputs = inputs.to("cuda")
    else:
        for key, value in list(inputs.items()):
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating_point", False):
                inputs[key] = value.to(dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result = {
        "model_path": args.model_path,
        "image": str(image_path),
        "device": "cuda" if use_cuda else "cpu",
        "output_text": output_text,
    }

    if args.json_out:
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
