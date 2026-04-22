#!/usr/bin/env python3
"""Reference implementation of ``validate_workload(...)`` for dots.mocr on N300.

This is the API answer to NewMind's second ask: a static check that
flags Category-B (silent) failure regimes BEFORE inference runs.

It consumes ``ocr_lab/validation/envelope.json`` and returns a dict of:
    {
        "ok": bool,
        "warnings": [str, ...],
        "errors":   [str, ...],
        "details":  {<knob>: <value>, ...}
    }

This is intentionally a small, dependency-free script so it can be
copy-pasted into the production service. Re-running the Phase 2 probes
will refresh the envelope.json values; this function does not need to be
edited unless new failure modes are discovered.

Usage as a module:

    from validate_workload import validate_workload
    report = validate_workload(
        pixel_values_shape=(8960, 588),  # (num_patches, dim)
        seq_len=2259,                     # unpacked vision seq from grid_thw
        max_seq_vision=4096,              # MAX_SEQ env passed to vision tower
        n_tt_vision_blocks=5,             # N_TT env
        decoder_prefill_len=2791,         # prompt token count post-vision
        kv_dtype="bfloat16",              # current production setting
        max_cache_seq=2976,               # tile-aligned upper bound
    )

Returns ``ok`` False if any error is present; warnings are non-blocking
but should be logged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_ENVELOPE_PATH = Path(__file__).resolve().parent.parent / "envelope.json"


def _load_envelope() -> dict:
    return json.loads(_ENVELOPE_PATH.read_text())


def validate_workload(
    *,
    pixel_values_shape: tuple[int, int],
    seq_len: int,
    max_seq_vision: int,
    n_tt_vision_blocks: int,
    decoder_prefill_len: int,
    kv_dtype: str = "bfloat16",
    max_cache_seq: Optional[int] = None,
    envelope: Optional[dict] = None,
) -> dict:
    """Validate a dots.mocr workload against the documented envelope.

    All shape/precision arguments are required EXCEPT ``max_cache_seq``,
    which we derive from ``decoder_prefill_len`` if omitted.
    """
    env = envelope if envelope is not None else _load_envelope()
    warnings: list[str] = []
    errors: list[str] = []

    # Derive padded vision sl
    padded_sl = ((seq_len // 2048) + 1) * 2048
    vis = env["components"]["vision_tower"]["knobs"]
    dec = env["components"]["decoder"]["knobs"]

    # ---- vision: padded_sl vs n_tt vs dtype ----
    silent_sl_threshold = vis["padded_sl"]["silent_failure_above_when_dtype_bf8"]
    if padded_sl >= silent_sl_threshold and n_tt_vision_blocks > 0:
        warnings.append(
            f"Vision padded_sl={padded_sl} (from unpacked seq_len={seq_len}) is at or above "
            f"the bfloat8_b silent-corruption threshold ({silent_sl_threshold}). "
            f"With N_TT={n_tt_vision_blocks} this may produce CJK / doubled-suffix garbage "
            f"in the OCR output. Recommended mitigations: reduce input image resolution to "
            f"keep padded_sl<={silent_sl_threshold//2}, set N_TT=0 (CPU vision only), or "
            f"override DropInVisionTransformer's dtype to ttnn.bfloat16."
        )

    # ---- vision: max_seq_len just sizes buffers ----
    if max_seq_vision < padded_sl:
        errors.append(
            f"max_seq_vision={max_seq_vision} < padded_sl={padded_sl}. The vision tower "
            f"will OOM or under-allocate. Pass MAX_SEQ>={padded_sl} (multiples of 2048)."
        )

    # ---- vision: depth-of-bfloat8_b ----
    n_tt_ok_max = vis["n_tt_blocks"]["validated_ok_for_keyword_5_of_5"][1]
    if n_tt_vision_blocks > n_tt_ok_max:
        warnings.append(
            f"N_TT={n_tt_vision_blocks} > {n_tt_ok_max} (5/5 keyword threshold). "
            f"The bfloat8_b error in deeper blocks degrades 'hükümler' detection."
        )

    # ---- decoder KV dtype ----
    if kv_dtype not in dec["kv_dtype"]["validated_ok"]:
        if kv_dtype in dec["kv_dtype"].get("rejected_with_fatal", []):
            errors.append(
                f"kv_dtype={kv_dtype!r} is rejected at SDPA op entry "
                f"(see audit/A4_fp32_kv.md). Use 'bfloat16'."
            )
        else:
            warnings.append(
                f"kv_dtype={kv_dtype!r} is outside the validated set "
                f"{dec['kv_dtype']['validated_ok']}. Behavior is unverified."
            )

    # ---- decoder prefill seq (precision drift) ----
    safe_below = dec["prefill_seq_for_argmax_stability"]["validated_ok_below"]
    silent_above = dec["prefill_seq_for_argmax_stability"]["silent_failure_above"]
    if decoder_prefill_len >= silent_above:
        warnings.append(
            f"decoder_prefill_len={decoder_prefill_len} >= {silent_above}: TT decoder argmax "
            f"is known to diverge from HF fp32 by step ~39 (see Case B2). For OCR keyword "
            f"accuracy, route this prompt to the HF (CPU fp32) decoder by setting "
            f"HF_DECODE=1 in the runner."
        )
    elif decoder_prefill_len > safe_below:
        warnings.append(
            f"decoder_prefill_len={decoder_prefill_len} is in the gray zone "
            f"({safe_below} < seq <= {silent_above}): occasional argmax flips possible. "
            f"Consider HF_DECODE=1 if accuracy is critical."
        )

    # ---- max_cache_seq sanity ----
    if max_cache_seq is None:
        max_cache_seq = ((decoder_prefill_len + 256 + 31) // 32) * 32
    cache_lo, cache_hi = dec["max_cache_seq"]["validated_ok"]
    if max_cache_seq > cache_hi:
        warnings.append(
            f"max_cache_seq={max_cache_seq} > {cache_hi} (upper end of validated range). "
            f"Risk of OOM at allocation time (Category A1)."
        )

    return {
        "ok": not errors,
        "warnings": warnings,
        "errors": errors,
        "details": {
            "padded_sl": padded_sl,
            "max_cache_seq": max_cache_seq,
            "kv_dtype": kv_dtype,
            "n_tt_vision_blocks": n_tt_vision_blocks,
            "decoder_prefill_len": decoder_prefill_len,
        },
    }


def _self_test() -> int:
    """Reproduce NewMind's two reported B-cases as a smoke test."""
    cases = [
        (
            "B1: 1120x1568 image, MAX_SEQ=4096, N_TT=5",
            dict(
                pixel_values_shape=(8960, 588),
                seq_len=2259,
                max_seq_vision=4096,
                n_tt_vision_blocks=5,
                decoder_prefill_len=2400,
                kv_dtype="bfloat16",
            ),
        ),
        (
            "B2: native resolution, seq=2791, bf16 KV",
            dict(
                pixel_values_shape=(0, 0),
                seq_len=2200,
                max_seq_vision=4096,
                n_tt_vision_blocks=11,
                decoder_prefill_len=2791,
                kv_dtype="bfloat16",
            ),
        ),
        (
            "Production 5/5: 672x952, MAX_SEQ=8192, N_TT=11",
            dict(
                pixel_values_shape=(2688, 588),
                seq_len=672,
                max_seq_vision=8192,
                n_tt_vision_blocks=11,
                decoder_prefill_len=1500,
                kv_dtype="bfloat16",
            ),
        ),
        (
            "A4-style fp32 KV (should error)",
            dict(
                pixel_values_shape=(2688, 588),
                seq_len=672,
                max_seq_vision=8192,
                n_tt_vision_blocks=11,
                decoder_prefill_len=1500,
                kv_dtype="float32",
            ),
        ),
    ]
    for label, kwargs in cases:
        rep = validate_workload(**kwargs)
        print(f"\n=== {label} ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
