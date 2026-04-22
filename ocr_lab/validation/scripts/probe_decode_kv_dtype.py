#!/usr/bin/env python3
"""B2.a: KV-cache dtype ablation.

Drives the same logits-comparison core as ocr_lab/probe_tt_decode_divergence.py
once with bfloat16 KV cache (current production), once with bfloat8_b KV cache,
and (for completeness) once attempting fp32 KV. The fp32 case is expected to
hit the SDPA dtype check (Category-A4); we capture that as evidence.

Outputs ocr_lab/validation/logs/decode_kv_dtype.json with per-run
first-divergence-step and cosine-at-step traces.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import write_log  # noqa: E402
from _decode_probe_core import ProbeConfig, run_probe, close_mesh  # noqa: E402

import ttnn  # noqa: E402


CASES = [
    ("bfloat16", ttnn.bfloat16),
    ("bfloat8_b", ttnn.bfloat8_b),
    ("float32 (expected to fail at SDPA)", ttnn.float32),
]


def main() -> int:
    out: list[dict] = []
    for label, dtype in CASES:
        cfg = ProbeConfig(label=f"kv_dtype={label}", kv_dtype=dtype, native=True, n_tt=10)
        print(f"\n=== {cfg.label} ===", flush=True)
        res = run_probe(cfg)
        out.append(
            {
                "label": res.label,
                "kv_dtype": res.kv_dtype,
                "seq": res.seq,
                "first_divergence_step": res.first_divergence_step,
                "divergence_count": res.divergence_count,
                "total_steps": res.total_steps,
                "cosine_at_steps": res.cosine_at_steps,
                "elapsed_s": res.elapsed_s,
                "error": res.error,
            }
        )
        print(json.dumps(out[-1], indent=2, ensure_ascii=False))
    close_mesh()
    write_log("decode_kv_dtype", {"cases": out})
    return 0


if __name__ == "__main__":
    sys.exit(main())
