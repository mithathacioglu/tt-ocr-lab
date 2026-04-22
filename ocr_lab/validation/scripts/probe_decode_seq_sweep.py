#!/usr/bin/env python3
"""B2.c: Sweep prefill sequence length and find the first-divergence step.

Drives the logits-comparison core at a range of prefill lengths obtained by
truncating the input_ids at a synthetic boundary. Each cell records:

  - seq:                     actual prefill length used
  - first_divergence_step:   first decode step where TT argmax != HF argmax
  - divergence_count:        total argmax mismatches across MAX_NEW steps
  - cosine_at_step_60:       cosine of TT-vs-HF logits at step 60 (the
                             critical "hükümler" position in NewMind's report)

The output table is the empirical evidence for the "precision-safe maximum
KV length" answer to NewMind's third ask.

Output: ocr_lab/validation/logs/decode_seq_sweep.json (+ .md)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from _common import write_log, PROJECT_ROOT  # noqa: E402
from _decode_probe_core import ProbeConfig, run_probe, close_mesh  # noqa: E402


# Default sweep: 8 prefill lengths around the failure threshold
DEFAULT_LENS = [500, 1000, 1500, 2000, 2400, 2791, 3500]


def main() -> int:
    raw = os.environ.get("SEQ_SWEEP")
    lengths = [int(x) for x in raw.split(",")] if raw else DEFAULT_LENS

    out: list[dict] = []
    for cut in lengths:
        cfg = ProbeConfig(
            label=f"seq={cut}",
            native=True,
            n_tt=10,
            max_new=int(os.environ.get("MAX_NEW", "100")),
            truncate_prompt_to=cut,
        )
        print(f"\n=== {cfg.label} ===", flush=True)
        res = run_probe(cfg)
        out.append(
            {
                "label": res.label,
                "seq": res.seq,
                "first_divergence_step": res.first_divergence_step,
                "divergence_count": res.divergence_count,
                "total_steps": res.total_steps,
                "cosine_at_60": res.cosine_at_steps.get(60),
                "cosine_at_steps": res.cosine_at_steps,
                "elapsed_s": res.elapsed_s,
                "error": res.error,
            }
        )
        print(json.dumps(out[-1], indent=2, ensure_ascii=False))
    close_mesh()
    payload = {"cases": out, "lengths": lengths}
    write_log("decode_seq_sweep", payload)

    # Markdown summary
    md = ["# B2.c — decode divergence vs prefill seq length", "",
          "| seq | first_div_step | div_count/total | cos@60 | elapsed_s |",
          "|---|---|---|---|---|"]
    for c in out:
        md.append(
            f"| {c['seq']} | {c['first_divergence_step']} | "
            f"{c['divergence_count']}/{c['total_steps']} | "
            f"{c['cosine_at_60']} | {c['elapsed_s']:.1f} |"
        )
    out_md = PROJECT_ROOT / "ocr_lab" / "validation" / "logs" / "decode_seq_sweep.md"
    out_md.write_text("\n".join(md))
    print(f"\nWrote {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
