#!/usr/bin/env python3
"""Accuracy sweep: compare output text across resolutions and token counts."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OCR_LAB = Path(__file__).resolve().parent
PROJECT_ROOT = OCR_LAB.parent
RUNNER = OCR_LAB / "run_dots_mocr_tt_full_hybrid.py"
IMAGE = OCR_LAB / "tmp_1pdf_page-1.png"

RESOLUTIONS = [
    (672, 952),   # referans
    (476, 674),   # hızlı güvenli
    (468, 662),   # alt güvenli sınır
]

TOKEN_COUNTS = [128, 256]

ACTIVATE = "source /home/mlops/dll_project/scripts/activate_tt_xla.sh"


def run_one(w, h, tokens, out_path):
    cmd = (
        f"{ACTIVATE} && python {RUNNER} "
        f"--image {IMAGE} "
        f"--fixed-page-width {w} --fixed-page-height {h} "
        f"--max-new-tokens {tokens} "
        f"--mlp-mode cpu_silu "
        f"--json-out {out_path}"
    )
    print(f"\n>>> {w}x{h} tok={tokens} ...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600, executable="/bin/bash")
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s  exit={result.returncode}", flush=True)
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[-500:]}", flush=True)
    return result.returncode == 0


def main():
    out_dir = OCR_LAB / "accuracy_sweep_results"
    out_dir.mkdir(exist_ok=True)

    all_results = []
    for w, h in RESOLUTIONS:
        for tok in TOKEN_COUNTS:
            fname = f"acc_{w}x{h}_tok{tok}.json"
            out_path = out_dir / fname
            ok = run_one(w, h, tok, out_path)
            if ok and out_path.exists():
                data = json.loads(out_path.read_text())
                all_results.append({
                    "resolution": f"{w}x{h}",
                    "tokens": tok,
                    "tt_text": data.get("tt_hybrid", {}).get("output_text", ""),
                    "cpu_text": data.get("cpu_baseline", {}).get("output_text", ""),
                    "match_cpu": data.get("match_cpu", False),
                    "tt_vision_ms": data.get("tt_hybrid", {}).get("tt_vision_ms", 0),
                    "decoder_ms": data.get("tt_hybrid", {}).get("decoder_total_ms", 0),
                    "total_ms": data.get("tt_hybrid", {}).get("total_ms", 0),
                })
            else:
                all_results.append({
                    "resolution": f"{w}x{h}",
                    "tokens": tok,
                    "error": True,
                })

    summary_path = out_dir / "accuracy_sweep_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    
    # Print comparison
    ref_texts = {}
    for r in all_results:
        if r["resolution"] == "672x952" and "error" not in r:
            ref_texts[r["tokens"]] = r["tt_text"]

    print("\n\n=== DOĞRULUK KARŞILAŞTIRMASI ===")
    for r in all_results:
        if "error" in r:
            print(f"\n{r['resolution']} tok={r['tokens']}: HATA")
            continue
        ref = ref_texts.get(r["tokens"], "")
        tt = r["tt_text"]
        
        # Character-level match
        import re
        def norm(s): return re.sub(r'\s+', ' ', s).strip()
        rn, tn = norm(ref), norm(tt)
        
        match_chars = sum(1 for a, b in zip(rn, tn) if a == b)
        total = max(len(rn), 1)
        pct = match_chars / total * 100
        
        print(f"\n{r['resolution']} tok={r['tokens']}:")
        print(f"  match_cpu: {r['match_cpu']}")
        print(f"  vs referans: {pct:.1f}% ({match_chars}/{total} char)")
        print(f"  süre: vision={r['tt_vision_ms']:.0f}ms decoder={r['decoder_ms']:.0f}ms toplam={r['total_ms']:.0f}ms")
        print(f"  metin: {repr(tt[:200])}")


if __name__ == "__main__":
    main()
