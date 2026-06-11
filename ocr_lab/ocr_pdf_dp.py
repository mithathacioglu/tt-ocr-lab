#!/usr/bin/env python3
"""dots.ocr PDF orchestrator: split a PDF across 4 N300 boards in parallel.

Two modes:
  CLI:    python ocr_pdf_dp.py <input.pdf> [-o output.md]
  Server: python ocr_pdf_dp.py --server [--port 8080]

Splitting strategy: 9 pages -> [3,2,2,2], 8 -> [2,2,2,2], 5 -> [2,1,1,1], etc.
Always front-loads the first worker with the extra page when count % 4 != 0.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --- Hardware config ---
PCI_BUSES = ["0000:31:00.0", "0000:4b:00.0", "0000:b1:00.0", "0000:ca:00.0"]
NUM_WORKERS = 4
TT_METAL_HOME = "/home/mlops/dll_project/tt-metal-fresh"
TEST_FILE = "models/experimental/tt_symbiote/tests/test_dots_ocr_t1_dp.py"
DEFAULT_DPI = 200
TARGET_W = 1240   # trace-friendly size for dots.ocr on N300
TARGET_H = 1754


def split_pages(total_pages: int, n_workers: int = NUM_WORKERS) -> list[list[int]]:
    """Distribute page numbers (1-indexed) across n_workers, front-load extras."""
    if total_pages <= 0:
        return [[] for _ in range(n_workers)]
    base, extra = divmod(total_pages, n_workers)
    sizes = [base + (1 if i < extra else 0) for i in range(n_workers)]
    out, p = [], 1
    for sz in sizes:
        out.append(list(range(p, p + sz)))
        p += sz
    return out


def extract_pages(pdf_path: Path, image_dir: Path, dpi: int = DEFAULT_DPI) -> int:
    """Render each page as PNG, then resize to TARGET_W x TARGET_H. Returns page count."""
    image_dir.mkdir(parents=True, exist_ok=True)
    info = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    pages = next(int(line.split(":")[1].strip()) for line in info.splitlines() if line.startswith("Pages:"))
    raw_prefix = image_dir / "raw"
    for i in range(1, pages + 1):
        subprocess.check_call(
            ["pdftoppm", "-r", str(dpi), "-f", str(i), "-l", str(i),
             str(pdf_path), str(raw_prefix), "-png"],
            stdout=subprocess.DEVNULL,
        )
        raw_path = next(image_dir.glob(f"raw-{i:0>1}*.png"))
        # Resize to trace shape
        from PIL import Image
        img = Image.open(raw_path)
        img.resize((TARGET_W, TARGET_H), Image.LANCZOS).save(image_dir / f"t1_page{i}.png")
        raw_path.unlink()
    return pages


def launch_worker(worker_id: int, pages: list[int], image_dir: Path, log_path: Path) -> subprocess.Popen:
    """Spawn one worker pinned to a single N300 board."""
    env = os.environ.copy()
    env["TT_METAL_HOME"] = TT_METAL_HOME
    env["MESH_DEVICE"] = "N300"
    env["TT_SYMBIOTE_RUN_MODE"] = "TRACED"
    env["PYTHONUNBUFFERED"] = "1"
    env["WORKER_ID"] = str(worker_id)
    env["WORKER_PAGES"] = ",".join(str(p) for p in pages)
    env["TT_METAL_PCI_BUS_IDS"] = PCI_BUSES[worker_id]
    env["DOTS_OCR_IMAGE_DIR"] = str(image_dir)
    venv_python = f"{TT_METAL_HOME}/python_env/bin/python"
    cmd = [venv_python, "-u", "-m", "pytest", f"{TT_METAL_HOME}/{TEST_FILE}",
           "-xvs", "--timeout=0"]
    return subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


def run_ocr(pdf_path: Path, output_md: Path | None = None) -> dict:
    """Top-level orchestration: split, launch, gather, merge."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # Stage pages into a temp dir (workers read from a known location)
    image_dir = Path("/home/mlops/dll_project/ocr_lab")  # current worker hard-codes this
    print(f"Extracting pages from {pdf_path.name}...", flush=True)
    total = extract_pages(pdf_path, image_dir)
    assignments = split_pages(total)
    print(f"  {total} pages, distribution: {[len(a) for a in assignments]}", flush=True)

    # Launch workers in parallel
    t0 = time.time()
    log_dir = Path(tempfile.mkdtemp(prefix="ocr_dp_"))
    procs = []
    for wid, pages in enumerate(assignments):
        if not pages:
            procs.append(None)
            continue
        lp = log_dir / f"w{wid}.log"
        print(f"  W{wid} -> pages {pages} -> {lp}", flush=True)
        procs.append(launch_worker(wid, pages, image_dir, lp))

    # Wait
    for wid, p in enumerate(procs):
        if p is None:
            continue
        rc = p.wait()
        print(f"  W{wid} exit={rc}", flush=True)
    wall = time.time() - t0

    # Collect per-worker JSONs
    pages_data = {}
    for wid in range(NUM_WORKERS):
        jp = Path(f"/tmp/t1_dp_worker_{wid}.json")
        if jp.exists():
            d = json.loads(jp.read_text(encoding="utf-8"))
            for r in d["results"]:
                pages_data[r["page"]] = {**r, "worker": wid}

    missing = [i for i in range(1, total + 1) if i not in pages_data]

    # Write consolidated markdown
    if output_md is None:
        output_md = pdf_path.with_suffix(".ocr.md")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_path.name} OCR Output (dots.ocr / 4×N300 DP)\n\n")
        f.write(f"- **Wall time:** {wall:.1f}s\n")
        f.write(f"- **Pages:** {total} (recovered {len(pages_data)})\n")
        if missing:
            f.write(f"- **Missing:** {missing}\n")
        f.write("\n| Page | Worker | Time | Tokens |\n|---|---|---|---|\n")
        for p in sorted(pages_data):
            d = pages_data[p]
            f.write(f"| {p} | W{d['worker']} | {d['time_s']:.1f}s | {d['tokens']} |\n")
        f.write("\n---\n\n")
        for p in sorted(pages_data):
            d = pages_data[p]
            f.write(f"## Page {p} — {d['time_s']:.1f}s / {d['tokens']} tokens (worker {d['worker']})\n\n")
            f.write(d["text"].strip() + "\n\n---\n\n")

    print(f"Wrote {output_md} ({output_md.stat().st_size} bytes, wall={wall:.1f}s)", flush=True)
    return {"wall_s": wall, "pages": total, "missing": missing, "output": str(output_md)}


# --- HTTP server mode ---

def run_server(port: int = 8080) -> None:
    from flask import Flask, request, send_file, jsonify
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def home():
        return ("<h1>dots.ocr DP server</h1>"
                "<form method=post enctype=multipart/form-data action=/ocr>"
                "<input type=file name=pdf accept=.pdf required>"
                "<button>OCR</button></form>")

    @app.route("/ocr", methods=["POST"])
    def ocr_endpoint():
        f = request.files.get("pdf")
        if not f:
            return jsonify(error="missing pdf"), 400
        tmp = Path(tempfile.mkdtemp(prefix="ocr_srv_")) / f.filename
        f.save(tmp)
        out_md = tmp.with_suffix(".ocr.md")
        try:
            result = run_ocr(tmp, out_md)
        except Exception as e:
            return jsonify(error=str(e)), 500
        if request.args.get("format") == "json":
            return jsonify({**result, "text": out_md.read_text(encoding="utf-8")})
        return send_file(out_md, as_attachment=True, download_name=out_md.name)

    print(f"OCR server listening on :{port}", flush=True)
    app.run(host="0.0.0.0", port=port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", help="input PDF (CLI mode)")
    ap.add_argument("-o", "--output", help="output markdown path")
    ap.add_argument("--server", action="store_true", help="HTTP server mode")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    if args.server:
        run_server(args.port)
        return

    if not args.pdf:
        ap.error("either provide a PDF or --server")
    out = Path(args.output) if args.output else None
    run_ocr(Path(args.pdf), out)


if __name__ == "__main__":
    main()
