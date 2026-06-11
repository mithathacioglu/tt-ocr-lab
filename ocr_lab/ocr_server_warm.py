#!/usr/bin/env python3
"""Warm persistent dots.ocr server.

Loads the TT pipeline once at startup and keeps it in memory. Subsequent
PDF requests skip the ~3min cold setup and only pay decode time per page.

Architecture:
  - Main process opens N300 mesh + loads dots.ocr pipeline + warmup once
  - Flask HTTP server accepts POST /ocr with a PDF file
  - For each request, pages are extracted and pipeline.generate() is called
  - Result returned as markdown (or JSON with ?format=json)

This holds device + ~3GB model in process memory for the lifetime of the server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Force tt_metal home + env before imports
TT_METAL_HOME = "/home/mlops/dll_project/tt-metal-fresh"
os.environ["TT_METAL_HOME"] = TT_METAL_HOME
os.environ.setdefault("TT_SYMBIOTE_RUN_MODE", "TRACED")
os.environ.setdefault("MESH_DEVICE", "N300")
sys.path.insert(0, TT_METAL_HOME)

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
TARGET_W, TARGET_H = 1240, 1754
PROMPT = "Please output the exact text in the image.\n\nReturn plain text only.\n"
MAX_NEW_TOKENS = 2000


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download
    return snapshot_download(DOTS_OCR_MODEL_ID)


class WarmPipeline:
    """Holds the dots.ocr pipeline + processor in memory across requests."""

    def __init__(self):
        import torch
        import ttnn
        from transformers import (
            AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor,
        )
        from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline

        self.torch = torch
        self.ttnn = ttnn

        model_path = _resolve_model_path()
        print(f"[server] model_path={model_path}", flush=True)

        # Set fabric config BEFORE opening mesh device (1D ring for N300 TP=2)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        except Exception as e:
            print(f"[server] fabric_config note: {e}", flush=True)
        print("[server] opening mesh device (1,2) ...", flush=True)
        self.mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 2),
            trace_region_size=300_000_000,
            num_command_queues=1,
        )

        # Load pipeline + processor
        print("[server] loading TTNNDotsOCRPipeline ...", flush=True)
        self.pipeline = TTNNDotsOCRPipeline.from_hf_model(
            model_path=model_path, device=self.mesh,
        )

        image_processor = AutoImageProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        video_processor = AutoVideoProcessor.from_pretrained(model_path)
        with open(os.path.join(model_path, "chat_template.json")) as f:
            chat_template = json.load(f)["chat_template"]
        self.processor = Qwen2_5_VLProcessor(
            image_processor, tokenizer, video_processor, chat_template=chat_template,
        )
        self.processor.image_token = "<|imgpad|>"
        self.processor.image_token_id = 151665
        print("[server] processor ready", flush=True)

        # Warmup using a dummy image of TARGET_W x TARGET_H
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        print("[server] warming up trace (dummy image)...", flush=True)
        dummy = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
        dpath = "/tmp/_warm_dummy.png"
        dummy.save(dpath)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": dpath},
            {"type": "text", "text": PROMPT},
        ]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(msgs)
        inputs = self.processor(text=[text], images=img_in, videos=vid_in,
                                padding=True, return_tensors="pt")
        self.pipeline.warmup(inputs["input_ids"],
                             pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                             image_grid_thw=inputs["image_grid_thw"])
        print("[server] warmup done — READY", flush=True)

    def _force_cleanup(self):
        """Lightweight cleanup between pages.

        Do NOT deallocate _decode_cache_position — the captured trace
        references its DRAM address and freeing it corrupts the trace.
        Just synchronize device and run Python GC.
        """
        import gc
        try:
            self.ttnn.synchronize_device(self.mesh)
        except Exception:
            pass
        gc.collect()

    def ocr_image(self, image_path: str) -> tuple[str, float, int]:
        """Run OCR on a single image. Returns (text, elapsed_s, num_tokens)."""
        import time
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": PROMPT},
        ]}]
        text_prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = self.processor(text=[text_prompt], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt")

        t0 = time.time()
        generated = self.pipeline.generate(
            inputs["input_ids"],
            pixel_values=inputs["pixel_values"].to(self.torch.bfloat16),
            image_grid_thw=inputs["image_grid_thw"],
            max_new_tokens=MAX_NEW_TOKENS,
        )
        self.ttnn.synchronize_device(self.mesh)
        elapsed = time.time() - t0
        decoded = self.processor.decode(generated, skip_special_tokens=True)
        return decoded, elapsed, len(generated)


def split_pdf_pages(pdf_path: Path, out_dir: Path) -> list[Path]:
    """Render PDF pages to PNGs at TARGET_W x TARGET_H."""
    import subprocess
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    info = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    n_pages = next(int(l.split(":")[1].strip()) for l in info.splitlines() if l.startswith("Pages:"))
    paths = []
    for i in range(1, n_pages + 1):
        raw = out_dir / f"raw-{i}.png"
        subprocess.check_call(
            ["pdftoppm", "-f", str(i), "-l", str(i), str(pdf_path),
             str(out_dir / "raw"), "-png"],
            stdout=subprocess.DEVNULL,
        )
        img = Image.open(raw)
        img.resize((TARGET_W, TARGET_H), Image.LANCZOS).save(out_dir / f"page{i}.png")
        raw.unlink()
        paths.append(out_dir / f"page{i}.png")
    return paths


def run_pdf(warm: WarmPipeline, pdf_path: Path) -> dict:
    """Process a PDF and return per-page results."""
    work_dir = Path(tempfile.mkdtemp(prefix="ocr_warm_"))
    pages = split_pdf_pages(pdf_path, work_dir)
    print(f"[server] processing {len(pages)} pages from {pdf_path.name}", flush=True)
    results = []
    t_total = time.time()
    for i, p in enumerate(pages, 1):
        text, t_s, n_tok = warm.ocr_image(str(p))
        print(f"[server]   page {i}: {t_s:.2f}s, {n_tok} tok", flush=True)
        results.append({"page": i, "time_s": t_s, "tokens": n_tok, "text": text})
        # Force cleanup between pages to combat state-degradation bug
        warm._force_cleanup()
    total = time.time() - t_total
    print(f"[server] total {len(pages)} pages: {total:.2f}s", flush=True)
    return {"pdf": pdf_path.name, "total_s": total, "pages": results}


def to_markdown(result: dict) -> str:
    lines = [f"# {result['pdf']} OCR (warm dots.ocr)", "",
             f"**Total wall time:** {result['total_s']:.2f}s",
             f"**Pages:** {len(result['pages'])}", "",
             "| Page | Time | Tokens |", "|---|---|---|"]
    for p in result["pages"]:
        lines.append(f"| {p['page']} | {p['time_s']:.1f}s | {p['tokens']} |")
    lines.append("\n---\n")
    for p in result["pages"]:
        lines.append(f"## Page {p['page']} — {p['time_s']:.1f}s / {p['tokens']} tokens\n")
        lines.append(p["text"].strip())
        lines.append("\n---\n")
    return "\n".join(lines)


def main_server(port: int = 8080):
    from flask import Flask, request, send_file, jsonify

    warm = WarmPipeline()
    app = Flask(__name__)

    @app.get("/")
    def home():
        return ("<h1>dots.ocr warm server</h1>"
                "<p>Pipeline loaded and ready.</p>"
                "<form method=post enctype=multipart/form-data action=/ocr>"
                "<input type=file name=pdf accept=.pdf required>"
                "<button>OCR</button></form>")

    @app.post("/ocr")
    def ocr():
        f = request.files.get("pdf")
        if not f:
            return jsonify(error="missing pdf"), 400
        tmp = Path(tempfile.mkdtemp(prefix="ocr_req_")) / f.filename
        f.save(tmp)
        result = run_pdf(warm, tmp)
        md = to_markdown(result)
        if request.args.get("format") == "json":
            return jsonify({**result, "markdown": md})
        out_md = tmp.with_suffix(".ocr.md")
        out_md.write_text(md, encoding="utf-8")
        return send_file(out_md, as_attachment=True, download_name=out_md.name)

    @app.get("/health")
    def health():
        return jsonify(status="ready")

    print(f"[server] listening on :{port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=False)  # single-threaded so we don't share device


def main_cli(pdf: Path, output: Path | None):
    warm = WarmPipeline()
    result = run_pdf(warm, pdf)
    md = to_markdown(result)
    out = output or pdf.with_suffix(".ocr.md")
    out.write_text(md, encoding="utf-8")
    print(f"[server] wrote {out} ({out.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("-o", "--output")
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    if args.server:
        main_server(args.port)
    elif args.pdf:
        main_cli(Path(args.pdf), Path(args.output) if args.output else None)
    else:
        ap.error("pass a PDF or --server")
