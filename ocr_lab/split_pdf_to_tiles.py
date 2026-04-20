#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def render_pdf_page(pdf_path: Path, page: int, dpi: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{pdf_path.stem}_page_{page}"
    cmd = [
        "pdftoppm",
        "-jpeg",
        "-r",
        str(dpi),
        "-f",
        str(page),
        "-singlefile",
        str(pdf_path),
        str(prefix),
    ]
    subprocess.run(cmd, check=True)
    return prefix.with_suffix(".jpg")


def split_image(image_path: Path, rows: int, cols: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    tile_w = width // cols
    tile_h = height // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            top = r * tile_h
            right = width if c == cols - 1 else (c + 1) * tile_w
            bottom = height if r == rows - 1 else (r + 1) * tile_h
            tile = img.crop((left, top, right, bottom))
            tile_path = out_dir / f"{image_path.stem}_r{r}_c{c}.jpg"
            tile.save(tile_path, quality=95)
            tiles.append(
                {
                    "row": r,
                    "col": c,
                    "bbox": [left, top, right, bottom],
                    "path": str(tile_path),
                    "size": [right - left, bottom - top],
                }
            )

    manifest = {
        "source_image": str(image_path),
        "rows": rows,
        "cols": cols,
        "image_size": [width, height],
        "tiles": tiles,
    }
    manifest_path = out_dir / f"{image_path.stem}_tiles.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    out_dir = Path(args.out_dir).resolve()
    render_dir = out_dir / "rendered"
    tile_dir = out_dir / "tiles"

    rendered = render_pdf_page(pdf_path, args.page, args.dpi, render_dir)
    manifest = split_image(rendered, args.rows, args.cols, tile_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
