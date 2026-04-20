from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_A4_WIDTH = 672
DEFAULT_A4_HEIGHT = 952


@dataclass(frozen=True)
class FixedPageMeta:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scaled_width: int
    scaled_height: int
    pad_left: int
    pad_top: int
    scale: float


def _load_bgr_image(image: str | Path | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    image_path = str(Path(image).resolve())
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    return bgr


def fixed_page_image(
    image: str | Path | Image.Image,
    *,
    width: int = DEFAULT_A4_WIDTH,
    height: int = DEFAULT_A4_HEIGHT,
    background: int = 255,
) -> tuple[Image.Image, FixedPageMeta]:
    bgr = _load_bgr_image(image)
    src_h, src_w = bgr.shape[:2]
    scale = min(width / src_w, height / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=interpolation)

    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    pad_left = (width - scaled_w) // 2
    pad_top = (height - scaled_h) // 2
    canvas[pad_top : pad_top + scaled_h, pad_left : pad_left + scaled_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    meta = FixedPageMeta(
        source_width=src_w,
        source_height=src_h,
        target_width=width,
        target_height=height,
        scaled_width=scaled_w,
        scaled_height=scaled_h,
        pad_left=pad_left,
        pad_top=pad_top,
        scale=scale,
    )
    return image_pil, meta


def fixed_page_message(
    image: str | Path | Image.Image,
    *,
    width: int = DEFAULT_A4_WIDTH,
    height: int = DEFAULT_A4_HEIGHT,
    background: int = 255,
) -> tuple[dict, dict]:
    image_pil, meta = fixed_page_image(image, width=width, height=height, background=background)
    return {
        "type": "image",
        "image": image_pil,
        "resized_width": width,
        "resized_height": height,
    }, {
        "source_width": meta.source_width,
        "source_height": meta.source_height,
        "target_width": meta.target_width,
        "target_height": meta.target_height,
        "scaled_width": meta.scaled_width,
        "scaled_height": meta.scaled_height,
        "pad_left": meta.pad_left,
        "pad_top": meta.pad_top,
        "scale": round(meta.scale, 6),
    }
