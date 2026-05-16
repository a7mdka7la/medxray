"""Image loading helpers.

Kept tiny on purpose. Chest X-rays come in lots of formats (PNG/JPG, DICOM,
sometimes grayscale, sometimes 16-bit). For this assignment we stick to the
PNG/JPG outputs from MIMIC's `files/` directory.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

ImageLike = Union[str, Path, bytes, Image.Image]


def to_pil(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_keep_aspect(img: Image.Image, max_side: int = 896) -> Image.Image:
    """Resize so the longest side is `max_side`, keep aspect ratio.

    Most VLMs we use prefer ~896px and choke or get slow on full-resolution
    DICOM exports.
    """
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = int(h * max_side / w)
    else:
        new_h = max_side
        new_w = int(w * max_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)
