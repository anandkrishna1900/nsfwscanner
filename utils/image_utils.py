"""
utils/image_utils.py — Discord attachment download utilities.

Provides async streaming download with size limit enforcement.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import aiohttp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


async def download_attachment(
    url: str,
    dest_path: str,
    max_size_mb: int = 50,
) -> str:
    """
    Stream-download a Discord attachment to dest_path.

    Args:
        url:         The attachment URL.
        dest_path:   Local file path to write to.
        max_size_mb: Maximum allowed file size in MB.

    Returns:
        dest_path on success.

    Raises:
        ValueError: If the file exceeds max_size_mb.
        IOError:    If the download fails (non-200 status).
        aiohttp.ClientError: On connection errors.
    """
    max_bytes = max_size_mb * 1024 * 1024
    downloaded = 0

    timeout = aiohttp.ClientTimeout(total=60, connect=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise IOError(
                    f"Failed to download attachment: HTTP {resp.status} from {url}"
                )

            with open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(65536):  # 64 KB chunks
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(
                            f"File too large: exceeds {max_size_mb} MB limit "
                            f"({downloaded / 1048576:.1f} MB downloaded so far)"
                        )
                    f.write(chunk)

    logger.debug("Downloaded %d bytes → %s", downloaded, dest_path)
    return dest_path


def guess_extension(filename: str, content_type: Optional[str] = None) -> str:
    """
    Guess a file extension from filename or content_type.

    Returns a dot-prefixed extension string, e.g. ".jpg".
    Falls back to ".jpg" if nothing matches.
    """
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".webm", ".mov", ".avi"):
            return ext

    if content_type:
        ct_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            "video/quicktime": ".mov",
        }
        for ct, ext in ct_map.items():
            if content_type.startswith(ct):
                return ext

    return ".jpg"


def prepare_image_for_onnx(
    image: Image.Image,
    target_size: int,
    normalize_imagenet: bool = False,
    to_chw: bool = False,
    to_bgr: bool = False,
) -> np.ndarray:
    """
    Robust preprocessing for ONNX models:
    - Handles RGBA transparency by alpha compositing onto a white background.
    - Pads the image to a square while preserving aspect ratio (avoids stretching/distortion).
    - Resizes to target_size using PIL.Image.BICUBIC.
    - Converts to a float32 numpy array normalized to [0.0, 1.0].
    - Optionally swaps RGB to BGR channels.
    - Optionally applies ImageNet normalization (mean and std).
    - Optionally transposes from HWC to CHW format.
    - Returns a 4D tensor with batch dimension (shape [1, C, H, W] or [1, H, W, C]).
    """
    # 1. Handle transparency
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        img = canvas.convert("RGB")
    else:
        img = image.convert("RGB")

    # 2. Pad to square preserving aspect ratio
    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2

    padded_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_img.paste(img, (pad_left, pad_top))

    # 3. Resize to target_size
    if max_dim != target_size:
        padded_img = padded_img.resize((target_size, target_size), Image.BICUBIC)

    # 4. Convert to float32 numpy array [0.0, 1.0]
    arr = np.array(padded_img, dtype=np.float32) / 255.0

    # 5. Swap RGB to BGR if requested
    if to_bgr:
        arr = arr[:, :, ::-1]

    # 6. Apply ImageNet normalization if requested
    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

    # 7. Transpose to CHW if requested
    if to_chw:
        arr = arr.transpose(2, 0, 1)

    # 8. Add batch dimension
    return np.expand_dims(arr, axis=0).copy()
