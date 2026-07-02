

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
                async for chunk in resp.content.iter_chunked(65536):
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
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image.convert("RGBA"))
        img = canvas.convert("RGB")
    else:
        img = image.convert("RGB")

    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2

    padded_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_img.paste(img, (pad_left, pad_top))

    if max_dim != target_size:
        padded_img = padded_img.resize((target_size, target_size), Image.BICUBIC)

    arr = np.array(padded_img, dtype=np.float32) / 255.0

    if to_bgr:
        arr = arr[:, :, ::-1]

    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

    if to_chw:
        arr = arr.transpose(2, 0, 1)

    return np.expand_dims(arr, axis=0).copy()


def generate_placeholder_image() -> bytes:
    """
    Generate a highly polished, premium-looking 300x150 dark-mode placeholder image
    with a vibrant red warning header accent and bold text indicating that the preview
    is unavailable due to flagged NSFW content.
    """
    import io
    from PIL import ImageDraw, ImageFont

    width, height = 300, 150
    # Create dark slate canvas
    img = Image.new("RGB", (width, height), color=(26, 27, 38)) # Rich Slate Dark
    draw = ImageDraw.Draw(img)
    
    # Draw a 5px high warning red bar at the top
    draw.rectangle([0, 0, width, 5], fill=(239, 68, 68))
    
    text_line1 = "⚠️ PREVIEW UNAVAILABLE"
    text_line2 = "[ FLAGGED NSFW ]"
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 14)
        font_sub = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        
    try:
        bbox1 = draw.textbbox((0, 0), text_line1, font=font_title)
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        bbox2 = draw.textbbox((0, 0), text_line2, font=font_sub)
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    except AttributeError:
        w1, h1 = draw.textsize(text_line1, font=font_title)
        w2, h2 = draw.textsize(text_line2, font=font_sub)
        
    x1 = (width - w1) // 2
    y1 = (height - h1 - h2 - 12) // 2
    
    x2 = (width - w2) // 2
    y2 = y1 + h1 + 12
    
    draw.text((x1, y1), text_line1, fill=(255, 255, 255), font=font_title)
    draw.text((x2, y2), text_line2, fill=(239, 68, 68), font=font_sub)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

