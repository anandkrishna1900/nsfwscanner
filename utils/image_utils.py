"""
utils/image_utils.py — Discord attachment download utilities.

Provides async streaming download with size limit enforcement.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import aiohttp

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
