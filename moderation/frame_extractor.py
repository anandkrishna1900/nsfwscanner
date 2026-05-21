"""
moderation/frame_extractor.py — Extract frames from images, GIFs, and videos.

Returns a list of RGB PIL Images.  Raises FileTooLargeError or VideoTooLongError
if size/duration limits are exceeded.  All methods are synchronous.

Frame extraction strategy by file type:
  - Static images (.jpg, .png, .webp, etc.): single frame
  - GIF (.gif):  ALL frames  — GIFs are short by design; full scan is fast
  - Video (.mp4, .avi, .mov, .webm, .mkv, .flv):
        Only the FIRST 30 SECONDS are scanned, at 1 frame every 3 seconds.
        This gives a maximum of 10 frames per video regardless of length.
        If the video is shorter than 30 seconds the whole file is scanned
        at the same rate (fewer than 10 frames — that's fine).
        Nothing beyond the 30-second mark is ever read or decoded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)

# Supported extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
GIF_EXTS   = {".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv"}

# Video scanning constants — hardcoded, not configurable
_VIDEO_MAX_SECS: int   = 30   # never scan past this timestamp
_VIDEO_FRAME_INTERVAL: float = 3.0   # one frame every 3 seconds → max 10 frames


# ── Custom exceptions ─────────────────────────────────────────────────────────

class FileTooLargeError(Exception):
    """Raised when the file exceeds the configured size limit."""


class VideoTooLongError(Exception):
    """Raised when the video exceeds the configured duration limit."""


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_frames(
    file_path: str,
    max_size_mb: int = 50,
    max_duration_secs: int = 300,
) -> List[Image.Image]:
    """
    Extract frames from the given file and return them as RGB PIL Images.

    Routing:
      - Static image  → [single PIL Image]
      - GIF           → all frames
      - Video         → first 30 s at 1 frame / 3 s  (max 10 frames)

    Args:
        file_path:         Absolute or relative path to the media file.
        max_size_mb:       Reject file if it exceeds this size in MB.
        max_duration_secs: Reject video if its total duration exceeds this
                           value (checked before decoding; NOT a scan cap —
                           the 30-second scan cap is separate and always applied).

    Raises:
        FileTooLargeError:  File size exceeds max_size_mb.
        VideoTooLongError:  Video duration exceeds max_duration_secs.

    Returns:
        List of RGB PIL Images (may be empty if the file is unreadable).
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    # ── Size guard ────────────────────────────────────────────────────────────
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise FileTooLargeError(
                f"File is {size_mb:.1f} MB, which exceeds the {max_size_mb} MB limit."
            )
    except FileTooLargeError:
        raise
    except Exception as e:
        logger.warning("Could not check file size for %s: %s", file_path, e)

    # ── Route by extension ────────────────────────────────────────────────────
    if ext in IMAGE_EXTS:
        return _extract_image(file_path)
    elif ext in GIF_EXTS:
        return _extract_gif(file_path)
    elif ext in VIDEO_EXTS:
        return _extract_video(file_path, max_duration_secs)
    else:
        # Unknown extension — attempt to open as a static image
        logger.debug("Unknown extension %r — attempting image open", ext)
        return _extract_image(file_path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL Image mode to RGB."""
    return img.convert("RGB") if img.mode != "RGB" else img


def _extract_image(file_path: str) -> List[Image.Image]:
    """Open a static image and return it as a single-element list."""
    try:
        img = Image.open(file_path)
        img.load()  # force decode so the file handle can be released
        return [_to_rgb(img)]
    except Exception as e:
        logger.warning("Failed to open image %s: %s", file_path, e)
        return []


def _extract_gif(file_path: str) -> List[Image.Image]:
    """
    Extract ALL frames from a GIF.

    GIFs are short by design (typically < 50 frames), so a full scan
    is both fast and the correct behaviour.
    """
    try:
        import imageio.v3 as iio
        import numpy as np

        frames: List[Image.Image] = []
        for frame_array in iio.imiter(file_path, plugin="pillow"):
            img = Image.fromarray(frame_array.astype("uint8"))
            frames.append(_to_rgb(img))

        logger.debug("GIF: extracted %d frame(s) from %s", len(frames), file_path)
        return frames

    except Exception as e:
        logger.warning("Failed to extract GIF frames from %s: %s", file_path, e)
        return []


def _extract_video(file_path: str, max_duration_secs: int) -> List[Image.Image]:
    """
    Extract frames from the FIRST 30 SECONDS of a video at 1 frame per 3 seconds.

    Maximum frames returned: 10  (30 s ÷ 3 s/frame)
    If the video is shorter than 30 s, the entire file is scanned at the
    same rate, yielding fewer than 10 frames — that's expected and fine.
    No frame at or beyond the 30-second mark is ever decoded.

    The total-duration guard (max_duration_secs) is checked first and
    raises VideoTooLongError if the video is too long overall — this is
    separate from the 30-second scan cap.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.warning("cv2 could not open video: %s", file_path)
            return []

        video_fps: float  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count: int  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration: float = frame_count / video_fps if video_fps > 0 else 0.0

        # Total-duration guard (e.g. reject videos > 5 minutes)
        if total_duration > max_duration_secs:
            cap.release()
            raise VideoTooLongError(
                f"Video is {total_duration:.0f}s, which exceeds the "
                f"{max_duration_secs}s limit."
            )

        # Scan cap: never go past 30 seconds
        scan_end_secs: float = min(total_duration, float(_VIDEO_MAX_SECS))

        # How many video frames between each captured sample
        frame_interval: int = max(1, int(video_fps * _VIDEO_FRAME_INTERVAL))

        # The frame index at which we must stop
        stop_frame: int = int(scan_end_secs * video_fps)

        frames: List[Image.Image] = []
        frame_idx: int = 0

        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            # Stop once we've passed the 30-second mark
            if frame_idx >= stop_frame:
                break

            if frame_idx % frame_interval == 0:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                frames.append(_to_rgb(Image.fromarray(rgb_frame)))

            frame_idx += 1

        cap.release()

        logger.debug(
            "Video: extracted %d frame(s) from first %.1fs of %s "
            "(total duration %.1fs, interval %.1fs)",
            len(frames),
            scan_end_secs,
            file_path,
            total_duration,
            _VIDEO_FRAME_INTERVAL,
        )
        return frames

    except VideoTooLongError:
        raise
    except Exception as e:
        logger.warning("Failed to extract video frames from %s: %s", file_path, e)
        return []
