"""
moderation/pipeline.py — Main orchestration pipeline.

Coordinates pre-filter → gatekeeper → branch(es) → verdict aggregation.
All heavy inference is wrapped in asyncio.to_thread() so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from config import BotConfig

logger = logging.getLogger(__name__)

# Severity ordering for aggregation
_SEVERITY: dict[str, int] = {"SAFE": 0, "REVIEW": 1, "BLOCK": 2}


@dataclass
class ScanResult:
    verdict: str                           # "BLOCK" | "REVIEW" | "SAFE"
    reason: str                            # human-readable e.g. "MALE_GENITALIA_EXPOSED (0.73)"
    branch: str                            # "real" | "anime" | "prefilter_skip" | "both"
    model: str                             # which model made the final call
    frame_index: Optional[int]            # which frame triggered it (None for images)
    processing_time_ms: float


# ── Module-level singletons (lazy init) ───────────────────────────────────────
_prefilter = None
_gatekeeper = None
_real_branch = None
_anime_branch = None
_initialized = False


def _ensure_models_initialized(config: "BotConfig") -> None:
    """Initialize all model singletons on first call."""
    global _prefilter, _gatekeeper, _real_branch, _anime_branch, _initialized
    if _initialized:
        return

    from moderation.prefilter import AdamCoddPrefilter
    from moderation.gatekeeper import ContentTypeRouter
    from moderation.real_branch import RealBranch
    from moderation.anime_branch import AnimeBranch

    logger.info("[Pipeline] Initializing models (first-run)…")

    # Pre-filter loads eagerly (it's the only model that stays in memory)
    _prefilter = AdamCoddPrefilter(cache_dir=config.model_cache_dir)

    # Gatekeeper: load lazily but keep in memory (small model)
    try:
        _gatekeeper = ContentTypeRouter(cache_dir=config.model_cache_dir)
    except Exception as e:
        logger.error("[Pipeline] Gatekeeper failed to load: %s — will skip routing", e)
        _gatekeeper = None

    # Branches: instantiated now but internally lazy
    _real_branch = RealBranch()

    try:
        _anime_branch = AnimeBranch(cache_dir=config.model_cache_dir)
    except Exception as e:
        logger.error("[Pipeline] AnimeBranch failed to load: %s — anime scanning disabled", e)
        _anime_branch = None

    _initialized = True
    logger.info("[Pipeline] All models initialized")


# ── Attachment download ────────────────────────────────────────────────────────

async def _download_attachment(
    url: str,
    dest_path: str,
    max_size_mb: int,
) -> None:
    """
    Stream-download a Discord attachment to dest_path.
    Raises ValueError if file exceeds max_size_mb.
    """
    max_bytes = max_size_mb * 1024 * 1024
    downloaded = 0

    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise IOError(f"HTTP {resp.status} when downloading {url}")

            with open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(65536):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(
                            f"Attachment exceeds {max_size_mb} MB size limit "
                            f"({downloaded / 1048576:.1f} MB so far)"
                        )
                    f.write(chunk)

    logger.debug("[Pipeline] Downloaded %d bytes → %s", downloaded, dest_path)


# ── Sync inference helpers (run inside asyncio.to_thread) ─────────────────────

def _run_prefilter(image, threshold: float = 0.25) -> bool:
    """Returns True if the image is worth checking (score >= threshold)."""
    return _prefilter.is_worth_checking(image, threshold=threshold)


def _run_gatekeeper(image) -> tuple[str, float]:
    """Returns (route, confidence) from gatekeeper."""
    if _gatekeeper is None:
        return ("uncertain", 0.0)
    return _gatekeeper.route(image)


def _run_real_branch(image):
    """Run the real/photo branch and return BranchResult."""
    return _real_branch.scan(image)


def _run_anime_branch(image):
    """Run the anime branch and return BranchResult."""
    if _anime_branch is None:
        from moderation.real_branch import BranchResult
        return BranchResult(verdict="SAFE")
    return _anime_branch.scan(image)


def _severity(verdict: str) -> int:
    return _SEVERITY.get(verdict, 0)


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def scan_attachment(
    attachment_url: str,
    config: "BotConfig",
) -> ScanResult:
    """
    Full pipeline: download → frame extraction → prefilter → gatekeeper → branch(es) → verdict.

    Always deletes the temp file in a finally block.
    Returns ScanResult with verdict BLOCK | REVIEW | SAFE.
    """
    start_time = time.monotonic()

    # Ensure models are ready (lazy init on first call)
    await asyncio.to_thread(_ensure_models_initialized, config)

    tmp_path: Optional[str] = None
    suffix = _guess_suffix(attachment_url)

    try:
        # ── Download ──────────────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await _download_attachment(
                attachment_url, tmp_path, config.max_video_size_mb
            )
        except ValueError as e:
            # File too large
            elapsed = (time.monotonic() - start_time) * 1000
            return ScanResult(
                verdict="SAFE",
                reason=str(e),
                branch="size_limit",
                model="none",
                frame_index=None,
                processing_time_ms=elapsed,
            )
        except Exception as e:
            logger.warning("[Pipeline] Download failed for %s: %s", attachment_url, e)
            elapsed = (time.monotonic() - start_time) * 1000
            return ScanResult(
                verdict="SAFE",
                reason=f"Download error: {e}",
                branch="download_error",
                model="none",
                frame_index=None,
                processing_time_ms=elapsed,
            )

        # ── Frame extraction ──────────────────────────────────────────────────
        from moderation.frame_extractor import (
            extract_frames,
            FileTooLargeError,
            VideoTooLongError,
        )

        try:
            frames = await asyncio.to_thread(
                extract_frames,
                tmp_path,
                config.max_video_size_mb,
                config.max_video_duration_secs,
            )
        except (FileTooLargeError, VideoTooLongError) as e:
            elapsed = (time.monotonic() - start_time) * 1000
            return ScanResult(
                verdict="SAFE",
                reason=str(e),
                branch="limit_exceeded",
                model="none",
                frame_index=None,
                processing_time_ms=elapsed,
            )

        if not frames:
            elapsed = (time.monotonic() - start_time) * 1000
            return ScanResult(
                verdict="SAFE",
                reason="No frames could be extracted",
                branch="extract_error",
                model="none",
                frame_index=None,
                processing_time_ms=elapsed,
            )

        logger.info("[Pipeline] Processing %d frame(s) from %s", len(frames), attachment_url)

        # ── Per-frame scan ────────────────────────────────────────────────────
        best_result: Optional[ScanResult] = None

        for frame_idx, frame in enumerate(frames):
            # Stage 0: Pre-filter
            worth_checking = await asyncio.to_thread(_run_prefilter, frame)
            if not worth_checking:
                logger.debug("[Pipeline] Frame %d skipped by prefilter", frame_idx)
                continue

            # Stage 1: Gatekeeper
            route, confidence = await asyncio.to_thread(_run_gatekeeper, frame)
            logger.debug("[Pipeline] Frame %d → route=%s (%.2f)", frame_idx, route, confidence)

            # Stage 2: Branch(es)
            if route == "real":
                branch_result = await asyncio.to_thread(_run_real_branch, frame)
                branch_name = "real"
                model_name = branch_result.model

            elif route == "anime":
                branch_result = await asyncio.to_thread(_run_anime_branch, frame)
                branch_name = "anime"
                model_name = branch_result.model

            else:
                # Uncertain: run both, take higher severity
                real_res = await asyncio.to_thread(_run_real_branch, frame)
                anime_res = await asyncio.to_thread(_run_anime_branch, frame)

                if _severity(real_res.verdict) >= _severity(anime_res.verdict):
                    branch_result = real_res
                    branch_name = "real"
                else:
                    branch_result = anime_res
                    branch_name = "anime"
                model_name = branch_result.model

            # Build ScanResult for this frame
            reason = _build_reason(branch_result)
            frame_scan = ScanResult(
                verdict=branch_result.verdict,
                reason=reason,
                branch=branch_name,
                model=model_name,
                frame_index=frame_idx if len(frames) > 1 else None,
                processing_time_ms=0,
            )

            # Track best result
            if best_result is None or _severity(frame_scan.verdict) > _severity(best_result.verdict):
                best_result = frame_scan

            # Early exit on BLOCK
            if frame_scan.verdict == "BLOCK":
                logger.info(
                    "[Pipeline] BLOCK on frame %d — stopping early. Reason: %s",
                    frame_idx,
                    reason,
                )
                break

        elapsed = (time.monotonic() - start_time) * 1000

        if best_result is None:
            # All frames skipped by prefilter
            return ScanResult(
                verdict="SAFE",
                reason="All frames cleared prefilter",
                branch="prefilter_skip",
                model="prefilter",
                frame_index=None,
                processing_time_ms=elapsed,
            )

        best_result.processing_time_ms = elapsed
        logger.info(
            "[Pipeline] Final verdict: %s | branch=%s | model=%s | %.0fms",
            best_result.verdict,
            best_result.branch,
            best_result.model,
            elapsed,
        )
        return best_result

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug("[Pipeline] Temp file deleted: %s", tmp_path)
            except Exception as e:
                logger.warning("[Pipeline] Failed to delete temp file %s: %s", tmp_path, e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _guess_suffix(url: str) -> str:
    """Guess file extension from URL."""
    from urllib.parse import urlparse
    path = urlparse(url).path.lower()
    for ext in (".gif", ".mp4", ".webm", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".webp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def _build_reason(branch_result) -> str:
    """Build a human-readable reason string from a BranchResult."""
    if not branch_result.detections:
        return f"rating={getattr(branch_result, 'rating', 'unknown')} (score={branch_result.max_score:.2f})"

    top = sorted(branch_result.detections, key=lambda x: x[1], reverse=True)
    parts = [f"{label} ({score:.2f})" for label, score, _ in top[:3]]
    return ", ".join(parts)
