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
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from config import BotConfig

logger = logging.getLogger(__name__)

# Severity ordering for aggregation
_SEVERITY: dict[str, int] = {
    "SAFE": 0,
    "SUGGESTIVE": 1,
    "REVIEW": 1,
    "NSFW": 2,
    "BLOCK": 2,
    "EXPLICIT": 3,
}


@dataclass
class ScanResult:
    verdict: str                           # "SAFE" | "SUGGESTIVE" | "REVIEW" | "NSFW" | "BLOCK" | "EXPLICIT"
    reason: str                            # human-readable e.g. "MALE_GENITALIA_EXPOSED (0.73)"
    branch: str                            # "real" | "anime" | "prefilter_skip" | "both"
    model: str                             # which model made the final call
    frame_index: Optional[int]            # which frame triggered it (None for images)
    processing_time_ms: float
    pipeline_steps: list[str] = field(default_factory=list)


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
    bypass_prefilter: bool = False,
) -> ScanResult:
    """
    Full pipeline: download → frame extraction → prefilter → gatekeeper → branch(es) → verdict.

    Always deletes the temp file in a finally block.
    Returns ScanResult with verdict SAFE | SUGGESTIVE | REVIEW | NSFW | BLOCK | EXPLICIT.
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
                pipeline_steps=[
                    "Limit Check:\n"
                    "  Status: Rejected\n"
                    f"  Reason: {str(e)}"
                ]
            )
        except Exception as e:
            logger.warning("[Pipeline] Download failed for %s: %s", attachment_url, e)
            # Re-raise so the caller (on_message) can skip this URL instead of
            # treating a stale/expired/blocked link as SAFE.
            raise IOError(f"Download error: {e}") from e

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
                pipeline_steps=[
                    "Frame Extraction Stage:\n"
                    "  Status: Rejected\n"
                    f"  Reason: {str(e)}"
                ]
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
                pipeline_steps=[
                    "Frame Extraction Stage:\n"
                    "  Status: Failed\n"
                    "  Reason: Zero frames could be parsed from media"
                ]
            )

        logger.info("[Pipeline] Processing %d frame(s) from %s", len(frames), attachment_url)

        # ── Per-frame scan ────────────────────────────────────────────────────
        best_result: Optional[ScanResult] = None

        for frame_idx, frame in enumerate(frames):
            steps = []
            w, h = frame.size

            # Stage 0: Pre-filter
            prefilter_score = await asyncio.to_thread(_prefilter.score, frame)
            worth_checking = prefilter_score >= 0.25 or bypass_prefilter

            verdict_text = "Passed to Gatekeeper Router"
            if not (prefilter_score >= 0.25):
                if bypass_prefilter:
                    verdict_text = "Passed to Gatekeeper Router (Forced via Test Mode)"
                else:
                    verdict_text = "Approved (SAFE) - Skipping deeper analysis"

            steps.append(
                f"Stage 0: Pre-filter\n"
                f"  Model: AdamCodd/vit-base-nsfw-detector (ONNX CPU)\n"
                f"  Resolution: {w}x{h}\n"
                f"  NSFW Score: {prefilter_score:.4f}\n"
                f"  Threshold: 0.25\n"
                f"  Verdict: {verdict_text}"
            )

            if not worth_checking:
                logger.debug("[Pipeline] Frame %d skipped by prefilter", frame_idx)
                frame_scan = ScanResult(
                    verdict="SAFE",
                    reason=f"Cleared prefilter (score={prefilter_score:.4f})",
                    branch="prefilter_skip",
                    model="prefilter",
                    frame_index=frame_idx if len(frames) > 1 else None,
                    processing_time_ms=0,
                    pipeline_steps=steps
                )
                if best_result is None or _severity(frame_scan.verdict) > _severity(best_result.verdict):
                    best_result = frame_scan
                continue

            # Stage 1: Gatekeeper
            route, confidence = await asyncio.to_thread(_run_gatekeeper, frame)
            logger.debug("[Pipeline] Frame %d → route=%s (%.2f)", frame_idx, route, confidence)

            steps.append(
                f"Stage 1: Gatekeeper Router\n"
                f"  Model: deepghs/anime_real_cls (ONNX CPU)\n"
                f"  Classification: {route}\n"
                f"  Confidence: {confidence:.2f}\n"
                f"  Action: " + ("Routed to Real/Photo Branch" if route == "real" else "Routed to Anime/Illustration Branch" if route == "anime" else "Uncertain - Routing to BOTH branches")
            )

            # Stage 2: Branch(es)
            if route == "real":
                branch_result = await asyncio.to_thread(_run_real_branch, frame)
                branch_name = "real"
                model_name = branch_result.model

                detections_str = "None found"
                if branch_result.detections:
                    detections_str = "\n" + "\n".join(f"    - {label}: {score:.2f}" for label, score, _ in branch_result.detections)

                steps.append(
                    f"Stage 2A: Real/Photo Branch\n"
                    f"  Model: NudeNet v3 (ONNX CPU)\n"
                    f"  Target Explicit Labels: FEMALE_GENITALIA_EXPOSED, MALE_GENITALIA_EXPOSED, ANUS_EXPOSED, MALE_GENITALIA_COVERED, FEMALE_BREAST_EXPOSED\n"
                    f"  Detected Explicit Labels: {detections_str}\n"
                    f"  Verdict: {branch_result.verdict}"
                )

            elif route == "anime":
                branch_result = await asyncio.to_thread(_run_anime_branch, frame)
                branch_name = "anime"
                model_name = branch_result.model

                tags_str = "None found"
                if branch_result.detections:
                    tags_str = "\n" + "\n".join(f"    - {tag}: {score:.2f}" for tag, score, _ in branch_result.detections)

                wdv3_exp = getattr(branch_result, 'wdv3_explicit', 0.0)
                ar_r18 = getattr(branch_result, 'anime_rating_r18', 0.0)
                gen_sc = getattr(branch_result, 'genital_score', 0.0)
                breast_sc = getattr(branch_result, 'breast_score', 0.0)
                sugg_sc = getattr(branch_result, 'suggestive_score', 0.0)
                fin_sc = getattr(branch_result, 'final_score', 0.0)

                steps.append(
                    f"Stage 2B: Anime/Hentai Branch (Score Fusion)\n"
                    f"  Models:\n"
                    f"    - SmilingWolf/wd-vit-large-tagger-v3 (ONNX CPU)\n"
                    f"    - deepghs/anime_rating (ONNX CPU)\n"
                    f"  Scores & Fusion Math:\n"
                    f"    - wdv3_explicit: {wdv3_exp:.4f} (weight: 0.30)\n"
                    f"    - anime_rating_r18: {ar_r18:.4f} (weight: 0.25)\n"
                    f"    - genital_score: {gen_sc:.4f} (weight: 0.30)\n"
                    f"    - breast_score: {breast_sc:.4f} (weight: 0.15)\n"
                    f"    - suggestive_score: {sugg_sc:.4f}\n"
                    f"    - Formula: ({wdv3_exp:.4f} * 0.30) + ({ar_r18:.4f} * 0.25) + ({gen_sc:.4f} * 0.30) + ({breast_sc:.4f} * 0.15) = {fin_sc:.4f}\n"
                    f"  Decision Tiers:\n"
                    f"    - EXPLICIT: genital_score >= 0.80\n"
                    f"    - NSFW: breast_score >= 0.75\n"
                    f"    - SUGGESTIVE: fused score >= 0.65\n"
                    f"    - SAFE: otherwise\n"
                    f"  Detected Explicit Tags: {tags_str}\n"
                    f"  Tagger Rating: {getattr(branch_result, 'rating', 'unknown')}\n"
                    f"  Verdict: {branch_result.verdict}"
                )

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

                real_detections_str = "None found"
                if real_res.detections:
                    real_detections_str = "\n" + "\n".join(f"      - {label}: {score:.2f}" for label, score, _ in real_res.detections)

                anime_tags_str = "None found"
                if anime_res.detections:
                    anime_tags_str = "\n" + "\n".join(f"      - {tag}: {score:.2f}" for tag, score, _ in anime_res.detections)

                anime_wdv3_exp = getattr(anime_res, 'wdv3_explicit', 0.0)
                anime_ar_r18 = getattr(anime_res, 'anime_rating_r18', 0.0)
                anime_gen_sc = getattr(anime_res, 'genital_score', 0.0)
                anime_breast_sc = getattr(anime_res, 'breast_score', 0.0)
                anime_sugg_sc = getattr(anime_res, 'suggestive_score', 0.0)
                anime_fin_sc = getattr(anime_res, 'final_score', 0.0)

                steps.append(
                    f"Stage 2: Uncertain (Both Branches Evaluated)\n"
                    f"  Real/Photo Branch:\n"
                    f"    Model: NudeNet v3 (ONNX CPU)\n"
                    f"    Detections: {real_detections_str}\n"
                    f"    Verdict: {real_res.verdict}\n"
                    f"  Anime/Illustration Branch:\n"
                    f"    Models: SmilingWolf/wd-vit-large-tagger-v3 & deepghs/anime_rating (ONNX CPU)\n"
                    f"    Scores & Fusion Math:\n"
                    f"      - wdv3_explicit: {anime_wdv3_exp:.4f} (weight: 0.30)\n"
                    f"      - anime_rating_r18: {anime_ar_r18:.4f} (weight: 0.25)\n"
                    f"      - genital_score: {anime_gen_sc:.4f} (weight: 0.30)\n"
                    f"      - breast_score: {anime_breast_sc:.4f} (weight: 0.15)\n"
                    f"      - suggestive_score: {anime_sugg_sc:.4f}\n"
                    f"      - Formula: ({anime_wdv3_exp:.4f} * 0.30) + ({anime_ar_r18:.4f} * 0.25) + ({anime_gen_sc:.4f} * 0.30) + ({anime_breast_sc:.4f} * 0.15) = {anime_fin_sc:.4f}\n"
                    f"    Tagger Rating: {getattr(anime_res, 'rating', 'unknown')}\n"
                    f"    Detected Explicit Tags: {anime_tags_str}\n"
                    f"    Verdict: {anime_res.verdict}\n"
                    f"  Decision Policy: Chosen higher severity verdict\n"
                    f"  Selected Branch: {branch_name}\n"
                    f"  Final Model: {model_name}\n"
                    f"  Verdict: {branch_result.verdict}"
                )

            # Build ScanResult for this frame
            reason = _build_reason(branch_result)
            frame_scan = ScanResult(
                verdict=branch_result.verdict,
                reason=reason,
                branch=branch_name,
                model=model_name,
                frame_index=frame_idx if len(frames) > 1 else None,
                processing_time_ms=0,
                pipeline_steps=steps
            )

            # Track best result
            if best_result is None or _severity(frame_scan.verdict) > _severity(best_result.verdict):
                best_result = frame_scan

            # Early exit on BLOCK
            if frame_scan.verdict in {"BLOCK", "NSFW", "EXPLICIT"}:
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
                pipeline_steps=[
                    "Pipeline Run:\n"
                    "  Verdict: SAFE\n"
                    "  Reason: All frames cleared prefilter"
                ]
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
    if hasattr(branch_result, 'final_score') and branch_result.final_score > 0:
        # Anime branch score fusion reason
        tags_part = ""
        if branch_result.detections:
            top = sorted(branch_result.detections, key=lambda x: x[1], reverse=True)
            tags_part = ", " + ", ".join(f"{label} ({score:.2f})" for label, score, _ in top[:2])
        breast = getattr(branch_result, "breast_score", 0.0)
        suggestive = getattr(branch_result, "suggestive_score", 0.0)
        return f"score={branch_result.final_score:.2f} (exp={branch_result.wdv3_explicit:.2f}, r18={branch_result.anime_rating_r18:.2f}, gen={branch_result.genital_score:.2f}, breast={breast:.2f}, sugg={suggestive:.2f}{tags_part})"

    if not branch_result.detections:
        return f"rating={getattr(branch_result, 'rating', 'unknown')} (score={branch_result.max_score:.2f})"

    top = sorted(branch_result.detections, key=lambda x: x[1], reverse=True)
    parts = [f"{label} ({score:.2f})" for label, score, _ in top[:3]]
    return ", ".join(parts)
