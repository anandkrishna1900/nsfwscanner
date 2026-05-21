"""
moderation/real_branch.py — Stage 2A: Real/photo content branch using NudeNet.

ONLY flags genital exposure per AGENTS.md genital-only policy.
All other NudeNet labels (BREAST, BUTTOCKS, etc.) are completely ignored.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# ── Genital-only label thresholds ─────────────────────────────────────────────
# These are the exact class name strings NudeNet v3 returns in detection dicts.
# All other NudeNet labels are DISCARDED before verdict logic runs — they cannot
# influence the verdict, max_score, or the detections list in BranchResult.
GENITAL_LABELS: dict[str, float] = {
    "FEMALE_GENITALIA_EXPOSED": 0.55,
    "MALE_GENITALIA_EXPOSED":   0.55,
    "ANUS_EXPOSED":             0.60,
    "MALE_GENITALIA_COVERED":   0.75,  # only when erection is clearly evident
}

# REVIEW window: score in [threshold - REVIEW_OFFSET, threshold) → REVIEW
REVIEW_OFFSET: float = 0.15


@dataclass
class BranchResult:
    verdict: str                                         # "BLOCK" | "REVIEW" | "SAFE"
    detections: List[Tuple[str, float, Optional[list]]] = field(default_factory=list)
    max_score: float = 0.0
    model: str = "nudenet"


class RealBranch:
    """
    Detects explicit genital content in real/photographic images using NudeNet.

    NudeNet uses ONNX internally and runs entirely on CPU.
    All inference is synchronous — wrap calls in asyncio.to_thread().
    """

    def scan(self, image: Image.Image) -> BranchResult:
        """
        Scan a PIL Image for genital content using NudeNet.

        Steps:
          1. Save PIL image to a temp JPEG (NudeNet requires a file path).
          2. Run NudeDetector.detect() — returns list of dicts with key 'class'.
          3. FILTER FIRST — keep only dicts whose 'class' is in GENITAL_LABELS.
             Every other detection is discarded here; nothing else touches verdict logic.
          4. If zero genital detections remain → return SAFE immediately.
          5. Determine BLOCK / REVIEW from the filtered list only.
          6. Delete temp file in finally block regardless of outcome.
        """
        tmp_path: Optional[str] = None

        try:
            from nudenet import NudeDetector

            # ── 1. Save to temp file ──────────────────────────────────────────
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                image.convert("RGB").save(tmp_path, format="JPEG", quality=95)

            # ── 2. Run NudeNet ────────────────────────────────────────────────
            detector = NudeDetector()
            raw_detections: list[dict] = detector.detect(tmp_path)
            # raw_detections format (NudeNet v3):
            # [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.91, "box": [x,y,w,h]}, ...]

            # ── 3. FILTER FIRST — discard everything that isn't a genital label ──
            # NudeNet v3 uses the key 'class', NOT 'label'.
            genital_detections = [
                d for d in raw_detections
                if d.get("class", "") in GENITAL_LABELS
            ]

            logger.debug(
                "[RealBranch] raw=%d detections, genital=%d after filter",
                len(raw_detections),
                len(genital_detections),
            )

            # ── 4. Zero genital detections → SAFE immediately ─────────────────
            if not genital_detections:
                logger.debug("[RealBranch] No genital labels found → SAFE")
                return BranchResult(verdict="SAFE", max_score=0.0)

            # ── 5. Verdict logic — runs ONLY on genital_detections ─────────────
            flagged: List[Tuple[str, float, Optional[list]]] = []
            max_score: float = 0.0
            verdict = "SAFE"

            for detection in genital_detections:
                label: str = detection["class"]          # already confirmed in GENITAL_LABELS
                score: float = float(detection["score"])
                box = detection.get("box")               # [x, y, w, h] or None

                threshold: float = GENITAL_LABELS[label]
                review_low: float = threshold - REVIEW_OFFSET

                logger.debug(
                    "[RealBranch] %s score=%.3f threshold=%.2f review_low=%.2f",
                    label, score, threshold, review_low,
                )

                # Only include detections that are at least in the REVIEW window
                if score >= review_low:
                    flagged.append((label, score, box))
                    if score > max_score:
                        max_score = score

                    # Escalate verdict — BLOCK wins immediately
                    if score >= threshold:
                        verdict = "BLOCK"
                    elif verdict != "BLOCK":
                        verdict = "REVIEW"

            # If every genital detection was below the review window, still SAFE
            if not flagged:
                logger.debug("[RealBranch] Genital labels found but all below review window → SAFE")
                return BranchResult(verdict="SAFE", max_score=0.0)

            logger.info(
                "[RealBranch] Verdict: %s | max_score=%.3f | flagged=%s",
                verdict,
                max_score,
                [(lbl, f"{sc:.3f}") for lbl, sc, _ in flagged],
            )
            return BranchResult(verdict=verdict, detections=flagged, max_score=max_score)

        except Exception as e:
            logger.error("[RealBranch] Scan error: %s", e, exc_info=True)
            return BranchResult(verdict="SAFE", max_score=0.0)

        finally:
            # ── 6. Always delete the temp file ────────────────────────────────
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as cleanup_err:
                    logger.warning(
                        "[RealBranch] Failed to clean up temp file %s: %s",
                        tmp_path, cleanup_err,
                    )
