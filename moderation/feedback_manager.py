"""
moderation/feedback_manager.py — Active-learning feedback collector.

Bridges ScanResult data → utils.database for storage.
Does NOT retrain models automatically — only prepares calibration data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from moderation.pipeline import ScanResult

logger = logging.getLogger(__name__)


def _extract_model_scores(result: ScanResult) -> Dict[str, Any]:
    """
    Serialize all model scores from a ScanResult into a flat dict for JSON storage.

    Fields extracted (where available):
        - prefilter_score
        - gatekeeper_route, gatekeeper_confidence
        - wdv3_explicit, anime_rating_r18, genital_score, breast_score
        - suggestive_score, final_score
        - nudenet_max_score, nudenet_labels
        - verdict, branch, model
    """
    scores: Dict[str, Any] = {
        "verdict": result.verdict,
        "branch": result.branch,
        "model": result.model,
        "processing_time_ms": result.processing_time_ms,
    }

    # Parse pipeline_steps to extract prefilter score and gatekeeper info
    for step in getattr(result, "pipeline_steps", []):
        lines = step.splitlines()
        for line in lines:
            stripped = line.strip()
            # Pre-filter score
            if stripped.startswith("NSFW Score:"):
                try:
                    scores["prefilter_score"] = float(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass
            # Gatekeeper route
            if stripped.startswith("Classification:"):
                scores["gatekeeper_route"] = stripped.split(":", 1)[1].strip()
            if stripped.startswith("Confidence:"):
                try:
                    scores["gatekeeper_confidence"] = float(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass
            # Anime branch fusion scores (stored in step lines like "- wdv3_explicit: 0.3450")
            for key in ("wdv3_explicit", "anime_rating_r18", "genital_score", "breast_score", "suggestive_score", "final_score"):
                prefix = f"- {key}:"
                if stripped.startswith(prefix):
                    try:
                        # Value may be "0.3450 (weight: 0.30)" — take first token
                        raw_val = stripped[len(prefix):].strip().split()[0]
                        scores[key] = float(raw_val)
                    except (ValueError, IndexError):
                        pass

    return scores


def _extract_detected_tags(result: ScanResult) -> List[Any]:
    """
    Extract detected tags or NudeNet labels from ScanResult.

    Scans pipeline_steps for detected tag lines.
    Returns a list of [tag, score] pairs.
    """
    tags: List[Any] = []
    for step in getattr(result, "pipeline_steps", []):
        lines = step.splitlines()
        in_detections = False
        for line in lines:
            stripped = line.strip()
            if "Detected Explicit Labels:" in stripped or "Detected Explicit Tags:" in stripped:
                in_detections = True
                continue
            if in_detections:
                if stripped.startswith("- "):
                    # Format: "- LABEL: 0.73" or "- tag: 0.85"
                    content = stripped[2:].strip()
                    if ":" in content:
                        label, _, score_str = content.partition(":")
                        try:
                            tags.append([label.strip(), float(score_str.strip())])
                        except ValueError:
                            tags.append([label.strip(), 0.0])
                elif stripped and not stripped.startswith("-"):
                    # New section started
                    in_detections = False
    return tags


async def record_moderation_feedback(
    *,
    message_id: str,
    guild_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,
    moderator_id: str,
    result: ScanResult,
    moderator_verdict: str,
) -> int:
    """
    Record moderator feedback for a moderation event.

    Parameters
    ----------
    message_id:
        Discord message ID of the flagged message.
    guild_id:
        Discord guild (server) ID.
    channel_id:
        Channel where the flagged content was posted.
    user_id:
        User who posted the flagged content.
    moderator_id:
        Discord ID of the moderator submitting feedback.
    result:
        The ScanResult produced by the pipeline.
    moderator_verdict:
        One of: "CORRECT", "FALSE_POSITIVE", "FALSE_NEGATIVE_NSFW", "FALSE_NEGATIVE_EXPLICIT"

    Returns
    -------
    int
        Row ID of the stored feedback record.
    """
    from utils.database import record_feedback

    model_scores = _extract_model_scores(result)
    detected_tags = _extract_detected_tags(result)

    row_id = await record_feedback(
        message_id=message_id,
        guild_id=guild_id,
        channel_id=channel_id,
        user_id=user_id,
        moderator_id=moderator_id,
        content_type=result.branch,
        predicted_verdict=result.verdict,
        moderator_verdict=moderator_verdict,
        model_scores=model_scores,
        detected_tags=detected_tags,
        branch=result.branch,
        model=result.model,
        processing_time_ms=result.processing_time_ms,
    )

    logger.info(
        "[FeedbackManager] Logged feedback: message=%s mod=%s predicted=%s correction=%s",
        message_id,
        moderator_id,
        result.verdict,
        moderator_verdict,
    )
    return row_id
