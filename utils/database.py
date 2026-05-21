"""
utils/database.py — Async SQLite database manager for moderator feedback.

Stores moderation verdicts, model scores (as JSON), and moderator corrections
for active-learning calibration data preparation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH: str = "./bot.db"
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS moderation_feedback (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id         TEXT NOT NULL,
    guild_id           TEXT,
    channel_id         TEXT,
    user_id            TEXT,
    moderator_id       TEXT,
    content_type       TEXT,
    predicted_verdict  TEXT,
    moderator_verdict  TEXT,
    model_scores       TEXT,
    detected_tags      TEXT,
    branch             TEXT,
    model              TEXT,
    processing_time_ms REAL,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique
    ON moderation_feedback (message_id, COALESCE(moderator_id, ''));
"""


async def init_db(db_path: str = "./bot.db") -> None:
    """Create the database and tables if they don't already exist."""
    global _DB_PATH
    _DB_PATH = db_path
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.executescript(_INIT_SQL)
        await db.commit()
    logger.info("[Database] Initialized at %s", _DB_PATH)


async def record_feedback(
    *,
    message_id: str,
    guild_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,
    moderator_id: Optional[str] = None,
    content_type: Optional[str] = None,
    predicted_verdict: Optional[str] = None,
    moderator_verdict: str,
    model_scores: Optional[Dict[str, Any]] = None,
    detected_tags: Optional[List[Any]] = None,
    branch: Optional[str] = None,
    model: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> int:
    """
    Insert or replace a feedback record.

    Returns the row ID of the inserted/updated record.
    Prevents duplicate submissions for the same (message_id, moderator_id).
    """
    model_scores_json = json.dumps(model_scores or {})
    detected_tags_json = json.dumps(detected_tags or [])

    async with aiosqlite.connect(_DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO moderation_feedback
                (message_id, guild_id, channel_id, user_id, moderator_id,
                 content_type, predicted_verdict, moderator_verdict,
                 model_scores, detected_tags, branch, model, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, COALESCE(moderator_id, '')) DO UPDATE SET
                moderator_verdict  = excluded.moderator_verdict,
                model_scores       = excluded.model_scores,
                detected_tags      = excluded.detected_tags,
                created_at         = CURRENT_TIMESTAMP
            """,
            (
                message_id,
                guild_id,
                channel_id,
                user_id,
                moderator_id,
                content_type,
                predicted_verdict,
                moderator_verdict,
                model_scores_json,
                detected_tags_json,
                branch,
                model,
                processing_time_ms,
            ),
        )
        await db.commit()
        row_id = cursor.lastrowid or 0

    logger.debug(
        "[Database] Feedback recorded: message=%s moderator=%s verdict=%s (row %d)",
        message_id,
        moderator_id,
        moderator_verdict,
        row_id,
    )
    return row_id


async def has_feedback(message_id: str, moderator_id: str) -> bool:
    """Return True if this moderator already submitted feedback on this message."""
    async with aiosqlite.connect(_DB_PATH) as db:
        async with db.execute(
            "SELECT 1 FROM moderation_feedback WHERE message_id = ? AND moderator_id = ? LIMIT 1",
            (message_id, moderator_id),
        ) as cursor:
            return await cursor.fetchone() is not None


async def get_feedback_stats(guild_id: str) -> Dict[str, Any]:
    """
    Compute feedback statistics for a guild.

    Returns a dict with:
        - total_logged: all feedback rows for this guild
        - correct: confirmed true positives
        - false_positives: moderator-marked FP
        - false_negatives: moderator-marked FN
        - fp_rate: false_positives / total_logged (0.0 if no data)
        - fn_rate: false_negatives / total_logged (0.0 if no data)
        - top_failed_tags: list of (tag, count) for tags in FP/FN records
        - accuracy: correct / total_logged
    """
    stats: Dict[str, Any] = {
        "total_logged": 0,
        "correct": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "fp_rate": 0.0,
        "fn_rate": 0.0,
        "accuracy": 0.0,
        "top_failed_tags": [],
    }

    async with aiosqlite.connect(_DB_PATH) as db:
        # Count verdicts
        async with db.execute(
            """
            SELECT moderator_verdict, COUNT(*) as cnt
            FROM moderation_feedback
            WHERE guild_id = ?
            GROUP BY moderator_verdict
            """,
            (guild_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        for verdict, count in rows:
            stats["total_logged"] += count
            if verdict == "CORRECT":
                stats["correct"] += count
            elif verdict == "FALSE_POSITIVE":
                stats["false_positives"] += count
            elif verdict == "FALSE_NEGATIVE":
                stats["false_negatives"] += count

        total = stats["total_logged"]
        if total > 0:
            stats["fp_rate"] = round(stats["false_positives"] / total * 100, 1)
            stats["fn_rate"] = round(stats["false_negatives"] / total * 100, 1)
            stats["accuracy"] = round(stats["correct"] / total * 100, 1)

        # Top failed tags from FP/FN records
        async with db.execute(
            """
            SELECT detected_tags FROM moderation_feedback
            WHERE guild_id = ?
              AND moderator_verdict IN ('FALSE_POSITIVE', 'FALSE_NEGATIVE')
              AND detected_tags IS NOT NULL
            """,
            (guild_id,),
        ) as cursor:
            tag_rows = await cursor.fetchall()

    tag_counts: Dict[str, int] = {}
    for (tags_json,) in tag_rows:
        try:
            tags = json.loads(tags_json or "[]")
            for entry in tags:
                # entry is either a tag string or a tuple/list [tag, score, ...]
                tag = entry[0] if isinstance(entry, (list, tuple)) else str(entry)
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        except Exception:
            pass

    stats["top_failed_tags"] = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return stats
