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

CREATE TABLE IF NOT EXISTS scan_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id         TEXT NOT NULL,
    guild_id           TEXT,
    channel_id         TEXT,
    user_id            TEXT,
    filename           TEXT,
    verdict            TEXT NOT NULL,
    branch             TEXT,
    model              TEXT,
    reason             TEXT,
    processing_time_ms REAL,
    cache_hit          INTEGER NOT NULL DEFAULT 0,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scan_log_guild
    ON scan_log (guild_id, created_at);
"""


async def init_db(db_path: str = "./bot.db") -> None:
    """Create the database and tables if they don't already exist."""
    global _DB_PATH
    _DB_PATH = db_path
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.executescript(_INIT_SQL)
        await db.commit()
    # Also initialise the hash cache table in the same DB
    from utils.hash_cache import init_hash_cache
    await init_hash_cache(db_path)
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


# ── Scan logging ──────────────────────────────────────────────────────────────

async def record_scan(
    *,
    message_id: str,
    guild_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,
    filename: Optional[str] = None,
    verdict: str,
    branch: Optional[str] = None,
    model: Optional[str] = None,
    reason: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
    cache_hit: bool = False,
) -> None:
    """Insert a scan event into the scan_log table."""
    try:
        async with aiosqlite.connect(_DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO scan_log
                    (message_id, guild_id, channel_id, user_id, filename,
                     verdict, branch, model, reason, processing_time_ms, cache_hit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id, guild_id, channel_id, user_id, filename,
                    verdict, branch, model, reason, processing_time_ms,
                    1 if cache_hit else 0,
                ),
            )
            await db.commit()
    except Exception as e:
        logger.warning("[Database] record_scan failed: %s", e)


# ── CSV export ────────────────────────────────────────────────────────────────

async def export_feedback_csv(guild_id: str) -> str:
    """
    Export all moderation_feedback rows for a guild as a CSV string.
    Returns the CSV text (may be very large — caller should write to BytesIO).
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "message_id", "user_id", "moderator_id",
        "predicted_verdict", "moderator_verdict",
        "branch", "model", "processing_time_ms",
        "detected_tags", "created_at",
    ])

    async with aiosqlite.connect(_DB_PATH) as db:
        async with db.execute(
            """
            SELECT id, message_id, user_id, moderator_id,
                   predicted_verdict, moderator_verdict,
                   branch, model, processing_time_ms,
                   detected_tags, created_at
            FROM moderation_feedback
            WHERE guild_id = ?
            ORDER BY created_at DESC
            """,
            (guild_id,),
        ) as cursor:
            async for row in cursor:
                writer.writerow(row)

    return output.getvalue()


async def get_scan_stats(guild_id: str) -> Dict[str, Any]:
    """
    Compute scan statistics for a guild from the scan_log table.

    Returns a dict with:
        - total_scans, blocked, reviewed, safe, cache_hits
        - avg_processing_ms
        - verdict_breakdown: dict of verdict -> count
    """
    stats: Dict[str, Any] = {
        "total_scans": 0,
        "blocked": 0,
        "reviewed": 0,
        "safe": 0,
        "cache_hits": 0,
        "avg_processing_ms": 0.0,
        "verdict_breakdown": {},
    }

    async with aiosqlite.connect(_DB_PATH) as db:
        async with db.execute(
            """
            SELECT verdict, COUNT(*) as cnt, AVG(processing_time_ms) as avg_ms,
                   SUM(cache_hit) as hits
            FROM scan_log
            WHERE guild_id = ?
            GROUP BY verdict
            """,
            (guild_id,),
        ) as cursor:
            rows = await cursor.fetchall()

    total_ms = 0.0
    ms_count = 0
    for verdict, count, avg_ms, hits in rows:
        stats["total_scans"] += count
        stats["cache_hits"] += hits or 0
        stats["verdict_breakdown"][verdict] = count
        if verdict in ("BLOCK", "NSFW", "EXPLICIT"):
            stats["blocked"] += count
        elif verdict in ("REVIEW", "SUGGESTIVE"):
            stats["reviewed"] += count
        elif verdict == "SAFE":
            stats["safe"] += count
        if avg_ms:
            total_ms += avg_ms * count
            ms_count += count

    if ms_count > 0:
        stats["avg_processing_ms"] = round(total_ms / ms_count, 1)

    return stats


# ── Automated cleanup ─────────────────────────────────────────────────────────

async def cleanup_old_records(
    db_path: str = "",
    feedback_days: int = 90,
    cache_days: int = 30,
    scan_log_days: int = 60,
) -> Dict[str, int]:
    """
    Delete old rows to keep the database trim.

    Rules:
    - moderation_feedback older than ``feedback_days`` → deleted
    - image_hash_cache older than ``cache_days`` AND hit_count < 3 → deleted
      (frequently-seen hashes are kept longer regardless)
    - scan_log older than ``scan_log_days`` → deleted

    Returns a dict of {table: rows_deleted}.
    """
    path = db_path or _DB_PATH
    deleted: Dict[str, int] = {"moderation_feedback": 0, "image_hash_cache": 0, "scan_log": 0}

    try:
        async with aiosqlite.connect(path) as db:
            cur = await db.execute(
                "DELETE FROM moderation_feedback WHERE created_at < datetime('now', ?)",
                (f"-{feedback_days} days",),
            )
            deleted["moderation_feedback"] = cur.rowcount

            cur = await db.execute(
                """
                DELETE FROM image_hash_cache
                WHERE last_seen_at < datetime('now', ?)
                  AND hit_count < 3
                """,
                (f"-{cache_days} days",),
            )
            deleted["image_hash_cache"] = cur.rowcount

            cur = await db.execute(
                "DELETE FROM scan_log WHERE created_at < datetime('now', ?)",
                (f"-{scan_log_days} days",),
            )
            deleted["scan_log"] = cur.rowcount

            await db.commit()

    except Exception as e:
        logger.error("[Database] cleanup_old_records error: %s", e)

    logger.info(
        "[Database] Cleanup complete — deleted: feedback=%d cache=%d scan_log=%d",
        deleted["moderation_feedback"],
        deleted["image_hash_cache"],
        deleted["scan_log"],
    )
    return deleted
