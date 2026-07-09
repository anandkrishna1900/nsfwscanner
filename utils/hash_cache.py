"""
utils/hash_cache.py — Perceptual image hash cache.

Computes a 64-bit pHash fingerprint for every scanned image and stores the
verdict in SQLite.  On subsequent scans of the same (or visually identical)
image the cached verdict is returned instantly, bypassing all AI inference.

Hash distance threshold: ≤ 8 bits different (out of 64) = "same image".
This tolerates minor compression artifacts and resizing without false-matches
between genuinely different images.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Hamming distance threshold — images this close are treated as identical.
PHASH_DISTANCE_THRESHOLD: int = 8

_DB_PATH: str = "./bot.db"

# ── SQL ────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS image_hash_cache (
    phash       TEXT PRIMARY KEY,
    verdict     TEXT NOT NULL,
    reason      TEXT,
    branch      TEXT,
    model       TEXT,
    hit_count   INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ── Initialiser (called by utils.database.init_db) ────────────────────────────

async def init_hash_cache(db_path: str = "./bot.db") -> None:
    """Create the image_hash_cache table if it does not exist."""
    global _DB_PATH
    _DB_PATH = db_path
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()
    logger.debug("[HashCache] Table ready at %s", _DB_PATH)


# ── pHash computation ─────────────────────────────────────────────────────────

def compute_phash(image: "Image.Image") -> str:
    """
    Compute a 64-bit perceptual hash of a PIL Image and return it as a
    16-character hex string.

    Returns an empty string on failure (caller should treat as cache-miss).
    """
    try:
        import imagehash
        ph = imagehash.phash(image, hash_size=8)   # 8×8 = 64 bits
        return str(ph)                             # 16 hex chars
    except Exception as e:
        logger.debug("[HashCache] phash computation failed: %s", e)
        return ""


def _hamming(a: str, b: str) -> int:
    """Hamming distance between two hex pHash strings (both must be 16 chars)."""
    try:
        ai = int(a, 16)
        bi = int(b, 16)
        xor = ai ^ bi
        return bin(xor).count("1")
    except Exception:
        return 999


# ── Cache lookup ──────────────────────────────────────────────────────────────

async def get_cached_verdict(
    phash_hex: str,
    db_path: str = "",
) -> Optional[dict]:
    """
    Look up a cached verdict for the given pHash.
    
    Uses a prefix filter to skip obviously-different hashes before doing
    the full Hamming distance check, reducing Python-side comparisons.
    """
    if not phash_hex:
        return None

    path = db_path or _DB_PATH
    # Use the first hex character as a coarse bucket filter.
    # Two hashes that differ in the first nibble differ by >= 1 bit,
    # but this still cuts ~93% of rows from the full scan.
    # For a more aggressive filter, use the first 2 chars (cuts ~98%).
    prefix = phash_hex[:1]

    try:
        async with aiosqlite.connect(path) as db:
            async with db.execute(
                "SELECT phash, verdict, reason, branch, model FROM image_hash_cache WHERE phash LIKE ?",
                (prefix + "%",),
            ) as cursor:
                rows = await cursor.fetchall()

            for stored_phash, verdict, reason, branch, model in rows:
                dist = _hamming(phash_hex, stored_phash)
                if dist <= PHASH_DISTANCE_THRESHOLD:
                    await db.execute(
                        """
                        UPDATE image_hash_cache
                        SET hit_count = hit_count + 1, last_seen_at = CURRENT_TIMESTAMP
                        WHERE phash = ?
                        """,
                        (stored_phash,),
                    )
                    await db.commit()
                    logger.debug(
                        "[HashCache] HIT phash=%s distance=%d verdict=%s",
                        phash_hex, dist, verdict,
                    )
                    return {"verdict": verdict, "reason": reason, "branch": branch, "model": model}

    except Exception as e:
        logger.warning("[HashCache] Lookup error: %s", e)

    return None


# ── Cache store ───────────────────────────────────────────────────────────────

async def store_cached_verdict(
    phash_hex: str,
    verdict: str,
    reason: str,
    branch: str,
    model: str,
    db_path: str = "",
) -> None:
    """
    Store a verdict in the hash cache.

    Only stores SAFE and BLOCK/NSFW/EXPLICIT verdicts — REVIEW/SUGGESTIVE are
    borderline cases and should re-run the full pipeline each time.
    """
    if not phash_hex:
        return
    if verdict not in ("SAFE", "BLOCK", "NSFW", "EXPLICIT"):
        return   # Don't cache borderline verdicts

    path = db_path or _DB_PATH
    try:
        async with aiosqlite.connect(path) as db:
            await db.execute(
                """
                INSERT INTO image_hash_cache (phash, verdict, reason, branch, model)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(phash) DO UPDATE SET
                    verdict      = excluded.verdict,
                    reason       = excluded.reason,
                    last_seen_at = CURRENT_TIMESTAMP,
                    hit_count    = hit_count + 1
                """,
                (phash_hex, verdict, reason, branch, model),
            )
            await db.commit()
        logger.debug("[HashCache] Stored phash=%s verdict=%s", phash_hex, verdict)
    except Exception as e:
        logger.warning("[HashCache] Store error: %s", e)
