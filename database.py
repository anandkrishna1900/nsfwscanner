# database.py — SQLite backend (replaces PostgreSQL/asyncpg)
#
# Uses Python's built-in sqlite3 via aiosqlite for async access.
# No external database server required — the DB is a single file: bot.db
#
# Drop-in replacement: all function signatures are identical to the
# previous asyncpg version so no other files need to change.

import aiosqlite
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

DB_PATH = os.getenv("SQLITE_DB_PATH", "./bot.db")

# Module-level connection (reused across calls)
_db: aiosqlite.Connection | None = None


async def init_db_pool():
    """Open the SQLite connection and create tables if they don't exist."""
    global _db
    try:
        _db = await aiosqlite.connect(DB_PATH)
        _db.row_factory = aiosqlite.Row   # allows dict-style access
        await _db.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
        await _db.execute("PRAGMA foreign_keys=ON")
        await init_tables()
        logger.info("✅ Connected to SQLite: %s", os.path.abspath(DB_PATH))
    except Exception as e:
        logger.error("❌ Failed to open SQLite database: %s", e)
        raise


async def get_db() -> aiosqlite.Connection:
    """Return the active database connection, opening it if needed."""
    global _db
    if _db is None:
        await init_db_pool()
    return _db


async def init_tables():
    """Create necessary tables if they don't exist."""
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS modlogs (
            case_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            action      TEXT    NOT NULL,
            reason      TEXT    NOT NULL,
            moderator_id INTEGER NOT NULL,
            timestamp   TEXT    DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS scheduled (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id        INTEGER NOT NULL,
            user_id         INTEGER NOT NULL,
            action          TEXT    NOT NULL,
            execute_at_ts   INTEGER NOT NULL,
            extra           TEXT
        )
    """)
    await db.commit()
    logger.info("✅ Database tables initialized (SQLite)")


async def close_db():
    """Close the SQLite connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("👋 SQLite connection closed.")


# ── Modlogs helpers ────────────────────────────────────────────────────────────

async def add_modlog(user_id: int, action: str, reason: str, moderator_id: int) -> int:
    """Insert a moderation log entry and return the new case_id."""
    db = await get_db()
    cursor = await db.execute(
        """
        INSERT INTO modlogs (user_id, action, reason, moderator_id)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, action, reason, moderator_id),
    )
    await db.commit()
    return cursor.lastrowid


async def get_modlogs(user_id: int) -> list[tuple]:
    """Return all modlogs for a user as a list of (case_id, user_id, action, reason, moderator_id, timestamp)."""
    db = await get_db()
    async with db.execute(
        """
        SELECT case_id, user_id, action, reason, moderator_id, timestamp
        FROM modlogs
        WHERE user_id = ?
        ORDER BY case_id ASC
        """,
        (user_id,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [tuple(r) for r in rows]


async def get_case(case_id: int) -> tuple | None:
    """Return a single modlog entry by case_id, or None if not found."""
    db = await get_db()
    async with db.execute(
        """
        SELECT case_id, user_id, action, reason, moderator_id, timestamp
        FROM modlogs
        WHERE case_id = ?
        """,
        (case_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return tuple(row) if row else None


async def clear_warns(user_id: int) -> None:
    """Delete all 'Warn' entries for a user."""
    db = await get_db()
    await db.execute(
        "DELETE FROM modlogs WHERE user_id = ? AND action = 'Warn'",
        (user_id,),
    )
    await db.commit()


# ── Scheduled actions helpers ──────────────────────────────────────────────────

async def add_scheduled(
    guild_id: int,
    user_id: int,
    action: str,
    execute_at_ts: int,
    extra: str | None = None,
) -> int:
    """Schedule an action and return its id."""
    db = await get_db()
    cursor = await db.execute(
        """
        INSERT INTO scheduled (guild_id, user_id, action, execute_at_ts, extra)
        VALUES (?, ?, ?, ?, ?)
        """,
        (guild_id, user_id, action, execute_at_ts, extra),
    )
    await db.commit()
    return cursor.lastrowid


async def get_due_scheduled(now_ts: int) -> list[tuple]:
    """Return all scheduled actions whose execute_at_ts <= now_ts."""
    db = await get_db()
    async with db.execute(
        """
        SELECT id, guild_id, user_id, action, execute_at_ts, extra
        FROM scheduled
        WHERE execute_at_ts <= ?
        """,
        (now_ts,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [tuple(r) for r in rows]


async def remove_scheduled_by_id(sched_id: int) -> None:
    """Delete a scheduled action by its id."""
    db = await get_db()
    await db.execute("DELETE FROM scheduled WHERE id = ?", (sched_id,))
    await db.commit()
