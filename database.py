# database.py
import asyncpg
import os
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

pool = None

async def init_db_pool():
    """Initialize the PostgreSQL connection pool."""
    global pool
    try:
        # Get credentials from env, or use defaults
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        database = os.getenv("POSTGRES_DB", "nsfwscanner")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")

        dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        pool = await asyncpg.create_pool(dsn)
        logger.info("✅ Connected to PostgreSQL!")
        
        # Initialize tables
        await init_tables()
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
        # Re-raise so bot doesn't start without DB if it's critical
        raise e

async def get_db():
    """Get the connection pool."""
    if not pool:
        await init_db_pool()
    return pool

async def init_tables():
    """Create necessary tables if they don't exist."""
    async with pool.acquire() as conn:
        # modlogs table
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS modlogs (
            case_id SERIAL PRIMARY KEY,
            user_id BIGINT,
            action TEXT,
            reason TEXT,
            moderator_id BIGINT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # scheduled actions table
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduled (
            id SERIAL PRIMARY KEY,
            guild_id BIGINT,
            user_id BIGINT,
            action TEXT,
            execute_at_ts BIGINT,
            extra TEXT
        );
        """)
        logger.info("✅ Database tables initialized.")

async def close_db():
    """Close the connection pool."""
    if pool:
        await pool.close()
        logger.info("👋 Database connection closed.")

# --- Modlogs Helpers ---

async def add_modlog(user_id: int, action: str, reason: str, moderator_id: int):
    """Add a moderation log entry."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO modlogs (user_id, action, reason, moderator_id, timestamp)
            VALUES ($1, $2, $3, $4, NOW())
            RETURNING case_id
            """,
            user_id, action, reason, moderator_id
        )
        return row['case_id']

async def get_modlogs(user_id: int):
    """Get all modlogs for a user."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT case_id, user_id, action, reason, moderator_id, 
                   to_char(timestamp, 'YYYY-MM-DD HH24:MI:SS') as timestamp_str
            FROM modlogs 
            WHERE user_id = $1 
            ORDER BY case_id ASC
            """,
            user_id
        )
        # Convert record objects to list of tuples for compatibility
        return [(r['case_id'], r['user_id'], r['action'], r['reason'], r['moderator_id'], r['timestamp_str']) for r in rows]

async def get_case(case_id: int):
    """Get a specific case by ID."""
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            """
            SELECT case_id, user_id, action, reason, moderator_id, 
                   to_char(timestamp, 'YYYY-MM-DD HH24:MI:SS') as timestamp_str
            FROM modlogs 
            WHERE case_id = $1
            """,
            case_id
        )
        if r:
            return (r['case_id'], r['user_id'], r['action'], r['reason'], r['moderator_id'], r['timestamp_str'])
        return None

async def clear_warns(user_id: int):
    """Clear warning logs for a user."""
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM modlogs WHERE user_id = $1 AND action = 'Warn'",
            user_id
        )

# --- Scheduled Actions Helpers ---

async def add_scheduled(guild_id: int, user_id: int, action: str, execute_at_ts: int, extra: str = None):
    """Add a scheduled action."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO scheduled (guild_id, user_id, action, execute_at_ts, extra)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            guild_id, user_id, action, execute_at_ts, extra
        )
        return row['id']

async def get_due_scheduled(now_ts: int):
    """Get actions that are due to be executed."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, guild_id, user_id, action, execute_at_ts, extra FROM scheduled WHERE execute_at_ts <= $1",
            now_ts
        )
        # Convert to list of tuples/objects as expected by consumer
        return [(r['id'], r['guild_id'], r['user_id'], r['action'], r['execute_at_ts'], r['extra']) for r in rows]

async def remove_scheduled_by_id(sched_id: int):
    """Remove a scheduled action by ID."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM scheduled WHERE id = $1", sched_id)
