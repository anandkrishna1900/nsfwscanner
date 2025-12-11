# database.py
import sqlite3
from datetime import datetime
import os

DB_FILE = "modlogs.db"

def get_conn():
    conn = sqlite3.connect(DB_FILE)
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # modlogs table
    c.execute("""
    CREATE TABLE IF NOT EXISTS modlogs (
        case_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT,
        reason TEXT,
        moderator_id INTEGER,
        timestamp TEXT
    );
    """)
    # scheduled actions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS scheduled (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id INTEGER,
        user_id INTEGER,
        action TEXT,             -- 'unban' | 'untimeout' | 'auto-unmute'
        execute_at_ts INTEGER,   -- unix ts UTC
        extra TEXT               -- optional JSON/text
    );
    """)
    conn.commit()
    conn.close()

# modlogs helpers
def add_modlog(user_id:int, action:str, reason:str, moderator_id:int):
    conn = get_conn()
    c = conn.cursor()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO modlogs (user_id, action, reason, moderator_id, timestamp) VALUES (?, ?, ?, ?, ?)",
              (user_id, action, reason, moderator_id, timestamp))
    conn.commit()
    case_id = c.lastrowid
    conn.close()
    return case_id

def get_modlogs(user_id:int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT case_id, user_id, action, reason, moderator_id, timestamp FROM modlogs WHERE user_id = ? ORDER BY case_id ASC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_case(case_id:int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT case_id, user_id, action, reason, moderator_id, timestamp FROM modlogs WHERE case_id = ?", (case_id,))
    row = c.fetchone()
    conn.close()
    return row

# scheduled helpers
def add_scheduled(guild_id:int, user_id:int, action:str, execute_at_ts:int, extra: str = None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO scheduled (guild_id, user_id, action, execute_at_ts, extra) VALUES (?, ?, ?, ?, ?)",
              (guild_id, user_id, action, execute_at_ts, extra))
    conn.commit()
    sid = c.lastrowid
    conn.close()
    return sid

def get_due_scheduled(now_ts:int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, guild_id, user_id, action, execute_at_ts, extra FROM scheduled WHERE execute_at_ts <= ?", (now_ts,))
    rows = c.fetchall()
    conn.close()
    return rows

def remove_scheduled_by_id(sched_id:int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM scheduled WHERE id = ?", (sched_id,))
    conn.commit()
    conn.close()
