"""
Database module with aiosqlite, WAL mode, and safe schema evolution.
Only additive changes allowed (ALTER TABLE ADD COLUMN ... DEFAULT ...).
No DROP or MODIFY operations.
"""
import asyncio
import aiosqlite
from pathlib import Path
from typing import Optional, Any, Dict, List

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

POSTS_DB = DATA_DIR / "posts.db"
LIMITS_DB = DATA_DIR / "limits.db"
ANALYTICS_DB = DATA_DIR / "analytics.db"


async def init_wal(db_path: Path) -> None:
    """Initialize WAL mode and busy_timeout for a database."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.commit()


async def safe_add_column(db_path: Path, table: str, column: str, col_type: str, default: Any) -> None:
    """Safely add a column if it doesn't exist (additive schema evolution only)."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in await cursor.fetchall()]
        if column not in columns:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}")
            await db.commit()


async def init_posts_db() -> None:
    """Initialize posts database with archive tables."""
    await init_wal(POSTS_DB)
    async with aiosqlite.connect(POSTS_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_at TIMESTAMP,
                validation_attempts INTEGER DEFAULT 0,
                creator_agent TEXT,
                validator_agent TEXT,
                publisher_agent TEXT,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                search_sources TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_posts_channel ON posts(channel_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at)")
        await db.commit()
    
    # Safe schema evolution examples
    await safe_add_column(POSTS_DB, "posts", "error_message", "TEXT", "NULL")


async def init_limits_db() -> None:
    """Initialize limits database for RPM/TPM tracking."""
    await init_wal(LIMITS_DB)
    async with aiosqlite.connect(LIMITS_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS usage_counters (
                agent_id TEXT PRIMARY KEY,
                rpm_count INTEGER DEFAULT 0,
                tpm_count INTEGER DEFAULT 0,
                last_reset_rpm TIMESTAMP,
                last_reset_tpm TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def init_analytics_db() -> None:
    """Initialize analytics database for economic tracking and audit logs."""
    await init_wal(ANALYTICS_DB)
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                model_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                agent_id TEXT,
                message TEXT,
                severity TEXT DEFAULT 'info',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dedup_key TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_agent ON usage_log(agent_id)")
        await db.commit()


async def init_all_dbs() -> None:
    """Initialize all databases."""
    await init_posts_db()
    await init_limits_db()
    await init_analytics_db()


# CRUD helpers for posts
async def create_post(channel_id: str, content: str, creator_agent: str) -> int:
    """Create a new post draft."""
    async with aiosqlite.connect(POSTS_DB) as db:
        cursor = await db.execute(
            "INSERT INTO posts (channel_id, content, creator_agent, status) VALUES (?, ?, ?, 'draft')",
            (channel_id, content, creator_agent)
        )
        await db.commit()
        return cursor.lastrowid


async def get_post(post_id: int) -> Optional[Dict[str, Any]]:
    """Get a post by ID."""
    async with aiosqlite.connect(POSTS_DB) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM posts WHERE id = ?", (post_id,))
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None


async def update_post(post_id: int, **kwargs) -> None:
    """Update post fields dynamically."""
    if not kwargs:
        return
    fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [post_id]
    async with aiosqlite.connect(POSTS_DB) as db:
        await db.execute(f"UPDATE posts SET {fields} WHERE id = ?", values)
        await db.commit()


async def get_archive_paginated(limit: int = 20, offset: int = 0, channel_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get paginated archive of posts."""
    async with aiosqlite.connect(POSTS_DB) as db:
        db.row_factory = aiosqlite.Row
        if channel_id:
            cursor = await db.execute(
                "SELECT * FROM posts WHERE channel_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (channel_id, limit, offset)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM posts ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def count_posts(channel_id: Optional[str] = None) -> int:
    """Count total posts for pagination."""
    async with aiosqlite.connect(POSTS_DB) as db:
        if channel_id:
            cursor = await db.execute("SELECT COUNT(*) FROM posts WHERE channel_id = ?", (channel_id,))
        else:
            cursor = await db.execute("SELECT COUNT(*) FROM posts")
        result = await cursor.fetchone()
        return result[0]


# Audit log helpers
async def log_audit(event_type: str, agent_id: Optional[str] = None, message: str = "", 
                    severity: str = "info", dedup_key: Optional[str] = None) -> None:
    """Log an audit event asynchronously."""
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.execute(
            "INSERT INTO audit_log (event_type, agent_id, message, severity, dedup_key) VALUES (?, ?, ?, ?, ?)",
            (event_type, agent_id, message, severity, dedup_key)
        )
        await db.commit()


# Usage tracking helpers
async def log_usage(agent_id: str, tokens_in: int, tokens_out: int, cost_usd: float, model_name: str) -> None:
    """Log token usage for economic tracking."""
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.execute(
            "INSERT INTO usage_log (agent_id, tokens_in, tokens_out, cost_usd, model_name) VALUES (?, ?, ?, ?, ?)",
            (agent_id, tokens_in, tokens_out, cost_usd, model_name)
        )
        await db.commit()
