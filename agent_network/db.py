"""
Database module for async SQLite operations with WAL mode and safe schema evolution.
Only additive changes allowed (ALTER TABLE ADD COLUMN). No DROP or MODIFY.
"""
import aiosqlite
import asyncio
from pathlib import Path
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def get_connection(db_name: str):
    """Get async connection with WAL mode enabled."""
    db_path = DATA_DIR / db_name
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.commit()
        yield conn


async def init_posts_db():
    """Initialize posts database with safe schema evolution."""
    async with get_connection("posts.db") as conn:
        # Create main posts table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_at TIMESTAMP,
                validation_result TEXT,
                tokens_used INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0
            )
        """)
        # Safe schema evolution: add columns if not exist
        await _safe_add_column(conn, "posts", "hashtags", "TEXT DEFAULT ''")
        await _safe_add_column(conn, "posts", "template_name", "TEXT DEFAULT ''")
        await _safe_add_column(conn, "posts", "search_sources", "TEXT DEFAULT ''")
        await conn.commit()


async def init_limits_db():
    """Initialize rate limits database for async-flushed counters."""
    async with get_connection("limits.db") as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL UNIQUE,
                rpm_count INTEGER DEFAULT 0,
                tpm_count INTEGER DEFAULT 0,
                last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()


async def init_analytics_db():
    """Initialize analytics database for economic tracking."""
    async with get_connection("analytics.db") as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT DEFAULT ''
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                severity TEXT DEFAULT 'info',
                message TEXT NOT NULL,
                details TEXT DEFAULT '',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()


async def _safe_add_column(conn: aiosqlite.Connection, table: str, column: str, definition: str):
    """Safely add column only if it doesn't exist."""
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in await cursor.fetchall()]
    if column not in columns:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


async def insert_post(channel_id: str, content: str, **kwargs) -> int:
    """Insert a new post and return its ID."""
    async with get_connection("posts.db") as conn:
        cursor = await conn.execute(
            """INSERT INTO posts (channel_id, content, hashtags, template_name, search_sources)
               VALUES (?, ?, ?, ?, ?)""",
            (channel_id, content, kwargs.get('hashtags', ''), 
             kwargs.get('template_name', ''), kwargs.get('search_sources', ''))
        )
        await conn.commit()
        return cursor.lastrowid


async def update_post_status(post_id: int, status: str, **kwargs):
    """Update post status and optional fields."""
    async with get_connection("posts.db") as conn:
        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]
        
        if 'validation_result' in kwargs:
            updates.append("validation_result = ?")
            params.append(kwargs['validation_result'])
        if 'published_at' in kwargs:
            updates.append("published_at = ?")
            params.append(kwargs['published_at'])
        if 'tokens_used' in kwargs:
            updates.append("tokens_used = ?")
            params.append(kwargs['tokens_used'])
        if 'cost_usd' in kwargs:
            updates.append("cost_usd = ?")
            params.append(kwargs['cost_usd'])
            
        params.append(post_id)
        await conn.execute(f"UPDATE posts SET {', '.join(updates)} WHERE id = ?", params)
        await conn.commit()


async def get_posts_paginated(limit: int = 20, offset: int = 0, 
                               status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get posts with server-side pagination."""
    async with get_connection("posts.db") as conn:
        if status:
            cursor = await conn.execute(
                "SELECT * FROM posts WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset)
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM posts ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in await cursor.fetchall()]


async def get_post_count(status: Optional[str] = None) -> int:
    """Get total post count for pagination."""
    async with get_connection("posts.db") as conn:
        if status:
            cursor = await conn.execute("SELECT COUNT(*) FROM posts WHERE status = ?", (status,))
        else:
            cursor = await conn.execute("SELECT COUNT(*) FROM posts")
        return (await cursor.fetchone())[0]


async def log_audit_event(event_type: str, message: str, 
                          severity: str = 'info', details: str = ''):
    """Log an audit event asynchronously."""
    async with get_connection("analytics.db") as conn:
        await conn.execute(
            "INSERT INTO audit_log (event_type, severity, message, details) VALUES (?, ?, ?, ?)",
            (event_type, severity, message, details)
        )
        await conn.commit()


async def record_usage(agent_id: str, tokens_in: int, tokens_out: int, 
                       cost_usd: float, model_name: str = ''):
    """Record API usage for economic tracking."""
    async with get_connection("analytics.db") as conn:
        await conn.execute(
            """INSERT INTO usage_stats (agent_id, tokens_in, tokens_out, cost_usd, model_name)
               VALUES (?, ?, ?, ?, ?)""",
            (agent_id, tokens_in, tokens_out, cost_usd, model_name)
        )
        await conn.commit()


async def get_audit_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent audit logs."""
    async with get_connection("analytics.db") as conn:
        cursor = await conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in await cursor.fetchall()]


async def init_all_dbs():
    """Initialize all databases."""
    await init_posts_db()
    await init_limits_db()
    await init_analytics_db()
