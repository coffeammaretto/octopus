"""
Authentication module with cookie sessions, CSRF middleware, and in-memory LRU cache.
Session lifecycle: In-memory LRU (TTL 24h, max 100, FIFO) + async flush to sessions.json via aiofiles.
"""
import asyncio
import hashlib
import hmac
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import aiofiles
from cryptography.fernet import Fernet
from passlib.hash import pbkdf2_sha256

from config_loader import init_session_secret


DATA_DIR = Path(__file__).parent / "data"
SESSIONS_FILE = DATA_DIR / "sessions.json"


class LRUSessionCache:
    """In-memory LRU cache for sessions with TTL and max size."""
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(hours=ttl_hours)
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start background flush task."""
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self) -> None:
        """Stop and force flush."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._force_flush()
    
    async def _flush_loop(self) -> None:
        """Flush sessions to disk every 60 seconds."""
        while True:
            try:
                await asyncio.sleep(60)
                await self._force_flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Session cache flush error: {e}")
    
    async def get(self, session_id: str) -> Optional[Dict]:
        """Get session data if valid."""
        async with self._lock:
            if session_id not in self._cache:
                return None
            
            session = self._cache[session_id]
            
            # Check TTL
            created_at = datetime.fromisoformat(session['created_at'])
            if datetime.utcnow() - created_at > self._ttl:
                del self._cache[session_id]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(session_id)
            return session['data'].copy()
    
    async def set(self, session_id: str, data: Dict) -> None:
        """Set session data."""
        async with self._lock:
            # If exists, remove first to update position
            if session_id in self._cache:
                del self._cache[session_id]
            
            # Evict oldest if at capacity (FIFO)
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[session_id] = {
                'created_at': datetime.utcnow().isoformat(),
                'data': data
            }
    
    async def delete(self, session_id: str) -> None:
        """Delete session."""
        async with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]
    
    async def _force_flush(self) -> None:
        """Force flush to sessions.json."""
        async with self._lock:
            snapshot = {k: v.copy() for k, v in self._cache.items()}
        
        try:
            async with aiofiles.open(SESSIONS_FILE, 'w') as f:
                await f.write(json.dumps(snapshot))
        except Exception as e:
            print(f"Failed to flush sessions: {e}")
    
    async def flush_now(self) -> None:
        """Public flush method for graceful shutdown."""
        await self._force_flush()
    
    async def load_from_disk(self) -> None:
        """Load existing sessions from disk on startup."""
        if not SESSIONS_FILE.exists():
            return
        
        try:
            async with aiofiles.open(SESSIONS_FILE, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            async with self._lock:
                # Filter expired sessions
                now = datetime.utcnow()
                for session_id, session_data in data.items():
                    created_at = datetime.fromisoformat(session_data['created_at'])
                    if now - created_at <= self._ttl:
                        if len(self._cache) < self._max_size:
                            self._cache[session_id] = session_data
        except Exception as e:
            print(f"Failed to load sessions from disk: {e}")


class AuthMiddleware:
    """Cookie-based session auth with CSRF protection."""
    
    def __init__(self, secret_key: bytes):
        self._secret_key = secret_key
        self._fernet = Fernet(base64_urlsafe_encode(secret_key))
        self._cache = LRUSessionCache()
    
    async def initialize(self) -> None:
        """Initialize auth system."""
        await self._cache.load_from_disk()
        await self._cache.start()
    
    async def cleanup(self) -> None:
        """Cleanup on shutdown."""
        await self._cache.stop()
    
    def create_session(self, user_id: str) -> str:
        """Create a new session and return session ID."""
        session_id = hashlib.sha256(os.urandom(32) + self._secret_key).hexdigest()
        # Store in cache asynchronously (fire and forget for now)
        asyncio.create_task(self._cache.set(session_id, {'user_id': user_id}))
        return session_id
    
    async def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return user_id if valid."""
        session = await self._cache.get(session_id)
        if session:
            return session.get('user_id')
        return None
    
    async def destroy_session(self, session_id: str) -> None:
        """Destroy a session."""
        await self._cache.delete(session_id)
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token."""
        token = os.urandom(32)
        return hmac.new(self._secret_key, token, hashlib.sha256).hexdigest()
    
    def validate_csrf_token(self, token: str, session_id: str) -> bool:
        """Validate CSRF token."""
        expected = hmac.new(self._secret_key, session_id.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(token, expected)


def base64_urlsafe_encode(key: bytes) -> bytes:
    """Convert a 32-byte key to a URL-safe base64-encoded 44-byte key for Fernet."""
    import base64
    key_hash = hashlib.sha256(key).digest()
    return base64.urlsafe_b64encode(key_hash)


# Global auth instance (initialized in main.py)
auth: Optional[AuthMiddleware] = None


async def init_auth() -> AuthMiddleware:
    """Initialize global auth instance."""
    global auth
    secret = await init_session_secret()
    auth = AuthMiddleware(secret)
    await auth.initialize()
    return auth


# Password hashing helpers
def hash_password(password: str) -> str:
    """Hash password using PBKDF2-SHA256."""
    return pbkdf2_sha256.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return pbkdf2_sha256.verify(password, hashed)
