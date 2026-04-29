"""
Rate limiter with in-memory dict + asyncio.Lock → Queue → async batch flush.
Updates limits.db every 30 seconds or on SIGTERM.
"""
import asyncio
import time
from typing import Dict, Optional
from datetime import datetime
import db


class RateLimiter:
    """Async rate limiter with batched DB flush."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._counters: Dict[str, Dict[str, any]] = {}  # agent_id -> {rpm, tpm, last_reset}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self):
        """Stop and force final flush."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._force_flush()
    
    async def _flush_loop(self):
        """Background task to flush counters every 30 seconds."""
        while self._running:
            await asyncio.sleep(30)
            await self._batch_flush()
    
    async def check_limit(self, agent_id: str, rpm_limit: int, tpm_limit: int) -> bool:
        """Check if request is within limits. Returns True if allowed."""
        async with self._lock:
            now = time.time()
            
            if agent_id not in self._counters:
                self._counters[agent_id] = {
                    'rpm': 0,
                    'tpm': 0,
                    'last_reset': now,
                    'minute_start': now
                }
            
            counter = self._counters[agent_id]
            
            # Reset RPM counter if minute passed
            if now - counter['minute_start'] >= 60:
                counter['rpm'] = 0
                counter['minute_start'] = now
            
            # Check limits
            if counter['rpm'] >= rpm_limit:
                return False
            if counter['tpm'] >= tpm_limit:
                return False
            
            return True
    
    async def increment(self, agent_id: str, tokens: int = 1):
        """Increment counters for an agent."""
        async with self._lock:
            now = time.time()
            
            if agent_id not in self._counters:
                self._counters[agent_id] = {
                    'rpm': 0,
                    'tpm': 0,
                    'last_reset': now,
                    'minute_start': now
                }
            
            counter = self._counters[agent_id]
            
            # Reset RPM counter if minute passed
            if now - counter['minute_start'] >= 60:
                counter['rpm'] = 0
                counter['minute_start'] = now
            
            counter['rpm'] += 1
            counter['tpm'] += tokens
            
            # Queue for async flush
            await self._queue.put({
                'agent_id': agent_id,
                'rpm': counter['rpm'],
                'tpm': counter['tpm'],
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def _batch_flush(self):
        """Flush queued updates to database."""
        if self._queue.empty():
            return
        
        updates = []
        while not self._queue.empty():
            try:
                updates.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        if not updates:
            return
        
        # Aggregate by agent_id (keep latest)
        aggregated = {u['agent_id']: u for u in updates}
        
        try:
            async with db.get_connection("limits.db") as conn:
                for agent_id, data in aggregated.items():
                    await conn.execute("""
                        INSERT INTO api_limits (agent_id, rpm_count, tpm_count, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(agent_id) DO UPDATE SET
                            rpm_count = excluded.rpm_count,
                            tpm_count = excluded.tpm_count,
                            updated_at = excluded.updated_at
                    """, (agent_id, data['rpm'], data['tpm'], data['timestamp']))
                await conn.commit()
        except Exception as e:
            # Log error but don't crash
            print(f"Rate limiter flush error: {e}")
    
    async def _force_flush(self):
        """Force immediate flush of all pending updates."""
        await self._batch_flush()
    
    def get_counter(self, agent_id: str) -> Optional[Dict]:
        """Get current counter for an agent (no lock, use carefully)."""
        return self._counters.get(agent_id)


# Global rate limiter instance
rate_limiter = RateLimiter()
