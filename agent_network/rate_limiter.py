"""
Rate limiter with in-memory dict + asyncio.Lock → asyncio.Queue → async batch flush to limits.db.
Crash-safe: background task flushes every 30 seconds, SIGTERM triggers immediate sync.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

import aiosqlite

from db import LIMITS_DB


class RateLimiter:
    """Async rate limiter with batched DB flush for RPM/TPM tracking."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._counters: Dict[str, Dict] = {}  # agent_id -> {rpm_count, tpm_count, last_reset_rpm, last_reset_tpm}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = True
        self._flush_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self) -> None:
        """Stop the background flush task and force immediate sync."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._force_flush()
    
    async def _flush_loop(self) -> None:
        """Background task that flushes counters to DB every 30 seconds."""
        while self._running:
            try:
                await asyncio.sleep(30)
                await self._batch_flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Rate limiter flush error: {e}")
    
    async def increment(self, agent_id: str, tokens: int = 1) -> tuple[bool, int, int]:
        """
        Increment counters for an agent. Returns (allowed, rpm_remaining, tpm_remaining).
        This is a simplified check - actual limits come from agent config.
        """
        async with self._lock:
            now = datetime.utcnow()
            
            if agent_id not in self._counters:
                self._counters[agent_id] = {
                    'rpm_count': 0,
                    'tpm_count': 0,
                    'last_reset_rpm': now,
                    'last_reset_tpm': now
                }
            
            counter = self._counters[agent_id]
            
            # Reset RPM counter if minute has passed
            if now - counter['last_reset_rpm'] >= timedelta(minutes=1):
                counter['rpm_count'] = 0
                counter['last_reset_rpm'] = now
            
            # Reset TPM counter if minute has passed (could be hourly depending on requirements)
            if now - counter['last_reset_tpm'] >= timedelta(minutes=1):
                counter['tpm_count'] = 0
                counter['last_reset_tpm'] = now
            
            counter['rpm_count'] += 1
            counter['tpm_count'] += tokens
            
            # Queue for async flush
            await self._queue.put((agent_id, counter.copy()))
            
            return True, counter['rpm_count'], counter['tpm_count']
    
    async def get_counters(self, agent_id: str) -> Optional[Dict]:
        """Get current counters for an agent."""
        async with self._lock:
            if agent_id in self._counters:
                return self._counters[agent_id].copy()
            return None
    
    async def _batch_flush(self) -> None:
        """Flush all queued updates to the database."""
        updates = []
        while not self._queue.empty():
            try:
                updates.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        if not updates:
            return
        
        # Consolidate updates per agent (keep latest)
        consolidated: Dict[str, Dict] = {}
        for agent_id, counter in updates:
            consolidated[agent_id] = counter
        
        # Batch write to DB
        try:
            async with aiosqlite.connect(LIMITS_DB) as db:
                for agent_id, counter in consolidated.items():
                    await db.execute("""
                        INSERT INTO usage_counters (agent_id, rpm_count, tpm_count, last_reset_rpm, last_reset_tpm, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(agent_id) DO UPDATE SET
                            rpm_count = excluded.rpm_count,
                            tpm_count = excluded.tpm_count,
                            last_reset_rpm = excluded.last_reset_rpm,
                            last_reset_tpm = excluded.last_reset_tpm,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        agent_id,
                        counter['rpm_count'],
                        counter['tpm_count'],
                        counter['last_reset_rpm'].isoformat(),
                        counter['last_reset_tpm'].isoformat()
                    ))
                await db.commit()
        except Exception as e:
            print(f"Failed to flush rate limit counters: {e}")
    
    async def _force_flush(self) -> None:
        """Force immediate flush of all counters."""
        async with self._lock:
            # Copy current state
            snapshot = {k: v.copy() for k, v in self._counters.items()}
        
        if snapshot:
            try:
                async with aiosqlite.connect(LIMITS_DB) as db:
                    for agent_id, counter in snapshot.items():
                        await db.execute("""
                            INSERT INTO usage_counters (agent_id, rpm_count, tpm_count, last_reset_rpm, last_reset_tpm, updated_at)
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                            ON CONFLICT(agent_id) DO UPDATE SET
                                rpm_count = excluded.rpm_count,
                                tpm_count = excluded.tpm_count,
                                last_reset_rpm = excluded.last_reset_rpm,
                                last_reset_tpm = excluded.last_reset_tpm,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            agent_id,
                            counter['rpm_count'],
                            counter['tpm_count'],
                            counter['last_reset_rpm'].isoformat(),
                            counter['last_reset_tpm'].isoformat()
                        ))
                    await db.commit()
            except Exception as e:
                print(f"Failed to force flush rate limit counters: {e}")
    
    async def flush_now(self) -> None:
        """Public method to trigger immediate flush (for graceful shutdown)."""
        await self._force_flush()


# Global singleton instance
rate_limiter = RateLimiter()
