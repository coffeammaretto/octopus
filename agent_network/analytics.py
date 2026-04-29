"""
Analytics module with async queues → batch flush → analytics.db.
Tracks usage_metadata: tokens_in/out, cost_usd for economic tracking.
"""
import asyncio
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

import aiosqlite

from db import ANALYTICS_DB


class AnalyticsCollector:
    """Async analytics collector with batched DB flush."""
    
    def __init__(self):
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
        """Background task that flushes analytics every 30 seconds."""
        while self._running:
            try:
                await asyncio.sleep(30)
                await self._batch_flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Analytics flush error: {e}")
    
    async def track_usage(self, agent_id: str, tokens_in: int, tokens_out: int, 
                          cost_usd: float, model_name: str) -> None:
        """Queue usage data for batched insert."""
        await self._queue.put({
            'type': 'usage',
            'agent_id': agent_id,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'cost_usd': cost_usd,
            'model_name': model_name,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def track_event(self, event_type: str, agent_id: Optional[str] = None,
                          message: str = "", severity: str = "info",
                          dedup_key: Optional[str] = None) -> None:
        """Queue audit event for batched insert."""
        await self._queue.put({
            'type': 'audit',
            'event_type': event_type,
            'agent_id': agent_id,
            'message': message,
            'severity': severity,
            'dedup_key': dedup_key,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _batch_flush(self) -> None:
        """Flush all queued items to the database."""
        usage_records: List[Dict] = []
        audit_records: List[Dict] = []
        
        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                if record['type'] == 'usage':
                    usage_records.append(record)
                elif record['type'] == 'audit':
                    audit_records.append(record)
            except asyncio.QueueEmpty:
                break
        
        try:
            async with aiosqlite.connect(ANALYTICS_DB) as db:
                # Batch insert usage records
                if usage_records:
                    await db.executemany("""
                        INSERT INTO usage_log (agent_id, tokens_in, tokens_out, cost_usd, model_name, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, [(r['agent_id'], r['tokens_in'], r['tokens_out'], r['cost_usd'], 
                           r['model_name'], r['timestamp']) for r in usage_records])
                
                # Batch insert audit records
                if audit_records:
                    await db.executemany("""
                        INSERT INTO audit_log (event_type, agent_id, message, severity, dedup_key, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, [(r['event_type'], r['agent_id'], r['message'], r['severity'],
                           r['dedup_key'], r['timestamp']) for r in audit_records])
                
                await db.commit()
        except Exception as e:
            print(f"Failed to flush analytics: {e}")
    
    async def _force_flush(self) -> None:
        """Force immediate flush of all queued records."""
        await self._batch_flush()
    
    async def flush_now(self) -> None:
        """Public method to trigger immediate flush (for graceful shutdown)."""
        await self._force_flush()
    
    async def get_total_cost(self, agent_id: Optional[str] = None) -> float:
        """Get total cost, optionally filtered by agent."""
        async with aiosqlite.connect(ANALYTICS_DB) as db:
            if agent_id:
                cursor = await db.execute(
                    "SELECT SUM(cost_usd) FROM usage_log WHERE agent_id = ?", (agent_id,)
                )
            else:
                cursor = await db.execute("SELECT SUM(cost_usd) FROM usage_log")
            result = await cursor.fetchone()
            return result[0] or 0.0
    
    async def get_usage_summary(self, days: int = 7) -> List[Dict]:
        """Get usage summary grouped by day."""
        async with aiosqlite.connect(ANALYTICS_DB) as db:
            cursor = await db.execute("""
                SELECT DATE(timestamp) as date, SUM(tokens_in) as tokens_in, 
                       SUM(tokens_out) as tokens_out, SUM(cost_usd) as cost_usd
                FROM usage_log
                WHERE timestamp >= datetime('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            return [{'date': r[0], 'tokens_in': r[1], 'tokens_out': r[2], 'cost_usd': r[3]} 
                    for r in rows]


# Global singleton instance
analytics_collector = AnalyticsCollector()
