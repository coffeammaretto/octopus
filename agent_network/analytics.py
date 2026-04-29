"""
Analytics module for economic tracking.
Async queues → batch flush → analytics.db
"""
import asyncio
from typing import Dict, Optional
from datetime import datetime
import db


class AnalyticsTracker:
    """Track API usage and costs with async batched writes."""
    
    def __init__(self):
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
        """Background task to flush every 30 seconds."""
        while self._running:
            await asyncio.sleep(30)
            await self._batch_flush()
    
    async def record(self, agent_id: str, tokens_in: int, tokens_out: int,
                     cost_usd: float, model_name: str = ''):
        """Record usage metrics."""
        await self._queue.put({
            'agent_id': agent_id,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'cost_usd': cost_usd,
            'model_name': model_name,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _batch_flush(self):
        """Flush queued records to database."""
        if self._queue.empty():
            return
        
        records = []
        while not self._queue.empty():
            try:
                records.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        if not records:
            return
        
        try:
            for record in records:
                await db.record_usage(
                    agent_id=record['agent_id'],
                    tokens_in=record['tokens_in'],
                    tokens_out=record['tokens_out'],
                    cost_usd=record['cost_usd'],
                    model_name=record['model_name']
                )
        except Exception as e:
            print(f"Analytics flush error: {e}")
    
    async def _force_flush(self):
        """Force immediate flush."""
        await self._batch_flush()
    
    def estimate_cost(self, tokens_in: int, tokens_out: int, 
                      model: str = 'gemini-2.0-flash') -> float:
        """Estimate cost based on token counts.
        
        Gemini 2.0 Flash pricing (approximate):
        - Input: $0.10 / 1M tokens
        - Output: $0.40 / 1M tokens
        """
        rates = {
            'gemini-2.0-flash': {'in': 0.10, 'out': 0.40},
            'gemini-1.5-pro': {'in': 1.25, 'out': 5.00},
            'gemini-1.5-flash': {'in': 0.075, 'out': 0.30},
        }
        rate = rates.get(model, rates['gemini-2.0-flash'])
        return (tokens_in * rate['in'] + tokens_out * rate['out']) / 1_000_000


# Global analytics tracker instance
analytics_tracker = AnalyticsTracker()
