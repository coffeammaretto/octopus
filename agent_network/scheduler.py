"""
APScheduler-based scheduler with asyncio.Semaphore(3) for concurrency control.
Manages scheduled content generation and publishing tasks.
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config_loader import config_loader
from agents.workflow import run_content_pipeline
from analytics import analytics_collector


class ContentScheduler:
    """Async scheduler for automated content generation and publishing."""
    
    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._semaphore = asyncio.Semaphore(3)  # Max 3 concurrent tasks
        self._running = False
    
    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        self._scheduler.start()
        await self._load_schedules()
    
    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False
        self._scheduler.shutdown(wait=True)
    
    async def _load_schedules(self) -> None:
        """Load schedules from channel configurations."""
        channels = await config_loader.get_enabled_channels()
        
        for channel in channels:
            if channel.posting_schedule:
                self.add_job(
                    channel_id=channel.id,
                    cron_expression=channel.posting_schedule
                )
    
    def add_job(self, channel_id: str, cron_expression: str, 
                topic_generator: Optional[str] = None) -> None:
        """Add a scheduled job for a channel."""
        
        async def run_pipeline():
            async with self._semaphore:
                try:
                    # Get enabled agents
                    creators = await config_loader.get_enabled_agents('creator')
                    validators = await config_loader.get_enabled_agents('validator')
                    publishers = await config_loader.get_enabled_agents('publisher')
                    
                    if not creators or not validators or not publishers:
                        await analytics_collector.track_event(
                            'schedule_error',
                            message=f"No enabled agents found for channel {channel_id}",
                            severity='error'
                        )
                        return
                    
                    # Select first available agent of each role
                    creator = creators[0]
                    validator = validators[0]
                    publisher = publishers[0]
                    
                    # Generate or use provided topic
                    topic = topic_generator or f"Auto-generated content for {channel_id}"
                    
                    # Run the pipeline
                    result = await run_content_pipeline(
                        topic=topic,
                        channel_id=channel_id,
                        creator_agent_id=creator.id,
                        validator_agent_id=validator.id,
                        publisher_agent_id=publisher.id
                    )
                    
                    if result.get('error_message'):
                        await analytics_collector.track_event(
                            'schedule_failed',
                            message=f"Pipeline failed for {channel_id}: {result['error_message']}",
                            severity='error'
                        )
                    else:
                        await analytics_collector.track_event(
                            'schedule_success',
                            message=f"Successfully published to {channel_id}",
                            severity='success'
                        )
                
                except Exception as e:
                    await analytics_collector.track_event(
                        'schedule_exception',
                        message=f"Scheduler error: {str(e)}",
                        severity='error'
                    )
        
        # Parse cron expression and add job
        try:
            parts = cron_expression.split()
            if len(parts) == 5:
                minute, hour, day, month, day_of_week = parts
                trigger = CronTrigger(
                    minute=minute,
                    hour=hour,
                    day=day,
                    month=month,
                    day_of_week=day_of_week
                )
                self._scheduler.add_job(
                    run_pipeline,
                    trigger=trigger,
                    id=f"scheduled_{channel_id}",
                    replace_existing=True
                )
        except Exception as e:
            print(f"Failed to add schedule for {channel_id}: {e}")
    
    def remove_job(self, channel_id: str) -> None:
        """Remove a scheduled job."""
        try:
            self._scheduler.remove_job(f"scheduled_{channel_id}")
        except Exception:
            pass
    
    def get_all_jobs(self) -> list[Dict[str, Any]]:
        """Get all scheduled jobs."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        return jobs


# Global singleton instance
scheduler = ContentScheduler()
