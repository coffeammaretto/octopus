"""
APScheduler-based scheduler with asyncio.Semaphore(3) concurrency.
Manages scheduled post generation for channels.
"""
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Optional, Dict, Any
from datetime import datetime

import db
from config_loader import config_loader, ChannelConfig, AgentConfig


class PostScheduler:
    """Async scheduler for automated post generation."""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._semaphore = asyncio.Semaphore(3)  # Max 3 concurrent jobs
        self._running = False
        self._jobs: Dict[str, Any] = {}
    
    async def start(self):
        """Start the scheduler."""
        self._running = True
        self.scheduler.start()
        await self._load_schedules()
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        self.scheduler.shutdown(wait=False)
    
    async def _load_schedules(self):
        """Load schedules from channel configs."""
        channels = config_loader.get_enabled_channels()
        for channel in channels:
            if channel.post_schedule:
                await self.add_job(channel)
    
    async def add_job(self, channel: ChannelConfig):
        """Add a scheduled job for a channel."""
        if not channel.post_schedule or not channel.enabled:
            return
        
        async with self._semaphore:
            # Parse cron-like schedule
            try:
                trigger = CronTrigger.from_crontab(channel.post_schedule)
                
                job = self.scheduler.add_job(
                    self._execute_post_generation,
                    trigger=trigger,
                    args=[channel],
                    id=f"channel_{channel.id}",
                    replace_existing=True
                )
                self._jobs[channel.id] = job
                
                # Log schedule
                await db.log_audit_event(
                    event_type="schedule_added",
                    message=f"Scheduled posts for {channel.name}",
                    severity="info",
                    details=f"Schedule: {channel.post_schedule}"
                )
            except Exception as e:
                await db.log_audit_event(
                    event_type="schedule_error",
                    message=f"Failed to schedule {channel.name}",
                    severity="error",
                    details=str(e)
                )
    
    async def remove_job(self, channel_id: str):
        """Remove a scheduled job."""
        if channel_id in self._jobs:
            self._jobs[channel_id].remove()
            del self._jobs[channel_id]
    
    async def _execute_post_generation(self, channel: ChannelConfig):
        """Execute post generation for a channel (with semaphore)."""
        async with self._semaphore:
            try:
                # Get enabled agents
                creators = config_loader.get_enabled_agents("creator")
                validators = config_loader.get_enabled_agents("validator")
                publishers = config_loader.get_enabled_agents("publisher")
                
                if not all([creators, validators, publishers]):
                    await db.log_audit_event(
                        event_type="post_generation_failed",
                        message=f"No enabled agents for channel {channel.name}",
                        severity="warning"
                    )
                    return
                
                # Use first enabled agent of each role
                creator_cfg = creators[0]
                validator_cfg = validators[0]
                publisher_cfg = publishers[0]
                
                # Check if API keys are configured
                if not all([creator_cfg.gemini_api_key, 
                           validator_cfg.gemini_api_key, 
                           publisher_cfg.gemini_api_key]):
                    await db.log_audit_event(
                        event_type="post_generation_failed",
                        message=f"Missing API keys for channel {channel.name}",
                        severity="error"
                    )
                    return
                
                # Import workflow here to avoid circular imports
                from agents.workflow import run_workflow
                
                # Generate topic (could be from template or AI-generated)
                topic = f"Автоматический пост для {channel.name}"
                
                result = await run_workflow(
                    topic=topic,
                    channel_id=channel.id,
                    creator_cfg=creator_cfg,
                    validator_cfg=validator_cfg,
                    publisher_cfg=publisher_cfg,
                    do_search=True
                )
                
                if result.get("error"):
                    await db.log_audit_event(
                        event_type="post_generation_failed",
                        message=f"Workflow error for {channel.name}: {result['error']}",
                        severity="error"
                    )
                else:
                    await db.log_audit_event(
                        event_type="post_generated",
                        message=f"Post generated for {channel.name}",
                        severity="info",
                        details=f"Post ID: {result.get('post_id')}"
                    )
                    
            except Exception as e:
                await db.log_audit_event(
                    event_type="post_generation_error",
                    message=f"Error generating post for {channel.name}",
                    severity="error",
                    details=str(e)
                )
    
    def get_next_run(self, channel_id: str) -> Optional[datetime]:
        """Get next scheduled run time for a channel."""
        if channel_id in self._jobs:
            return self._jobs[channel_id].next_run_time
        return None


# Global scheduler instance
post_scheduler = PostScheduler()
