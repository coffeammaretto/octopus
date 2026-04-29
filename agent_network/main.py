"""
Main FastAPI application with auth middleware, logging filter, signal handlers, /api/health.
Entry point for uvicorn (must run with --workers 1).
"""
import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import project modules
from db import init_all_dbs, get_archive_paginated, count_posts, log_audit, get_post
from config_loader import config_loader, initialize_configs, ChannelsConfig, ChannelConfig, AgentsConfig, AgentConfig
from rate_limiter import rate_limiter
from analytics import analytics_collector
from auth import init_auth, auth, hash_password, verify_password
from scheduler import scheduler
from agents.workflow import run_content_pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "web" / "static"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TRACEBACK SANITIZATION FILTER
# Masks API keys and tokens in logs
# ============================================================================
class TokenSanitizerFilter(logging.Filter):
    """Filter that masks sensitive data in log records."""
    
    PATTERNS = [
        (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]{20,}["\']?', '***API_KEY_REDACTED***'),
        (r'["\']?token["\']?\s*[:=]\s*["\']?[\w-]{20,}["\']?', '***TOKEN_REDACTED***'),
        (r'Bearer\s+[\w-]{20,}', 'Bearer ***REDACTED***'),
        (r'sk-[\w-]{32,}', '***KEY_REDACTED***'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
            record.msg = msg
        
        if hasattr(record, 'args') and record.args:
            args = []
            for arg in record.args:
                arg_str = str(arg)
                for pattern, replacement in self.PATTERNS:
                    arg_str = re.sub(pattern, replacement, arg_str, flags=re.IGNORECASE)
                args.append(arg_str)
            record.args = tuple(args)
        
        return True


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "app.log", encoding='utf-8')
    ]
)

logger = logging.getLogger("agent_network")
logger.addFilter(TokenSanitizerFilter())

# ============================================================================
# LIFESPAN MANAGER
# Handles startup and shutdown events
# ============================================================================
_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting AI Agent Network...")
    
    try:
        # Initialize databases
        await init_all_dbs()
        logger.info("Databases initialized")
        
        # Initialize configs
        await initialize_configs()
        logger.info("Configuration loaded")
        
        # Initialize auth
        await init_auth()
        logger.info("Authentication initialized")
        
        # Start background tasks
        await rate_limiter.start()
        await analytics_collector.start()
        await scheduler.start()
        logger.info("Background services started")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    finally:
        # Shutdown - graceful cleanup
        logger.info("Shutting down gracefully...")
        
        # Flush all pending data
        await rate_limiter.stop()
        await analytics_collector.stop()
        await scheduler.stop()
        
        if auth:
            await auth.cleanup()
        
        logger.info("Shutdown complete")


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="AI Agent Network",
    description="Autonomous Telegram Content Generation System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================================
# AUTH DEPENDENCY
# ============================================================================
async def get_current_user(request: Request) -> Optional[str]:
    """Get current user from session cookie."""
    if not auth:
        return None
    
    session_id = request.cookies.get("session_id")
    if not session_id:
        return None
    
    user_id = await auth.validate_session(session_id)
    return user_id


async def require_auth(request: Request) -> str:
    """Require authentication for protected routes."""
    user_id = await get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Serve the main SPA."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AI Agent Network API"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint for systemd/monitoring."""
    uptime = time.time() - _start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/stats")
async def get_stats(user_id: str = Depends(require_auth)):
    """Get system statistics."""
    total_posts = await count_posts()
    total_cost = await analytics_collector.get_total_cost()
    
    # Get today's posts
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    return {
        "totalPosts": total_posts,
        "totalCost": total_cost,
        "todayPosts": 0,  # Implement date-filtered count
        "tokensIn": 0,
        "tokensOut": 0,
        "activeJobs": len(scheduler.get_all_jobs())
    }


@app.get("/api/channels")
async def list_channels(user_id: str = Depends(require_auth)):
    """List all configured channels."""
    config = await config_loader.load_channels()
    return [channel.model_dump() for channel in config.channels]


@app.post("/api/channels")
async def create_channel(channel_data: dict, user_id: str = Depends(require_auth)):
    """Create or update a channel."""
    config = await config_loader.load_channels()
    
    # Check if channel exists
    existing_idx = None
    for i, ch in enumerate(config.channels):
        if ch.id == channel_data.get('id') or ch.chat_id == channel_data.get('chat_id'):
            existing_idx = i
            break
    
    if existing_idx is not None:
        # Update existing
        channel = ChannelConfig(
            id=config.channels[existing_idx].id,
            name=channel_data.get('name', config.channels[existing_idx].name),
            chat_id=channel_data.get('chat_id', config.channels[existing_idx].chat_id),
            enabled=channel_data.get('enabled', config.channels[existing_idx].enabled),
            posting_schedule=channel_data.get('posting_schedule'),
            templates=channel_data.get('templates', []),
            hashtags=channel_data.get('hashtags', [])
        )
        config.channels[existing_idx] = channel
    else:
        # Create new
        import uuid
        channel = ChannelConfig(
            id=str(uuid.uuid4())[:8],
            name=channel_data.get('name', 'New Channel'),
            chat_id=channel_data.get('chat_id', ''),
            posting_schedule=channel_data.get('posting_schedule'),
            templates=channel_data.get('templates', []),
            hashtags=channel_data.get('hashtags', [])
        )
        config.channels.append(channel)
    
    await config_loader.save_channels(config)
    
    # Update scheduler
    if channel.posting_schedule:
        scheduler.add_job(channel.id, channel.posting_schedule)
    else:
        scheduler.remove_job(channel.id)
    
    await analytics_collector.track_event(
        'channel_updated',
        agent_id=user_id,
        message=f"Channel {channel.name} updated"
    )
    
    return channel.model_dump()


@app.delete("/api/channels/{channel_id}")
async def delete_channel(channel_id: str, user_id: str = Depends(require_auth)):
    """Delete a channel."""
    config = await config_loader.load_channels()
    config.channels = [ch for ch in config.channels if ch.id != channel_id]
    await config_loader.save_channels(config)
    
    scheduler.remove_job(channel_id)
    
    await analytics_collector.track_event(
        'channel_deleted',
        agent_id=user_id,
        message=f"Channel {channel_id} deleted"
    )
    
    return {"status": "ok"}


@app.get("/api/agents")
async def list_agents(user_id: str = Depends(require_auth)):
    """List all configured agents."""
    config = await config_loader.load_agents()
    return [agent.model_dump() for agent in config.agents]


@app.post("/api/agents")
async def create_agent(agent_data: dict, user_id: str = Depends(require_auth)):
    """Create or update an agent."""
    config = await config_loader.load_agents()
    
    import uuid
    agent = AgentConfig(
        id=agent_data.get('id', str(uuid.uuid4())[:8]),
        name=agent_data.get('name', 'New Agent'),
        role=agent_data.get('role', 'creator'),
        model_name=agent_data.get('model_name', 'gemini-1.5-pro'),
        enabled=agent_data.get('enabled', True),
        api_key_env_var=agent_data.get('api_key_env_var', 'GEMINI_API_KEY'),
        rpm_limit=agent_data.get('rpm_limit', 60),
        tpm_limit=agent_data.get('tpm_limit', 100000),
        system_prompt=agent_data.get('system_prompt')
    )
    
    # Check if exists
    existing_idx = None
    for i, a in enumerate(config.agents):
        if a.id == agent.id:
            existing_idx = i
            break
    
    if existing_idx is not None:
        config.agents[existing_idx] = agent
    else:
        config.agents.append(agent)
    
    await config_loader.save_agents(config)
    
    await analytics_collector.track_event(
        'agent_updated',
        agent_id=user_id,
        message=f"Agent {agent.name} updated"
    )
    
    return agent.model_dump()


@app.get("/api/archive")
async def get_archive(
    limit: int = 20,
    offset: int = 0,
    channel_id: Optional[str] = None,
    user_id: str = Depends(require_auth)
):
    """Get paginated archive of posts."""
    posts = await get_archive_paginated(limit=limit, offset=offset, channel_id=channel_id)
    total = await count_posts(channel_id)
    
    return {
        "posts": posts,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/archive/{post_id}")
async def get_archive_post(post_id: int, user_id: str = Depends(require_auth)):
    """Get a specific post by ID."""
    post = await get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


@app.get("/api/events")
async def get_recent_events(limit: int = 10, user_id: str = Depends(require_auth)):
    """Get recent audit events."""
    # Query from analytics.db
    import aiosqlite
    from db import ANALYTICS_DB
    
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


@app.get("/api/analytics/usage")
async def get_usage_analytics(days: int = 7, user_id: str = Depends(require_auth)):
    """Get usage analytics for chart."""
    return await analytics_collector.get_usage_summary(days=days)


@app.get("/api/logs/stream")
async def stream_logs(request: Request):
    """Server-Sent Events endpoint for live log streaming."""
    # Simple implementation - in production use proper log tailing
    async def generate():
        log_file = DATA_DIR / "app.log"
        last_size = 0
        
        while True:
            if await request.is_disconnected():
                break
            
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                        last_size = f.tell()
                    
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            # Parse log line
                            parts = line.split(' ', 3)
                            if len(parts) >= 4:
                                log_entry = {
                                    'id': f"{time.time()}_{hash(line)}",
                                    'timestamp': parts[0] + ' ' + parts[1],
                                    'level': parts[2].strip('[]'),
                                    'message': parts[3] if len(parts) > 3 else ''
                                }
                                yield f"data: {json.dumps(log_entry)}\n\n"
                except Exception:
                    pass
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/generate")
async def trigger_generation(data: dict, user_id: str = Depends(require_auth)):
    """Manually trigger content generation pipeline."""
    topic = data.get('topic')
    channel_id = data.get('channel_id')
    
    if not topic or not channel_id:
        raise HTTPException(status_code=400, detail="topic and channel_id required")
    
    # Get enabled agents
    creators = await config_loader.get_enabled_agents('creator')
    validators = await config_loader.get_enabled_agents('validator')
    publishers = await config_loader.get_enabled_agents('publisher')
    
    if not creators or not validators or not publishers:
        raise HTTPException(status_code=500, detail="No enabled agents configured")
    
    try:
        result = await run_content_pipeline(
            topic=topic,
            channel_id=channel_id,
            creator_agent_id=creators[0].id,
            validator_agent_id=validators[0].id,
            publisher_agent_id=publishers[0].id
        )
        
        return result
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backup")
async def download_backup(user_id: str = Depends(require_auth)):
    """Download system backup."""
    backup_data = {
        "channels": (await config_loader.load_channels()).model_dump(),
        "agents": (await config_loader.load_agents()).model_dump(),
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
    
    return Response(
        content=json.dumps(backup_data, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=backup_{datetime.utcnow().strftime('%Y%m%d')}.json"
        }
    )


@app.post("/api/import")
async def import_config(request: Request, user_id: str = Depends(require_auth)):
    """Import configuration from backup file."""
    form = await request.form()
    file = form.get("file")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        content = await file.read()
        backup_data = json.loads(content.decode('utf-8'))
        
        # Validate structure
        if "channels" not in backup_data or "agents" not in backup_data:
            raise ValueError("Invalid backup format")
        
        # Apply configuration
        if "channels" in backup_data:
            channels_config = ChannelsConfig(**backup_data["channels"])
            await config_loader.save_channels(channels_config)
        
        if "agents" in backup_data:
            agents_config = AgentsConfig(**backup_data["agents"])
            await config_loader.save_agents(agents_config)
        
        await analytics_collector.track_event(
            'config_imported',
            agent_id=user_id,
            message="Configuration imported from backup",
            severity='success'
        )
        
        return {"status": "ok", "message": "Configuration imported successfully"}
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@app.post("/api/login")
async def login(request: Request):
    """Login endpoint (simple demo - implement proper auth)."""
    data = await request.json()
    username = data.get('username')
    password = data.get('password')
    
    # Demo credentials (implement proper user management)
    if username == "admin" and password == os.environ.get("ADMIN_PASSWORD", "admin"):
        if auth:
            session_id = auth.create_session(username)
            
            response = JSONResponse({"status": "ok"})
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
                max_age=86400,  # 24 hours
                samesite="lax"
            )
            return response
    
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/logout")
async def logout_api(request: Request):
    """Logout API endpoint."""
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("session_id")
    return response


@app.get("/logout")
async def logout():
    """Logout and redirect to login."""
    response = Response(content="<script>window.location.href='/'</script>", media_type="text/html")
    response.delete_cookie("session_id")
    return response


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================
def handle_signal(signum, frame):
    """Handle termination signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    # IMPORTANT: Must run with workers=1 for proper async behavior
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False,
        log_level="info"
    )
