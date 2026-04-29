"""
Main FastAPI application with CSRF/CORS/Host middleware.
Strict localhost-only access (127.0.0.1:8000).
Double Submit Cookie pattern for CSRF protection.
Content-Security-Policy for XSS protection.
"""
import os
import sys
import json
import signal
import asyncio
import secrets
from pathlib import Path
from typing import Optional, List, Any
from datetime import datetime

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import db
import analytics
from config_loader import config_loader, AgentConfig, ChannelConfig
from rate_limiter import rate_limiter
from scheduler import post_scheduler

# Initialize FastAPI app
app = FastAPI(title="Agent Network Control", version="1.0.0")

# Mount static files
static_path = Path(__file__).parent / "web" / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class HostValidationMiddleware(BaseHTTPMiddleware):
    """Validate Host header to prevent DNS rebinding attacks."""
    
    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        # Only allow 127.0.0.1:8000 or localhost:8000
        allowed_hosts = ["127.0.0.1:8000", "localhost:8000", "127.0.0.1", "localhost"]
        
        # Strip port for comparison
        host_without_port = host.split(":")[0] if host else ""
        
        if host not in allowed_hosts and host_without_port not in ["127.0.0.1", "localhost"]:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Invalid host"}
            )
        
        return await call_next(request)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    Double Submit Cookie CSRF protection.
    - Sets csrf_token cookie on GET requests
    - Validates X-CSRF-Token header on POST/PUT/DELETE requests
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Set CSRF token cookie on GET requests (including index.html)
        if request.method == "GET" and request.url.path in ["/", "/index.html"]:
            if "csrf_token" not in request.cookies:
                csrf_token = secrets.token_hex(32)
                response.set_cookie(
                    key="csrf_token",
                    value=csrf_token,
                    max_age=86400,  # 24 hours
                    httponly=False,  # Must be readable by JS
                    samesite="strict",
                    secure=False,  # False for localhost
                    path="/"
                )
        
        # Validate CSRF token on state-changing requests
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            csrf_cookie = request.cookies.get("csrf_token")
            csrf_header = request.headers.get("x-csrf-token")
            
            if not csrf_cookie or not csrf_header:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token missing"}
                )
            
            if not secrets.compare_digest(csrf_cookie, csrf_header):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token mismatch"}
                )
        
        return response


class CSPMiddleware(BaseHTTPMiddleware):
    """Add Content-Security-Policy headers for XSS protection."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Only apply CSP to HTML responses (not static files)
        if request.url.path in ["/", "/index.html"] or request.url.path.startswith("/api/"):
            # Strict CSP for HTML and API responses
            # 'unsafe-eval' is REQUIRED for Alpine.js runtime template compilation
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "connect-src 'self' http://127.0.0.1:8000 https://cdn.jsdelivr.net; "
                "img-src 'self' data: https:; "
                "frame-ancestors 'none';"
            )
            response.headers["Content-Security-Policy"] = csp
        
        return response


# Apply middleware in order
app.add_middleware(CSPMiddleware)
app.add_middleware(CSRFProtectionMiddleware)
app.add_middleware(HostValidationMiddleware)

# CORS middleware - strict localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],
    allow_credentials=False,  # No credentials for security
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)


@app.on_event("startup")
async def startup():
    """Initialize databases, config, and background tasks on startup."""
    await db.init_all_dbs()
    await config_loader.load_all()
    await rate_limiter.start()
    await analytics.analytics_tracker.start()
    await post_scheduler.start()
    
    # Log startup
    await db.log_audit_event(
        event_type="system_startup",
        message="Agent Network started",
        severity="info"
    )


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown with forced flushes."""
    await rate_limiter.stop()
    await analytics.analytics_tracker.stop()
    await post_scheduler.stop()
    
    await db.log_audit_event(
        event_type="system_shutdown",
        message="Agent Network stopped",
        severity="info"
    )


# Signal handler for graceful shutdown
def handle_signal(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    asyncio.create_task(shutdown())
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signal)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main SPA with explicit UTF-8 encoding."""
    index_path = static_path / "index.html"
    return HTMLResponse(
        content=index_path.read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8"
    )


@app.get("/api/agents")
async def get_agents():
    """Get all agent configurations."""
    agents = config_loader.get_agents()
    return {"agents": [a.model_dump() for a in agents.values()]}


@app.post("/api/agents")
async def save_agents(request: Request, data: dict):
    """Save agent configurations."""
    try:
        agents = data.get("agents", [])
        await config_loader.save_agents(agents)
        
        await db.log_audit_event(
            event_type="config_updated",
            message="Agents configuration updated",
            severity="info"
        )
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/channels")
async def get_channels():
    """Get all channel configurations."""
    channels = config_loader.get_channels()
    return {"channels": [c.model_dump() for c in channels.values()]}


@app.post("/api/channels")
async def save_channels(request: Request, data: dict):
    """Save channel configurations."""
    try:
        channels = data.get("channels", [])
        await config_loader.save_channels(channels)
        
        # Reload scheduler jobs
        await post_scheduler._load_schedules()
        
        await db.log_audit_event(
            event_type="config_updated",
            message="Channels configuration updated",
            severity="info"
        )
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/archive")
async def get_archive(limit: int = 20, offset: int = 0, status: Optional[str] = None):
    """Get posts archive with server-side pagination."""
    posts = await db.get_posts_paginated(limit=limit, offset=offset, status=status)
    total = await db.get_post_count(status=status)
    
    return {
        "posts": posts,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent audit logs."""
    logs = await db.get_audit_logs(limit=limit)
    return {"logs": logs}


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    total_posts = await db.get_post_count()
    pending_posts = await db.get_post_count(status="pending")
    published_posts = await db.get_post_count(status="published")
    
    # Get cost from analytics (simplified)
    total_cost = 0.0
    tokens_in = 0
    tokens_out = 0
    
    try:
        async with db.get_connection("analytics.db") as conn:
            cursor = await conn.execute("SELECT SUM(cost_usd), SUM(tokens_in), SUM(tokens_out) FROM usage_stats")
            row = await cursor.fetchone()
            total_cost = row[0] or 0.0
            tokens_in = row[1] or 0
            tokens_out = row[2] or 0
    except:
        pass
    
    return {
        "totalPosts": total_posts,
        "pendingPosts": pending_posts,
        "publishedPosts": published_posts,
        "totalCost": f"${total_cost:.4f}",
        "tokensIn": tokens_in,
        "tokensOut": tokens_out
    }


@app.post("/api/generate")
async def generate_post(request: Request, data: dict):
    """Manually trigger post generation workflow."""
    try:
        topic = data.get("topic", "")
        channel_id = data.get("channel_id", "")
        do_search = data.get("do_search", False)
        template_name = data.get("template_name", "")
        
        if not topic or not channel_id:
            raise HTTPException(status_code=400, detail="Topic and channel_id required")
        
        # Get enabled agents
        creators = config_loader.get_enabled_agents("creator")
        validators = config_loader.get_enabled_agents("validator")
        publishers = config_loader.get_enabled_agents("publisher")
        
        if not all([creators, validators, publishers]):
            raise HTTPException(status_code=500, detail="No enabled agents configured")
        
        creator_cfg = creators[0]
        validator_cfg = validators[0]
        publisher_cfg = publishers[0]
        
        # Check API keys
        if not all([creator_cfg.gemini_api_key, 
                   validator_cfg.gemini_api_key, 
                   publisher_cfg.gemini_api_key]):
            raise HTTPException(status_code=500, detail="Missing Gemini API keys")
        
        # Import and run workflow
        from agents.workflow import run_workflow
        
        result = await run_workflow(
            topic=topic,
            channel_id=channel_id,
            creator_cfg=creator_cfg,
            validator_cfg=validator_cfg,
            publisher_cfg=publisher_cfg,
            template_name=template_name,
            do_search=do_search
        )
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "post_id": result.get("post_id"),
            "status": "approved" if result.get("is_approved") else "rejected",
            "tokens_used": result.get("tokens_used", 0),
            "cost_usd": result.get("cost_usd", 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backup")
async def backup_config():
    """Download full configuration backup."""
    agents = config_loader.get_agents()
    channels = config_loader.get_channels()
    
    backup = {
        "backup_date": datetime.utcnow().isoformat(),
        "agents": [a.model_dump() for a in agents.values()],
        "channels": [c.model_dump() for c in channels.values()]
    }
    
    return Response(
        content=json.dumps(backup, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=agent_network_backup.json"
        }
    )


@app.post("/api/restore")
async def restore_config(request: Request, data: dict):
    """Restore configuration from backup."""
    try:
        # Validate backup structure
        if "agents" not in data or "channels" not in data:
            raise HTTPException(status_code=400, detail="Invalid backup format")
        
        # Restore agents
        await config_loader.save_agents(data["agents"])
        
        # Restore channels
        await config_loader.save_channels(data["channels"])
        
        # Reload scheduler
        await post_scheduler._load_schedules()
        
        await db.log_audit_event(
            event_type="config_restored",
            message="Configuration restored from backup",
            severity="info"
        )
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    # Strict localhost binding only
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
        reload=False,
        log_level="info"
    )
