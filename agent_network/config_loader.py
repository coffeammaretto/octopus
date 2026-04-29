"""
Configuration loader with asyncio.Lock, aiofiles, and Pydantic validation.
Atomic save via .tmp file + os.replace() for crash safety.
"""
import json
import asyncio
import aiofiles
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ValidationError

CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_DIR.mkdir(exist_ok=True)


class AgentConfig(BaseModel):
    """Agent configuration model."""
    id: str
    name: str
    role: str  # creator, validator, publisher
    gemini_api_key: str
    model: str = "gemini-2.0-flash"
    rpm_limit: int = 60
    tpm_limit: int = 100000
    enabled: bool = True


class ChannelConfig(BaseModel):
    """Telegram channel configuration model."""
    id: str
    name: str
    chat_id: str
    topic_id: Optional[int] = None
    enabled: bool = True
    post_schedule: Optional[str] = None  # cron-like schedule


class ConfigLoader:
    """Thread-safe async configuration loader with atomic saves."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._agents: Dict[str, AgentConfig] = {}
        self._channels: Dict[str, ChannelConfig] = {}
        self._agents_file = CONFIG_DIR / "agents.json"
        self._channels_file = CONFIG_DIR / "channels.json"
    
    async def load_all(self):
        """Load all configurations from disk."""
        async with self._lock:
            await self._load_agents()
            await self._load_channels()
    
    async def _load_agents(self):
        """Load agents configuration."""
        if not self._agents_file.exists():
            # Create default config
            default_agents = [
                {
                    "id": "creator_1",
                    "name": "Content Creator",
                    "role": "creator",
                    "gemini_api_key": "",
                    "model": "gemini-2.0-flash",
                    "rpm_limit": 60,
                    "tpm_limit": 100000,
                    "enabled": True
                },
                {
                    "id": "validator_1",
                    "name": "Content Validator",
                    "role": "validator",
                    "gemini_api_key": "",
                    "model": "gemini-2.0-flash",
                    "rpm_limit": 60,
                    "tpm_limit": 100000,
                    "enabled": True
                },
                {
                    "id": "publisher_1",
                    "name": "Content Publisher",
                    "role": "publisher",
                    "gemini_api_key": "",
                    "model": "gemini-2.0-flash",
                    "rpm_limit": 30,
                    "tpm_limit": 50000,
                    "enabled": True
                }
            ]
            await self._atomic_save(self._agents_file, {"agents": default_agents})
            self._agents = {a["id"]: AgentConfig(**a) for a in default_agents}
        else:
            async with aiofiles.open(self._agents_file, 'r') as f:
                data = json.loads(await f.read())
                self._agents = {a["id"]: AgentConfig(**a) for a in data.get("agents", [])}
    
    async def _load_channels(self):
        """Load channels configuration."""
        if not self._channels_file.exists():
            # Create default empty config
            await self._atomic_save(self._channels_file, {"channels": []})
            self._channels = {}
        else:
            async with aiofiles.open(self._channels_file, 'r') as f:
                data = json.loads(await f.read())
                self._channels = {c["id"]: ChannelConfig(**c) for c in data.get("channels", [])}
    
    async def _atomic_save(self, filepath: Path, data: Dict[str, Any]):
        """Atomically save config via .tmp file + os.replace()."""
        tmp_path = filepath.with_suffix('.tmp')
        content = json.dumps(data, indent=2)
        
        # Use run_in_executor for blocking file I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_file_sync, tmp_path, content)
        await loop.run_in_executor(None, os.replace, str(tmp_path), str(filepath))
    
    def _write_file_sync(self, path: Path, content: str):
        """Synchronous file write for executor."""
        with open(path, 'w') as f:
            f.write(content)
    
    async def save_agents(self, agents: List[Dict[str, Any]]):
        """Save agents configuration atomically."""
        async with self._lock:
            # Validate all agents first
            validated = [AgentConfig(**a) for a in agents]
            self._agents = {a.id: a for a in validated}
            await self._atomic_save(self._agents_file, {"agents": agents})
    
    async def save_channels(self, channels: List[Dict[str, Any]]):
        """Save channels configuration atomically."""
        async with self._lock:
            # Validate all channels first
            validated = [ChannelConfig(**c) for c in channels]
            self._channels = {c.id: c for c in validated}
            await self._atomic_save(self._channels_file, {"channels": channels})
    
    def get_agents(self) -> Dict[str, AgentConfig]:
        """Get all agents."""
        return self._agents.copy()
    
    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get single agent by ID."""
        return self._agents.get(agent_id)
    
    def get_enabled_agents(self, role: Optional[str] = None) -> List[AgentConfig]:
        """Get enabled agents, optionally filtered by role."""
        agents = [a for a in self._agents.values() if a.enabled]
        if role:
            agents = [a for a in agents if a.role == role]
        return agents
    
    def get_channels(self) -> Dict[str, ChannelConfig]:
        """Get all channels."""
        return self._channels.copy()
    
    def get_enabled_channels(self) -> List[ChannelConfig]:
        """Get enabled channels."""
        return [c for c in self._channels.values() if c.enabled]
    
    async def update_agent_api_key(self, agent_id: str, api_key: str):
        """Update single agent API key."""
        async with self._lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                updated = agent.model_dump()
                updated["gemini_api_key"] = api_key
                agents_list = [a.model_dump() for a in self._agents.values()]
                for i, a in enumerate(agents_list):
                    if a["id"] == agent_id:
                        agents_list[i] = updated
                        break
                await self._atomic_save(self._agents_file, {"agents": agents_list})
                self._agents[agent_id] = AgentConfig(**updated)


# Global config loader instance
config_loader = ConfigLoader()
