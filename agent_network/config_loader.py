"""
Configuration loader with asyncio.Lock, aiofiles, Pydantic validation, and atomic save.
Zero-Manual-Config Policy: All settings managed exclusively via web UI.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ValidationError
import aiofiles

CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS_FILE = CONFIG_DIR / "channels.json"
AGENTS_FILE = CONFIG_DIR / "agents.json"
SESSION_SECRET_FILE = CONFIG_DIR / "session_secret.key"


class ChannelConfig(BaseModel):
    """Telegram channel configuration."""
    id: str
    name: str
    chat_id: str
    enabled: bool = True
    posting_schedule: Optional[str] = None  # cron expression
    templates: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)


class ChannelsConfig(BaseModel):
    """Channels configuration container."""
    channels: List[ChannelConfig] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Agent configuration with per-agent API key routing."""
    id: str
    name: str
    role: str  # creator, validator, publisher
    model_name: str = "gemini-1.5-pro"
    enabled: bool = True
    # API key is loaded from environment or secure storage, NEVER stored in config files
    api_key_env_var: str = "GEMINI_API_KEY"
    rpm_limit: int = 60
    tpm_limit: int = 100000
    system_prompt: Optional[str] = None


class AgentsConfig(BaseModel):
    """Agents configuration container."""
    agents: List[AgentConfig] = Field(default_factory=list)


class ConfigLoader:
    """Thread-safe async configuration loader with atomic writes."""
    
    def __init__(self):
        self._channels_lock = asyncio.Lock()
        self._agents_lock = asyncio.Lock()
        self._channels_cache: Optional[ChannelsConfig] = None
        self._agents_cache: Optional[AgentsConfig] = None
    
    async def _read_json(self, path: Path) -> Dict[str, Any]:
        """Read JSON file asynchronously."""
        if not path.exists():
            return {}
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            return json.loads(content) if content.strip() else {}
    
    async def _write_json_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON atomically using temp file + os.replace."""
        tmp_path = path.with_suffix('.tmp')
        bak_path = path.with_suffix('.bak')
        
        # Backup existing file
        if path.exists():
            os.replace(path, bak_path)
        
        try:
            async with aiofiles.open(tmp_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            os.replace(tmp_path, path)
        except Exception as e:
            # Rollback on error
            if bak_path.exists():
                os.replace(bak_path, path)
            raise e
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    
    async def load_channels(self) -> ChannelsConfig:
        """Load channels configuration with lock."""
        async with self._channels_lock:
            data = await self._read_json(CHANNELS_FILE)
            try:
                self._channels_cache = ChannelsConfig(**data)
            except ValidationError:
                self._channels_cache = ChannelsConfig()
            return self._channels_cache
    
    async def save_channels(self, config: ChannelsConfig) -> None:
        """Save channels configuration atomically."""
        async with self._channels_lock:
            await self._write_json_atomic(CHANNELS_FILE, config.model_dump())
            self._channels_cache = config
    
    async def load_agents(self) -> AgentsConfig:
        """Load agents configuration with lock."""
        async with self._agents_lock:
            data = await self._read_json(AGENTS_FILE)
            try:
                self._agents_cache = AgentsConfig(**data)
            except ValidationError:
                self._agents_cache = AgentsConfig()
            return self._agents_cache
    
    async def save_agents(self, config: AgentsConfig) -> None:
        """Save agents configuration atomically."""
        async with self._agents_lock:
            await self._write_json_atomic(AGENTS_FILE, config.model_dump())
            self._agents_cache = config
    
    async def get_channel(self, channel_id: str) -> Optional[ChannelConfig]:
        """Get a specific channel by ID."""
        config = await self.load_channels()
        for ch in config.channels:
            if ch.id == channel_id:
                return ch
        return None
    
    async def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get a specific agent by ID."""
        config = await self.load_agents()
        for ag in config.agents:
            if ag.id == agent_id:
                return ag
        return None
    
    async def get_enabled_agents(self, role: Optional[str] = None) -> List[AgentConfig]:
        """Get all enabled agents, optionally filtered by role."""
        config = await self.load_agents()
        agents = [a for a in config.agents if a.enabled]
        if role:
            agents = [a for a in agents if a.role == role]
        return agents
    
    async def get_enabled_channels(self) -> List[ChannelConfig]:
        """Get all enabled channels."""
        config = await self.load_channels()
        return [c for c in config.channels if c.enabled]


# Global singleton instance
config_loader = ConfigLoader()


async def init_session_secret() -> bytes:
    """Initialize or load session secret key."""
    if not SESSION_SECRET_FILE.exists():
        secret = os.urandom(32)
        async with aiofiles.open(SESSION_SECRET_FILE, 'wb') as f:
            await f.write(secret)
        return secret
    async with aiofiles.open(SESSION_SECRET_FILE, 'rb') as f:
        return await f.read()


async def initialize_configs() -> None:
    """Initialize default configurations if they don't exist."""
    # Initialize channels
    if not CHANNELS_FILE.exists():
        default_channels = ChannelsConfig(channels=[
            ChannelConfig(id="default", name="Default Channel", chat_id="@default_channel")
        ])
        await config_loader.save_channels(default_channels)
    
    # Initialize agents
    if not AGENTS_FILE.exists():
        default_agents = AgentsConfig(agents=[
            AgentConfig(id="creator_1", name="Creator Agent", role="creator"),
            AgentConfig(id="validator_1", name="Validator Agent", role="validator"),
            AgentConfig(id="publisher_1", name="Publisher Agent", role="publisher")
        ])
        await config_loader.save_agents(default_agents)
    
    # Initialize session secret
    await init_session_secret()
