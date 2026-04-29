"""
Agent tools: Search chain with Gemini Native Grounding → Tavily/Serper API (async, circuit breaker) → Cache → Fallback.
No HTML scraping allowed.
"""
import asyncio
import hashlib
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import httpx

CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True)


class CircuitBreaker:
    """Simple circuit breaker for external APIs."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        import time
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
    
    def can_execute(self) -> bool:
        import time
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open


class SearchTools:
    """Search tools with fallback chain."""
    
    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._tavily_breaker = CircuitBreaker()
        self._serper_breaker = CircuitBreaker()
        self._cache_file = CACHE_DIR / "search_cache.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load search cache from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    self._cache = json.load(f)
            except:
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f)
        except:
            pass
    
    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    async def search(self, query: str, use_grounding: bool = True) -> List[Dict[str, Any]]:
        """
        Search with fallback chain:
        1. Gemini Native Grounding (handled in workflow)
        2. Tavily API (async with circuit breaker)
        3. Serper API (async with circuit breaker)
        4. Cache
        5. [SEARCH_FALLBACK]
        """
        cache_key = self._get_cache_key(query)
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try Tavily
        if self._tavily_breaker.can_execute():
            result = await self._tavily_search(query)
            if result:
                self._tavily_breaker.record_success()
                self._cache[cache_key] = result
                self._save_cache()
                return result
            else:
                self._tavily_breaker.record_failure()
        
        # Try Serper
        if self._serper_breaker.can_execute():
            result = await self._serper_search(query)
            if result:
                self._serper_breaker.record_success()
                self._cache[cache_key] = result
                self._save_cache()
                return result
            else:
                self._serper_breaker.record_failure()
        
        # Return fallback marker
        return [{"title": "SEARCH_FALLBACK", "content": "No search results available"}]
    
    async def _tavily_search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Async Tavily search."""
        import os
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "api_key": api_key, "max_results": 5}
                )
                if response.status_code == 200:
                    data = response.json()
                    return [
                        {"title": r.get("title", ""), "content": r.get("content", ""), "url": r.get("url", "")}
                        for r in data.get("results", [])
                    ]
        except Exception:
            pass
        return None
    
    async def _serper_search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Async Serper search."""
        import os
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query},
                    headers={"X-API-KEY": api_key}
                )
                if response.status_code == 200:
                    data = response.json()
                    return [
                        {"title": r.get("title", ""), "content": r.get("snippet", ""), "url": r.get("link", "")}
                        for r in data.get("organic", [])[:5]
                    ]
        except Exception:
            pass
        return None


# Global search tools instance
search_tools = SearchTools()
