"""
Agent tools module with search chain: Gemini Native Grounding → Tavily/Serper API (async, circuit breaker) → Cache → Fallback.
No HTML scraping allowed. Uses stable HTTP/API clients only.
"""
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx

# Search result cache (simple in-memory with TTL)
_search_cache: Dict[str, Dict] = {}
_cache_lock = asyncio.Lock()
_CACHE_TTL = timedelta(hours=1)


class CircuitBreaker:
    """Simple circuit breaker for external API calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time: Optional[datetime] = None
        self._state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self._state == "open":
            if self._last_failure_time and \
               datetime.utcnow() - self._last_failure_time > timedelta(seconds=self._recovery_timeout):
                self._state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self._state == "half-open":
                self._state = "closed"
                self._failure_count = 0
            return result
        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            if self._failure_count >= self._failure_threshold:
                self._state = "open"
            raise


# Circuit breakers for different search providers
_tavily_breaker = CircuitBreaker()
_serper_breaker = CircuitBreaker()


async def get_from_cache(query: str) -> Optional[List[Dict]]:
    """Get search results from cache if available and not expired."""
    async with _cache_lock:
        if query in _search_cache:
            cached = _search_cache[query]
            if datetime.utcnow() - cached['timestamp'] < _CACHE_TTL:
                return cached['results']
            else:
                del _search_cache[query]
    return None


async def set_cache(query: str, results: List[Dict]) -> None:
    """Store search results in cache."""
    async with _cache_lock:
        _search_cache[query] = {
            'timestamp': datetime.utcnow(),
            'results': results
        }


async def search_with_gemini_grounding(query: str, client) -> Optional[List[Dict]]:
    """
    Use Gemini's native grounding/search capability.
    This is the primary search method.
    """
    try:
        # Gemini with grounding enabled via google_search_retrieval tool
        response = await client.generate_content_async(
            query,
            tools=[{'google_search_retrieval': {}}]
        )
        
        # Extract search metadata if available
        if hasattr(response, 'grounding_metadata') and response.grounding_metadata:
            sources = []
            if hasattr(response.grounding_metadata, 'search_entry_point'):
                # Process grounding metadata
                pass
            return sources
        
        return None
    except Exception as e:
        print(f"Gemini grounding failed: {e}")
        return None


async def search_with_tavily(query: str, api_key: str) -> List[Dict]:
    """Search using Tavily API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5
            }
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for result in data.get('results', []):
            results.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'source': 'tavily'
            })
        return results


async def search_with_serper(query: str, api_key: str) -> List[Dict]:
    """Search using Serper API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            },
            json={
                "q": query,
                "num": 5
            }
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for organic in data.get('organic', []):
            results.append({
                'title': organic.get('title', ''),
                'url': organic.get('link', ''),
                'content': organic.get('snippet', ''),
                'source': 'serper'
            })
        return results


async def perform_search(query: str, tavily_api_key: Optional[str] = None, 
                         serper_api_key: Optional[str] = None) -> List[Dict]:
    """
    Perform search with fallback chain:
    1. Check cache
    2. Gemini native grounding
    3. Tavily API (with circuit breaker)
    4. Serper API (with circuit breaker)
    5. Return empty list (fallback)
    """
    # Step 1: Check cache
    cached = await get_from_cache(query)
    if cached:
        return cached
    
    # Step 2: Try Gemini grounding (requires genai client passed separately)
    # This would be called from the agent tool with the client
    # For now, skip to API fallbacks
    
    # Step 3: Try Tavily
    if tavily_api_key:
        try:
            results = await _tavily_breaker.call(search_with_tavily, query, tavily_api_key)
            if results:
                await set_cache(query, results)
                return results
        except Exception as e:
            print(f"Tavily search failed: {e}")
    
    # Step 4: Try Serper
    if serper_api_key:
        try:
            results = await _serper_breaker.call(search_with_serper, query, serper_api_key)
            if results:
                await set_cache(query, results)
                return results
        except Exception as e:
            print(f"Serper search failed: {e}")
    
    # Step 5: Fallback - return empty
    return []


def format_search_results(results: List[Dict]) -> str:
    """Format search results for LLM consumption."""
    if not results:
        return "[SEARCH_FALLBACK] No search results available."
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    Summary: {r['content']}")
    
    return "\n\n".join(formatted)


class SearchTool:
    """Search tool for agent use with full fallback chain."""
    
    def __init__(self, tavily_api_key: Optional[str] = None, serper_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key
        self.serper_api_key = serper_api_key
    
    async def search(self, query: str) -> str:
        """Perform search and return formatted results."""
        results = await perform_search(
            query,
            tavily_api_key=self.tavily_api_key,
            serper_api_key=self.serper_api_key
        )
        return format_search_results(results)
