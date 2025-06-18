"""
Redis-based caching service.
"""
import json
from typing import Optional, Any
import redis.asyncio as redis
from app.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """Redis-based caching service."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
    
    async def init(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration."""
        if not self.redis_client:
            return False
        
        try:
            serialized = json.dumps(value)
            await self.redis_client.setex(key, expire, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


# Singleton instance
cache_service = CacheService()

async def init_cache():
    """Initialize cache service."""
    await cache_service.init()

async def get_cache() -> CacheService:
    """Get cache service instance."""
    return cache_service
