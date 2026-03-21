"""
Rate Limiter Module

Simple in-memory rate limiter to prevent API abuse.
Tracks requests per IP and per user_id with configurable limits.
"""

import time
import threading
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Requests per minute by IP
    ip_requests_per_minute: int = 60
    
    # Requests per minute by user_id for expensive operations
    user_generations_per_minute: int = 5
    
    # Global generation limit per minute
    global_generations_per_minute: int = 100
    
    # Enable/disable rate limiting
    enabled: bool = True


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    
    Usage:
        limiter = get_rate_limiter()
        
        # Check if request is allowed
        if not limiter.is_allowed(key="ip:192.168.1.1", limit=60):
            raise HTTPException(429, "Rate limit exceeded")
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._config = RateLimitConfig()
    
    def configure(self, config: RateLimitConfig):
        """Update rate limiter configuration."""
        self._config = config
    
    @property
    def config(self) -> RateLimitConfig:
        """Get current configuration."""
        return self._config
    
    def _cleanup_old_requests(self, key: str, window_seconds: int = 60):
        """Remove requests older than the window."""
        cutoff = time.time() - window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
    
    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """
        Check if a request is allowed for the given key.
        
        Args:
            key: Identifier for rate limiting (e.g., "ip:1.2.3.4" or "user:123")
            limit: Maximum requests allowed in window
            window_seconds: Window duration (default 60s)
        
        Returns:
            True if allowed, False if rate limited
        """
        if not self._config.enabled:
            return True
        
        with self._lock:
            self._cleanup_old_requests(key, window_seconds)
            
            if len(self._requests[key]) >= limit:
                return False
            
            self._requests[key].append(time.time())
            return True
    
    def get_remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        """Get remaining requests for a key."""
        with self._lock:
            self._cleanup_old_requests(key, window_seconds)
            return max(0, limit - len(self._requests[key]))
    
    def get_reset_time(self, key: str, window_seconds: int = 60) -> Optional[float]:
        """Get time until the oldest request expires from the window."""
        with self._lock:
            if not self._requests[key]:
                return None
            oldest = min(self._requests[key])
            return oldest + window_seconds - time.time()
    
    def reset(self, key: Optional[str] = None):
        """Reset rate limits for a specific key or all keys."""
        with self._lock:
            if key:
                self._requests.pop(key, None)
            else:
                self._requests.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Applies IP-based rate limiting to all requests.
    Generation-specific limits are applied in the route handlers.
    """
    
    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Skip rate limiting if disabled
        if not self.limiter.config.enabled:
            return await call_next(request)
        
        # Skip health check endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        ip_key = f"ip:{client_ip}"
        
        # Check IP rate limit
        if not self.limiter.is_allowed(
            key=ip_key,
            limit=self.limiter.config.ip_requests_per_minute
        ):
            remaining = self.limiter.get_remaining(ip_key, self.limiter.config.ip_requests_per_minute)
            reset_time = self.limiter.get_reset_time(ip_key)
            
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests from {client_ip}",
                    "retry_after_seconds": int(reset_time) if reset_time else 60
                },
                headers={
                    "X-RateLimit-Limit": str(self.limiter.config.ip_requests_per_minute),
                    "X-RateLimit-Remaining": str(remaining),
                    "Retry-After": str(int(reset_time) if reset_time else 60)
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        remaining = self.limiter.get_remaining(ip_key, self.limiter.config.ip_requests_per_minute)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.config.ip_requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxies."""
        # Check X-Forwarded-For header (for reverse proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        return request.client.host if request.client else "unknown"


def check_generation_rate_limit(user_id: str) -> bool:
    """
    Check if a user can make a generation request.
    Call this from the /for-match endpoint before starting generation.
    
    Returns:
        True if allowed, raises HTTPException if rate limited
    """
    limiter = get_rate_limiter()
    
    if not limiter.config.enabled:
        return True
    
    user_key = f"user_gen:{user_id}"
    global_key = "global_gen"
    
    # Check user-specific limit
    if not limiter.is_allowed(
        key=user_key,
        limit=limiter.config.user_generations_per_minute
    ):
        reset_time = limiter.get_reset_time(user_key)
        logger.warning(f"Generation rate limit exceeded for user: {user_id}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Generation rate limit exceeded",
                "message": f"User {user_id} has exceeded generation limit",
                "retry_after_seconds": int(reset_time) if reset_time else 60
            }
        )
    
    # Check global limit
    if not limiter.is_allowed(
        key=global_key,
        limit=limiter.config.global_generations_per_minute
    ):
        reset_time = limiter.get_reset_time(global_key)
        logger.warning("Global generation rate limit exceeded")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Service busy",
                "message": "Too many generation requests globally",
                "retry_after_seconds": int(reset_time) if reset_time else 60
            }
        )
    
    return True


# Singleton accessor
_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
