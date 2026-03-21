# Core module for monitoring, rate limiting, and error handling

from app.core.metrics import get_metrics, MetricsCollector, MetricsSnapshot
from app.core.rate_limiter import (
    get_rate_limiter,
    RateLimiter,
    RateLimitConfig,
    RateLimitMiddleware,
    check_generation_rate_limit
)
from app.core.error_handling import (
    ErrorCategory,
    ServiceError,
    DatabaseError,
    LLMError,
    PistonError,
    GenerationError,
    VerificationError,
    ErrorHandlingMiddleware,
    create_error_response,
    with_fallback,
    safe_get
)

__all__ = [
    # Metrics
    "get_metrics",
    "MetricsCollector",
    "MetricsSnapshot",
    
    # Rate Limiting
    "get_rate_limiter",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitMiddleware",
    "check_generation_rate_limit",
    
    # Error Handling
    "ErrorCategory",
    "ServiceError",
    "DatabaseError",
    "LLMError",
    "PistonError",
    "GenerationError",
    "VerificationError",
    "ErrorHandlingMiddleware",
    "create_error_response",
    "with_fallback",
    "safe_get",
]
