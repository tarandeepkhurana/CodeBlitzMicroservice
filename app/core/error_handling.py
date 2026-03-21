"""
Error Handling Module

Provides global exception handling with:
- Graceful degradation
- Structured error responses
- Error categorization for monitoring
"""

import traceback
from enum import Enum
from typing import Optional, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from app.core.metrics import get_metrics

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for monitoring."""
    
    # Client errors
    VALIDATION = "validation_error"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    BAD_REQUEST = "bad_request"
    
    # External service errors
    DATABASE_ERROR = "database_error"
    LLM_API_ERROR = "llm_api_error"
    PISTON_ERROR = "piston_error"
    
    # Internal errors
    GENERATION_FAILED = "generation_failed"
    VERIFICATION_FAILED = "verification_failed"
    INTERNAL_ERROR = "internal_error"
    
    # Unknown
    UNKNOWN = "unknown_error"


class ServiceError(Exception):
    """
    Base exception for service-level errors.
    Use this for errors that should be exposed to clients with specific messages.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        status_code: int = 500,
        details: Optional[dict] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.status_code = status_code
        self.details = details or {}
        self.recoverable = recoverable


class DatabaseError(ServiceError):
    """Database-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE_ERROR,
            status_code=503,
            details=details,
            recoverable=True
        )


class LLMError(ServiceError):
    """LLM API-related errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.LLM_API_ERROR,
            status_code=503,
            details=details,
            recoverable=True
        )


class PistonError(ServiceError):
    """PISTON execution engine errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.PISTON_ERROR,
            status_code=503,
            details=details,
            recoverable=True
        )


class GenerationError(ServiceError):
    """Variant generation errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.GENERATION_FAILED,
            status_code=422,
            details=details,
            recoverable=True
        )


class VerificationError(ServiceError):
    """Code verification errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VERIFICATION_FAILED,
            status_code=422,
            details=details,
            recoverable=True
        )


def create_error_response(
    status_code: int,
    error_type: str,
    message: str,
    details: Optional[dict] = None,
    request_id: Optional[str] = None
) -> dict:
    """Create a structured error response."""
    response = {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    if request_id:
        response["error"]["request_id"] = request_id
    
    return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    
    Catches all unhandled exceptions and converts them to structured responses.
    Records errors in metrics for monitoring.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""
        metrics = get_metrics()
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # FastAPI HTTP exceptions (including rate limiting)
            error_type = self._categorize_http_exception(e.status_code)
            metrics.record_error(error_type)
            
            # Return the HTTPException as-is (FastAPI handles it)
            raise
            
        except ServiceError as e:
            # Our custom service errors
            logger.error(f"Service error [{e.category}]: {e.message}", exc_info=True)
            metrics.record_error(e.category.value)
            
            return JSONResponse(
                status_code=e.status_code,
                content=create_error_response(
                    status_code=e.status_code,
                    error_type=e.category.value,
                    message=e.message,
                    details=e.details
                )
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            metrics.record_error(ErrorCategory.UNKNOWN.value)
            
            # Log full traceback for debugging
            traceback_str = traceback.format_exc()
            logger.error(f"Traceback:\n{traceback_str}")
            
            # Return generic error to client (don't expose internal details)
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    status_code=500,
                    error_type=ErrorCategory.INTERNAL_ERROR.value,
                    message="An unexpected error occurred. Please try again later.",
                    details={"hint": "If this persists, contact support"}
                )
            )
    
    def _categorize_http_exception(self, status_code: int) -> str:
        """Categorize HTTP exception for metrics."""
        if status_code == 429:
            return ErrorCategory.RATE_LIMITED.value
        elif status_code == 404:
            return ErrorCategory.NOT_FOUND.value
        elif status_code == 422:
            return ErrorCategory.VALIDATION.value
        elif 400 <= status_code < 500:
            return ErrorCategory.BAD_REQUEST.value
        else:
            return ErrorCategory.INTERNAL_ERROR.value


# ═══════════════════════════════════════════════════════════════════════════════
# Graceful Degradation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def with_fallback(
    primary_func,
    fallback_func,
    error_types: tuple = (Exception,),
    log_primary_error: bool = True
):
    """
    Execute primary function, fall back to secondary on failure.
    
    Usage:
        result = await with_fallback(
            primary_func=lambda: expensive_llm_call(),
            fallback_func=lambda: cached_result(),
            error_types=(LLMError, TimeoutError)
        )
    """
    try:
        return await primary_func()
    except error_types as e:
        if log_primary_error:
            logger.warning(f"Primary function failed, using fallback: {e}")
        return await fallback_func()


def safe_get(data: dict, *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Usage:
        value = safe_get(response, "data", "question", "title", default="Unknown")
    """
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return default
        if result is None:
            return default
    return result
