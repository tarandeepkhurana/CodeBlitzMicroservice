"""
Metrics API Route

Provides endpoint for monitoring and metrics.
"""

from dataclasses import asdict
from fastapi import APIRouter
from app.core import get_metrics

router = APIRouter(prefix="/metrics", tags=["monitoring"])


@router.get("")
async def get_all_metrics():
    """
    Get all service metrics.
    
    Returns detailed metrics about:
    - Request counts and response times
    - Generation success/failure rates
    - Cache hit/miss rates
    - LLM API costs
    - Per-language verification pass rates
    - Error counts by type
    """
    metrics = get_metrics()
    snapshot = metrics.get_snapshot()
    return {
        "success": True,
        "data": asdict(snapshot)
    }


@router.get("/summary")
async def get_metrics_summary():
    """
    Get a formatted summary of metrics (human-readable).
    """
    metrics = get_metrics()
    return {
        "success": True,
        "summary": metrics.get_summary_str()
    }


@router.get("/quick")
async def get_quick_metrics():
    """
    Get quick essential metrics only.
    
    Returns a minimal set of metrics for quick monitoring:
    - Total requests
    - Generation success rate
    - Cache hit rate
    - LLM cost
    """
    metrics = get_metrics()
    snapshot = metrics.get_snapshot()
    
    return {
        "success": True,
        "data": {
            "uptime_hours": round(snapshot.uptime_seconds / 3600, 2),
            "requests": {
                "total": snapshot.total_requests,
                "last_hour": snapshot.requests_last_hour
            },
            "generation": {
                "success_rate_percent": snapshot.generation_success_rate,
                "succeeded": snapshot.generations_succeeded,
                "failed": snapshot.generations_failed
            },
            "cache": {
                "hit_rate_percent": snapshot.cache_hit_rate,
                "hits": snapshot.cache_hits,
                "misses": snapshot.cache_misses
            },
            "llm": {
                "total_cost_usd": snapshot.llm_total_cost,
                "calls": snapshot.llm_calls
            },
            "response_time_ms": {
                "avg": snapshot.avg_response_time_ms,
                "p95": snapshot.p95_response_time_ms
            }
        }
    }
