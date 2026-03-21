"""
Metrics and Monitoring Module

Tracks:
- Request counts and response times
- Generation success/failure rates
- Cache hit/miss rates
- LLM API costs
- Per-language verification pass rates
- Error counts by type
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics for API response."""
    uptime_seconds: float
    total_requests: int
    
    # Generation metrics
    generations_attempted: int
    generations_succeeded: int
    generations_failed: int
    generation_success_rate: float
    
    # Cache metrics
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    
    # Verification metrics
    verifications_total: int
    verification_pass_rates: dict  # per language
    
    # LLM metrics
    llm_calls: int
    llm_total_cost: float
    llm_avg_response_time_ms: float
    
    # Error metrics
    errors_by_type: dict
    
    # Response time metrics
    avg_response_time_ms: float
    p95_response_time_ms: float
    
    # Recent stats (last hour)
    requests_last_hour: int
    generations_last_hour: int


class MetricsCollector:
    """
    Thread-safe metrics collector for the microservice.
    
    Usage:
        metrics = get_metrics()
        metrics.record_request_start()
        metrics.record_request_end(success=True)
        metrics.record_generation(success=True, cost=0.015)
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
        self._start_time = time.time()
        self._lock = threading.Lock()
        
        # Request metrics
        self._total_requests = 0
        self._response_times: list[float] = []  # Keep last 1000
        self._request_timestamps: list[float] = []  # For hourly tracking
        
        # Generation metrics
        self._generations_attempted = 0
        self._generations_succeeded = 0
        self._generations_failed = 0
        self._generation_timestamps: list[float] = []
        
        # Cache metrics
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Verification metrics
        self._verifications_total = 0
        self._verification_passed = defaultdict(int)  # per language
        self._verification_failed = defaultdict(int)  # per language
        
        # LLM metrics
        self._llm_calls = 0
        self._llm_total_cost = 0.0
        self._llm_response_times: list[float] = []
        
        # Error tracking
        self._errors_by_type = defaultdict(int)
        
        # Fix attempt tracking
        self._fix_attempts = 0
        self._fix_successes = 0
    
    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.__init__()
            self._initialized = True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Request Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_request(self, duration_ms: float):
        """Record a completed request."""
        with self._lock:
            self._total_requests += 1
            self._request_timestamps.append(time.time())
            self._response_times.append(duration_ms)
            
            # Keep only last 1000 response times
            if len(self._response_times) > 1000:
                self._response_times = self._response_times[-1000:]
            
            # Keep only last hour of timestamps
            hour_ago = time.time() - 3600
            self._request_timestamps = [t for t in self._request_timestamps if t > hour_ago]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Generation Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_generation_attempt(self):
        """Record the start of a generation attempt."""
        with self._lock:
            self._generations_attempted += 1
            self._generation_timestamps.append(time.time())
            
            # Keep only last hour
            hour_ago = time.time() - 3600
            self._generation_timestamps = [t for t in self._generation_timestamps if t > hour_ago]
    
    def record_generation_success(self):
        """Record a successful generation."""
        with self._lock:
            self._generations_succeeded += 1
    
    def record_generation_failure(self, reason: str = "unknown"):
        """Record a failed generation."""
        with self._lock:
            self._generations_failed += 1
            self._errors_by_type[f"generation_{reason}"] += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Cache Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Verification Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_verification(self, language: str, passed: int, total: int):
        """Record verification results for a language."""
        with self._lock:
            self._verifications_total += 1
            self._verification_passed[language] += passed
            self._verification_failed[language] += (total - passed)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LLM Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_llm_call(self, cost: float, duration_ms: float):
        """Record an LLM API call."""
        with self._lock:
            self._llm_calls += 1
            self._llm_total_cost += cost
            self._llm_response_times.append(duration_ms)
            
            # Keep only last 100 response times
            if len(self._llm_response_times) > 100:
                self._llm_response_times = self._llm_response_times[-100:]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Fix Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_fix_attempt(self, success: bool):
        """Record a fix attempt."""
        with self._lock:
            self._fix_attempts += 1
            if success:
                self._fix_successes += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Error Tracking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def record_error(self, error_type: str):
        """Record an error by type."""
        with self._lock:
            self._errors_by_type[error_type] += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Snapshot
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of current metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            
            # Calculate rates
            gen_total = self._generations_succeeded + self._generations_failed
            gen_success_rate = (self._generations_succeeded / gen_total * 100) if gen_total > 0 else 0.0
            
            cache_total = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / cache_total * 100) if cache_total > 0 else 0.0
            
            # Calculate verification pass rates per language
            verification_rates = {}
            for lang in set(list(self._verification_passed.keys()) + list(self._verification_failed.keys())):
                passed = self._verification_passed[lang]
                failed = self._verification_failed[lang]
                total = passed + failed
                verification_rates[lang] = (passed / total * 100) if total > 0 else 0.0
            
            # Calculate response time percentiles
            avg_response = sum(self._response_times) / len(self._response_times) if self._response_times else 0.0
            p95_response = 0.0
            if self._response_times:
                sorted_times = sorted(self._response_times)
                p95_idx = int(len(sorted_times) * 0.95)
                p95_response = sorted_times[min(p95_idx, len(sorted_times) - 1)]
            
            # LLM avg response time
            llm_avg = sum(self._llm_response_times) / len(self._llm_response_times) if self._llm_response_times else 0.0
            
            # Hourly counts
            hour_ago = time.time() - 3600
            requests_last_hour = len([t for t in self._request_timestamps if t > hour_ago])
            generations_last_hour = len([t for t in self._generation_timestamps if t > hour_ago])
            
            return MetricsSnapshot(
                uptime_seconds=uptime,
                total_requests=self._total_requests,
                generations_attempted=self._generations_attempted,
                generations_succeeded=self._generations_succeeded,
                generations_failed=self._generations_failed,
                generation_success_rate=round(gen_success_rate, 2),
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                cache_hit_rate=round(cache_hit_rate, 2),
                verifications_total=self._verifications_total,
                verification_pass_rates={k: round(v, 2) for k, v in verification_rates.items()},
                llm_calls=self._llm_calls,
                llm_total_cost=round(self._llm_total_cost, 4),
                llm_avg_response_time_ms=round(llm_avg, 2),
                errors_by_type=dict(self._errors_by_type),
                avg_response_time_ms=round(avg_response, 2),
                p95_response_time_ms=round(p95_response, 2),
                requests_last_hour=requests_last_hour,
                generations_last_hour=generations_last_hour
            )
    
    def get_summary_str(self) -> str:
        """Get a human-readable summary string."""
        s = self.get_snapshot()
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    METRICS SUMMARY                           ║
╠══════════════════════════════════════════════════════════════╣
║ Uptime: {s.uptime_seconds/3600:.1f}h  |  Requests: {s.total_requests}  |  Last hour: {s.requests_last_hour}
╠══════════════════════════════════════════════════════════════╣
║ GENERATION                                                   
║   Attempted: {s.generations_attempted}  |  Success: {s.generations_succeeded}  |  Failed: {s.generations_failed}
║   Success Rate: {s.generation_success_rate}%
╠══════════════════════════════════════════════════════════════╣
║ CACHE
║   Hits: {s.cache_hits}  |  Misses: {s.cache_misses}  |  Hit Rate: {s.cache_hit_rate}%
╠══════════════════════════════════════════════════════════════╣
║ LLM
║   Calls: {s.llm_calls}  |  Cost: ${s.llm_total_cost:.4f}  |  Avg: {s.llm_avg_response_time_ms:.0f}ms
╠══════════════════════════════════════════════════════════════╣
║ VERIFICATION PASS RATES
║   {', '.join(f'{k}: {v}%' for k, v in s.verification_pass_rates.items()) or 'N/A'}
╠══════════════════════════════════════════════════════════════╣
║ RESPONSE TIMES
║   Avg: {s.avg_response_time_ms:.0f}ms  |  P95: {s.p95_response_time_ms:.0f}ms
╚══════════════════════════════════════════════════════════════╝
"""


# Singleton accessor
_metrics: Optional[MetricsCollector] = None

def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
