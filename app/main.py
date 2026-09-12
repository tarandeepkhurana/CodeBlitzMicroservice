"""
CodeBlitz Microservice - FastAPI Entry Point

A microservice for:
- Generating question variants for 1v1 coding battles
- Executing user code against test cases
- Managing question selection based on user history
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import health, question, metrics, pool
from app.clients.piston_client import get_piston_client, close_piston_client
from app.core import (
    get_rate_limiter,
    RateLimitMiddleware,
    RateLimitConfig,
    ErrorHandlingMiddleware,
    get_metrics
)

# ═══════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════════════════

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S"
)

# Suppress noisy third-party loggers
for noisy_logger in [
    "httpcore", "httpx", "hpack", "openai", "urllib3",
    "httpcore.http11", "httpcore.http2", "httpcore.connection",
    "openai._base_client", "hpack.hpack", "hpack.table"
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan Events
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    
    - On startup: Initialize PISTON client, verify connectivity, prewarm runtimes
    - On shutdown: Clean up connections
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"PISTON URL: {settings.PISTON_URL}")
    
    # Initialize PISTON client
    try:
        piston = await get_piston_client()
        if await piston.health_check():
            logger.info("✅ PISTON API connected")
            
            # Prewarm Java and C++ runtimes (JVM and compiler cold start fix)
            logger.info("🔥 Prewarming runtimes (Java JVM, C++ compiler)...")
            await prewarm_runtimes(piston)
            logger.info("✅ Runtimes prewarmed")
        else:
            logger.warning("⚠️ PISTON API not responding - some features may not work")
    except Exception as e:
        logger.error(f"❌ Failed to connect to PISTON: {e}")
    
    # Keep the verified-question pool stocked in the background
    from app.services.pool import start_pool_worker, stop_pool_worker
    await start_pool_worker()

    yield

    await stop_pool_worker()
    
    # Shutdown
    logger.info("Shutting down...")
    await close_piston_client()
    logger.info("Cleanup complete")


async def prewarm_runtimes(piston):
    """
    Prewarm Java JVM and C++ compiler to avoid cold start timeouts.
    Run simple programs in each language to initialize the runtimes.
    """
    # Simple prewarm programs
    prewarm_codes = {
        "python": "print('warm')",
        "java": """
public class Main {
    public static void main(String[] args) {
        System.out.println("warm");
    }
}
""",
        "cpp": """
#include <iostream>
int main() {
    std::cout << "warm" << std::endl;
    return 0;
}
"""
    }
    
    for lang, code in prewarm_codes.items():
        try:
            result = await piston.execute(lang, code, "")
            if result.success:
                logger.info(f"  ✓ {lang} warmed up")
            else:
                logger.warning(f"  ✗ {lang} prewarm failed: {result.error_message}")
        except Exception as e:
            logger.warning(f"  ✗ {lang} prewarm error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## CodeBlitz Question Variant Generator Microservice
    
    ### Features
    - **Question Selection**: Smart question selection based on user ratings and history
    - **Code Execution**: Execute user code against test cases via PISTON API
    - **Variant Generation**: Generate harder variants of base questions (coming soon)
    
    ### Supported Languages
    - Python 3.10
    - Java 15
    - C++ (GCC 10.2)
    """,
    lifespan=lifespan
)

# ═══════════════════════════════════════════════════════════════════════════════
# CORS Middleware
# ═══════════════════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Rate Limiting Middleware
# ═══════════════════════════════════════════════════════════════════════════════

# Configure rate limiter
rate_limiter = get_rate_limiter()
rate_limiter.configure(RateLimitConfig(
    ip_requests_per_minute=60,           # 60 requests/min per IP
    user_generations_per_minute=5,       # 5 variant generations/min per user
    global_generations_per_minute=100,   # 100 generations/min total
    enabled=True                         # Enable rate limiting
))

app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)
logger.info("✅ Rate limiting middleware enabled")

# ═══════════════════════════════════════════════════════════════════════════════
# Error Handling Middleware
# ═══════════════════════════════════════════════════════════════════════════════

app.add_middleware(ErrorHandlingMiddleware)
logger.info("✅ Error handling middleware enabled")

# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

app.include_router(health.router)
app.include_router(question.router)
app.include_router(metrics.router)
app.include_router(pool.router)


# ═══════════════════════════════════════════════════════════════════════════════
# Root Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Run with Uvicorn (for development)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )
