"""
Verified question pool ("stock") and its restock worker.

Matches must never wait for an LLM. They take a ready, fully verified variant
off the shelf; this module keeps the shelf stocked.

- The worker wakes every POOL_CHECK_INTERVAL_SECONDS, looks at how many
  validated variants exist per difficulty, and generates at most
  POOL_MAX_PER_CYCLE replacements for the most depleted difficulty.
- When the pool is full it does nothing at all (no LLM calls, no cost).
- POST /api/v1/pool/fill triggers a refill by hand (e.g. before a demo).

Generation is serialized by a lock: at most one pool generation at a time, so
it never competes with a live match generation for PISTON or API budget.
"""

import asyncio
import logging
from typing import Optional

from app.clients.supabase_client import get_supabase_client
from app.config import settings
from app.services.variant_service import get_variant_service

logger = logging.getLogger(__name__)

DIFFICULTIES = ("Easy", "Medium", "Hard")

# Representative rating per difficulty, used as "user context" when a variant
# is generated for the pool instead of for two specific players.
POOL_RATING = {"Easy": 1000, "Medium": 1400, "Hard": 1800}

_generation_lock = asyncio.Lock()
_worker_task: Optional[asyncio.Task] = None


def pool_status() -> dict:
    """How many verified variants are ready per difficulty."""
    counts = get_supabase_client().count_validated_variants_by_difficulty()
    target = settings.POOL_TARGET_PER_DIFFICULTY
    return {
        "target_per_difficulty": target,
        "difficulties": {
            d: {"ready": counts.get(d, 0), "missing": max(0, target - counts.get(d, 0))}
            for d in DIFFICULTIES
        },
        "total_ready": sum(counts.get(d, 0) for d in DIFFICULTIES),
    }


def _most_depleted(status: dict) -> Optional[str]:
    """Difficulty furthest below target, or None when the pool is full."""
    gaps = [(info["missing"], d) for d, info in status["difficulties"].items() if info["missing"] > 0]
    if not gaps:
        return None
    return max(gaps)[1]


async def restock_once(max_generations: int = 1, difficulty: Optional[str] = None) -> dict:
    """Generate up to `max_generations` verified variants for depleted difficulties."""
    generated, failed = [], []

    if _generation_lock.locked():
        logger.info("[POOL] Another generation is running, skipping this cycle")
        return {"skipped": "generation already in progress", "generated": [], "failed": []}

    async with _generation_lock:
        for _ in range(max_generations):
            status = pool_status()
            target = difficulty or _most_depleted(status)
            if not target:
                break

            ready = status["difficulties"][target]["ready"]
            logger.info(f"[POOL] {target} has {ready}/{status['target_per_difficulty']} — generating one...")

            result = await get_variant_service().generate_for_pool(target)
            if result.success:
                generated.append({"difficulty": target, "variant_id": result.variant_id,
                                  "title": (result.variant or {}).get("title")})
                logger.info(f"[POOL] ✓ Added {target}: \"{(result.variant or {}).get('title', '')[:45]}\"")
            else:
                failed.append({"difficulty": target, "error": result.error})
                logger.warning(f"[POOL] ✗ {target} generation failed: {result.error}")
                break  # don't hammer the LLM when something is broken

    return {"generated": generated, "failed": failed, "status": pool_status()}


async def _worker_loop() -> None:
    interval = settings.POOL_CHECK_INTERVAL_SECONDS
    # Let startup (and PISTON prewarm) settle before the first check.
    await asyncio.sleep(min(30, interval))
    while True:
        try:
            status = pool_status()
            if _most_depleted(status):
                await restock_once(max_generations=settings.POOL_MAX_PER_CYCLE)
            else:
                logger.debug(f"[POOL] Full: {status['total_ready']} variants ready")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[POOL] Restock cycle failed: {e}")
        await asyncio.sleep(interval)


async def start_pool_worker() -> None:
    """Start the background restock loop (called from the FastAPI lifespan)."""
    global _worker_task
    if not settings.POOL_ENABLED:
        logger.info("⏸️  Question pool worker disabled (POOL_ENABLED=false)")
        return
    if _worker_task and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_worker_loop())
    logger.info(
        f"✅ Question pool worker started (target {settings.POOL_TARGET_PER_DIFFICULTY} per difficulty, "
        f"checks every {settings.POOL_CHECK_INTERVAL_SECONDS}s)"
    )
    status = pool_status()
    logger.info(
        "   Pool stock: "
        + ", ".join(
            f"{difficulty} {info['ready']}/{status['target_per_difficulty']}"
            for difficulty, info in status["difficulties"].items()
        )
    )


async def stop_pool_worker() -> None:
    """Stop the background restock loop."""
    global _worker_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    _worker_task = None
