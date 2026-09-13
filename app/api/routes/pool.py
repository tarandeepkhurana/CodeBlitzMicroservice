"""
Question pool endpoints.

- GET  /api/v1/pool/status - how many verified questions are ready per difficulty
- POST /api/v1/pool/fill   - generate now (used before a demo, or to bootstrap)
"""

import logging
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.services import pool as pool_service
from app.services import reverify as reverify_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pool", tags=["Pool"])


class FillRequest(BaseModel):
    """How many questions to generate now, and optionally for which difficulty."""
    count: int = Field(default=1, ge=1, le=20)
    difficulty: Optional[Literal["Easy", "Medium", "Hard"]] = None


@router.get("/status")
async def get_pool_status():
    """Verified questions ready to be served, per difficulty."""
    return {"success": True, **pool_service.pool_status()}


@router.post("/fill")
async def fill_pool(request: FillRequest):
    """
    Generate verified variants right now (blocking until done).

    Each one costs LLM calls and takes ~30-90s, so keep `count` small.
    """
    logger.info(f"[POOL] Manual fill requested: count={request.count} difficulty={request.difficulty}")
    result = await pool_service.restock_once(
        max_generations=request.count,
        difficulty=request.difficulty,
    )
    return {"success": not result.get("failed"), **result}


class ReverifyRequest(BaseModel):
    """Re-check questions stored before verification existed."""
    limit: int = Field(default=5, ge=1, le=100)
    use_llm: bool = True          # let the LLM write the missing helper programs
    only_unverified: bool = True  # skip questions that already have a report
    force_helpers: bool = False   # always rewrite the cross-check oracle


@router.post("/reverify")
async def reverify_existing(request: ReverifyRequest, background_tasks: BackgroundTasks):
    """
    Put already-stored questions through the current verification checks.

    Runs in the background (each question takes ~1 minute); poll
    GET /api/v1/pool/reverify/status for progress.
    """
    if reverify_service.progress()["running"]:
        return {"success": False, "message": "A re-verification run is already in progress",
                **reverify_service.progress()}

    background_tasks.add_task(
        reverify_service.reverify_all,
        limit=request.limit,
        use_llm=request.use_llm,
        only_unverified=request.only_unverified,
        force_helpers=request.force_helpers,
    )
    logger.info(f"[REVERIFY] Requested for up to {request.limit} variant(s)")
    return {"success": True, "message": f"Re-verifying up to {request.limit} question(s) in the background"}


@router.get("/reverify/status")
async def reverify_status():
    """Progress of the current (or last) re-verification run."""
    return {"success": True, **reverify_service.progress()}
