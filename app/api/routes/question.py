"""
Question-related API endpoints.

Endpoints:
- POST /api/v1/questions/for-match - Trigger async variant generation (fire-and-forget)
- GET /api/v1/questions/variant-status/{queue_id} - Check generation status
- POST /api/v1/questions/execute - Execute user code against test cases
- POST /api/v1/questions/test-piston - Direct PISTON test (no DB)
"""

import logging
import asyncio
from typing import Literal
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.schemas.api import (
    MatchQuestionRequest,
    MatchQuestionResponse,
    AsyncGenerationResponse,
    VariantStatusResponse,
    ExecuteCodeRequest,
    ExecuteCodeResponse,
    TestResult
)
from app.clients.supabase_client import get_supabase_client
from app.clients.piston_client import get_piston_client
from app.services.variant_service import get_variant_service
from app.config import settings
from app.core import check_generation_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["Questions"])


# ═══════════════════════════════════════════════════════════════════════════════
# Direct PISTON Test (for debugging)
# ═══════════════════════════════════════════════════════════════════════════════

class DirectTestRequest(BaseModel):
    """Direct code execution request for testing."""
    language: Literal["python", "java", "cpp"]
    code: str
    stdin: str = ""


class DirectTestResponse(BaseModel):
    """Direct execution response."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int = -1  # Default value for when None is returned
    error: str | None = None


@router.post("/test-piston", response_model=DirectTestResponse)
async def test_piston_direct(request: DirectTestRequest) -> DirectTestResponse:
    """
    Direct PISTON test - execute code without database lookup.
    
    Useful for testing PISTON connectivity and code execution.
    """
    try:
        piston = await get_piston_client()
        result = await piston.execute(
            language=request.language,
            code=request.code,
            stdin=request.stdin
        )
        
        return DirectTestResponse(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            error=result.error_message
        )
    except Exception as e:
        logger.error(f"PISTON test failed: {e}")
        return DirectTestResponse(
            success=False,
            stdout="",
            stderr="",
            exit_code=-1,
            error=str(e)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Async Variant Generation (fire-and-forget pattern)
# ═══════════════════════════════════════════════════════════════════════════════

async def _generate_variant_background(
    queue_id: str,
    match_id: str,
    user1_id: str,
    user2_id: str,
    user1_rating: int,
    user2_rating: int,
    exclude_base_question_ids: list[str]
):
    """
    Background task: Generate variant and update queue status.
    
    Called asynchronously after /for-match returns immediately.
    """
    db = get_supabase_client()
    
    try:
        # Ensure queue entry exists (creates if missing - helps during testing)
        existing = db.get_queue_entry(queue_id)
        if not existing:
            # Use first excluded base as shown_base_question_id (Q1 that was served)
            shown_base = exclude_base_question_ids[0] if exclude_base_question_ids else None
            db.create_queue_entry(queue_id, match_id, user1_id, user2_id, "generating", shown_base)
        else:
            db.update_queue_status(queue_id, "generating")
        
        avg_rating = (user1_rating + user2_rating) / 2
        target = 'Easy' if avg_rating < 1200 else 'Medium' if avg_rating < 1600 else 'Hard'
        
        logger.info("=" * 50)
        logger.info(f"[ASYNC] Variant generation started for queue_id={queue_id[:8]}...")
        logger.info(f"        Users: {user1_id[:8]}..({user1_rating}) vs {user2_id[:8]}..({user2_rating})")
        logger.info(f"        Target: {target} (avg: {avg_rating})")
        logger.info(f"        Excluding {len(exclude_base_question_ids)} base question(s)")
        
        if not settings.OPENAI_API_KEY:
            logger.warning("        ✗ OpenAI not configured")
            db.update_queue_status(queue_id, "failed", error_message="OpenAI not configured")
            return
        
        service = get_variant_service()
        result = await service.get_variant_for_match_async(
            user1_id=user1_id,
            user2_id=user2_id,
            user1_rating=user1_rating,
            user2_rating=user2_rating,
            match_id=match_id,
            exclude_base_ids=exclude_base_question_ids
        )
        
        if not result.success:
            logger.error(f"[ASYNC ERROR] {result.error}")
            db.update_queue_status(queue_id, "failed", error_message=result.error)
            return
        
        # Success - update queue with variant ID
        db.update_queue_status(
            queue_id=queue_id,
            status="completed",
            generated_variant_id=result.variant_id,
            generation_base_question_id=result.base_question_id
        )
        
        # Record user history for the variant
        db.record_user_question_v2(user1_id, result.base_question_id, result.variant_id, "variant")
        db.record_user_question_v2(user2_id, result.base_question_id, result.variant_id, "variant")
        
        logger.info("=" * 50)
        logger.info(f"[ASYNC DONE] queue_id={queue_id[:8]}... variant_id={result.variant_id[:8]}...")
        logger.info(f"             Title: \"{result.variant.get('title', 'N/A')[:45]}\"")
        logger.info(f"             Time: {result.generation_time_ms}ms")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"[ASYNC ERROR] Background generation failed: {e}")
        db.update_queue_status(queue_id, "failed", error_message=str(e))


@router.post("/for-match", response_model=AsyncGenerationResponse)
async def trigger_variant_generation(
    request: MatchQuestionRequest,
    background_tasks: BackgroundTasks
) -> AsyncGenerationResponse:
    """
    Trigger async variant generation for a match.
    
    This is a fire-and-forget endpoint:
    - Returns immediately with acknowledgment
    - Variant generation runs in background
    - Node.js polls /variant-status/{queue_id} for completion
    
    Workflow:
    1. Node.js shows base question to users immediately
    2. Node.js creates queue entry and calls this endpoint
    3. This endpoint returns immediately
    4. Background task generates variant
    5. Node.js polls for completion
    6. When ready, Node.js can serve the variant as next question
    """
    logger.info(f"[FOR-MATCH] Received async generation request: queue_id={request.queue_id[:8]}...")
    
    # Rate limit check
    try:
        check_generation_rate_limit(request.user1_id)
        check_generation_rate_limit(request.user2_id)
    except Exception as e:
        return AsyncGenerationResponse(
            success=False,
            queue_id=request.queue_id,
            message=str(e)
        )
    
    # Schedule background task
    background_tasks.add_task(
        _generate_variant_background,
        queue_id=request.queue_id,
        match_id=request.match_id,
        user1_id=request.user1_id,
        user2_id=request.user2_id,
        user1_rating=request.user1_rating,
        user2_rating=request.user2_rating,
        exclude_base_question_ids=request.exclude_base_question_ids
    )
    
    # Return immediately
    return AsyncGenerationResponse(
        success=True,
        queue_id=request.queue_id,
        message="Variant generation started"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Variant Generation Status Check
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/variant-status/{queue_id}", response_model=VariantStatusResponse)
async def get_variant_status(queue_id: str) -> VariantStatusResponse:
    """
    Check the status of an async variant generation.
    
    Called by Node.js to poll for completion.
    
    Status values:
    - pending: Generation not yet started
    - generating: Generation in progress
    - completed: Variant ready (variant_id returned)
    - failed: Generation failed (error returned)
    """
    db = get_supabase_client()
    
    queue_entry = db.get_queue_entry(queue_id)
    
    if not queue_entry:
        raise HTTPException(status_code=404, detail=f"Queue entry not found: {queue_id}")
    
    return VariantStatusResponse(
        success=True,
        queue_id=queue_id,
        status=queue_entry.get("status", "pending"),
        variant_id=queue_entry.get("generated_variant_id"),
        error=queue_entry.get("error_message")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Execute User Code
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/execute", response_model=ExecuteCodeResponse)
async def execute_code(request: ExecuteCodeRequest) -> ExecuteCodeResponse:
    """
    Execute user code against test cases.
    
    Steps:
    1. Fetch question data (stdin_wrappers, test_cases)
    2. Build executable code (wrapper + user code)
    3. Execute each test case via PISTON
    4. Compare outputs and return results
    """
    try:
        db = get_supabase_client()
        piston = await get_piston_client()
        
        # Fetch question data
        question = db.get_base_question_by_id(request.question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
        # Get wrapper for the language
        stdin_wrappers = question.get("stdin_wrappers", {})
        wrapper = stdin_wrappers.get(request.language)
        
        if not wrapper:
            raise HTTPException(
                status_code=400, 
                detail=f"No wrapper available for {request.language}"
            )
        
        # Build executable code (wrapper uses {USER_SOLUTION_CODE} placeholder)
        executable_code = wrapper.replace("{USER_SOLUTION_CODE}", request.user_code)
        
        # Get test cases
        test_cases = question.get("test_cases", [])
        if not test_cases:
            raise HTTPException(status_code=500, detail="No test cases available")
        
        # Execute all test cases in parallel
        piston_results = await piston.execute_test_cases_parallel(
            language=request.language,
            code=executable_code,
            test_cases=test_cases,
            early_exit_on_failure=False
        )
        
        # Build response
        results = []
        passed_count = 0
        
        for r in piston_results:
            if r.passed:
                passed_count += 1
            
            result = TestResult(
                test_index=r.test_index,
                passed=r.passed,
                is_hidden=r.is_hidden,
                error=r.error
            )
            
            # Only show details for non-hidden tests
            if not r.is_hidden:
                result.stdin = r.stdin
                result.expected_stdout = r.expected_stdout
                result.actual_stdout = r.actual_stdout
            
            results.append(result)
        
        return ExecuteCodeResponse(
            success=passed_count == len(test_cases),
            total_tests=len(test_cases),
            passed_tests=passed_count,
            results=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return ExecuteCodeResponse(
            success=False,
            total_tests=0,
            passed_tests=0,
            results=[],
            error=str(e)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Get Question by ID (utility endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/{question_id}")
async def get_question(question_id: str):
    """Get question details by ID."""
    db = get_supabase_client()
    
    question = db.get_base_question_by_id(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Don't expose solution code or hidden test cases
    return {
        "id": question["id"],
        "title": question["title"],
        "difficulty": question["difficulty"],
        "category": question.get("category"),
        "problem_statement": question["problem_statement"],
        "input_format": question.get("input_format"),
        "output_format": question.get("output_format"),
        "constraints": question.get("constraints"),
        "examples": question.get("examples"),
        "function_template": question.get("function_template")
    }
