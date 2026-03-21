"""
Question-related API endpoints.

Endpoints:
- POST /api/v1/questions/for-match - Get question for a match
- POST /api/v1/questions/execute - Execute user code against test cases
- POST /api/v1/questions/test-piston - Direct PISTON test (no DB)
"""

import logging
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.api import (
    MatchQuestionRequest,
    MatchQuestionResponse,
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
# Get Question for Match (Steps 1-9 from workflow)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/for-match", response_model=MatchQuestionResponse)
async def get_question_for_match(request: MatchQuestionRequest) -> MatchQuestionResponse:
    """
    Get a question for a 1v1 match.
    """
    # Rate limit check for variant generation (per-user limit)
    check_generation_rate_limit(request.user1_id)
    check_generation_rate_limit(request.user2_id)
    
    avg_rating = (request.user1_rating + request.user2_rating) / 2
    target = 'Easy' if avg_rating < 1200 else 'Medium' if avg_rating < 1600 else 'Hard'
    
    logger.info("=" * 50)
    logger.info(f"[STEP 1] FOR-MATCH REQUEST")
    logger.info(f"         Users: {request.user1_id[:8]}..({request.user1_rating}) vs {request.user2_id[:8]}..({request.user2_rating})")
    logger.info(f"         Target: {target} (avg: {avg_rating})")
    
    try:
        if not settings.OPENAI_API_KEY:
            logger.warning("         ✗ OpenAI not configured, using base questions")
            return await _get_base_question_fallback(request)
        
        service = get_variant_service()
        result = await service.get_variant_for_match(
            user1_id=request.user1_id,
            user2_id=request.user2_id,
            user1_rating=request.user1_rating,
            user2_rating=request.user2_rating,
            match_id=request.match_id
        )
        
        if not result.success:
            logger.error(f"[ERROR] {result.error}")
            return MatchQuestionResponse(success=False, error=result.error or "Failed to generate question")
        
        logger.info("=" * 50)
        logger.info(f"[DONE] Source: {result.source}, Time: {result.generation_time_ms}ms")
        logger.info(f"       Title: \"{result.variant.get('title', 'N/A')[:45]}\"")
        logger.info("=" * 50)
        
        # Handle both "difficulty" (base questions) and "effective_difficulty" (variants)
        difficulty = result.variant.get("difficulty") or result.variant.get("effective_difficulty", "Medium")
        
        return MatchQuestionResponse(
            success=True,
            question={
                "variant_id": result.variant_id,
                "title": result.variant.get("title"),
                "problem_statement": result.variant.get("problem_statement"),
                "input_format": result.variant.get("input_format", {}),
                "output_format": result.variant.get("output_format", {}),
                "constraints": result.variant.get("constraints", []),
                "examples": result.variant.get("examples", []),
                "difficulty": difficulty,
                "function_template": result.variant.get("function_template", {}),
                "stdin_wrappers": result.variant.get("stdin_wrappers", {})
            },
            metadata={
                "source": result.source,
                "base_question_id": result.base_question_id,
                "generation_time_ms": result.generation_time_ms
            }
        )
    
    except Exception as e:
        logger.error(f"[ERROR] get_question_for_match: {e}")
        return MatchQuestionResponse(
            success=False,
            error=str(e)
        )



async def _get_base_question_fallback(request: MatchQuestionRequest) -> MatchQuestionResponse:
    """Fallback to base questions when OpenAI is not configured."""
    db = get_supabase_client()
    
    avg_rating = (request.user1_rating + request.user2_rating) / 2
    target_difficulty = 'Easy' if avg_rating < 1200 else 'Medium' if avg_rating < 1600 else 'Hard'
    
    # Get user history
    user1_history = db.get_user_seen_questions(request.user1_id)
    user2_history = db.get_user_seen_questions(request.user2_id)
    
    seen_base_ids = list(set(
        user1_history["base_question_ids"] + 
        user2_history["base_question_ids"]
    ))
    
    # Get base question
    questions = db.get_base_questions_by_difficulty(
        difficulty=target_difficulty,
        exclude_ids=seen_base_ids,
        limit=1
    )
    
    if not questions:
        questions = db.get_all_base_questions(limit=1)
    
    if not questions:
        return MatchQuestionResponse(
            success=False,
            error="No questions available"
        )
    
    question = questions[0]
    logger.info(f"[FALLBACK] Using base question: {question['title'][:40]}...")
    
    # Note: Not recording history since variant_id FK requires a real variant
    # User history will not track this base question usage
    
    # Filter to only Python/Java/C++ for consistency
    supported_langs = ["python", "java", "cpp"]
    function_template = {k: v for k, v in (question.get("function_template") or {}).items() if k in supported_langs}
    stdin_wrappers = {k: v for k, v in (question.get("stdin_wrappers") or {}).items() if k in supported_langs}
    
    return MatchQuestionResponse(
        success=True,
        question={
            "variant_id": question["id"],  # This is base question ID, client handles differently
            "title": question["title"],
            "problem_statement": question["problem_statement"],
            "input_format": question.get("input_format", {}),
            "output_format": question.get("output_format", {}),
            "constraints": question.get("constraints", []),
            "examples": question.get("examples", []),
            "difficulty": question["difficulty"],
            "function_template": function_template,
            "stdin_wrappers": stdin_wrappers
        },
        metadata={
            "source": "base_question",
            "base_question_id": question.get("leetcode_id"),
            "generation_time_ms": 0
        }
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
