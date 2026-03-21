"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Question Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class MatchQuestionRequest(BaseModel):
    """Request from Match Service for a question."""
    match_id: str = Field(..., description="Unique match identifier")
    user1_id: str = Field(..., description="First user's ID")
    user2_id: str = Field(..., description="Second user's ID")
    user1_rating: int = Field(1200, ge=0, le=3500, description="User 1's rating")
    user2_rating: int = Field(1200, ge=0, le=3500, description="User 2's rating")
    preferred_categories: list[str] = Field(default=[], description="Preferred question categories")
    exclude_categories: list[str] = Field(default=[], description="Categories to exclude")


class QuestionInput(BaseModel):
    """Input parameter definition."""
    name: str
    type: str
    description: Optional[str] = None


class QuestionConstraint(BaseModel):
    """Question constraint."""
    text: str
    variable: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None


class QuestionExample(BaseModel):
    """Question example."""
    input: dict
    output: str | int | list
    explanation: Optional[str] = None


class QuestionData(BaseModel):
    """Full question data returned to Match Service."""
    variant_id: str
    title: str
    problem_statement: str
    input_format: dict
    output_format: dict
    constraints: list
    examples: list
    difficulty: str
    function_template: dict = Field(default_factory=dict, description="Code templates per language {python, java, cpp}")
    stdin_wrappers: dict = Field(default_factory=dict, description="Execution wrappers per language")


class MatchQuestionResponse(BaseModel):
    """Response with question for match."""
    success: bool
    question: Optional[QuestionData] = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=lambda: {})


# ═══════════════════════════════════════════════════════════════════════════════
# Code Execution Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class ExecuteCodeRequest(BaseModel):
    """Request to execute user code."""
    question_id: str = Field(..., description="Question/variant ID")
    language: Literal["python", "java", "cpp"] = Field(..., description="Programming language")
    user_code: str = Field(..., description="User's solution code")
    use_generator: bool = Field(False, description="Generate fresh test cases")


class TestResult(BaseModel):
    """Single test case result."""
    test_index: int
    passed: bool
    is_hidden: bool
    stdin: Optional[str] = None  # Only shown if not hidden
    expected_stdout: Optional[str] = None  # Only shown if not hidden
    actual_stdout: Optional[str] = None  # Only shown if not hidden
    error: Optional[str] = None


class ExecuteCodeResponse(BaseModel):
    """Response with execution results."""
    success: bool
    total_tests: int
    passed_tests: int
    results: list[TestResult]
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Health Check Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    services: dict[str, bool]


class PistonHealthResponse(BaseModel):
    """PISTON-specific health check."""
    available: bool
    languages: list[str] = []
    error: Optional[str] = None
