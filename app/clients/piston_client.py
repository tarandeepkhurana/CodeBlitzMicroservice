"""
PISTON API Client with robust error handling, retries, and parallel execution.

Best Practices Implemented:
- Connection pooling via httpx.AsyncClient
- Exponential backoff retries
- Semaphore-controlled concurrency
- Proper timeout handling
- Health checks before batch execution
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """Result of a single code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int = -1  # Default for error cases
    compile_output: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None


@dataclass
class TestCaseResult:
    """Result of running a test case."""
    test_index: int
    passed: bool
    stdin: str
    expected_stdout: str
    actual_stdout: str
    is_hidden: bool
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class PistonError(Exception):
    """Base exception for PISTON errors."""
    pass


class PistonConnectionError(PistonError):
    """Raised when PISTON API is unreachable."""
    pass


class PistonTimeoutError(PistonError):
    """Raised when execution times out."""
    pass


class PistonExecutionError(PistonError):
    """Raised when code execution fails."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# PISTON Client
# ═══════════════════════════════════════════════════════════════════════════════

class PistonClient:
    """
    Async PISTON API client with connection pooling and retry logic.
    
    Usage:
        async with PistonClient() as client:
            result = await client.execute("python", code, stdin)
            results = await client.execute_parallel(test_cases)
    """
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(settings.PISTON_MAX_CONCURRENT)
        self._base_url = settings.PISTON_URL
    
    async def __aenter__(self):
        """Initialize HTTP client with connection pooling."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.PISTON_EXECUTE_TIMEOUT,
                write=10.0,
                pool=5.0
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
    
    # ═══════════════════════════════════════════════════════════════
    # Health Check
    # ═══════════════════════════════════════════════════════════════
    
    async def health_check(self) -> bool:
        """Check if PISTON API is available."""
        try:
            response = await self._client.get("/api/v2/runtimes", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"PISTON health check failed: {e}")
            return False
    
    async def wait_for_ready(self, max_wait: int = 30, interval: float = 1.0) -> bool:
        """Wait for PISTON to become ready."""
        elapsed = 0
        while elapsed < max_wait:
            if await self.health_check():
                logger.info("PISTON API is ready")
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        logger.error(f"PISTON not ready after {max_wait}s")
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # Single Execution
    # ═══════════════════════════════════════════════════════════════
    
    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def execute(
        self,
        language: str,
        code: str,
        stdin: str = "",
        timeout_ms: int = 10000
    ) -> ExecutionResult:
        """
        Execute code via PISTON API.
        
        Args:
            language: "python", "java", or "cpp"
            code: Complete executable code
            stdin: Input to provide to the program
            timeout_ms: Execution timeout in milliseconds
        
        Returns:
            ExecutionResult with stdout, stderr, exit_code
        """
        # Map language to PISTON format
        lang_config = settings.SUPPORTED_LANGUAGES.get(language)
        if not lang_config:
            raise ValueError(f"Unsupported language: {language}")
        
        piston_language = lang_config.get("language", language)
        version = lang_config["version"]
        filename = lang_config["filename"]
        
        # Use PISTON defaults (like the working batch_test files did)
        payload = {
            "language": piston_language,
            "version": version,
            "files": [{"name": filename, "content": code}],
            "stdin": stdin
        }
        
        try:
            async with self._semaphore:
                response = await self._client.post(
                    "/api/v2/execute",
                    json=payload
                )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"PISTON error {response.status_code}: {error_text}")
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    error_message=f"PISTON API error: {error_text}"
                )
            
            data = response.json()
            
            # Check for compilation errors or timeout (Java, C++)
            compile_result = data.get("compile", {})
            compile_code = compile_result.get("code")
            compile_status = compile_result.get("status")
            
            # Handle compilation timeout (status "TO")
            if compile_status == "TO":
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="Compilation timed out",
                    exit_code=-1,
                    compile_output=compile_result.get("message", ""),
                    error_message="Compilation timeout"
                )
            
            # Handle compilation errors
            if compile_code is not None and compile_code != 0:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=compile_result.get("stderr", ""),
                    exit_code=compile_code if compile_code is not None else -1,
                    compile_output=compile_result.get("output", ""),
                    error_message="Compilation failed"
                )
            
            run_result = data.get("run", {})
            run_code = run_result.get("code")
            run_status = run_result.get("status")
            
            # Output larger than PISTON's output_max_size (process gets killed)
            if run_status in ("OL", "EL"):
                return ExecutionResult(
                    success=False,
                    stdout=run_result.get("stdout", "").strip(),
                    stderr="Output limit exceeded",
                    exit_code=-1,
                    error_message="Output limit exceeded (PISTON output_max_size)"
                )

            # Handle runtime timeout
            if run_status == "TO":
                return ExecutionResult(
                    success=False,
                    stdout=run_result.get("stdout", "").strip(),
                    stderr="Time limit exceeded",
                    exit_code=-1,
                    error_message="Runtime timeout"
                )
            
            return ExecutionResult(
                success=run_code == 0,
                stdout=run_result.get("stdout", "").strip(),
                stderr=run_result.get("stderr", ""),
                exit_code=run_code if run_code is not None else -1,
                compile_output=compile_result.get("output") if compile_result else None
            )
        
        except httpx.TimeoutException as e:
            logger.error(f"PISTON timeout: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error_message="Execution timed out"
            )
        except httpx.ConnectError as e:
            logger.error(f"PISTON connection error: {e}")
            raise PistonConnectionError(f"Cannot connect to PISTON: {e}")
        except Exception as e:
            logger.error(f"PISTON unexpected error: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error_message=str(e)
            )
    
    # ═══════════════════════════════════════════════════════════════
    # Parallel Execution (for test cases)
    # ═══════════════════════════════════════════════════════════════
    
    async def execute_test_case(
        self,
        index: int,
        language: str,
        code: str,
        test_case: dict
    ) -> TestCaseResult:
        """Execute a single test case."""
        stdin = test_case.get("stdin", "")
        expected = test_case.get("expected_stdout", "")
        is_hidden = test_case.get("is_hidden", False)
        
        result = await self.execute(language, code, stdin)
        
        passed = result.success and result.stdout == expected
        
        return TestCaseResult(
            test_index=index,
            passed=passed,
            stdin=stdin,
            expected_stdout=expected,
            actual_stdout=result.stdout,
            is_hidden=is_hidden,
            error=result.error_message or result.stderr if not result.success else None
        )
    
    async def execute_test_cases_parallel(
        self,
        language: str,
        code: str,
        test_cases: list[dict],
        early_exit_on_failure: bool = False
    ) -> list[TestCaseResult]:
        """
        Execute multiple test cases in parallel with controlled concurrency.
        
        Args:
            language: Programming language
            code: Complete executable code with wrapper
            test_cases: List of {"stdin": ..., "expected_stdout": ..., "is_hidden": ...}
            early_exit_on_failure: Stop on first failure (useful for quick feedback)
        
        Returns:
            List of TestCaseResult
        """
        # Ensure PISTON is ready before batch execution
        if not await self.health_check():
            logger.warning("PISTON health check failed, waiting for ready...")
            if not await self.wait_for_ready(max_wait=10):
                raise PistonConnectionError("PISTON API not available")
        
        results: list[TestCaseResult] = []
        
        if early_exit_on_failure:
            # Sequential execution with early exit
            for i, tc in enumerate(test_cases):
                result = await self.execute_test_case(i, language, code, tc)
                results.append(result)
                if not result.passed:
                    break
        else:
            # Parallel execution with semaphore-controlled concurrency
            tasks = [
                self.execute_test_case(i, language, code, tc)
                for i, tc in enumerate(test_cases)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_results = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    processed_results.append(TestCaseResult(
                        test_index=i,
                        passed=False,
                        stdin=test_cases[i].get("stdin", ""),
                        expected_stdout=test_cases[i].get("expected_stdout", ""),
                        actual_stdout="",
                        is_hidden=test_cases[i].get("is_hidden", False),
                        error=str(r)
                    ))
                else:
                    processed_results.append(r)
            results = processed_results
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton Client for Dependency Injection
# ═══════════════════════════════════════════════════════════════════════════════

_client_instance: Optional[PistonClient] = None


async def get_piston_client() -> PistonClient:
    """Get PISTON client instance (for FastAPI dependency injection)."""
    global _client_instance
    if _client_instance is None:
        _client_instance = PistonClient()
        await _client_instance.__aenter__()
    return _client_instance


async def close_piston_client():
    """Close PISTON client (call on shutdown)."""
    global _client_instance
    if _client_instance:
        await _client_instance.__aexit__(None, None, None)
        _client_instance = None
