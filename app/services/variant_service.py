"""
Variant Generation Service.

Handles the complete workflow for generating question variants:
1. Select base question (avoiding user history)
2. Generate variant via GPT-4o
3. Verify solution against test cases via PISTON
4. Retry/fix if verification fails
5. Store validated variant in database
"""

import logging
import time
import hashlib
import json
from typing import Optional
from dataclasses import dataclass

from app.clients.openai_client import get_openai_client
from app.clients.piston_client import get_piston_client
from app.clients.supabase_client import get_supabase_client
from app.config import settings
from app.core import get_metrics

logger = logging.getLogger(__name__)
metrics = get_metrics()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VariantResult:
    """Result of variant generation."""
    success: bool
    variant: Optional[dict] = None
    variant_id: Optional[str] = None
    base_question_id: Optional[str] = None
    generation_time_ms: int = 0
    source: str = "generated"  # "generated", "cached", "base_question"
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of solution verification."""
    success: bool
    passed_tests: int = 0
    total_tests: int = 0
    failures: list = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Variant Service
# ═══════════════════════════════════════════════════════════════════════════════

class VariantService:
    """
    Service for generating and verifying question variants.
    
    Flow:
    1. Check cache for validated variant
    2. If no cache: select base question, generate variant
    3. Verify solution via PISTON
    4. If fails: send failures to LLM for fix, re-verify (up to 2 fix attempts)
    5. If fix fails: try fresh generation
    6. Store validated variant
    """
    
    MAX_GENERATION_ATTEMPTS = 2  # Fresh generation attempts per base question
    MAX_FIX_ATTEMPTS = 2  # Fix attempts per failed generation
    
    def __init__(self):
        self.openai = get_openai_client()
        self.db = get_supabase_client()
    
    def _fix_java_wrapper_structure(self, stdin_wrappers: dict, base_question: dict = None) -> dict:
        """
        Auto-fix Java wrapper issues:
        1. Move {user_solution} outside Main class if it's inside
        2. Add System.exit(0) to prevent timeout (JVM hanging on stdin)
        3. REJECT Scanner usage - will be marked as error for LLM to fix
        
        Returns modified stdin_wrappers and sets a flag if Scanner was detected.
        """
        java_wrapper = stdin_wrappers.get("java", "")
        if not java_wrapper or "{user_solution}" not in java_wrapper:
            return stdin_wrappers
        
        stdin_wrappers = stdin_wrappers.copy()
        
        # FIX 1: Move {user_solution} outside Main class
        placeholder_pos = java_wrapper.find("{user_solution}")
        last_brace_pos = java_wrapper.rfind("}")
        
        if placeholder_pos < last_brace_pos:
            logger.warning("⚠️ Auto-fixing Java wrapper: moving {user_solution} outside Main class")
            java_wrapper = java_wrapper.replace("{user_solution}", "")
            java_wrapper = java_wrapper.rstrip() + "\n\n{user_solution}"
            stdin_wrappers["java"] = java_wrapper
            logger.info("✓ Java wrapper auto-fixed (structure)")
        
        # FIX 2: Add System.exit(0) if not present - CRITICAL to prevent timeout!
        if "System.exit(0)" not in java_wrapper and "System.exit(0);" not in java_wrapper:
            logger.warning("⚠️ Auto-fixing Java wrapper: adding System.exit(0) to prevent timeout")
            # Find the last System.out.println in main() and add System.exit(0) after it
            # Look for pattern: System.out.println(...); followed by whitespace/newline then }
            import re
            # Add System.exit(0); before the closing brace of main() method
            # Pattern: find "System.out.println" followed eventually by "}}" (end of main, end of Main class)
            # Simpler approach: find last println before {user_solution} and add exit after
            
            # Find the position just before {user_solution}
            user_sol_pos = java_wrapper.find("{user_solution}")
            if user_sol_pos > 0:
                # Find the last closing brace of Main class (before {user_solution})
                main_part = java_wrapper[:user_sol_pos]
                # Find last System.out.println in main part
                last_println = main_part.rfind("System.out.println")
                if last_println != -1:
                    # Find the semicolon after this println
                    semicolon_pos = main_part.find(";", last_println)
                    if semicolon_pos != -1:
                        # Insert System.exit(0); after the println
                        java_wrapper = (
                            java_wrapper[:semicolon_pos + 1] + 
                            "\n        System.exit(0);" + 
                            java_wrapper[semicolon_pos + 1:]
                        )
                        stdin_wrappers["java"] = java_wrapper
                        logger.info("✓ Java wrapper auto-fixed (added System.exit(0))")
        
        # FIX 3: Detect Scanner - replace with BufferedReader pattern if simple
        if "Scanner" in java_wrapper and "new Scanner(System.in)" in java_wrapper:
            logger.warning("⚠️ Java wrapper uses Scanner - attempting auto-conversion to BufferedReader")
            
            # Simple pattern replacement for common Scanner usage
            fixed = java_wrapper
            
            # Add IOException import if not present
            if "throws IOException" not in fixed and "throws Exception" not in fixed:
                fixed = fixed.replace(
                    "public static void main(String[] args) {",
                    "public static void main(String[] args) throws IOException {"
                )
                fixed = fixed.replace(
                    "public static void main(String[] args){",
                    "public static void main(String[] args) throws IOException {"
                )
            
            # Replace Scanner creation with BufferedReader
            fixed = fixed.replace(
                "Scanner sc = new Scanner(System.in);",
                "BufferedReader br = new BufferedReader(new InputStreamReader(System.in));"
            )
            fixed = fixed.replace(
                "Scanner scanner = new Scanner(System.in);",
                "BufferedReader br = new BufferedReader(new InputStreamReader(System.in));"
            )
            
            # Replace common Scanner methods - this is approximate
            # sc.nextInt() → Integer.parseInt(br.readLine().trim())
            # sc.nextLine() → br.readLine()
            # For complex cases, LLM will need to fix
            
            # Remove import java.util.Scanner if present and add BufferedReader imports
            if "import java.util.Scanner;" in fixed:
                fixed = fixed.replace("import java.util.Scanner;", "")
            if "import java.io.*;" not in fixed:
                # Add import after first import statement
                if "import java.util.*;" in fixed:
                    fixed = fixed.replace("import java.util.*;", "import java.util.*;\nimport java.io.*;")
                elif "import " in fixed:
                    first_import = fixed.find("import ")
                    end_of_first_import = fixed.find(";", first_import)
                    fixed = fixed[:end_of_first_import+1] + "\nimport java.io.*;" + fixed[end_of_first_import+1:]
            
            if fixed != java_wrapper:
                stdin_wrappers["java"] = fixed
                logger.info("✓ Java wrapper auto-converted Scanner→BufferedReader (partial)")
        
        return stdin_wrappers
    
    async def get_variant_for_match(
        self,
        user1_id: str,
        user2_id: str,
        user1_rating: int,
        user2_rating: int,
        match_id: Optional[str] = None
    ) -> VariantResult:
        """
        Get a question variant for a 1v1 match.
        
        Steps per MICROSERVICE_WORKFLOW.excalidraw:
        1. Get user context (seen questions)
        2. Check variant cache
        3. Select base question
        4. Generate variant via LLM
        5. Verify solution
        6. Store and return
        """
        start_time = time.time()
        
        try:
            # STEP 2: Get user history
            logger.info("[STEP 2] Fetching user history...")
            user1_history = self.db.get_user_seen_questions(user1_id)
            user2_history = self.db.get_user_seen_questions(user2_id)
            
            seen_base_ids = list(set(
                user1_history["base_question_ids"] + 
                user2_history["base_question_ids"]
            ))
            seen_variant_ids = list(set(
                user1_history["variant_ids"] + 
                user2_history["variant_ids"]
            ))
            
            logger.info(f"         Exclusions: {len(seen_base_ids)} base, {len(seen_variant_ids)} variants")
            
            # Calculate target difficulty
            avg_rating = (user1_rating + user2_rating) / 2
            target_difficulty = self._get_target_difficulty(avg_rating)
            logger.info(f"         Target: {target_difficulty} (avg rating: {avg_rating})")
            
            # STEP 3: Check variant cache
            logger.info("[STEP 3] Checking variant cache...")
            cached = self.db.get_cached_variant(
                difficulty=target_difficulty,
                exclude_ids=seen_variant_ids
            )
            
            if cached:
                elapsed = int((time.time() - start_time) * 1000)
                logger.info(f"         ✓ CACHE HIT: \"{cached.get('title', 'N/A')[:40]}\"")
                
                # Track cache hit
                metrics.record_cache_hit()
                metrics.record_request(elapsed)
                
                self._record_user_history(user1_id, user2_id, cached["base_question_id"], cached["id"])
                
                return VariantResult(
                    success=True,
                    variant=cached,
                    variant_id=cached["id"],
                    base_question_id=cached["base_question_id"],
                    generation_time_ms=elapsed,
                    source="cached"
                )
            
            logger.info("         ✗ Cache miss")
            metrics.record_cache_miss()
            
            # STEP 4: Select base question
            logger.info("[STEP 4] Selecting base question...")
            base_questions = self.db.get_base_questions_by_difficulty(
                difficulty=target_difficulty,
                exclude_ids=seen_base_ids,
                limit=3
            )
            
            if not base_questions:
                base_questions = self.db.get_all_base_questions(limit=1)
            
            if not base_questions:
                return VariantResult(success=False, error="No base questions available")
            
            logger.info(f"         Found {len(base_questions)} candidates: {[q['title'][:25] for q in base_questions]}")
            
            # STEP 5-7: Generate & verify
            for base_question in base_questions:
                logger.info(f"[STEP 5] Generating variant from: \"{base_question['title'][:40]}\"")
                
                result = await self._generate_and_verify_variant(
                    base_question=base_question,
                    user1_rating=user1_rating,
                    user2_rating=user2_rating
                )
                
                if result.success:
                    elapsed = int((time.time() - start_time) * 1000)
                    result.generation_time_ms = elapsed
                    
                    # Track successful generation
                    metrics.record_generation_success()
                    metrics.record_request(elapsed)
                    
                    logger.info("[STEP 8] Recording user history...")
                    self._record_user_history(user1_id, user2_id, result.base_question_id, result.variant_id)
                    
                    return result
            
            # All generation attempts failed - return base question (no history recorded for base)
            elapsed = int((time.time() - start_time) * 1000)
            logger.warning("[FALLBACK] Using base question directly (no variant generated)")
            
            # Track generation failure (fallback to base)
            metrics.record_generation_failure("fallback_to_base")
            metrics.record_request(elapsed)
            
            base = base_questions[0]
            # Note: Don't record history for base question - it would violate FK constraint
            # User will see this base question again, which is acceptable for fallback
            
            return VariantResult(
                success=True,
                variant=self._format_base_as_variant(base),
                variant_id=base["id"],
                base_question_id=base["id"],
                generation_time_ms=elapsed,
                source="base_question"
            )
            
        except Exception as e:
            logger.error(f"[ERROR] Variant generation failed: {str(e)[:80]}")
            elapsed = int((time.time() - start_time) * 1000)
            
            # Track generation error
            metrics.record_generation_failure("exception")
            metrics.record_error("generation_exception")
            metrics.record_request(elapsed)
            
            return VariantResult(
                success=False,
                generation_time_ms=elapsed,
                error=str(e)
            )
    
    async def get_variant_for_match_async(
        self,
        user1_id: str,
        user2_id: str,
        user1_rating: int,
        user2_rating: int,
        match_id: Optional[str] = None,
        exclude_base_ids: Optional[list[str]] = None
    ) -> VariantResult:
        """
        Generate a variant for async queue processing.
        
        Same as get_variant_for_match but:
        - Accepts exclude_base_ids (bases shown to users or used for previous variants)
        - Does NOT record user history (caller handles it)
        - Does NOT use cache (always generate fresh)
        """
        start_time = time.time()
        
        try:
            # Get user history
            logger.info("[ASYNC S2] Fetching user history...")
            user1_history = self.db.get_user_seen_questions(user1_id)
            user2_history = self.db.get_user_seen_questions(user2_id)
            
            seen_base_ids = list(set(
                user1_history["base_question_ids"] + 
                user2_history["base_question_ids"]
            ))
            
            # Add explicitly excluded base questions to exclusion list
            if exclude_base_ids:
                for base_id in exclude_base_ids:
                    if base_id not in seen_base_ids:
                        seen_base_ids.append(base_id)
            
            logger.info(f"           Exclusions: {len(seen_base_ids)} base questions")
            
            # Calculate target difficulty
            avg_rating = (user1_rating + user2_rating) / 2
            target_difficulty = self._get_target_difficulty(avg_rating)
            logger.info(f"           Target: {target_difficulty} (avg rating: {avg_rating})")
            
            # Select base question (always generate fresh, no cache)
            logger.info("[ASYNC S3] Selecting base question (excluding shown)...")
            base_questions = self.db.get_base_questions_by_difficulty(
                difficulty=target_difficulty,
                exclude_ids=seen_base_ids,
                limit=3
            )
            
            if not base_questions:
                # Try all difficulties if none found
                base_questions = self.db.get_all_base_questions(limit=3)
                # Filter out excluded
                base_questions = [q for q in base_questions if q["id"] not in seen_base_ids]
            
            if not base_questions:
                return VariantResult(success=False, error="No base questions available for generation")
            
            logger.info(f"           Found {len(base_questions)} candidates: {[q['title'][:25] for q in base_questions]}")
            
            # Generate & verify
            for base_question in base_questions:
                logger.info(f"[ASYNC S4] Generating variant from: \"{base_question['title'][:40]}\"")
                
                result = await self._generate_and_verify_variant(
                    base_question=base_question,
                    user1_rating=user1_rating,
                    user2_rating=user2_rating
                )
                
                if result.success:
                    elapsed = int((time.time() - start_time) * 1000)
                    result.generation_time_ms = elapsed
                    
                    # Track successful generation
                    metrics.record_generation_success()
                    metrics.record_request(elapsed)
                    
                    # Note: caller records user history, not us
                    return result
            
            # All generation attempts failed
            elapsed = int((time.time() - start_time) * 1000)
            logger.error("[ASYNC ERROR] All variant generation attempts failed")
            
            metrics.record_generation_failure("all_attempts_failed")
            metrics.record_request(elapsed)
            
            return VariantResult(
                success=False,
                generation_time_ms=elapsed,
                error="Failed to generate variant after multiple attempts"
            )
            
        except Exception as e:
            logger.error(f"[ASYNC ERROR] Variant generation failed: {str(e)[:80]}")
            elapsed = int((time.time() - start_time) * 1000)
            
            metrics.record_generation_failure("exception")
            metrics.record_error("generation_exception")
            metrics.record_request(elapsed)
            
            return VariantResult(
                success=False,
                generation_time_ms=elapsed,
                error=str(e)
            )
    
    async def _generate_and_verify_variant(
        self,
        base_question: dict,
        user1_rating: int,
        user2_rating: int
    ) -> VariantResult:
        """
        Generate a variant and verify it passes all test cases.
        
        Flow:
        1. Generate variant via LLM
        2. Verify all 3 languages
        3. If fails → send failures to LLM for fix → re-verify
        4. Repeat fix up to MAX_FIX_ATTEMPTS times
        5. If still fails → try fresh generation
        """
        import os
        from datetime import datetime
        
        # Create debug folder
        debug_dir = "debug_variants"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        user_context = {
            "avg_rating": (user1_rating + user2_rating) / 2,
            "skill_level": self._get_skill_level(user1_rating, user2_rating)
        }
        
        for gen_attempt in range(self.MAX_GENERATION_ATTEMPTS):
            logger.info(f"         LLM generation {gen_attempt + 1}/{self.MAX_GENERATION_ATTEMPTS}...")
            
            # Track generation attempt
            metrics.record_generation_attempt()
            
            # Generate variant via LLM
            variant_data = await self.openai.generate_variant(
                base_question=base_question,
                user_context=user_context
            )
            
            if not variant_data:
                logger.warning(f"         ✗ LLM returned None")
                continue
            
            test_count = len(variant_data.get('test_cases', []))
            logger.info(f"         ✓ Generated with {test_count} test cases")
            
            # DEBUG: Save LLM response
            debug_file = f"{debug_dir}/{timestamp}_gen{gen_attempt+1}.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(variant_data, f, indent=2)
            logger.info(f"         📁 Saved: {debug_file}")
            
            # Verify initial generation
            logger.info("[STEP 6] Verifying solution via PISTON...")
            verification = await self._verify_variant(variant_data, base_question)
            
            # DEBUG: Save verification failures
            if not verification.success and verification.failures:
                fail_file = f"{debug_dir}/{timestamp}_gen{gen_attempt+1}_failures.json"
                with open(fail_file, "w", encoding="utf-8") as f:
                    json.dump(verification.failures, f, indent=2)
                logger.info(f"         📁 Failures saved: {fail_file}")
            
            if verification.success:
                store_result = await self._store_and_return_variant(variant_data, base_question)
                if store_result.success:
                    return store_result
                # Storage failed (validation issue) - continue to fix loop
                logger.warning(f"         Storage validation failed, trying fixes...")
            
            # Verification failed - try fix loop
            failed_langs = set(f["language"] for f in (verification.failures or []))
            logger.warning(f"         ✗ Verification FAILED: {verification.passed_tests}/{verification.total_tests} (langs: {failed_langs})")
            
            # FIX LOOP: Send failures to LLM, re-verify
            current_variant = variant_data
            for fix_attempt in range(self.MAX_FIX_ATTEMPTS):
                logger.info(f"         Sending failures to LLM for fix ({fix_attempt + 1}/{self.MAX_FIX_ATTEMPTS})...")
                
                fixed_variant = await self.openai.fix_variant(
                    variant=current_variant,
                    failures=verification.failures or [],
                    base_question=base_question  # Pass base for context
                )
                
                if not fixed_variant:
                    logger.warning(f"         ✗ LLM fix returned None")
                    metrics.record_fix_attempt(success=False)
                    break  # Can't fix, try fresh generation
                
                # VALIDATION: Ensure LLM didn't remove {user_solution} placeholders
                placeholder_error = self._validate_placeholders(fixed_variant)
                if placeholder_error:
                    logger.warning(f"         ✗ Fix broke wrappers: {placeholder_error}")
                    logger.info("         Keeping previous version, trying next fix...")
                    continue  # Skip this fix, try another
                
                # Re-verify fixed variant
                logger.info("         Re-verifying fixed variant...")
                
                # Debug: Log what we got from fix
                tc_count = len(fixed_variant.get('test_cases', []))
                sol_langs = list(fixed_variant.get('solution_code', {}).keys())
                wrap_langs = list(fixed_variant.get('stdin_wrappers', {}).keys())
                logger.info(f"         Fixed variant: {tc_count} tests, solutions: {sol_langs}, wrappers: {wrap_langs}")
                
                verification = await self._verify_variant(fixed_variant, base_question)
                
                if verification.success:
                    logger.info(f"         ✓ Fix successful! {verification.passed_tests}/{verification.total_tests}")
                    metrics.record_fix_attempt(success=True)
                    store_result = await self._store_and_return_variant(fixed_variant, base_question)
                    if store_result.success:
                        return store_result
                    # Storage failed (validation) - continue to next fix or fresh generation
                    logger.warning(f"         Storage validation failed after fix, continuing...")
                    continue
                
                # Still failing - log details and track
                metrics.record_fix_attempt(success=False)
                if verification.total_tests == 0:
                    logger.error(f"         ✗ Verification returned 0 tests! Error: {verification.error}")
                else:
                    failed_langs = set(f["language"] for f in (verification.failures or []))
                    logger.warning(f"         ✗ Still failing: {verification.passed_tests}/{verification.total_tests} (langs: {failed_langs})")
                current_variant = fixed_variant  # Use fixed version for next fix attempt
            
            logger.info("         Fix attempts exhausted, trying fresh generation...")
        
        return VariantResult(success=False, error=f"Generation/fix failed after {self.MAX_GENERATION_ATTEMPTS} generations × {self.MAX_FIX_ATTEMPTS} fixes")
    
    async def _store_and_return_variant(self, variant_data: dict, base_question: dict) -> VariantResult:
        """Store verified variant and return result."""
        # VALIDATION: Ensure we have exactly 10 test cases
        test_count = len(variant_data.get("test_cases", []))
        if test_count < 10:
            logger.error(f"         ❌ Cannot store: only {test_count} test cases (need 10)")
            return VariantResult(success=False, error=f"Only {test_count} test cases, need 10")
        
        logger.info("[STEP 7] Storing validated variant...")
        stored = self._store_variant(
            variant_data=variant_data,
            base_question_id=base_question["id"],
            base_difficulty=base_question["difficulty"]
        )
        
        if stored:
            logger.info(f"         ✓ Stored: {stored['id'][:8]}...")
            return VariantResult(
                success=True,
                variant=stored,
                variant_id=stored["id"],
                base_question_id=base_question["id"],
                source="generated"
            )
        else:
            logger.error("         ✗ DB storage failed")
            return VariantResult(success=False, error="DB storage failed")
    
    async def _verify_variant(self, variant_data: dict, base_question: dict = None) -> VerificationResult:
        """
        Verify variant's solution passes all test cases for ALL 3 languages.
        
        Uses the variant's own stdin_wrappers with solution_code inserted.
        All languages must pass all tests for verification to succeed.
        
        Args:
            variant_data: The variant to verify
            base_question: Original base question (for auto-fix reference)
        """
        test_cases = variant_data.get("test_cases", [])
        solution_code = variant_data.get("solution_code", {})
        stdin_wrappers = variant_data.get("stdin_wrappers", {})
        
        if not test_cases:
            return VerificationResult(success=False, error="No test cases in variant")
        
        if not stdin_wrappers:
            return VerificationResult(success=False, error="No stdin_wrappers in variant")
        
        # Pre-validate and auto-fix Java wrapper structure
        stdin_wrappers = self._fix_java_wrapper_structure(stdin_wrappers, base_question)
        
        # Languages to verify (all 3 must pass)
        languages = ["python", "java", "cpp"]
        
        try:
            piston = await get_piston_client()
            import time
            start_time = time.time()
            
            all_failures = []
            total_passed = 0
            total_tests = 0
            
            for lang in languages:
                solution = solution_code.get(lang)
                wrapper = stdin_wrappers.get(lang)
                
                if not solution:
                    return VerificationResult(
                        success=False, 
                        error=f"No {lang} solution in variant"
                    )
                if not wrapper:
                    return VerificationResult(
                        success=False,
                        error=f"No {lang} stdin_wrapper in variant"
                    )
                
                # Build executable: replace placeholder with solution
                # Support both {USER_SOLUTION_CODE} and {user_solution} placeholders
                if "{USER_SOLUTION_CODE}" in wrapper:
                    executable = wrapper.replace("{USER_SOLUTION_CODE}", solution)
                elif "{user_solution}" in wrapper:
                    executable = wrapper.replace("{user_solution}", solution)
                else:
                    return VerificationResult(
                        success=False,
                        error=f"No placeholder in {lang} stdin_wrapper"
                    )
                
                # Execute all test cases for this language
                results = await piston.execute_test_cases_parallel(
                    language=lang,
                    code=executable,
                    test_cases=test_cases,
                    early_exit_on_failure=False
                )
                
                passed = sum(1 for r in results if r.passed)
                total_passed += passed
                total_tests += len(test_cases)
                
                # Track verification result for this language
                metrics.record_verification(lang, passed, len(test_cases))
                
                # Collect failures with FULL details for debugging
                for r in results:
                    if not r.passed:
                        all_failures.append({
                            "language": lang,
                            "index": r.test_index,
                            "stdin": r.stdin,  # Full input
                            "expected": r.expected_stdout,  # Full expected
                            "actual": r.actual_stdout,  # Full actual output
                            "error": r.error,
                            "stderr": getattr(r, 'stderr', None)  # Include stderr if available
                        })
                
                logger.info(f"         {lang.upper()}: {passed}/{len(test_cases)} passed")
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            all_passed = len(all_failures) == 0
            
            logger.info(f"         TOTAL: {total_passed}/{total_tests} in {elapsed_ms}ms")
            
            return VerificationResult(
                success=all_passed,
                passed_tests=total_passed,
                total_tests=total_tests,
                failures=all_failures if all_failures else None  # Return ALL failures for debugging
            )
            
        except Exception as e:
            logger.error(f"         ✗ Verification error: {str(e)[:80]}")
            return VerificationResult(success=False, error=str(e))
    
    def _store_variant(
        self,
        variant_data: dict,
        base_question_id: str,
        base_difficulty: str
    ) -> Optional[dict]:
        """Store validated variant in database."""
        try:
            variant_hash = self._generate_variant_hash(variant_data)
            
            # Check for duplicate
            existing = self.db.find_variant_by_hash(variant_hash)
            if existing:
                logger.info(f"         Deduped to existing: {existing['id'][:8]}...")
                return existing
            
            # Build modifications summary
            modifications = variant_data.get("modifications", [])
            modifications_summary = "; ".join(modifications) if modifications else None
            
            # Normalize complexity strings (must fit varchar(30))
            expected_time = self._normalize_complexity(
                variant_data.get("expected_time_complexity"), "O(n)"
            )
            expected_space = self._normalize_complexity(
                variant_data.get("expected_space_complexity"), "O(1)"
            )
            allowed_time = self._normalize_complexity(
                variant_data.get("allowed_time_complexity") or expected_time, "O(n^2)"
            )
            allowed_space = self._normalize_complexity(
                variant_data.get("allowed_space_complexity") or expected_space, "O(n)"
            )
            
            # Store in database with all fields including stdin_wrappers and function_template
            stored = self.db.store_variant(
                base_question_id=base_question_id,
                title=variant_data.get("title"),
                problem_statement=variant_data.get("problem_statement"),
                effective_difficulty=base_difficulty,
                difficulty_delta=0,
                input_format=variant_data.get("input_format"),
                output_format=variant_data.get("output_format"),
                constraints=variant_data.get("constraints"),
                examples=variant_data.get("examples"),
                test_cases=variant_data.get("test_cases"),
                solution_code=variant_data.get("solution_code"),
                stdin_wrappers=variant_data.get("stdin_wrappers"),
                function_template=variant_data.get("function_template"),
                variant_hash=variant_hash,
                modifications_summary=modifications_summary,
                solution_explanation=variant_data.get("solution_explanation"),
                expected_time_complexity=expected_time,
                expected_space_complexity=expected_space,
                allowed_time_complexity=allowed_time,
                allowed_space_complexity=allowed_space
            )
            
            return stored
            
        except Exception as e:
            logger.error(f"         ✗ DB store failed: {str(e)[:50]}")
            return None
    
    def _normalize_complexity(self, complexity: str, default: str) -> str:
        """
        Normalize complexity string to fit varchar(30).
        Extracts Big-O notation from verbose descriptions.
        
        Examples:
            "O(Ln) where Ln is the length..." -> "O(Ln)"
            "O(n log n) due to sorting" -> "O(n log n)"
            None -> default
        """
        if not complexity:
            return default
        
        complexity = str(complexity).strip()
        
        # If already short enough, return as-is
        if len(complexity) <= 30:
            return complexity
        
        # Try to extract just the O(...) part
        import re
        match = re.match(r'(O\([^)]+\))', complexity, re.IGNORECASE)
        if match:
            extracted = match.group(1)
            if len(extracted) <= 30:
                return extracted
        
        # Fallback: truncate to 30 chars
        return complexity[:27] + "..."
    
    def _generate_variant_hash(self, variant_data: dict) -> str:
        """Generate hash for deduplication."""
        content = json.dumps({
            "title": variant_data.get("title"),
            "problem_statement": variant_data.get("problem_statement"),
            "constraints": variant_data.get("constraints")
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:64]
    
    def _record_user_history(
        self,
        user1_id: str,
        user2_id: str,
        base_question_id: str,
        variant_id: str
    ):
        """Record that users have seen this question."""
        try:
            # Order: user_id, variant_id, base_question_id
            self.db.record_user_question(user1_id, variant_id, base_question_id)
            self.db.record_user_question(user2_id, variant_id, base_question_id)
        except Exception as e:
            logger.warning(f"Failed to record user history: {e}")
    
    def _format_base_as_variant(self, base: dict) -> dict:
        """Format base question as variant response."""
        # Filter to only Python/Java/C++
        supported_langs = ["python", "java", "cpp"]
        function_template = {k: v for k, v in (base.get("function_template") or {}).items() if k in supported_langs}
        stdin_wrappers = {k: v for k, v in (base.get("stdin_wrappers") or {}).items() if k in supported_langs}
        
        return {
            "id": base["id"],
            "title": base["title"],
            "problem_statement": base["problem_statement"],
            "input_format": base.get("input_format"),
            "output_format": base.get("output_format"),
            "constraints": base.get("constraints"),
            "examples": base.get("examples"),
            "difficulty": base["difficulty"],
            "function_template": function_template,
            "stdin_wrappers": stdin_wrappers
        }
    
    def _get_target_difficulty(self, avg_rating: float) -> str:
        """Map average rating to difficulty."""
        if avg_rating < 1200:
            return "Easy"
        elif avg_rating < 1600:
            return "Medium"
        else:
            return "Hard"
    
    def _get_skill_level(self, rating1: int, rating2: int) -> str:
        """Get skill level description for LLM context."""
        avg = (rating1 + rating2) / 2
        if avg < 1000:
            return "beginner"
        elif avg < 1400:
            return "intermediate"
        elif avg < 1800:
            return "advanced"
        else:
            return "expert"
    
    def _validate_placeholders(self, variant_data: dict) -> Optional[str]:
        """
        Validate that all stdin_wrappers still have the {user_solution} placeholder.
        Returns error message if invalid, None if valid.
        """
        wrappers = variant_data.get("stdin_wrappers", {})
        if not wrappers:
            return "No stdin_wrappers"
        
        for lang in ["python", "java", "cpp"]:
            wrapper = wrappers.get(lang, "")
            if not wrapper:
                return f"Missing {lang} wrapper"
            # Check for either placeholder format
            if "{user_solution}" not in wrapper and "{USER_SOLUTION_CODE}" not in wrapper:
                return f"Missing placeholder in {lang} wrapper"
        
        return None  # Valid


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_variant_service: Optional[VariantService] = None


def get_variant_service() -> VariantService:
    """Get variant service instance."""
    global _variant_service
    if _variant_service is None:
        _variant_service = VariantService()
    return _variant_service
