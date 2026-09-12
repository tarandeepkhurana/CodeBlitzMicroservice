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
from app.clients.supabase_client import get_supabase_client
from app.services.verification import verify_variant, python_code
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
        2. Verify it (see app/services/verification.py): ground truth by
           execution, brute-force differential testing, cross-language agreement
        3. If it fails → send the concrete failures to the LLM for a fix → re-verify
        4. Repeat fix up to MAX_FIX_ATTEMPTS times
        5. If still failing → try a fresh generation

        Only a variant that passes verification is stored, and it is stored
        exactly as verified (ground-truth test outputs, same wrappers).
        """
        import os
        from datetime import datetime

        # Create debug folder
        debug_dir = "debug_variants"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def dump(name: str, data) -> None:
            path = f"{debug_dir}/{timestamp}_{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"         📁 Saved: {path}")

        user_context = {
            "avg_rating": (user1_rating + user2_rating) / 2,
            "skill_level": self._get_skill_level(user1_rating, user2_rating)
        }

        llm_rounds = 0  # generations + fixes, recorded as validation_attempts

        for gen_attempt in range(self.MAX_GENERATION_ATTEMPTS):
            logger.info(f"         LLM generation {gen_attempt + 1}/{self.MAX_GENERATION_ATTEMPTS}...")
            metrics.record_generation_attempt()

            variant_data = await self.openai.generate_variant(
                base_question=base_question,
                user_context=user_context
            )
            llm_rounds += 1

            if not variant_data:
                logger.warning("         ✗ LLM returned None")
                continue

            logger.info(f"         ✓ Generated with {len(variant_data.get('test_cases', []))} test cases")
            dump(f"gen{gen_attempt + 1}", variant_data)

            # Independent oracle: written from the problem statement alone, so
            # agreement between it and solution_code is real evidence (if the
            # same call wrote both, they could share the same misreading).
            logger.info("         Writing independent cross-check oracle...")
            helpers = await self.openai.write_verification_helpers(variant_data)
            llm_rounds += 1
            if not helpers:
                logger.warning("         ✗ Could not write the oracle")
                continue
            variant_data["brute_force_solution"] = helpers["brute_force_solution"]
            if not python_code(variant_data.get("test_generator_code")):
                variant_data["test_generator_code"] = helpers["test_generator_code"]

            logger.info("[STEP 6] Verifying via PISTON (truth, brute force, 3 languages)...")
            outcome = await verify_variant(variant_data)

            # Fix loop: send concrete failures back to the LLM, re-verify
            current_variant = variant_data
            fix_attempt = 0
            while not outcome.success and fix_attempt < self.MAX_FIX_ATTEMPTS:
                fix_attempt += 1
                dump(f"gen{gen_attempt + 1}_fix{fix_attempt - 1}_failures", outcome.failures)
                kinds = sorted({f.get("kind", "?") for f in outcome.failures})
                logger.warning(f"         ✗ Verification failed ({len(outcome.failures)} failures: {kinds})")
                # A problem with the oracle itself (too slow, crashed, weak
                # generator) is repaired by rewriting it from the statement
                # alone — never by showing it the reference solution.
                oracle_kinds = {"brute_force_too_slow", "brute_force_error",
                                "generator_error", "too_few_random_cases"}
                if kinds and set(kinds) <= oracle_kinds:
                    logger.info(f"         Rewriting the oracle ({fix_attempt}/{self.MAX_FIX_ATTEMPTS})...")
                    helpers = await self.openai.write_verification_helpers(current_variant)
                    llm_rounds += 1
                    if not helpers:
                        metrics.record_fix_attempt(success=False)
                        break
                    current_variant = {**current_variant, **helpers}
                    outcome = await verify_variant(current_variant)
                    metrics.record_fix_attempt(success=outcome.success)
                    continue

                logger.info(f"         Sending failures to LLM for fix ({fix_attempt}/{self.MAX_FIX_ATTEMPTS})...")

                fixed_variant = await self.openai.fix_variant(
                    variant=current_variant,
                    failures=outcome.failures,
                    base_question=base_question
                )
                llm_rounds += 1

                if not fixed_variant:
                    logger.warning("         ✗ LLM fix returned None")
                    metrics.record_fix_attempt(success=False)
                    break  # Can't fix, try fresh generation

                # Ensure the LLM didn't remove {user_solution} placeholders
                placeholder_error = self._validate_placeholders(fixed_variant)
                if placeholder_error:
                    logger.warning(f"         ✗ Fix broke wrappers: {placeholder_error}")
                    metrics.record_fix_attempt(success=False)
                    continue  # Keep previous version, try another fix

                current_variant = fixed_variant
                outcome = await verify_variant(current_variant)
                metrics.record_fix_attempt(success=outcome.success)

            if outcome.success:
                logger.info("         ✓ Verification passed")
                store_result = await self._store_and_return_variant(
                    outcome.variant, base_question, outcome.report, llm_rounds
                )
                if store_result.success:
                    return store_result
                logger.warning("         Storage failed, trying fresh generation...")
            else:
                dump(f"gen{gen_attempt + 1}_final_failures", outcome.failures)
                logger.info("         Fix attempts exhausted, trying fresh generation...")

        return VariantResult(success=False, error=f"Generation/fix failed after {self.MAX_GENERATION_ATTEMPTS} generations × {self.MAX_FIX_ATTEMPTS} fixes")
    
    async def _store_and_return_variant(
        self,
        variant_data: dict,
        base_question: dict,
        report: dict,
        llm_rounds: int
    ) -> VariantResult:
        """Store a verified variant (exactly as verified) and return result."""
        # VALIDATION: Ensure we have at least 10 test cases
        test_count = len(variant_data.get("test_cases", []))
        if test_count < 10:
            logger.error(f"         ❌ Cannot store: only {test_count} test cases (need 10)")
            return VariantResult(success=False, error=f"Only {test_count} test cases, need 10")

        logger.info("[STEP 7] Storing validated variant...")
        stored = self._store_variant(
            variant_data=variant_data,
            base_question_id=base_question["id"],
            base_difficulty=base_question["difficulty"],
            report=report,
            llm_rounds=llm_rounds
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
    
    def _store_variant(
        self,
        variant_data: dict,
        base_question_id: str,
        base_difficulty: str,
        report: dict,
        llm_rounds: int
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
                allowed_space_complexity=allowed_space,
                # Same shape as base_questions.test_generator_code ({"python": ...}),
                # plus the evidence needed to re-run and justify verification.
                test_generator_code={
                    "python": python_code(variant_data.get("test_generator_code")),
                    "brute_force_python": python_code(variant_data.get("brute_force_solution")),
                    "verification_report": report,
                },
                edge_case_types=[
                    str(e) for e in (variant_data.get("edge_case_types") or []) if e
                ][:20] or None,
                validation_attempts=llm_rounds
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
    
    async def generate_for_pool(self, difficulty: str) -> VariantResult:
        """
        Generate one verified variant for the question pool.

        No users involved: the base question is chosen from the ones with the
        fewest variants so far, so the pool spreads across topics instead of
        piling up variants of the same question.
        """
        from app.services.pool import POOL_RATING

        rating = POOL_RATING.get(difficulty, 1400)
        start_time = time.time()

        base_questions = self.db.get_base_questions_by_difficulty(difficulty=difficulty, limit=10)
        if not base_questions:
            return VariantResult(success=False, error=f"No base questions for difficulty {difficulty}")

        # Prefer base questions that have produced the fewest variants
        variant_counts = self.db.count_variants_by_base_question()
        base_questions.sort(key=lambda q: variant_counts.get(q["id"], 0))

        for base_question in base_questions[:self.MAX_GENERATION_ATTEMPTS]:
            logger.info(f"[POOL] Generating {difficulty} variant from: \"{base_question['title'][:40]}\"")
            result = await self._generate_and_verify_variant(
                base_question=base_question,
                user1_rating=rating,
                user2_rating=rating,
            )
            if result.success:
                result.generation_time_ms = int((time.time() - start_time) * 1000)
                metrics.record_generation_success()
                return result
            metrics.record_generation_failure("pool_generation_failed")

        return VariantResult(
            success=False,
            generation_time_ms=int((time.time() - start_time) * 1000),
            error=f"Could not generate a verified {difficulty} variant",
        )

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
