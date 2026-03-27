"""
Supabase Client for database operations.

Tables:
- base_questions: Original LeetCode-style questions with solutions, test cases, wrappers
- question_variants: LLM-generated variants (currently empty)
- user_question_history: Track which questions users have seen (currently empty)
- match_questions: Questions assigned to matches (currently empty)
"""

import logging
import random
from typing import Optional
from uuid import UUID

from supabase import create_client, Client

from app.config import settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Async-compatible Supabase client wrapper."""
    
    def __init__(self):
        self._client: Optional[Client] = None
    
    def _get_client(self) -> Client:
        """Get or create Supabase client."""
        if self._client is None:
            self._client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
        return self._client
    
    @property
    def client(self) -> Client:
        return self._get_client()
    
    # ═══════════════════════════════════════════════════════════════
    # Base Questions
    # ═══════════════════════════════════════════════════════════════
    
    def get_base_question_by_id(self, question_id: str) -> Optional[dict]:
        """Fetch a base question by ID."""
        try:
            response = self.client.table("base_questions").select("*").eq("id", question_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching base question {question_id}: {e}")
            return None
    
    def get_base_question_by_leetcode_id(self, leetcode_id: str) -> Optional[dict]:
        """Fetch a base question by LeetCode ID."""
        try:
            response = self.client.table("base_questions").select("*").eq("leetcode_id", leetcode_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching base question leetcode_id={leetcode_id}: {e}")
            return None
    
    def get_base_questions_by_difficulty(
        self,
        difficulty: str,
        exclude_ids: list[str] = None,
        category: str = None,
        limit: int = 10
    ) -> list[dict]:
        """
        Fetch base questions by difficulty, excluding already-seen ones.
        
        Args:
            difficulty: "Easy", "Medium", "Hard" (capitalized)
            exclude_ids: List of base_question UUIDs to exclude
            category: Optional category filter
            limit: Max results
        """
        try:
            query = self.client.table("base_questions").select("*").eq("difficulty", difficulty)
            
            if exclude_ids:
                query = query.not_.in_("id", exclude_ids)
            
            if category:
                query = query.eq("category", category)
            
            # Fetch more than needed, then shuffle for randomization
            fetch_limit = min(limit * 3, 50)  # Fetch up to 3x or max 50
            response = query.limit(fetch_limit).execute()
            data = response.data or []
            
            # Randomize selection
            random.shuffle(data)
            return data[:limit]
        except Exception as e:
            logger.error(f"Error fetching base questions: {e}")
            return []
    
    def get_all_base_questions(self, limit: int = 100) -> list[dict]:
        """Get all base questions (randomized)."""
        try:
            # Fetch more than needed for randomization
            fetch_limit = min(limit * 2, 200)
            response = self.client.table("base_questions").select("*").limit(fetch_limit).execute()
            data = response.data or []
            
            # Randomize selection
            random.shuffle(data)
            return data[:limit]
        except Exception as e:
            logger.error(f"Error fetching all base questions: {e}")
            return []
    
    # ═══════════════════════════════════════════════════════════════
    # Question Variants
    # ═══════════════════════════════════════════════════════════════
    
    def get_validated_variant(
        self,
        difficulty: str,
        exclude_variant_ids: list[str] = None,
        exclude_base_ids: list[str] = None
    ) -> Optional[dict]:
        """
        Get a validated variant that users haven't seen.
        
        Args:
            difficulty: Target difficulty
            exclude_variant_ids: Variants already seen by users
            exclude_base_ids: Base questions already seen
        """
        try:
            query = self.client.table("question_variants").select("*")\
                .eq("validation_status", "validated")\
                .eq("effective_difficulty", difficulty)
            
            if exclude_variant_ids:
                query = query.not_.in_("id", exclude_variant_ids)
            
            if exclude_base_ids:
                query = query.not_.in_("base_question_id", exclude_base_ids)
            
            response = query.limit(1).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error fetching validated variant: {e}")
            return None
    
    def insert_variant(self, variant_data: dict) -> Optional[dict]:
        """Insert a new question variant."""
        try:
            response = self.client.table("question_variants").insert(variant_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error inserting variant: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════
    # User Question History
    # ═══════════════════════════════════════════════════════════════
    
    def get_user_seen_questions(self, user_id: str) -> dict:
        """
        Get questions a user has seen.
        
        Returns:
            {"variant_ids": [...], "base_question_ids": [...]}
        """
        try:
            response = self.client.table("user_question_history")\
                .select("variant_id, base_question_id")\
                .eq("user_id", user_id)\
                .execute()
            
            data = response.data or []
            return {
                "variant_ids": [r["variant_id"] for r in data if r.get("variant_id")],
                "base_question_ids": [r["base_question_id"] for r in data if r.get("base_question_id")]
            }
        except Exception as e:
            logger.error(f"Error fetching user history for {user_id}: {e}")
            return {"variant_ids": [], "base_question_ids": []}
    
    def record_user_question(
        self,
        user_id: str,
        variant_id: str,
        base_question_id: str
    ) -> bool:
        """Record that a user has seen a question."""
        try:
            self.client.table("user_question_history").insert({
                "user_id": user_id,
                "variant_id": variant_id,
                "base_question_id": base_question_id
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error recording user question: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════
    # Match Questions
    # ═══════════════════════════════════════════════════════════════
    
    def record_match_question(
        self,
        match_id: str,
        user1_id: str,
        user2_id: str,
        variant_id: str,
        question_data: dict
    ) -> bool:
        """Record question assigned to a match."""
        try:
            self.client.table("match_questions").insert({
                "match_id": match_id,
                "user1_id": user1_id,
                "user2_id": user2_id,
                "variant_id": variant_id,
                "question_data": question_data
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error recording match question: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════
    # Variant Cache & Storage (for variant generation)
    # ═══════════════════════════════════════════════════════════════
    
    def get_cached_variant(
        self,
        difficulty: str,
        exclude_ids: list[str] = None
    ) -> Optional[dict]:
        """
        Get a cached validated variant.
        
        Args:
            difficulty: Target difficulty
            exclude_ids: Variant IDs to exclude (already seen)
        """
        try:
            query = self.client.table("question_variants")\
                .select("*")\
                .eq("validation_status", "validated")\
                .eq("effective_difficulty", difficulty)
            
            if exclude_ids:
                query = query.not_.in_("id", exclude_ids)
            
            response = query.limit(1).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting cached variant: {e}")
            return None
    
    def find_variant_by_hash(self, variant_hash: str) -> Optional[dict]:
        """Find variant by hash for deduplication."""
        try:
            response = self.client.table("question_variants")\
                .select("*")\
                .eq("variant_hash", variant_hash)\
                .limit(1)\
                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error finding variant by hash: {e}")
            return None
    
    def store_variant(
        self,
        base_question_id: str,
        title: str,
        problem_statement: str,
        effective_difficulty: str,
        difficulty_delta: int,
        input_format: dict,
        output_format: dict,
        constraints: list,
        examples: list,
        test_cases: list,
        solution_code: dict,
        stdin_wrappers: dict,
        function_template: dict,
        variant_hash: str,
        modifications_summary: str = None,
        solution_explanation: str = None,
        expected_time_complexity: str = "O(n)",
        expected_space_complexity: str = "O(1)",
        allowed_time_complexity: str = "O(n^2)",
        allowed_space_complexity: str = "O(n)"
    ) -> Optional[dict]:
        """Store a new validated variant with all execution fields."""
        try:
            variant_data = {
                "base_question_id": base_question_id,
                "title": title,
                "problem_statement": problem_statement,
                "effective_difficulty": effective_difficulty,
                "difficulty_delta": difficulty_delta,
                "input_format": input_format,
                "output_format": output_format,
                "constraints": constraints,
                "examples": examples,
                "test_cases": test_cases,
                "solution_code": solution_code,
                "stdin_wrappers": stdin_wrappers,
                "function_template": function_template,
                "variant_hash": variant_hash,
                "modifications_summary": modifications_summary,
                "solution_explanation": solution_explanation,
                "expected_time_complexity": expected_time_complexity,
                "expected_space_complexity": expected_space_complexity,
                "allowed_time_complexity": allowed_time_complexity,
                "allowed_space_complexity": allowed_space_complexity,
                "validation_status": "validated"
            }
            
            response = self.client.table("question_variants").insert(variant_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error storing variant: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════
    # Variant Generation Queue (async generation tracking)
    # ═══════════════════════════════════════════════════════════════
    
    def get_queue_entry(self, queue_id: str) -> Optional[dict]:
        """Get a queue entry by ID."""
        try:
            response = self.client.table("variant_generation_queue")\
                .select("*")\
                .eq("id", queue_id)\
                .single()\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching queue entry {queue_id}: {e}")
            return None
    
    def create_queue_entry(
        self,
        queue_id: str,
        match_id: str,
        user1_id: str,
        user2_id: str,
        status: str = "pending",
        shown_base_question_id: str = None
    ) -> bool:
        """
        Create a queue entry if it doesn't exist.
        
        Used by microservice when Node.js hasn't created the entry yet (e.g., during testing).
        """
        try:
            entry = {
                "id": queue_id,
                "match_id": match_id,
                "user1_id": user1_id,
                "user2_id": user2_id,
                "status": status
            }
            if shown_base_question_id:
                entry["shown_base_question_id"] = shown_base_question_id
            
            self.client.table("variant_generation_queue").insert(entry).execute()
            logger.info(f"Created queue entry: {queue_id[:8]}...")
            return True
        except Exception as e:
            # Entry might already exist, which is fine
            if "duplicate key" in str(e).lower():
                return True
            logger.error(f"Error creating queue entry: {e}")
            return False
    
    def update_queue_status(
        self,
        queue_id: str,
        status: str,
        generated_variant_id: str = None,
        generation_base_question_id: str = None,
        error_message: str = None
    ) -> bool:
        """Update queue entry status."""
        try:
            update_data = {"status": status}
            
            if generated_variant_id:
                update_data["generated_variant_id"] = generated_variant_id
            if generation_base_question_id:
                update_data["generation_base_question_id"] = generation_base_question_id
            if error_message:
                update_data["error_message"] = error_message
            if status in ("completed", "failed"):
                update_data["completed_at"] = "now()"
            
            self.client.table("variant_generation_queue")\
                .update(update_data)\
                .eq("id", queue_id)\
                .execute()
            return True
        except Exception as e:
            logger.error(f"Error updating queue status: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════
    # User Question History (updated for question_type)
    # ═══════════════════════════════════════════════════════════════
    
    def record_user_question_v2(
        self,
        user_id: str,
        base_question_id: str,
        variant_id: str = None,
        question_type: str = "variant"
    ) -> bool:
        """
        Record that a user has seen a question.
        
        Args:
            user_id: User UUID
            base_question_id: Base question UUID
            variant_id: Variant UUID (None for base questions)
            question_type: "base" or "variant"
        """
        try:
            data = {
                "user_id": user_id,
                "base_question_id": base_question_id,
                "question_type": question_type
            }
            if variant_id:
                data["variant_id"] = variant_id
            
            self.client.table("user_question_history").insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Error recording user question: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_db_instance: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get Supabase client instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SupabaseClient()
    return _db_instance
