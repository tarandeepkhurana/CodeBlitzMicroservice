"""
Hardness assessment — is a variant really a different, harder question, or the
same problem with new words?

Verification (verification.py) only proves a question is CORRECT. This module
decides whether it is WORTH SERVING, using evidence a machine can check:

1. Base-solution test (the strong one)
   Run the BASE question's own solution against the VARIANT's tests.
   - passes everything  -> same problem reworded -> REJECTED as cosmetic
   - runs but gets answers wrong -> genuinely a different problem -> +100
   - crashes on every input -> inconclusive (no points, no rejection)

2. Statement similarity
   Near-identical wording AND an unchanged required algorithm -> REJECTED as a
   relabelling (e.g. "Sudoku with digits 0-8 instead of 1-9"), even though the
   base solution technically fails such a variant.

3. Required algorithm changed
   The variant's expected time complexity differs from the base's -> +50.

4. Edge cases added
   Recorded in the report but worth NO points: the categories are written by the
   same model that wrote the question, so they are claims, not checks.

5. Model judge (rejection filter only)
   A separate cheap LLM call asks "meaningfully different, or relabelled?".
   A "relabelled" verdict rejects the variant. It never earns points and is
   recorded as a judgement, clearly separate from the machine-checked evidence -
   it exists because no automatic check catches "Sudoku with digits 0-8".

The points are stored as `difficulty_delta`; a question's difficulty score is
its base score (Easy 1000 / Medium 1400 / Hard 1800) plus that delta. Serving
questions by score is step 4.

Still to come: "constraint tightened" evidence, which needs large generated
inputs where a brute-force solution runs out of time but the reference does not.
"""

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.clients.piston_client import get_piston_client
from app.services.verification import build_program, python_code

logger = logging.getLogger(__name__)

NEW_RULE_POINTS = 100
COMPLEXITY_CHANGED_POINTS = 50
EDGE_CASE_POINTS = 25
MAX_EDGE_CASE_POINTS = 75

# Statements this similar, with nothing else changed, are just a rewording.
COSMETIC_SIMILARITY = 0.85

BASE_SCORE = {"Easy": 1000, "Medium": 1400, "Hard": 1800}

# Only recognised complexity classes count. Unknown strings (e.g. "O(recursive)"
# or "O(9^(n*n))") must never earn points - different notation is not evidence.
COMPLEXITY_RANK = {
    "o(1)": 0, "o(logn)": 1, "o(log^2n)": 1, "o(sqrt(n))": 2,
    "o(n)": 3, "o(n+m)": 3, "o(v+e)": 3,
    "o(nlogn)": 4, "o(nlogk)": 4, "o(nloglogn)": 4,
    "o(n^2)": 5, "o(n*m)": 5, "o(n^2logn)": 6, "o(n^3)": 7,
    "o(2^n)": 8, "o(n!)": 9,
}


@dataclass
class HardnessResult:
    """What we can prove about a variant's difficulty."""
    cosmetic: bool                      # same problem reworded -> do not store
    delta: int = 0                      # points earned, stored as difficulty_delta
    reason: Optional[str] = None        # why it was called cosmetic
    evidence: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def report(self) -> dict:
        return {
            "difficulty_delta": self.delta,
            "evidence": self.evidence,
            "cosmetic": self.cosmetic,
            "reason": self.reason,
            **self.details,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _first_function_name(code: str) -> Optional[str]:
    match = re.search(r"^\s*def\s+(\w+)\s*\(", code or "", re.MULTILINE)
    return match.group(1) if match else None


def _rename_function(code: str, old: str, new: str) -> str:
    """Rename a function so the base solution fits the variant's wrapper."""
    if not old or not new or old == new:
        return code
    return re.sub(rf"\b{re.escape(old)}\b", new, code)


def _normalize_statement(text: str) -> str:
    """Lowercase words only — ignores formatting, numbers and punctuation."""
    return " ".join(re.findall(r"[a-z]+", (text or "").lower()))


def statement_similarity(base_statement: str, variant_statement: str) -> float:
    """0.0 (completely different) to 1.0 (identical wording)."""
    return difflib.SequenceMatcher(
        None, _normalize_statement(base_statement), _normalize_statement(variant_statement)
    ).ratio()


def _canonical_complexity(raw: str) -> str:
    return (
        (raw or "")
        .lower()
        .replace(" ", "")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("√n", "sqrt(n)")
        .replace("m*n", "n*m")
        .replace("m+n", "n+m")
    )


def _edge_case_types(question: dict) -> set:
    values = question.get("edge_case_types") or []
    if isinstance(values, str):
        values = [values]
    return {str(v).strip().lower() for v in values if v}


# ═══════════════════════════════════════════════════════════════════════════════
# The base-solution test
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_base_solution_on_variant(variant: dict, base_question: dict) -> dict:
    """
    Run the base question's own python solution against the variant's tests.

    Returns {"ran": bool, "passed": int, "failed": int, "errored": int, "total": int}.
    """
    base_solution = python_code((base_question.get("solution_code") or {}).get("python"))
    variant_solution = python_code((variant.get("solution_code") or {}).get("python"))
    wrapper = (variant.get("stdin_wrappers") or {}).get("python")
    tests = variant.get("test_cases") or []

    if not base_solution or not variant_solution or not wrapper or not tests:
        return {"ran": False, "passed": 0, "failed": 0, "errored": 0, "total": len(tests)}

    # The variant's wrapper calls the variant's function name.
    adapted = _rename_function(
        base_solution,
        _first_function_name(base_solution),
        _first_function_name(variant_solution),
    )
    program = build_program(wrapper, adapted)
    if program is None:
        return {"ran": False, "passed": 0, "failed": 0, "errored": 0, "total": len(tests)}

    piston = await get_piston_client()
    passed = failed = errored = 0
    for test in tests:
        try:
            result = await piston.execute("python", program, str(test.get("stdin", "")))
        except Exception as e:  # connection problems
            logger.warning(f"         Base-solution check could not run: {str(e)[:60]}")
            return {"ran": False, "passed": 0, "failed": 0, "errored": 0, "total": len(tests)}
        if not result.success:
            errored += 1
        elif result.stdout == str(test.get("expected_stdout", "")):
            passed += 1
        else:
            failed += 1

    return {
        "ran": errored < len(tests),  # at least one input produced a real answer
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "total": len(tests),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

async def assess_hardness(variant: dict, base_question: dict,
                          use_judge: bool = True) -> HardnessResult:
    """Decide whether a verified variant is worth serving, and score it."""
    # Cheap signals first (no code execution)
    similarity = statement_similarity(
        base_question.get("problem_statement", ""), variant.get("problem_statement", "")
    )
    base_rank = COMPLEXITY_RANK.get(_canonical_complexity(base_question.get("expected_time_complexity")))
    variant_rank = COMPLEXITY_RANK.get(_canonical_complexity(variant.get("expected_time_complexity")))
    complexity_changed = (
        base_rank is not None and variant_rank is not None and base_rank != variant_rank
    )

    # The strong signal: does the base question's own solution still solve this?
    base_check = await _run_base_solution_on_variant(variant, base_question)
    base_still_solves = (
        base_check["ran"]
        and base_check["failed"] == 0
        and base_check["passed"] == base_check["total"]
    )

    details = {
        "base_solution_check": base_check,
        "statement_similarity": round(similarity, 3),
        "complexity_changed": complexity_changed,
    }

    # ── Rejections ───────────────────────────────────────────────────────────
    if base_still_solves:
        return HardnessResult(
            cosmetic=True,
            reason="the base question's own solution passes every test of this variant, "
                   "so it is the same problem reworded",
            details=details,
        )

    if similarity >= COSMETIC_SIMILARITY and not complexity_changed:
        return HardnessResult(
            cosmetic=True,
            reason=f"the statement is {similarity:.0%} the same as the base question and the "
                   f"required algorithm is unchanged - a relabelling, not a new problem",
            details=details,
        )

    # ── Evidence worth points ────────────────────────────────────────────────
    evidence = []
    delta = 0

    if base_check["ran"] and base_check["failed"] > 0:
        delta += NEW_RULE_POINTS
        evidence.append({
            "claim": "different problem, not a rewording",
            "proof": f"the base question's solution gets {base_check['failed']} of "
                     f"{base_check['total']} tests wrong",
            "points": NEW_RULE_POINTS,
        })

    if complexity_changed:
        delta += COMPLEXITY_CHANGED_POINTS
        evidence.append({
            "claim": "a different algorithm is required",
            "proof": f"expected time complexity {base_question.get('expected_time_complexity')} "
                     f"-> {variant.get('expected_time_complexity')}",
            "points": COMPLEXITY_CHANGED_POINTS,
        })

    # Edge-case categories are written by the same model that wrote the question,
    # so they are recorded for the report but earn NO points - only things a
    # machine checked are worth points.
    new_edges = _edge_case_types(variant) - _edge_case_types(base_question)
    details["new_edge_case_types"] = sorted(new_edges)

    # Last gate: a model judgement, because no automatic check catches a
    # relabelling like "Sudoku with digits 0-8 instead of 1-9".
    if use_judge:
        from app.clients.openai_client import get_openai_client

        judgement = await get_openai_client().judge_hardness(base_question, variant, details)
        details["judge"] = judgement
        if judgement and judgement["verdict"] == "relabelled":
            return HardnessResult(
                cosmetic=True,
                reason=f"model judge: {judgement['reason']}",
                details=details,
            )

    difficulty = base_question.get("difficulty", "Medium")
    details["base_difficulty"] = difficulty
    details["difficulty_score"] = BASE_SCORE.get(difficulty, 1400) + delta

    return HardnessResult(cosmetic=False, delta=delta, evidence=evidence, details=details)
