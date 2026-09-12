"""
Question verification — proves a generated question is correct before it can be
served. Everything here is decided by executing code on PISTON; nothing the LLM
claims about outputs is trusted.

Checks
------
1. Ground truth by execution
   Expected outputs are produced by running the Python reference solution on
   every test input. The LLM's own predicted outputs are only recorded for the
   report ("how often was the LLM right?"), never used as answers.

2. Differential testing (brute force)
   An independent, obviously-correct brute-force solution must print the same
   output as the reference on every test input AND on up to MAX_RANDOM_CASES
   random small inputs from the question's generator. Two independently written
   solutions agreeing is strong evidence both are right.

3. Cross-language agreement
   The Java and C++ reference solutions (inside their stdin wrappers) must print
   exactly what Python prints on every test — the same comparison the backend
   uses when judging players.

4. Starter code compatibility
   The starter code players receive (function_template) must compile inside
   the wrapper in every language — otherwise a correct player solution written
   from the template could never be submitted (e.g. wrapper calls a static
   method, template declares an instance method).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.clients.piston_client import ExecutionResult, get_piston_client

logger = logging.getLogger(__name__)

LANGUAGES = ("python", "java", "cpp")
PLACEHOLDERS = ("{user_solution}", "{USER_SOLUTION_CODE}")
VISIBLE_TESTS = 3
MIN_TESTS = 10

MAX_RANDOM_CASES = 200
MIN_RANDOM_CASES = 10  # fewer than this is too weak to count as evidence
# CPU seconds the in-sandbox harness may spend (PISTON kills Python at 3s).
HARNESS_CPU_BUDGET_S = 2.0
# CPU seconds any single program run inside the harness may spend. Keeps one
# slow brute force (e.g. repeated subtraction on 10^9) from killing the run.
HARNESS_CASE_CPU_LIMIT_S = 0.5

REPORT_VERSION = 1


@dataclass
class VerificationOutcome:
    """Result of verifying one question."""
    success: bool
    variant: dict  # copy of the variant with ground-truth expected outputs
    failures: list = field(default_factory=list)
    report: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_program(wrapper: str, code: str) -> Optional[str]:
    """Insert code into a stdin wrapper (first placeholder, like the backend)."""
    for placeholder in PLACEHOLDERS:
        if placeholder in wrapper:
            return wrapper.replace(placeholder, code, 1)
    return None


def python_code(value) -> Optional[str]:
    """Accept either {"python": code} or a plain code string."""
    if isinstance(value, dict):
        value = value.get("python")
    return value if isinstance(value, str) and value.strip() else None


def _error_text(result: ExecutionResult) -> str:
    return (
        result.error_message
        or (result.compile_output or "").strip()
        or (result.stderr or "").strip()
        or f"exit code {result.exit_code}"
    )[:300]


def _template_problem(language: str, result: ExecutionResult) -> Optional[str]:
    """A compile/name error from running the starter template, or None.

    The template's placeholder return value may legitimately crash the wrapper
    or print a wrong answer — only errors that stop the code from even
    starting count as a mismatch.
    """
    text = f"{result.stderr or ''}\n{result.compile_output or ''}"
    if language == "python":
        for err in ("SyntaxError", "IndentationError", "NameError"):
            if err in text:
                return text.strip()[-300:]
    elif language == "java":
        if "error: compilation failed" in text or result.error_message == "Compilation failed":
            return text.strip()[:300]
    elif result.error_message in ("Compilation failed", "Compilation timeout"):
        return text.strip()[:300] or result.error_message
    return None


async def _run_on_inputs(piston, language: str, program: str, inputs: list[str]) -> list[ExecutionResult]:
    """Run one program once per input (concurrency is limited inside the client)."""

    async def one(stdin: str) -> ExecutionResult:
        try:
            return await piston.execute(language, program, stdin)
        except Exception as e:  # connection errors etc.
            return ExecutionResult(success=False, stdout="", stderr="", error_message=str(e))

    return list(await asyncio.gather(*(one(s) for s in inputs)))


# ═══════════════════════════════════════════════════════════════════════════════
# Differential-testing harness (runs inside the PISTON Python sandbox)
# ═══════════════════════════════════════════════════════════════════════════════
# Runs the reference and brute-force programs in-process with redirected
# stdin/stdout, so hundreds of comparisons fit in ONE sandbox execution.
# Its output must stay under PISTON's 1KB output limit, so it prints a single
# compact JSON summary.

_HARNESS = r'''
import io, json, random, signal, sys, time

REF_SRC = __REF__
BRUTE_SRC = __BRUTE__
GEN_SRC = __GEN__
INPUTS = __INPUTS__
TRUTH = __TRUTH__
MAX_RANDOM = __MAX_RANDOM__
BUDGET = __BUDGET__

REAL_OUT = sys.stdout
START = time.process_time()


def over_budget():
    return time.process_time() - START > BUDGET


def clip(s, n=70):
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def same(a, b):
    norm = lambda s: "\n".join(line.rstrip() for line in s.strip().splitlines())
    return norm(a) == norm(b)


class CpuTimeout(Exception):
    pass


def _on_cpu_alarm(signum, frame):
    raise CpuTimeout()


try:
    signal.signal(signal.SIGVTALRM, _on_cpu_alarm)
    HAVE_TIMER = True
except (AttributeError, ValueError):
    HAVE_TIMER = False


def run(code, data, cpu_limit=__CASE_LIMIT__):
    """Run one program with redirected stdin/stdout and its own CPU guard, so a
    single slow program cannot burn the whole sandbox budget."""
    old_in, old_out = sys.stdin, sys.stdout
    buf = io.BytesIO()
    out = io.TextIOWrapper(buf, encoding="utf-8", write_through=True)
    sys.stdin = io.TextIOWrapper(io.BytesIO(data.encode("utf-8")), encoding="utf-8")
    sys.stdout = out
    err = None
    if HAVE_TIMER:
        signal.setitimer(signal.ITIMER_VIRTUAL, cpu_limit)
    try:
        exec(code, {"__name__": "__main__", "__builtins__": __builtins__})
    except SystemExit:
        pass
    except CpuTimeout:
        err = "TIMEOUT: used more than %.1fs CPU on one input" % cpu_limit
    except BaseException as e:
        err = clip(type(e).__name__ + ": " + str(e), 120)
    finally:
        if HAVE_TIMER:
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        try:
            out.flush()
        except Exception:
            pass
        sys.stdin, sys.stdout = old_in, old_out
    return buf.getvalue().decode("utf-8", "replace"), err


summary = {"tests": 0, "ref_matches_truth": 0, "test_mismatches": [],
           "random": 0, "random_mismatches": [], "errors": [], "stopped": None}

try:
    REF = compile(REF_SRC, "<reference>", "exec")
    BRUTE = compile(BRUTE_SRC, "<brute_force>", "exec")
except SyntaxError as e:
    summary["errors"].append({"where": "compile", "error": clip(e, 150)})
    REF = BRUTE = None

if REF is not None:
    # Phase 1: brute force on every test input vs the executed ground truth
    for i, data in enumerate(INPUTS):
        r, rerr = run(REF, data)
        if rerr is None and same(r, TRUTH[i]):
            summary["ref_matches_truth"] += 1
        b, berr = run(BRUTE, data)
        summary["tests"] += 1
        if berr is not None:
            if len(summary["errors"]) < 3:
                summary["errors"].append({"where": "brute_force", "test": i, "stdin": clip(data), "error": berr})
        elif not same(b, TRUTH[i]) and len(summary["test_mismatches"]) < 2:
            summary["test_mismatches"].append({"test": i, "stdin": clip(data), "reference": clip(TRUTH[i]), "brute_force": clip(b)})
        if over_budget():
            summary["stopped"] = "cpu budget reached during tests"
            break

    # Phase 2: random small inputs from the generator, reference vs brute force
    gen = None
    try:
        ns = {"__name__": "generator"}
        exec(compile(GEN_SRC, "<generator>", "exec"), ns)
        gen = ns["generate_test_case"]
    except BaseException as e:
        summary["errors"].append({"where": "generator", "error": clip(type(e).__name__ + ": " + str(e), 150)})

    if gen is not None and summary["stopped"] is None:
        for seed in range(MAX_RANDOM):
            if over_budget():
                summary["stopped"] = "cpu budget reached"
                break
            if len(summary["random_mismatches"]) >= 2 or len(summary["errors"]) >= 3:
                break
            random.seed(seed)
            try:
                try:
                    case = gen(size="small")
                except TypeError:
                    case = gen()
                data = case["stdin"] if isinstance(case, dict) else str(case)
            except BaseException as e:
                summary["errors"].append({"where": "generator", "seed": seed, "error": clip(type(e).__name__ + ": " + str(e), 150)})
                break
            r, rerr = run(REF, data)
            b, berr = run(BRUTE, data)
            summary["random"] += 1
            if rerr is not None:
                summary["errors"].append({"where": "reference", "seed": seed, "stdin": clip(data), "error": rerr})
            elif berr is not None:
                summary["errors"].append({"where": "brute_force", "seed": seed, "stdin": clip(data), "error": berr})
            elif not same(r, b):
                summary["random_mismatches"].append({"seed": seed, "stdin": clip(data), "reference": clip(r), "brute_force": clip(b)})

# Stay under the sandbox's 1KB output limit: drop details until it fits.
text = json.dumps(summary)
for key in ("errors", "random_mismatches", "test_mismatches"):
    while len(text) > 950 and summary[key]:
        summary[key].pop()
        summary["truncated"] = True
        text = json.dumps(summary)
REAL_OUT.write(text + "\n")
'''


def _build_harness(ref_program: str, brute_program: str, generator: str,
                   inputs: list[str], truth: list[str]) -> str:
    # repr() of str / list[str] is always a valid Python literal.
    return (
        _HARNESS
        .replace("__REF__", repr(ref_program))
        .replace("__BRUTE__", repr(brute_program))
        .replace("__GEN__", repr(generator))
        .replace("__INPUTS__", repr(inputs))
        .replace("__TRUTH__", repr(truth))
        .replace("__MAX_RANDOM__", str(MAX_RANDOM_CASES))
        .replace("__BUDGET__", str(HARNESS_CPU_BUDGET_S))
        .replace("__CASE_LIMIT__", str(HARNESS_CASE_CPU_LIMIT_S))
    )


async def _differential_check(piston, python_wrapper: str, reference: str, brute: str,
                              generator: str, inputs: list[str], truth: list[str]) -> dict:
    """Run the harness once; returns its summary (or {"harness_error": ...})."""
    harness = _build_harness(
        build_program(python_wrapper, reference),
        build_program(python_wrapper, brute),
        generator,
        inputs,
        truth,
    )
    result = await piston.execute("python", harness, "")
    if not result.success:
        return {"harness_error": _error_text(result)}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"harness_error": f"unreadable harness output: {result.stdout[:200]!r}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _structural_problems(variant: dict) -> list[str]:
    problems = []
    tests = variant.get("test_cases") or []
    if len(tests) < MIN_TESTS:
        problems.append(f"only {len(tests)} test_cases (need at least {MIN_TESTS})")
    solutions = variant.get("solution_code") or {}
    wrappers = variant.get("stdin_wrappers") or {}
    for lang in LANGUAGES:
        if not solutions.get(lang):
            problems.append(f"solution_code.{lang} is missing")
        wrapper = wrappers.get(lang)
        if not wrapper:
            problems.append(f"stdin_wrappers.{lang} is missing")
        elif not any(p in wrapper for p in PLACEHOLDERS):
            problems.append(f"stdin_wrappers.{lang} has no {{user_solution}} placeholder")
    if not python_code(variant.get("brute_force_solution")):
        problems.append("brute_force_solution.python is missing")
    if not python_code(variant.get("test_generator_code")):
        problems.append("test_generator_code.python is missing")
    return problems


async def verify_variant(variant: dict) -> VerificationOutcome:
    """Run all correctness checks on a generated variant."""
    started = time.time()
    problems = _structural_problems(variant)
    if problems:
        return VerificationOutcome(
            success=False,
            variant=variant,
            failures=[{"kind": "missing_fields", "detail": "; ".join(problems)}],
        )

    piston = await get_piston_client()
    tests = variant["test_cases"]
    inputs = [str(tc.get("stdin", "")) for tc in tests]
    wrappers = variant["stdin_wrappers"]
    solutions = variant["solution_code"]
    brute = python_code(variant["brute_force_solution"])
    generator = python_code(variant["test_generator_code"])
    failures: list[dict] = []

    # ── Check 1: ground truth = output of the Python reference ──────────────
    py_program = build_program(wrappers["python"], solutions["python"])
    py_results = await _run_on_inputs(piston, "python", py_program, inputs)
    truth: list[str] = []
    for i, result in enumerate(py_results):
        if result.success:
            truth.append(result.stdout)
        else:
            failures.append({
                "kind": "reference_error", "language": "python", "index": i,
                "stdin": inputs[i], "error": _error_text(result),
            })
    if failures:
        return VerificationOutcome(success=False, variant=variant, failures=failures)

    # ── Checks 2, 3 and 4 run concurrently ───────────────────────────────────
    other_langs = [lang for lang in LANGUAGES if lang != "python"]
    templates = variant.get("function_template") or {}
    template_langs = [lang for lang in LANGUAGES if templates.get(lang)]
    diff, *rest = await asyncio.gather(
        _differential_check(piston, wrappers["python"], solutions["python"], brute, generator, inputs, truth),
        *(_run_on_inputs(piston, lang, build_program(wrappers[lang], solutions[lang]), inputs)
          for lang in other_langs),
        *(_run_on_inputs(piston, lang, build_program(wrappers[lang], templates[lang]), inputs[:1])
          for lang in template_langs),
    )
    lang_results = rest[:len(other_langs)]
    template_results = rest[len(other_langs):]

    # Check 4: starter code compiles inside the wrapper
    starter_code = {}
    for lang in LANGUAGES:
        if lang not in template_langs:
            starter_code[lang] = "missing"
            failures.append({"kind": "template_mismatch", "language": lang,
                             "error": f"function_template.{lang} is missing"})
            continue
        problem = _template_problem(lang, template_results[template_langs.index(lang)][0])
        starter_code[lang] = "compiles" if problem is None else "does not compile"
        if problem:
            failures.append({"kind": "template_mismatch", "language": lang, "error": problem})

    # Check 2: differential testing
    if "harness_error" in diff:
        failures.append({"kind": "harness_error", "error": diff["harness_error"]})
    else:
        if diff["tests"] and diff["ref_matches_truth"] < diff["tests"]:
            failures.append({"kind": "harness_incompatible"})
        for m in diff["test_mismatches"] + diff["random_mismatches"]:
            failures.append({"kind": "brute_mismatch", **m})
        for e in diff["errors"]:
            where = e.get("where", "unknown")
            too_slow = "TIMEOUT" in str(e.get("error", ""))
            failures.append({"kind": f"{where}_too_slow" if too_slow else f"{where}_error", **e})
        if not diff["errors"] and diff["random"] < MIN_RANDOM_CASES:
            failures.append({"kind": "too_few_random_cases", "checked": diff["random"],
                             "stopped": diff.get("stopped")})

    # Check 3: cross-language agreement (exact match, like the backend judge)
    cross_language = {"python": f"{len(inputs)}/{len(inputs)}"}
    for lang, results in zip(other_langs, lang_results):
        agree = 0
        for i, result in enumerate(results):
            if result.success and result.stdout == truth[i]:
                agree += 1
            else:
                failures.append({
                    "kind": "language_mismatch", "language": lang, "index": i,
                    "stdin": inputs[i], "expected": truth[i], "actual": result.stdout,
                    "error": None if result.success else _error_text(result),
                })
        cross_language[lang] = f"{agree}/{len(inputs)}"

    # Ground-truth test cases: first VISIBLE_TESTS shown to players, rest hidden.
    llm_right = sum(
        1 for tc, out in zip(tests, truth)
        if "expected_stdout" in tc and str(tc["expected_stdout"]).strip() == out
    )
    verified_tests = [
        {"stdin": inputs[i], "expected_stdout": truth[i], "is_hidden": i >= VISIBLE_TESTS}
        for i in range(len(inputs))
    ]
    verified_variant = {**variant, "test_cases": verified_tests}

    report = {
        "version": REPORT_VERSION,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "ground_truth": {"method": "executed python reference solution", "tests": len(inputs)},
        "differential_testing": {
            "brute_force_agrees_on_tests": f"{diff.get('tests', 0) - len(diff.get('test_mismatches', []))}/{len(inputs)}",
            "random_inputs_checked": diff.get("random", 0),
            "random_mismatches": len(diff.get("random_mismatches", [])),
        },
        "cross_language_agreement": cross_language,
        "starter_code": starter_code,
        "llm_predicted_outputs_correct": f"{llm_right}/{len(inputs)}",
        "duration_ms": int((time.time() - started) * 1000),
    }

    logger.info(
        f"         Verification: truth={len(inputs)} tests | brute tests "
        f"{report['differential_testing']['brute_force_agrees_on_tests']} | random "
        f"{diff.get('random', 0)} checked | langs {cross_language} | "
        f"LLM predictions right {llm_right}/{len(inputs)} | failures={len(failures)}"
    )

    return VerificationOutcome(
        success=not failures,
        variant=verified_variant,
        failures=failures,
        report=report,
    )
