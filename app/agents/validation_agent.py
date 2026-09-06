"""
Validation Agent -- Day 5/6.

Runs the deterministic checks in order:
  1. Required files present
  2. ast.parse syntax check on every .py file
  3. Custom endpoints actually implemented (not still 501 stubs)
  4. Actually RUN pytest against the generated project in a temp directory
  5. Tie test results back to specific FR-IDs for real requirement coverage
     (not just "the code exists" -- "a passing test proves this FR works")

This intentionally runs pytest locally via subprocess -- per the project's
security stance, arbitrary generated code is never executed in the public
deployed Space, only here in a local/dev context.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

from app.schemas import SpecOutput, GeneratedFile, ValidationReport
from app.codegen.context import build_template_context

STUB_MARKER = "Not implemented yet -- Day 4 Coding Agent fills this in"

EXPECTED_FILES = [
    "main.py", "database.py", "models.py", "schemas.py",
    "services.py", "frontend.html", "requirements.txt", "README.md", "tests/test_api.py",
]


def _check_files_present(spec: SpecOutput, all_files: list[GeneratedFile]) -> tuple[bool, list[str]]:
    expected = list(EXPECTED_FILES)
    if spec.auth_enabled:
        expected.append("security.py")
    present = {f.path for f in all_files}
    missing = [p for p in expected if p not in present]
    return len(missing) == 0, missing


def _check_syntax(all_files: list[GeneratedFile]) -> list[str]:
    errors = []
    for f in all_files:
        if f.path.endswith(".py"):
            try:
                ast.parse(f.content)
            except SyntaxError as e:
                errors.append(f"{f.path}: {e}")
    return errors


def _count_implemented_endpoints(spec: SpecOutput, main_content: str) -> tuple[int, int]:
    required = len(spec.endpoints)
    remaining_stubs = main_content.count(STUB_MARKER)
    detected = required - remaining_stubs
    return detected, required


def _run_pytest(all_files: list[GeneratedFile]) -> tuple[dict[str, bool], dict[str, str]]:
    """Writes all files to a temp dir and runs pytest for real. Returns
    ({test_name: passed}, {test_name: failure_text}) for every collected test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in all_files:
            path = os.path.join(tmp_dir, f.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f.content)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_api.py", "-v", "--tb=short"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout

        if result.returncode not in (0, 1) or (result.returncode == 5):
            print(f"[Validation Agent] WARNING: pytest exited with code {result.returncode}")
            if result.stderr:
                print(f"[Validation Agent] stderr:\n{result.stderr}")
            if not output.strip():
                print(f"[Validation Agent] stdout was empty -- pytest may have failed to run at all")

    test_results: dict[str, bool] = {}
    for line in output.splitlines():
        match = re.match(r"tests/test_api\.py::(\w+)\s+(PASSED|FAILED|ERROR)", line)
        if match:
            test_name, status = match.groups()
            test_results[test_name] = status == "PASSED"

    # Parse per-test failure blocks out of pytest's "FAILURES" section, e.g.:
    #   ___________________ test_something ___________________
    #   ... traceback ...
    failure_details: dict[str, str] = {}
    for m in re.finditer(r"_{3,} (test_\w+) _{3,}\n(.*?)(?=\n_{3,} test_|\Z)", output, re.DOTALL):
        test_name, block = m.groups()
        failure_details[test_name] = block.strip()[:2000]  # cap length fed back to the LLM

    return test_results, failure_details


def _build_fr_coverage_map(spec: SpecOutput) -> dict[str, list[str]]:
    """Maps each deterministic/custom test function name -> the FR-IDs it covers."""
    context = build_template_context(spec)
    mapping: dict[str, list[str]] = {}

    for entity in context["entities"]:
        var = entity["var_name"]
        if entity["create_fr_ids"]:
            mapping[f"test_create_{var}"] = entity["create_fr_ids"]
        if entity["list_fr_ids"]:
            mapping[f"test_list_{var}"] = entity["list_fr_ids"]
        if entity["get_one_fr_ids"]:
            mapping[f"test_get_one_{var}"] = entity["get_one_fr_ids"]
        if entity["update_fr_ids"]:
            mapping[f"test_update_{var}"] = entity["update_fr_ids"]
        if entity["delete_fr_ids"]:
            mapping[f"test_delete_{var}"] = entity["delete_fr_ids"]

    for ep in context["custom_endpoints"]:
        if ep["fr_ids"]:
            mapping[f"test_{ep['func_name']}"] = ep["fr_ids"]

    return mapping


def _compute_requirement_coverage(spec: SpecOutput, test_results: dict[str, bool]) -> float:
    total_frs = len(spec.functional_requirements)
    if total_frs == 0:
        return 100.0

    fr_map = _build_fr_coverage_map(spec)
    covered_frs: set[str] = set()
    for test_name, fr_ids in fr_map.items():
        if test_results.get(test_name):  # only counts if the test actually passed
            covered_frs.update(fr_ids)

    return round(100.0 * len(covered_frs) / total_frs, 1)


def _extract_function_source(file_content: str, func_name: str) -> str | None:
    """Pulls one function's exact source text out of a file, for embedding
    similarity scoring against its functional requirement's text."""
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(file_content, node)
    return None


def _compute_embedding_similarity(spec: SpecOutput, test_file: GeneratedFile) -> float | None:
    """
    For each FR with a covering test, scores how semantically close the
    FR's acceptance criteria is to the actual test that's supposed to
    verify it -- a sanity signal that tests aren't just passing by
    accident (e.g. an empty assert True) but plausibly test the right
    thing. Returns None if there's nothing to score against (e.g. import
    of sentence-transformers isn't available in this environment).
    """
    try:
        from app.rag.retrieval import embedding_similarity
    except Exception:
        return None

    fr_map = _build_fr_coverage_map(spec)
    if not fr_map:
        return None

    scores = []
    for test_name, fr_ids in fr_map.items():
        source = _extract_function_source(test_file.content, test_name)
        if source is None:
            continue
        for fr_id in fr_ids:
            fr = next((f for f in spec.functional_requirements if f.id == fr_id), None)
            if fr is None:
                continue
            try:
                score = embedding_similarity(fr.acceptance_criteria, source)
                scores.append(score)
            except Exception:
                continue

    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def run_validation_agent_full(
    spec: SpecOutput,
    main_files: list[GeneratedFile],
    test_file: GeneratedFile,
    correction_cycles_used: int = 0,
) -> tuple[ValidationReport, dict[str, str]]:
    """Same as run_validation_agent, but also returns per-test failure text
    (test_name -> traceback excerpt) for the correction loop to act on."""
    all_files = main_files + [test_file]

    files_ok, missing = _check_files_present(spec, all_files)
    syntax_errors = _check_syntax(all_files)

    main_content = next(f.content for f in all_files if f.path == "main.py")
    endpoints_detected, endpoints_required = _count_implemented_endpoints(spec, main_content)

    if syntax_errors:
        # Don't bother running pytest against code that doesn't even parse.
        test_results: dict[str, bool] = {}
        failure_details: dict[str, str] = {}
    else:
        test_results, failure_details = _run_pytest(all_files)

    tests_total = len(test_results)
    tests_passed = sum(1 for passed in test_results.values() if passed)
    requirement_coverage_pct = _compute_requirement_coverage(spec, test_results)
    avg_embedding_similarity = _compute_embedding_similarity(spec, test_file)

    if syntax_errors:
        overall_status = "FAIL"
    elif tests_total == 0 or tests_passed < tests_total or endpoints_detected < endpoints_required or not files_ok:
        overall_status = "PARTIAL" if tests_passed > 0 else "FAIL"
    else:
        overall_status = "PASS"

    if missing:
        syntax_errors = syntax_errors + [f"Missing expected file: {m}" for m in missing]

    failed_test_names = [name for name, passed in test_results.items() if not passed]

    report = ValidationReport(
        files_ok=files_ok,
        syntax_errors=syntax_errors,
        endpoints_detected=endpoints_detected,
        endpoints_required=endpoints_required,
        tests_passed=tests_passed,
        tests_total=tests_total,
        failed_test_names=failed_test_names,
        requirement_coverage_pct=requirement_coverage_pct,
        avg_embedding_similarity=avg_embedding_similarity,
        correction_cycles_used=correction_cycles_used,
        overall_status=overall_status,
    )
    return report, failure_details


def run_validation_agent(
    spec: SpecOutput,
    main_files: list[GeneratedFile],
    test_file: GeneratedFile,
    correction_cycles_used: int = 0,
) -> ValidationReport:
    report, _ = run_validation_agent_full(spec, main_files, test_file, correction_cycles_used)
    return report


if __name__ == "__main__":
    from app.agents.coding_agent import run_coding_agent
    from app.agents.testing_agent import run_testing_agent
    from app.rag.example_bank import EXAMPLE_BANK

    spec = EXAMPLE_BANK[0]
    main_files = run_coding_agent(spec)
    test_file = run_testing_agent(spec)
    report = run_validation_agent(spec, main_files, test_file)
    print(report.model_dump_json(indent=2))
