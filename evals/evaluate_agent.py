"""Run deterministic checks against the LangGraph agent."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.graph import AgentResult, run


CASES_PATH = Path(__file__).with_name("agent_cases.jsonl")
REFUSAL_MARKERS = (
    "i do not know",
    "i don't know",
    "cannot answer",
    "can't answer",
    "only answer questions about",
    "can only answer questions about",
    "not enough information",
    "insufficient evidence",
)


def load_cases() -> list[dict[str, Any]]:
    cases = []
    with CASES_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                case = json.loads(line)
                case["_line"] = line_number
                cases.append(case)
    return cases


def truncate(value: str, limit: int = 120) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def has_refusal(answer: str) -> bool:
    lower_answer = answer.lower()
    return any(marker in lower_answer for marker in REFUSAL_MARKERS)


def check_expectations(case: dict[str, Any], result: AgentResult) -> list[str]:
    failures = []
    expected = case.get("expected", {})
    answer = result.answer
    lower_answer = answer.lower()

    if "route" in expected and result.route != expected["route"]:
        failures.append(f"route expected={expected['route']!r} actual={result.route!r}")

    allowed_routes = expected.get("allowed_routes", [])
    if allowed_routes and result.route not in allowed_routes:
        failures.append(f"allowed_routes expected_one_of={allowed_routes!r} actual={result.route!r}")

    for text in expected.get("must_contain", []):
        if text.lower() not in lower_answer:
            failures.append(
                f"must_contain expected={text!r} actual_answer={truncate(answer)!r}"
            )

    any_terms = expected.get("must_contain_any", [])
    if any_terms and not any(text.lower() in lower_answer for text in any_terms):
        failures.append(
            "must_contain_any expected_one_of="
            f"{any_terms!r} actual_answer={truncate(answer)!r}"
        )

    for text in expected.get("must_not_contain", []):
        if text.lower() in lower_answer:
            failures.append(
                f"must_not_contain expected_absent={text!r} actual_answer={truncate(answer)!r}"
            )

    if "refusal" in expected:
        actual = has_refusal(answer)
        if actual is not expected["refusal"]:
            failures.append(
                f"refusal expected={expected['refusal']!r} actual={actual!r} "
                f"actual_answer={truncate(answer)!r}"
            )

    if "has_sources" in expected:
        actual = bool(result.sources)
        if actual is not expected["has_sources"]:
            failures.append(
                f"has_sources expected={expected['has_sources']!r} actual={actual!r}"
            )

    return failures


def evaluate_case(case: dict[str, Any]) -> tuple[bool, list[str], AgentResult | None]:
    try:
        result = run(case["question"])
        failures = check_expectations(case, result)
    except Exception as exc:  # noqa: BLE001 - crashes are reported per case.
        return False, [f"exception expected=none actual={type(exc).__name__}: {exc}"], None

    return not failures, failures, result


def print_debug(result: AgentResult | None) -> None:
    if result is None:
        return

    print(f"  route: {result.route}")
    if result.route_reason:
        print(f"  route_reason: {truncate(result.route_reason)}")

    if result.sources:
        print("  sources:")
        for source, section in result.sources:
            print(f"    - {source} ({section})")

    if result.rewritten_queries:
        print("  rewritten_queries:")
        for query in result.rewritten_queries:
            print(f"    - {query}")

    if result.retrieval_attempts:
        print("  retrieval_attempts:")
        for query, chunk_ids in result.retrieval_attempts:
            chunks = ", ".join(chunk_ids) if chunk_ids else "no chunks"
            print(f"    - {query} -> {chunks}")

    if result.grading_attempts:
        print("  grading_attempts:")
        for query, sufficient, reason in result.grading_attempts:
            print(f"    - {query} -> sufficient={sufficient}: {truncate(reason)}")


def print_summary(family_counts: dict[str, list[int]], total_passed: int, total: int) -> None:
    print("\nSummary")
    print("Family        Passed/Total")
    for family in sorted(family_counts):
        passed, count = family_counts[family]
        print(f"{family:<13} {passed}/{count}")
    print(f"Overall       {total_passed}/{total}")


def main() -> int:
    cases = load_cases()
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total_passed = 0

    for case in cases:
        passed, failures, result = evaluate_case(case)
        family = case["family"]
        family_counts[family][1] += 1

        if passed:
            total_passed += 1
            family_counts[family][0] += 1

        status = "PASS" if passed else "FAIL"
        print(f"{status} {family} {case['id']}")
        for failure in failures:
            print(f"  - {failure}")
        if not passed:
            print_debug(result)

    print_summary(family_counts, total_passed, len(cases))
    return 0 if total_passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
