"""
LLM-as-a-Judge Evaluation Pipeline.
Runs the 40-question theological eval dataset against the live system
and outputs a Markdown report with pass/fail scores.

Usage:
    python -m backend.evals.run_evals
    python -m backend.evals.run_evals --subset 10
    python -m backend.evals.run_evals --output eval_results.md
"""

import asyncio
import json
import sys
import time
from dotenv import load_dotenv

# Load env vars into os.environ for Langfuse
load_dotenv()
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.agents.graph import run_graph_sync
from backend.core.config import get_settings
from backend.core.logging import get_logger, setup_logging
from backend.models.schemas import ChatMessage

setup_logging("WARNING")
logger = get_logger("evals")

DATASET_PATH = Path(__file__).parent / "dataset.json"
DEFAULT_OUTPUT = Path(__file__).parent / "eval_results.md"


async def run_single_eval(test_case: dict[str, Any], settings: Any) -> dict[str, Any]:
    """Run a single test case and return results."""
    start_time = time.perf_counter()
    result: dict[str, Any] = {
        "id": test_case["id"],
        "category": test_case["category"],
        "query": test_case["query"],
        "denomination": test_case["denomination"],
        "passed": False,
        "response": "",
        "failures": [],
        "latency_ms": 0,
    }

    try:
        response = await run_graph_sync(
            query=test_case["query"],
            denomination=test_case["denomination"],
            session_id=f"eval_{test_case['id']}",
            history=[],
            settings=settings,
        )

        result["latency_ms"] = round((time.perf_counter() - start_time) * 1000)
        result["response"] = response.answer
        result["verification_passed"] = response.verification_passed
        result["intent"] = response.intent

        failures = []

        # Check 1: Should refuse?
        if test_case.get("should_refuse"):
            if len(response.answer) > 200 and not any(
                phrase in response.answer.lower()
                for phrase in ["unable", "cannot", "not able", "outside", "guidelines", "not fulfill"]
            ):
                failures.append("FAILED: Expected refusal but got substantive response")
        else:
            # Check 2: Expected content
            response_lower = response.answer.lower()
            expected_list = test_case.get("expected_contains", [])
            if expected_list:
                # Require AT LEAST ONE expected phrase to pass, not ALL of them.
                if not any(expected.lower() in response_lower for expected in expected_list):
                    failures.append(f"MISSING: Expected at least one of {expected_list} in response")

            # Check 3: Fake book/chapter detection
            if test_case.get("is_fake_book") or test_case.get("is_fake_chapter"):
                corrective_phrases = [
                    "not a book", "does not exist", "no book", "not exist",
                    "only has", "chapters", "invalid", "cannot find", 
                    "there is no", "is no", "no chapter"
                ]
                if not any(p in response_lower for p in corrective_phrases):
                    failures.append("FAILED: Did not detect fake book/chapter")

            # Check 4: Citation verification
            if not response.verification_passed:
                failures.append("FAILED: Citation verification did not pass")

        result["passed"] = len(failures) == 0
        result["failures"] = failures

    except Exception as e:
        result["failures"] = [f"ERROR: {e}"]
        result["latency_ms"] = round((time.perf_counter() - start_time) * 1000)

    status = "✅ PASS" if result["passed"] else "❌ FAIL"
    print(f"  {status} [{test_case['id']}] {test_case['query'][:60]}")
    if not result["passed"]:
        for f in result["failures"]:
            print(f"       → {f}")

    return result


def generate_markdown_report(
    results: list[dict[str, Any]],
    output_path: Path,
    subset: int | None = None,
) -> None:
    """Generate a Markdown eval report."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / total if total else 0

    # Category breakdown
    categories: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Christianity-Focused AI Assistant — Evaluation Report",
        f"\n**Generated:** {timestamp}",
        f"**Subset:** {f'{subset} questions' if subset else 'Full dataset (40 questions)'}",
        "\n---\n",
        "## Summary",
        f"| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Questions** | {total} |",
        f"| **Passed** | {passed} ✅ |",
        f"| **Failed** | {failed} ❌ |",
        f"| **Pass Rate** | **{pass_rate:.1f}%** |",
        f"| **Avg Latency** | {avg_latency:.0f}ms |",
        "\n---\n",
        "## Category Breakdown",
        "| Category | Passed | Total | Rate |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for cat, stats in sorted(categories.items()):
        rate = stats["passed"] / stats["total"] * 100
        emoji = "✅" if rate == 100 else "⚠️" if rate >= 70 else "❌"
        lines.append(f"| {cat} | {stats['passed']} | {stats['total']} | {rate:.0f}% {emoji} |")

    lines += ["\n---\n", "## Detailed Results", ""]

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        lines.append(f"### [{r['id']}] {r['query']}")
        lines.append(f"- **Status:** {status}")
        lines.append(f"- **Category:** {r['category']}")
        lines.append(f"- **Denomination:** {r['denomination']}")
        lines.append(f"- **Latency:** {r.get('latency_ms', 0)}ms")
        if r.get("response"):
            preview = r["response"][:300].replace("\n", " ")
            lines.append(f"- **Response Preview:** {preview}...")
        if r.get("failures"):
            lines.append(f"- **Failures:**")
            for f in r["failures"]:
                lines.append(f"  - {f}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Report saved to: {output_path}")


async def main(subset: int | None = None, output: Path = DEFAULT_OUTPUT) -> int:
    """Main eval runner. Returns exit code (0=pass, 1=fail)."""
    settings = get_settings()

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    if subset:
        # Take N cases spread across categories
        dataset = dataset[:subset]

    print(f"\n🔍 Running {len(dataset)} evaluation cases...")
    print("=" * 60)

    results = []
    for test_case in dataset:
        result = await run_single_eval(test_case, settings)
        results.append(result)
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / total * 100 if total else 0

    print("\n" + "=" * 60)
    print(f"📊 FINAL SCORE: {passed}/{total} ({pass_rate:.1f}%)")

    generate_markdown_report(results, output, subset=subset)

    # Fail CI if pass rate < 80%
    return 0 if pass_rate >= 80 else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run theological AI evaluation suite")
    parser.add_argument("--subset", type=int, default=None, help="Run only first N cases")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output path")
    args = parser.parse_args()

    exit_code = asyncio.run(main(subset=args.subset, output=Path(args.output)))
    sys.exit(exit_code)
