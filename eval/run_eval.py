"""
Run the NeuralRays RAG evaluation harness.

This script:
1. Loads golden questions from eval/golden_questions.jsonl
2. Calls the chatbot answer_question() function
3. Scores each answer for keyword coverage and citation accuracy
4. Saves a timestamped JSON results file
5. Prints a summary in the terminal

Run from the project root:

    python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow this script to import chatbot.py from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from chatbot import answer_question  # noqa: E402
from scoring import score_answer  # noqa: E402


GOLDEN_QUESTIONS_FILE = PROJECT_ROOT / "eval" / "golden_questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_golden_questions(file_path: Path) -> list[dict[str, Any]]:
    """
    Load golden questions from a JSONL file.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Golden questions file not found: {file_path}")

    questions = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {file_path}"
                ) from error

    return questions


def run_single_evaluation(question_item: dict[str, Any]) -> dict[str, Any]:
    """
    Run one evaluation question through the chatbot and score the answer.
    """

    question_id = question_item["id"]
    question = question_item["question"]
    expected_keywords = question_item.get("expected_keywords", [])
    expected_sources = question_item.get("expected_sources", [])

    chatbot_result = answer_question(question)

    answer = chatbot_result.get("answer", "")
    sources = chatbot_result.get("sources", [])

    score = score_answer(
        question_id=question_id,
        question=question,
        answer=answer,
        sources=sources,
        expected_keywords=expected_keywords,
        expected_sources=expected_sources,
    )

    return asdict(score)


def calculate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate summary metrics for an evaluation run.
    """

    total_questions = len(results)

    if total_questions == 0:
        return {
            "total_questions": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "average_keyword_score": 0.0,
            "average_citation_score": 0.0,
            "average_overall_score": 0.0,
        }

    passed = sum(1 for result in results if result["passed"])
    failed = total_questions - passed

    average_keyword_score = sum(
        result["keyword_score"] for result in results
    ) / total_questions

    average_citation_score = sum(
        result["citation_score"] for result in results
    ) / total_questions

    average_overall_score = sum(
        result["overall_score"] for result in results
    ) / total_questions

    return {
        "total_questions": total_questions,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total_questions, 3),
        "average_keyword_score": round(average_keyword_score, 3),
        "average_citation_score": round(average_citation_score, 3),
        "average_overall_score": round(average_overall_score, 3),
    }


def save_results(results: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    """
    Save evaluation results to a timestamped JSON file.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = RESULTS_DIR / f"eval_results_{timestamp}.json"

    output_data = {
        "run_timestamp": timestamp,
        "summary": summary,
        "results": results,
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    return output_file


def print_summary(summary: dict[str, Any], output_file: Path) -> None:
    """
    Print a clean evaluation summary in the terminal.
    """

    print("\nRAG Evaluation Summary")
    print("-" * 40)
    print(f"Total questions:        {summary['total_questions']}")
    print(f"Passed:                 {summary['passed']}")
    print(f"Failed:                 {summary['failed']}")
    print(f"Pass rate:              {summary['pass_rate'] * 100:.1f}%")
    print(f"Average keyword score:  {summary['average_keyword_score']}")
    print(f"Average citation score: {summary['average_citation_score']}")
    print(f"Average overall score:  {summary['average_overall_score']}")
    print("-" * 40)
    print(f"Results saved to:       {output_file}")


def print_failures(results: list[dict[str, Any]]) -> None:
    """
    Print failed questions so they can be improved.
    """

    failed_results = [result for result in results if not result["passed"]]

    if not failed_results:
        print("\nNo failed questions.")
        return

    print("\nFailed Questions")
    print("-" * 40)

    for result in failed_results:
        print(f"\nID: {result['question_id']}")
        print(f"Question: {result['question']}")
        print(f"Keyword score: {result['keyword_score']}")
        print(f"Citation score: {result['citation_score']}")

        if result["missing_keywords"]:
            print(f"Missing keywords: {', '.join(result['missing_keywords'])}")

        if result["missing_sources"]:
            print(f"Missing sources: {', '.join(result['missing_sources'])}")


def main() -> None:
    """
    Main evaluation entry point.
    """

    golden_questions = load_golden_questions(GOLDEN_QUESTIONS_FILE)

    results = []

    for question_item in golden_questions:
        print(f"Evaluating {question_item['id']}: {question_item['question']}")
        result = run_single_evaluation(question_item)
        results.append(result)

    summary = calculate_summary(results)
    output_file = save_results(results, summary)

    print_summary(summary, output_file)
    print_failures(results)


if __name__ == "__main__":
    main()