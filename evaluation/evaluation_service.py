"""
Domain-agnostic RAG evaluation service.

General flow:
Documents -> Chunking + Embedding + Metadata -> Shared Vector Store
-> tenant_id filter -> Evaluation Service -> Auto Question Generator
-> RAG Service -> Metric Adapter -> Results

Run from the project root:

    python -m evaluation.evaluation_service --tenant_id neuralrays --adapter simple
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from evaluation.config import (
    DEFAULT_ADAPTER,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MAX_QUESTIONS,
    DEFAULT_TENANT_ID,
)
from evaluation.dataset_generator import (
    generate_questions_from_chunks,
    load_chunks_from_vector_store,
)
from evaluation.metric_adapters.base import EvaluationMetricAdapter
from evaluation.metric_adapters.deepeval_adapter import DeepEvalMetricAdapter
from evaluation.metric_adapters.ragas_adapter import RagasMetricAdapter
from evaluation.metric_adapters.simple_adapter import SimpleMetricAdapter
from evaluation.models import EvaluationResult, EvaluationRun, RAGAnswer
from evaluation.storage import save_evaluation_run

# Allow imports from project root, including chatbot.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from chatbot import answer_question  # noqa: E402


def get_metric_adapter(adapter_name: str) -> EvaluationMetricAdapter:
    """
    Return the selected metric adapter.
    """

    adapter_name = adapter_name.lower().strip()

    if adapter_name == "simple":
        return SimpleMetricAdapter()

    if adapter_name == "ragas":
        return RagasMetricAdapter()

    if adapter_name == "deepeval":
        return DeepEvalMetricAdapter()

    raise ValueError(
        f"Unsupported adapter '{adapter_name}'. Use one of: simple, ragas, deepeval."
    )


def run_rag_service(question: str, tenant_id: str, question_id: str, expected_context: str) -> RAGAnswer:
    """
    Send the question to the existing RAG chatbot.

    Current limitation:
    The existing chatbot is not fully tenant-aware yet and does not return
    retrieved contexts. For this first version, the generated source chunk is
    used as the evaluation context.
    """

    chatbot_response = answer_question(question)

    return RAGAnswer(
        tenant_id=tenant_id,
        question_id=question_id,
        question=question,
        answer=chatbot_response.get("answer", ""),
        sources=chatbot_response.get("sources", []),
        contexts=[expected_context],
    )


def calculate_pass_status(metrics: dict[str, float]) -> bool:
    """
    Decide whether a question passed.

    Hallucination is inverted because lower is better.
    """

    faithfulness = metrics.get("faithfulness", 0.0)
    answer_relevancy = metrics.get("answer_relevancy", 0.0)
    context_precision = metrics.get("context_precision", 0.0)
    hallucination = metrics.get("hallucination", 1.0)

    return (
        faithfulness >= 0.30
        and answer_relevancy >= 0.20
        and context_precision >= 0.20
        and hallucination <= 0.70
    )


def calculate_summary(results: list[EvaluationResult]) -> dict[str, float | int]:
    """
    Calculate dashboard-ready summary metrics.
    """

    total_questions = len(results)

    if total_questions == 0:
        return {
            "total_questions": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "average_faithfulness": 0.0,
            "average_correctness": 0.0,
            "average_context_precision": 0.0,
            "average_answer_relevancy": 0.0,
            "average_hallucination": 0.0,
            "average_source_presence": 0.0,
        }

    passed = sum(1 for result in results if result.passed)
    failed = total_questions - passed

    def average_metric(metric_name: str) -> float:
        return round(
            sum(result.metrics.get(metric_name, 0.0) for result in results) / total_questions,
            3,
        )

    return {
        "total_questions": total_questions,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total_questions, 3),
        "average_faithfulness": average_metric("faithfulness"),
        "average_correctness": average_metric("correctness"),
        "average_context_precision": average_metric("context_precision"),
        "average_answer_relevancy": average_metric("answer_relevancy"),
        "average_hallucination": average_metric("hallucination"),
        "average_source_presence": average_metric("source_presence"),
    }


def run_evaluation(
    tenant_id: str,
    adapter_name: str,
    collection_name: str,
    max_questions: int,
) -> EvaluationRun:
    """
    Run a full tenant evaluation.
    """

    print(f"\nLoading chunks for tenant: {tenant_id}")
    chunks = load_chunks_from_vector_store(
        tenant_id=tenant_id,
        collection_name=collection_name,
    )

    print(f"Chunks loaded: {len(chunks)}")

    if not chunks:
        raise ValueError(
            f"No chunks found for tenant '{tenant_id}'. Check ingestion and metadata."
        )

    print("Generating evaluation questions...")
    generated_questions = generate_questions_from_chunks(
        chunks=chunks,
        max_questions=max_questions,
    )

    print(f"Generated questions: {len(generated_questions)}")

    adapter = get_metric_adapter(adapter_name)

    results: list[EvaluationResult] = []

    for generated_question in generated_questions:
        print(f"Evaluating {generated_question.question_id}: {generated_question.question}")

        rag_answer = run_rag_service(
            question=generated_question.question,
            tenant_id=tenant_id,
            question_id=generated_question.question_id,
            expected_context=generated_question.expected_context,
        )

        metric_scores = adapter.evaluate(
            rag_answer=rag_answer,
            expected_context=generated_question.expected_context,
        )

        metrics = {
            metric.name: metric.score
            for metric in metric_scores
        }

        explanations = {
            metric.name: metric.explanation
            for metric in metric_scores
        }

        passed = calculate_pass_status(metrics)

        results.append(
            EvaluationResult(
                tenant_id=tenant_id,
                question_id=generated_question.question_id,
                question=generated_question.question,
                answer=rag_answer.answer,
                sources=rag_answer.sources,
                source_url=generated_question.source_url,
                metrics=metrics,
                passed=passed,
                explanations=explanations,
            )
        )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = calculate_summary(results)

    return EvaluationRun(
        tenant_id=tenant_id,
        adapter_name=adapter.name,
        run_timestamp=run_timestamp,
        total_questions=len(results),
        summary=summary,
        results=results,
    )


def print_summary(evaluation_run: EvaluationRun, output_file: Path) -> None:
    """
    Print the evaluation summary in the terminal.
    """

    summary = evaluation_run.summary

    print("\nDomain-Agnostic RAG Evaluation Summary")
    print("-" * 55)
    print(f"Tenant ID:                 {evaluation_run.tenant_id}")
    print(f"Adapter:                   {evaluation_run.adapter_name}")
    print(f"Total questions:           {summary['total_questions']}")
    print(f"Passed:                    {summary['passed']}")
    print(f"Failed:                    {summary['failed']}")
    print(f"Pass rate:                 {summary['pass_rate'] * 100:.1f}%")
    print(f"Average faithfulness:      {summary['average_faithfulness']}")
    print(f"Average correctness:       {summary['average_correctness']}")
    print(f"Average context precision: {summary['average_context_precision']}")
    print(f"Average answer relevancy:  {summary['average_answer_relevancy']}")
    print(f"Average hallucination:     {summary['average_hallucination']}")
    print(f"Average source presence:   {summary['average_source_presence']}")
    print("-" * 55)
    print(f"Results saved to:          {output_file}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run domain-agnostic RAG evaluation."
    )

    parser.add_argument(
        "--tenant_id",
        default=DEFAULT_TENANT_ID,
        help="Tenant/customer namespace to evaluate.",
    )

    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        choices=["simple", "ragas", "deepeval"],
        help="Metric adapter to use.",
    )

    parser.add_argument(
        "--collection_name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name.",
    )

    parser.add_argument(
        "--max_questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help="Maximum number of auto-generated questions.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """

    args = parse_args()

    evaluation_run = run_evaluation(
        tenant_id=args.tenant_id,
        adapter_name=args.adapter,
        collection_name=args.collection_name,
        max_questions=args.max_questions,
    )

    output_file = save_evaluation_run(evaluation_run)

    print_summary(evaluation_run, output_file)


if __name__ == "__main__":
    main()