"""
Domain-agnostic RAG evaluation service.

This service:
1. Loads chunks for a tenant from ChromaDB
2. Generates evaluation questions automatically
3. Runs the chatbot
4. Scores answer quality
5. Measures operational metrics such as latency and estimated cost
6. Saves JSON results
7. Prints a summary for quick comparison

Run from project root:

    python -m evaluation.evaluation_service --tenant_id neuralrays --adapter simple --max_questions 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.dataset_generator import (
    generate_questions_from_chunks,
    load_chunks_from_vector_store,
)
from evaluation.metric_adapters.simple_adapter import SimpleMetricAdapter
from evaluation.models import (
    EvaluationQuestionResult,
    EvaluationRunResult,
    GeneratedQuestion,
    OperationalMetrics,
)


RESULTS_ROOT = Path("evaluation/results")

# These default to zero because the current prototype does not call a paid LLM.
# They can be changed later for OpenAI/Azure/Gemini cost estimation.
DEFAULT_INPUT_COST_PER_1K_TOKENS = float(
    os.getenv("EVAL_INPUT_COST_PER_1K_TOKENS", "0.0")
)
DEFAULT_OUTPUT_COST_PER_1K_TOKENS = float(
    os.getenv("EVAL_OUTPUT_COST_PER_1K_TOKENS", "0.0")
)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.

    This is a lightweight approximation for local evaluation.
    A common rough estimate is around 1.3 tokens per word.
    """

    if not text:
        return 0

    word_count = len(text.split())

    return max(1, math.ceil(word_count * 1.3))


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_1k: float = DEFAULT_INPUT_COST_PER_1K_TOKENS,
    output_cost_per_1k: float = DEFAULT_OUTPUT_COST_PER_1K_TOKENS,
) -> float:
    """
    Estimate LLM cost.

    Since the current prototype uses local models and rule-based generation,
    the default is zero. This function makes the framework ready for paid LLMs.
    """

    input_cost = (prompt_tokens / 1000.0) * input_cost_per_1k
    output_cost = (completion_tokens / 1000.0) * output_cost_per_1k

    return round(input_cost + output_cost, 8)


def load_metric_adapter(adapter_name: str) -> SimpleMetricAdapter:
    """
    Load a metric adapter by name.
    """

    adapter_name = adapter_name.strip().lower()

    if adapter_name == "simple":
        return SimpleMetricAdapter()

    raise ValueError(
        f"Unsupported adapter: {adapter_name}. Currently supported: simple"
    )


def load_chatbot_instance() -> Any:
    """
    Load the project chatbot.

    The current NeuralRays chatbot has separate retrieve_relevant_chunks and
    build_answer methods, which lets us measure retrieval and generation timing
    separately. If a future chatbot only exposes answer_question, the service
    can still be adapted.
    """

    from chatbot import get_chatbot

    return get_chatbot()


def run_chatbot_with_timing(
    chatbot: Any,
    question: str,
) -> tuple[str, list[str], str, OperationalMetrics]:
    """
    Run the chatbot and measure retrieval/generation timing.

    Returns:
        answer
        sources
        retrieved_context
        operational_metrics
    """

    retrieval_start = time.perf_counter()

    if hasattr(chatbot, "retrieve_relevant_chunks") and hasattr(chatbot, "build_answer"):
        chunks = chatbot.retrieve_relevant_chunks(question)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        generation_start = time.perf_counter()
        answer = chatbot.build_answer(question, chunks)
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

        sources: list[str] = []
        retrieved_context_parts: list[str] = []

        for chunk in chunks:
            chunk_url = getattr(chunk, "url", "")

            if chunk_url and chunk_url not in sources:
                sources.append(chunk_url)

            chunk_text = getattr(chunk, "text", "")

            if chunk_text:
                retrieved_context_parts.append(chunk_text)

        retrieved_context = "\n\n".join(retrieved_context_parts)

    else:
        response = chatbot.answer_question(question)
        total_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        answer = str(response.get("answer", ""))
        sources = list(response.get("sources", []))
        retrieved_context = ""

        retrieval_latency_ms = 0.0
        generation_latency_ms = total_latency_ms

    total_latency_ms = retrieval_latency_ms + generation_latency_ms

    prompt_text = f"Question:\n{question}\n\nContext:\n{retrieved_context}"
    estimated_prompt_tokens = estimate_tokens(prompt_text)
    estimated_completion_tokens = estimate_tokens(answer)
    estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens

    estimated_cost = estimate_cost_usd(
        prompt_tokens=estimated_prompt_tokens,
        completion_tokens=estimated_completion_tokens,
    )

    operational_metrics = OperationalMetrics(
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimated_completion_tokens=estimated_completion_tokens,
        estimated_total_tokens=estimated_total_tokens,
        estimated_cost_usd=estimated_cost,
    )

    return answer, sources, retrieved_context, operational_metrics


def evaluate_question(
    generated_question: GeneratedQuestion,
    chatbot: Any,
    adapter: SimpleMetricAdapter,
) -> EvaluationQuestionResult:
    """
    Evaluate one generated question.
    """

    answer, sources, retrieved_context, operational_metrics = run_chatbot_with_timing(
        chatbot=chatbot,
        question=generated_question.question,
    )

    scores = adapter.score(
        generated_question=generated_question,
        answer=answer,
        sources=sources,
        retrieved_context=retrieved_context,
    )

    # Keep pass/fail simple and stable for local evaluation.
    # Source presence is required because this is a grounded RAG system.
    passed = (
        scores.source_presence >= 1.0
        and scores.overall_quality_score >= 0.35
    )

    return EvaluationQuestionResult(
        question_id=generated_question.question_id,
        question=generated_question.question,
        source_url=generated_question.source_url,
        answer=answer,
        sources=sources,
        passed=passed,
        scores=scores,
        operational_metrics=operational_metrics,
        metadata={
            "source_chunk_id": generated_question.source_chunk_id,
            "question_metadata": generated_question.metadata,
        },
    )


def average(values: list[float]) -> float:
    """
    Safely average values.
    """

    if not values:
        return 0.0

    return round(mean(values), 4)


def summarise_results(
    tenant_id: str,
    adapter_name: str,
    question_results: list[EvaluationQuestionResult],
) -> EvaluationRunResult:
    """
    Create an evaluation run summary.
    """

    total_questions = len(question_results)
    passed = sum(1 for result in question_results if result.passed)
    failed = total_questions - passed

    pass_rate = round((passed / total_questions) * 100, 1) if total_questions else 0.0

    average_scores = {
        "faithfulness": average([result.scores.faithfulness for result in question_results]),
        "correctness": average([result.scores.correctness for result in question_results]),
        "context_precision": average([result.scores.context_precision for result in question_results]),
        "answer_relevancy": average([result.scores.answer_relevancy for result in question_results]),
        "hallucination": average([result.scores.hallucination for result in question_results]),
        "source_presence": average([result.scores.source_presence for result in question_results]),
        "overall_quality_score": average([result.scores.overall_quality_score for result in question_results]),
    }

    average_operational_metrics = {
        "retrieval_latency_ms": average([
            result.operational_metrics.retrieval_latency_ms
            for result in question_results
        ]),
        "generation_latency_ms": average([
            result.operational_metrics.generation_latency_ms
            for result in question_results
        ]),
        "total_latency_ms": average([
            result.operational_metrics.total_latency_ms
            for result in question_results
        ]),
        "estimated_prompt_tokens": average([
            float(result.operational_metrics.estimated_prompt_tokens)
            for result in question_results
        ]),
        "estimated_completion_tokens": average([
            float(result.operational_metrics.estimated_completion_tokens)
            for result in question_results
        ]),
        "estimated_total_tokens": average([
            float(result.operational_metrics.estimated_total_tokens)
            for result in question_results
        ]),
        "estimated_cost_usd": average([
            result.operational_metrics.estimated_cost_usd
            for result in question_results
        ]),
        "latency_score": average([
            result.operational_metrics.latency_score
            for result in question_results
        ]),
        "cost_score": average([
            result.operational_metrics.cost_score
            for result in question_results
        ]),
    }

    average_overall_rag_score = average([
        result.overall_rag_score for result in question_results
    ])

    return EvaluationRunResult(
        tenant_id=tenant_id,
        adapter=adapter_name,
        total_questions=total_questions,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        average_scores=average_scores,
        average_operational_metrics=average_operational_metrics,
        average_overall_rag_score=average_overall_rag_score,
        question_results=question_results,
        metadata={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "cost_note": (
                "Estimated cost defaults to 0.0 because this prototype uses local models "
                "and does not call a paid external LLM."
            ),
        },
    )


def save_evaluation_result(
    result: EvaluationRunResult,
) -> Path:
    """
    Save evaluation result JSON.
    """

    tenant_results_dir = RESULTS_ROOT / result.tenant_id
    tenant_results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = tenant_results_dir / f"evaluation_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def print_summary(result: EvaluationRunResult, output_path: Path) -> None:
    """
    Print a readable CLI summary.
    """

    print("\nDomain-Agnostic RAG Evaluation Summary")
    print("-------------------------------------------------------")
    print(f"Tenant ID:                 {result.tenant_id}")
    print(f"Adapter:                   {result.adapter}")
    print(f"Total questions:           {result.total_questions}")
    print(f"Passed:                    {result.passed}")
    print(f"Failed:                    {result.failed}")
    print(f"Pass rate:                 {result.pass_rate}%")

    print(f"Average faithfulness:      {result.average_scores['faithfulness']}")
    print(f"Average correctness:       {result.average_scores['correctness']}")
    print(f"Average context precision: {result.average_scores['context_precision']}")
    print(f"Average answer relevancy:  {result.average_scores['answer_relevancy']}")
    print(f"Average hallucination:     {result.average_scores['hallucination']}")
    print(f"Average source presence:   {result.average_scores['source_presence']}")

    print("\nOperational Metrics")
    print("-------------------------------------------------------")
    print(f"Avg retrieval latency ms:  {result.average_operational_metrics['retrieval_latency_ms']}")
    print(f"Avg generation latency ms: {result.average_operational_metrics['generation_latency_ms']}")
    print(f"Avg total latency ms:      {result.average_operational_metrics['total_latency_ms']}")
    print(f"Avg prompt tokens:         {result.average_operational_metrics['estimated_prompt_tokens']}")
    print(f"Avg completion tokens:     {result.average_operational_metrics['estimated_completion_tokens']}")
    print(f"Avg total tokens:          {result.average_operational_metrics['estimated_total_tokens']}")
    print(f"Avg estimated cost USD:    {result.average_operational_metrics['estimated_cost_usd']}")
    print(f"Avg overall RAG score:     {result.average_overall_rag_score}")
    print("-------------------------------------------------------")
    print(f"Saved JSON result:         {output_path}")


def run_evaluation(
    tenant_id: str,
    adapter_name: str,
    max_questions: int,
) -> EvaluationRunResult:
    """
    Run the full evaluation.
    """

    print(f"Loading chunks for tenant: {tenant_id}")

    chunks = load_chunks_from_vector_store(tenant_id=tenant_id)

    print(f"Chunks loaded: {len(chunks)}")
    print("Generating evaluation questions...")

    generated_questions = generate_questions_from_chunks(
        chunks=chunks,
        max_questions=max_questions,
    )

    print(f"Generated questions: {len(generated_questions)}")

    adapter = load_metric_adapter(adapter_name)
    chatbot = load_chatbot_instance()

    question_results: list[EvaluationQuestionResult] = []

    for generated_question in generated_questions:
        print(f"Evaluating {generated_question.question_id}: {generated_question.question}")

        question_result = evaluate_question(
            generated_question=generated_question,
            chatbot=chatbot,
            adapter=adapter,
        )

        question_results.append(question_result)

    return summarise_results(
        tenant_id=tenant_id,
        adapter_name=adapter_name,
        question_results=question_results,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run domain-agnostic RAG evaluation."
    )

    parser.add_argument(
        "--tenant_id",
        required=True,
        help="Tenant ID to evaluate, for example neuralrays.",
    )

    parser.add_argument(
        "--adapter",
        default="simple",
        help="Metric adapter to use. Currently supported: simple.",
    )

    parser.add_argument(
        "--max_questions",
        type=int,
        default=10,
        help="Maximum number of generated questions to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint.
    """

    args = parse_args()

    result = run_evaluation(
        tenant_id=args.tenant_id,
        adapter_name=args.adapter,
        max_questions=args.max_questions,
    )

    output_path = save_evaluation_result(result)
    print_summary(result, output_path)


if __name__ == "__main__":
    main()