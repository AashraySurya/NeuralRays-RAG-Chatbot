"""
Data models for the domain-agnostic RAG evaluation framework.

The evaluation framework is intended to be reusable across tenants, customers,
domains and document types.

It stores both:
- quality metrics, such as faithfulness and hallucination
- operational metrics, such as latency, token usage and estimated cost
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """
    Represents a chunk loaded from the vector store.
    """

    tenant_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedQuestion:
    """
    Represents an automatically generated evaluation question.
    """

    tenant_id: str
    question_id: str
    question: str
    source_chunk_id: str
    source_url: str | None
    expected_context: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationScores:
    """
    Quality metrics for one evaluated answer.
    """

    faithfulness: float
    correctness: float
    context_precision: float
    answer_relevancy: float
    hallucination: float
    source_presence: float

    @property
    def overall_quality_score(self) -> float:
        """
        Calculate a single quality score from the main RAG quality metrics.

        Hallucination is inverted because lower hallucination is better.
        """

        positive_scores = [
            self.faithfulness,
            self.correctness,
            self.context_precision,
            self.answer_relevancy,
            self.source_presence,
            1.0 - self.hallucination,
        ]

        return round(sum(positive_scores) / len(positive_scores), 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperationalMetrics:
    """
    Operational metrics for one evaluated answer.

    The current prototype uses a local embedding model and rule-based answer
    generation, so actual external LLM cost is usually zero. The estimated cost
    fields are included so the framework is ready for OpenAI/Azure/Gemini later.
    """

    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    estimated_total_tokens: int
    estimated_cost_usd: float

    @property
    def latency_score(self) -> float:
        """
        Convert latency into a score between 0 and 1.

        Lower latency is better.
        """

        return round(1.0 / (1.0 + (self.total_latency_ms / 10000.0)), 4)

    @property
    def cost_score(self) -> float:
        """
        Convert estimated cost into a score between 0 and 1.

        Lower cost is better.
        """

        return round(1.0 / (1.0 + self.estimated_cost_usd), 4)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["latency_score"] = self.latency_score
        data["cost_score"] = self.cost_score
        return data


@dataclass
class EvaluationQuestionResult:
    """
    Full evaluation result for one question.
    """

    question_id: str
    question: str
    source_url: str | None
    answer: str
    sources: list[str]
    passed: bool
    scores: EvaluationScores
    operational_metrics: OperationalMetrics
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_rag_score(self) -> float:
        """
        Calculate an overall score combining quality, latency and cost.

        Weighting:
        - 80% quality
        - 15% latency
        - 5% cost

        This keeps answer quality as the main priority while still measuring
        production-relevant operational performance.
        """

        quality_component = self.scores.overall_quality_score * 0.80
        latency_component = self.operational_metrics.latency_score * 0.15
        cost_component = self.operational_metrics.cost_score * 0.05

        return round(
            quality_component + latency_component + cost_component,
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "source_url": self.source_url,
            "answer": self.answer,
            "sources": self.sources,
            "passed": self.passed,
            "scores": self.scores.to_dict(),
            "operational_metrics": self.operational_metrics.to_dict(),
            "overall_quality_score": self.scores.overall_quality_score,
            "overall_rag_score": self.overall_rag_score,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationRunResult:
    """
    Full result for one evaluation run.
    """

    tenant_id: str
    adapter: str
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    average_scores: dict[str, float]
    average_operational_metrics: dict[str, float]
    average_overall_rag_score: float
    question_results: list[EvaluationQuestionResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "adapter": self.adapter,
            "total_questions": self.total_questions,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "average_scores": self.average_scores,
            "average_operational_metrics": self.average_operational_metrics,
            "average_overall_rag_score": self.average_overall_rag_score,
            "question_results": [
                result.to_dict() for result in self.question_results
            ],
            "metadata": self.metadata,
        }