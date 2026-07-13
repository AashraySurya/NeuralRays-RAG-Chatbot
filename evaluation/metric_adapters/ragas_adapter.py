"""
Optional RAGAS adapter.

This file is a placeholder for plugging in RAGAS later.
The evaluation service can already use the simple adapter without an external LLM key.
"""

from __future__ import annotations

from evaluation.metric_adapters.base import EvaluationMetricAdapter
from evaluation.models import MetricScore, RAGAnswer


class RagasMetricAdapter(EvaluationMetricAdapter):
    """
    Placeholder adapter for future RAGAS integration.
    """

    name = "ragas"

    def evaluate(self, rag_answer: RAGAnswer, expected_context: str) -> list[MetricScore]:
        raise NotImplementedError(
            "RAGAS adapter is not configured yet. Use --adapter simple for now."
        )