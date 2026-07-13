"""
Optional DeepEval adapter.

This file is a placeholder for plugging in DeepEval later.
The evaluation service can already use the simple adapter without an external LLM key.
"""

from __future__ import annotations

from evaluation.metric_adapters.base import EvaluationMetricAdapter
from evaluation.models import MetricScore, RAGAnswer


class DeepEvalMetricAdapter(EvaluationMetricAdapter):
    """
    Placeholder adapter for future DeepEval integration.
    """

    name = "deepeval"

    def evaluate(self, rag_answer: RAGAnswer, expected_context: str) -> list[MetricScore]:
        raise NotImplementedError(
            "DeepEval adapter is not configured yet. Use --adapter simple for now."
        )