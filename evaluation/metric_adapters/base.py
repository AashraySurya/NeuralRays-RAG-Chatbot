"""
Base interface for plug-and-play evaluation metric adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation.models import MetricScore, RAGAnswer


class EvaluationMetricAdapter(ABC):
    """
    All metric adapters must follow this interface.
    """

    name: str

    @abstractmethod
    def evaluate(self, rag_answer: RAGAnswer, expected_context: str) -> list[MetricScore]:
        """
        Evaluate one RAG answer against the expected context.
        """

        raise NotImplementedError