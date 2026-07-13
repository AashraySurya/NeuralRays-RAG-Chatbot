"""
Simple local metric adapter.

This adapter is domain-agnostic and does not require an LLM API key.

It provides proxy scores for:
- faithfulness
- correctness
- context_precision
- answer_relevancy
- hallucination
"""

from __future__ import annotations

import re

from evaluation.metric_adapters.base import EvaluationMetricAdapter
from evaluation.models import MetricScore, RAGAnswer


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "as", "by", "from", "that",
    "this", "it", "its", "they", "their", "be", "can", "does", "do",
    "what", "which", "who", "where", "how", "about", "into", "using",
    "user", "question", "answer", "information", "provide", "provides",
}


def tokenize(text: str) -> set[str]:
    """
    Convert text into useful lowercase tokens.
    """

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word
        for word in words
        if len(word) > 2 and word not in STOPWORDS
    }


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Avoid division by zero.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


class SimpleMetricAdapter(EvaluationMetricAdapter):
    """
    Simple local scorer for early-stage RAG evaluation.

    This is not a replacement for RAGAS or DeepEval, but it allows the
    evaluation service to run without external API keys.
    """

    name = "simple"

    def evaluate(self, rag_answer: RAGAnswer, expected_context: str) -> list[MetricScore]:
        question_tokens = tokenize(rag_answer.question)
        answer_tokens = tokenize(rag_answer.answer)
        context_tokens = tokenize(expected_context)

        faithfulness = self._calculate_faithfulness(answer_tokens, context_tokens)
        correctness = self._calculate_correctness(answer_tokens, context_tokens)
        context_precision = self._calculate_context_precision(question_tokens, context_tokens)
        answer_relevancy = self._calculate_answer_relevancy(question_tokens, answer_tokens)
        hallucination = round(1.0 - faithfulness, 3)
        source_presence = 1.0 if rag_answer.sources else 0.0

        return [
            MetricScore(
                name="faithfulness",
                score=faithfulness,
                explanation="Proxy score based on how much of the answer is supported by the source context.",
            ),
            MetricScore(
                name="correctness",
                score=correctness,
                explanation="Proxy score based on overlap between the answer and expected source context.",
            ),
            MetricScore(
                name="context_precision",
                score=context_precision,
                explanation="Proxy score based on whether the context contains terms relevant to the question.",
            ),
            MetricScore(
                name="answer_relevancy",
                score=answer_relevancy,
                explanation="Proxy score based on overlap between the question and answer.",
            ),
            MetricScore(
                name="hallucination",
                score=hallucination,
                explanation="Proxy hallucination risk. Lower is better.",
            ),
            MetricScore(
                name="source_presence",
                score=source_presence,
                explanation="Checks whether the RAG answer returned source URLs.",
            ),
        ]

    def _calculate_faithfulness(
        self,
        answer_tokens: set[str],
        context_tokens: set[str],
    ) -> float:
        if not answer_tokens:
            return 0.0

        supported_tokens = answer_tokens.intersection(context_tokens)

        return round(safe_divide(len(supported_tokens), len(answer_tokens)), 3)

    def _calculate_correctness(
        self,
        answer_tokens: set[str],
        context_tokens: set[str],
    ) -> float:
        if not context_tokens:
            return 0.0

        matched_context_tokens = context_tokens.intersection(answer_tokens)

        return round(safe_divide(len(matched_context_tokens), len(context_tokens)), 3)

    def _calculate_context_precision(
        self,
        question_tokens: set[str],
        context_tokens: set[str],
    ) -> float:
        if not question_tokens:
            return 0.0

        matched_question_tokens = question_tokens.intersection(context_tokens)

        return round(safe_divide(len(matched_question_tokens), len(question_tokens)), 3)

    def _calculate_answer_relevancy(
        self,
        question_tokens: set[str],
        answer_tokens: set[str],
    ) -> float:
        if not question_tokens:
            return 0.0

        matched_question_tokens = question_tokens.intersection(answer_tokens)

        return round(safe_divide(len(matched_question_tokens), len(question_tokens)), 3)