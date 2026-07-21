"""
Simple local metric adapter for RAG evaluation.

This adapter does not require external APIs. It provides lightweight proxy
metrics for local development and regression testing.

It is not a replacement for RAGAS or DeepEval, but it is useful for repeatable
before/after comparisons during prototype development.
"""

from __future__ import annotations

import re

from evaluation.models import EvaluationScores, GeneratedQuestion


STOPWORDS = {
    "what", "does", "do", "is", "are", "the", "a", "an", "of", "to", "in",
    "for", "with", "and", "or", "on", "by", "from", "about", "tell", "me",
    "please", "can", "could", "would", "should", "neuralrays", "neural",
    "rays", "ai", "company", "website", "page", "offer", "offers", "list",
    "lists", "describe", "described", "information", "provided", "covered",
    "main", "points", "this", "that", "these", "those", "using", "into",
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


def score_overlap(reference_text: str, candidate_text: str) -> float:
    """
    Calculate overlap between two pieces of text.
    """

    reference_words = tokenize(reference_text)
    candidate_words = tokenize(candidate_text)

    if not reference_words:
        return 0.0

    overlap = reference_words.intersection(candidate_words)

    return round(len(overlap) / len(reference_words), 4)


def score_candidate_grounding(candidate_text: str, reference_text: str) -> float:
    """
    Score how much of the answer appears grounded in the expected context.
    """

    candidate_words = tokenize(candidate_text)
    reference_words = tokenize(reference_text)

    if not candidate_words:
        return 0.0

    overlap = candidate_words.intersection(reference_words)

    return round(len(overlap) / len(candidate_words), 4)


def source_presence_score(expected_source_url: str | None, sources: list[str]) -> float:
    """
    Score whether the answer returned sources.

    If an expected source URL exists, reward exact matching.
    Otherwise, reward any source.
    """

    if not sources:
        return 0.0

    if expected_source_url:
        normalised_expected = expected_source_url.rstrip("/").lower()

        normalised_sources = {
            source.rstrip("/").lower()
            for source in sources
        }

        if normalised_expected in normalised_sources:
            return 1.0

    return 1.0


class SimpleMetricAdapter:
    """
    Local metric adapter.
    """

    name = "simple"

    def score(
        self,
        generated_question: GeneratedQuestion,
        answer: str,
        sources: list[str],
        retrieved_context: str = "",
    ) -> EvaluationScores:
        """
        Score one chatbot answer.
        """

        expected_context = generated_question.expected_context or ""
        question = generated_question.question or ""

        # How much of the answer is supported by expected context.
        faithfulness = score_candidate_grounding(
            candidate_text=answer,
            reference_text=expected_context,
        )

        # How much of the expected context appears in the answer.
        correctness = score_overlap(
            reference_text=expected_context,
            candidate_text=answer,
        )

        # Whether retrieved context matches the expected context.
        # If retrieved context is missing, fall back to answer/context overlap.
        if retrieved_context:
            context_precision = score_overlap(
                reference_text=expected_context,
                candidate_text=retrieved_context,
            )
        else:
            context_precision = correctness

        # Whether the answer addresses the question.
        answer_relevancy = score_overlap(
            reference_text=question,
            candidate_text=answer,
        )

        source_presence = source_presence_score(
            expected_source_url=generated_question.source_url,
            sources=sources,
        )

        # Hallucination proxy: inverse of faithfulness.
        hallucination = round(1.0 - faithfulness, 4)

        return EvaluationScores(
            faithfulness=faithfulness,
            correctness=correctness,
            context_precision=context_precision,
            answer_relevancy=answer_relevancy,
            hallucination=hallucination,
            source_presence=source_presence,
        )