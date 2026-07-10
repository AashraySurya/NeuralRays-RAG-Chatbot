"""
Simple scoring utilities for the NeuralRays RAG evaluation harness.

This file scores chatbot answers using:
- keyword coverage
- citation/source URL accuracy
- overall pass/fail result

This is a lightweight first version of an evaluation harness.
It does not require an LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalScore:
    """
    Stores the score for one evaluated question.
    """

    question_id: str
    question: str
    answer: str
    sources: list[str]
    expected_keywords: list[str]
    expected_sources: list[str]
    keyword_score: float
    citation_score: float
    overall_score: float
    passed: bool
    missing_keywords: list[str]
    missing_sources: list[str]


def normalise_text(text: str) -> str:
    """
    Normalise text for simple case-insensitive matching.
    """

    return text.lower().strip()


def normalise_url(url: str) -> str:
    """
    Normalise URLs so small differences like trailing slashes do not break scoring.
    """

    return url.lower().strip().rstrip("/")


def calculate_keyword_score(answer: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    """
    Calculate what proportion of expected keywords appear in the answer.
    """

    if not expected_keywords:
        return 1.0, []

    answer_normalised = normalise_text(answer)

    matched_count = 0
    missing_keywords = []

    for keyword in expected_keywords:
        if normalise_text(keyword) in answer_normalised:
            matched_count += 1
        else:
            missing_keywords.append(keyword)

    keyword_score = matched_count / len(expected_keywords)

    return keyword_score, missing_keywords


def calculate_citation_score(actual_sources: list[str], expected_sources: list[str]) -> tuple[float, list[str]]:
    """
    Calculate whether expected source URLs were returned.

    If no expected sources are provided, citation score is treated as 1.0.
    This is useful for questions where the correct answer is that information
    was not found on the website, such This is useful for questions where the correct answer is that information
 as pricing.
    """

    if not expected_sources:
        return 1.0, []

    actual_normalised = {normalise_url(source) for source in actual_sources}

    matched_count = 0
    missing_sources = []

    for expected_source in expected_sources:
        expected_normalised = normalise_url(expected_source)

        source_found = any(
            expected_normalised in actual_source
            for actual_source in actual_normalised
        )

        if source_found:
            matched_count += 1
        else:
            missing_sources.append(expected_source)

    citation_score = matched_count / len(expected_sources)

    return citation_score, missing_sources


def score_answer(
    question_id: str,
    question: str,
    answer: str,
    sources: list[str],
    expected_keywords: list[str],
    expected_sources: list[str],
    keyword_threshold: float = 0.6,
    citation_threshold: float = 0.8,
) -> EvalScore:
    """
    Score one chatbot answer.
    """

    keyword_score, missing_keywords = calculate_keyword_score(
        answer=answer,
        expected_keywords=expected_keywords,
    )

    citation_score, missing_sources = calculate_citation_score(
        actual_sources=sources,
        expected_sources=expected_sources,
    )

    overall_score = round((keyword_score + citation_score) / 2, 3)

    passed = (
        keyword_score >= keyword_threshold
        and citation_score >= citation_threshold
    )

    return EvalScore(
        question_id=question_id,
        question=question,
        answer=answer,
        sources=sources,
        expected_keywords=expected_keywords,
        expected_sources=expected_sources,
        keyword_score=round(keyword_score, 3),
        citation_score=round(citation_score, 3),
        overall_score=overall_score,
        passed=passed,
        missing_keywords=missing_keywords,
        missing_sources=missing_sources,
    )