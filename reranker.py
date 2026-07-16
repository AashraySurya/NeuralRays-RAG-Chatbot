"""
Reranker adapter for the NeuralRays RAG chatbot.

This module sits after vector search and before answer generation.

Flow:
User question
-> vector search from ChromaDB
-> reranker.py reorders retrieved chunks
-> chatbot.py uses the highest ranked chunks

The reranker uses:
- vector similarity
- keyword overlap
- exact phrase matching
- page/source priority
- section heading priority
- chunk type priority
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


STOPWORDS = {
    "what", "does", "do", "is", "are", "the", "a", "an", "of", "to", "in",
    "for", "with", "and", "or", "on", "by", "from", "about", "tell", "me",
    "please", "can", "could", "would", "should", "neuralrays", "neural",
    "rays", "ai", "company", "website", "page", "offer", "offers", "list",
    "lists", "describe", "described", "information",
}


@dataclass
class RerankableChunk:
    """
    Minimal representation of a retrieved chunk for reranking.
    """

    text: str
    url: str
    title: str
    distance: float
    section_heading: str = ""
    chunk_type: str = ""
    metadata: dict[str, Any] | None = None
    rerank_score: float = 0.0


def normalise_text(text: str) -> str:
    """
    Lowercase and trim text.
    """

    return text.lower().strip()


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


def extract_phrases(question: str) -> list[str]:
    """
    Extract useful phrases from a question.

    This gives extra score when the question contains phrases like:
    - cloud transformation
    - data strategy
    - success stories
    - product director
    """

    question = normalise_text(question)

    known_phrases = [
        "cloud transformation",
        "digital services",
        "digital service",
        "ai services",
        "ai service",
        "data strategy",
        "data science",
        "ai driven automation",
        "ai-driven automation",
        "product and platform engineering",
        "digital assurance",
        "success stories",
        "case studies",
        "contact information",
        "phone number",
        "email address",
        "core team",
        "product director",
        "technical director",
        "technical lead",
        "finance executive",
        "devops architect",
    ]

    return [
        phrase
        for phrase in known_phrases
        if phrase in question
    ]


def keyword_overlap_score(question: str, text: str) -> float:
    """
    Score overlap between question words and chunk text.
    """

    question_words = tokenize(question)
    text_words = tokenize(text)

    if not question_words:
        return 0.0

    overlap = question_words.intersection(text_words)

    return len(overlap) / len(question_words)


def phrase_match_score(question: str, text: str, title: str, section_heading: str) -> float:
    """
    Score exact phrase matches in chunk text, title or section heading.
    """

    phrases = extract_phrases(question)

    if not phrases:
        return 0.0

    combined_text = normalise_text(f"{title} {section_heading} {text}")

    matches = sum(1 for phrase in phrases if phrase in combined_text)

    return matches / len(phrases)


def vector_similarity_score(distance: float) -> float:
    """
    Convert ChromaDB distance into a similarity-style score.

    ChromaDB distance is better when lower.
    """

    if distance < 0:
        distance = 0.0

    return 1.0 / (1.0 + distance)


def page_priority_score(intent: str, url: str, title: str) -> float:
    """
    Score pages that are likely to match the user's intent.
    """

    url = normalise_text(url)
    title = normalise_text(title)
    combined = f"{url} {title}"

    if intent == "contact":
        if "contact" in combined:
            return 2.5
        return -0.5

    if intent in {"ai_services", "automation"}:
        if "ai-services" in combined or "ai service" in combined:
            return 2.5
        if "contact" in combined:
            return -0.5

    if intent in {"digital_services", "cloud"}:
        if "digital-service" in combined or "digital services" in combined:
            return 2.5
        if "contact" in combined:
            return -0.5

    if intent == "success_stories":
        if "success-stories" in combined or "success stories" in combined:
            return 2.5
        return -0.25

    if intent == "team":
        if "about" in combined:
            return 2.5
        if "contact" in combined:
            return -0.5

    if intent == "about":
        if "about" in combined:
            return 1.8
        if url.rstrip("/") == "https://neuralrays.ai":
            return 1.2

    if intent == "pricing":
        return 0.0

    return 0.0


def section_priority_score(intent: str, section_heading: str, chunk_type: str) -> float:
    """
    Score chunks using layout-aware metadata from chunker.py.
    """

    section_heading = normalise_text(section_heading)
    chunk_type = normalise_text(chunk_type)
    combined = f"{section_heading} {chunk_type}"

    if intent == "contact" and any(term in combined for term in ["contact", "location", "address"]):
        return 1.5

    if intent == "team" and any(term in combined for term in ["team", "ceo", "director", "lead", "executive"]):
        return 1.5

    if intent in {"ai_services", "automation"} and any(
        term in combined
        for term in ["service", "services", "automation", "consulting", "data"]
    ):
        return 1.2

    if intent in {"digital_services", "cloud"} and any(
        term in combined
        for term in ["digital", "cloud", "platform", "engineering", "assurance"]
    ):
        return 1.2

    if intent == "success_stories" and any(term in combined for term in ["case", "story", "success"]):
        return 1.2

    return 0.0


def source_diversity_penalty(chunk: RerankableChunk, seen_urls: set[str]) -> float:
    """
    Apply a small penalty if many chunks come from the same source.

    This helps the top results include the best page while still avoiding
    one source flooding every top slot.
    """

    if chunk.url in seen_urls:
        return -0.15

    return 0.0


def calculate_rerank_score(
    question: str,
    intent: str,
    chunk: RerankableChunk,
) -> float:
    """
    Calculate the final rerank score for one chunk.
    """

    vector_score = vector_similarity_score(chunk.distance)
    keyword_score = keyword_overlap_score(question, chunk.text)
    phrase_score = phrase_match_score(
        question=question,
        text=chunk.text,
        title=chunk.title,
        section_heading=chunk.section_heading,
    )
    page_score = page_priority_score(intent, chunk.url, chunk.title)
    section_score = section_priority_score(
        intent=intent,
        section_heading=chunk.section_heading,
        chunk_type=chunk.chunk_type,
    )

    final_score = (
        vector_score * 1.0
        + keyword_score * 2.0
        + phrase_score * 2.5
        + page_score
        + section_score
    )

    return round(final_score, 4)


def rerank_chunks(
    question: str,
    intent: str,
    chunks: list[RerankableChunk],
    top_n: int = 6,
) -> list[RerankableChunk]:
    """
    Rerank retrieved chunks and return the strongest results.
    """

    if not chunks:
        return []

    scored_chunks: list[RerankableChunk] = []

    for chunk in chunks:
        chunk.rerank_score = calculate_rerank_score(
            question=question,
            intent=intent,
            chunk=chunk,
        )
        scored_chunks.append(chunk)

    scored_chunks.sort(key=lambda item: item.rerank_score, reverse=True)

    final_chunks: list[RerankableChunk] = []
    seen_urls: set[str] = set()

    for chunk in scored_chunks:
        if len(final_chunks) >= top_n:
            break

        adjusted_score = chunk.rerank_score + source_diversity_penalty(chunk, seen_urls)
        chunk.rerank_score = round(adjusted_score, 4)

        final_chunks.append(chunk)
        seen_urls.add(chunk.url)

    final_chunks.sort(key=lambda item: item.rerank_score, reverse=True)

    return final_chunks