"""
Reusable data models for the domain-agnostic evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """
    Represents one chunk retrieved from the shared vector store.
    """

    tenant_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedQuestion:
    """
    Represents one automatically generated evaluation question.
    """

    tenant_id: str
    question_id: str
    question: str
    source_chunk_id: str
    source_url: str | None
    expected_context: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    """
    Represents the answer returned by the RAG service.
    """

    tenant_id: str
    question_id: str
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


@dataclass
class MetricScore:
    """
    Represents one metric score.
    """

    name: str
    score: float
    explanation: str


@dataclass
class EvaluationResult:
    """
    Represents the full evaluation result for one question.
    """

    tenant_id: str
    question_id: str
    question: str
    answer: str
    sources: list[str]
    source_url: str | None
    metrics: dict[str, float]
    passed: bool
    explanations: dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationRun:
    """
    Represents one complete evaluation run.
    """

    tenant_id: str
    adapter_name: str
    run_timestamp: str
    total_questions: int
    summary: dict[str, float | int]
    results: list[EvaluationResult]