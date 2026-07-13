"""
Auto question generation for the domain-agnostic evaluation service.

This module reads chunks from the shared vector store and creates evaluation
questions automatically. It does not hardcode NeuralRays-specific questions.
"""

from __future__ import annotations

import re
from typing import Any

import chromadb

from evaluation.config import (
    CHROMA_DB_PATH,
    DEFAULT_COLLECTION_NAME,
    MIN_CHUNK_CHARACTERS,
)
from evaluation.models import DocumentChunk, GeneratedQuestion


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "your", "their", "they", "have", "will", "can", "are", "our", "you",
    "using", "through", "more", "than", "such", "also", "these", "those",
    "service", "services", "solution", "solutions", "company", "business",
}


def load_chunks_from_vector_store(
    tenant_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[DocumentChunk]:
    """
    Load chunks for a tenant from ChromaDB.

    If chunks do not yet have tenant_id metadata, the function treats them as
    belonging to the supplied tenant_id. This allows the current NeuralRays
    prototype to work while tenant-aware ingestion is added later.
    """

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB path not found: {CHROMA_DB_PATH}. Run python ingest.py first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name=collection_name)

    result = collection.get(include=["documents", "metadatas"])

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    chunks: list[DocumentChunk] = []

    for index, document_text in enumerate(documents):
        metadata = metadatas[index] or {}
        chunk_id = ids[index] if index < len(ids) else f"chunk_{index + 1}"

        chunk_tenant_id = metadata.get("tenant_id", tenant_id)

        if chunk_tenant_id != tenant_id:
            continue

        if not document_text or len(document_text.strip()) < MIN_CHUNK_CHARACTERS:
            continue

        chunks.append(
            DocumentChunk(
                tenant_id=tenant_id,
                chunk_id=chunk_id,
                text=document_text,
                metadata=metadata,
            )
        )

    return chunks


def extract_keywords(text: str, limit: int = 5) -> list[str]:
    """
    Extract simple keywords from chunk text.
    """

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())

    counts: dict[str, int] = {}

    for word in words:
        if len(word) < 4 or word in STOPWORDS:
            continue

        counts[word] = counts.get(word, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    return [word for word, _count in sorted_words[:limit]]


def get_title_from_metadata(metadata: dict[str, Any]) -> str:
    """
    Find a readable document title from metadata.
    """

    title = (
        metadata.get("title")
        or metadata.get("page_title")
        or metadata.get("document_title")
        or metadata.get("source_url")
        or metadata.get("url")
        or "this document"
    )

    return str(title).strip()


def get_source_url_from_metadata(metadata: dict[str, Any]) -> str | None:
    """
    Find source URL from metadata.
    """

    source_url = metadata.get("source_url") or metadata.get("url")

    if source_url is None:
        return None

    return str(source_url)


def generate_question_for_chunk(chunk: DocumentChunk, question_number: int) -> GeneratedQuestion:
    """
    Generate one evaluation question from a document chunk.
    """

    title = get_title_from_metadata(chunk.metadata)
    source_url = get_source_url_from_metadata(chunk.metadata)
    keywords = extract_keywords(chunk.text, limit=3)

    if keywords:
        keyword_text = ", ".join(keywords)
        question = f"What does {title} say about {keyword_text}?"
    else:
        question = f"What are the main points covered in {title}?"

    return GeneratedQuestion(
        tenant_id=chunk.tenant_id,
        question_id=f"auto_q{question_number:04d}",
        question=question,
        source_chunk_id=chunk.chunk_id,
        source_url=source_url,
        expected_context=chunk.text,
        metadata={
            "title": title,
            "keywords": keywords,
        },
    )


def generate_questions_from_chunks(
    chunks: list[DocumentChunk],
    max_questions: int,
) -> list[GeneratedQuestion]:
    """
    Generate a list of evaluation questions from chunks.
    """

    questions: list[GeneratedQuestion] = []

    for chunk in chunks:
        if len(questions) >= max_questions:
            break

        question = generate_question_for_chunk(
            chunk=chunk,
            question_number=len(questions) + 1,
        )

        questions.append(question)

    return questions