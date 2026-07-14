"""
Auto question generation for the domain-agnostic evaluation service.

This module reads chunks from the shared vector store and creates evaluation
questions automatically. It does not hardcode NeuralRays-specific questions.

Important design choice:
Questions are generated at source-document/page level, not single-chunk level.
This means the expected context is the combined text from all chunks belonging
to the same source document/page.

This avoids unfair failures where a broad answer is correct but is scored
against only one small chunk.
"""

from __future__ import annotations

import re
from collections import defaultdict
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
    "read", "help", "helps", "provide", "provides", "provided", "full",
    "first", "high", "quality", "results", "result", "client", "clients",
    "website", "page", "data", "digital", "work", "works", "make", "made",
    "use", "used", "based", "including", "include", "includes", "main",
    "point", "points", "overview", "learn", "explore", "more", "lean",
}


def load_chunks_from_vector_store(
    tenant_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[DocumentChunk]:
    """
    Load chunks for a tenant from ChromaDB.

    If chunks do not yet have tenant_id metadata, the function treats them as
    belonging to the supplied tenant_id. This keeps the current prototype
    working while tenant-aware ingestion is added later.
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


def clean_title(title: str) -> str:
    """
    Clean titles so generated questions read naturally.
    """

    title = title.strip()

    if not title:
        return "this document"

    title = title.replace("–", "-")
    title = re.sub(r"\s+", " ", title)

    return title


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

    return clean_title(str(title).strip())


def get_source_url_from_metadata(metadata: dict[str, Any]) -> str | None:
    """
    Find source URL from metadata.
    """

    source_url = metadata.get("source_url") or metadata.get("url")

    if source_url is None:
        return None

    return str(source_url)


def tokenize(text: str) -> list[str]:
    """
    Tokenise text into lowercase words.
    """

    return re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())


def extract_keywords(text: str, limit: int = 5) -> list[str]:
    """
    Extract useful keywords from document text.

    These are saved in metadata for inspection, but the generator no longer
    creates weak homepage questions such as "What does this say about lean?"
    """

    words = tokenize(text)
    counts: dict[str, int] = {}

    for word in words:
        if len(word) < 4:
            continue

        if word in STOPWORDS:
            continue

        counts[word] = counts.get(word, 0) + 1

    sorted_words = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [word for word, _count in sorted_words[:limit]]


def infer_document_type(title: str, source_url: str | None, text: str) -> str:
    """
    Infer a general document type from title, URL and text.

    This keeps the evaluation layer domain-agnostic while producing questions
    that match the type of source document.
    """

    title_lower = title.lower()
    url_lower = (source_url or "").lower()
    text_lower = text.lower()

    combined = f"{title_lower} {url_lower}"

    if any(term in combined for term in ["contact", "location", "address"]):
        return "contact"

    if any(term in combined for term in ["about", "team", "people", "leadership"]):
        return "about"

    if any(
        term in combined
        for term in ["success", "case-study", "case-studies", "story", "stories"]
    ):
        return "case_study"

    if any(
        term in combined
        for term in ["ai-service", "ai-services", "artificial-intelligence"]
    ):
        return "ai_services"

    if any(term in combined for term in ["digital-service", "digital-services"]):
        return "digital_services"

    if any(term in combined for term in ["service", "services", "capabilities", "solutions"]):
        return "services"

    if any(
        term in text_lower
        for term in ["policy", "contract", "clause", "agreement", "compliance"]
    ):
        return "policy"

    return "general"


def build_question_for_document_type(
    document_type: str,
    title: str,
    keywords: list[str],
) -> str:
    """
    Generate a natural question based on inferred document type.

    For general pages, this deliberately asks a broad summary question instead
    of asking about a single extracted keyword.
    """

    if document_type == "contact":
        return f"What contact or location information is provided in {title}?"

    if document_type == "about":
        return f"What does {title} say about the organisation and its team?"

    if document_type == "case_study":
        return f"What case studies or success stories are described in {title}?"

    if document_type == "ai_services":
        return f"What AI services or capabilities are described in {title}?"

    if document_type == "digital_services":
        return f"What digital services or capabilities are described in {title}?"

    if document_type == "services":
        return f"What services or capabilities are described in {title}?"

    if document_type == "policy":
        return f"What are the key requirements or clauses described in {title}?"

    return f"What are the main points covered in {title}?"


def group_chunks_by_source(
    chunks: list[DocumentChunk],
) -> list[list[DocumentChunk]]:
    """
    Group chunks by source URL.

    If source URL is missing, fall back to title.
    """

    grouped: dict[str, list[DocumentChunk]] = defaultdict(list)

    for chunk in chunks:
        source_url = get_source_url_from_metadata(chunk.metadata)
        title = get_title_from_metadata(chunk.metadata)

        group_key = source_url or title or chunk.chunk_id

        grouped[group_key].append(chunk)

    return list(grouped.values())


def sort_chunks_by_index(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    Sort chunks in document order using chunk_index metadata.
    """

    return sorted(
        chunks,
        key=lambda chunk: int(chunk.metadata.get("chunk_index", 0)),
    )


def build_combined_context(chunks: list[DocumentChunk]) -> str:
    """
    Combine all chunks from the same source page/document into one context.
    """

    sorted_chunks = sort_chunks_by_index(chunks)

    context_parts = []

    seen_sections: set[str] = set()

    for chunk in sorted_chunks:
        section_heading = str(chunk.metadata.get("section_heading", "")).strip()

        if section_heading and section_heading not in seen_sections:
            context_parts.append(f"Section: {section_heading}")
            seen_sections.add(section_heading)

        context_parts.append(chunk.text)

    return "\n\n".join(context_parts).strip()


def generate_question_for_source_group(
    chunks: list[DocumentChunk],
    question_number: int,
) -> GeneratedQuestion:
    """
    Generate one evaluation question from a source page/document group.
    """

    sorted_chunks = sort_chunks_by_index(chunks)
    first_chunk = sorted_chunks[0]

    title = get_title_from_metadata(first_chunk.metadata)
    source_url = get_source_url_from_metadata(first_chunk.metadata)
    expected_context = build_combined_context(sorted_chunks)
    keywords = extract_keywords(expected_context, limit=5)

    document_type = infer_document_type(
        title=title,
        source_url=source_url,
        text=expected_context,
    )

    question = build_question_for_document_type(
        document_type=document_type,
        title=title,
        keywords=keywords,
    )

    source_chunk_ids = [chunk.chunk_id for chunk in sorted_chunks]

    return GeneratedQuestion(
        tenant_id=first_chunk.tenant_id,
        question_id=f"auto_q{question_number:04d}",
        question=question,
        source_chunk_id=",".join(source_chunk_ids),
        source_url=source_url,
        expected_context=expected_context,
        metadata={
            "title": title,
            "source_url": source_url,
            "keywords": keywords,
            "document_type": document_type,
            "generation_method": "source_level_pattern_based",
            "source_chunk_count": len(sorted_chunks),
        },
    )


def generate_questions_from_chunks(
    chunks: list[DocumentChunk],
    max_questions: int,
) -> list[GeneratedQuestion]:
    """
    Generate evaluation questions from source-level groups.

    This creates one question per source document/page, using all chunks from
    that source as the expected context.
    """

    questions: list[GeneratedQuestion] = []

    source_groups = group_chunks_by_source(chunks)

    for source_group in source_groups:
        if len(questions) >= max_questions:
            break

        if not source_group:
            continue

        generated_question = generate_question_for_source_group(
            chunks=source_group,
            question_number=len(questions) + 1,
        )

        questions.append(generated_question)

    return questions