"""
Semantic / layout-aware chunker for the RAG pipeline.

This module improves on fixed-size word chunking by preserving:
- source URL
- page title
- section heading
- chunk index
- tenant_id
- chunk type

It is designed to support the enterprise RAG flow where documents are chunked,
embedded with metadata, stored in a shared vector store, and later filtered by
tenant_id / namespace.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_TENANT_ID = "neuralrays"
DEFAULT_CHUNK_SIZE_WORDS = 220
DEFAULT_CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 25


@dataclass
class SourcePage:
    """
    Represents one source page/document before chunking.
    """

    url: str
    title: str
    text: str
    tenant_id: str = DEFAULT_TENANT_ID
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticChunk:
    """
    Represents one chunk ready for embedding and vector storage.
    """

    chunk_id: str
    tenant_id: str
    source_url: str
    title: str
    section_heading: str
    text: str
    chunk_index: int
    chunk_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalise_whitespace(text: str) -> str:
    """
    Clean repeated whitespace while keeping readable text.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def word_count(text: str) -> int:
    """
    Count words in a text string.
    """

    return len(re.findall(r"\b\w+\b", text))


def slugify(text: str, max_length: int = 60) -> str:
    """
    Create a safe slug for chunk IDs.
    """

    text = text.lower().strip()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")

    if not text:
        return "document"

    return text[:max_length].strip("-")


def create_hash(text: str, length: int = 10) -> str:
    """
    Create a short stable hash.
    """

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def create_chunk_id(
    tenant_id: str,
    source_url: str,
    section_heading: str,
    chunk_index: int,
    text: str,
) -> str:
    """
    Create a stable unique chunk ID.
    """

    source_slug = slugify(source_url or "source")
    section_slug = slugify(section_heading or "section", max_length=35)
    text_hash = create_hash(text)

    return f"{tenant_id}_{source_slug}_{section_slug}_{chunk_index:04d}_{text_hash}"


def is_probable_heading(line: str) -> bool:
    """
    Detect likely headings from plain extracted website text.

    This is intentionally generic and does not depend on a specific domain.
    """

    clean_line = line.strip()

    if not clean_line:
        return False

    if len(clean_line) > 90:
        return False

    if word_count(clean_line) > 10:
        return False

    # Lines ending with normal sentence punctuation are usually paragraph text.
    if clean_line.endswith((".", ",", ";", ":")):
        return False

    heading_keywords = {
        "about",
        "services",
        "contact",
        "team",
        "success",
        "stories",
        "case",
        "studies",
        "platform",
        "engineering",
        "automation",
        "consulting",
        "cloud",
        "digital",
        "ai",
        "data",
        "overview",
        "mission",
        "vision",
        "leadership",
    }

    tokens = set(re.findall(r"[a-zA-Z]+", clean_line.lower()))

    if tokens.intersection(heading_keywords):
        return True

    # Title-case or uppercase short lines are often headings.
    letters = re.sub(r"[^A-Za-z]", "", clean_line)

    if letters and clean_line.istitle():
        return True

    if letters and clean_line.isupper():
        return True

    return False


def split_into_blocks(text: str) -> list[str]:
    """
    Split page text into blocks.

    This keeps paragraph-like units together where possible.
    """

    text = normalise_whitespace(text)

    if not text:
        return []

    # Prefer blank-line paragraphs where available.
    if "\n\n" in text:
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    else:
        # Fallback: split on line breaks.
        blocks = [block.strip() for block in text.split("\n") if block.strip()]

    # If extraction produced one huge block, split softly on sentence boundaries.
    final_blocks: list[str] = []

    for block in blocks:
        if word_count(block) <= DEFAULT_CHUNK_SIZE_WORDS * 2:
            final_blocks.append(block)
            continue

        sentence_parts = re.split(r"(?<=[.!?])\s+", block)

        current = ""

        for sentence in sentence_parts:
            if not sentence.strip():
                continue

            candidate = f"{current} {sentence}".strip()

            if word_count(candidate) > DEFAULT_CHUNK_SIZE_WORDS:
                if current:
                    final_blocks.append(current.strip())
                current = sentence
            else:
                current = candidate

        if current:
            final_blocks.append(current.strip())

    return final_blocks


def group_blocks_into_sections(
    blocks: list[str],
    default_heading: str,
) -> list[tuple[str, list[str]]]:
    """
    Group blocks under detected headings.

    Returns:
        [
            ("Section heading", ["paragraph 1", "paragraph 2"]),
            ...
        ]
    """

    sections: list[tuple[str, list[str]]] = []
    current_heading = default_heading
    current_blocks: list[str] = []

    for block in blocks:
        if is_probable_heading(block):
            if current_blocks:
                sections.append((current_heading, current_blocks))

            current_heading = block.strip()
            current_blocks = []
        else:
            current_blocks.append(block)

    if current_blocks:
        sections.append((current_heading, current_blocks))

    if not sections and blocks:
        sections.append((default_heading, blocks))

    return sections


def create_overlapping_windows(
    words: list[str],
    chunk_size_words: int,
    overlap_words: int,
) -> list[str]:
    """
    Split a large word list into overlapping word windows.
    """

    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]

        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = max(end - overlap_words, start + 1)

    return chunks


def infer_chunk_type(section_heading: str, text: str) -> str:
    """
    Infer a simple chunk type for metadata.
    """

    combined = f"{section_heading} {text}".lower()

    if "|" in text or "\t" in text:
        return "table_or_structured_text"

    if any(term in combined for term in ["contact", "email", "phone", "location", "address"]):
        return "contact"

    if any(term in combined for term in ["team", "ceo", "director", "lead", "executive", "architect"]):
        return "team"

    if any(term in combined for term in ["success", "case study", "case studies", "story", "stories"]):
        return "case_study"

    if any(term in combined for term in ["service", "services", "consulting", "automation", "cloud", "engineering"]):
        return "services"

    return "section"


def create_chunks_for_section(
    page: SourcePage,
    section_heading: str,
    section_blocks: list[str],
    start_index: int,
    chunk_size_words: int,
    overlap_words: int,
) -> list[SemanticChunk]:
    """
    Create chunks from one document section.
    """

    section_text = "\n\n".join(section_blocks).strip()

    if not section_text:
        return []

    section_words = section_text.split()

    chunk_texts = create_overlapping_windows(
        words=section_words,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    chunks: list[SemanticChunk] = []

    for offset, chunk_text in enumerate(chunk_texts):
        chunk_index = start_index + offset
        chunk_type = infer_chunk_type(section_heading, chunk_text)

        chunk_id = create_chunk_id(
            tenant_id=page.tenant_id,
            source_url=page.url,
            section_heading=section_heading,
            chunk_index=chunk_index,
            text=chunk_text,
        )

        metadata = {
            "tenant_id": page.tenant_id,
            "source_url": page.url,
            "url": page.url,
            "title": page.title,
            "section_heading": section_heading,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "word_count": word_count(chunk_text),
        }

        chunks.append(
            SemanticChunk(
                chunk_id=chunk_id,
                tenant_id=page.tenant_id,
                source_url=page.url,
                title=page.title,
                section_heading=section_heading,
                text=chunk_text,
                chunk_index=chunk_index,
                chunk_type=chunk_type,
                metadata=metadata,
            )
        )

    return chunks


def parse_page(raw_page: Any, tenant_id: str = DEFAULT_TENANT_ID) -> SourcePage:
    """
    Convert a raw page dictionary or object into a SourcePage.

    Supports dictionaries like:
        {"url": "...", "title": "...", "text": "..."}

    Also supports dataclass/object style pages with .url, .title, .text.
    """

    if isinstance(raw_page, dict):
        url = str(raw_page.get("url", "")).strip()
        title = str(raw_page.get("title", "")).strip() or url or "Untitled document"
        text = str(raw_page.get("text", "")).strip()
        raw_metadata = raw_page.get("metadata", {}) or {}
        page_tenant_id = str(raw_page.get("tenant_id", tenant_id)).strip() or tenant_id
    else:
        url = str(getattr(raw_page, "url", "")).strip()
        title = str(getattr(raw_page, "title", "")).strip() or url or "Untitled document"
        text = str(getattr(raw_page, "text", "")).strip()
        raw_metadata = getattr(raw_page, "metadata", {}) or {}
        page_tenant_id = str(getattr(raw_page, "tenant_id", tenant_id)).strip() or tenant_id

    return SourcePage(
        url=url,
        title=title,
        text=normalise_whitespace(text),
        tenant_id=page_tenant_id,
        metadata=raw_metadata,
    )


def create_chunks_from_pages(
    pages: list[Any],
    tenant_id: str = DEFAULT_TENANT_ID,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[SemanticChunk]:
    """
    Create semantic/layout-aware chunks from pages.

    This is the main function that ingest.py should call.
    """

    all_chunks: list[SemanticChunk] = []

    for raw_page in pages:
        page = parse_page(raw_page, tenant_id=tenant_id)

        if not page.text:
            continue

        blocks = split_into_blocks(page.text)
        sections = group_blocks_into_sections(
            blocks=blocks,
            default_heading=page.title or "Document",
        )

        page_chunks: list[SemanticChunk] = []

        for section_heading, section_blocks in sections:
            section_chunks = create_chunks_for_section(
                page=page,
                section_heading=section_heading,
                section_blocks=section_blocks,
                start_index=len(page_chunks) + 1,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
            )

            page_chunks.extend(section_chunks)

        all_chunks.extend(page_chunks)

    return all_chunks


def load_pages_from_json(file_path: Path) -> list[dict[str, Any]]:
    """
    Load pages from a JSON file.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Pages file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def preview_chunks(chunks: list[SemanticChunk], limit: int = 5) -> None:
    """
    Print a small preview of chunks for local testing.
    """

    print(f"Total chunks created: {len(chunks)}")
    print("-" * 60)

    for chunk in chunks[:limit]:
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Title: {chunk.title}")
        print(f"URL: {chunk.source_url}")
        print(f"Section: {chunk.section_heading}")
        print(f"Type: {chunk.chunk_type}")
        print(f"Words: {word_count(chunk.text)}")
        print(f"Text preview: {chunk.text[:250]}...")
        print("-" * 60)


def main() -> None:
    """
    Test the chunker directly.

    Run:
        python chunker.py
    """

    data_file = Path("data/pages.json")

    pages = load_pages_from_json(data_file)

    chunks = create_chunks_from_pages(
        pages=pages,
        tenant_id=DEFAULT_TENANT_ID,
    )

    preview_chunks(chunks)


if __name__ == "__main__":
    main()