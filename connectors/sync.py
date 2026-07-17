"""
Sync enterprise source documents into ChromaDB.

This module demonstrates the source connector ingestion flow:

Mock enterprise source
-> preserve ACL metadata
-> chunk documents
-> embed chunks
-> store chunks in ChromaDB
-> search with ACL filtering
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from connectors.acl import acl_reason_from_metadata, encode_acl_list
from connectors.mock_source import MockEnterpriseSource
from connectors.models import AuthorisedDocumentResult, EnterpriseDocument, SourceChunk, UserContext


CHROMA_DB_PATH = Path("data/chroma_db")
ENTERPRISE_COLLECTION_NAME = "enterprise_source_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE_WORDS = 120
DEFAULT_CHUNK_OVERLAP_WORDS = 20


def setup_logging() -> None:
    """
    Set up simple logging.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def chunk_text(
    text: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into overlapping word chunks.
    """

    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start_index = 0

    while start_index < len(words):
        end_index = start_index + chunk_size_words
        chunk_words = words[start_index:end_index]
        chunks.append(" ".join(chunk_words))

        if end_index >= len(words):
            break

        start_index = max(0, end_index - overlap_words)

    return chunks


def create_chunks_from_documents(
    documents: list[EnterpriseDocument],
) -> list[SourceChunk]:
    """
    Create source chunks from enterprise documents while preserving ACL metadata.
    """

    chunks: list[SourceChunk] = []

    for document in documents:
        text_chunks = chunk_text(document.content)

        for index, chunk_text_value in enumerate(text_chunks):
            chunks.append(
                SourceChunk(
                    chunk_id=f"{document.document_id}_chunk_{index}",
                    document_id=document.document_id,
                    title=document.title,
                    source_type=document.source_type,
                    source_url=document.source_url,
                    text=chunk_text_value,
                    chunk_index=index,
                    allowed_users=document.allowed_users,
                    allowed_groups=document.allowed_groups,
                    metadata=document.metadata,
                )
            )

    return chunks


def chunk_metadata(chunk: SourceChunk) -> dict[str, Any]:
    """
    Prepare ChromaDB metadata for a source chunk.
    """

    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "source_type": chunk.source_type,
        "source_url": chunk.source_url,
        "chunk_index": chunk.chunk_index,
        "allowed_users": encode_acl_list(chunk.allowed_users),
        "allowed_groups": encode_acl_list(chunk.allowed_groups),
    }

    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value

    return metadata


def reset_enterprise_collection() -> None:
    """
    Delete the existing enterprise collection if it exists.

    This does not delete the existing NeuralRays website collection.
    """

    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    try:
        client.delete_collection(ENTERPRISE_COLLECTION_NAME)
        logging.info("Deleted existing collection: %s", ENTERPRISE_COLLECTION_NAME)
    except Exception:
        logging.info("No existing enterprise collection found")


def sync_enterprise_documents(reset_collection: bool = True) -> None:
    """
    Sync mock enterprise documents into ChromaDB.
    """

    setup_logging()

    source = MockEnterpriseSource()
    documents = source.fetch_documents()

    logging.info("Fetched %s enterprise documents", len(documents))

    chunks = create_chunks_from_documents(documents)

    logging.info("Created %s enterprise chunks", len(chunks))

    if reset_collection:
        reset_enterprise_collection()

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    collection = client.get_or_create_collection(
        name=ENTERPRISE_COLLECTION_NAME,
        metadata={
            "description": "Mock enterprise source documents with ACL metadata"
        },
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=texts,
        metadatas=[chunk_metadata(chunk) for chunk in chunks],
        embeddings=embeddings,
    )

    print("\nEnterprise Source Sync Summary")
    print("-" * 50)
    print(f"Documents fetched:  {len(documents)}")
    print(f"Chunks created:     {len(chunks)}")
    print(f"Collection:         {ENTERPRISE_COLLECTION_NAME}")
    print(f"ChromaDB path:      {CHROMA_DB_PATH}")
    print("ACL metadata:       preserved")
    print("-" * 50)


def search_authorised_chunks(
    query: str,
    user_context: UserContext,
    top_k: int = 5,
    initial_k: int = 20,
) -> list[AuthorisedDocumentResult]:
    """
    Search enterprise chunks and return only results the user is authorised to see.
    """

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB path does not exist: {CHROMA_DB_PATH}. Run python connectors/sync.py first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(ENTERPRISE_COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

    total_count = collection.count()

    if total_count == 0:
        return []

    number_of_results = min(initial_k, total_count)

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    authorised_results: list[AuthorisedDocumentResult] = []

    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}

        acl_reason = acl_reason_from_metadata(
            user_context=user_context,
            metadata=metadata,
        )

        if acl_reason is None:
            continue

        score = 1.0 / (1.0 + float(distance))

        authorised_results.append(
            AuthorisedDocumentResult(
                chunk_id=str(metadata.get("chunk_id", "")),
                document_id=str(metadata.get("document_id", "")),
                title=str(metadata.get("title", "Untitled")),
                source_type=str(metadata.get("source_type", "")),
                source_url=str(metadata.get("source_url", "")),
                text=text,
                score=round(score, 4),
                acl_reason=acl_reason,
                metadata=metadata,
            )
        )

        if len(authorised_results) >= top_k:
            break

    return authorised_results


def main() -> None:
    """
    Run enterprise source sync from the command line.
    """

    sync_enterprise_documents(reset_collection=True)


if __name__ == "__main__":
    main()