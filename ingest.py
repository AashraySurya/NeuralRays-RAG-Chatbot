"""
Ingestion pipeline for the NeuralRays RAG chatbot.

This script:
1. Loads crawled website pages from data/pages.json
2. Uses chunker.py to create semantic/layout-aware chunks
3. Generates local embeddings using sentence-transformers
4. Stores chunks, embeddings and metadata in ChromaDB

Run from the project root:

    python ingest.py
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_TENANT_ID,
    SemanticChunk,
    create_chunks_from_pages,
)


DATA_FILE = Path("data/pages.json")
CHROMA_DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "neuralrays_website"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def setup_logging() -> None:
    """
    Configure logging for the ingestion script.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def load_pages(file_path: Path) -> list[dict[str, Any]]:
    """
    Load crawled pages from data/pages.json.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Pages file not found: {file_path}. Run python crawler.py first."
        )

    with file_path.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    if not isinstance(pages, list):
        raise ValueError("data/pages.json must contain a list of page objects.")

    logging.info("Loaded %s pages from %s", len(pages), file_path)

    return pages


def reset_chroma_database(db_path: Path) -> None:
    """
    Delete the existing ChromaDB folder so ingestion starts cleanly.
    """

    if db_path.exists():
        logging.info("Removing existing ChromaDB database at %s", db_path)
        shutil.rmtree(db_path)

    db_path.mkdir(parents=True, exist_ok=True)


def create_embeddings(
    model: SentenceTransformer,
    chunks: list[SemanticChunk],
) -> list[list[float]]:
    """
    Generate embeddings for each semantic chunk.
    """

    texts = [chunk.text for chunk in chunks]

    logging.info("Generating embeddings for %s chunks", len(texts))

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def prepare_documents(chunks: list[SemanticChunk]) -> list[str]:
    """
    Prepare chunk texts for ChromaDB.
    """

    return [chunk.text for chunk in chunks]


def prepare_ids(chunks: list[SemanticChunk]) -> list[str]:
    """
    Prepare stable chunk IDs for ChromaDB.
    """

    return [chunk.chunk_id for chunk in chunks]


def prepare_metadata(chunks: list[SemanticChunk]) -> list[dict[str, Any]]:
    """
    Prepare metadata for ChromaDB.

    ChromaDB metadata values should be simple types:
    str, int, float or bool.
    """

    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        metadata = {
            "tenant_id": chunk.tenant_id,
            "source_url": chunk.source_url,
            "url": chunk.source_url,
            "title": chunk.title,
            "section_heading": chunk.section_heading,
            "chunk_index": chunk.chunk_index,
            "chunk_type": chunk.chunk_type,
            "word_count": len(chunk.text.split()),
        }

        # Include any additional simple metadata from chunker.py
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        metadatas.append(metadata)

    return metadatas


def save_chunks_to_chromadb(
    chunks: list[SemanticChunk],
    embeddings: list[list[float]],
    db_path: Path,
    collection_name: str,
) -> None:
    """
    Save semantic chunks, metadata and embeddings to ChromaDB.
    """

    logging.info("Saving chunks to ChromaDB collection: %s", collection_name)

    client = chromadb.PersistentClient(path=str(db_path))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "Semantic/layout-aware NeuralRays website chunks",
        },
    )

    documents = prepare_documents(chunks)
    metadatas = prepare_metadata(chunks)
    ids = prepare_ids(chunks)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    logging.info("Saved %s chunks to ChromaDB", len(chunks))


def print_ingestion_summary(chunks: list[SemanticChunk]) -> None:
    """
    Print a useful summary after ingestion.
    """

    chunk_types: dict[str, int] = {}
    source_urls: set[str] = set()

    for chunk in chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        source_urls.add(chunk.source_url)

    print("\nIngestion Summary")
    print("-" * 50)
    print(f"Tenant ID:           {DEFAULT_TENANT_ID}")
    print(f"Total chunks:        {len(chunks)}")
    print(f"Source pages:        {len(source_urls)}")
    print(f"Embedding model:     {EMBEDDING_MODEL_NAME}")
    print(f"ChromaDB path:       {CHROMA_DB_PATH}")
    print(f"Collection:          {COLLECTION_NAME}")
    print(f"Chunk size words:    {DEFAULT_CHUNK_SIZE_WORDS}")
    print(f"Chunk overlap words: {DEFAULT_CHUNK_OVERLAP_WORDS}")
    print("-" * 50)

    print("Chunk types:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"  - {chunk_type}: {count}")

    print("-" * 50)


def main() -> None:
    """
    Main ingestion entry point.
    """

    setup_logging()

    pages = load_pages(DATA_FILE)

    logging.info("Creating semantic/layout-aware chunks")

    chunks = create_chunks_from_pages(
        pages=pages,
        tenant_id=DEFAULT_TENANT_ID,
        chunk_size_words=DEFAULT_CHUNK_SIZE_WORDS,
        overlap_words=DEFAULT_CHUNK_OVERLAP_WORDS,
    )

    if not chunks:
        raise ValueError("No chunks were created. Check data/pages.json and chunker.py.")

    logging.info("Created %s semantic/layout-aware chunks", len(chunks))

    reset_chroma_database(CHROMA_DB_PATH)

    logging.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = create_embeddings(
        model=model,
        chunks=chunks,
    )

    save_chunks_to_chromadb(
        chunks=chunks,
        embeddings=embeddings,
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    print_ingestion_summary(chunks)


if __name__ == "__main__":
    main()