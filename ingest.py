"""
Ingestion pipeline for the Neural Rays RAG chatbot.

This script loads crawled website content from data/pages.json, splits the text
into smaller chunks, creates local embeddings, and stores them in ChromaDB.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Input file created by crawler.py
DATA_FILE = Path("data/pages.json")

# Folder where ChromaDB will store the local vector database
CHROMA_DB_PATH = Path("data/chroma_db")

# Name of the ChromaDB collection
COLLECTION_NAME = "neuralrays_website"

# Local embedding model, so no OpenAI key is needed
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


@dataclass
class Page:
    """
    Represents one website page loaded from pages.json.
    """

    url: str
    title: str
    text: str


@dataclass
class TextChunk:
    """
    Represents one smaller text chunk created from a website page.
    """

    chunk_id: str
    text: str
    url: str
    title: str
    chunk_index: int


def setup_logging() -> None:
    """
    Set up logging so we can see ingestion progress in the terminal.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def load_pages(file_path: Path = DATA_FILE) -> list[Page]:
    """
    Load crawled website pages from the JSON file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run python crawler.py first."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        raw_pages = json.load(file)

    pages = [
        Page(
            url=page["url"],
            title=page.get("title", "Untitled Page"),
            text=page["text"],
        )
        for page in raw_pages
    ]

    logging.info("Loaded %s pages from %s", len(pages), file_path)

    return pages


def split_text_into_chunks(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Overlap helps avoid losing meaning when useful information sits on a chunk boundary.
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    start_index = 0

    while start_index < len(words):
        end_index = start_index + chunk_size_words
        chunk_words = words[start_index:end_index]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            chunks.append(chunk_text.strip())

        start_index += chunk_size_words - chunk_overlap_words

    return chunks


def create_chunks_from_pages(pages: list[Page]) -> list[TextChunk]:
    """
    Convert website pages into smaller chunks ready for embedding.
    """

    chunks: list[TextChunk] = []

    for page_index, page in enumerate(pages):
        page_chunks = split_text_into_chunks(page.text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = f"page-{page_index}-chunk-{chunk_index}"

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    url=page.url,
                    title=page.title,
                    chunk_index=chunk_index,
                )
            )

    logging.info("Created %s text chunks", len(chunks))

    return chunks


def reset_chroma_database(db_path: Path = CHROMA_DB_PATH) -> None:
    """
    Delete the existing local ChromaDB folder so ingestion starts cleanly.
    """

    if db_path.exists():
        shutil.rmtree(db_path)
        logging.info("Deleted existing vector database at %s", db_path)


def create_embeddings(
    chunks: list[TextChunk],
    model_name: str = EMBEDDING_MODEL_NAME,
) -> list[list[float]]:
    """
    Create embeddings for all chunks using a local sentence-transformers model.
    """

    logging.info("Loading embedding model: %s", model_name)

    model = SentenceTransformer(model_name)

    chunk_texts = [chunk.text for chunk in chunks]

    logging.info("Generating embeddings for %s chunks", len(chunk_texts))

    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def save_chunks_to_chromadb(
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> None:
    """
    Store chunks, embeddings, and metadata in ChromaDB.
    """

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Neural Rays website content for RAG chatbot"
        },
    )

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]

    metadatas = [
        {
            "url": chunk.url,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logging.info(
        "Stored %s chunks in ChromaDB collection '%s'",
        collection.count(),
        COLLECTION_NAME,
    )


def main() -> None:
    """
    Run the full ingestion pipeline.
    """

    setup_logging()

    pages = load_pages()
    chunks = create_chunks_from_pages(pages)

    if not chunks:
        raise ValueError("No chunks were created. Check data/pages.json content.")

    reset_chroma_database()

    embeddings = create_embeddings(chunks)
    save_chunks_to_chromadb(chunks, embeddings)

    logging.info("Ingestion completed successfully.")


if __name__ == "__main__":
    main()