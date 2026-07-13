"""
Configuration for the domain-agnostic RAG evaluation service.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_COLLECTION_NAME = "neuralrays_website"

EVALUATION_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

DEFAULT_TENANT_ID = "neuralrays"
DEFAULT_ADAPTER = "simple"

DEFAULT_MAX_QUESTIONS = 20
MIN_CHUNK_CHARACTERS = 120