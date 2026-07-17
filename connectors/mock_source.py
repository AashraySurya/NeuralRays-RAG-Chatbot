"""
Mock enterprise source connector.

This simulates an enterprise content source such as:
- Confluence
- SharePoint
- S3
- Internal knowledge base
- Database-backed document store

The purpose is to show the connector pattern without needing real enterprise
credentials during the prototype.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from connectors.models import EnterpriseDocument


DEFAULT_SAMPLE_DOCUMENTS_PATH = Path("connectors/sample_documents.jsonl")


def load_sample_documents(
    file_path: Path | str = DEFAULT_SAMPLE_DOCUMENTS_PATH,
) -> list[EnterpriseDocument]:
    """
    Load mock enterprise documents from a JSONL file.

    Each line should contain one JSON object with:
    - document_id
    - title
    - source_type
    - source_url
    - content
    - allowed_users
    - allowed_groups
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Sample documents file not found: {path}")

    documents: list[EnterpriseDocument] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw_document: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {error}"
                ) from error

            documents.append(
                EnterpriseDocument(
                    document_id=str(raw_document["document_id"]),
                    title=str(raw_document["title"]),
                    source_type=str(raw_document["source_type"]),
                    source_url=str(raw_document["source_url"]),
                    content=str(raw_document["content"]),
                    allowed_users=list(raw_document.get("allowed_users", [])),
                    allowed_groups=list(raw_document.get("allowed_groups", [])),
                    metadata=dict(raw_document.get("metadata", {})),
                )
            )

    return documents


class MockEnterpriseSource:
    """
    Mock source connector that simulates fetching documents from an enterprise system.
    """

    def __init__(
        self,
        file_path: Path | str = DEFAULT_SAMPLE_DOCUMENTS_PATH,
    ) -> None:
        self.file_path = Path(file_path)

    def fetch_documents(self) -> list[EnterpriseDocument]:
        """
        Fetch all documents from the mock source.
        """

        return load_sample_documents(self.file_path)