"""
Data models for enterprise source connectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserContext:
    """
    Represents the user making a retrieval request.

    email:
        The user's email address.

    groups:
        The user's enterprise groups, for example:
        ["engineering"], ["hr"], ["leadership"].
    """

    email: str
    groups: list[str] = field(default_factory=list)


@dataclass
class EnterpriseDocument:
    """
    Represents a document from an enterprise source system.
    """

    document_id: str
    title: str
    source_type: str
    source_url: str
    content: str
    allowed_users: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceChunk:
    """
    Represents a chunk created from an enterprise document.
    """

    chunk_id: str
    document_id: str
    title: str
    source_type: str
    source_url: str
    text: str
    chunk_index: int
    allowed_users: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorisedDocumentResult:
    """
    Represents an authorised search result returned to a user.
    """

    chunk_id: str
    document_id: str
    title: str
    source_type: str
    source_url: str
    text: str
    score: float
    acl_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    