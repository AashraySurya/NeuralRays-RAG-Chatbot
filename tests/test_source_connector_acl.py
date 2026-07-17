"""
Tests for the source connector and ACL enforcement.

Run from project root:

    python -m pytest tests/test_source_connector_acl.py
"""

from __future__ import annotations

from connectors.acl import can_access_document, filter_authorised_documents
from connectors.mcp_server import search_authorised_documents_in_memory
from connectors.mock_source import load_sample_documents
from connectors.models import EnterpriseDocument, UserContext


def test_public_document_is_accessible_to_any_user() -> None:
    document = EnterpriseDocument(
        document_id="public_doc",
        title="Public Handbook",
        source_type="confluence",
        source_url="confluence://company/handbook",
        content="General company handbook.",
        allowed_users=[],
        allowed_groups=["*"],
    )

    user = UserContext(
        email="anyone@example.com",
        groups=["engineering"],
    )

    assert can_access_document(user, document) is True


def test_group_restricted_document_is_accessible_to_matching_group() -> None:
    document = EnterpriseDocument(
        document_id="hr_doc",
        title="HR Policy",
        source_type="sharepoint",
        source_url="sharepoint://hr/policy",
        content="Salary review policy.",
        allowed_users=[],
        allowed_groups=["hr"],
    )

    user = UserContext(
        email="hr.user@example.com",
        groups=["hr"],
    )

    assert can_access_document(user, document) is True


def test_group_restricted_document_is_denied_to_wrong_group() -> None:
    document = EnterpriseDocument(
        document_id="hr_doc",
        title="HR Policy",
        source_type="sharepoint",
        source_url="sharepoint://hr/policy",
        content="Salary review policy.",
        allowed_users=[],
        allowed_groups=["hr"],
    )

    user = UserContext(
        email="engineer@example.com",
        groups=["engineering"],
    )

    assert can_access_document(user, document) is False


def test_user_specific_access_allows_named_user() -> None:
    document = EnterpriseDocument(
        document_id="architecture_doc",
        title="Architecture",
        source_type="confluence",
        source_url="confluence://engineering/architecture",
        content="RAG architecture.",
        allowed_users=["aashray@example.com"],
        allowed_groups=[],
    )

    user = UserContext(
        email="aashray@example.com",
        groups=[],
    )

    assert can_access_document(user, document) is True


def test_empty_acl_is_denied_by_default() -> None:
    document = EnterpriseDocument(
        document_id="locked_doc",
        title="Locked Document",
        source_type="s3",
        source_url="s3://locked/document",
        content="Restricted content.",
        allowed_users=[],
        allowed_groups=[],
    )

    user = UserContext(
        email="user@example.com",
        groups=["engineering"],
    )

    assert can_access_document(user, document) is False


def test_filter_authorised_documents_returns_only_allowed_documents() -> None:
    documents = [
        EnterpriseDocument(
            document_id="public_doc",
            title="Public Handbook",
            source_type="confluence",
            source_url="confluence://company/handbook",
            content="General content.",
            allowed_users=[],
            allowed_groups=["*"],
        ),
        EnterpriseDocument(
            document_id="hr_doc",
            title="HR Salary Policy",
            source_type="sharepoint",
            source_url="sharepoint://hr/salary",
            content="Salary content.",
            allowed_users=[],
            allowed_groups=["hr"],
        ),
    ]

    user = UserContext(
        email="engineer@example.com",
        groups=["engineering"],
    )

    authorised_documents = filter_authorised_documents(user, documents)

    assert len(authorised_documents) == 1
    assert authorised_documents[0].document_id == "public_doc"


def test_sample_documents_load_successfully() -> None:
    documents = load_sample_documents()

    assert len(documents) >= 5

    document_ids = {document.document_id for document in documents}

    assert "doc_public_handbook" in document_ids
    assert "doc_hr_salary_policy" in document_ids
    assert "doc_engineering_architecture" in document_ids


def test_in_memory_search_respects_acl_for_engineering_user() -> None:
    documents = load_sample_documents()

    user = UserContext(
        email="aashray@example.com",
        groups=["engineering"],
    )

    results = search_authorised_documents_in_memory(
        query="architecture vector database",
        user_context=user,
        documents=documents,
        top_k=5,
    )

    result_ids = {result.document_id for result in results}

    assert "doc_engineering_architecture" in result_ids
    assert "doc_hr_salary_policy" not in result_ids


def test_in_memory_search_denies_hr_document_to_engineering_user() -> None:
    documents = load_sample_documents()

    user = UserContext(
        email="aashray@example.com",
        groups=["engineering"],
    )

    results = search_authorised_documents_in_memory(
        query="salary compensation review",
        user_context=user,
        documents=documents,
        top_k=5,
    )

    result_ids = {result.document_id for result in results}

    assert "doc_hr_salary_policy" not in result_ids


def test_in_memory_search_allows_hr_document_to_hr_user() -> None:
    documents = load_sample_documents()

    user = UserContext(
        email="hr.user@example.com",
        groups=["hr"],
    )

    results = search_authorised_documents_in_memory(
        query="salary compensation review",
        user_context=user,
        documents=documents,
        top_k=5,
    )

    result_ids = {result.document_id for result in results}

    assert "doc_hr_salary_policy" in result_ids