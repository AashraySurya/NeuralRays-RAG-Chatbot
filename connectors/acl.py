"""
ACL enforcement for enterprise source documents.

This module checks whether a user is allowed to access a document or chunk.

The rule is:
- "*" in allowed_users means all users can access the document
- "*" in allowed_groups means all groups can access the document
- otherwise, the user's email must be in allowed_users
- or at least one of the user's groups must be in allowed_groups
- if both lists are empty, deny by default
"""

from __future__ import annotations

import json

from connectors.models import EnterpriseDocument, SourceChunk, UserContext


def normalise_email(email: str) -> str:
    """
    Normalise an email address for comparison.
    """

    return email.strip().lower()


def normalise_group(group: str) -> str:
    """
    Normalise a group name for comparison.
    """

    return group.strip().lower()


def normalise_list(values: list[str]) -> list[str]:
    """
    Normalise a list of string values.
    """

    return [normalise_group(value) for value in values]


def acl_reason_for_document(
    user_context: UserContext,
    allowed_users: list[str],
    allowed_groups: list[str],
) -> str | None:
    """
    Return the reason access is allowed.

    Returns None if access should be denied.
    """

    user_email = normalise_email(user_context.email)
    user_groups = set(normalise_list(user_context.groups))

    document_users = set(normalise_email(user) for user in allowed_users)
    document_groups = set(normalise_list(allowed_groups))

    if "*" in document_users:
        return "allowed_by_public_user_wildcard"

    if "*" in document_groups:
        return "allowed_by_public_group_wildcard"

    if not document_users and not document_groups:
        return None

    if user_email in document_users:
        return "allowed_by_user"

    matching_groups = user_groups.intersection(document_groups)

    if matching_groups:
        return f"allowed_by_group:{sorted(matching_groups)[0]}"

    return None


def can_access_document(
    user_context: UserContext,
    document: EnterpriseDocument,
) -> bool:
    """
    Return True if the user can access the enterprise document.
    """

    return (
        acl_reason_for_document(
            user_context=user_context,
            allowed_users=document.allowed_users,
            allowed_groups=document.allowed_groups,
        )
        is not None
    )


def can_access_chunk(
    user_context: UserContext,
    chunk: SourceChunk,
) -> bool:
    """
    Return True if the user can access the source chunk.
    """

    return (
        acl_reason_for_document(
            user_context=user_context,
            allowed_users=chunk.allowed_users,
            allowed_groups=chunk.allowed_groups,
        )
        is not None
    )


def filter_authorised_documents(
    user_context: UserContext,
    documents: list[EnterpriseDocument],
) -> list[EnterpriseDocument]:
    """
    Return only documents the user is allowed to access.
    """

    return [
        document
        for document in documents
        if can_access_document(user_context, document)
    ]


def encode_acl_list(values: list[str]) -> str:
    """
    Encode an ACL list for ChromaDB metadata.

    ChromaDB metadata values should be simple scalar values, so we store lists as JSON strings.
    """

    return json.dumps(values)


def decode_acl_list(value: str | None) -> list[str]:
    """
    Decode an ACL list from ChromaDB metadata.
    """

    if not value:
        return []

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []

    if not isinstance(decoded, list):
        return []

    return [str(item) for item in decoded]


def acl_reason_from_metadata(
    user_context: UserContext,
    metadata: dict,
) -> str | None:
    """
    Check ACL access using ChromaDB metadata.
    """

    allowed_users = decode_acl_list(metadata.get("allowed_users"))
    allowed_groups = decode_acl_list(metadata.get("allowed_groups"))

    return acl_reason_for_document(
        user_context=user_context,
        allowed_users=allowed_users,
        allowed_groups=allowed_groups,
    )