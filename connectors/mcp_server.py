"""
MCP-style server for authorised enterprise source search.

This is a lightweight MCP-style prototype.

It does not depend on the official MCP SDK yet. Instead, it exposes a simple
tool interface that mirrors the idea of an MCP server:

- list available tools
- call a named tool with arguments
- return structured results

Main tool:
    search_authorised_documents

This demonstrates how a real MCP connector could enforce ACL-aware retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from connectors.acl import acl_reason_for_document, can_access_document
from connectors.mock_source import MockEnterpriseSource
from connectors.models import AuthorisedDocumentResult, EnterpriseDocument, UserContext
from connectors.sync import search_authorised_chunks


def tokenize(text: str) -> set[str]:
    """
    Tokenise text for lightweight in-memory search.
    """

    return {
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", text)
        if len(word) > 2
    }


def in_memory_score(query: str, document: EnterpriseDocument) -> float:
    """
    Simple keyword overlap score for in-memory connector tests/demo.
    """

    query_words = tokenize(query)

    searchable_text = (
        f"{document.title} "
        f"{document.source_type} "
        f"{document.content}"
    )

    document_words = tokenize(searchable_text)

    if not query_words:
        return 0.0

    overlap = query_words.intersection(document_words)

    return len(overlap) / len(query_words)


def search_authorised_documents_in_memory(
    query: str,
    user_context: UserContext,
    documents: list[EnterpriseDocument],
    top_k: int = 5,
) -> list[AuthorisedDocumentResult]:
    """
    Search mock documents in memory while enforcing ACL permissions.

    This is useful for quick tests without loading the embedding model.
    """

    results: list[AuthorisedDocumentResult] = []

    for document in documents:
        if not can_access_document(user_context, document):
            continue

        score = in_memory_score(query, document)

        if score <= 0:
            continue

        acl_reason = acl_reason_for_document(
            user_context=user_context,
            allowed_users=document.allowed_users,
            allowed_groups=document.allowed_groups,
        )

        results.append(
            AuthorisedDocumentResult(
                chunk_id=f"{document.document_id}_in_memory",
                document_id=document.document_id,
                title=document.title,
                source_type=document.source_type,
                source_url=document.source_url,
                text=document.content,
                score=round(score, 4),
                acl_reason=acl_reason or "allowed",
                metadata=document.metadata,
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)

    return results[:top_k]


class MCPStyleServer:
    """
    Lightweight MCP-style server.

    The real MCP version can later expose the same tools using the MCP SDK.
    """

    def list_tools(self) -> list[dict[str, Any]]:
        """
        List available MCP-style tools.
        """

        return [
            {
                "name": "search_authorised_documents",
                "description": (
                    "Search enterprise source documents while enforcing user and group ACL permissions."
                ),
                "input_schema": {
                    "query": "string",
                    "user_email": "string",
                    "user_groups": "list[string]",
                    "top_k": "integer",
                    "use_vector_store": "boolean",
                },
            },
            {
                "name": "list_available_sources",
                "description": "List mock enterprise source systems available to the connector.",
                "input_schema": {},
            },
        ]

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call a named MCP-style tool.
        """

        if tool_name == "list_available_sources":
            return self._list_available_sources()

        if tool_name == "search_authorised_documents":
            return self._search_authorised_documents(arguments)

        raise ValueError(f"Unknown MCP-style tool: {tool_name}")

    def _list_available_sources(self) -> dict[str, Any]:
        """
        Return mock source systems.
        """

        return {
            "sources": [
                {
                    "source_type": "confluence",
                    "description": "Mock Confluence knowledge base documents",
                },
                {
                    "source_type": "sharepoint",
                    "description": "Mock SharePoint policy and operations documents",
                },
                {
                    "source_type": "s3",
                    "description": "Mock S3 technical and analytics documents",
                },
            ]
        }

    def _search_authorised_documents(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Search authorised documents.

        By default this tries vector search against ChromaDB.
        Set use_vector_store=False for in-memory mock search.
        """

        query = str(arguments.get("query", "")).strip()
        user_email = str(arguments.get("user_email", "")).strip()
        user_groups = list(arguments.get("user_groups", []))
        top_k = int(arguments.get("top_k", 5))
        use_vector_store = bool(arguments.get("use_vector_store", True))

        if not query:
            raise ValueError("query is required")

        if not user_email:
            raise ValueError("user_email is required")

        user_context = UserContext(
            email=user_email,
            groups=[str(group) for group in user_groups],
        )

        if use_vector_store:
            results = search_authorised_chunks(
                query=query,
                user_context=user_context,
                top_k=top_k,
            )
        else:
            source = MockEnterpriseSource()
            documents = source.fetch_documents()

            results = search_authorised_documents_in_memory(
                query=query,
                user_context=user_context,
                documents=documents,
                top_k=top_k,
            )

        return {
            "query": query,
            "user_email": user_context.email,
            "user_groups": user_context.groups,
            "results": [asdict(result) for result in results],
            "result_count": len(results),
            "acl_enforced": True,
        }


def demo() -> None:
    """
    Run a simple command-line demo.
    """

    server = MCPStyleServer()

    print("\nAvailable MCP-style tools:")
    print(json.dumps(server.list_tools(), indent=2))

    print("\nDemo 1: Engineering user searching for architecture")
    engineering_response = server.call_tool(
        "search_authorised_documents",
        {
            "query": "architecture authentication vector database",
            "user_email": "aashray@example.com",
            "user_groups": ["engineering"],
            "top_k": 5,
            "use_vector_store": False,
        },
    )

    print(json.dumps(engineering_response, indent=2))

    print("\nDemo 2: HR user searching for salary policy")
    hr_response = server.call_tool(
        "search_authorised_documents",
        {
            "query": "salary bands compensation review",
            "user_email": "hr.user@example.com",
            "user_groups": ["hr"],
            "top_k": 5,
            "use_vector_store": False,
        },
    )

    print(json.dumps(hr_response, indent=2))

    print("\nDemo 3: Engineering user searching for HR salary policy")
    denied_response = server.call_tool(
        "search_authorised_documents",
        {
            "query": "salary bands compensation review",
            "user_email": "aashray@example.com",
            "user_groups": ["engineering"],
            "top_k": 5,
            "use_vector_store": False,
        },
    )

    print(json.dumps(denied_response, indent=2))

    print(
        "\nNotice: the engineering user does not receive the restricted HR document."
    )


if __name__ == "__main__":
    demo()