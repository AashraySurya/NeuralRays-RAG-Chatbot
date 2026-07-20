"""
Tests for the extended PII protection pipeline.

Run from project root:

    python -m pytest tests/test_pii_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pii.config import PIIPipelineConfig
from pii.ingestion_guard import PIIIngestionGuard
from pii.output_validator import PIIOutputValidator
from pii.token_store import PIITokenStore


def create_config(
    tmp_path: Path,
    masking_mode: str,
) -> PIIPipelineConfig:
    """
    Create a test config using temporary paths.
    """

    return PIIPipelineConfig(
        masking_mode=masking_mode,
        customer_id="cust",
        audit_log_path=tmp_path / "audit_log.jsonl",
        token_store_path=tmp_path / "pii_tokens.sqlite3",
    )


def test_partial_masking_email_for_ingestion(tmp_path: Path) -> None:
    config = create_config(
        tmp_path=tmp_path,
        masking_mode="partial_mask",
    )

    guard = PIIIngestionGuard(config=config)

    result = guard.process_document(
        document_id="doc_1",
        text="Customer email is rahul@gmail.com.",
    )

    assert result.has_pii is True
    assert "rahul@gmail.com" not in result.safe_text
    assert "rah***@gmail.com" in result.safe_text
    assert result.pii_counts["EMAIL"] == 1


def test_full_masking_email_for_ingestion(tmp_path: Path) -> None:
    config = create_config(
        tmp_path=tmp_path,
        masking_mode="full_mask",
    )

    guard = PIIIngestionGuard(config=config)

    result = guard.process_document(
        document_id="doc_1",
        text="Customer email is rahul@gmail.com.",
    )

    assert result.has_pii is True
    assert "rahul@gmail.com" not in result.safe_text
    assert "***************" in result.safe_text
    assert result.pii_counts["EMAIL"] == 1


def test_tokenisation_email_for_ingestion(tmp_path: Path) -> None:
    config = create_config(
        tmp_path=tmp_path,
        masking_mode="tokenise",
    )

    guard = PIIIngestionGuard(config=config)

    result = guard.process_document(
        document_id="doc_1",
        text="Customer email is rahul@gmail.com.",
        source="test_ingestion",
    )

    assert result.has_pii is True
    assert "rahul@gmail.com" not in result.safe_text
    assert "<email_cust_001>" in result.safe_text
    assert result.tokens_created == ["<email_cust_001>"]

    restored_text = guard.restore_tokens(result.safe_text)

    assert restored_text == "Customer email is rahul@gmail.com."


def test_sqlite_token_store_maps_token_to_original_value(tmp_path: Path) -> None:
    token_store = PIITokenStore(
        db_path=tmp_path / "tokens.sqlite3",
    )

    token = token_store.create_token(
        pii_type="EMAIL",
        original_value="rahul@gmail.com",
        customer_id="cust",
        source="test",
    )

    assert token == "<email_cust_001>"
    assert token_store.get_original_value(token) == "rahul@gmail.com"
    assert token_store.count_tokens() == 1


def test_multiple_pii_types_are_masked_before_vector_storage(tmp_path: Path) -> None:
    config = create_config(
        tmp_path=tmp_path,
        masking_mode="partial_mask",
    )

    guard = PIIIngestionGuard(config=config)

    raw_text = (
        "Name: John Smith. "
        "Email: john@example.com. "
        "Phone: +91 9876543210. "
        "PAN: ABCDE1234F. "
        "Aadhaar: 1234 5678 9012. "
        "Address: 12 MG Road Chennai 600001."
    )

    result = guard.process_document(
        document_id="doc_sensitive",
        text=raw_text,
    )

    assert "john@example.com" not in result.safe_text
    assert "+91 9876543210" not in result.safe_text
    assert "ABCDE1234F" not in result.safe_text
    assert "1234 5678 9012" not in result.safe_text
    assert "John Smith" not in result.safe_text
    assert "12 MG Road Chennai 600001" not in result.safe_text

    assert result.pii_counts["EMAIL"] == 1
    assert result.pii_counts["PHONE"] == 1
    assert result.pii_counts["PAN"] == 1
    assert result.pii_counts["AADHAAR"] == 1
    assert result.pii_counts["NAME"] == 1
    assert result.pii_counts["ADDRESS"] == 1


def test_audit_log_records_mask_but_not_original_pii(tmp_path: Path) -> None:
    config = create_config(
        tmp_path=tmp_path,
        masking_mode="partial_mask",
    )

    guard = PIIIngestionGuard(config=config)

    guard.process_document(
        document_id="doc_1",
        text="Customer email is rahul@gmail.com.",
        source="test_audit",
    )

    audit_text = config.audit_log_path.read_text(encoding="utf-8")

    assert "EMAIL" in audit_text
    assert "rah***@gmail.com" in audit_text
    assert "rahul@gmail.com" not in audit_text


def test_output_validator_detects_raw_pii_leakage() -> None:
    validator = PIIOutputValidator()

    result = validator.validate(
        output_text="Please contact rahul@gmail.com for more information."
    )

    assert result.is_safe is False
    assert result.detected_pii["EMAIL"] == ["rahul@gmail.com"]
    assert result.violation_count == 1


def test_output_validator_allows_masked_output() -> None:
    validator = PIIOutputValidator()

    result = validator.validate(
        output_text="Please contact rah***@gmail.com for more information."
    )

    assert result.is_safe is True
    assert result.detected_pii == {}


def test_output_validator_can_allow_pii_when_explicitly_permitted() -> None:
    validator = PIIOutputValidator()

    result = validator.validate(
        output_text="Please contact rahul@gmail.com for more information.",
        allow_pii=True,
    )

    assert result.is_safe is True
    assert result.detected_pii["EMAIL"] == ["rahul@gmail.com"]


def test_output_validator_assert_safe_raises_on_pii() -> None:
    validator = PIIOutputValidator()

    with pytest.raises(ValueError):
        validator.assert_safe(
            output_text="Please contact rahul@gmail.com for more information."
        )