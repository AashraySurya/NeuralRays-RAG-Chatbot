"""
Tests for the PII redaction module.

Run from project root:

    python -m pytest tests/test_pii_redactor.py
"""

from __future__ import annotations

import json

from pii.redactor import PIIRedactor, redact_text


def test_redacts_email(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    result = redactor.redact("Please contact john@example.com for details.")

    assert "[EMAIL_1]" in result.redacted_text
    assert "john@example.com" not in result.redacted_text
    assert result.redaction_map["[EMAIL_1]"] == "john@example.com"
    assert result.pii_counts["EMAIL"] == 1


def test_redacts_indian_phone_number(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    result = redactor.redact("My phone number is +91 9876543210.")

    assert "[PHONE_1]" in result.redacted_text
    assert "+91 9876543210" not in result.redacted_text
    assert result.redaction_map["[PHONE_1]"] == "+91 9876543210"
    assert result.pii_counts["PHONE"] == 1


def test_redacts_aadhaar_number(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    result = redactor.redact("My Aadhaar is 1234 5678 9012.")

    assert "[AADHAAR_1]" in result.redacted_text
    assert "1234 5678 9012" not in result.redacted_text
    assert result.redaction_map["[AADHAAR_1]"] == "1234 5678 9012"
    assert result.pii_counts["AADHAAR"] == 1


def test_redacts_pan_number(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    result = redactor.redact("My PAN is ABCDE1234F.")

    assert "[PAN_1]" in result.redacted_text
    assert "ABCDE1234F" not in result.redacted_text
    assert result.redaction_map["[PAN_1]"] == "ABCDE1234F"
    assert result.pii_counts["PAN"] == 1


def test_redacts_labelled_name(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    result = redactor.redact("Name: John Smith. The request is approved.")

    assert "[NAME_1]" in result.redacted_text
    assert "John Smith" not in result.redacted_text
    assert result.redaction_map["[NAME_1]"] == "John Smith"
    assert result.pii_counts["NAME"] == 1


def test_redacts_labelled_address(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    result = redactor.redact("Address: 12 MG Road Chennai 600001.")

    assert "[ADDRESS_1]" in result.redacted_text
    assert "12 MG Road Chennai 600001" not in result.redacted_text
    assert result.redaction_map["[ADDRESS_1]"] == "12 MG Road Chennai 600001"
    assert result.pii_counts["ADDRESS"] == 1


def test_multiple_pii_values_are_redacted(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    text = (
        "Name: John Smith. Email: john@example.com. "
        "Phone: +91 9876543210. PAN: ABCDE1234F."
    )

    result = redactor.redact(text)

    assert "[NAME_1]" in result.redacted_text
    assert "[EMAIL_1]" in result.redacted_text
    assert "[PHONE_1]" in result.redacted_text
    assert "[PAN_1]" in result.redacted_text

    assert result.pii_counts["NAME"] == 1
    assert result.pii_counts["EMAIL"] == 1
    assert result.pii_counts["PHONE"] == 1
    assert result.pii_counts["PAN"] == 1


def test_restore_redacted_text(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    original_text = "Please contact john@example.com or +91 9876543210."
    result = redactor.redact(original_text)

    restored_text = redactor.restore(
        redacted_text=result.redacted_text,
        redaction_map=result.redaction_map,
    )

    assert restored_text == original_text


def test_audit_log_is_created_without_original_pii(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"
    redactor = PIIRedactor(audit_log_path=audit_file, source="test")

    redactor.redact("My email is john@example.com.")

    assert audit_file.exists()

    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 1

    event = json.loads(lines[0])

    assert event["event_type"] == "PII_REDACTION"
    assert event["pii_type"] == "EMAIL"
    assert event["mask"] == "[EMAIL_1]"
    assert event["source"] == "test"

    # The audit log must not store the original PII value.
    assert "john@example.com" not in lines[0]


def test_redact_text_helper_function(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"

    result = redact_text(
        text="Contact john@example.com.",
        audit_log_path=audit_file,
        source="helper_test",
    )

    assert "[EMAIL_1]" in result.redacted_text
    assert result.redaction_map["[EMAIL_1]"] == "john@example.com"