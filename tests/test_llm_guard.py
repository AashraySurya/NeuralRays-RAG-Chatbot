"""
Tests for the LLM PII guard.

Run from project root:

    python -m pytest tests/test_llm_guard.py
"""

from __future__ import annotations

from pii.llm_guard import LLMPiiGuard, prepare_redacted_prompt


def test_prepare_payload_redacts_question_and_context(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"

    guard = LLMPiiGuard(
        audit_log_path=audit_file,
        source="test_llm_guard",
    )

    question = "Can you summarise the case for john@example.com?"
    contexts = [
        "Name: John Smith. Phone: +91 9876543210. PAN: ABCDE1234F."
    ]

    payload = guard.prepare_payload(
        question=question,
        contexts=contexts,
    )

    assert payload.has_pii is True

    assert "john@example.com" not in payload.redacted_question
    assert "[EMAIL_1]" in payload.redacted_question

    assert "John Smith" not in payload.redacted_contexts[0]
    assert "+91 9876543210" not in payload.redacted_contexts[0]
    assert "ABCDE1234F" not in payload.redacted_contexts[0]

    assert "[NAME_1]" in payload.redacted_contexts[0]
    assert "[PHONE_1]" in payload.redacted_contexts[0]
    assert "[PAN_1]" in payload.redacted_contexts[0]

    assert payload.redaction_map["[EMAIL_1]"] == "john@example.com"
    assert payload.redaction_map["[NAME_1]"] == "John Smith"
    assert payload.redaction_map["[PHONE_1]"] == "+91 9876543210"
    assert payload.redaction_map["[PAN_1]"] == "ABCDE1234F"


def test_build_redacted_prompt_removes_original_pii(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"

    guard = LLMPiiGuard(
        audit_log_path=audit_file,
        source="test_llm_guard",
    )

    guarded_prompt = guard.build_redacted_prompt(
        question="What should I tell Priya? Her email is priya@example.com.",
        contexts=[
            "Aadhaar: 1234 5678 9012. Address: 12 MG Road Chennai 600001."
        ],
    )

    prompt = guarded_prompt.redacted_prompt

    assert "priya@example.com" not in prompt
    assert "1234 5678 9012" not in prompt
    assert "12 MG Road Chennai 600001" not in prompt

    assert "[EMAIL_1]" in prompt
    assert "[AADHAAR_1]" in prompt
    assert "[ADDRESS_1]" in prompt

    assert guarded_prompt.payload.has_pii is True


def test_restore_model_output(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"

    guard = LLMPiiGuard(
        audit_log_path=audit_file,
        source="test_llm_guard",
    )

    guarded_prompt = guard.build_redacted_prompt(
        question="Can you contact john@example.com?",
        contexts=["Phone: +91 9876543210."],
    )

    fake_model_output = "The user can be contacted at [EMAIL_1] or [PHONE_1]."

    restored_output = guard.restore_model_output(
        model_output=fake_model_output,
        redaction_map=guarded_prompt.payload.redaction_map,
    )

    assert restored_output == (
        "The user can be contacted at john@example.com or +91 9876543210."
    )


def test_prepare_payload_without_pii(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"

    guard = LLMPiiGuard(
        audit_log_path=audit_file,
        source="test_llm_guard",
    )

    payload = guard.prepare_payload(
        question="What services does the company offer?",
        contexts=["The company offers AI consulting and cloud transformation."],
    )

    assert payload.has_pii is False
    assert payload.redaction_map == {}
    assert payload.redacted_question == "What services does the company offer?"
    assert payload.redacted_contexts == [
        "The company offers AI consulting and cloud transformation."
    ]


def test_prepare_redacted_prompt_helper(tmp_path):
    audit_file = tmp_path / "audit_log.jsonl"

    guarded_prompt = prepare_redacted_prompt(
        question="Please review the record for john@example.com.",
        contexts=["PAN: ABCDE1234F."],
        audit_log_path=audit_file,
        source="helper_test",
    )

    assert "john@example.com" not in guarded_prompt.redacted_prompt
    assert "ABCDE1234F" not in guarded_prompt.redacted_prompt

    assert "[EMAIL_1]" in guarded_prompt.redacted_prompt
    assert "[PAN_1]" in guarded_prompt.redacted_prompt