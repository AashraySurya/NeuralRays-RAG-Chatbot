"""
PII redaction module.

This module detects and masks common sensitive values before text is sent
to an external LLM API.

Supported first-version PII types:
- Email addresses
- Indian phone numbers
- Aadhaar numbers
- PAN numbers
- Simple labelled names
- Simple labelled addresses

The masking is reversible within the RedactionResult mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pii.audit import PIIAuditLogger


@dataclass
class RedactionResult:
    """
    Result returned after redacting text.
    """

    original_text: str
    redacted_text: str
    redaction_map: dict[str, str] = field(default_factory=dict)
    pii_counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_pii(self) -> bool:
        """
        Whether any PII was detected.
        """

        return bool(self.redaction_map)


class PIIRedactor:
    """
    Detects and masks PII in text.

    The redaction map allows masked values to be restored later if needed.
    """

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    AADHAAR_PATTERN = re.compile(
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    )

    PAN_PATTERN = re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    )

    PHONE_PATTERN = re.compile(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
    )

    NAME_PATTERN = re.compile(
        r"\b(?:my name is|name is|name:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        re.IGNORECASE,
    )

    ADDRESS_PATTERN = re.compile(
        r"\b(?:address is|address:)\s+([^.\n]+(?:road|street|lane|avenue|drive|chennai|dubai|london|mumbai|delhi|bangalore|pune|hyderabad|kolkata|[0-9]{6})[^.\n]*)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        audit_log_path: Path | str = Path("pii/audit_log.jsonl"),
        source: str = "rag_pipeline",
    ) -> None:
        self.audit_logger = PIIAuditLogger(audit_log_path)
        self.source = source

    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII from text and return redacted text plus reversible mapping.
        """

        redacted_text = text
        redaction_map: dict[str, str] = {}
        pii_counts: dict[str, int] = {}

        # Specific structured identifiers first.
        redacted_text = self._redact_pattern(
            text=redacted_text,
            pii_type="EMAIL",
            pattern=self.EMAIL_PATTERN,
            redaction_map=redaction_map,
            pii_counts=pii_counts,
        )

        redacted_text = self._redact_pattern(
            text=redacted_text,
            pii_type="AADHAAR",
            pattern=self.AADHAAR_PATTERN,
            redaction_map=redaction_map,
            pii_counts=pii_counts,
        )

        redacted_text = self._redact_pattern(
            text=redacted_text,
            pii_type="PAN",
            pattern=self.PAN_PATTERN,
            redaction_map=redaction_map,
            pii_counts=pii_counts,
        )

        redacted_text = self._redact_pattern(
            text=redacted_text,
            pii_type="PHONE",
            pattern=self.PHONE_PATTERN,
            redaction_map=redaction_map,
            pii_counts=pii_counts,
        )

        # Labelled free-text PII.
        redacted_text = self._redact_labelled_group(
            text=redacted_text,
            pii_type="NAME",
            pattern=self.NAME_PATTERN,
            redaction_map=redaction_map,
            pii_counts=pii_counts,
        )

        redacted_text = self._redact_labelled_group(
            text=redacted_text,
            pii_type="ADDRESS",
            pattern=self.ADDRESS_PATTERN,
            redaction_map=redaction_map,
            pii_counts=pii_counts,
        )

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            redaction_map=redaction_map,
            pii_counts=pii_counts,
        )

    def restore(self, redacted_text: str, redaction_map: dict[str, str]) -> str:
        """
        Restore masked values using a redaction map.
        """

        restored_text = redacted_text

        # Replace longer masks first to avoid accidental partial replacement.
        for mask in sorted(redaction_map.keys(), key=len, reverse=True):
            restored_text = restored_text.replace(mask, redaction_map[mask])

        return restored_text

    def redact_before_external_llm(self, text: str) -> RedactionResult:
        """
        Convenience method showing where this module fits in the RAG flow.

        Use this before sending a prompt, user question, or retrieved context
        to an external LLM provider.
        """

        return self.redact(text)

    def _next_mask(
        self,
        pii_type: str,
        pii_counts: dict[str, int],
    ) -> str:
        """
        Create the next mask token for a PII type.
        """

        pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1

        return f"[{pii_type}_{pii_counts[pii_type]}]"

    def _store_mapping_and_audit(
        self,
        pii_type: str,
        original_value: str,
        redaction_map: dict[str, str],
        pii_counts: dict[str, int],
    ) -> str:
        """
        Store reversible mapping and write audit event.
        """

        mask = self._next_mask(pii_type, pii_counts)
        redaction_map[mask] = original_value

        self.audit_logger.log_redaction(
            pii_type=pii_type,
            mask=mask,
            source=self.source,
        )

        return mask

    def _redact_pattern(
        self,
        text: str,
        pii_type: str,
        pattern: re.Pattern[str],
        redaction_map: dict[str, str],
        pii_counts: dict[str, int],
    ) -> str:
        """
        Redact regex matches where the whole match is the sensitive value.
        """

        def replacement(match: re.Match[str]) -> str:
            original_value = match.group(0)

            return self._store_mapping_and_audit(
                pii_type=pii_type,
                original_value=original_value,
                redaction_map=redaction_map,
                pii_counts=pii_counts,
            )

        return pattern.sub(replacement, text)

    def _redact_labelled_group(
        self,
        text: str,
        pii_type: str,
        pattern: re.Pattern[str],
        redaction_map: dict[str, str],
        pii_counts: dict[str, int],
    ) -> str:
        """
        Redact only the captured sensitive value in labelled patterns.

        Example:
            "Name: John Smith" -> "Name: [NAME_1]"
        """

        def replacement(match: re.Match[str]) -> str:
            full_match = match.group(0)
            sensitive_value = match.group(1).strip()

            mask = self._store_mapping_and_audit(
                pii_type=pii_type,
                original_value=sensitive_value,
                redaction_map=redaction_map,
                pii_counts=pii_counts,
            )

            return full_match.replace(sensitive_value, mask)

        return pattern.sub(replacement, text)


def redact_text(
    text: str,
    audit_log_path: Path | str = Path("pii/audit_log.jsonl"),
    source: str = "rag_pipeline",
) -> RedactionResult:
    """
    Simple helper function for callers that do not need to manage a PIIRedactor instance.
    """

    redactor = PIIRedactor(
        audit_log_path=audit_log_path,
        source=source,
    )

    return redactor.redact(text)