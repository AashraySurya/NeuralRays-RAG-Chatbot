"""
PII ingestion guard.

This module protects the document ingestion stage.

Flow:

Raw document text
-> PII detection
-> config-based redaction/tokenisation
-> safe text stored in Vector DB
-> audit event written

This means sensitive information can be handled before it reaches ChromaDB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pii.audit import PIIAuditLogger
from pii.config import PIIPipelineConfig, default_pii_config
from pii.token_store import PIITokenStore


@dataclass
class IngestionGuardResult:
    """
    Result returned after processing a document for ingestion.
    """

    document_id: str
    original_text: str
    safe_text: str
    masking_mode: str
    pii_counts: dict[str, int] = field(default_factory=dict)
    redaction_map: dict[str, str] = field(default_factory=dict)
    tokens_created: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_pii(self) -> bool:
        """
        Whether any PII was detected.
        """

        return bool(self.pii_counts)


class PIIIngestionGuard:
    """
    Applies PII protection before documents are stored in the vector database.
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
        config: PIIPipelineConfig | None = None,
    ) -> None:
        self.config = config or default_pii_config()
        self.config.validate()

        self.audit_logger = PIIAuditLogger(self.config.audit_log_path)
        self.token_store = PIITokenStore(self.config.token_store_path)

    def process_document(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str = "document_ingestion",
    ) -> IngestionGuardResult:
        """
        Detect and mask/tokenise PII before storing document text.
        """

        if not self.config.redact_at_ingestion:
            return IngestionGuardResult(
                document_id=document_id,
                original_text=text,
                safe_text=text,
                masking_mode="none",
                metadata=metadata or {},
            )

        safe_text = text
        pii_counts: dict[str, int] = {}
        redaction_map: dict[str, str] = {}
        tokens_created: list[str] = []

        # Order matters: Aadhaar before phone so Aadhaar is not partly detected as a phone number.
        safe_text = self._replace_full_match_pattern(
            text=safe_text,
            pii_type="EMAIL",
            pattern=self.EMAIL_PATTERN,
            pii_counts=pii_counts,
            redaction_map=redaction_map,
            tokens_created=tokens_created,
            source=source,
            metadata=metadata,
        )

        safe_text = self._replace_full_match_pattern(
            text=safe_text,
            pii_type="AADHAAR",
            pattern=self.AADHAAR_PATTERN,
            pii_counts=pii_counts,
            redaction_map=redaction_map,
            tokens_created=tokens_created,
            source=source,
            metadata=metadata,
        )

        safe_text = self._replace_full_match_pattern(
            text=safe_text,
            pii_type="PAN",
            pattern=self.PAN_PATTERN,
            pii_counts=pii_counts,
            redaction_map=redaction_map,
            tokens_created=tokens_created,
            source=source,
            metadata=metadata,
        )

        safe_text = self._replace_full_match_pattern(
            text=safe_text,
            pii_type="PHONE",
            pattern=self.PHONE_PATTERN,
            pii_counts=pii_counts,
            redaction_map=redaction_map,
            tokens_created=tokens_created,
            source=source,
            metadata=metadata,
        )

        safe_text = self._replace_captured_group_pattern(
            text=safe_text,
            pii_type="NAME",
            pattern=self.NAME_PATTERN,
            pii_counts=pii_counts,
            redaction_map=redaction_map,
            tokens_created=tokens_created,
            source=source,
            metadata=metadata,
        )

        safe_text = self._replace_captured_group_pattern(
            text=safe_text,
            pii_type="ADDRESS",
            pattern=self.ADDRESS_PATTERN,
            pii_counts=pii_counts,
            redaction_map=redaction_map,
            tokens_created=tokens_created,
            source=source,
            metadata=metadata,
        )

        return IngestionGuardResult(
            document_id=document_id,
            original_text=text,
            safe_text=safe_text,
            masking_mode=self.config.masking_mode,
            pii_counts=pii_counts,
            redaction_map=redaction_map,
            tokens_created=tokens_created,
            metadata=metadata or {},
        )

    def restore_tokens(self, text: str) -> str:
        """
        Restore tokenised placeholders from SQLite.
        """

        return self.token_store.remap_text(text)

    def _replace_full_match_pattern(
        self,
        text: str,
        pii_type: str,
        pattern: re.Pattern[str],
        pii_counts: dict[str, int],
        redaction_map: dict[str, str],
        tokens_created: list[str],
        source: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """
        Replace matches where the whole regex match is the PII value.
        """

        def replacement(match: re.Match[str]) -> str:
            original_value = match.group(0)

            return self._mask_value(
                pii_type=pii_type,
                original_value=original_value,
                pii_counts=pii_counts,
                redaction_map=redaction_map,
                tokens_created=tokens_created,
                source=source,
                metadata=metadata,
            )

        return pattern.sub(replacement, text)

    def _replace_captured_group_pattern(
        self,
        text: str,
        pii_type: str,
        pattern: re.Pattern[str],
        pii_counts: dict[str, int],
        redaction_map: dict[str, str],
        tokens_created: list[str],
        source: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """
        Replace only the captured sensitive group in labelled values.

        Example:
            Name: John Smith
            Name: [NAME_1]
        """

        def replacement(match: re.Match[str]) -> str:
            full_match = match.group(0)
            original_value = match.group(1).strip()

            masked_value = self._mask_value(
                pii_type=pii_type,
                original_value=original_value,
                pii_counts=pii_counts,
                redaction_map=redaction_map,
                tokens_created=tokens_created,
                source=source,
                metadata=metadata,
            )

            return full_match.replace(original_value, masked_value)

        return pattern.sub(replacement, text)

    def _mask_value(
        self,
        pii_type: str,
        original_value: str,
        pii_counts: dict[str, int],
        redaction_map: dict[str, str],
        tokens_created: list[str],
        source: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """
        Mask or tokenise one PII value according to config.
        """

        pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1

        if self.config.masking_mode == "partial_mask":
            masked_value = self._partial_mask(pii_type, original_value)

        elif self.config.masking_mode == "full_mask":
            masked_value = self._full_mask(original_value)

        elif self.config.masking_mode == "tokenise":
            masked_value = self.token_store.create_token(
                pii_type=pii_type,
                original_value=original_value,
                customer_id=self.config.customer_id,
                source=source,
                metadata=metadata,
            )
            tokens_created.append(masked_value)

        else:
            raise ValueError(f"Unsupported masking mode: {self.config.masking_mode}")

        redaction_map[masked_value] = original_value

        self.audit_logger.log_redaction(
            pii_type=pii_type,
            mask=masked_value,
            source=source,
            extra_metadata={
                "masking_mode": self.config.masking_mode,
                "customer_id": self.config.customer_id,
            },
        )

        return masked_value

    def _full_mask(self, value: str) -> str:
        """
        Fully mask a sensitive value with stars.
        """

        if not value:
            return ""

        return "*" * len(value)

    def _partial_mask(self, pii_type: str, value: str) -> str:
        """
        Partially mask a sensitive value while keeping limited readability.
        """

        if pii_type == "EMAIL":
            return self._partial_mask_email(value)

        if pii_type == "PHONE":
            digits = re.sub(r"\D", "", value)

            if len(digits) <= 4:
                return "*" * len(value)

            return "*" * (len(value) - 4) + value[-4:]

        if pii_type == "AADHAAR":
            digits = re.sub(r"\D", "", value)

            if len(digits) >= 4:
                return "**** **** " + digits[-4:]

            return "*" * len(value)

        if pii_type == "PAN":
            if len(value) >= 4:
                return value[:2] + "***" + value[-2:]

            return "*" * len(value)

        if pii_type == "NAME":
            return self._partial_mask_name(value)

        if pii_type == "ADDRESS":
            return self._partial_mask_address(value)

        return self._full_mask(value)

    def _partial_mask_email(self, email: str) -> str:
        """
        Partially mask an email address.

        Example:
            rahul@gmail.com -> rah***@gmail.com
        """

        if "@" not in email:
            return self._full_mask(email)

        local_part, domain = email.split("@", 1)

        if len(local_part) <= 3:
            masked_local = local_part[:1] + "***"
        else:
            masked_local = local_part[:3] + "***"

        return f"{masked_local}@{domain}"

    def _partial_mask_name(self, name: str) -> str:
        """
        Partially mask a name.

        Example:
            John Smith -> J*** S***
        """

        parts = name.split()
        masked_parts = []

        for part in parts:
            if len(part) <= 1:
                masked_parts.append("*")
            else:
                masked_parts.append(part[0] + "***")

        return " ".join(masked_parts)

    def _partial_mask_address(self, address: str) -> str:
        """
        Partially mask an address.
        """

        address = address.strip()

        if len(address) <= 8:
            return "*" * len(address)

        return address[:4] + "***" + address[-4:]