"""
Audit logging for PII redaction.

The audit log records that redaction happened without storing the original
sensitive value. This helps provide a compliance-friendly audit trail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_LOG_PATH = Path("pii/audit_log.jsonl")


class PIIAuditLogger:
    """
    Writes PII redaction events to a JSONL audit log.

    Important:
    The original PII value is not written to the audit log.
    """

    def __init__(self, audit_log_path: Path | str = DEFAULT_AUDIT_LOG_PATH) -> None:
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_redaction(
        self,
        pii_type: str,
        mask: str,
        source: str = "unknown",
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Write one redaction event to the audit log.
        """

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "PII_REDACTION",
            "pii_type": pii_type,
            "mask": mask,
            "source": source,
        }

        if extra_metadata:
            event["metadata"] = extra_metadata

        with self.audit_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")