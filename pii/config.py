"""
Configuration for the PII protection pipeline.

This config controls how sensitive information is handled before it is stored
in the vector database or sent to an external LLM.

Supported masking modes:
- partial_mask: keeps a small amount of readable information, e.g. rah***@gmail.com
- full_mask: replaces the value with stars, e.g. **************
- tokenise: replaces the value with a placeholder, e.g. <email_cust_001>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MaskingMode = Literal["partial_mask", "full_mask", "tokenise"]


@dataclass
class PIIPipelineConfig:
    """
    Configuration for PII handling across the RAG pipeline.
    """

    masking_mode: MaskingMode = "partial_mask"
    customer_id: str = "cust"
    audit_log_path: Path = Path("pii/audit_log.jsonl")
    token_store_path: Path = Path("pii/pii_tokens.sqlite3")

    # External LLM safety control.
    allow_raw_pii_to_external_llm: bool = False

    # Ingestion controls.
    redact_at_ingestion: bool = True
    store_original_in_vector_db: bool = False

    # Output controls.
    validate_outputs: bool = True
    allow_pii_in_final_output: bool = False

    def validate(self) -> None:
        """
        Validate the config values.
        """

        allowed_modes = {"partial_mask", "full_mask", "tokenise"}

        if self.masking_mode not in allowed_modes:
            raise ValueError(
                f"Invalid masking_mode: {self.masking_mode}. "
                f"Expected one of: {sorted(allowed_modes)}"
            )

        if not self.customer_id.strip():
            raise ValueError("customer_id cannot be empty")


def default_pii_config() -> PIIPipelineConfig:
    """
    Return the default PII pipeline config.
    """

    return PIIPipelineConfig()