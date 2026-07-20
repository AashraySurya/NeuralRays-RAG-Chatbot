"""
SQLite token store for reversible PII tokenisation.

When masking_mode='tokenise', sensitive values are replaced with placeholders
such as:

    <email_cust_001>

The original value is stored locally in SQLite so it can be remapped later
if the user is authorised.

This is deliberately separate from the audit log.
The audit log should never store original PII values.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TokenRecord:
    """
    One stored PII token mapping.
    """

    token: str
    pii_type: str
    original_value: str
    customer_id: str
    source: str
    created_at: str
    metadata: dict[str, Any]


class PIITokenStore:
    """
    Stores token-to-original-value mappings in SQLite.
    """

    def __init__(self, db_path: Path | str = Path("pii/pii_tokens.sqlite3")) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_database()

    def _connect(self) -> sqlite3.Connection:
        """
        Open a SQLite connection.
        """

        return sqlite3.connect(self.db_path)

    def _initialise_database(self) -> None:
        """
        Create the token mapping table if it does not already exist.
        """

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pii_tokens (
                    token TEXT PRIMARY KEY,
                    pii_type TEXT NOT NULL,
                    original_value TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def create_token(
        self,
        pii_type: str,
        original_value: str,
        customer_id: str,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create and store a new placeholder token.
        """

        normalised_pii_type = self._normalise_token_part(pii_type)
        normalised_customer_id = self._normalise_token_part(customer_id)

        next_number = self._next_token_number(
            pii_type=normalised_pii_type,
            customer_id=normalised_customer_id,
        )

        token = f"<{normalised_pii_type}_{normalised_customer_id}_{next_number:03d}>"

        created_at = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pii_tokens (
                    token,
                    pii_type,
                    original_value,
                    customer_id,
                    source,
                    created_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    normalised_pii_type,
                    original_value,
                    normalised_customer_id,
                    source,
                    created_at,
                    metadata_json,
                ),
            )

            connection.commit()

        return token

    def get_original_value(self, token: str) -> str | None:
        """
        Return the original value for a token.
        """

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT original_value
                FROM pii_tokens
                WHERE token = ?
                """,
                (token,),
            ).fetchone()

        if row is None:
            return None

        return str(row[0])

    def get_record(self, token: str) -> TokenRecord | None:
        """
        Return the full token record.
        """

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    token,
                    pii_type,
                    original_value,
                    customer_id,
                    source,
                    created_at,
                    metadata_json
                FROM pii_tokens
                WHERE token = ?
                """,
                (token,),
            ).fetchone()

        if row is None:
            return None

        metadata = json.loads(row[6])

        return TokenRecord(
            token=str(row[0]),
            pii_type=str(row[1]),
            original_value=str(row[2]),
            customer_id=str(row[3]),
            source=str(row[4]),
            created_at=str(row[5]),
            metadata=metadata,
        )

    def remap_text(self, text: str) -> str:
        """
        Replace any stored tokens in text with their original values.
        """

        tokens = re.findall(r"<[a-z0-9_]+_[a-z0-9_]+_\d{3}>", text)

        remapped_text = text

        for token in tokens:
            original_value = self.get_original_value(token)

            if original_value is not None:
                remapped_text = remapped_text.replace(token, original_value)

        return remapped_text

    def count_tokens(self) -> int:
        """
        Count stored tokens.
        """

        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM pii_tokens").fetchone()

        return int(row[0])

    def _next_token_number(self, pii_type: str, customer_id: str) -> int:
        """
        Return the next token number for a PII type and customer.
        """

        pattern_prefix = f"<{pii_type}_{customer_id}_"

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM pii_tokens
                WHERE token LIKE ?
                """,
                (f"{pattern_prefix}%",),
            ).fetchone()

        return int(row[0]) + 1

    def _normalise_token_part(self, value: str) -> str:
        """
        Make a value safe for use inside a token.
        """

        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        value = value.strip("_")

        if not value:
            return "unknown"

        return value