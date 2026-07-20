"""
Output validator for PII leakage.

This module checks the final model/chatbot output before it is returned to a user.

The purpose is to catch accidental raw PII leakage in the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class OutputValidationResult:
    """
    Result of validating a final output.
    """

    output_text: str
    is_safe: bool
    detected_pii: dict[str, list[str]] = field(default_factory=dict)

    @property
    def violation_count(self) -> int:
        """
        Count the number of detected PII values.
        """

        return sum(len(values) for values in self.detected_pii.values())


class PIIOutputValidator:
    """
    Validates whether final output contains raw PII.
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

    def validate(
        self,
        output_text: str,
        allow_pii: bool = False,
    ) -> OutputValidationResult:
        """
        Validate the output text.

        If allow_pii=True, detected PII is reported but the output is considered safe.
        """

        detected_pii = self.detect_pii(output_text)

        is_safe = allow_pii or not detected_pii

        return OutputValidationResult(
            output_text=output_text,
            is_safe=is_safe,
            detected_pii=detected_pii,
        )

    def detect_pii(self, text: str) -> dict[str, list[str]]:
        """
        Detect raw PII values in output text.
        """

        detected: dict[str, list[str]] = {}

        self._add_matches(
            detected=detected,
            pii_type="EMAIL",
            matches=self.EMAIL_PATTERN.findall(text),
        )

        self._add_matches(
            detected=detected,
            pii_type="AADHAAR",
            matches=self.AADHAAR_PATTERN.findall(text),
        )

        self._add_matches(
            detected=detected,
            pii_type="PAN",
            matches=self.PAN_PATTERN.findall(text),
        )

        self._add_matches(
            detected=detected,
            pii_type="PHONE",
            matches=self.PHONE_PATTERN.findall(text),
        )

        return detected

    def assert_safe(
        self,
        output_text: str,
        allow_pii: bool = False,
    ) -> None:
        """
        Raise an error if unsafe PII is detected.
        """

        result = self.validate(
            output_text=output_text,
            allow_pii=allow_pii,
        )

        if not result.is_safe:
            raise ValueError(
                f"Output contains raw PII and is not safe to return: {result.detected_pii}"
            )

    def _add_matches(
        self,
        detected: dict[str, list[str]],
        pii_type: str,
        matches: list[str],
    ) -> None:
        """
        Add matches to the detected PII dictionary.
        """

        clean_matches = []

        for match in matches:
            if isinstance(match, tuple):
                match = "".join(match)

            match = str(match).strip()

            if match:
                clean_matches.append(match)

        if clean_matches:
            detected[pii_type] = clean_matches