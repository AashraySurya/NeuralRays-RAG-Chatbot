"""
LLM guard for PII-safe prompt preparation.

This module shows where the PII redaction module fits in the RAG flow:

User question / retrieved context
-> PII redaction
-> external LLM API
-> optional restoration of masked values

The current project does not yet use an external LLM API, but this guard makes
the PII module ready for future OpenAI / Claude / Gemini / Azure OpenAI usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pii.redactor import PIIRedactor


@dataclass
class GuardedPayload:
    """
    Redacted inputs ready to be sent to an external LLM.
    """

    original_question: str
    original_contexts: list[str]
    redacted_question: str
    redacted_contexts: list[str]
    redaction_map: dict[str, str] = field(default_factory=dict)
    pii_counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_pii(self) -> bool:
        """
        Whether PII was found in the question or contexts.
        """

        return bool(self.redaction_map)


@dataclass
class GuardedPrompt:
    """
    Full redacted prompt plus the payload metadata.
    """

    redacted_prompt: str
    payload: GuardedPayload


class LLMPiiGuard:
    """
    Prepares PII-safe prompts before external LLM calls.
    """

    def __init__(
        self,
        audit_log_path: Path | str = Path("pii/audit_log.jsonl"),
        source: str = "llm_guard",
    ) -> None:
        self.redactor = PIIRedactor(
            audit_log_path=audit_log_path,
            source=source,
        )

    def prepare_payload(
        self,
        question: str,
        contexts: list[str],
    ) -> GuardedPayload:
        """
        Redact PII from a user question and retrieved contexts.

        The question and contexts are redacted together so mask IDs remain
        unique across the whole LLM prompt.
        """

        combined_text = self._combine_question_and_contexts(
            question=question,
            contexts=contexts,
        )

        redaction_result = self.redactor.redact_before_external_llm(combined_text)

        redacted_question = self._extract_section(
            text=redaction_result.redacted_text,
            section_name="QUESTION",
        )

        redacted_contexts = []

        for index in range(len(contexts)):
            redacted_context = self._extract_section(
                text=redaction_result.redacted_text,
                section_name=f"CONTEXT_{index}",
            )
            redacted_contexts.append(redacted_context)

        return GuardedPayload(
            original_question=question,
            original_contexts=contexts,
            redacted_question=redacted_question,
            redacted_contexts=redacted_contexts,
            redaction_map=redaction_result.redaction_map,
            pii_counts=redaction_result.pii_counts,
        )

    def build_redacted_prompt(
        self,
        question: str,
        contexts: list[str],
        system_instruction: str | None = None,
    ) -> GuardedPrompt:
        """
        Build a redacted RAG prompt ready for an external LLM.

        The returned prompt should be safe to send to an external API.
        """

        payload = self.prepare_payload(
            question=question,
            contexts=contexts,
        )

        instruction = system_instruction or (
            "Answer the user question using only the provided context. "
            "If the answer is not supported by the context, say that the "
            "information is not available in the provided documents."
        )

        context_block = "\n\n".join(
            f"[Context {index + 1}]\n{context}"
            for index, context in enumerate(payload.redacted_contexts)
        )

        redacted_prompt = (
            f"{instruction}\n\n"
            f"Question:\n{payload.redacted_question}\n\n"
            f"Context:\n{context_block}\n"
        )

        return GuardedPrompt(
            redacted_prompt=redacted_prompt,
            payload=payload,
        )

    def restore_model_output(
        self,
        model_output: str,
        redaction_map: dict[str, str],
    ) -> str:
        """
        Restore masked values in an LLM response if needed.

        In many enterprise settings, you may choose not to restore PII in the
        final answer. This method is available for controlled use cases.
        """

        return self.redactor.restore(
            redacted_text=model_output,
            redaction_map=redaction_map,
        )

    def _combine_question_and_contexts(
        self,
        question: str,
        contexts: list[str],
    ) -> str:
        """
        Combine question and contexts into labelled sections before redaction.
        """

        sections = [self._wrap_section("QUESTION", question)]

        for index, context in enumerate(contexts):
            sections.append(
                self._wrap_section(f"CONTEXT_{index}", context)
            )

        return "\n".join(sections)

    def _wrap_section(self, section_name: str, content: str) -> str:
        """
        Wrap a section in markers so it can be extracted after redaction.
        """

        return (
            f"<<<START_{section_name}>>>\n"
            f"{content}\n"
            f"<<<END_{section_name}>>>"
        )

    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Extract one labelled section from combined redacted text.
        """

        start_marker = f"<<<START_{section_name}>>>\n"
        end_marker = f"\n<<<END_{section_name}>>>"

        start_index = text.find(start_marker)

        if start_index == -1:
            return ""

        content_start = start_index + len(start_marker)
        end_index = text.find(end_marker, content_start)

        if end_index == -1:
            return ""

        return text[content_start:end_index]


def prepare_redacted_prompt(
    question: str,
    contexts: list[str],
    audit_log_path: Path | str = Path("pii/audit_log.jsonl"),
    source: str = "llm_guard",
) -> GuardedPrompt:
    """
    Convenience helper for preparing a redacted LLM prompt.
    """

    guard = LLMPiiGuard(
        audit_log_path=audit_log_path,
        source=source,
    )

    return guard.build_redacted_prompt(
        question=question,
        contexts=contexts,
    )