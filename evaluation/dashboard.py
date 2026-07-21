"""
HTML dashboard for the domain-agnostic RAG evaluation framework.

Run from project root:

    python -m evaluation.dashboard --tenant_id neuralrays
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("evaluation/results")


def find_latest_result_file(tenant_id: str) -> Path:
    """
    Find the latest evaluation JSON result for a tenant.
    """

    tenant_dir = RESULTS_ROOT / tenant_id

    if not tenant_dir.exists():
        raise FileNotFoundError(f"No results directory found for tenant: {tenant_id}")

    result_files = sorted(
        tenant_dir.glob("evaluation_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not result_files:
        raise FileNotFoundError(f"No evaluation JSON files found for tenant: {tenant_id}")

    return result_files[0]


def load_result(path: Path) -> dict[str, Any]:
    """
    Load evaluation result JSON.
    """

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_float(value: Any, decimals: int = 4) -> str:
    """
    Format numeric values safely.
    """

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def card(title: str, value: Any, note: str = "") -> str:
    """
    Render a dashboard metric card.
    """

    return f"""
    <div class="card">
      <div class="card-title">{html.escape(title)}</div>
      <div class="card-value">{html.escape(str(value))}</div>
      <div class="card-note">{html.escape(note)}</div>
    </div>
    """


def render_question_rows(question_results: list[dict[str, Any]]) -> str:
    """
    Render per-question result rows.
    """

    rows = []

    for result in question_results:
        scores = result.get("scores", {})
        operational = result.get("operational_metrics", {})

        status = "PASS" if result.get("passed") else "FAIL"
        status_class = "pass" if result.get("passed") else "fail"

        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(result.get("question_id", "")))}</td>
              <td>{html.escape(str(result.get("question", "")))}</td>
              <td class="{status_class}">{status}</td>
              <td>{format_float(scores.get("faithfulness"))}</td>
              <td>{format_float(scores.get("correctness"))}</td>
              <td>{format_float(scores.get("context_precision"))}</td>
              <td>{format_float(scores.get("answer_relevancy"))}</td>
              <td>{format_float(scores.get("hallucination"))}</td>
              <td>{format_float(operational.get("total_latency_ms"), 2)}</td>
              <td>{format_float(operational.get("estimated_total_tokens"), 0)}</td>
              <td>{format_float(result.get("overall_rag_score"))}</td>
            </tr>
            """
        )

    return "\n".join(rows)


def build_dashboard_html(result: dict[str, Any]) -> str:
    """
    Build the full dashboard HTML.
    """

    tenant_id = result.get("tenant_id", "")
    adapter = result.get("adapter", "")
    average_scores = result.get("average_scores", {})
    average_operational = result.get("average_operational_metrics", {})
    question_results = result.get("question_results", [])
    metadata = result.get("metadata", {})

    quality_cards = [
        card("Pass Rate", f"{result.get('pass_rate')}%", "Overall pass percentage"),
        card("Faithfulness", format_float(average_scores.get("faithfulness")), "Grounding in context"),
        card("Correctness", format_float(average_scores.get("correctness")), "Answer/context overlap"),
        card("Context Precision", format_float(average_scores.get("context_precision")), "Retrieval quality proxy"),
        card("Answer Relevancy", format_float(average_scores.get("answer_relevancy")), "Question/answer match"),
        card("Hallucination", format_float(average_scores.get("hallucination")), "Lower is better"),
        card("Source Presence", format_float(average_scores.get("source_presence")), "Source URL returned"),
        card("Overall RAG Score", format_float(result.get("average_overall_rag_score")), "Quality + latency + cost"),
    ]

    operational_cards = [
        card("Avg Retrieval Latency", f"{format_float(average_operational.get('retrieval_latency_ms'), 2)} ms", "Vector retrieval time"),
        card("Avg Generation Latency", f"{format_float(average_operational.get('generation_latency_ms'), 2)} ms", "Answer building time"),
        card("Avg Total Latency", f"{format_float(average_operational.get('total_latency_ms'), 2)} ms", "End-to-end answer time"),
        card("Avg Prompt Tokens", format_float(average_operational.get("estimated_prompt_tokens"), 0), "Estimated input tokens"),
        card("Avg Completion Tokens", format_float(average_operational.get("estimated_completion_tokens"), 0), "Estimated output tokens"),
        card("Avg Total Tokens", format_float(average_operational.get("estimated_total_tokens"), 0), "Prompt + completion"),
        card("Estimated Cost USD", format_float(average_operational.get("estimated_cost_usd"), 8), "Defaults to zero locally"),
        card("Latency Score", format_float(average_operational.get("latency_score")), "Higher is better"),
    ]

    question_rows = render_question_rows(question_results)

    created_at = metadata.get("created_at", "")
    cost_note = metadata.get("cost_note", "")

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RAG Evaluation Dashboard - {html.escape(str(tenant_id))}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #f5f7fb;
      margin: 0;
      padding: 24px;
      color: #1f2937;
    }}

    h1, h2 {{
      margin-bottom: 8px;
    }}

    .subtitle {{
      color: #6b7280;
      margin-bottom: 24px;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}

    .card {{
      background: #ffffff;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }}

    .card-title {{
      font-size: 13px;
      color: #6b7280;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .card-value {{
      font-size: 26px;
      font-weight: bold;
      margin-bottom: 6px;
      color: #111827;
    }}

    .card-note {{
      font-size: 12px;
      color: #6b7280;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }}

    th, td {{
      padding: 12px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}

    th {{
      background: #111827;
      color: #ffffff;
      font-weight: 600;
    }}

    .pass {{
      color: #047857;
      font-weight: bold;
    }}

    .fail {{
      color: #b91c1c;
      font-weight: bold;
    }}

    .note {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      padding: 14px;
      border-radius: 10px;
      color: #7c2d12;
      margin-bottom: 24px;
    }}
  </style>
</head>
<body>
  <h1>Domain-Agnostic RAG Evaluation Dashboard</h1>
  <div class="subtitle">
    Tenant: <strong>{html.escape(str(tenant_id))}</strong> |
    Adapter: <strong>{html.escape(str(adapter))}</strong> |
    Created: <strong>{html.escape(str(created_at))}</strong>
  </div>

  <h2>Quality Metrics</h2>
  <div class="grid">
    {''.join(quality_cards)}
  </div>

  <h2>Operational Metrics</h2>
  <div class="grid">
    {''.join(operational_cards)}
  </div>

  <div class="note">
    {html.escape(str(cost_note))}
  </div>

  <h2>Per-Question Results</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Question</th>
        <th>Status</th>
        <th>Faithfulness</th>
        <th>Correctness</th>
        <th>Context Precision</th>
        <th>Answer Relevancy</th>
        <th>Hallucination</th>
        <th>Total Latency ms</th>
        <th>Total Tokens</th>
        <th>Overall RAG Score</th>
      </tr>
    </thead>
    <tbody>
      {question_rows}
    </tbody>
  </table>
</body>
</html>
"""


def save_dashboard(
    result_path: Path,
    result: dict[str, Any],
) -> Path:
    """
    Save dashboard HTML beside the JSON result.
    """

    output_path = result_path.with_suffix(".html")

    dashboard_html = build_dashboard_html(result)

    output_path.write_text(
        dashboard_html,
        encoding="utf-8",
    )

    return output_path


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard for latest evaluation result."
    )

    parser.add_argument(
        "--tenant_id",
        required=True,
        help="Tenant ID, for example neuralrays.",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint.
    """

    args = parse_args()

    result_path = find_latest_result_file(args.tenant_id)
    result = load_result(result_path)

    dashboard_path = save_dashboard(
        result_path=result_path,
        result=result,
    )

    print("\nEvaluation dashboard generated")
    print("-------------------------------------------------------")
    print(f"Input JSON:  {result_path}")
    print(f"Output HTML: {dashboard_path}")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()