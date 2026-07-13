"""
Simple HTML dashboard for saved evaluation results.

Run from project root:

    python -m evaluation.dashboard --tenant_id neuralrays
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.config import EVALUATION_RESULTS_DIR


def get_latest_result_file(tenant_id: str) -> Path:
    tenant_dir = EVALUATION_RESULTS_DIR / tenant_id

    if not tenant_dir.exists():
        raise FileNotFoundError(f"No results folder found for tenant: {tenant_id}")

    result_files = sorted(tenant_dir.glob("evaluation_run_*.json"))

    if not result_files:
        raise FileNotFoundError(f"No JSON result files found for tenant: {tenant_id}")

    return result_files[-1]


def load_result(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_dashboard_html(data: dict) -> str:
    summary = data["summary"]
    results = data["results"]

    rows = ""

    for result in results:
        metrics = result["metrics"]
        status = "PASS" if result["passed"] else "FAIL"

        rows += f"""
        <tr>
          <td>{result["question_id"]}</td>
          <td>{status}</td>
          <td>{result["question"]}</td>
          <td>{metrics.get("faithfulness", 0)}</td>
          <td>{metrics.get("correctness", 0)}</td>
          <td>{metrics.get("context_precision", 0)}</td>
          <td>{metrics.get("answer_relevancy", 0)}</td>
          <td>{metrics.get("hallucination", 0)}</td>
        </tr>
        """

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RAG Evaluation Dashboard</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 30px;
      background: #f7fbff;
      color: #102033;
    }}

    .card {{
      background: white;
      border: 1px solid #d8e9ef;
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 8px 24px rgba(16, 32, 51, 0.08);
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}

    .metric {{
      background: #f0f8ff;
      border: 1px solid #d6ebff;
      border-radius: 12px;
      padding: 14px;
    }}

    .metric strong {{
      display: block;
      color: #607086;
      font-size: 0.85rem;
    }}

    .metric span {{
      display: block;
      margin-top: 6px;
      font-size: 1.4rem;
      font-weight: 800;
      color: #1657c8;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
    }}

    th, td {{
      padding: 10px;
      border-bottom: 1px solid #d8e9ef;
      text-align: left;
      font-size: 0.9rem;
      vertical-align: top;
    }}

    th {{
      background: #eaf6ff;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>RAG Evaluation Dashboard</h1>
    <p><strong>Tenant:</strong> {data["tenant_id"]}</p>
    <p><strong>Adapter:</strong> {data["adapter_name"]}</p>
    <p><strong>Run timestamp:</strong> {data["run_timestamp"]}</p>
  </div>

  <div class="card">
    <h2>Summary Metrics</h2>
    <div class="grid">
      <div class="metric"><strong>Total Questions</strong><span>{summary["total_questions"]}</span></div>
      <div class="metric"><strong>Passed</strong><span>{summary["passed"]}</span></div>
      <div class="metric"><strong>Failed</strong><span>{summary["failed"]}</span></div>
      <div class="metric"><strong>Pass Rate</strong><span>{summary["pass_rate"] * 100:.1f}%</span></div>
      <div class="metric"><strong>Faithfulness</strong><span>{summary["average_faithfulness"]}</span></div>
      <div class="metric"><strong>Correctness</strong><span>{summary["average_correctness"]}</span></div>
      <div class="metric"><strong>Context Precision</strong><span>{summary["average_context_precision"]}</span></div>
      <div class="metric"><strong>Answer Relevancy</strong><span>{summary["average_answer_relevancy"]}</span></div>
      <div class="metric"><strong>Hallucination</strong><span>{summary["average_hallucination"]}</span></div>
      <div class="metric"><strong>Source Presence</strong><span>{summary["average_source_presence"]}</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Question-Level Results</h2>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Status</th>
          <th>Question</th>
          <th>Faithfulness</th>
          <th>Correctness</th>
          <th>Context Precision</th>
          <th>Answer Relevancy</th>
          <th>Hallucination</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def save_dashboard(json_file: Path, html_content: str) -> Path:
    html_file = json_file.with_suffix(".html")

    with html_file.open("w", encoding="utf-8") as file:
        file.write(html_content)

    return html_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation dashboard.")
    parser.add_argument("--tenant_id", default="neuralrays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    latest_json_file = get_latest_result_file(args.tenant_id)
    data = load_result(latest_json_file)

    html_content = build_dashboard_html(data)
    html_file = save_dashboard(latest_json_file, html_content)

    print(f"Dashboard saved to: {html_file}")


if __name__ == "__main__":
    main()