"""
Storage utilities for evaluation results.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.config import EVALUATION_RESULTS_DIR
from evaluation.models import EvaluationRun


def save_evaluation_run(evaluation_run: EvaluationRun) -> Path:
    """
    Save an evaluation run to JSON.
    """

    tenant_results_dir = EVALUATION_RESULTS_DIR / evaluation_run.tenant_id
    tenant_results_dir.mkdir(parents=True, exist_ok=True)

    output_file = tenant_results_dir / f"evaluation_run_{evaluation_run.run_timestamp}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(asdict(evaluation_run), file, indent=2, ensure_ascii=False)

    return output_file


def load_evaluation_run(file_path: Path) -> dict[str, Any]:
    """
    Load one saved evaluation run.
    """

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_latest_evaluation_file(tenant_id: str) -> Path:
    """
    Return the latest evaluation result file for a tenant.
    """

    tenant_results_dir = EVALUATION_RESULTS_DIR / tenant_id

    if not tenant_results_dir.exists():
        raise FileNotFoundError(f"No evaluation results found for tenant: {tenant_id}")

    result_files = sorted(tenant_results_dir.glob("evaluation_run_*.json"))

    if not result_files:
        raise FileNotFoundError(f"No evaluation result files found for tenant: {tenant_id}")

    return result_files[-1]