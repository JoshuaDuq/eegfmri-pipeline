#!/usr/bin/env python3
"""Developer refactor helper for ML output path reorganization.

This script is intentionally not part of runtime code paths. It rewrites
`eeg_pipeline/analysis/machine_learning/orchestration.py` in-place based on
string/regex substitutions and is meant for one-off development use.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CREATION_BLOCK_OLD = """    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)"""

CREATION_BLOCK_NEW = """    plots_dir = results_dir / "plots"
    data_dir = results_dir / "data"
    metrics_dir = results_dir / "metrics"
    models_dir = results_dir / "models"
    null_dir = results_dir / "null"
    reports_dir = results_dir / "reports"
    importance_dir = results_dir / "importance"

    for d in [results_dir, plots_dir, data_dir, metrics_dir, models_dir, null_dir, reports_dir, importance_dir]:
        ensure_dir(d)"""


REPLACEMENTS = {
    r'results_dir \/ "included_subjects\.tsv"': 'reports_dir / "included_subjects.tsv"',
    r'results_dir \/ "excluded_subjects\.tsv"': 'reports_dir / "excluded_subjects.tsv"',
    r'results_dir \/ "reproducibility_info\.json"': 'reports_dir / "reproducibility_info.json"',
    r'results_dir \/ f"best_params_\{model_name\}\.jsonl"': 'models_dir / f"best_params_{model_name}.jsonl"',
    r'results_dir \/ f"best_params_\{model_type\}\.tsv"': 'models_dir / f"best_params_{model_type}.tsv"',
    r'results_dir \/ "loso_predictions\.tsv"': 'data_dir / "loso_predictions.tsv"',
    r'results_dir \/ "cv_predictions\.tsv"': 'data_dir / "cv_predictions.tsv"',
    r'results_dir \/ "loso_indices\.tsv"': 'data_dir / "loso_indices.tsv"',
    r'results_dir \/ "cv_indices\.tsv"': 'data_dir / "cv_indices.tsv"',
    r'results_dir \/ "baseline_predictions\.tsv"': 'data_dir / "baseline_predictions.tsv"',
    r'results_dir \/ "prediction_intervals\.tsv"': 'data_dir / "prediction_intervals.tsv"',
    r'results_dir \/ "per_subject_correlations\.tsv"': 'metrics_dir / "per_subject_correlations.tsv"',
    r'results_dir \/ "per_subject_errors\.tsv"': 'metrics_dir / "per_subject_errors.tsv"',
    r'results_dir \/ "pooled_metrics\.json"': 'metrics_dir / "pooled_metrics.json"',
    r'results_dir \/ "per_subject_metrics\.tsv"': 'metrics_dir / "per_subject_metrics.tsv"',
    r'results_dir \/ "calibration_data\.json"': 'metrics_dir / "calibration_data.json"',
    r'results_dir \/ "model_comparison\.tsv"': 'metrics_dir / "model_comparison.tsv"',
    r'results_dir \/ "model_comparison_summary\.json"': 'metrics_dir / "model_comparison_summary.json"',
    r'results_dir \/ "incremental_validity\.tsv"': 'metrics_dir / "incremental_validity.tsv"',
    r'results_dir \/ "incremental_validity_summary\.json"': 'metrics_dir / "incremental_validity_summary.json"',
    r'results_dir \/ "per_subject_uncertainty\.tsv"': 'metrics_dir / "per_subject_uncertainty.tsv"',
    r'results_dir \/ "uncertainty_metrics\.json"': 'metrics_dir / "uncertainty_metrics.json"',
    r'results_dir \/ f"loso_null_\{model_name\}\.npz"': 'null_dir / f"loso_null_{model_name}.npz"',
    r'results_dir \/ f"loso_null_\{model_type\}\.npz"': 'null_dir / f"loso_null_{model_type}.npz"',
    r'results_dir \/ f"cv_null_\{model_name\}\.npz"': 'null_dir / f"cv_null_{model_name}.npz"',
    r'results_dir \/ f"cv_null_\{model_type\}\.npz"': 'null_dir / f"cv_null_{model_type}.npz"',
    r'results_dir \/ f"classification_null_\{model_type\}\.npz"': 'null_dir / f"classification_null_{model_type}.npz"',
    r'results_dir \/ "within_subject_null\.npz"': 'null_dir / "within_subject_null.npz"',
    r'results_dir \/ "permutation_importance\.tsv"': 'importance_dir / "permutation_importance.tsv"',
    r'results_dir \/ "permutation_importance_by_group_band\.tsv"': 'importance_dir / "permutation_importance_by_group_band.tsv"',
    r'results_dir \/ "permutation_importance_by_group_band_roi\.tsv"': 'importance_dir / "permutation_importance_by_group_band_roi.tsv"',
    r'results_dir \/ "shap_importance\.tsv"': 'importance_dir / "shap_importance.tsv"',
    r'results_dir \/ "shap_importance_by_group_band\.tsv"': 'importance_dir / "shap_importance_by_group_band.tsv"',
    r'results_dir \/ "shap_importance_by_group_band_roi\.tsv"': 'importance_dir / "shap_importance_by_group_band_roi.tsv"',
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-off ML orchestration rewriter")
    parser.add_argument(
        "--target",
        type=Path,
        default=repo_root() / "eeg_pipeline" / "analysis" / "machine_learning" / "orchestration.py",
        help="Path to orchestration.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print change summary without writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target.resolve()
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    content = target.read_text()
    updated = content.replace(CREATION_BLOCK_OLD, CREATION_BLOCK_NEW)
    for old, new in REPLACEMENTS.items():
        updated = re.sub(old, new, updated)

    if updated == content:
        print(f"No changes needed: {target}")
        return 0

    if args.dry_run:
        print(f"Would update: {target}")
        return 0

    target.write_text(updated)
    print(f"Updated: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
