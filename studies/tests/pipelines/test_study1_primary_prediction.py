from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config


def test_load_primary_prediction_summary_joins_configured_primary_folds(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.primary_prediction import (
        load_primary_prediction_summary,
    )

    report_path = write_primary_prediction_report(tmp_path)

    summary = load_primary_prediction_summary(report_path, load_study1_config())

    assert summary.targets == ("NPS", "SIIPS1")
    assert len(summary.participant_performance) == 8
    assert len(summary.cohort_performance) == 2
    assert summary.cohort_performance["target"].tolist() == ["NPS", "SIIPS1"]
    assert set(summary.participant_performance["model"]) == {"elasticnet"}
    assert set(summary.participant_performance["feature_spec"]) == {"alpha_beta_gamma"}
    assert np.allclose(
        summary.participant_performance["delta_r2"],
        summary.participant_performance["r2"] - summary.participant_performance["r2_nuisance"],
    )


def test_load_primary_prediction_summary_rejects_missing_target(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.primary_prediction import (
        load_primary_prediction_summary,
    )

    report_path = write_primary_prediction_report(tmp_path)
    report = pd.read_csv(report_path, sep="\t")
    report.loc[report["target"].eq("SIIPS1"), "feature_spec"] = "gamma"
    report.to_csv(report_path, sep="\t", index=False)

    with pytest.raises(
        ValueError,
        match="primary target set does not match figure configuration",
    ):
        load_primary_prediction_summary(report_path, load_study1_config())


def test_load_primary_prediction_summary_rejects_duplicate_subject(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.primary_prediction import (
        load_primary_prediction_summary,
    )

    report_path = write_primary_prediction_report(tmp_path)
    fold_path = first_fold_path(report_path)
    folds = pd.read_csv(fold_path, sep="\t")
    folds.loc[1, "test_subject"] = folds.loc[0, "test_subject"]
    folds.to_csv(fold_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="duplicate held-out subjects"):
        load_primary_prediction_summary(report_path, load_study1_config())


def test_load_primary_prediction_summary_rejects_delta_arithmetic_mismatch(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.primary_prediction import (
        load_primary_prediction_summary,
    )

    report_path = write_primary_prediction_report(tmp_path)
    fold_path = first_fold_path(report_path)
    folds = pd.read_csv(fold_path, sep="\t")
    folds.loc[0, "delta_r2"] += 0.01
    folds.to_csv(fold_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="delta_r2 does not equal r2 - r2_nuisance"):
        load_primary_prediction_summary(report_path, load_study1_config())


def test_load_primary_prediction_summary_rejects_report_fold_mean_mismatch(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.primary_prediction import (
        load_primary_prediction_summary,
    )

    report_path = write_primary_prediction_report(tmp_path)
    report = pd.read_csv(report_path, sep="\t")
    report.loc[0, "mean_delta_r2"] += 0.01
    report.to_csv(report_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="fold mean does not match report mean"):
        load_primary_prediction_summary(report_path, load_study1_config())


def test_load_primary_prediction_summary_requires_fold_table(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.primary_prediction import (
        load_primary_prediction_summary,
    )

    report_path = write_primary_prediction_report(tmp_path)
    first_fold_path(report_path).unlink()

    with pytest.raises(FileNotFoundError, match="fold table does not exist"):
        load_primary_prediction_summary(report_path, load_study1_config())


def write_primary_prediction_report(tmp_path: Path) -> Path:
    report_rows: list[dict[str, object]] = []
    subjects = ("sub-01", "sub-02", "sub-03", "sub-04")
    for target_index, target in enumerate(("NPS", "SIIPS1")):
        metrics_root = (
            tmp_path
            / "feature_benchmark"
            / "primary"
            / target
            / "alpha_beta_gamma"
            / "model_comparison"
            / "metrics"
        )
        metrics_root.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_root / "model_comparison_summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        nuisance = np.asarray((-0.20, 0.10, -0.40, 0.05), dtype=float)
        nuisance += target_index * 0.03
        delta = np.asarray((0.04, 0.02, -0.10, 0.08), dtype=float)
        delta -= target_index * 0.01
        full = nuisance + delta
        folds = pd.DataFrame(
            {
                "model": ["elasticnet"] * 4,
                "fold": range(4),
                "test_subject": subjects,
                "r2_nuisance": nuisance,
                "r2": full,
                "delta_r2": delta,
            }
        )
        folds.to_csv(metrics_root / "model_comparison.tsv", sep="\t", index=False)
        mean_full = float(full.mean())
        mean_delta = float(delta.mean())
        report_rows.append(
            {
                "lane": "feature_benchmark",
                "analysis_partition": "primary",
                "target": target,
                "feature_spec": "alpha_beta_gamma",
                "model": "elasticnet",
                "mean_nuisance_r2": float(nuisance.mean()),
                "mean_r2": mean_full,
                "ci_low_r2": mean_full - 0.20,
                "ci_high_r2": mean_full + 0.20,
                "mean_delta_r2": mean_delta,
                "ci_low_delta_r2": mean_delta - 0.08,
                "ci_high_delta_r2": mean_delta + 0.08,
                "p_value_delta_r2": 0.2,
                "p_value_delta_r2_holm": 0.4,
                "n_folds": 4,
                "n_subjects_included": 4,
                "summary_path": str(summary_path),
            }
        )

    distractor = report_rows[0].copy()
    distractor["feature_spec"] = "gamma"
    report_rows.append(distractor)
    report_path = tmp_path / "reports" / "study1_report.tsv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, sep="\t", index=False)
    return report_path


def first_fold_path(report_path: Path) -> Path:
    report = pd.read_csv(report_path, sep="\t")
    summary_path = Path(report.iloc[0]["summary_path"])
    return summary_path.parent / "model_comparison.tsv"
