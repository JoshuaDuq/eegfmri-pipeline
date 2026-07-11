from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config

FEATURE_SPECS = (
    "alpha",
    "beta",
    "gamma",
    "alpha_beta",
    "alpha_beta_gamma",
)


def test_load_spectral_specificity_summary_joins_configured_folds(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.spectral_specificity import (
        load_spectral_specificity_summary,
    )

    report_path = write_spectral_report(tmp_path)

    summary = load_spectral_specificity_summary(report_path, load_study1_config())

    assert summary.targets == ("NPS", "SIIPS1")
    assert summary.feature_specs == FEATURE_SPECS
    assert len(summary.participant_effects) == 30
    assert len(summary.cohort_effects) == 10
    assert summary.cohort_effects["feature_order"].tolist() == list(range(5)) * 2
    assert summary.participant_effects.groupby(["target", "feature_spec"]).size().eq(3).all()
    assert np.allclose(
        summary.participant_effects["delta_r2"],
        summary.participant_effects["r2"] - summary.participant_effects["r2_nuisance"],
    )


def test_load_spectral_specificity_summary_rejects_missing_feature(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.spectral_specificity import (
        load_spectral_specificity_summary,
    )

    report_path = write_spectral_report(tmp_path)
    report = pd.read_csv(report_path, sep="\t")
    report.loc[
        report["target"].eq("SIIPS1") & report["feature_spec"].eq("gamma"),
        "feature_spec",
    ] = "theta"
    report.to_csv(report_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="feature set does not match figure configuration"):
        load_spectral_specificity_summary(report_path, load_study1_config())


def test_load_spectral_specificity_summary_rejects_duplicate_subject(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.spectral_specificity import (
        load_spectral_specificity_summary,
    )

    report_path = write_spectral_report(tmp_path)
    fold_path = first_fold_path(report_path)
    folds = pd.read_csv(fold_path, sep="\t")
    folds.loc[1, "test_subject"] = folds.loc[0, "test_subject"]
    folds.to_csv(fold_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="duplicate held-out subjects"):
        load_spectral_specificity_summary(report_path, load_study1_config())


def test_load_spectral_specificity_summary_rejects_delta_arithmetic(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.spectral_specificity import (
        load_spectral_specificity_summary,
    )

    report_path = write_spectral_report(tmp_path)
    fold_path = first_fold_path(report_path)
    folds = pd.read_csv(fold_path, sep="\t")
    folds.loc[0, "delta_r2"] += 0.01
    folds.to_csv(fold_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="delta_r2 does not equal r2 - r2_nuisance"):
        load_spectral_specificity_summary(report_path, load_study1_config())


def test_load_spectral_specificity_summary_rejects_report_fold_mean(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.spectral_specificity import (
        load_spectral_specificity_summary,
    )

    report_path = write_spectral_report(tmp_path)
    report = pd.read_csv(report_path, sep="\t")
    report.loc[0, "mean_delta_r2"] += 0.01
    report.to_csv(report_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="fold mean does not match report mean"):
        load_spectral_specificity_summary(report_path, load_study1_config())


def test_load_spectral_specificity_summary_requires_fold_table(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.spectral_specificity import (
        load_spectral_specificity_summary,
    )

    report_path = write_spectral_report(tmp_path)
    first_fold_path(report_path).unlink()

    with pytest.raises(FileNotFoundError, match="fold table does not exist"):
        load_spectral_specificity_summary(report_path, load_study1_config())


def write_spectral_report(tmp_path: Path) -> Path:
    report_rows: list[dict[str, object]] = []
    subjects = ("sub-01", "sub-02", "sub-03")
    for target_index, target in enumerate(("NPS", "SIIPS1")):
        for feature_order, feature_spec in enumerate(FEATURE_SPECS):
            metrics_root = (
                tmp_path
                / "feature_benchmark"
                / "primary"
                / target
                / feature_spec
                / "model_comparison"
                / "metrics"
            )
            metrics_root.mkdir(parents=True, exist_ok=True)
            summary_path = metrics_root / "model_comparison_summary.json"
            summary_path.write_text("{}\n", encoding="utf-8")
            nuisance = np.asarray((-0.20, -0.05, 0.08), dtype=float)
            delta = np.asarray((-0.04, 0.01, 0.06), dtype=float)
            delta += feature_order * 0.01 - target_index * 0.015
            full = nuisance + delta
            pd.DataFrame(
                {
                    "model": ["elasticnet"] * 3,
                    "fold": range(3),
                    "test_subject": subjects,
                    "r2_nuisance": nuisance,
                    "r2": full,
                    "delta_r2": delta,
                }
            ).to_csv(metrics_root / "model_comparison.tsv", sep="\t", index=False)
            mean = float(delta.mean())
            report_rows.append(
                {
                    "lane": "feature_benchmark",
                    "analysis_partition": "primary",
                    "target": target,
                    "feature_spec": feature_spec,
                    "model": "elasticnet",
                    "mean_delta_r2": mean,
                    "ci_low_delta_r2": mean - 0.05,
                    "ci_high_delta_r2": mean + 0.05,
                    "p_value_delta_r2": 0.2,
                    "p_value_delta_r2_holm": 0.5,
                    "n_folds": 3,
                    "n_subjects_included": 3,
                    "summary_path": str(summary_path),
                }
            )
    distractor = report_rows[0].copy()
    distractor["model"] = "ridge"
    report_rows.append(distractor)
    report_path = tmp_path / "reports" / "study1_report.tsv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, sep="\t", index=False)
    return report_path


def first_fold_path(report_path: Path) -> Path:
    report = pd.read_csv(report_path, sep="\t")
    return Path(report.iloc[0]["summary_path"]).parent / "model_comparison.tsv"
