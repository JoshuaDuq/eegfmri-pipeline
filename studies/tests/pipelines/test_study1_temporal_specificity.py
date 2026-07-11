from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.temporal_controls import (
    resolve_temporal_control_windows,
)

EXPECTED_WINDOWS = (
    "prestimulus_wide",
    "immediate_prestimulus",
    "ramp_up",
    "early_plateau",
    "mid_plateau",
    "late_plateau",
)


def test_load_temporal_specificity_summary_joins_current_protocol_folds(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.temporal_specificity import (
        load_temporal_specificity_summary,
    )

    config = load_study1_config()
    report_path = _write_temporal_report(tmp_path, config)

    summary = load_temporal_specificity_summary(report_path, config)

    assert summary.targets == ("NPS", "SIIPS1")
    assert summary.windows == EXPECTED_WINDOWS
    assert len(summary.cohort_effects) == 12
    assert len(summary.participant_effects) == 36
    assert summary.participant_effects.groupby(["target", "window_name"]).size().eq(3).all()
    assert summary.cohort_effects["window_order"].tolist() == list(range(6)) * 2
    assert set(summary.participant_effects["model"]) == {"elasticnet"}


def test_load_temporal_specificity_summary_rejects_legacy_window_set(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.temporal_specificity import (
        load_temporal_specificity_summary,
    )

    config = load_study1_config()
    report_path = _write_temporal_report(tmp_path, config)
    report = pd.read_csv(report_path, sep="\t")
    replacements = {
        "early_plateau": "early_shifted_active",
        "mid_plateau": "late_shifted_active",
        "late_plateau": "late_ramp_down",
    }
    report["temporal_control_window"] = report["temporal_control_window"].replace(replacements)
    report.to_csv(report_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="window set does not match current configuration"):
        load_temporal_specificity_summary(report_path, config)


def test_load_temporal_specificity_summary_rejects_report_fold_mean_mismatch(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.temporal_specificity import (
        load_temporal_specificity_summary,
    )

    config = load_study1_config()
    report_path = _write_temporal_report(tmp_path, config)
    report = pd.read_csv(report_path, sep="\t")
    report.loc[0, "mean_delta_r2"] += 0.01
    report.to_csv(report_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="fold mean does not match report mean"):
        load_temporal_specificity_summary(report_path, config)


def test_load_temporal_specificity_summary_rejects_duplicate_held_out_subject(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.temporal_specificity import (
        load_temporal_specificity_summary,
    )

    config = load_study1_config()
    report_path = _write_temporal_report(tmp_path, config)
    report = pd.read_csv(report_path, sep="\t")
    fold_path = Path(report.loc[0, "summary_path"]).parent / "model_comparison.tsv"
    folds = pd.read_csv(fold_path, sep="\t")
    folds.loc[1, "test_subject"] = folds.loc[0, "test_subject"]
    folds.to_csv(fold_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="duplicate held-out subjects"):
        load_temporal_specificity_summary(report_path, config)


def test_load_temporal_specificity_summary_requires_fold_table(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.temporal_specificity import (
        load_temporal_specificity_summary,
    )

    config = load_study1_config()
    report_path = _write_temporal_report(tmp_path, config)
    report = pd.read_csv(report_path, sep="\t")
    fold_path = Path(report.loc[0, "summary_path"]).parent / "model_comparison.tsv"
    fold_path.unlink()

    with pytest.raises(FileNotFoundError, match="fold table does not exist"):
        load_temporal_specificity_summary(report_path, config)


def test_load_temporal_specificity_summary_excludes_primary_partition(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.temporal_specificity import (
        load_temporal_specificity_summary,
    )

    config = load_study1_config()
    report_path = _write_temporal_report(tmp_path, config)
    report = pd.read_csv(report_path, sep="\t")
    primary = report.iloc[[0]].copy()
    primary["analysis_partition"] = "primary"
    primary["feature_spec"] = "alpha_beta_gamma"
    primary["temporal_control_window"] = pd.NA
    primary["temporal_control_kind"] = pd.NA
    pd.concat([report, primary], ignore_index=True).to_csv(report_path, sep="\t", index=False)

    summary = load_temporal_specificity_summary(report_path, config)

    assert len(summary.cohort_effects) == 12
    assert summary.cohort_effects["feature_spec"].str.startswith("temporal_").all()


def _write_temporal_report(tmp_path: Path, config) -> Path:
    report_rows: list[dict[str, object]] = []
    windows = resolve_temporal_control_windows(config)
    for target_index, target in enumerate(("NPS", "SIIPS1")):
        for window_index, window in enumerate(windows):
            metrics_root = (
                tmp_path
                / "feature_benchmark"
                / "temporal_control"
                / target
                / window.feature_spec
                / "model_comparison"
                / "metrics"
            )
            metrics_root.mkdir(parents=True, exist_ok=True)
            summary_path = metrics_root / "model_comparison_summary.json"
            summary_path.write_text("{}\n", encoding="utf-8")
            delta_r2 = (
                np.asarray(
                    [-0.03, 0.01, 0.05],
                    dtype=float,
                )
                + target_index * 0.02
                + window_index * 0.01
            )
            fold_rows = pd.DataFrame(
                {
                    "model": ["elasticnet"] * 3,
                    "fold": [0, 1, 2],
                    "test_subject": ["sub-01", "sub-02", "sub-03"],
                    "delta_r2": delta_r2,
                }
            )
            fold_rows.to_csv(metrics_root / "model_comparison.tsv", sep="\t", index=False)
            mean = float(delta_r2.mean())
            report_rows.append(
                {
                    "lane": "feature_benchmark",
                    "analysis_partition": "temporal_control",
                    "target": target,
                    "feature_spec": window.feature_spec,
                    "temporal_control_window": window.name,
                    "temporal_control_kind": window.kind,
                    "model": "elasticnet",
                    "mean_delta_r2": mean,
                    "ci_low_delta_r2": mean - 0.04,
                    "ci_high_delta_r2": mean + 0.04,
                    "p_value_delta_r2_holm": 0.5,
                    "n_folds": 3,
                    "n_subjects_included": 3,
                    "summary_path": str(summary_path),
                }
            )
    report_path = tmp_path / "reports" / "study1_report.tsv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, sep="\t", index=False)
    return report_path
