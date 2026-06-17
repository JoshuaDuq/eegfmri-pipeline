from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from studies.tests.test_support import DotConfig


def _config(root: Path, *, sensitivity_outputs: list[dict[str, str]] | None = None) -> DotConfig:
    reporting = {}
    if sensitivity_outputs is not None:
        reporting["sensitivity_output_roots"] = sensitivity_outputs

    deriv_root = str(root / "derivatives")
    return DotConfig(
        {
            "deriv_root": deriv_root,
            "paths": {"deriv_root": deriv_root},
            "study1": {
                "outputs": {"root_name": "study1"},
                "targets": {"names": ["NPS", "SIIPS1"]},
                "feature_benchmark": {
                    "n_perm": 10,
                    "max_invalid_permutation_fraction": 0.20,
                },
                "reporting": reporting,
            },
        }
    )


def _study1_root(config: DotConfig) -> Path:
    return Path(config.get("paths.deriv_root")) / "group" / "multimodal" / "study1"


def _write_primary_target_table(root: Path) -> None:
    targets = pd.DataFrame(
        [
            _target_row("sub-0001", 1, 1, 45.3, 1.0, 100.0),
            _target_row("sub-0001", 1, 2, 49.3, 5.0, 500.0),
            _target_row("sub-0001", 2, 1, 45.3, 1.2, 115.0),
            _target_row("sub-0001", 2, 2, 49.3, 5.2, 515.0),
            _target_row("sub-0002", 1, 1, 45.3, 2.0, 200.0),
            _target_row("sub-0002", 1, 2, 49.3, 6.0, 600.0),
            _target_row("sub-0002", 2, 1, 45.3, 2.2, 215.0),
            _target_row("sub-0002", 2, 2, 49.3, 6.2, 615.0),
        ]
    )
    target_dir = root / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(target_dir / "primary_targets.parquet", index=False)
    targets.to_csv(target_dir / "primary_targets.tsv", sep="\t", index=False)


def _target_row(
    subject_id: str,
    block: int,
    within_block_trial: int,
    stimulus_temp: float,
    nps: float,
    siips1: float,
) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "task": "pain",
        "block": block,
        "acquisition_run": block,
        "trial_index": within_block_trial,
        "within_block_trial": within_block_trial,
        "onset": float(within_block_trial),
        "duration": 1.0,
        "NPS": nps,
        "SIIPS1": siips1,
        "NPS_fmri_n_voxels": 10,
        "NPS_fmri_scoring_mask_sha256": "nps-mask",
        "SIIPS1_fmri_n_voxels": 10,
        "SIIPS1_fmri_scoring_mask_sha256": "siips-mask",
        "hrf_weighted_framewise_displacement": 0.01,
        "hrf_weighted_std_dvars": 0.02,
        "hrf_weighted_fp1_fp2_high_frequency_power": 0.03,
        "residual_ecg_coupling": 0.04,
        "stimulus_temp": stimulus_temp,
        "selected_surface": 1,
    }


def _write_clean_events(config: DotConfig) -> None:
    deriv_root = Path(config.get("paths.deriv_root"))
    for subject_id in ("sub-0001", "sub-0002"):
        event_dir = deriv_root / "preprocessed" / "eeg" / subject_id / "eeg"
        event_dir.mkdir(parents=True, exist_ok=True)
        subject_rating_offset = 0.0 if subject_id == "sub-0001" else 20.0
        events = pd.DataFrame(
            [
                _event_row(1, 1, 45.3, 0, 110.0 + subject_rating_offset),
                _event_row(1, 2, 49.3, 1, 170.0 + subject_rating_offset),
                _event_row(2, 1, 45.3, 0, 112.0 + subject_rating_offset),
                _event_row(2, 2, 49.3, 1, 172.0 + subject_rating_offset),
            ]
        )
        events.to_csv(
            event_dir / f"{subject_id}_task-pain_proc-clean_events.tsv",
            sep="\t",
            index=False,
        )


def _event_row(
    run_id: int,
    trial_number: int,
    stimulus_temp: float,
    pain_binary: int,
    rating: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "trial_number": trial_number,
        "stimulus_temp": stimulus_temp,
        "selected_surface": 1,
        "pain_binary_coded": pain_binary,
        "vas_final_coded_rating": rating,
        "residual_ecg_coupling": 0.04,
        "fp1_fp2_high_frequency_power": 0.05,
    }


def _write_feature_summary(
    root: Path,
    target: str,
    feature_spec: str,
    *,
    partition: str = "primary",
    include_incremental_metrics: bool = True,
) -> None:
    metrics_dir = (
        root
        / "feature_benchmark"
        / partition
        / target
        / feature_spec
        / "model_comparison"
        / "metrics"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "elasticnet": _model_metrics(0.20, include_incremental_metrics=include_incremental_metrics),
        "ridge": _model_metrics(0.10, include_incremental_metrics=include_incremental_metrics),
        "subject_selection": {
            "n_requested": 2,
            "n_included": 2,
            "n_excluded": 0,
            "excluded_fraction": 0.0,
        },
    }
    (metrics_dir / "model_comparison_summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": "elasticnet",
                "fold": 0,
                "test_subject": "sub-0001",
                "best_params": "{'model__alpha': 0.1}",
            },
            {
                "model": "ridge",
                "fold": 0,
                "test_subject": "sub-0001",
                "best_params": "{'model__alpha': 1.0}",
            },
        ]
    ).to_csv(metrics_dir / "model_comparison.tsv", sep="\t", index=False)


def _model_metrics(
    mean_r2: float,
    *,
    include_incremental_metrics: bool,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "mean_r2": mean_r2,
        "std_r2": 0.01,
        "ci_low_r2": mean_r2 - 0.02,
        "ci_high_r2": mean_r2 + 0.02,
        "overall_r2": mean_r2,
        "p_value_r2": 0.1,
        "mean_mae": 0.2,
        "std_mae": 0.01,
        "ci_low_mae": 0.18,
        "ci_high_mae": 0.22,
        "n_perm": 10,
        "n_perm_requested": 10,
        "n_perm_completed": 10,
        "n_perm_attempted": 11,
        "n_invalid_permutations": 1,
        "n_folds": 2,
    }
    if include_incremental_metrics:
        metrics.update(
            {
                "mean_nuisance_r2": 0.05,
                "std_nuisance_r2": 0.01,
                "mean_delta_r2": mean_r2 - 0.05,
                "std_delta_r2": 0.01,
                "ci_low_delta_r2": mean_r2 - 0.07,
                "ci_high_delta_r2": mean_r2 - 0.03,
                "p_value_delta_r2": 0.1,
            }
        )
    return metrics


def test_report_writes_full_picture_bundle(tmp_path, monkeypatch) -> None:
    from studies.pain_study.study1 import reporting

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_primary_target_table(root)
    _write_clean_events(cfg)
    for target in ("NPS", "SIIPS1"):
        _write_feature_summary(root, target, "alpha")

    monkeypatch.setattr(reporting, "PRIMARY_BAND_PRESETS", {"alpha": ["alpha"]})

    report_path = reporting.write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    full_picture_root = report_path.parent / "full_picture"

    assert "temporal_control_interpretation" not in report.columns

    assert (full_picture_root / "full_picture_manifest.json").exists()
    assert (full_picture_root / "primary_feature_model_summary.tsv").exists()
    assert (full_picture_root / "model_leaderboard_by_mean_r2.tsv").exists()
    assert (full_picture_root / "model_leaderboard_by_delta_r2.tsv").exists()
    assert (full_picture_root / "target_by_stimulus_temp.tsv").exists()
    assert (full_picture_root / "target_by_subject_and_stimulus_temp.tsv").exists()
    assert (full_picture_root / "target_qc_metrics.tsv").exists()

    by_temp = pd.read_csv(full_picture_root / "target_by_stimulus_temp.tsv", sep="\t")
    assert by_temp["stimulus_temp"].tolist() == [45.3, 49.3]
    assert by_temp["mean_NPS"].tolist() == [1.6, 5.6]
    assert by_temp["mean_SIIPS1"].tolist() == [157.5, 557.5]

    target_qc = pd.read_csv(full_picture_root / "target_qc_metrics.tsv", sep="\t")
    assert set(target_qc["target"]) == {"NPS", "SIIPS1"}
    assert "target_interpretation" not in target_qc.columns
    assert "validity_limitations" not in target_qc.columns
    assert "expected_construct_relation" not in target_qc.columns
    assert "scope_sensitivity" not in target_qc.columns
    siips1 = target_qc.loc[target_qc["target"] == "SIIPS1"].iloc[0]
    assert siips1["siips1_rating_beyond_temperature_nps_r"] > 0.0


def test_report_compares_configured_sensitivity_roots(tmp_path, monkeypatch) -> None:
    from studies.pain_study.study1 import reporting

    cfg = _config(
        tmp_path,
        sensitivity_outputs=[{"label": "raw_target", "root_name": "study1_raw"}],
    )
    root = _study1_root(cfg)
    raw_root = Path(cfg.get("paths.deriv_root")) / "group" / "multimodal" / "study1_raw"
    _write_primary_target_table(root)
    _write_clean_events(cfg)
    for target in ("NPS", "SIIPS1"):
        _write_feature_summary(root, target, "alpha")
        _write_feature_summary(
            raw_root,
            target,
            "alpha",
            include_incremental_metrics=False,
        )

    monkeypatch.setattr(reporting, "PRIMARY_BAND_PRESETS", {"alpha": ["alpha"]})

    report_path = reporting.write_study1_report(task="pain", config=cfg)
    full_picture_root = report_path.parent / "full_picture"
    sensitivity = pd.read_csv(
        full_picture_root / "configured_sensitivity_model_summary.tsv",
        sep="\t",
    )
    pivot = pd.read_csv(full_picture_root / "primary_sensitivity_comparison.tsv", sep="\t")

    assert set(sensitivity["analysis_label"]) == {"study1", "raw_target"}
    assert set(sensitivity["target"]) == {"NPS", "SIIPS1"}
    raw_rows = sensitivity.loc[sensitivity["analysis_label"] == "raw_target"]
    assert raw_rows["mean_delta_r2"].isna().all()
    assert "mean_r2_raw_target" in pivot.columns
    assert "mean_delta_r2_study1" in pivot.columns
