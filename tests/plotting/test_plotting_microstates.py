from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from eeg_pipeline.plotting.features import microstates as microstates_plots

matplotlib.use("Agg", force=True)


def _comparison_config(*, segment: str | None = "active") -> dict:
    comparisons = {
        "comparison_column": "condition",
        "comparison_values": ["control", "pain"],
        "comparison_labels": ["Control", "Pain"],
    }
    if segment is not None:
        comparisons["comparison_segment"] = segment
    return {"plotting": {"comparisons": comparisons}}


def _events_df(n_rows: int = 6) -> pd.DataFrame:
    labels = ["control"] * (n_rows // 2) + ["pain"] * (n_rows - (n_rows // 2))
    return pd.DataFrame({"condition": labels, "binary_outcome": [0, 0, 0, 1, 1, 1][:n_rows]})


def _microstates_df(*, include_baseline: bool = False) -> pd.DataFrame:
    data = {
        "microstates_active_broadband_global_coverage_a": [0.10, 0.11, 0.12, 0.31, 0.32, 0.33],
        "microstates_active_broadband_global_duration_ms_a": [40.0, 42.0, 44.0, 60.0, 62.0, 64.0],
        "microstates_active_broadband_global_occurrence_hz_a": [2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
    }
    if include_baseline:
        data.update(
            {
                "microstates_baseline_broadband_global_coverage_a": [
                    0.20,
                    0.21,
                    0.22,
                    0.23,
                    0.24,
                    0.25,
                ],
                "microstates_baseline_broadband_global_duration_ms_a": [
                    35.0,
                    36.0,
                    37.0,
                    38.0,
                    39.0,
                    40.0,
                ],
                "microstates_baseline_broadband_global_occurrence_hz_a": [
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                ],
            }
        )
    return pd.DataFrame(data)


def test_microstates_plot_requires_explicit_configured_comparison(tmp_path: Path) -> None:
    features_df = _microstates_df()
    events_df = _events_df()

    with pytest.raises(ValueError, match="explicit comparison"):
        microstates_plots.plot_microstates_by_condition(
            features_df=features_df,
            events_df=events_df,
            subject="01",
            save_dir=tmp_path,
            logger=logging.getLogger("test_microstates_plot_comparison"),
            config={},
        )


def test_microstates_plot_requires_explicit_segment_when_multiple_segments(
    tmp_path: Path,
) -> None:
    features_df = _microstates_df(include_baseline=True)
    events_df = _events_df()

    with pytest.raises(ValueError, match="comparison segment"):
        microstates_plots.plot_microstates_by_condition(
            features_df=features_df,
            events_df=events_df,
            subject="01",
            save_dir=tmp_path,
            logger=logging.getLogger("test_microstates_plot_segment"),
            config=_comparison_config(segment=None),
        )


def test_microstates_plot_rejects_feature_event_length_mismatch(tmp_path: Path) -> None:
    features_df = _microstates_df()
    events_df = _events_df(n_rows=5)

    with pytest.raises(ValueError, match="row count mismatch"):
        microstates_plots.plot_microstates_by_condition(
            features_df=features_df,
            events_df=events_df,
            subject="01",
            save_dir=tmp_path,
            logger=logging.getLogger("test_microstates_plot_length"),
            config=_comparison_config(),
        )


def test_microstates_plot_skips_trial_level_stats_for_subject_fitted_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features_df = _microstates_df()
    features_df.attrs["microstate_template_source"] = "subject_fitted"
    events_df = _events_df()
    call_count = 0

    def fake_mannwhitneyu(*args, **kwargs):
        del args, kwargs
        nonlocal call_count
        call_count += 1
        return 0.0, 0.5

    monkeypatch.setattr(microstates_plots, "mannwhitneyu", fake_mannwhitneyu)
    monkeypatch.setattr(microstates_plots, "save_fig", lambda *args, **kwargs: None)

    microstates_plots.plot_microstates_by_condition(
        features_df=features_df,
        events_df=events_df,
        subject="01",
        save_dir=tmp_path,
        logger=logging.getLogger("test_microstates_plot_subject_fitted"),
        config=_comparison_config(),
    )

    assert call_count == 0


def test_microstates_plot_runs_trial_level_stats_for_fixed_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features_df = _microstates_df()
    features_df.attrs["microstate_template_source"] = "fixed"
    events_df = _events_df()
    call_count = 0

    def fake_mannwhitneyu(*args, **kwargs):
        del args, kwargs
        nonlocal call_count
        call_count += 1
        return 0.0, 0.5

    monkeypatch.setattr(microstates_plots, "mannwhitneyu", fake_mannwhitneyu)
    monkeypatch.setattr(microstates_plots, "save_fig", lambda *args, **kwargs: None)

    microstates_plots.plot_microstates_by_condition(
        features_df=features_df,
        events_df=events_df,
        subject="01",
        save_dir=tmp_path,
        logger=logging.getLogger("test_microstates_plot_fixed"),
        config=_comparison_config(),
    )

    assert call_count == 3
