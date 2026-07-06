from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.tests.pipelines.test_study2_source_stage import _source_stage_frame


def test_prepare_source_stage_association_inputs_preserves_trial_row_indices() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import (
        prepare_source_stage_association_inputs,
    )

    frame = _source_stage_frame()
    inputs = prepare_source_stage_association_inputs(
        frame.iloc[::-1].reset_index(drop=True),
        config=load_study2_config(),
    )

    assert inputs.qc.source_stage_criteria_met is True
    assert inputs.retained_row_indices.tolist() == list(range(len(frame)))
    assert inputs.score.shape == (len(frame),)
    assert inputs.design.shape[0] == len(frame)
    assert len(inputs.design_columns) == inputs.design.shape[1]
    assert "onset" in inputs.design_columns


def test_compute_subject_source_association_map_uses_combined_score_design() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_subject_source_association_map,
    )

    frame = _source_stage_frame()
    source_power = _source_power_from_column(frame["eta_combined_z"].to_numpy(dtype=float))

    result = compute_subject_source_association_map(
        frame,
        source_power,
        band="alpha",
        config=load_study2_config(),
    )

    assert result.subject_id == "sub-0001"
    assert result.source_band == "alpha"
    assert result.qc.band == "combined"
    assert result.qc.source_stage_criteria_met is True
    assert result.association is not None
    assert result.association.partial_r.shape == (3,)
    assert result.association.valid_vertices.tolist() == [True, True, False]
    assert result.association.partial_r[0] > 0.95
    assert result.association.partial_r[1] < -0.95


def test_compute_band_unique_subject_source_association_map_adjusts_other_bands() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_band_unique_subject_source_association_map,
    )

    frame = _source_stage_frame()
    alpha_score = frame["eta_alpha_z"].to_numpy(dtype=float)
    source_power = _source_power_from_column(alpha_score)

    result = compute_band_unique_subject_source_association_map(
        frame,
        source_power,
        band="alpha",
        config=load_study2_config(),
    )

    assert result.source_band == "alpha"
    assert result.qc.band == "alpha"
    assert result.qc.source_stage_criteria_met is True
    assert result.association is not None
    assert result.association.partial_r[0] > 0.95
    assert result.association.partial_r[1] < -0.95


def test_compute_subject_source_association_map_returns_qc_without_map_when_criteria_unmet() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_subject_source_association_map,
    )

    frame = _source_stage_frame(n_runs=2)
    source_power = np.ones((len(frame), 3), dtype=float)

    result = compute_subject_source_association_map(
        frame,
        source_power,
        band="alpha",
        config=load_study2_config(),
    )

    assert result.qc.source_stage_criteria_met is False
    assert result.association is None
    assert result.qc.unmet_criteria == (
        "min_valid_runs_per_subject",
        "min_retained_trials_per_subject",
    )


def test_compute_subject_source_association_map_rejects_misaligned_source_power() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_subject_source_association_map,
    )

    frame = _source_stage_frame()
    source_power = np.ones((len(frame) - 1, 3), dtype=float)

    with pytest.raises(ValueError, match="same number of rows"):
        compute_subject_source_association_map(
            frame,
            source_power,
            band="alpha",
            config=load_study2_config(),
        )


def test_compute_cohort_source_association_maps_stacks_source_stage_subject_maps() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_cohort_source_association_maps,
    )

    frame = _cohort_source_stage_frame()
    source_power_by_subject = _source_power_by_subject(
        frame,
        column="eta_combined_z",
    )

    result = compute_cohort_source_association_maps(
        frame,
        source_power_by_subject,
        band="alpha",
        config=load_study2_config(),
    )

    assert result.source_band == "alpha"
    assert result.subject_ids == ("sub-0001", "sub-0002")
    assert result.fisher_z_maps.shape == (2, 3)
    assert result.partial_r_maps.shape == (2, 3)
    assert result.qc["subject_id"].tolist() == ["sub-0001", "sub-0002", "sub-0003"]
    assert result.qc["source_stage_criteria_met"].tolist() == [True, True, False]
    assert result.partial_r_maps[:, 0].min() > 0.95
    assert result.partial_r_maps[:, 1].max() < -0.95


def test_compute_cohort_source_association_maps_aligns_source_power_by_trial_id() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_cohort_source_association_maps,
    )

    full_frame = _source_stage_frame().assign(
        subject_id="sub-0001",
        trial_id=np.arange(1, 67),
    )
    retained_frame = (
        full_frame.loc[~full_frame["trial_id"].isin([4, 8, 12])]
        .copy()
        .reset_index(drop=True)
    )
    source_power = np.zeros((len(full_frame), 3), dtype=float)
    retained_rows = retained_frame["trial_id"].to_numpy(dtype=int) - 1
    source_power[retained_rows, :] = _source_power_from_column(
        retained_frame["eta_combined_z"].to_numpy(dtype=float)
    )

    result = compute_cohort_source_association_maps(
        retained_frame,
        {"sub-0001": source_power},
        band="alpha",
        config=load_study2_config(),
    )

    assert result.subject_ids == ("sub-0001",)
    assert result.partial_r_maps.shape == (1, 3)
    assert result.partial_r_maps[0, 0] > 0.95
    assert result.partial_r_maps[0, 1] < -0.95


def test_compute_cohort_source_association_maps_requires_trial_id() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_cohort_source_association_maps,
    )

    frame = _cohort_source_stage_frame(include_subject_with_unmet_criteria=False)
    source_power_by_subject = _source_power_by_subject(frame, column="eta_combined_z")

    with pytest.raises(ValueError, match="missing required trial column"):
        compute_cohort_source_association_maps(
            frame.drop(columns=["trial_id"]),
            source_power_by_subject,
            band="alpha",
            config=load_study2_config(),
        )


def test_compute_band_unique_cohort_source_association_maps_uses_band_scores() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_band_unique_cohort_source_association_maps,
    )

    frame = _cohort_source_stage_frame(include_subject_with_unmet_criteria=False)
    source_power_by_subject = _source_power_by_subject(frame, column="eta_alpha_z")

    result = compute_band_unique_cohort_source_association_maps(
        frame,
        source_power_by_subject,
        band="alpha",
        config=load_study2_config(),
    )

    assert result.source_band == "alpha"
    assert result.subject_ids == ("sub-0001", "sub-0002")
    assert result.qc["band"].tolist() == ["alpha", "alpha"]
    assert result.fisher_z_maps.shape == (2, 3)
    assert result.partial_r_maps[:, 0].min() > 0.95


def test_compute_cohort_source_association_maps_requires_source_power_per_subject() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_maps import (
        compute_cohort_source_association_maps,
    )

    frame = _cohort_source_stage_frame(include_subject_with_unmet_criteria=False)
    source_power_by_subject = {
        "sub-0001": _source_power_from_column(
            frame.loc[frame["subject_id"] == "sub-0001", "eta_combined_z"].to_numpy(dtype=float)
        )
    }

    with pytest.raises(ValueError, match="missing source_power"):
        compute_cohort_source_association_maps(
            frame,
            source_power_by_subject,
            band="alpha",
            config=load_study2_config(),
        )


def _source_power_from_column(score: np.ndarray) -> np.ndarray:
    centered = score - np.mean(score)
    return np.column_stack(
        [
            centered,
            -centered,
            np.ones(len(centered), dtype=float),
        ]
    )


def _cohort_source_stage_frame(*, include_subject_with_unmet_criteria: bool = True):
    frames = [
        _source_stage_frame().assign(subject_id="sub-0001"),
        _source_stage_frame().assign(subject_id="sub-0002"),
    ]
    if include_subject_with_unmet_criteria:
        frames.append(_source_stage_frame(n_runs=2).assign(subject_id="sub-0003"))
    return pd.concat(frames, ignore_index=True)


def _source_power_by_subject(frame, *, column: str) -> dict[str, np.ndarray]:
    return {
        str(subject_id): _source_power_from_column(subject_frame[column].to_numpy(dtype=float))
        for subject_id, subject_frame in frame.groupby("subject_id", sort=True)
    }
