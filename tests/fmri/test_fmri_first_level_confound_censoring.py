"""First-level GLMs must censor fMRIPrep's leading non-finite confound rows.

fMRIPrep always writes ``n/a`` in row 0 of every ``*_derivative1`` column and of
``framewise_displacement``. The first-level GLM path must handle that the same
way the trial-wise path already does: censor the affected volume through
``sample_masks`` rather than rejecting the run or zero-filling the regressor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

from fmri_pipeline.analysis.contrast_builder import (
    ContrastBuilderConfig,
    fit_first_level_glm_multi_run,
)


def _make_cfg(**overrides: Any) -> ContrastBuilderConfig:
    base: Dict[str, Any] = dict(
        enabled=True,
        input_source="fmriprep",
        fmriprep_space="T1w",
        require_fmriprep=False,
        contrast_type="t-test",
        condition1=None,
        condition2=None,
        condition_a_column="trial_type",
        condition_a_value="pain",
        condition_b_column="trial_type",
        condition_b_value="rest",
        formula=None,
        name="pain-vs-rest",
        runs=[1],
        hrf_model="spm",
        drift_model="cosine",
        high_pass_hz=0.008,
        low_pass_hz=None,
        output_type="z-score",
        resample_to_freesurfer=False,
        confounds_strategy="auto",
    )
    base.update(overrides)
    return ContrastBuilderConfig(**base)


def _write_events(path: Path) -> Path:
    pd.DataFrame(
        {
            "onset": [10.0, 40.0, 70.0, 100.0],
            "duration": [10.0, 10.0, 10.0, 10.0],
            "trial_type": ["pain", "rest", "pain", "rest"],
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def _write_fmriprep_confounds(path: Path, *, n_volumes: int = 60) -> Path:
    """Write a confounds TSV shaped like fMRIPrep's, with n/a in row 0."""
    rng = np.random.default_rng(0)
    frame = {}
    for axis in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"):
        values = rng.normal(scale=0.05, size=n_volumes)
        frame[axis] = values
        derivative = np.diff(values, prepend=np.nan)
        frame[f"{axis}_derivative1"] = derivative
        frame[f"{axis}_power2"] = values**2
        frame[f"{axis}_derivative1_power2"] = derivative**2
    frame["white_matter"] = rng.normal(size=n_volumes)
    frame["csf"] = rng.normal(size=n_volumes)
    fd = rng.gamma(2.0, 0.05, size=n_volumes)
    fd[0] = np.nan
    frame["framewise_displacement"] = fd
    pd.DataFrame(frame).to_csv(path, sep="\t", index=False, na_rep="n/a")
    return path


class _RecordingFirstLevelModel:
    """Minimal stand-in that records what the builder passes to ``fit``."""

    def __init__(self) -> None:
        self.fit_kwargs: Dict[str, Any] = {}
        self.design_matrices_: List[pd.DataFrame] = []

    def fit(self, imgs: Any, events: Any = None, confounds: Any = None, **kwargs: Any) -> None:
        self.fit_kwargs = {"imgs": imgs, "events": events, "confounds": confounds, **kwargs}
        n_runs = len(imgs) if isinstance(imgs, (list, tuple)) else 1
        sample_masks = kwargs.get("sample_masks")
        retained_counts = (
            [len(mask) for mask in sample_masks] if sample_masks is not None else [60] * n_runs
        )
        self.design_matrices_ = [
            pd.DataFrame(
                {
                    "cond_a_pain": np.zeros(retained_count),
                    "constant": np.ones(retained_count),
                }
            )
            for retained_count in retained_counts
        ]


def _run_multi_run_fit(
    tmp_path: Path,
    *,
    n_runs: int = 2,
    confounds_strategy: str = "auto",
) -> _RecordingFirstLevelModel:
    bold_paths: List[Path] = []
    events_paths: List[Path] = []
    confounds_paths: List[Optional[Path]] = []
    for run in range(1, n_runs + 1):
        bold = tmp_path / f"run-0{run}_bold.nii.gz"
        bold.write_bytes(b"")
        bold_paths.append(bold)
        events_paths.append(_write_events(tmp_path / f"run-0{run}_events.tsv"))
        confounds_paths.append(_write_fmriprep_confounds(tmp_path / f"run-0{run}_confounds.tsv"))

    model = _RecordingFirstLevelModel()
    with (
        patch("fmri_pipeline.analysis.contrast_builder._validate_events_against_bold_run"),
        patch("fmri_pipeline.analysis.contrast_builder._validate_consistent_trs", return_value=2.0),
        patch(
            "fmri_pipeline.analysis.contrast_builder._build_intersection_brain_mask",
            return_value="mask",
        ),
        patch(
            "fmri_pipeline.analysis.contrast_builder._build_first_level_model",
            return_value=model,
        ),
        patch("fmri_pipeline.analysis.contrast_builder._validate_design_matrices"),
    ):
        model.glm_result = fit_first_level_glm_multi_run(
            bold_paths=bold_paths,
            events_paths=events_paths,
            confounds_paths=confounds_paths,
            cfg=_make_cfg(confounds_strategy=confounds_strategy),
        )
    return model


def test_multi_run_fit_does_not_reject_fmriprep_leading_nan_confounds(tmp_path: Path) -> None:
    """The stock 'auto' strategy must run against unmodified fMRIPrep confounds."""
    model = _run_multi_run_fit(tmp_path)
    assert model.fit_kwargs, "GLM was never fitted"


def test_multi_run_fit_censors_leading_nan_volume_via_sample_masks(tmp_path: Path) -> None:
    model = _run_multi_run_fit(tmp_path)

    sample_masks = model.fit_kwargs.get("sample_masks")
    assert sample_masks is not None, "sample_masks was not passed to FirstLevelModel.fit"
    assert len(sample_masks) == 2
    for mask in sample_masks:
        retained = np.asarray(mask, dtype=int)
        assert 0 not in retained, "leading non-finite volume was not censored"
        assert retained.size == 59


def test_multi_run_fit_records_the_exact_retained_frame_indices(tmp_path: Path) -> None:
    model = _run_multi_run_fit(tmp_path)

    assert model.glm_result.retained_frame_indices == [
        tuple(range(1, 60)),
        tuple(range(1, 60)),
    ]


def test_multi_run_fit_never_passes_nonfinite_confounds_to_nilearn(tmp_path: Path) -> None:
    model = _run_multi_run_fit(tmp_path)

    confounds = model.fit_kwargs.get("confounds")
    assert isinstance(confounds, list) and confounds
    for frame in confounds:
        values = frame.to_numpy(dtype=float)
        assert np.isfinite(values).all(), "non-finite confound values reached the GLM"


def test_multi_run_fit_standardizes_confounds_over_retained_volumes_only(
    tmp_path: Path,
) -> None:
    """The censored volume must not contribute to the confound scaling.

    The placeholder written into a censored row is the retained-volume mean, so it
    lands at ~0 once standardized. What distinguishes that from zero-filling is
    whether the mean and scale were computed with the censored row excluded.
    """
    model = _run_multi_run_fit(tmp_path)

    frame = model.fit_kwargs["confounds"][0]
    retained = np.asarray(model.fit_kwargs["sample_masks"][0], dtype=int)
    retained_values = frame.to_numpy(dtype=float)[retained, :]

    assert np.isfinite(retained_values).all()
    np.testing.assert_allclose(retained_values.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(retained_values.std(axis=0), 1.0, atol=1e-9)


def test_multi_run_fit_omits_sample_masks_when_confounds_are_disabled(tmp_path: Path) -> None:
    model = _run_multi_run_fit(tmp_path, confounds_strategy="none")

    assert model.fit_kwargs["confounds"] is None
    assert model.fit_kwargs.get("sample_masks") is None
