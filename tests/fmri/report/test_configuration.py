"""The report has to carry the configuration every result was produced under.

A result shown without its settings cannot be reproduced from and cannot be compared
against another study: an HRF basis, a high-pass cutoff, or a different resolved
confound set changes the numbers, and none of those differences is visible in any map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report import subject
from fmri_pipeline.analysis.report.manifest import (
    ContrastManifest,
    model_settings_from_config,
    read_manifest,
    write_report_manifest,
)


@dataclass
class _Cfg:
    hrf_model: str = "spm + derivative"
    drift_model: Optional[str] = "cosine"
    high_pass_hz: float = 0.008
    low_pass_hz: Optional[float] = None
    confounds_strategy: str = "motion24+wmcsf"
    auto_compcor_n: int = 5
    output_type: str = "z-score"
    fmriprep_space: str = "T1w"
    resample_to_freesurfer: bool = False
    contrast_type: str = "t-test"
    formula: Optional[str] = None


def _manifest(**overrides) -> ContrastManifest:
    base = dict(
        subject="sub-01",
        task="heat",
        contrast_name="pain-vs-warm",
        space="native",
        stat_map=Path("z.nii.gz"),
        effect_map=None,
        variance_map=None,
        mask=None,
        threshold_mode="z",
        z_threshold=2.3,
        fdr_q=0.05,
        cluster_min_voxels=0,
        two_sided=True,
        radiological=False,
        design_matrices=(),
        contrast_vector=None,
        contrast_columns=(),
        included_runs=("run-01",),
        excluded_runs=(),
        bold_paths=(),
        confounds_paths=(),
        t_r=0.9,
        smoothing_fwhm=6.0,
        signal_scaling=False,
        confound_strategy="motion24+wmcsf",
        model_settings=model_settings_from_config(_Cfg()),
        confound_columns=("trans_x", "rot_z", "white_matter"),
    )
    base.update(overrides)
    return ContrastManifest(**base)


# --- extraction -----------------------------------------------------------


def test_the_settings_that_change_the_numbers_are_extracted() -> None:
    settings = dict(model_settings_from_config(_Cfg()))
    assert settings["HRF model"] == "spm + derivative"
    assert settings["Drift model"] == "cosine"
    assert settings["Confound strategy"] == "motion24+wmcsf"


def test_a_high_pass_cutoff_is_given_in_seconds_as_well_as_hertz() -> None:
    # Cutoffs are configured in Hz and reported in the literature in seconds.
    settings = dict(model_settings_from_config(_Cfg(high_pass_hz=0.008)))
    assert "0.008 Hz" in settings["High-pass cutoff"]
    assert "125 s" in settings["High-pass cutoff"]


def test_an_absent_drift_model_is_named_rather_than_left_blank() -> None:
    settings = dict(model_settings_from_config(_Cfg(drift_model=None)))
    assert settings["Drift model"] == "none"


def test_a_config_without_a_setting_simply_omits_it() -> None:
    # A config that gains or loses an option must not break the manifest.
    @dataclass
    class _Sparse:
        hrf_model: str = "glover"

    settings = dict(model_settings_from_config(_Sparse()))
    assert settings == {"HRF model": "glover"}


def test_no_config_yields_no_settings() -> None:
    assert model_settings_from_config(None) == ()


def test_an_unreadable_setting_is_omitted_rather_than_guessed() -> None:
    # A wrong value here is worse than a missing one: the reader cannot check it
    # against the maps.
    class _Exploding:
        hrf_model = "glover"

        @property
        def high_pass_hz(self):
            raise RuntimeError("boom")

    assert dict(model_settings_from_config(_Exploding())) == {"HRF model": "glover"}


# --- round trip -----------------------------------------------------------


def test_the_settings_survive_a_manifest_round_trip(tmp_path: Path) -> None:
    stat = tmp_path / "z.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(stat))
    written = write_report_manifest(
        contrast_dir=tmp_path,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat,
        run_meta={"tr": 0.9, "confound_columns": ["trans_x", "csf"]},
        contrast_cfg=_Cfg(),
    )
    loaded = read_manifest(written)
    assert dict(loaded.model_settings)["HRF model"] == "spm + derivative"
    assert loaded.confound_columns == ("trans_x", "csf")


def test_a_manifest_without_configuration_still_loads(tmp_path: Path) -> None:
    stat = tmp_path / "z.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(stat))
    written = write_report_manifest(
        contrast_dir=tmp_path,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat,
        run_meta={"tr": 0.9},
    )
    loaded = read_manifest(written)
    assert loaded.model_settings == ()
    assert loaded.confound_columns == ()


# --- the report section ---------------------------------------------------


def test_the_configuration_section_states_the_model_settings() -> None:
    text = str(subject.build_configuration_section([_manifest()]))
    assert "HRF model" in text and "spm + derivative" in text
    assert "High-pass cutoff" in text


def test_the_configuration_section_names_every_confound_column() -> None:
    # A count would not let a reader tell one "auto" resolution from another, which
    # is the whole reason for recording them.
    text = str(subject.build_configuration_section([_manifest()]))
    for column in ("trans_x", "rot_z", "white_matter"):
        assert column in text


def test_the_configuration_is_recorded_per_contrast_not_once(tmp_path: Path) -> None:
    # Contrasts of one subject need not share a model: formula, confound strategy and
    # space are all configurable per contrast.
    first = _manifest(contrast_name="a", model_settings=model_settings_from_config(_Cfg()))
    second = _manifest(
        contrast_name="b",
        model_settings=model_settings_from_config(_Cfg(hrf_model="fir")),
    )
    text = str(subject.build_configuration_section([first, second]))
    assert "spm + derivative" in text and "fir" in text


def test_the_configuration_section_is_collapsed() -> None:
    # Reference material, not something that competes with the result.
    assert subject.build_configuration_section([_manifest()]).collapsed


def test_a_manifest_without_settings_says_so_rather_than_showing_nothing() -> None:
    text = str(subject.build_configuration_section([_manifest(model_settings=())]))
    # The remaining settings still render, so the section is not empty.
    assert "Signal scaling" in text


def test_manifests_carrying_nothing_at_all_explain_the_gap() -> None:
    section = subject.build_configuration_section([])
    assert "No model configuration was recorded" in str(section)


# --- the machine-readable sidecar -----------------------------------------


def test_the_configuration_is_also_written_as_json(tmp_path: Path) -> None:
    # Checking a cohort was fit under one configuration is not a question anyone
    # should answer by opening ninety reports.
    path = subject.write_configuration_json([_manifest()], out_path=tmp_path / "config.json")
    payload = json.loads(path.read_text())
    assert payload["subject"] == "sub-01"
    assert payload["contrasts"][0]["model_settings"]["HRF model"] == "spm + derivative"
    assert payload["contrasts"][0]["confound_columns"] == [
        "trans_x",
        "rot_z",
        "white_matter",
    ]


def test_the_json_carries_the_thresholding_as_well_as_the_model(tmp_path: Path) -> None:
    path = subject.write_configuration_json([_manifest()], out_path=tmp_path / "config.json")
    contrast = json.loads(path.read_text())["contrasts"][0]
    assert contrast["threshold_mode"] == "z"
    assert contrast["z_threshold"] == pytest.approx(2.3)
    assert contrast["two_sided"] is True


def test_the_json_is_stable_across_writes(tmp_path: Path) -> None:
    # Sorted keys, so a cohort's configs can be diffed against one another.
    first = subject.write_configuration_json([_manifest()], out_path=tmp_path / "a.json")
    second = subject.write_configuration_json([_manifest()], out_path=tmp_path / "b.json")
    assert first.read_text() == second.read_text()
