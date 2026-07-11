from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

TEMPERATURES = (41.3, 42.3, 43.3, 44.3, 45.3, 46.3)


def test_temperature_events_encode_centered_effect_and_nuisance_structure() -> None:
    from studies.pain_study.study1.figures.fmri_construct_data import (
        build_temperature_events,
    )

    raw, retained = subject_events()
    events = build_temperature_events(raw, retained)

    target = events.loc[events["trial_type"].eq("temperature_linear")]
    assert np.allclose(target.groupby("run")["modulation"].mean(), 0.0)
    assert np.allclose(
        target.loc[target["run"].eq(1), "modulation"],
        np.asarray(TEMPERATURES) - np.mean(TEMPERATURES),
    )
    assert set(events["trial_type"]) >= {
        "plateau_mean",
        "temperature_linear",
        "within_run_trial_order",
        "nuisance_fixation_rest",
        "nuisance_stimulation_ramp_up",
    }


def test_rating_events_encode_within_temperature_intensity_in_ten_point_units() -> None:
    from studies.pain_study.study1.figures.fmri_construct_data import build_rating_events

    raw, retained = subject_events()
    events = build_rating_events(raw, retained)

    rating = events.loc[events["trial_type"].eq("rating_within_temperature")]
    assert np.allclose(rating.groupby("stimulus_temp")["modulation"].mean(), 0.0)
    assert np.allclose(
        rating.loc[rating["run"].eq(1), "modulation"],
        -0.25,
    )
    assert np.allclose(
        rating.loc[rating["run"].eq(2), "modulation"],
        0.25,
    )
    temperature_regressors = {f"temperature_{value:g}" for value in TEMPERATURES}
    assert temperature_regressors.issubset(set(events["trial_type"]))


def test_event_design_rejects_duplicate_retained_trial_keys() -> None:
    from studies.pain_study.study1.figures.fmri_construct_data import (
        build_temperature_events,
    )

    raw, retained = subject_events()

    with pytest.raises(ValueError, match="duplicate retained trial keys"):
        build_temperature_events(raw, pd.concat([retained, retained.iloc[[0]]]))


def test_event_design_rejects_surface_changes_within_run() -> None:
    from studies.pain_study.study1.figures.fmri_construct_data import (
        build_temperature_events,
    )

    raw, retained = subject_events()
    raw.loc[(raw["run_id"] == 1) & (raw["trial_number"] == 2), "selected_surface"] = 9

    with pytest.raises(ValueError, match="one thermode surface per run"):
        build_temperature_events(raw, retained)


def test_event_design_rejects_unmatched_retained_trials() -> None:
    from studies.pain_study.study1.figures.fmri_construct_data import build_rating_events

    raw, retained = subject_events()
    retained.loc[0, "within_run_trial"] = 999

    with pytest.raises(ValueError, match="without matching raw fMRI plateau events"):
        build_rating_events(raw, retained)


def test_event_design_models_unkeyed_nontrial_events_as_nuisance() -> None:
    from studies.pain_study.study1.figures.fmri_construct_data import (
        build_temperature_events,
    )

    raw, retained = subject_events()
    raw = pd.concat(
        [
            raw,
            pd.DataFrame(
                {
                    "run_id": [1],
                    "trial_number": [np.nan],
                    "onset": [1.0],
                    "duration": [2.0],
                    "trial_type": ["instructions"],
                    "stim_phase": [np.nan],
                    "stimulus_temp": [np.nan],
                    "selected_surface": [np.nan],
                    "pain_binary_coded": [np.nan],
                    "vas_final_coded_rating": [np.nan],
                }
            ),
        ],
        ignore_index=True,
    )

    events = build_temperature_events(raw, retained)

    assert "nuisance_instructions" in set(events["trial_type"])


def test_rating_design_rejects_zero_within_temperature_variation() -> None:
    from studies.pain_study.study1.figures.fmri_construct_data import build_rating_events

    raw, retained = subject_events()
    retained["within_scale_intensity"] = retained.groupby("stimulus_temp")[
        "within_scale_intensity"
    ].transform("mean")

    with pytest.raises(ValueError, match="within-temperature rating variation"):
        build_rating_events(raw, retained)


def test_first_level_settings_use_prespecified_study1_glm() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.fmri_construct_models import (
        first_level_settings,
    )

    settings = first_level_settings(load_study1_config())

    assert settings.hrf_model == "spm"
    assert settings.high_pass_hz == 0.008
    assert settings.smoothing_fwhm == 6.0
    assert settings.confounds_strategy == "motion24"
    assert settings.max_condition_number == 3000.0


def test_fit_subject_effects_recovers_known_temperature_and_rating_signals(
    tmp_path: Path,
) -> None:
    import nibabel as nib
    from nilearn.glm.first_level import make_first_level_design_matrix

    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.fmri_construct_data import (
        FmriRunInput,
        build_subject_designs,
    )
    from studies.pain_study.study1.figures.fmri_construct_models import (
        first_level_settings,
        fit_subject_effects,
    )

    raw, retained = single_run_events()
    mask_path = tmp_path / "mask.nii.gz"
    bold_path = tmp_path / "bold.nii.gz"
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), affine), mask_path)
    run_input = FmriRunInput(
        subject_id="sub-0001",
        run=1,
        bold_path=bold_path,
        mask_path=mask_path,
        confounds_path=tmp_path / "unused.tsv",
        raw_events_path=tmp_path / "events.tsv",
        raw_events=raw,
        retained_trials=retained,
    )
    designs = build_subject_designs((run_input,))
    frame_times = np.arange(160, dtype=float)
    regressors: dict[str, np.ndarray] = {}
    for design in designs:
        matrix = make_first_level_design_matrix(
            frame_times,
            events=design.events,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=0.008,
        )
        regressors[design.estimand] = matrix[design.target_column].to_numpy(dtype=float)
    rng = np.random.default_rng(4)
    data = 100.0 + rng.normal(0.0, 0.5, size=(5, 5, 5, len(frame_times)))
    data[2, 2, 2, :] += 4.0 * regressors["temperature"]
    data[3, 3, 3, :] += 6.0 * regressors["rating"]
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    image.header.set_zooms((1.0, 1.0, 1.0, 1.0))
    nib.save(image, bold_path)

    settings = replace(
        first_level_settings(load_study1_config()),
        confounds_strategy="none",
        smoothing_fwhm=None,
        max_condition_number=1e8,
        min_target_efficiency=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = fit_subject_effects(
            subject_runs=(run_input,),
            designs=designs,
            settings=settings,
        )

    assert set(result.effect_images) == {"temperature", "rating"}
    assert result.effect_images["temperature"].get_fdata()[2, 2, 2] > 0.5
    assert result.effect_images["rating"].get_fdata()[3, 3, 3] > 0.5
    assert set(result.design_audit["target_column"]) == {
        "temperature_linear",
        "rating_within_temperature",
    }


def test_group_inference_is_deterministic_and_uses_participants() -> None:
    import nibabel as nib

    from studies.pain_study.study1.figures.fmri_construct_models import (
        GroupInferenceSettings,
        run_group_inference,
    )

    affine = np.eye(4)
    mask = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), affine)
    rng = np.random.default_rng(12)
    effects = []
    for _subject in range(12):
        values = rng.normal(0.0, 0.02, size=(5, 5, 5))
        values[2, 2, 2] += 3.0
        values[1, 1, 1] -= 2.0
        effects.append(nib.Nifti1Image(values.astype(np.float32), affine))
    settings = GroupInferenceSettings(
        n_permutations=128,
        two_sided=True,
        alpha=0.05,
        random_state=9,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = run_group_inference(
            effects,
            analysis_mask=mask,
            estimand="temperature",
            settings=settings,
        )
        second = run_group_inference(
            effects,
            analysis_mask=mask,
            estimand="temperature",
            settings=settings,
        )

    assert first.n_subjects == 12
    assert np.allclose(first.mean_effect.get_fdata(), second.mean_effect.get_fdata())
    assert np.allclose(first.neg_log10_fwe_p.get_fdata(), second.neg_log10_fwe_p.get_fdata())
    assert np.array_equal(
        np.asanyarray(first.significance_mask.dataobj),
        np.asanyarray(second.significance_mask.dataobj),
    )
    assert np.asanyarray(first.significance_mask.dataobj).dtype == np.uint8
    assert set(first.peaks["sign"]) == {"negative", "positive"}
    assert first.peaks["n_voxels"].ge(1).all()


def test_build_summary_preserves_identical_participant_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nibabel as nib

    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.fmri_construct_models import (
        GroupMapResult,
        SubjectEffectResult,
    )
    import studies.pain_study.study1.figures.fmri_construct_validity as module

    affine = np.eye(4)
    mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), affine)
    runs = tuple(
        SimpleNamespace(subject_id=subject_id, run=1) for subject_id in ("sub-0001", "sub-0002")
    )
    monkeypatch.setattr(module, "load_fmri_run_inputs", lambda **kwargs: runs)
    monkeypatch.setattr(module, "build_subject_designs", lambda subject_runs: ())

    def fit_subject_effects(*, subject_runs, designs, settings):
        subject_id = subject_runs[0].subject_id
        value = float(int(subject_id[-1]))
        image = nib.Nifti1Image(np.full((3, 3, 3), value, dtype=np.float32), affine)
        return SubjectEffectResult(
            subject_id=subject_id,
            effect_images={"temperature": image, "rating": image},
            analysis_mask=mask,
            design_audit=pd.DataFrame({"subject_id": [subject_id], "estimand": ["temperature"]}),
        )

    def run_group_inference(images, *, analysis_mask, estimand, settings):
        mean = nib.Nifti1Image(
            np.mean([image.get_fdata() for image in images], axis=0),
            affine,
        )
        return GroupMapResult(
            estimand=estimand,
            mean_effect=mean,
            neg_log10_fwe_p=nib.Nifti1Image(np.zeros((3, 3, 3)), affine),
            significance_mask=nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.uint8), affine),
            peaks=pd.DataFrame(),
            n_subjects=len(images),
        )

    monkeypatch.setattr(module, "fit_subject_effects", fit_subject_effects)
    monkeypatch.setattr(module, "run_group_inference", run_group_inference)

    summary = module.build_fmri_construct_validity_summary(
        task="thermalactive",
        config=load_study1_config(),
    )

    assert summary.subjects["subject_id"].tolist() == ["sub-0001", "sub-0002"]
    assert summary.n_subjects == 2
    assert not summary.article_ready
    assert set(summary.subject_effects) == {"temperature", "rating"}
    assert all(len(images) == 2 for images in summary.subject_effects.values())


def subject_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    retained_rows: list[dict[str, object]] = []
    for run in (1, 2):
        for trial, temperature in enumerate(TEMPERATURES, start=1):
            rating = 10.0 * trial + 5.0 * (run - 1)
            raw_rows.extend(
                [
                    {
                        "run_id": run,
                        "trial_number": trial,
                        "onset": 30.0 * trial,
                        "duration": 7.5,
                        "trial_type": "stimulation",
                        "stim_phase": "plateau",
                        "stimulus_temp": temperature,
                        "selected_surface": run,
                        "pain_binary_coded": 0,
                        "vas_final_coded_rating": rating,
                    },
                    {
                        "run_id": run,
                        "trial_number": trial,
                        "onset": 30.0 * trial - 3.0,
                        "duration": 3.0,
                        "trial_type": "stimulation",
                        "stim_phase": "ramp_up",
                        "stimulus_temp": temperature,
                        "selected_surface": run,
                        "pain_binary_coded": 0,
                        "vas_final_coded_rating": rating,
                    },
                    {
                        "run_id": run,
                        "trial_number": trial,
                        "onset": 30.0 * trial - 10.0,
                        "duration": 7.0,
                        "trial_type": "fixation_rest",
                        "stim_phase": np.nan,
                        "stimulus_temp": temperature,
                        "selected_surface": run,
                        "pain_binary_coded": 0,
                        "vas_final_coded_rating": rating,
                    },
                ]
            )
            retained_rows.append(
                {
                    "subject_id": "sub-0001",
                    "run": run,
                    "within_run_trial": trial,
                    "stimulus_temp": temperature,
                    "selected_surface": run,
                    "pain_binary_coded": 0,
                    "vas_final_coded_rating": rating,
                    "within_scale_intensity": rating,
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(retained_rows)


def single_run_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    retained_rows: list[dict[str, object]] = []
    temperatures = TEMPERATURES * 2
    rating_offsets = (2.0, 5.0, 9.0, 4.0, 7.0, 12.0)
    for trial, temperature in enumerate(temperatures, start=1):
        repeat = 0 if trial <= len(TEMPERATURES) else 1
        temperature_index = (trial - 1) % len(TEMPERATURES)
        rating = 30.0 + temperature_index + repeat * rating_offsets[temperature_index]
        onset = 8.0 + 11.0 * (trial - 1)
        raw_rows.extend(
            [
                {
                    "run_id": 1,
                    "trial_number": trial,
                    "onset": onset,
                    "duration": 5.0,
                    "trial_type": "stimulation",
                    "stim_phase": "plateau",
                    "stimulus_temp": temperature,
                    "selected_surface": 1,
                    "pain_binary_coded": 0,
                    "vas_final_coded_rating": rating,
                },
                {
                    "run_id": 1,
                    "trial_number": trial,
                    "onset": onset - 3.0,
                    "duration": 3.0,
                    "trial_type": "stimulation",
                    "stim_phase": "ramp_up",
                    "stimulus_temp": temperature,
                    "selected_surface": 1,
                    "pain_binary_coded": 0,
                    "vas_final_coded_rating": rating,
                },
            ]
        )
        retained_rows.append(
            {
                "subject_id": "sub-0001",
                "run": 1,
                "within_run_trial": trial,
                "stimulus_temp": temperature,
                "selected_surface": 1,
                "pain_binary_coded": 0,
                "vas_final_coded_rating": rating,
                "within_scale_intensity": rating,
            }
        )
    return pd.DataFrame(raw_rows), pd.DataFrame(retained_rows)
