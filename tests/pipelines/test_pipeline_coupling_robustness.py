from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from studies.pain_study.analysis.eeg_bold_coupling import (
    _add_within_between_predictors,
    _negative_control_passes,
    _sign_match,
    _write_group_adjudication_summary,
    _build_alternative_roi_specs,
    _build_overridden_nuisance_config,
    _leave_one_out_refits,
    _merge_trialwise_tables,
    _summarize_leave_one_out_refits,
    _shuffle_bold_table_within_run,
)
from studies.pain_study.analysis.eeg_bold_nuisance import (
    CouplingNuisanceConfig,
    apply_trial_censoring,
    compute_dvars_table,
)
from studies.pain_study.analysis.eeg_bold_sensitivity import (
    CouplingSensitivityConfig,
)
from studies.pain_study.analysis.eeg_bold_statistics import (
    CellSpec,
    CouplingStatisticsConfig,
    _prepare_model_table,
)


def test_apply_trial_censoring_excludes_trials_above_dvars_threshold() -> None:
    merged = pd.DataFrame(
        {
            "trial_key": ["1|trial-001", "1|trial-002", "2|trial-001"],
            "run_num": [1, 1, 2],
            "temperature": [44.3, 44.3, 45.3],
            "temperature_sq": [1962.49, 1962.49, 2052.09],
            "fd": [0.1, 0.1, 0.1],
            "dvars": [1.2, 3.1, 1.8],
            "eeg_artifact": [0.5, 0.5, 0.5],
            "exp_global": [0, 1, 2],
            "block_start": [1, 0, 1],
        }
    )
    nuisance_cfg = CouplingNuisanceConfig.from_config(
        {
            "eeg_bold_coupling": {
                "nuisance": {
                    "fd": {
                        "enabled": True,
                        "output_column": "fd",
                        "source_column": "framewise_displacement",
                        "hrf_model": "spm",
                    },
                    "dvars": {
                        "enabled": True,
                        "output_column": "dvars",
                        "source_column": "std_dvars",
                        "hrf_model": "spm",
                        "censor_above": 2.5,
                    },
                    "eeg_artifact": {
                        "enabled": False,
                        "output_column": "eeg_artifact",
                        "required_components": [
                            "event_peripheral_low_gamma_power",
                            "event_residual_ecg_coupling",
                        ],
                        "component_z_threshold": 5.0,
                        "composite_threshold": 3.0,
                        "event_numeric_columns": [
                            "peripheral_low_gamma_power",
                            "residual_ecg_coupling",
                        ],
                        "global_amplitude": {"enabled": False, "window": [-5.0, 10.5]},
                        "ecg_amplitude": {"enabled": False, "channels": [], "window": [-5.0, 10.5]},
                        "peripheral_band_power": {
                            "enabled": False,
                            "channels": [],
                            "band": [30.0, 45.0],
                            "window": [3.0, 10.5],
                        },
                    },
                    "censoring": {"require_all_model_terms_finite": True},
                }
            }
        },
        active_window=(3.0, 10.5),
        epoch_window=(-7.0, 15.0),
        default_hrf_model="spm",
    )

    kept, qc_table, summary = apply_trial_censoring(
        merged_table=merged,
        model_terms=[
            "temperature",
            "temperature_sq",
            "fd",
            "dvars",
            "eeg_artifact",
            "exp_global",
            "block_start",
        ],
        nuisance_cfg=nuisance_cfg,
    )

    assert kept["trial_key"].tolist() == ["1|trial-001", "2|trial-001"]
    assert bool(qc_table.loc[qc_table["trial_key"] == "1|trial-002", "exclude_dvars_threshold"].iloc[0])
    assert qc_table.loc[qc_table["trial_key"] == "1|trial-002", "exclude_reason"].iloc[0] == "dvars_threshold"
    assert summary["n_excluded_dvars_threshold"] == 1


def test_compute_dvars_table_adds_trialwise_dvars_column(
    monkeypatch,
    tmp_path: Path,
) -> None:
    confounds_path = tmp_path / "confounds.tsv"
    pd.DataFrame({"std_dvars": [0.5, 1.0, 1.5, 2.0]}).to_csv(
        confounds_path,
        sep="\t",
        index=False,
    )
    bold_path = tmp_path / "bold.nii.gz"
    bold_path.write_bytes(b"")
    trial_table = pd.DataFrame(
        {
            "run_num": [1, 1],
            "onset": [0.0, 1.0],
            "duration": [1.0, 1.0],
            "trial_key": ["1|trial-001", "1|trial-002"],
        }
    )
    nuisance_cfg = CouplingNuisanceConfig.from_config(
        {
            "eeg_bold_coupling": {
                "nuisance": {
                    "fd": {
                        "enabled": False,
                        "output_column": "fd",
                        "source_column": "framewise_displacement",
                        "hrf_model": "spm",
                    },
                    "dvars": {
                        "enabled": True,
                        "output_column": "dvars",
                        "source_column": "std_dvars",
                        "hrf_model": "spm",
                        "censor_above": 2.5,
                    },
                    "eeg_artifact": {
                        "enabled": False,
                        "output_column": "eeg_artifact",
                        "required_components": [],
                        "component_z_threshold": None,
                        "composite_threshold": None,
                        "event_numeric_columns": [],
                        "global_amplitude": {"enabled": False, "window": [-5.0, 10.5]},
                        "ecg_amplitude": {"enabled": False, "channels": [], "window": [-5.0, 10.5]},
                        "peripheral_band_power": {
                            "enabled": False,
                            "channels": [],
                            "band": [30.0, 45.0],
                            "window": [3.0, 10.5],
                        },
                    },
                    "censoring": {"require_all_model_terms_finite": True},
                }
            }
        },
        active_window=(3.0, 10.5),
        epoch_window=(-7.0, 15.0),
        default_hrf_model="spm",
    )

    monkeypatch.setattr(
        "studies.pain_study.analysis.eeg_bold_nuisance.discover_fmriprep_preproc_bold",
        lambda **_kwargs: bold_path,
    )
    monkeypatch.setattr(
        "studies.pain_study.analysis.eeg_bold_nuisance.discover_confounds",
        lambda **_kwargs: confounds_path,
    )
    monkeypatch.setattr(
        "studies.pain_study.analysis.eeg_bold_nuisance.get_tr_from_bold",
        lambda _path: 1.0,
    )
    monkeypatch.setattr(
        "studies.pain_study.analysis.eeg_bold_nuisance.nib.load",
        lambda _path: SimpleNamespace(shape=(2, 2, 2, 4)),
    )

    out = compute_dvars_table(
        subject="001",
        task="pain",
        deriv_root=tmp_path,
        trial_table=trial_table,
        nuisance_cfg=nuisance_cfg,
        fmri_space="T1w",
    )

    assert "dvars" in out.columns
    assert out["dvars"].notna().all()


def test_shuffle_bold_table_within_run_preserves_run_specific_value_sets() -> None:
    trial_table = pd.DataFrame(
        {
            "trial_key": [
                "1|trial-001",
                "1|trial-002",
                "1|trial-003",
                "2|trial-001",
                "2|trial-002",
                "2|trial-003",
            ],
            "run_num": [1, 1, 1, 2, 2, 2],
        }
    )
    bold_table = pd.DataFrame(
        {
            "trial_key": trial_table["trial_key"].tolist(),
            "bold_midcingulate_pre_sma": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        }
    )

    shuffled = _shuffle_bold_table_within_run(
        subject="001",
        trial_table=trial_table,
        bold_table=bold_table,
        config={"project": {"random_state": 7}},
    )

    merged = trial_table.merge(shuffled, on="trial_key", how="inner", validate="one_to_one")
    run_1_values = sorted(
        merged.loc[merged["run_num"] == 1, "bold_midcingulate_pre_sma"].tolist()
    )
    run_2_values = sorted(
        merged.loc[merged["run_num"] == 2, "bold_midcingulate_pre_sma"].tolist()
    )

    assert run_1_values == [10.0, 11.0, 12.0]
    assert run_2_values == [20.0, 21.0, 22.0]
    assert not shuffled.equals(bold_table)


def test_merge_trialwise_tables_rejects_mismatched_overlapping_event_metadata() -> None:
    eeg_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "run_num": [1],
            "onset": [3.0],
            "duration": [7.5],
            "trial_number": [1],
            "events_stim_phase": ["plateau"],
            "events_selected_surface": ["arm"],
            "eeg_midcingulate_pre_sma_alpha": [0.2],
        }
    )
    trial_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "run_num": [1],
            "onset": [3.0],
            "duration": [7.5],
            "trial_number": [1],
            "events_stim_phase": ["plateau"],
            "events_selected_surface": ["leg"],
        }
    )
    bold_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "bold_midcingulate_pre_sma": [1.0],
        }
    )

    try:
        _merge_trialwise_tables(
            eeg_table=eeg_table,
            trial_table=trial_table,
            bold_table=bold_table,
        )
    except ValueError as exc:
        assert "events_selected_surface" in str(exc)
    else:
        raise AssertionError("Expected mismatched overlapping event metadata to fail.")


def test_merge_trialwise_tables_coalesces_matching_overlaps() -> None:
    eeg_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "run_num": [1],
            "onset": [3.0],
            "duration": [7.5],
            "trial_number": [1],
            "events_stim_phase": ["plateau"],
            "events_selected_surface": ["arm"],
            "eeg_midcingulate_pre_sma_alpha": [0.2],
        }
    )
    trial_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "run_num": [1],
            "onset": [3.0],
            "duration": [7.5],
            "trial_number": [1],
            "events_stim_phase": ["plateau"],
            "events_selected_surface": ["arm"],
        }
    )
    bold_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "bold_midcingulate_pre_sma": [1.0],
        }
    )

    merged = _merge_trialwise_tables(
        eeg_table=eeg_table,
        trial_table=trial_table,
        bold_table=bold_table,
    )

    assert "events_stim_phase_x" not in merged.columns
    assert "events_stim_phase_y" not in merged.columns
    assert merged["events_stim_phase"].tolist() == ["plateau"]
    assert merged["events_selected_surface"].tolist() == ["arm"]


def test_merge_trialwise_tables_preserves_distinct_eeg_and_fmri_timing() -> None:
    eeg_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "run_num": [1],
            "onset": [3.0],
            "duration": [0.001],
            "trial_number": [1],
            "events_stim_phase": ["plateau"],
            "eeg_midcingulate_pre_sma_alpha": [0.2],
        }
    )
    trial_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "run_num": [1],
            "onset": [3.625],
            "duration": [7.5],
            "trial_number": [1],
            "events_stim_phase": ["plateau"],
        }
    )
    bold_table = pd.DataFrame(
        {
            "trial_key": ["1|trial-001"],
            "bold_midcingulate_pre_sma": [1.0],
        }
    )

    merged = _merge_trialwise_tables(
        eeg_table=eeg_table,
        trial_table=trial_table,
        bold_table=bold_table,
    )

    assert merged["eeg_onset"].tolist() == [3.0]
    assert merged["eeg_duration"].tolist() == [0.001]
    assert merged["onset"].tolist() == [3.625]
    assert merged["duration"].tolist() == [7.5]


def test_artifact_model_sensitivity_config_parses_named_nuisance_overrides() -> None:
    cfg = CouplingSensitivityConfig.from_config(
        {
            "eeg_bold_coupling": {
                "sensitivities": {
                    "painful_only": {},
                    "alternative_fmri": {},
                    "delta_temperature": {},
                    "temperature_categorical": {},
                    "residualized_correlation": {},
                    "primary_permutation": {},
                    "source_methods": {},
                    "artifact_models": {
                        "enabled": True,
                        "items": [
                            {
                                "name": "expanded_artifact",
                                "nuisance_overrides": {
                                    "eeg_artifact": {
                                        "required_components": [
                                            "event_residual_ecg_coupling",
                                            "event_peripheral_low_gamma_power",
                                            "global_amplitude",
                                        ],
                                        "global_amplitude": {
                                            "enabled": True,
                                            "window": [-5.0, 10.5],
                                        },
                                    }
                                },
                            }
                        ],
                    },
                }
            }
        }
    )

    assert cfg.artifact_models.enabled is True
    assert len(cfg.artifact_models.items) == 1
    assert cfg.artifact_models.items[0].name == "expanded_artifact"
    assert (
        cfg.artifact_models.items[0]
        .nuisance_overrides["eeg_artifact"]["global_amplitude"]["enabled"]
        is True
    )


def test_build_overridden_nuisance_config_expands_artifact_components() -> None:
    base = CouplingNuisanceConfig.from_config(
        {
            "eeg_bold_coupling": {
                "nuisance": {
                    "fd": {
                        "enabled": True,
                        "output_column": "fd",
                        "source_column": "framewise_displacement",
                        "hrf_model": "spm",
                        "censor_above": 0.5,
                    },
                    "dvars": {
                        "enabled": True,
                        "output_column": "dvars",
                        "source_column": "std_dvars",
                        "hrf_model": "spm",
                        "censor_above": 2.5,
                    },
                    "eeg_artifact": {
                        "enabled": True,
                        "output_column": "eeg_artifact",
                        "required_components": [
                            "event_residual_ecg_coupling",
                            "event_peripheral_low_gamma_power",
                        ],
                        "component_z_threshold": 5.0,
                        "composite_threshold": 3.0,
                        "event_numeric_columns": [
                            "residual_ecg_coupling",
                            "peripheral_low_gamma_power",
                        ],
                        "global_amplitude": {"enabled": False, "window": [-5.0, 10.5]},
                        "ecg_amplitude": {"enabled": False, "channels": [], "window": [-5.0, 10.5]},
                        "peripheral_band_power": {
                            "enabled": False,
                            "channels": [],
                            "band": [30.0, 45.0],
                            "window": [3.0, 10.5],
                        },
                    },
                    "censoring": {"require_all_model_terms_finite": True},
                }
            }
        },
        active_window=(3.0, 10.5),
        epoch_window=(-7.0, 15.0),
        default_hrf_model="spm",
    )

    expanded = _build_overridden_nuisance_config(
        base_nuisance_cfg=base,
        nuisance_overrides={
            "eeg_artifact": {
                "required_components": [
                    "event_residual_ecg_coupling",
                    "event_peripheral_low_gamma_power",
                    "global_amplitude",
                ],
                "global_amplitude": {"enabled": True, "window": [-5.0, 10.5]},
            }
        },
        active_window=(3.0, 10.5),
        epoch_window=(-7.0, 15.0),
        default_hrf_model="spm",
    )

    assert expanded.eeg_artifact.global_amplitude.enabled is True
    assert expanded.fd.censor_above == 0.5
    assert expanded.dvars.censor_above == 2.5
    assert expanded.eeg_artifact.required_components == (
        "event_residual_ecg_coupling",
        "event_peripheral_low_gamma_power",
        "global_amplitude",
    )


def test_within_between_sensitivity_config_parses_enabled_output_name() -> None:
    cfg = CouplingSensitivityConfig.from_config(
        {
            "eeg_bold_coupling": {
                "sensitivities": {
                    "painful_only": {},
                    "alternative_fmri": {},
                    "delta_temperature": {},
                    "temperature_categorical": {},
                    "residualized_correlation": {},
                    "primary_permutation": {},
                    "source_methods": {},
                    "artifact_models": {},
                    "within_between": {
                        "enabled": True,
                        "output_name": "within_between",
                    },
                }
            }
        }
    )

    assert cfg.within_between.enabled is True
    assert cfg.within_between.output_name == "within_between"


def test_prepare_model_table_keeps_nonstandardized_subject_mean_term() -> None:
    table = pd.DataFrame(
        {
            "subject": ["01", "01", "02", "02"],
            "run_num": [1, 1, 1, 1],
            "trial_position": [1, 2, 1, 2],
            "onset": [0.0, 1.0, 0.0, 1.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "predictor_within": [-1.0, 1.0, -2.0, 2.0],
            "predictor_subject_mean": [10.0, 10.0, 20.0, 20.0],
            "outcome": [0.1, 0.2, 0.3, 0.4],
            "temperature": [44.0, 45.0, 44.0, 45.0],
        }
    )

    model_table, numeric_terms, factor_terms, variance_column = _prepare_model_table(
        table=table,
        predictor_column="predictor_within",
        outcome_column="outcome",
        outcome_variance_column=None,
        model_terms=("temperature", "predictor_subject_mean"),
        categorical_terms=(),
        nonstandardized_terms=("predictor_subject_mean",),
        include_run_fixed_effect=True,
        use_outcome_variance=False,
    )

    assert variance_column is None
    assert factor_terms == []
    assert numeric_terms == ["temperature", "predictor_subject_mean"]
    assert model_table["predictor_subject_mean"].tolist() == [10.0, 10.0, 20.0, 20.0]


def test_add_within_between_predictors_creates_centered_and_subject_mean_columns() -> None:
    merged = pd.DataFrame(
        {
            "eeg_midcingulate_pre_sma_alpha": [1.0, 2.0, 4.0],
            "bold_midcingulate_pre_sma": [0.2, 0.4, 0.6],
        }
    )

    out = _add_within_between_predictors(
        merged_table=merged,
        base_cells=[
            type(
                "CellStub",
                (),
                {
                    "predictor_column": "eeg_midcingulate_pre_sma_alpha",
                },
            )()
        ],
    )

    assert out["eeg_midcingulate_pre_sma_alpha_subject_mean"].tolist() == [7.0 / 3.0] * 3
    assert out["eeg_midcingulate_pre_sma_alpha_within_subject"].tolist() == [
        1.0 - (7.0 / 3.0),
        2.0 - (7.0 / 3.0),
        4.0 - (7.0 / 3.0),
    ]


def test_leave_one_out_refits_calls_fit_for_each_subject_and_run(monkeypatch) -> None:
    pooled = pd.DataFrame(
        {
            "subject": ["01", "01", "02", "02"],
            "run_num": [1, 2, 1, 2],
            "trial_position": [1, 1, 1, 1],
            "onset": [0.0, 0.0, 0.0, 0.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "predictor": [0.1, 0.2, 0.3, 0.4],
            "outcome": [1.0, 1.1, 1.2, 1.3],
        }
    )
    calls: list[tuple[str, list[str], list[int]]] = []

    def fake_fit_mixedlm_cell(*, pooled_table, cell, stats_cfg):
        calls.append(
            (
                str(cell.analysis_id),
                pooled_table["subject"].astype(str).tolist(),
                pooled_table["run_num"].astype(int).tolist(),
            )
        )
        return {
            "analysis_id": cell.analysis_id,
            "family": cell.family,
            "roi": cell.roi,
            "band": cell.band,
            "predictor_column": cell.predictor_column,
            "outcome_column": cell.outcome_column,
            "outcome_variance_column": "",
            "n_trials": int(len(pooled_table)),
            "n_subjects": int(pooled_table["subject"].nunique()),
            "n_runs": int(pooled_table[["subject", "run_num"]].drop_duplicates().shape[0]),
            "beta": float(len(pooled_table)),
            "se": 1.0,
            "z_value": float(len(pooled_table)),
            "p_value": 0.05,
            "ci_low": 0.0,
            "ci_high": 1.0,
            "rho": 0.0,
            "converged": True,
            "singular": False,
            "loglik": 0.0,
            "aic": 0.0,
            "bic": 0.0,
            "status": "ok",
            "interpretable": True,
            "message": "",
        }

    monkeypatch.setattr(
        "studies.pain_study.analysis.eeg_bold_coupling.fit_mixedlm_cell",
        fake_fit_mixedlm_cell,
    )

    cell = CellSpec(
        analysis_id="confirmatory__roi__alpha",
        family="confirmatory",
        roi="roi",
        band="alpha",
        predictor_column="predictor",
        outcome_column="outcome",
        outcome_variance_column=None,
        model_terms=tuple(),
        categorical_terms=tuple(),
    )
    stats_cfg = CouplingStatisticsConfig(
        backend="nlme_lme_ar1",
        min_trials_per_subject=20,
        min_runs_per_subject=2,
        alpha=0.05,
        include_run_fixed_effect=True,
        use_outcome_variance=False,
        rscript_path="Rscript",
        fit_method="reml",
        max_iterations=200,
        em_iterations=50,
        singular_tolerance=1.0e-8,
    )

    subject_refits = _leave_one_out_refits(
        pooled_subset=pooled,
        cell=cell,
        stats_cfg=stats_cfg,
        refit_type="subject",
        holdout_values=["01", "02"],
    )
    run_refits = _leave_one_out_refits(
        pooled_subset=pooled,
        cell=cell,
        stats_cfg=stats_cfg,
        refit_type="run",
        holdout_values=[1, 2],
    )

    assert subject_refits["holdout"].tolist() == ["01", "02"]
    assert run_refits["holdout"].tolist() == ["1", "2"]
    assert len(calls) == 4


def test_summarize_leave_one_out_refits_reports_sign_flips_against_reference() -> None:
    detail = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha"] * 4,
            "refit_type": ["subject", "subject", "run", "run"],
            "beta": [0.3, -0.2, 0.1, 0.2],
            "interpretable": [True, True, True, True],
        }
    )
    reference = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha"],
            "family": ["confirmatory"],
            "beta": [0.25],
            "p_value": [0.01],
        }
    )

    summary = _summarize_leave_one_out_refits(
        detail=detail,
        reference_results=reference,
    )

    subject_row = summary.loc[summary["refit_type"] == "subject"].iloc[0]
    run_row = summary.loc[summary["refit_type"] == "run"].iloc[0]

    assert int(subject_row["n_sign_flips"]) == 1
    assert bool(subject_row["sign_stable"]) is False
    assert int(run_row["n_sign_flips"]) == 0
    assert bool(run_row["sign_stable"]) is True


def test_anatomical_specificity_config_parses_control_roi_set() -> None:
    cfg = CouplingSensitivityConfig.from_config(
        {
            "eeg_bold_coupling": {
                "sensitivities": {
                    "painful_only": {},
                    "alternative_fmri": {},
                    "delta_temperature": {},
                    "temperature_categorical": {},
                    "residualized_correlation": {},
                    "primary_permutation": {},
                    "source_methods": {},
                    "artifact_models": {},
                    "within_between": {},
                    "anatomical_specificity": {
                        "enabled": True,
                        "items": [
                            {
                                "name": "control_rois",
                                "rois": [
                                    {
                                        "name": "left_operculo_control",
                                        "template_subject": "fsaverage",
                                        "parcellation": "aparc.a2009s",
                                        "annot_labels": ["G_Ins_lg_and_S_cent_ins-lh"],
                                    }
                                ],
                            }
                        ],
                    },
                }
            }
        }
    )

    assert cfg.anatomical_specificity.enabled is True
    assert cfg.anatomical_specificity.items[0].name == "control_rois"
    assert cfg.anatomical_specificity.items[0].rois[0]["name"] == "left_operculo_control"


def test_build_alternative_roi_specs_accepts_annot_defined_control_rois() -> None:
    specs = _build_alternative_roi_specs(
        roi_items=[
            {
                "name": "left_operculo_control",
                "template_subject": "fsaverage",
                "parcellation": "aparc.a2009s",
                "annot_labels": [
                    "G_Ins_lg_and_S_cent_ins-lh",
                    "G_and_S_subcentral-lh",
                ],
            },
            {
                "name": "posterior_midcingulate_control",
                "template_subject": "fsaverage",
                "parcellation": "aparc.a2009s",
                "annot_labels": [
                    "G_and_S_cingul-Mid-Post-lh",
                    "G_and_S_cingul-Mid-Post-rh",
                ],
            },
        ],
        default_template_subject="fsaverage",
    )

    assert [spec.name for spec in specs] == [
        "left_operculo_control",
        "posterior_midcingulate_control",
    ]
    assert specs[0].annot_labels == (
        "G_Ins_lg_and_S_cent_ins-lh",
        "G_and_S_subcentral-lh",
    )


def test_sign_match_requires_finite_nonzero_same_direction() -> None:
    assert _sign_match(reference_beta=0.4, candidate_beta=0.2) is True
    assert _sign_match(reference_beta=0.4, candidate_beta=-0.2) is False
    assert _sign_match(reference_beta=0.0, candidate_beta=0.2) is False
    assert _sign_match(reference_beta=0.4, candidate_beta=float("nan")) is False


def test_negative_control_passes_when_not_interpretable_or_nonsignificant() -> None:
    uninterpretable = type("Row", (), {"interpretable": False, "p_value": 0.001})()
    nonsignificant = type("Row", (), {"interpretable": True, "p_value": 0.7})()
    significant = type("Row", (), {"interpretable": True, "p_value": 0.01})()

    assert _negative_control_passes(row=uninterpretable, alpha=0.05) is True
    assert _negative_control_passes(row=nonsignificant, alpha=0.05) is True
    assert _negative_control_passes(row=significant, alpha=0.05) is False


def test_write_group_adjudication_summary_marks_overall_pass(tmp_path: Path) -> None:
    group_dir = tmp_path / "group"
    (group_dir / "sensitivities" / "eloreta").mkdir(parents=True, exist_ok=True)
    (group_dir / "sensitivities" / "expanded_artifact").mkdir(parents=True, exist_ok=True)
    (group_dir / "sensitivities" / "within_between").mkdir(parents=True, exist_ok=True)
    (group_dir / "negative_controls" / "trial_shuffle").mkdir(parents=True, exist_ok=True)
    (group_dir / "robustness").mkdir(parents=True, exist_ok=True)

    confirmatory = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha"],
            "family": ["confirmatory"],
            "roi": ["roi"],
            "band": ["alpha"],
            "beta": [0.3],
            "p_value": [0.01],
            "p_holm": [0.02],
            "interpretable": [True],
            "significant_holm": [True],
        }
    )
    sensitivity_match = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha"],
            "family": ["sensitivity_source_method"],
            "beta": [0.2],
            "interpretable": [True],
            "p_value": [0.2],
        }
    )
    artifact_match = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha"],
            "family": ["sensitivity_artifact_model"],
            "beta": [0.1],
            "interpretable": [True],
            "p_value": [0.2],
        }
    )
    within_between = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha"],
            "family": ["sensitivity_within_between"],
            "beta": [0.25],
            "interpretable": [True],
            "p_value": [0.2],
        }
    )
    negative_control = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha"],
            "family": ["negative_control_trial_shuffle"],
            "beta": [0.05],
            "interpretable": [True],
            "p_value": [0.6],
        }
    )
    loo = pd.DataFrame(
        {
            "analysis_id": ["confirmatory__roi__alpha", "confirmatory__roi__alpha"],
            "refit_type": ["subject", "run"],
            "sign_stable": [True, True],
        }
    )

    sensitivity_match.to_csv(group_dir / "sensitivities" / "eloreta" / "group_results.tsv", sep="\t", index=False)
    artifact_match.to_csv(group_dir / "sensitivities" / "expanded_artifact" / "group_results.tsv", sep="\t", index=False)
    within_between.to_csv(group_dir / "sensitivities" / "within_between" / "group_results.tsv", sep="\t", index=False)
    negative_control.to_csv(group_dir / "negative_controls" / "trial_shuffle" / "group_results.tsv", sep="\t", index=False)
    loo.to_csv(group_dir / "robustness" / "leave_one_out_summary.tsv", sep="\t", index=False)

    config = {
        "eeg_bold_coupling": {
            "sensitivities": {
                "painful_only": {},
                "alternative_fmri": {},
                "delta_temperature": {},
                "temperature_categorical": {},
                "residualized_correlation": {},
                "primary_permutation": {},
                "source_methods": {
                    "enabled": True,
                    "items": [{"name": "eloreta", "method": "eloreta", "bands": ["alpha"]}],
                },
                "anatomical_specificity": {},
                "artifact_models": {
                    "enabled": True,
                    "items": [{"name": "expanded_artifact", "nuisance_overrides": {"eeg_artifact": {"global_amplitude": {"enabled": True}}}}],
                },
                "within_between": {"enabled": True, "output_name": "within_between"},
            },
            "negative_controls": {
                "trial_shuffle": {"enabled": True, "output_name": "trial_shuffle"}
            },
        }
    }

    _write_group_adjudication_summary(
        group_dir=group_dir,
        group_results=confirmatory,
        config=config,
        alpha=0.05,
    )

    adjudication = pd.read_csv(group_dir / "robustness" / "adjudication_summary.tsv", sep="\t")
    row = adjudication.iloc[0]
    assert bool(row["all_source_methods_pass"]) is True
    assert bool(row["all_artifact_models_pass"]) is True
    assert bool(row["within_between_pass"]) is True
    assert bool(row["leave_one_subject_pass"]) is True
    assert bool(row["leave_one_run_pass"]) is True
    assert bool(row["negative_control_pass"]) is True
    assert bool(row["overall_robust_pass"]) is True
