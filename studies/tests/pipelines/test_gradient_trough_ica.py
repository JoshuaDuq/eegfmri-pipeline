from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_refine_trough_indices_finds_run_specific_local_minima() -> None:
    from studies.pain_study.gradient_trough_ica.trough_selection import (
        refine_trough_indices,
    )

    signal = np.full(100, 10.0)
    signal[[22, 71]] = [1.0, 2.0]

    indices = refine_trough_indices(
        signal,
        reference_indices=np.array([20, 70]),
        search_radius_samples=3,
    )

    np.testing.assert_array_equal(indices, [22, 71])


def test_derive_plateau_windows_selects_contiguous_flat_trough_bottoms() -> None:
    from studies.pain_study.gradient_trough_ica.trough_selection import (
        derive_plateau_windows,
    )

    signal = np.full(50, 10.0)
    signal[7:14] = [8.0, 4.0, 2.0, 2.0, 2.0, 4.0, 8.0]
    signal[27:34] = [8.0, 5.0, 3.0, 3.0, 3.0, 5.0, 8.0]

    windows = derive_plateau_windows(
        signal,
        trough_indices=np.array([10, 30]),
        depth_fraction=0.2,
        minimum_samples=3,
        maximum_samples=5,
    )

    assert [(window.start, window.stop) for window in windows] == [(9, 12), (29, 32)]
    assert [window.trough_index for window in windows] == [10, 30]
    assert windows[0].threshold == pytest.approx(3.6)
    assert windows[1].threshold == pytest.approx(4.4)


def test_derive_plateau_windows_rejects_nonflat_trough() -> None:
    from studies.pain_study.gradient_trough_ica.trough_selection import (
        derive_plateau_windows,
    )

    signal = np.full(30, 10.0)
    signal[15] = 1.0

    with pytest.raises(ValueError, match="shorter than 3 samples"):
        derive_plateau_windows(
            signal,
            trough_indices=np.array([15]),
            depth_fraction=0.2,
            minimum_samples=3,
            maximum_samples=10,
        )


def test_default_configuration_encodes_approved_ica_and_tfr_contract() -> None:
    from studies.pain_study.gradient_trough_ica.configuration import (
        CONFIG_PATH,
        GradientTroughIcaConfig,
    )

    config = GradientTroughIcaConfig.load(CONFIG_PATH, validate_paths=False)

    assert config.participants == ("sub-0014", "sub-0015")
    assert config.expected_runs == (1, 2, 3, 4, 5, 6)
    assert config.sampling_frequency_hz == 1_000.0
    assert config.highpass_hz == 40.0
    assert config.plateau_depth_fraction == 0.2
    assert config.tfr_window_s == 0.8
    assert config.tfr_smoothing_hz == 5.0
    assert config.low_temperatures_c == (44.3, 45.3)
    assert config.high_temperatures_c == (48.3, 49.3)
    assert Path(config.output_root).name == "gradient_trough_ica"


def test_select_thermal_trials_requires_ordered_complete_run() -> None:
    from studies.pain_study.gradient_trough_ica.exporter import select_thermal_trials

    events = pd.DataFrame(
        {
            "trial_type": ["other", "Trig_therm/T  1", "Trig_therm/T  1"],
            "onset": [0.0, 10.0, 20.0],
            "run_id": [1, 1, 1],
            "trial_number": [np.nan, 1, 2],
            "stimulus_temp": [np.nan, 44.3, 48.3],
            "selected_surface": [np.nan, 1, 2],
            "pain_binary_coded": [np.nan, 0, 1],
            "vas_final_coded_rating": [np.nan, 12, 80],
        }
    )

    trials = select_thermal_trials(
        events,
        run_id=1,
        thermal_marker="Trig_therm/T  1",
        expected_trial_count=2,
    )

    assert trials["trial_number"].tolist() == [1, 2]
    assert trials["stimulus_temp"].tolist() == [44.3, 48.3]


def test_concatenate_selected_intervals_preserves_only_requested_samples() -> None:
    from studies.pain_study.gradient_trough_ica.exporter import (
        concatenate_selected_intervals,
    )

    data = np.arange(2 * 12, dtype=float).reshape(2, 12)
    selected = concatenate_selected_intervals(data, [(1, 4), (8, 10)])

    np.testing.assert_array_equal(selected, data[:, [1, 2, 3, 8, 9]])


def test_concatenate_selected_intervals_rejects_overlap() -> None:
    from studies.pain_study.gradient_trough_ica.exporter import (
        concatenate_selected_intervals,
    )

    with pytest.raises(ValueError, match="strictly ordered and non-overlapping"):
        concatenate_selected_intervals(np.zeros((2, 12)), [(1, 5), (4, 8)])


def test_match_relative_event_onsets_uses_first_volume_as_clock_zero() -> None:
    from studies.pain_study.gradient_trough_ica.exporter import (
        match_relative_event_onsets,
    )

    matches = match_relative_event_onsets(
        np.array([22.849, 71.134]),
        np.array([42.704, 90.989]),
        first_volume_onset_s=19.855,
        event_volume_onset_s=0.0,
        tolerance_s=0.002,
    )

    np.testing.assert_array_equal(matches, [0, 1])


def test_match_relative_event_onsets_handles_source_clock_events() -> None:
    from studies.pain_study.gradient_trough_ica.exporter import (
        match_relative_event_onsets,
    )

    matches = match_relative_event_onsets(
        np.array([42.256, 90.541]),
        np.array([42.256, 90.541]),
        first_volume_onset_s=19.407,
        event_volume_onset_s=19.407,
        tolerance_s=0.002,
    )

    np.testing.assert_array_equal(matches, [0, 1])


def test_matlab_pipeline_preserves_filter_and_tfr_contract() -> None:
    matlab_root = Path(__file__).parents[2] / "pain_study" / "gradient_trough_ica" / "matlab"
    fit_text = (matlab_root / "fitGradientTroughIca.m").read_text(encoding="utf-8")
    initialize_text = (matlab_root / "initializeGradientTroughFieldTrip.m").read_text(
        encoding="utf-8"
    )
    tfr_text = (matlab_root / "computeGradientTroughTfr.m").read_text(encoding="utf-8")
    plot_text = (matlab_root / "plotGradientTroughResults.m").read_text(encoding="utf-8")

    assert "cfg.method = 'runica'" in fit_text
    assert "cfg.runica.extended = 1" in fit_text
    assert "cfg.numcomponent = dataRank" in fit_text
    assert "cfg.runica.pca" not in fit_text
    assert 'fullfile(fieldTripRoot, "external", "eeglab")' in initialize_text
    assert 'addpath(fieldTripEeglabRoot, "-begin")' in initialize_text
    assert 'which("runica")' in initialize_text
    assert "gradientTrough:RunicaPathConflict" in initialize_text
    assert "cfg.unmixing = componentFit.unmixing" in fit_text
    assert "gradientTrough.broadband_data" in fit_text
    assert "cfg.method = 'mtmconvol'" in tfr_text
    assert "cfg.taper = 'dpss'" in tfr_text
    assert "cfg.t_ftimwin" in tfr_text
    assert "cfg.tapsmofrq" in tfr_text
    assert "ft_freqbaseline" in tfr_text
    assert "highDb.powspctrm - lowDb.powspctrm" in tfr_text
    assert "exportgraphics" in plot_text


def test_restrict_intervals_requires_complete_stimulation_plateau_containment() -> None:
    from studies.pain_study.gradient_trough_ica.stimulation_plateau_exporter import (
        restrict_intervals_to_stimulation_plateaus,
    )

    intervals = np.array(
        [
            [127, 130, 1, 1],
            [130, 135, 1, 2],
            [200, 205, 1, 3],
            [205, 207, 1, 4],
            [335, 340, 2, 1],
        ]
    )
    selected = restrict_intervals_to_stimulation_plateaus(
        intervals,
        trial_trigger_samples=np.array([100, 300]),
        trial_numbers=np.array([1, 2]),
        temperatures_c=np.array([44.3, 48.3]),
        plateau_offsets_samples=(30, 105),
    )

    assert [(item.start, item.stop) for item in selected] == [
        (130, 135),
        (200, 205),
        (335, 340),
    ]
    assert [item.trial_number for item in selected] == [1, 1, 2]
    assert [item.temperature_c for item in selected] == [44.3, 44.3, 48.3]


def test_restrict_intervals_rejects_trial_without_eligible_trough() -> None:
    from studies.pain_study.gradient_trough_ica.stimulation_plateau_exporter import (
        restrict_intervals_to_stimulation_plateaus,
    )

    with pytest.raises(ValueError, match="Trial 2 has no fully contained trough interval"):
        restrict_intervals_to_stimulation_plateaus(
            np.array([[130, 135, 1, 1]]),
            trial_trigger_samples=np.array([100, 300]),
            trial_numbers=np.array([1, 2]),
            temperatures_c=np.array([44.3, 48.3]),
            plateau_offsets_samples=(30, 105),
        )


def test_stimulation_plateau_configuration_is_isolated() -> None:
    from studies.pain_study.gradient_trough_ica.stimulation_plateau_exporter import (
        CONFIG_PATH as STIMULATION_CONFIG_PATH,
        StimulationPlateauExportConfig,
    )

    config = StimulationPlateauExportConfig.load(
        STIMULATION_CONFIG_PATH,
        validate_paths=False,
    )

    assert config.plateau_start_s == 3.0
    assert config.plateau_stop_s == 10.5
    assert config.base.output_root.name == "gradient_trough_ica_stimulation_plateau"
    assert config.source_export_root.name == "gradient_trough_ica"
