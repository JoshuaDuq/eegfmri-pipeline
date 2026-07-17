from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.band_power_epoch_evolution import (
    BandPowerEpochSummary,
)

BANDS = (
    {"name": "alpha", "label": "Alpha", "frequency_hz": [8.0, 12.9]},
    {"name": "beta", "label": "Beta", "frequency_hz": [13.0, 30.0]},
)
PRIMARY_BANDS = (
    {"name": "alpha", "label": "Alpha", "frequency_hz": [8.0, 12.9]},
    {"name": "beta", "label": "Beta", "frequency_hz": [13.0, 30.0]},
    {
        "name": "gamma_low_clean",
        "label": "Low gamma",
        "frequency_hz": [30.1, 38.0],
    },
    {
        "name": "gamma_mid_clean",
        "label": "Mid gamma",
        "frequency_hz": [43.0, 56.0],
    },
    {
        "name": "gamma_high_clean",
        "label": "High gamma",
        "frequency_hz": [67.0, 77.0],
    },
)


def test_subject_timecourses_use_retained_trials_and_exclude_frontal_channels() -> None:
    from studies.pain_study.study1.figures.band_power_epoch_evolution import (
        compute_subject_band_timecourses,
    )

    times = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    frequencies = np.asarray([10.0, 20.0])
    channel_names = ("Fp1", "Cz", "Pz")
    trial_ids = np.asarray([10, 11, 12])
    power = np.ones((3, 3, 2, 5), dtype=float)

    # A large frontal-only signal must not affect the primary channel scope.
    power[:, 0, :, :] = 10_000.0
    # Trial 10 doubles after onset; trial 11 quadruples; trial 12 is not retained.
    power[0, 1:, :, 2:] = 2.0
    power[1, 1:, :, 2:] = 4.0
    power[2, 1:, :, 2:] = 100.0

    result = compute_subject_band_timecourses(
        subject_id="sub-0001",
        tfr_data=power,
        times=times,
        frequencies=frequencies,
        channel_names=channel_names,
        trial_ids=trial_ids,
        retained_trial_ids=np.asarray([10, 11]),
        band_specs=BANDS,
        baseline_window=(-1.0, -0.01),
        display_window=(-1.0, 1.0),
        excluded_channels=("Fp1", "Fp2"),
    )

    alpha = result.loc[result["band"].eq("alpha")].reset_index(drop=True)
    expected_active = np.mean([10.0 * np.log10(2.0), 10.0 * np.log10(4.0)])
    assert alpha["time_s"].tolist() == times.tolist()
    assert alpha.loc[alpha["time_s"] < 0.0, "power_db"].tolist() == [0.0, 0.0]
    assert alpha.loc[alpha["time_s"] >= 0.0, "power_db"].tolist() == [
        expected_active,
        expected_active,
        expected_active,
    ]
    assert result["n_retained_trials"].unique().tolist() == [2]
    assert result["n_channels"].unique().tolist() == [2]
    assert result["included_channels"].unique().tolist() == ["Cz,Pz"]


def test_cohort_summary_bootstraps_participants_as_complete_trajectories() -> None:
    from studies.pain_study.study1.figures.band_power_epoch_evolution import (
        build_band_power_epoch_summary,
    )

    rows = []
    for subject_index, subject_id in enumerate(("sub-0001", "sub-0002", "sub-0003")):
        for band_index, band in enumerate(("alpha", "beta")):
            for time_s in (-1.0, 0.0, 1.0):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "band": band,
                        "time_s": time_s,
                        "power_db": subject_index + band_index + time_s,
                        "n_retained_trials": 4,
                        "n_channels": 2,
                        "included_channels": "Cz,Pz",
                    }
                )
    subject_timecourses = pd.DataFrame(rows)

    first = build_band_power_epoch_summary(
        subject_timecourses,
        band_specs=BANDS,
        bootstrap_iterations=200,
        confidence_level=0.95,
        seed=42,
    )
    second = build_band_power_epoch_summary(
        subject_timecourses,
        band_specs=BANDS,
        bootstrap_iterations=200,
        confidence_level=0.95,
        seed=42,
    )

    pd.testing.assert_frame_equal(first.cohort_timecourses, second.cohort_timecourses)
    alpha = first.cohort_timecourses.loc[first.cohort_timecourses["band"].eq("alpha")]
    assert alpha["mean_power_db"].tolist() == [0.0, 1.0, 2.0]
    assert alpha["n_subjects"].tolist() == [3, 3, 3]
    assert first.subject_ids == ("sub-0001", "sub-0002", "sub-0003")
    assert first.bands == ("alpha", "beta")
    assert first.times.tolist() == [-1.0, 0.0, 1.0]


def test_build_band_power_epoch_figure_aligns_five_bands_below_protocol_bar() -> None:
    from studies.pain_study.study1.figures.band_power_epoch_evolution_plot import (
        build_band_power_epoch_figure,
    )

    figure = build_band_power_epoch_figure(_publication_summary(), load_study1_config())

    try:
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 145.0 / 25.4))
        protocol_axis, *band_axes = figure.axes
        assert protocol_axis.get_label() == "protocol"
        assert [axis.get_label() for axis in band_axes] == [
            "band:alpha",
            "band:beta",
            "band:gamma_low_clean",
            "band:gamma_mid_clean",
            "band:gamma_high_clean",
        ]
        assert all(np.allclose(axis.get_xlim(), (-5.0, 14.5)) for axis in figure.axes)
        assert all(
            axis.get_position().x0 == pytest.approx(band_axes[0].get_position().x0)
            and axis.get_position().width == pytest.approx(band_axes[0].get_position().width)
            for axis in band_axes
        )
        assert [text.get_text() for text in protocol_axis.texts] == [
            "Baseline",
            "Ramp-up",
            "Plateau",
            "Ramp-down",
            "Post-stimulus",
        ]
        assert len(protocol_axis.patches) == 5
        assert all(len(axis.patches) == 0 for axis in band_axes)
        assert [axis.get_xlabel() for axis in band_axes] == [
            "",
            "",
            "",
            "",
            "Time from stimulus onset (s)",
        ]
        assert all(axis.get_ylim() == pytest.approx((-2.0, 2.0)) for axis in band_axes)
    finally:
        plt.close(figure)


def test_band_power_epoch_figure_reports_estimand_and_sample_metadata() -> None:
    from studies.pain_study.study1.figures.band_power_epoch_evolution_plot import (
        build_band_power_epoch_figure,
    )

    figure = build_band_power_epoch_figure(_publication_summary(), load_study1_config())

    try:
        figure_text = [text.get_text() for text in figure.texts]
        assert "Global EEG band-power change during thermal stimulation" in figure_text
        assert (
            "Retained-trial means · n=4 participants · trials/participant: "
            "median 30, range 30–30 · 62 EEG channels"
        ) in figure_text
        assert len(figure.legends) == 1
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Participant retained-trial mean",
            "Equally weighted cohort mean",
            "Pointwise 95% participant-bootstrap CI",
        ]
        assert [text.get_text() for text in figure.axes[1].texts] == [
            "Alpha",
            "8–12.9 Hz",
        ]
        assert all(
            text.get_position()[1] >= 0.9 and text.get_bbox_patch() is not None
            for axis in figure.axes[1:]
            for text in axis.texts
        )
    finally:
        plt.close(figure)


def test_loader_uses_only_primary_target_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.band_power_epoch_evolution as module

    targets = pd.DataFrame(
        {
            "subject_id": ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
            "task": ["thermalactive"] * 4,
            "run": [1, 1, 1, 1],
            "within_run_trial": [1, 2, 1, 2],
        }
    )
    events = pd.DataFrame(
        {
            "run_id": [1, 1, 1],
            "trial_number": [1, 2, 3],
            "trial_id": [10, 11, 12],
        }
    )

    class FakeEpochs:
        def __init__(self, indices: np.ndarray | None = None):
            self.indices = np.arange(3) if indices is None else np.asarray(indices)

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, indices: np.ndarray) -> "FakeEpochs":
            return FakeEpochs(self.indices[np.asarray(indices)])

    def fake_tfr(epochs, *_args, **_kwargs):
        assert epochs.indices.tolist() == [0, 1]
        data = np.ones((2, 3, 5, 7), dtype=float)
        data[:, 1:, :, [1, 3]] = 9.0
        data[:, 1:, :, 5:] = 5.0
        return SimpleNamespace(
            data=data,
            times=np.asarray([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0]),
            freqs=np.asarray([10.0, 20.0, 35.0, 50.0, 70.0]),
            ch_names=["Fp1", "Cz", "Pz"],
        )

    monkeypatch.setattr(module, "load_primary_target_table", lambda _config: targets)
    monkeypatch.setattr(
        module,
        "load_epochs_for_analysis",
        lambda **_kwargs: (FakeEpochs(), events.copy()),
    )
    monkeypatch.setattr(module, "compute_tfr_morlet", fake_tfr)

    config = load_study1_config()
    config["study1"]["figures"]["validity"]["bootstrap"]["iterations"] = 20
    config["study1"]["figures"]["band_power_epoch_evolution"]["temporal_stride"] = 2
    summary = module.load_band_power_epoch_summary(task="thermalactive", config=config)

    assert summary.subject_ids == ("sub-0001", "sub-0002")
    assert summary.bands == tuple(str(spec["name"]) for spec in PRIMARY_BANDS)
    assert summary.subject_timecourses["n_retained_trials"].unique().tolist() == [2]
    assert summary.subject_timecourses["n_channels"].unique().tolist() == [2]
    assert summary.times.tolist() == [-5.0, -3.0, -1.0, 1.0]
    alpha = summary.subject_timecourses.loc[
        summary.subject_timecourses["band"].eq("alpha")
        & summary.subject_timecourses["time_s"].eq(1.0)
    ]
    full_resolution_baseline = np.mean([1.0, 9.0, 1.0, 9.0, 1.0])
    assert alpha["power_db"].tolist() == pytest.approx(
        [10.0 * np.log10(5.0 / full_resolution_baseline)] * 2
    )


def test_band_power_epoch_writer_creates_svg_and_two_audit_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_band_power_epoch_evolution as module

    summary = _publication_summary()
    monkeypatch.setattr(module, "load_band_power_epoch_summary", lambda **kwargs: summary)
    outputs = [
        module.write_band_power_epoch_evolution(
            task="thermalactive",
            config=load_study1_config(),
            output_path=tmp_path / directory / "band_power_epoch_evolution.svg",
        )
        for directory in ("first", "second")
    ]

    first, second = outputs
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    assert float(root.attrib["width"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        183.0,
        abs=0.01,
    )
    assert float(root.attrib["height"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        145.0,
        abs=0.01,
    )
    assert {path.name for path in first.svg.parent.iterdir()} == {
        "band_power_epoch_evolution.svg",
        "band_power_epoch_evolution_by_subject.tsv",
        "band_power_epoch_evolution_by_subject.parquet",
        "band_power_epoch_evolution_summary.tsv",
        "band_power_epoch_evolution_summary.parquet",
    }
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.subject_parquet),
        summary.subject_timecourses,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.summary_parquet),
        summary.cohort_timecourses,
    )


def _publication_summary() -> BandPowerEpochSummary:
    times = np.asarray([-5.0, 0.0, 3.0, 10.5, 12.5, 14.5])
    subjects = tuple(f"sub-{index:04d}" for index in range(4))
    bands = tuple(str(spec["name"]) for spec in PRIMARY_BANDS)
    subject_rows = []
    cohort_rows = []
    for band_index, band in enumerate(bands):
        for subject_index, subject_id in enumerate(subjects):
            for time_s in times:
                subject_rows.append(
                    {
                        "subject_id": subject_id,
                        "band": band,
                        "time_s": time_s,
                        "power_db": 0.2 * band_index + 0.1 * subject_index + time_s / 20.0,
                        "n_retained_trials": 30,
                        "n_channels": 62,
                        "included_channels": "Cz,Pz",
                    }
                )
        for time_s in times:
            mean = 0.2 * band_index + 0.15 + time_s / 20.0
            cohort_rows.append(
                {
                    "band": band,
                    "time_s": time_s,
                    "mean_power_db": mean,
                    "ci_low": mean - 0.1,
                    "ci_high": mean + 0.1,
                    "n_subjects": len(subjects),
                }
            )
    return BandPowerEpochSummary(
        subject_timecourses=pd.DataFrame(subject_rows),
        cohort_timecourses=pd.DataFrame(cohort_rows),
        subject_ids=subjects,
        bands=bands,
        times=times,
    )
