from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import signal


def test_temperature_model_recovers_adjusted_slope_and_audits_missing_metadata() -> None:
    from studies.pain_study.study1.figures.band_time_frequency_model import (
        build_temperature_model,
    )

    temperature_levels = np.asarray([44.3, 45.3, 46.3, 47.3, 48.3, 49.3])
    rng = np.random.default_rng(17)
    temperatures = np.concatenate([rng.permutation(temperature_levels) for _ in range(4)])
    run = np.repeat([1, 2, 3, 4], 6)
    surface = np.tile([1, 2, 3, 1, 3, 2], 4)
    trial_number = np.tile(np.arange(1, 7), 4)
    events = pd.DataFrame(
        {
            "epoch_index": np.arange(25),
            "stimulus_temp": np.append(temperatures, np.nan),
            "run_id": np.append(run, 4),
            "selected_surface": np.append(surface, 2),
            "trial_number": np.append(trial_number, np.nan),
        }
    )

    model = build_temperature_model(
        events,
        n_epochs=25,
        temperatures=(44.3, 45.3, 46.3, 47.3, 48.3, 49.3),
    )
    nuisance_signal = 0.4 * run + 0.2 * surface + 0.1 * trial_number
    outcome = 2.5 * temperatures + nuisance_signal

    assert model.epoch_indices.tolist() == list(range(24))
    assert model.excluded_epoch_indices.tolist() == [24]
    assert model.n_clean_epochs == 25
    assert model.n_model_trials == 24
    assert model.temperature_weights @ outcome == pytest.approx(2.5)
    assert model.design_rank == model.design_columns
    assert np.isfinite(model.condition_number)


def test_variable_cycle_hanning_power_matches_direct_tapered_fourier_estimate() -> None:
    from studies.pain_study.study1.figures.band_time_frequency_model import (
        compute_variable_cycle_hanning_power,
    )

    sampling_frequency_hz = 100.0
    epoch_start_s = -2.0
    samples = np.arange(400, dtype=float) / sampling_frequency_hz
    data = (np.sin(2.0 * np.pi * 10.0 * samples) + 0.25 * np.sin(2.0 * np.pi * 20.0 * samples))[
        None, None, :
    ]

    result = compute_variable_cycle_hanning_power(
        data,
        sampling_frequency_hz=sampling_frequency_hz,
        epoch_start_s=epoch_start_s,
        frequencies=np.asarray([10.0, 20.0]),
        n_cycles=4.0,
        time_step_s=0.05,
        time_window=(-1.0, 1.0),
    )

    center_index = np.flatnonzero(np.isclose(result.times, 0.0)).item()
    demeaned = data[0, 0] - data[0, 0].mean()
    for frequency_index, frequency in enumerate(result.frequencies):
        window_samples = int(round(4.0 / frequency * sampling_frequency_hz))
        half = window_samples // 2
        sample_index = int(round((0.0 - epoch_start_s) * sampling_frequency_hz))
        segment = demeaned[sample_index - half : sample_index - half + window_samples]
        taper = signal.windows.hann(window_samples, sym=True)
        relative_samples = np.arange(window_samples) - (window_samples - 1) / 2.0
        wave = taper * np.exp(-2j * np.pi * frequency * relative_samples / sampling_frequency_hz)
        expected_power = np.abs(segment @ wave / np.linalg.norm(taper)) ** 2
        assert result.power[0, 0, frequency_index, center_index] == pytest.approx(expected_power)
    assert result.window_durations_s.tolist() == pytest.approx([0.4, 0.2])
    assert np.diff(result.times) == pytest.approx(0.05)


def test_temperature_slope_batch_normalizes_each_trial_channel_before_regression() -> None:
    from studies.pain_study.study1.figures.band_time_frequency_model import (
        summarize_temperature_slope_batch,
    )

    times = np.asarray([-0.75, -0.5, 0.0, 0.25])
    frequencies = np.asarray([10.0, 20.0])
    power = np.ones((2, 2, 2, 4), dtype=float)
    power[0, 0, :, 2:] = 2.0
    power[0, 1, :, 2:] = 8.0
    power[1, 0, :, 2:] = 4.0
    power[1, 1, :, 2:] = 16.0

    result = summarize_temperature_slope_batch(
        power=power,
        times=times,
        frequencies=frequencies,
        window_durations_s=np.asarray([0.5, 0.25]),
        channel_names=("Fp1", "Cz"),
        baseline_window=(-1.0, -0.01),
        display_window=(-1.0, 0.5),
        temperature_weights=np.asarray([-0.5, 0.5]),
    )

    expected = 0.5 * 10.0 * np.log10(2.0)
    assert result.slope_sum[:, :2] == pytest.approx(0.0)
    assert result.slope_sum[:, 2:] == pytest.approx(expected)
    assert result.channel_names == ("Fp1", "Cz")
    assert result.n_trials == 2


def test_discover_latest_clean_epochs_selects_newest_file_per_subject(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.band_time_frequency import (
        discover_latest_clean_epochs,
    )

    older = _clean_epoch_path(tmp_path / "older", "sub-0001")
    newer = _clean_epoch_path(tmp_path / "newer", "sub-0001")
    second_subject = _clean_epoch_path(tmp_path / "current", "sub-0002")
    excluded = _clean_epoch_path(tmp_path / "current", "sub-0006")
    pilot = _clean_epoch_path(tmp_path / "current", "sub-pilot001")
    _touch_at(older, 1_000)
    _touch_at(newer, 2_000)
    _touch_at(second_subject, 3_000)
    _touch_at(excluded, 4_000)
    _touch_at(pilot, 5_000)

    sources = discover_latest_clean_epochs(
        tmp_path,
        task="thermalactive",
        excluded_subjects=("sub-0006",),
    )

    assert [source.subject_id for source in sources] == ["sub-0001", "sub-0002"]
    assert [source.path for source in sources] == [newer, second_subject]
    assert [source.modified_time_ns for source in sources] == [2_000, 3_000]


def test_discover_latest_clean_epochs_rejects_ambiguous_newest_files(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.band_time_frequency import (
        discover_latest_clean_epochs,
    )

    first = _clean_epoch_path(tmp_path / "first", "sub-0001")
    second = _clean_epoch_path(tmp_path / "second", "sub-0001")
    _touch_at(first, 2_000)
    _touch_at(second, 2_000)

    with pytest.raises(ValueError, match="Ambiguous newest final-clean epochs"):
        discover_latest_clean_epochs(tmp_path, task="thermalactive")


def test_build_band_tfr_summary_equally_weights_complete_participant_maps() -> None:
    from studies.pain_study.study1.figures.band_time_frequency import (
        build_band_tfr_summary,
    )

    rows = []
    for subject_index, subject_id in enumerate(("sub-0001", "sub-0002", "sub-0003")):
        for band_index, band in enumerate(("alpha", "beta")):
            for frequency_hz in (10.0, 11.0):
                for time_s in (-1.0, 0.0):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "band": band,
                            "frequency_hz": frequency_hz,
                            "time_s": time_s,
                            "temperature_slope_db_per_c": (
                                subject_index + band_index + frequency_hz / 10.0 + time_s
                            ),
                        }
                    )
    subject_maps = pd.DataFrame(rows)
    source_audit = _source_audit(
        subjects=("sub-0001", "sub-0002", "sub-0003"),
        clean_trials=(20, 21, 22),
        model_trials=(20, 21, 22),
        channel_lists=("Fp1,Cz", "Fp1,Cz", "Fp1,Cz"),
    )

    summary = build_band_tfr_summary(
        subject_maps,
        source_audit,
        band_names=("alpha", "beta"),
    )

    alpha = summary.cohort_maps.loc[summary.cohort_maps["band"].eq("alpha")]
    assert alpha["mean_temperature_slope_db_per_c"].tolist() == pytest.approx([1.0, 2.0, 1.1, 2.1])
    assert alpha["n_subjects"].tolist() == [3, 3, 3, 3]
    assert summary.subject_ids == ("sub-0001", "sub-0002", "sub-0003")
    assert summary.bands == ("alpha", "beta")
    assert summary.source_audit.equals(source_audit)


def test_build_band_tfr_summary_rejects_incomplete_participant_grid() -> None:
    from studies.pain_study.study1.figures.band_time_frequency import (
        build_band_tfr_summary,
    )

    subject_maps, source_audit = _summary_inputs()
    subject_maps = subject_maps.iloc[:-1].copy()

    with pytest.raises(ValueError, match="complete grid"):
        build_band_tfr_summary(
            subject_maps,
            source_audit,
            band_names=("alpha",),
        )


def test_build_band_tfr_summary_preserves_each_participants_available_channels() -> None:
    from studies.pain_study.study1.figures.band_time_frequency import (
        build_band_tfr_summary,
    )

    subject_maps, source_audit = _summary_inputs()
    source_audit.loc[source_audit["subject_id"].eq("sub-0002"), "included_channels"] = "Cz,Pz"

    summary = build_band_tfr_summary(
        subject_maps,
        source_audit,
        band_names=("alpha",),
    )

    assert summary.source_audit["included_channels"].tolist() == ["Fp1,Cz", "Cz,Pz"]


def test_load_band_tfr_summary_audits_modelled_trials_and_all_eeg_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.band_time_frequency as module
    from studies.pain_study.study1.config.loader import load_study1_config

    first = _clean_epoch_path(tmp_path, "sub-0001")
    second = _clean_epoch_path(tmp_path, "sub-0002")
    _touch_at(first, 1_000)
    _touch_at(second, 2_000)

    class FakeEpochs:
        def __init__(self, factor: float):
            self.factor = factor
            self.info = {"sfreq": 500.0}
            self.times = np.asarray([-7.0, 15.0])
            self.ch_names = ["Fp1", "Cz"]

        def __len__(self) -> int:
            return 3

    def read_epochs(path: Path, **_kwargs) -> FakeEpochs:
        factor = 2.0 if "sub-0001" in path.name else 4.0
        return FakeEpochs(factor)

    model = SimpleNamespace(
        epoch_indices=np.asarray([0, 1]),
        excluded_epoch_indices=np.asarray([2]),
        temperature_weights=np.asarray([-0.5, 0.5]),
        n_clean_epochs=3,
        n_model_trials=2,
        design_rank=2,
        design_columns=2,
        condition_number=1.0,
    )

    def compute_subject(*, subject_id, epochs, event_path, model, frequencies, **_kwargs):
        factor = 2.0 if subject_id == "sub-0001" else 4.0
        return module.SubjectSlopeMap(
            slope=np.full((len(frequencies), 3), factor),
            times=np.asarray([-5.0, 0.0, 14.5]),
            frequencies=frequencies,
            channel_names=("Fp1", "Cz"),
            analysis_sampling_frequency_hz=200.0,
            model=model,
            event_path=event_path,
        )

    monkeypatch.setattr(module.mne, "read_epochs", read_epochs)
    monkeypatch.setattr(
        module,
        "_load_matching_events",
        lambda path, **_kwargs: (path.with_suffix(".tsv"), pd.DataFrame()),
    )
    monkeypatch.setattr(module, "build_temperature_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(module, "_compute_subject_slope", compute_subject)
    config = load_study1_config()
    config["study1"]["figures"]["band_time_frequency"] = {
        "frequency_step_hz": 1.0,
        "n_cycles": 7.0,
        "time_step_s": 0.05,
        "analysis_sampling_frequency_hz": 200.0,
        "trial_batch_size": 2,
        "display_window_s": [-5.0, 14.5],
        "excluded_subjects": [],
    }

    summary = module.load_band_tfr_summary(
        task="thermalactive",
        derivative_root=tmp_path,
        config=config,
    )

    assert summary.subject_ids == ("sub-0001", "sub-0002")
    assert summary.bands == (
        "alpha",
        "beta",
        "gamma_low_clean",
        "gamma_mid_clean",
        "gamma_high_clean",
    )
    assert summary.subject_maps["time_s"].unique().tolist() == [-5.0, 0.0, 14.5]
    assert summary.source_audit["n_clean_trials"].tolist() == [3, 3]
    assert summary.source_audit["n_model_trials"].tolist() == [2, 2]
    assert summary.source_audit["n_excluded_missing_metadata"].tolist() == [1, 1]
    assert summary.source_audit["n_eeg_channels"].tolist() == [2, 2]
    assert summary.source_audit["included_channels"].tolist() == ["Cz,Fp1", "Cz,Fp1"]
    active_alpha = summary.cohort_maps.loc[
        summary.cohort_maps["band"].eq("alpha") & summary.cohort_maps["time_s"].eq(0.0),
        "mean_temperature_slope_db_per_c",
    ]
    assert active_alpha.tolist() == pytest.approx([3.0] * 5)


def test_band_tfr_figures_use_separate_scales_and_precise_estimand_headers() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.band_time_frequency_plot import (
        build_cohort_band_tfr_figure,
        build_participant_band_tfr_figure,
        cohort_band_color_limit,
        participant_band_color_limit,
    )

    summary = _publication_summary()
    config = load_study1_config()
    config["study1"]["figures"]["band_time_frequency"] = {
        "dimensions_mm": {"width": 183.0, "height": 105.0},
        "display_window_s": [-5.0, 14.5],
        "n_cycles": 7.0,
        "time_step_s": 0.05,
        "color_percentile": 99.0,
    }
    participant_limit = participant_band_color_limit(summary, "alpha", percentile=99.0)
    cohort_limit = cohort_band_color_limit(summary, "alpha", percentile=99.0)

    participant = build_participant_band_tfr_figure(
        summary,
        band="alpha",
        subject_id="sub-0001",
        color_limit=participant_limit,
        config=config,
    )
    cohort = build_cohort_band_tfr_figure(
        summary,
        band="alpha",
        color_limit=cohort_limit,
        config=config,
    )

    try:
        assert np.allclose(participant.get_size_inches(), (183.0 / 25.4, 105.0 / 25.4))
        assert np.allclose(cohort.get_size_inches(), participant.get_size_inches())
        participant_text = [text.get_text() for text in participant.texts]
        cohort_text = [text.get_text() for text in cohort.texts]
        assert "Alpha-band EEG temperature modulation · sub-0001" in participant_text
        assert (
            "Adjusted OLS slope (dB/°C) · run + thermode surface + within-run trial order"
        ) in participant_text
        assert (
            "All EEG channels retained through normalization · 7-cycle Hanning "
            "mtmconvol · 0.05 s step"
        ) in participant_text
        assert (
            "19/20 modelled/clean trials · 1 excluded for missing metadata · "
            "2 EEG channels · source modified 2026-07-17 12:00 UTC"
        ) in participant_text
        assert "Alpha-band EEG temperature modulation · cohort" in cohort_text
        assert (
            "Equal-weight participant slopes · n=2 · modelled trials: range 19–20 · "
            "missing-metadata exclusions: total 2 · "
            "EEG channels/participant: range 2–3"
        ) in cohort_text
        assert participant.axes[0].get_label() == "band-tfr"
        assert participant.axes[0].get_xlabel() == "Time from stimulus onset (s)"
        assert participant.axes[0].get_ylabel() == "Frequency (Hz)"
        assert participant.axes[1].get_ylabel() == "Temperature slope (dB/°C)"
        assert [text.get_text() for text in participant.axes[0].texts] == [
            "Baseline",
            "Ramp-up",
            "Plateau",
            "Ramp-down",
            "Post-stimulus",
        ]
        assert participant.axes[0].collections[0].norm.vmin == pytest.approx(-participant_limit)
        assert participant.axes[0].collections[0].norm.vmax == pytest.approx(participant_limit)
        assert cohort.axes[0].collections[0].norm.vmin == pytest.approx(-cohort_limit)
        assert cohort.axes[0].collections[0].norm.vmax == pytest.approx(cohort_limit)
        assert cohort_limit < participant_limit
    finally:
        plt.close(participant)
        plt.close(cohort)


def test_band_tfr_writer_creates_participant_cohort_and_audit_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_band_time_frequency as module
    from studies.pain_study.study1.config.loader import load_study1_config

    summary = _publication_summary()
    monkeypatch.setattr(module, "load_band_tfr_summary", lambda **_kwargs: summary)
    config = load_study1_config()
    config["study1"]["figures"]["band_time_frequency"] = {
        "dimensions_mm": {"width": 183.0, "height": 105.0},
        "display_window_s": [-5.0, 14.5],
        "n_cycles": 7.0,
        "time_step_s": 0.05,
        "color_percentile": 99.0,
    }

    first = module.write_band_time_frequency(
        task="thermalactive",
        derivative_root=tmp_path,
        config=config,
        output_dir=tmp_path / "first",
    )
    second = module.write_band_time_frequency(
        task="thermalactive",
        derivative_root=tmp_path,
        config=config,
        output_dir=tmp_path / "second",
    )

    assert [path.name for path in first.participant_svgs] == [
        "band_time_frequency_alpha_sub-0001.svg",
        "band_time_frequency_alpha_sub-0002.svg",
    ]
    assert [path.name for path in first.cohort_svgs] == ["band_time_frequency_alpha_cohort.svg"]
    assert [path.read_bytes() for path in first.participant_svgs + first.cohort_svgs] == [
        path.read_bytes() for path in second.participant_svgs + second.cohort_svgs
    ]
    assert sorted(path.name for path in first.output_dir.iterdir()) == [
        "band_time_frequency_alpha_cohort.svg",
        "band_time_frequency_alpha_sub-0001.svg",
        "band_time_frequency_alpha_sub-0002.svg",
        "band_time_frequency_by_subject.parquet",
        "band_time_frequency_by_subject.tsv",
        "band_time_frequency_sources.parquet",
        "band_time_frequency_sources.tsv",
        "band_time_frequency_summary.parquet",
        "band_time_frequency_summary.tsv",
    ]
    root = ElementTree.parse(first.cohort_svgs[0]).getroot()
    assert float(root.attrib["width"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        183.0,
        abs=0.01,
    )
    assert float(root.attrib["height"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        105.0,
        abs=0.01,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.subject_parquet),
        summary.subject_maps,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.cohort_parquet),
        summary.cohort_maps,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.sources_parquet),
        summary.source_audit,
    )


def test_band_tfr_main_passes_explicit_roots_and_prints_every_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import studies.pain_study.study1.figures.plot_band_time_frequency as module
    from studies.pain_study.study1.config.loader import load_study1_config

    participant = tmp_path / "band_time_frequency_alpha_sub-0001.svg"
    cohort = tmp_path / "band_time_frequency_alpha_cohort.svg"
    table = tmp_path / "audit.tsv"
    outputs = module.BandTfrFigurePaths(
        output_dir=tmp_path,
        participant_svgs=(participant,),
        cohort_svgs=(cohort,),
        subject_tsv=table,
        subject_parquet=table.with_suffix(".parquet"),
        cohort_tsv=table,
        cohort_parquet=table.with_suffix(".parquet"),
        sources_tsv=table,
        sources_parquet=table.with_suffix(".parquet"),
    )
    captured = {}
    monkeypatch.setattr(module, "load_config", lambda _path: load_study1_config())
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *_args: None)

    def write(**kwargs):
        captured.update(kwargs)
        return outputs

    monkeypatch.setattr(module, "write_band_time_frequency", write)

    result = module.main(
        [
            "--config",
            "pipeline.yaml",
            "--study1-config",
            "study1.yaml",
            "--task",
            "thermalactive",
            "--derivative-root",
            str(tmp_path / "derivatives"),
            "--output-dir",
            str(tmp_path / "figures"),
        ]
    )

    assert result == outputs
    assert captured["task"] == "thermalactive"
    assert captured["derivative_root"] == tmp_path / "derivatives"
    assert captured["output_dir"] == tmp_path / "figures"
    assert capsys.readouterr().out.splitlines() == [str(participant), str(cohort)]


def _summary_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_maps = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "band": "alpha",
                "frequency_hz": frequency_hz,
                "time_s": time_s,
                "temperature_slope_db_per_c": subject_index + frequency_hz + time_s,
            }
            for subject_index, subject_id in enumerate(("sub-0001", "sub-0002"))
            for frequency_hz in (10.0, 11.0)
            for time_s in (-1.0, 0.0)
        ]
    )
    source_audit = _source_audit(
        subjects=("sub-0001", "sub-0002"),
        clean_trials=(20, 21),
        model_trials=(20, 21),
        channel_lists=("Fp1,Cz", "Fp1,Cz"),
    )
    return subject_maps, source_audit


def _publication_summary():
    from studies.pain_study.study1.figures.band_time_frequency import (
        build_band_tfr_summary,
    )

    times = (-5.0, 0.0, 3.0, 10.5, 12.5, 14.5)
    subject_maps = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "band": "alpha",
                "frequency_hz": frequency_hz,
                "time_s": time_s,
                "temperature_slope_db_per_c": (
                    (subject_index + 1) * (time_s + 5.1) / 100.0 + frequency_hz / 500.0
                ),
            }
            for subject_index, subject_id in enumerate(("sub-0001", "sub-0002"))
            for frequency_hz in (8.0, 10.0, 12.9)
            for time_s in times
        ]
    )
    source_audit = _source_audit(
        subjects=("sub-0001", "sub-0002"),
        clean_trials=(20, 21),
        model_trials=(19, 20),
        channel_lists=("Cz,Fp1", "Cz,Fp1,Pz"),
        timestamps=("2026-07-17T12:00:00+00:00", "2026-07-17T12:01:00+00:00"),
    )
    return build_band_tfr_summary(
        subject_maps,
        source_audit,
        band_names=("alpha",),
    )


def _source_audit(
    *,
    subjects: tuple[str, ...],
    clean_trials: tuple[int, ...],
    model_trials: tuple[int, ...],
    channel_lists: tuple[str, ...],
    timestamps: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    count = len(subjects)
    if timestamps is None:
        timestamps = tuple(f"2026-07-17T12:{index:02d}:00+00:00" for index in range(count))
    excluded = tuple(clean - model for clean, model in zip(clean_trials, model_trials, strict=True))
    return pd.DataFrame(
        {
            "subject_id": subjects,
            "source_file": tuple(f"{subject}.fif" for subject in subjects),
            "event_file": tuple(f"{subject}.tsv" for subject in subjects),
            "modified_time_ns": tuple(range(1, count + 1)),
            "modified_time_utc": timestamps,
            "n_clean_trials": clean_trials,
            "n_model_trials": model_trials,
            "n_excluded_missing_metadata": excluded,
            "design_rank": (8,) * count,
            "design_columns": (8,) * count,
            "design_condition_number": (2.0,) * count,
            "n_eeg_channels": tuple(len(channels.split(",")) for channels in channel_lists),
            "included_channels": channel_lists,
            "source_sampling_frequency_hz": (500.0,) * count,
            "analysis_sampling_frequency_hz": (200.0,) * count,
        }
    )


def _clean_epoch_path(root: Path, subject_id: str) -> Path:
    return root / subject_id / "eeg" / f"{subject_id}_task-thermalactive_proc-clean_epo.fif"


def _touch_at(path: Path, modified_time_ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, ns=(modified_time_ns, modified_time_ns))
