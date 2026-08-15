"""Before-and-after spectra.

A figure comparing two datasets is only worth reading if both sides were measured
identically and actually correspond to each other. Most of these pin that, rather than the
drawing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from studies.pain_study.scripts.line_comb import psd as ps


def _raw(sfreq=250.0, seconds=120.0, line_hz=57.25, amplitude=5e-6, n_channels=4):
    import mne

    times = np.arange(int(sfreq * seconds)) / sfreq
    rng = np.random.default_rng(4)
    data = rng.normal(scale=1e-6, size=(n_channels, times.size))
    data += amplitude * np.sin(2.0 * np.pi * line_hz * times)
    info = mne.create_info([f"EEG{i:02d}" for i in range(n_channels)], sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose="ERROR"), times


class TestSettings:
    def test_the_window_defaults_to_the_removals_own(self):
        class Config:
            def get(self, key, default=None):
                return {"line_comb_removal.estimation_window_s": 32.0}.get(key, default)

        assert ps.PsdSettings.from_config(Config()).window_s == 32.0

    @pytest.mark.parametrize(
        "kwargs",
        [{"window_s": 0.0}, {"window_s": -1.0}, {"overlap": 1.0}, {"overlap": -0.1}],
    )
    def test_an_impossible_setting_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            ps.PsdSettings(**kwargs)

    def test_a_reversed_band_is_refused(self):
        with pytest.raises(ValueError, match="band_hz"):
            ps.PsdSettings(band_hz=(100.0, 1.0))


class TestSpectrum:
    def test_the_resolution_follows_the_window(self):
        raw, _ = _raw()
        settings = ps.PsdSettings(window_s=20.0)

        freqs, _ = ps.channel_median_psd(raw, settings)

        assert freqs[1] - freqs[0] == pytest.approx(1.0 / 20.0, rel=1e-6)

    def test_the_planted_line_lands_where_it_was_put(self):
        raw, _ = _raw(line_hz=57.25)

        freqs, psd = ps.channel_median_psd(raw, ps.PsdSettings(window_s=20.0))

        assert freqs[int(np.argmax(psd))] == pytest.approx(57.25, abs=0.05)

    def test_only_eeg_channels_are_measured(self):
        """An ECG trace is orders larger; averaging it in would set the level."""
        import mne

        sfreq = 250.0
        times = np.arange(int(sfreq * 120)) / sfreq
        data = np.zeros((2, times.size))
        data[0] = 1e-6 * np.sin(2.0 * np.pi * 20.0 * times)
        data[1] = 1.0 * np.sin(2.0 * np.pi * 20.0 * times)  # a volt-scale auxiliary channel
        raw = mne.io.RawArray(
            data, mne.create_info(["Cz", "ECG"], sfreq, ["eeg", "ecg"]), verbose="ERROR"
        )

        _, psd = ps.channel_median_psd(raw, ps.PsdSettings(window_s=20.0))

        # Within a decade of the EEG channel alone, not of the volt-scale one.
        assert psd.max() < 1e-3

    def test_a_recording_shorter_than_one_segment_is_refused(self):
        raw, _ = _raw(seconds=5.0)

        with pytest.raises(ValueError, match="psd.window_s"):
            ps.channel_median_psd(raw, ps.PsdSettings(window_s=54.0))

    def test_the_band_bounds_what_is_returned(self):
        raw, _ = _raw()

        freqs, _ = ps.channel_median_psd(raw, ps.PsdSettings(window_s=20.0, band_hz=(10.0, 40.0)))

        assert freqs.min() >= 10.0 and freqs.max() <= 40.0


def _bids(root: Path, *, line_amplitude: float) -> Path:
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    raw, _ = _raw(amplitude=line_amplitude)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    path = BIDSPath(subject="0001", task="rest", datatype="eeg", root=root, extension=".vhdr")
    write_raw_bids(raw, path, format="BrainVision", allow_preload=True, verbose="ERROR")
    return next(root.rglob("*_eeg.vhdr"))


class TestComparison:
    def test_a_removed_line_shows_as_a_loss_at_its_frequency(self, tmp_path):
        source = _bids(tmp_path / "src", line_amplitude=5e-6)
        cleaned = _bids(tmp_path / "clean", line_amplitude=0.0)
        settings = ps.PsdSettings(window_s=20.0)

        freqs, arms = ps.compare_recording(source, [("line-cleaned", cleaned)], settings)

        change = ps.to_db(arms["line-cleaned"]) - ps.to_db(arms["source"])
        assert change[int(np.argmin(change))] < -20.0
        assert freqs[int(np.argmin(change))] == pytest.approx(57.25, abs=0.1)

    def test_untouched_frequencies_are_unchanged(self, tmp_path):
        source = _bids(tmp_path / "src", line_amplitude=5e-6)
        cleaned = _bids(tmp_path / "clean", line_amplitude=0.0)

        freqs, arms = ps.compare_recording(
            source, [("line-cleaned", cleaned)], ps.PsdSettings(window_s=20.0)
        )

        change = ps.to_db(arms["line-cleaned"]) - ps.to_db(arms["source"])
        away = np.abs(freqs - 57.25) > 2.0
        assert np.max(np.abs(change[away])) < 1.0

    def test_a_derivative_with_a_different_channel_set_is_refused(self, tmp_path):
        import mne
        from mne_bids import BIDSPath, write_raw_bids

        source = _bids(tmp_path / "src", line_amplitude=5e-6)
        raw, _ = _raw(n_channels=3)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        path = BIDSPath(
            subject="0001", task="rest", datatype="eeg", root=tmp_path / "odd", extension=".vhdr"
        )
        write_raw_bids(raw, path, format="BrainVision", allow_preload=True, verbose="ERROR")
        other = next((tmp_path / "odd").rglob("*_eeg.vhdr"))

        with pytest.raises(ValueError, match="channel set differs"):
            ps.compare_recording(source, [("line-cleaned", other)], ps.PsdSettings(window_s=20.0))

    def test_a_derivative_of_a_different_length_is_refused(self, tmp_path):
        import mne
        from mne_bids import BIDSPath, write_raw_bids

        source = _bids(tmp_path / "src", line_amplitude=5e-6)
        raw, _ = _raw(seconds=100.0)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        path = BIDSPath(
            subject="0001", task="rest", datatype="eeg", root=tmp_path / "short", extension=".vhdr"
        )
        write_raw_bids(raw, path, format="BrainVision", allow_preload=True, verbose="ERROR")
        other = next((tmp_path / "short").rglob("*_eeg.vhdr"))

        with pytest.raises(ValueError, match="length differs"):
            ps.compare_recording(source, [("line-cleaned", other)], ps.PsdSettings(window_s=20.0))


class TestFigures:
    def test_the_cohort_figure_is_written(self, tmp_path):
        freqs = np.arange(1.0, 100.0, 0.1)
        arms = {
            "source": np.ones((3, freqs.size)) * 1e-12,
            "line-cleaned": np.ones((3, freqs.size)) * 5e-13,
        }
        path = tmp_path / "psd.png"

        ps.figure_cohort(freqs, arms, path, n_recordings=3)

        assert path.is_file() and path.stat().st_size > 0

    def test_the_per_recording_figure_is_written(self, tmp_path):
        freqs = np.arange(1.0, 100.0, 0.1)
        per_recording = {
            f"sub-000{i}_task-rest_eeg": {
                "source": np.ones(freqs.size) * 1e-12,
                "line-cleaned": np.ones(freqs.size) * 5e-13,
            }
            for i in range(4)
        }
        path = tmp_path / "per_recording.png"

        ps.figure_per_recording(freqs, per_recording, path)

        assert path.is_file() and path.stat().st_size > 0


def test_the_stage_is_reachable_from_the_cli():
    from studies.pain_study.cli import line_comb as cli

    assert "psd" in cli.MODES


def test_the_stage_refuses_before_apply_has_run(tmp_path, monkeypatch):
    import argparse

    class Config:
        def path(self, name, override=None):
            return tmp_path / name

        def get(self, key, default=None):
            return default

    monkeypatch.setattr(
        "studies.pain_study.scripts.workflow_config.load_workflow_config",
        lambda *a, **k: Config(),
    )

    with pytest.raises(FileNotFoundError, match="Run `line-comb apply` first"):
        ps.run(argparse.Namespace(config=None, bids_root=None, report_dir=None))
