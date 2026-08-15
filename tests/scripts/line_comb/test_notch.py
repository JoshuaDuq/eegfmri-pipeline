"""The wide-notch stage.

What it has to get right is narrow: convert measured edges into a filter, take out the
band, leave everything else alone, and write a BIDS derivative whose sidecars are
byte-identical. There is no estimator here to certify, so the tests are about the contract
rather than about a fit.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from studies.pain_study.scripts.line_comb import notch as nt


class _Config:
    """The two accessors NotchSettings.from_config uses."""

    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


def _config(**overrides):
    values = {
        "frequency_bands": {
            "delta": [1.0, 3.9],
            "theta": [4.0, 7.9],
            "alpha": [8.0, 12.9],
            "beta": [13.0, 30.0],
            "gamma": [30.1, 80.0],
            # Carved out when the harmonics were thought unremovable; must be ignored.
            "gamma_mid_clean": [43.0, 56.0],
        }
    }
    values.update(overrides)
    return _Config(values)


class TestNotchBand:
    def test_edges_become_a_centre_and_a_width(self):
        band = nt.NotchBand(56.8, 57.7)

        assert band.centre_hz == pytest.approx(57.25)
        assert band.width_hz == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "low, high",
        [(57.7, 56.8), (57.0, 57.0), (-1.0, 5.0), (0.0, 5.0), (np.nan, 5.0)],
    )
    def test_a_band_that_is_not_a_span_is_refused(self, low, high):
        with pytest.raises(ValueError):
            nt.NotchBand(low, high)

    def test_overlap_is_zero_for_disjoint_spans(self):
        band = nt.NotchBand(56.8, 57.7)

        assert band.overlap_hz(30.1, 80.0) == pytest.approx(0.9)
        assert band.overlap_hz(1.0, 3.9) == 0.0


class TestSettings:
    def test_the_default_band_is_the_measured_cluster(self):
        settings = nt.NotchSettings.from_config(_config())

        assert [(b.low_hz, b.high_hz) for b in settings.bands] == [(56.8, 57.7)]

    def test_configured_bands_replace_the_default(self):
        settings = nt.NotchSettings.from_config(_config(notch_bands=[[20.0, 21.0], [40.0, 41.0]]))

        assert [b.centre_hz for b in settings.bands] == [20.5, 40.5]

    def test_only_the_canonical_bands_are_reported_against(self):
        """The *_clean variants no longer describe anything and must not appear."""
        settings = nt.NotchSettings.from_config(_config())

        assert [name for name, _, _ in settings.analysed_bands] == list(nt.CANONICAL_BANDS)

    def test_overlapping_bands_are_refused(self):
        with pytest.raises(ValueError, match="must not overlap"):
            nt.NotchSettings.from_config(_config(notch_bands=[[56.0, 57.5], [57.0, 58.0]]))

    @pytest.mark.parametrize("entry", [[57.0], [57.0, 58.0, 59.0], "57-58", {"low": 57.0}])
    def test_a_band_that_is_not_a_pair_is_refused(self, entry):
        with pytest.raises(ValueError):
            nt.NotchSettings.from_config(_config(notch_bands=[entry]))

    def test_an_empty_band_list_is_refused(self):
        with pytest.raises(ValueError, match="non-empty"):
            nt.NotchSettings.from_config(_config(notch_bands=[]))


def _raw_with_lines(sfreq=500.0, seconds=80.0):
    """Noise plus a tone inside the notch band and one well outside it."""
    import mne

    times = np.arange(int(sfreq * seconds)) / sfreq
    rng = np.random.default_rng(11)
    data = rng.normal(scale=1e-6, size=(3, times.size))
    data += 5e-6 * np.sin(2.0 * np.pi * 57.25 * times)  # inside 56.8-57.7
    data += 5e-6 * np.sin(2.0 * np.pi * 45.0 * times)  # far outside
    info = mne.create_info(["Cz", "Pz", "Oz"], sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose="ERROR"), times


def _amplitude(values, frequency_hz, times):
    basis = np.exp(-2j * np.pi * frequency_hz * times)
    return 2.0 * abs(np.vdot(basis, values)) / values.size


class TestFiltering:
    def test_the_band_goes_and_the_rest_stays(self):
        raw, times = _raw_with_lines()
        settings = nt.NotchSettings.from_config(_config())

        filtered = nt.notch_eeg(raw, settings)

        before = raw.get_data()[0]
        after = filtered.get_data()[0]
        assert _amplitude(after, 57.25, times) < 0.05 * _amplitude(before, 57.25, times)
        assert _amplitude(after, 45.0, times) == pytest.approx(
            _amplitude(before, 45.0, times), rel=0.02
        )

    def test_non_eeg_channels_are_untouched(self):
        import mne

        sfreq = 500.0
        times = np.arange(int(sfreq * 80)) / sfreq
        data = np.tile(5e-6 * np.sin(2.0 * np.pi * 57.25 * times), (2, 1))
        info = mne.create_info(["Cz", "ECG"], sfreq, ["eeg", "ecg"])
        raw = mne.io.RawArray(data, info, verbose="ERROR")

        filtered = nt.notch_eeg(raw, nt.NotchSettings.from_config(_config()))

        assert not np.allclose(filtered.get_data()[0], raw.get_data()[0])
        assert np.array_equal(filtered.get_data()[1], raw.get_data()[1])

    def test_a_band_reaching_nyquist_is_refused(self):
        raw, _ = _raw_with_lines(sfreq=120.0)
        settings = nt.NotchSettings.from_config(_config(notch_bands=[[58.0, 61.0]]))

        with pytest.raises(ValueError, match="Nyquist"):
            nt.notch_eeg(raw, settings)

    def test_a_recording_without_eeg_channels_is_refused(self):
        import mne

        raw = mne.io.RawArray(
            np.zeros((1, 1000)),
            mne.create_info(["ECG"], 500.0, "ecg"),
            verbose="ERROR",
        )

        with pytest.raises(ValueError, match="at least one EEG channel"):
            nt.notch_eeg(raw, nt.NotchSettings.from_config(_config()))


class TestMetrics:
    def test_the_in_band_loss_and_the_band_costs_are_reported(self):
        freqs = np.arange(0.0, 100.0, 0.05)
        psd_before = np.ones((2, freqs.size))
        psd_after = psd_before.copy()
        psd_after[:, (freqs >= 56.8) & (freqs <= 57.7)] = 0.01
        settings = nt.NotchSettings.from_config(_config())

        rows = nt.notch_metrics(freqs, psd_before, psd_after, settings)

        assert len(rows) == 1
        row = rows[0]
        assert row["band_hz"] == "56.8-57.7 Hz"
        assert row["in_band_change_db"] == pytest.approx(-20.0, abs=0.2)
        # 0.9 Hz of a 49.9 Hz band.
        assert row["gamma_width_share"] == pytest.approx(0.9 / 49.9, rel=1e-3)
        assert row["delta_width_share"] == 0.0
        # Gamma loses only the notched slice, so its total change is small but non-zero.
        assert -1.0 < row["gamma_change_db"] < 0.0
        assert row["delta_change_db"] == pytest.approx(0.0, abs=1e-12)

    def test_band_power_refuses_a_span_with_no_bins(self):
        freqs = np.arange(0.0, 10.0, 1.0)

        with pytest.raises(ValueError, match="No frequency bin"):
            nt.band_power(freqs, np.ones((1, freqs.size)), 20.0, 21.0)


def _bids_fixture(tmp_path: Path) -> Path:
    """A one-subject BIDS root written by MNE-BIDS, standing in for the cleaned copy."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    sfreq = 500.0
    times = np.arange(int(sfreq * 80)) / sfreq
    rng = np.random.default_rng(5)
    data = rng.normal(scale=1e-6, size=(3, times.size))
    data += 5e-6 * np.sin(2.0 * np.pi * 57.25 * times)
    info = mne.create_info(["Cz", "Pz", "Oz"], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

    root = tmp_path / "eeg_linecleaned"
    path = BIDSPath(
        subject="0001",
        task=rm_task(),
        run="1",
        datatype="eeg",
        root=root,
        extension=".vhdr",
    )
    write_raw_bids(raw, path, format="BrainVision", allow_preload=True, verbose="ERROR")
    return root


def rm_task() -> str:
    from studies.pain_study.scripts.line_comb import remove as rm

    return rm.TASK


class TestWriting:
    def test_only_the_binaries_differ_and_the_band_is_gone(self, tmp_path):
        from studies.pain_study.scripts.line_comb import remove as rm

        source = _bids_fixture(tmp_path)
        destination = tmp_path / "notched"
        destination.mkdir()
        rm.mirror_sidecars(source, destination)
        settings = nt.NotchSettings.from_config(_config())
        vhdr = next(source.rglob("*_eeg.vhdr"))

        rows = nt.notch_run(vhdr, destination, source, settings)

        assert rows[0]["recording"] == vhdr.stem
        assert rows[0]["in_band_change_db"] < -10.0
        for original in sorted(source.rglob("*")):
            # mirror_sidecars copies everything but the binaries and MNE-BIDS' lock files.
            if original.is_dir() or original.suffix in {".eeg", ".lock"}:
                continue
            mirrored = destination / original.relative_to(source)
            assert mirrored.read_bytes() == original.read_bytes(), original.name
        written = destination / vhdr.relative_to(source).with_suffix(".eeg")
        assert written.read_bytes() != vhdr.with_suffix(".eeg").read_bytes()

    def test_corruption_of_the_written_binary_is_caught(self, tmp_path, monkeypatch):
        from studies.pain_study.scripts.line_comb import remove as rm

        source = _bids_fixture(tmp_path)
        destination = tmp_path / "notched"
        destination.mkdir()
        rm.mirror_sidecars(source, destination)
        vhdr = next(source.rglob("*_eeg.vhdr"))

        honest_write = rm.write_eeg_binary

        def corrupt(vhdr_path, target, data):
            honest_write(vhdr_path, target, np.asarray(data) * 2.0)

        monkeypatch.setattr(nt.rm, "write_eeg_binary", corrupt)

        with pytest.raises(RuntimeError, match="round-trip tolerance"):
            nt.notch_run(vhdr, destination, source, nt.NotchSettings.from_config(_config()))

    def test_the_description_records_the_bands_that_made_it(self, tmp_path):
        from studies.pain_study.scripts.line_comb import remove as rm

        source = _bids_fixture(tmp_path)
        destination = tmp_path / "notched"
        destination.mkdir()
        rm.mirror_sidecars(source, destination)
        settings = nt.NotchSettings.from_config(_config())

        path = nt.write_derivative_description(destination, source, settings)

        described = json.loads(path.read_text(encoding="utf-8"))
        assert described["DatasetType"] == "derivative"
        entry = next(e for e in described["GeneratedBy"] if e["Name"] == "line-comb notch")
        assert entry["Parameters"]["bands_hz"] == [[56.8, 57.7]]
        assert described["SourceDatasets"][0]["URL"].endswith(source.name)

    def test_a_second_run_replaces_its_own_entry_rather_than_stacking(self, tmp_path):
        from studies.pain_study.scripts.line_comb import remove as rm

        source = _bids_fixture(tmp_path)
        destination = tmp_path / "notched"
        destination.mkdir()
        rm.mirror_sidecars(source, destination)
        settings = nt.NotchSettings.from_config(_config())

        nt.write_derivative_description(destination, source, settings)
        path = nt.write_derivative_description(destination, source, settings)

        described = json.loads(path.read_text(encoding="utf-8"))
        named = [e for e in described["GeneratedBy"] if e["Name"] == "line-comb notch"]
        assert len(named) == 1

    def test_a_missing_source_description_is_refused(self, tmp_path):
        destination = tmp_path / "notched"
        destination.mkdir()

        with pytest.raises(FileNotFoundError, match="was not mirrored"):
            nt.write_derivative_description(
                destination, tmp_path / "src", nt.NotchSettings.from_config(_config())
            )


def test_the_stage_is_reachable_from_the_cli():
    from studies.pain_study.cli import line_comb as cli

    assert "notch" in cli.MODES


def test_the_stage_refuses_a_subject_subset():
    """A subset would write a derivative covering part of the cohort."""
    import argparse

    from studies.pain_study.cli import line_comb as cli

    args = argparse.Namespace(mode="notch", subjects=["sub-0001"])

    with pytest.raises(ValueError, match="must use all recordings"):
        cli.run_line_comb(args, [], None)


def test_the_stage_refuses_when_apply_has_not_run(tmp_path, monkeypatch):
    import argparse

    class Config:
        def path(self, name, override=None):
            return tmp_path / name

        def get(self, key, default=None):
            return _config().get(key, default)

    monkeypatch.setattr(
        "studies.pain_study.scripts.workflow_config.load_workflow_config",
        lambda *a, **k: Config(),
    )

    with pytest.raises(FileNotFoundError, match="Run `line-comb apply` first"):
        nt.run(argparse.Namespace(config=None, bids_root=None, output_root=None, report_dir=None))


def test_an_existing_output_root_is_not_mixed_into(tmp_path, monkeypatch):
    import argparse

    source = _bids_fixture(tmp_path)
    existing = tmp_path / "already_there"
    existing.mkdir()

    class Config:
        def path(self, name, override=None):
            return {"output_root": source, "notched_root": existing, "removal_dir": tmp_path}[name]

        def get(self, key, default=None):
            return _config().get(key, default)

    monkeypatch.setattr(
        "studies.pain_study.scripts.workflow_config.load_workflow_config",
        lambda *a, **k: Config(),
    )

    with pytest.raises(FileExistsError, match="Refusing to mix"):
        nt.run(argparse.Namespace(config=None, bids_root=None, output_root=None, report_dir=None))
    shutil.rmtree(existing)
