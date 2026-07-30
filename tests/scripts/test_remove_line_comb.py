"""Tests for the line-comb removal runner."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.utils.config.loader import load_config
from studies.pain_study.analysis import line_removal as lr
from studies.pain_study.scripts import remove_line_comb as rlc


@pytest.fixture
def brainvision_run(tmp_path):
    """A small BrainVision file written the way the BIDS dataset writes them."""
    import mne

    mne.set_log_level("ERROR")
    sfreq, n_times = 1000.0, 4000
    rng = np.random.default_rng(0)
    data = rng.normal(scale=2e-5, size=(4, n_times))
    info = mne.create_info(["Fp1", "Cz", "Oz", "ECG"], sfreq, ["eeg", "eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(data, info)
    path = tmp_path / "sub-0001_task-thermalactive_run-1_eeg.vhdr"
    mne.export.export_raw(path, raw, fmt="brainvision", overwrite=True)
    return path, raw


class TestRemovalSettings:
    def test_reads_the_packaged_config(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.nominal_fundamental_hz == pytest.approx(1.2)
        assert settings.harmonic_range == (24, 79)
        assert settings.removal_harmonic_range == (22, 79)
        assert settings.filter_length == "20s"
        assert settings.mt_bandwidth == pytest.approx(0.6)

    def test_isolated_lines_match_the_diagnosis(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert set(settings.isolated_hz) == set(lr.ISOLATED_NOMINAL_HZ)

    def test_the_isolated_window_cannot_reach_a_comb_line(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        for frequency in settings.isolated_hz:
            spacing = settings.nominal_fundamental_hz
            distance = abs(frequency - round(frequency / spacing) * spacing)
            assert distance > settings.isolated_search_hz

    def test_notch_width_is_configured_rather_than_left_to_mne(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.notch_width_ratio == pytest.approx(450.0)
        assert settings.notch_width_min_hz == pytest.approx(0.05)
        # MNE's own default would empty a quarter of the band.
        assert settings.notch_width_ratio > 200.0

    def test_removal_reaches_below_the_fit_but_spares_harmonic_11(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.removal_harmonic_range[0] < settings.harmonic_range[0]
        assert settings.removal_harmonic_range[0] == 22  # 26.40 Hz
        assert settings.removal_harmonic_range[0] > 11  # 13.23 Hz stays

    def test_the_configured_width_keeps_the_band_inside_the_gate(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        low, high = settings.removal_harmonic_range
        targets = [
            settings.nominal_fundamental_hz * k
            for k in range(low, high + 1)
            if not 59.5 <= settings.nominal_fundamental_hz * k <= 60.5
        ]
        widths = lr.notch_widths_for(
            targets, ratio=settings.notch_width_ratio, minimum_hz=settings.notch_width_min_hz
        )
        fraction = lr.removed_band_fraction(np.arange(0, 120, 0.05), targets, widths)
        assert fraction <= lr.PreservationGate().max_band_fraction_removed

    def test_the_comb_window_cannot_reach_the_next_harmonic(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.search_hz < settings.nominal_fundamental_hz / 2

    def test_falls_back_when_the_block_is_absent(self):
        class Empty:
            def get(self, key, default=None):
                return default

        assert rlc.RemovalSettings.from_config(Empty()) == rlc.RemovalSettings()

    def test_config_values_override_the_defaults(self):
        class Fake:
            def get(self, key, default=None):
                return {"harmonic_range": [10, 20], "mt_bandwidth": 0.9, "filter_length": "8s"}

        settings = rlc.RemovalSettings.from_config(Fake())
        assert settings.harmonic_range == (10, 20)
        assert settings.mt_bandwidth == pytest.approx(0.9)
        assert settings.filter_length == "8s"


class TestChannelScaling:
    def test_reads_names_and_resolutions(self, brainvision_run):
        path, raw = brainvision_run
        names, resolutions = rlc.parse_channel_scaling(path)
        assert names == raw.ch_names
        assert np.all(resolutions > 0)

    def test_rejects_a_format_it_cannot_write(self, tmp_path):
        path = tmp_path / "x.vhdr"
        path.write_text(
            "BinaryFormat=INT_16\nDataOrientation=MULTIPLEXED\nCh1=Fp1,,0.5,µV\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="IEEE_FLOAT_32"):
            rlc.parse_channel_scaling(path)

    def test_rejects_a_vectorised_layout(self, tmp_path):
        path = tmp_path / "x.vhdr"
        path.write_text(
            "BinaryFormat=IEEE_FLOAT_32\nDataOrientation=VECTORIZED\nCh1=Fp1,,0.5,µV\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="MULTIPLEXED"):
            rlc.parse_channel_scaling(path)

    def test_ignores_a_coordinates_section(self, tmp_path):
        """Headers written by Analyzer carry a second block of ``Ch<N>=`` lines.

        Those hold three comma-separated numbers rather than the four fields of
        ``[Channel Infos]``. A pattern whose character classes admit newlines runs one
        coordinate line into the next and parses a resolution of ``"-72\\nCh2=1"``.
        """
        path = tmp_path / "x.vhdr"
        path.write_text(
            "BinaryFormat=IEEE_FLOAT_32\n"
            "DataOrientation=MULTIPLEXED\n"
            "[Channel Infos]\n"
            "Ch1=Fp1,,0.5,µV\n"
            "Ch2=Cz,,0.5,µV\n"
            "Ch3=Oz,,0.5,µV\n"
            "[Coordinates]\n"
            "Ch1=1,-90,-72\n"
            "Ch2=1,45,90\n"
            "Ch3=1,0,0\n",
            encoding="utf-8",
        )

        names, resolutions = rlc.parse_channel_scaling(path)

        assert names == ["Fp1", "Cz", "Oz"]
        assert np.allclose(resolutions, 0.5)


class TestWriteEegBinary:
    def test_round_trips_through_the_original_header(self, brainvision_run, tmp_path):
        import mne

        path, raw = brainvision_run
        modified = raw.get_data() * 0.5
        destination = tmp_path / "out" / path.with_suffix(".eeg").name
        destination.parent.mkdir()
        rlc.write_eeg_binary(path, destination, modified)

        # Reuse the original header, which is exactly what the runner does.
        for suffix in (".vhdr", ".vmrk"):
            (destination.parent / path.with_suffix(suffix).name).write_bytes(
                path.with_suffix(suffix).read_bytes()
            )
        back = mne.io.read_raw_brainvision(
            destination.parent / path.name, preload=True, verbose="ERROR"
        )
        # float32 storage, so the comparison has to be relative to full scale.
        deviation = np.max(np.abs(back.get_data() - modified))
        assert deviation < rlc.ROUNDTRIP_RELATIVE_TOLERANCE * np.max(np.abs(modified))
        assert deviation > 0  # it really did go through float32

    def test_rejects_a_channel_count_mismatch(self, brainvision_run, tmp_path):
        path, raw = brainvision_run
        with pytest.raises(ValueError, match="header describes"):
            rlc.write_eeg_binary(path, tmp_path / "o.eeg", raw.get_data()[:2])


class TestMirrorSidecars:
    def test_copies_everything_except_binaries(self, tmp_path):
        source = tmp_path / "src"
        (source / "sub-01" / "eeg").mkdir(parents=True)
        (source / "dataset_description.json").write_text("{}", encoding="utf-8")
        (source / "sub-01" / "eeg" / "a_eeg.vhdr").write_text("h", encoding="utf-8")
        (source / "sub-01" / "eeg" / "a_eeg.eeg").write_bytes(b"\x00" * 16)
        (source / "sub-01" / "eeg" / "a_eeg.vhdr.lock").write_text("", encoding="utf-8")

        destination = tmp_path / "dst"
        assert rlc.mirror_sidecars(source, destination) == 2
        assert (destination / "dataset_description.json").exists()
        assert (destination / "sub-01" / "eeg" / "a_eeg.vhdr").exists()
        assert not (destination / "sub-01" / "eeg" / "a_eeg.eeg").exists()
        assert not (destination / "sub-01" / "eeg" / "a_eeg.vhdr.lock").exists()


class TestDiscoverRuns:
    def _make(self, root, subject, run):
        directory = root / subject / "eeg"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{subject}_task-thermalactive_run-{run}_eeg.vhdr"
        path.write_text("", encoding="utf-8")
        return path

    def test_finds_every_task_run(self, tmp_path):
        for subject in ("sub-0001", "sub-0002"):
            for run in (1, 2):
                self._make(tmp_path, subject, run)
        assert len(rlc.discover_runs(tmp_path, None)) == 4

    def test_filters_by_subject(self, tmp_path):
        for subject in ("sub-0001", "sub-0002"):
            self._make(tmp_path, subject, 1)
        found = rlc.discover_runs(tmp_path, ["sub-0002"])
        assert len(found) == 1
        assert found[0].parent.parent.name == "sub-0002"

    def test_raises_when_nothing_matches(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No task runs"):
            rlc.discover_runs(tmp_path, None)


class TestRunSpectrum:
    def test_returns_a_tr_commensurate_grid(self):
        import mne

        mne.set_log_level("ERROR")
        sfreq = 1000.0
        n_times = int(sfreq * 0.9 * rlc.ESTIMATION_TR_COUNT * 2)
        rng = np.random.default_rng(1)
        info = mne.create_info(["Fp1", "Cz", "Oz"], sfreq, "eeg")
        raw = mne.io.RawArray(rng.normal(scale=1e-5, size=(3, n_times)), info)
        freqs, spectrum_db, prominence = rlc.run_spectrum(raw)
        assert freqs[1] == pytest.approx(1.0 / (0.9 * rlc.ESTIMATION_TR_COUNT))
        assert spectrum_db.shape == freqs.shape == prominence.shape
        # a comb line lands on a bin centre
        assert np.min(np.abs(freqs - 54.0)) < 1e-9

    def test_rejects_a_recording_shorter_than_one_block(self):
        import mne

        mne.set_log_level("ERROR")
        info = mne.create_info(["Fp1"], 1000.0, "eeg")
        raw = mne.io.RawArray(np.zeros((1, 1000)), info)
        with pytest.raises(ValueError, match="shorter than one estimation block"):
            rlc.run_spectrum(raw)
