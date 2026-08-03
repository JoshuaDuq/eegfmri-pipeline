"""Tests for the line-comb removal runner."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.utils.config.loader import load_config
from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts.line_comb import remove as rlc


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
        assert settings.removal_harmonic_range == (22, 82)
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


def test_removal_settings_reads_the_mains_exclusion_flag():
    """Whether mains is left to the pipeline's FIR notch is a setting, not a constant."""

    class _Config:
        def __init__(self, block):
            self._block = block

        def get(self, key):
            return self._block if key == "line_comb_removal" else None

    assert rlc.RemovalSettings.from_config(_Config({})).exclude_mains is True
    assert rlc.RemovalSettings.from_config(_Config({"exclude_mains": False})).exclude_mains is False


def _synthetic_spectrum(peaks=(), *, f0=1.2, harmonics=(24, 79), df=0.002):
    """A spectrum carrying a full comb plus the given isolated peaks."""
    freqs = np.arange(1.0, 100.0, df)
    spectrum = np.zeros_like(freqs)
    sigma = 0.109 / 2.355

    def add(centre, height):
        spectrum[:] = np.maximum(
            spectrum, height * np.exp(-0.5 * ((freqs - centre) / sigma) ** 2)
        )

    for k in range(harmonics[0], harmonics[1] + 1):
        add(k * f0, 14.0)
    for centre, height in peaks:
        add(centre, height)
    return freqs, spectrum, spectrum.copy()


def test_the_configured_list_is_used_when_detection_is_off():
    """Detection is opt-in; with it off nothing about the existing behaviour changes."""
    settings = rlc.RemovalSettings(detect_isolated=False, isolated_hz=(47.0362, 94.0748))
    freqs, spec, prom = _synthetic_spectrum(peaks=[(94.3453, 26.0)])
    assert rlc.isolated_nominals(freqs, spec, prom, settings) == (47.0362, 94.0748)


def test_detection_finds_the_line_the_configured_list_would_have_missed():
    """94.3453 Hz is 0.27 Hz from the curated seed, outside its 0.15 Hz window."""
    settings = rlc.RemovalSettings(detect_isolated=True, isolated_hz=(94.0748,))
    freqs, spec, prom = _synthetic_spectrum(peaks=[(94.3453, 26.0)])
    found = rlc.isolated_nominals(freqs, spec, prom, settings)
    assert any(abs(f - 94.3453) < 0.02 for f in found), found


def test_a_session_agrees_on_one_nominal_list_across_its_runs():
    """Pooling lines estimates up position by position and refuses runs that disagree.

    ``combine_estimates`` raises when two runs carry a different number of isolated lines,
    which per-run detection produces the moment a line sits either side of the prominence
    floor in different runs. The session list is therefore resolved once, from every run.
    """
    settings = rlc.RemovalSettings(detect_isolated=True)
    # The line recurs in two of the three runs, so it clears the recurrence rule, and the
    # third run is the one that has to be given the nominal anyway for pooling to line up.
    strong = _synthetic_spectrum(peaks=[(47.04, 24.0), (94.34, 26.0)])
    also = _synthetic_spectrum(peaks=[(47.04, 24.0), (94.34, 22.0)])
    without = _synthetic_spectrum(peaks=[(47.04, 24.0)])

    nominals = rlc.session_nominals([strong, also, without], settings)
    assert any(abs(f - 94.34) < 0.02 for f in nominals), (
        "a line the session carries must be offered to every run in it"
    )

    estimates = [
        lr.estimate_comb(
            freqs, spec, prom,
            nominal_hz=settings.nominal_fundamental_hz,
            harmonic_range=settings.harmonic_range,
            isolated_nominal_hz=nominals,
            search_hz=settings.search_hz,
            isolated_search_hz=settings.isolated_search_hz,
            min_prominence_db=settings.min_prominence_db,
        )
        for freqs, spec, prom in (strong, also, without)
    ]
    lr.combine_estimates(estimates)  # must not raise


def test_the_session_list_respects_the_budget():
    settings = rlc.RemovalSettings(detect_isolated=True, max_isolated_lines=2)
    spectra = [_synthetic_spectrum(peaks=[(47.04, 24.0), (94.34, 26.0), (30.5, 20.0)])]
    assert len(rlc.session_nominals(spectra, settings)) <= 2


def test_the_session_budget_is_spent_on_the_strongest_lines_not_the_lowest():
    """A cap that truncates by frequency throws away exactly what matters.

    Measured on sub-0000, whose six runs union to twenty candidate lines against a cap of
    sixteen: sorting by frequency and truncating dropped 93.944 Hz -- the 94 Hz line this
    detection exists to catch -- while keeping weak lines at 20.037 and 21.093 Hz for no
    reason but that they sit lower in the spectrum.
    """
    settings = rlc.RemovalSettings(detect_isolated=True, max_isolated_lines=2)
    # All four recur, so recurrence is not what decides this -- the budget is, and it has
    # to spend itself on the two strongest rather than the two lowest.
    peaks = [(21.093, 11.0), (20.037, 10.5), (47.043, 23.0), (94.344, 27.0)]
    spectra = [_synthetic_spectrum(peaks=peaks), _synthetic_spectrum(peaks=peaks)]
    kept = rlc.session_nominals(spectra, settings)

    assert len(kept) == 2
    assert any(abs(f - 94.344) < 0.02 for f in kept), (
        f"the strongest line was dropped by the cap: {kept}"
    )
    assert any(abs(f - 47.043) < 0.02 for f in kept), kept
    assert not any(f < 25.0 for f in kept), (
        f"weak low-frequency lines were kept ahead of strong ones: {kept}"
    )


def test_the_session_list_is_returned_in_frequency_order():
    """Ranking happens on strength; the result is still ordered for the manifest."""
    settings = rlc.RemovalSettings(detect_isolated=True)
    spectra = [_synthetic_spectrum(peaks=[(94.344, 27.0), (47.043, 23.0), (28.1, 19.0)])]
    kept = rlc.session_nominals(spectra, settings)
    assert list(kept) == sorted(kept)


def test_a_line_seen_in_only_one_run_of_a_session_is_not_taken():
    """The runs of a session are the replication that separates a line from a fluctuation.

    Measured on sub-0000: of twenty candidates unioned across its six runs, fifteen appeared
    in exactly one run and five appeared in five or six. The recurring five are the known
    lines -- 28.278, 57.296, 58.185, 82.204 and 93.944 Hz. sub-0008 is the clean case, where
    all seven of its lines appear in all six runs. A line on one machine minutes apart does
    not come and go; a noise peak clearing the floor once does.
    """
    settings = rlc.RemovalSettings(detect_isolated=True)
    # 63.0 Hz is 0.6 Hz from the nearest comb position and 2.35 Hz from the nearest probe
    # tone, so if it is rejected it is the recurrence rule doing it and nothing else.
    real = (47.043, 23.0)
    once = (63.0, 18.0)
    spectra = [
        _synthetic_spectrum(peaks=[real, once]),
        _synthetic_spectrum(peaks=[real]),
        _synthetic_spectrum(peaks=[real]),
    ]
    kept = rlc.session_nominals(spectra, settings)

    assert any(abs(f - real[0]) < 0.02 for f in kept), kept
    assert not any(abs(f - once[0]) < 0.02 for f in kept), (
        f"a peak present in one run of three was taken for the session: {kept}"
    )


def test_a_single_run_session_can_still_contribute_lines():
    """The recurrence rule must not empty a session that has only one run to offer."""
    settings = rlc.RemovalSettings(detect_isolated=True)
    kept = rlc.session_nominals([_synthetic_spectrum(peaks=[(47.043, 23.0)])], settings)
    assert any(abs(f - 47.043) < 0.02 for f in kept), kept
