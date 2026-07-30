from pathlib import Path

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.bcg.sources import (
    RunPair,
    discover_run_pairs,
    validate_pair,
    write_corrected_recording,
)


def _touch(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("")


def test_discover_pairs_matches_subject_and_run(tmp_path):
    unc, cor = tmp_path / "unc", tmp_path / "cor"
    _touch(unc, "ThermalPainEEGFMRI_run1_sub0009_2026-06-22_a_pulse_markers.vhdr")
    _touch(unc, "ThermalPainEEGFMRI_run2_sub0009_2026-06-22_b_pulse_markers.vhdr")
    _touch(cor, "ThermalPainEEGFMRI_run1_sub0009_2026-06-22_a_corrected.vhdr")

    pairs = discover_run_pairs(unc, cor)

    assert [(p.subject, p.run) for p in pairs] == [("sub0009", "1")]


def test_discover_pairs_ignores_appledouble_files(tmp_path):
    unc, cor = tmp_path / "unc", tmp_path / "cor"
    _touch(unc, "ThermalPainEEGFMRI_run1_sub0005_x_pulse_markers.vhdr")
    _touch(unc, "._ThermalPainEEGFMRI_run1_sub0005_x_pulse_markers.vhdr")
    _touch(cor, "ThermalPainEEGFMRI_run1_sub0005_x_corrected.vhdr")

    pairs = discover_run_pairs(unc, cor)

    assert len(pairs) == 1


def _write_brainvision(root: Path, stem: str, data_uv, sfreq=1000.0):
    root.mkdir(parents=True, exist_ok=True)
    names = [f"EEG{i:02d}" for i in range(data_uv.shape[0] - 1)] + ["ECG"]
    types = ["eeg"] * (data_uv.shape[0] - 1) + ["ecg"]
    info = mne.create_info(names, sfreq, ch_types=types)
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")
    mne.export.export_raw(
        root / f"{stem}.vhdr", raw, fmt="brainvision", overwrite=True, verbose="ERROR"
    )
    return root / f"{stem}.vhdr"


def test_validate_pair_reports_alignment_and_ecg_identity(tmp_path):
    rng = np.random.default_rng(0)
    shared_ecg = rng.normal(0, 50, 4000)
    base = rng.normal(0, 10, (3, 4000))

    unc = np.vstack([base, shared_ecg])
    cor = np.vstack([base + 5.0, shared_ecg])  # EEG differs, ECG identical

    a = _write_brainvision(tmp_path / "unc", "ThermalPainEEGFMRI_run1_sub0009_a", unc)
    b = _write_brainvision(tmp_path / "cor", "ThermalPainEEGFMRI_run1_sub0009_b", cor)

    result = validate_pair(RunPair("sub0009", "1", a, b))

    assert result.aligned is True
    assert result.ecg_max_abs_diff_uv < 1e-3
    assert result.status == "ok"


def test_validate_pair_flags_length_mismatch(tmp_path):
    rng = np.random.default_rng(1)
    a = _write_brainvision(
        tmp_path / "unc", "ThermalPainEEGFMRI_run1_sub0009_a", rng.normal(0, 10, (2, 4000))
    )
    b = _write_brainvision(
        tmp_path / "cor", "ThermalPainEEGFMRI_run1_sub0009_b", rng.normal(0, 10, (2, 3000))
    )

    result = validate_pair(RunPair("sub0009", "1", a, b))

    assert result.aligned is False
    assert result.status == "length_mismatch"


def _write_vectorized_brainvision(root: Path, stem: str, data_uv, sfreq=1000.0):
    """A BrainVision triplet in the export's own layout: VECTORIZED, float32, blank scale.

    `mne.export.export_raw` writes MULTIPLEXED with an explicit resolution, so it cannot
    stand in for the Analyzer export here.
    """
    root.mkdir(parents=True, exist_ok=True)
    names = [f"EEG{i:02d}" for i in range(data_uv.shape[0] - 1)] + ["ECG"]
    header = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "[Common Infos]",
        f"DataFile={stem}.eeg",
        f"MarkerFile={stem}.vmrk",
        "DataFormat=BINARY",
        "DataOrientation=VECTORIZED",
        f"NumberOfChannels={len(names)}",
        f"SamplingInterval={1e6 / sfreq:.0f}",
        "[Binary Infos]",
        "BinaryFormat=IEEE_FLOAT_32",
        "[Channel Infos]",
    ]
    header += [f"Ch{i + 1}={name},,,µV" for i, name in enumerate(names)]
    (root / f"{stem}.vhdr").write_text("\n".join(header) + "\n", encoding="utf-8")
    (root / f"{stem}.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={stem}.eeg\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,20260622104812121000\n"
        "Mk2=Stimulus,S  1,1000,1,0\n"
        "Mk3=Pulse Artifact,R,1500,1,0\n",
        encoding="utf-8",
    )
    np.asarray(data_uv, dtype=float).astype("<f4").tofile(root / f"{stem}.eeg")
    return root / f"{stem}.vhdr"


def test_written_recording_round_trips_and_keeps_its_markers(tmp_path):
    """Rewriting the binary must leave header and markers byte-identical.

    The stimulus markers are the whole point of the recording downstream, and rebuilding
    the file through `mne.export.export_raw` silently drops them.
    """
    rng = np.random.default_rng(7)
    original = rng.normal(0, 30.0, (4, 5000))
    source = _write_vectorized_brainvision(tmp_path / "src", "run1_sub0009_corrected", original)

    corrected = original.copy()
    corrected[0, 1000:2000] -= 12.5  # a gap stretch this stage would replace

    written = write_corrected_recording(source, tmp_path / "out", corrected)

    assert written.name == source.name
    for suffix in (".vhdr", ".vmrk"):
        assert written.with_suffix(suffix).read_bytes() == source.with_suffix(suffix).read_bytes()

    mne.set_log_level("ERROR")
    back = mne.io.read_raw_brainvision(written, preload=True, verbose="ERROR")
    assert np.allclose(back.get_data() * 1e6, corrected, atol=1e-3)
    assert "Stimulus/S  1" in set(back.annotations.description)
    assert any(d.split("/")[-1].strip() == "R" for d in back.annotations.description)


def test_written_recording_leaves_untouched_channels_bit_identical(tmp_path):
    """The ECG channel is carried through, not regenerated."""
    rng = np.random.default_rng(8)
    original = rng.normal(0, 30.0, (4, 5000))
    source = _write_vectorized_brainvision(tmp_path / "src", "run1_sub0009_corrected", original)

    corrected = original.copy()
    corrected[0] += 5.0

    written = write_corrected_recording(source, tmp_path / "out", corrected)

    stored = np.fromfile(written.with_suffix(".eeg"), dtype="<f4").reshape(4, 5000)
    assert np.array_equal(stored[3], original[3].astype("<f4"))  # ECG untouched
    assert not np.array_equal(stored[0], original[0].astype("<f4"))


def test_writer_refuses_a_channel_count_the_header_does_not_describe(tmp_path):
    rng = np.random.default_rng(9)
    source = _write_vectorized_brainvision(
        tmp_path / "src", "run1_sub0009_corrected", rng.normal(0, 30.0, (4, 5000))
    )

    with pytest.raises(ValueError, match="describes 4 channels"):
        write_corrected_recording(source, tmp_path / "out", np.zeros((3, 5000)))
