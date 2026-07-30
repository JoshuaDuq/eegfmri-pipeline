from pathlib import Path

import mne
import numpy as np

from eeg_pipeline.preprocessing.bcg.sources import RunPair, discover_run_pairs, validate_pair


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
