from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_pipeline.analysis.qc.scanner_harmonic_comb import (
    ParticipantSpectrum,
    ScannerCombParameters,
    Spectrum,
)
from eeg_pipeline.preprocessing.pipeline import scanner_harmonic_qc as module
from tests.utils.pipelines_test_utils import DotConfig


def _parameters() -> ScannerCombParameters:
    return ScannerCombParameters(
        welch_duration_seconds=2.0,
        frequency_resolution_hz=0.5,
        bootstrap_resamples=20,
    )


def _participant(participant: str, offset: float = 0.0) -> ParticipantSpectrum:
    frequencies = np.arange(15.0, 90.5, 0.5)
    power = offset + np.sin(frequencies / 10.0)
    for frequency in (20.0, 41.0, 61.0, 82.0):
        power[np.argmin(np.abs(frequencies - frequency))] += 8.0
    return ParticipantSpectrum(participant, frequencies, power)


def test_load_input_participant_aggregates_runs_equally(monkeypatch, tmp_path: Path) -> None:
    paths = [
        SimpleNamespace(fpath=tmp_path / "run-1.vhdr", run="1"),
        SimpleNamespace(fpath=tmp_path / "run-2.vhdr", run="2"),
    ]
    for path in paths:
        path.fpath.touch()
    spectra = iter(
        [
            Spectrum(np.array([20.0, 21.0]), np.array([0.0, 2.0])),
            Spectrum(np.array([20.0, 21.0]), np.array([10.0, 6.0])),
        ]
    )
    monkeypatch.setattr(module, "find_matching_paths", lambda *args, **kwargs: paths)
    monkeypatch.setattr(module, "read_raw_bids", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "compute_raw_comb_spectrum", lambda *args: next(spectra))

    observed = module._load_input_participant(
        participant="0001",
        task="thermalactive",
        bids_root=tmp_path,
        input_extension=".vhdr",
        parameters=_parameters(),
    )

    np.testing.assert_allclose(observed.power_db, [5.0, 4.0])


def test_load_final_participant_requires_clean_epochs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(module, "find_clean_epochs_path", lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="sub-0002.*clean epochs"):
        module._load_final_participant(
            participant="0002",
            task="thermalactive",
            deriv_root=tmp_path,
            parameters=_parameters(),
        )


def test_run_scanner_harmonic_qc_writes_task_outputs(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "bids").mkdir()
    (tmp_path / "derivatives").mkdir()
    monkeypatch.setattr(
        module,
        "_load_input_participant",
        lambda participant, **kwargs: _participant(participant),
    )
    monkeypatch.setattr(
        module,
        "_load_final_participant",
        lambda participant, **kwargs: _participant(participant, -3.0),
    )

    def fake_write(summary, *, output_dir, task):
        assert summary.participant_ids == ("0001", "0002")
        assert output_dir == tmp_path / "derivatives" / "preprocessed" / "eeg" / "qc"
        png = output_dir / f"task-{task}_desc-scannerharmoniccomb_qc.png"
        tsv = output_dir / f"task-{task}_desc-scannerharmoniccomb_qc.tsv"
        output_dir.mkdir(parents=True)
        png.write_bytes(b"png")
        tsv.write_text("frequency_hz\n", encoding="utf-8")
        return png, tsv

    monkeypatch.setattr(module, "write_scanner_harmonic_comb", fake_write)

    outputs = module.run_scanner_harmonic_qc(
        subjects=["0001", "0002"],
        task="thermalactive",
        bids_root=tmp_path / "bids",
        deriv_root=tmp_path / "derivatives",
        input_extension=".vhdr",
        parameters=_parameters(),
    )

    assert outputs.png_path.exists()
    assert outputs.tsv_path.exists()


def test_run_scanner_harmonic_qc_rejects_duplicate_subjects(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="subjects must be unique"):
        module.run_scanner_harmonic_qc(
            subjects=["0001", "0001"],
            task="thermalactive",
            bids_root=tmp_path / "bids",
            deriv_root=tmp_path / "derivatives",
            input_extension=".vhdr",
            parameters=_parameters(),
        )


def test_scanner_comb_parameters_from_config_maps_required_values() -> None:
    config = DotConfig(
        {
            "project": {"random_state": 17},
            "preprocessing": {
                "scanner_harmonic_qc": {
                    "frequency_range_hz": [15.0, 90.0],
                    "welch_duration_seconds": 4.0,
                    "frequency_resolution_hz": 0.25,
                    "bootstrap_resamples": 10_000,
                    "confidence_level": 0.95,
                }
            },
        }
    )

    parameters = module.scanner_comb_parameters_from_config(config)

    assert parameters == ScannerCombParameters(
        frequency_min_hz=15.0,
        frequency_max_hz=90.0,
        welch_duration_seconds=4.0,
        frequency_resolution_hz=0.25,
        bootstrap_resamples=10_000,
        confidence_level=0.95,
        random_seed=17,
    )


def test_scanner_comb_parameters_from_config_rejects_unknown_keys() -> None:
    config = DotConfig(
        {
            "project": {"random_state": 17},
            "preprocessing": {
                "scanner_harmonic_qc": {
                    "frequency_range_hz": [15.0, 90.0],
                    "welch_duration_seconds": 4.0,
                    "frequency_resolution_hz": 0.25,
                    "bootstrap_resamples": 10_000,
                    "confidence_level": 0.95,
                    "enabled": True,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="Unknown scanner harmonic QC config keys: enabled"):
        module.scanner_comb_parameters_from_config(config)
