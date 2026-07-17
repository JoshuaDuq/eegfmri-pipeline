from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.analysis.qc.scanner_harmonic_comb import ScannerCombSummary
from eeg_pipeline.plotting.scanner_harmonic_comb import (
    SCANNER_COMB_COLUMNS,
    build_scanner_harmonic_comb_figure,
    write_scanner_harmonic_comb,
)


def _summary() -> ScannerCombSummary:
    frequencies = np.arange(15.0, 90.25, 0.25)
    input_median = -35.0 - 5.0 * np.log10(frequencies)
    final_median = input_median - 3.0
    for frequency, height in zip((20.0, 41.0, 61.0, 82.0), (8.0, 14.0, 18.0, 13.0)):
        index = int(np.argmin(np.abs(frequencies - frequency)))
        input_median[index] += height
        final_median[index] += 0.5 * height
    return ScannerCombSummary(
        participant_ids=("0001", "0002", "0003"),
        frequencies_hz=frequencies,
        input_median_db=input_median,
        input_ci_low_db=input_median - 1.0,
        input_ci_high_db=input_median + 1.0,
        final_median_db=final_median,
        final_ci_low_db=final_median - 0.8,
        final_ci_high_db=final_median + 0.8,
        harmonic_frequencies_hz=(20.0, 41.0, 61.0, 82.0),
    )


def test_comb_figure_contains_two_lines_four_windows_and_references() -> None:
    figure = build_scanner_harmonic_comb_figure(_summary(), task="thermalactive")
    axis = figure.axes[0]

    assert len(figure.axes) == 5
    assert len(axis.lines) == 6
    assert len(axis.patches) == 4
    assert axis.get_xlim() == pytest.approx((15.0, 90.0))
    assert axis.get_xlabel() == "Frequency (Hz)"
    assert axis.get_ylabel() == "PSD (dB V²/Hz)"
    assert axis.get_title() == "Full scanner-harmonic comb"
    assert "3 participants" in figure.get_suptitle()


def test_write_scanner_harmonic_comb_outputs_png_and_tsv(tmp_path: Path) -> None:
    png_path, tsv_path = write_scanner_harmonic_comb(
        _summary(),
        output_dir=tmp_path,
        task="thermalactive",
    )

    assert png_path.name == "task-thermalactive_desc-scannerharmoniccomb_qc.png"
    assert png_path.stat().st_size > 0
    assert tsv_path.name == "task-thermalactive_desc-scannerharmoniccomb_qc.tsv"
    frame = pd.read_csv(tsv_path, sep="\t")
    assert frame.columns.tolist() == list(SCANNER_COMB_COLUMNS)
    assert frame["n_participants"].nunique() == 1
    assert frame["n_participants"].iloc[0] == 3
    np.testing.assert_allclose(frame["frequency_hz"], _summary().frequencies_hz)


def test_comb_writer_rejects_empty_task(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task must be non-empty"):
        write_scanner_harmonic_comb(_summary(), output_dir=tmp_path, task="")
