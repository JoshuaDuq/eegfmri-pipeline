"""Analyzer correction and beat-marker agreement, as tables.

One function per table the report section used to render. Each returns a DataFrame
carrying the columns that table carried, so the numbers survive the move out of the MNE
report and can be written as TSV instead. What the numbers mean is in README.md.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from eeg_pipeline.preprocessing.report.style import run_label
from studies.pain_study.analysis.bcg.report import (
    AnalyzerCorrectionQc,
    MarkerAgreement,
    _QC_COLUMNS,
)


def analyzer_qc_frame(qc: AnalyzerCorrectionQc) -> pd.DataFrame:
    # The same source columns the table selected, in the same order, unformatted: a TSV
    # keeps the numbers and leaves the rendering to whatever reads it.
    present = [spec.source for spec in _QC_COLUMNS if spec.source in qc.runs.columns]
    return qc.runs.loc[:, present].copy()


def marker_agreement_frame(agreements: Sequence[MarkerAgreement]) -> pd.DataFrame:
    rows = [
        {
            "run": run_label(agreement.recording_id),
            "n_markers": agreement.n_markers,
            "n_detected": agreement.n_detected,
            "n_matched": agreement.n_matched,
            "matched_fraction": agreement.matched_fraction,
            "marker_precision": agreement.marker_precision,
            "median_lag_ms": (
                None if agreement.median_lag_s is None else agreement.median_lag_s * 1000.0
            ),
            "lag_iqr_ms": (
                None if agreement.lag_iqr_s is None else agreement.lag_iqr_s * 1000.0
            ),
        }
        for agreement in agreements
    ]
    return pd.DataFrame(rows)


def unmeasured_runs(qc: AnalyzerCorrectionQc) -> dict[str, list[str]]:
    # Named rather than dropped: a run missing from the table above has not been measured
    # and found clean, it has not been measured. Both lists are accounted for in README.md.
    return {
        "outside_configured_bounds": list(qc.runs_outside_bounds),
        "detected_from_ecg_channel": list(qc.fallback_runs),
    }


__all__ = ["analyzer_qc_frame", "marker_agreement_frame", "unmeasured_runs"]
