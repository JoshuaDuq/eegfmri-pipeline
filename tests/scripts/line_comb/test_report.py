"""The outcome report must follow the participant-specific transform provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.scripts.line_comb import report
from studies.pain_study.scripts.line_comb.remove import RemovalSettings


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recording": "sub-0001_task-x_run-1_eeg",
                "fundamental_hz": 1.2,
                "isolated_hz": "47.04;57.22",
                "adjacent_hz": "27.72",
            },
            {
                "recording": "sub-0002_task-x_run-1_eeg",
                "fundamental_hz": 1.199,
                "isolated_hz": "94.31",
                "adjacent_hz": "81.50",
            },
        ]
    )


def test_report_targets_are_resolved_per_subject_from_the_manifest():
    targets = report.subject_artifact_targets(
        _manifest(),
        ("sub-0001", "sub-0002"),
        RemovalSettings(removal_harmonic_range=(22, 83), high_hz=99.8),
    )

    assert any(abs(value - 47.04) < 1e-9 for value in targets["sub-0001"])
    assert any(abs(value - 27.72) < 1e-9 for value in targets["sub-0001"])
    assert not any(abs(value - 47.04) < 1e-9 for value in targets["sub-0002"])
    assert any(abs(value - 94.31) < 1e-9 for value in targets["sub-0002"])
    assert any(abs(value - 81.50) < 1e-9 for value in targets["sub-0002"])
    assert not any(59.5 <= value <= 60.5 for values in targets.values() for value in values)


def test_report_refuses_missing_subject_provenance():
    with pytest.raises(ValueError, match="no manifest rows for sub-0003"):
        report.subject_artifact_targets(
            _manifest(),
            ("sub-0001", "sub-0003"),
            RemovalSettings(),
        )


def test_artifact_share_uses_each_subjects_own_targets():
    freqs = np.arange(1.0, 100.0, 0.1)
    psd = np.ones((2, freqs.size))
    psd[0, np.argmin(abs(freqs - 47.0))] = 100.0
    psd[1, np.argmin(abs(freqs - 94.0))] = 100.0

    shares, counts = report.artifact_share_by_subject(
        freqs,
        psd,
        (45.0, 95.0),
        {"sub-0001": (47.0,), "sub-0002": (94.0,)},
        ("sub-0001", "sub-0002"),
        half_width_bins=10,
    )

    assert np.all(shares > 0.0)
    assert counts.tolist() == [1, 1]
