from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.utils.pipelines_test_utils import DotConfig


def test_public_signature_target_loader_supports_study1_config_root() -> None:
    from eeg_pipeline.utils.data.fmri_signature_targets import (
        load_fmri_signature_target_for_subject,
    )

    cfg = DotConfig(
        {
            "study1": {
                "targets": {
                    "method": "lss",
                    "contrast_name": "contrast",
                    "signature_name": "NPS",
                    "metric": "dot",
                    "normalization": "none",
                    "round_decimals": 3,
                    "timing_audit": {
                        "onset_offset_s": 3.0,
                        "onset_tolerance_s": 0.010,
                        "plateau_duration_s": 7.5,
                        "duration_tolerance_s": 0.010,
                    },
                }
            }
        }
    )

    events_df = pd.DataFrame(
        {
            "run": [1, 1],
            "trial_number": [1, 2],
            "onset": [22.150, 65.084],
            "duration": [0.001, 0.001],
        }
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sig_dir = (
            root / "sub-0001" / "fmri" / "lss" / "task-task" / "contrast-contrast" / "signatures"
        )
        sig_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "run": ["run-01", "run-01"],
                "run_num": [1, 1],
                "trial_index": [1, 2],
                "signature": ["NPS", "NPS"],
                "dot": [1.25, 2.50],
                "onset": [25.150, 68.084],
                "duration": [7.5, 7.5],
            }
        ).to_csv(sig_dir / "trial_signature_expression.tsv", sep="\t", index=False)

        pd.DataFrame(
            {
                "run": ["run-01", "run-01"],
                "trial_index": [1, 2],
                "events_trial_number": [1, 2],
            }
        ).to_csv(sig_dir.parent / "trials.tsv", sep="\t", index=False)

        y, y_label, _extra = load_fmri_signature_target_for_subject(
            subject_raw="0001",
            task="task",
            deriv_root=root,
            config=cfg,
            events_df=events_df,
            logger=logging.getLogger(__name__),
            config_path="study1.targets",
        )

    arr = np.asarray(y, dtype=float)
    assert y_label == "fmri_signature.lss.contrast.NPS.dot"
    assert np.all(np.isfinite(arr))
    assert np.allclose(arr, np.array([1.25, 2.50], dtype=float))


def test_signature_target_loader_uses_configured_primary_target_table() -> None:
    from eeg_pipeline.utils.data.fmri_signature_targets import (
        load_fmri_signature_target_for_subject,
    )

    events_df = pd.DataFrame(
        {
            "run": [1, 1],
            "trial_number": [1, 2],
            "onset": [22.150, 65.084],
            "duration": [0.001, 0.001],
        }
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        target_path = root / "primary_targets.parquet"
        pd.DataFrame(
            {
                "subject_id": ["sub-0001", "sub-0001"],
                "task": ["task", "task"],
                "run": [1, 1],
                "trial_index": [1, 2],
                "onset": [22.150, 65.084],
                "duration": [0.001, 0.001],
                "NPS": [10.0, 20.0],
                "pain_binary_coded": [1, 0],
                "stimulus_temp": [47.0, 44.0],
            }
        ).to_parquet(target_path, index=False)

        cfg = DotConfig(
            {
                "machine_learning": {
                    "fmri_signature": {
                        "method": "lss",
                        "contrast_name": "contrast",
                        "signature_name": "NPS",
                        "target_column": "NPS",
                        "target_table_path": str(target_path),
                        "metric": "dot",
                        "normalization": "none",
                        "round_decimals": 3,
                        "timing_audit": {
                            "onset_offset_s": 3.0,
                            "onset_tolerance_s": 0.010,
                            "plateau_duration_s": 7.5,
                            "duration_tolerance_s": 0.010,
                        },
                    }
                }
            }
        )

        y, y_label, extra = load_fmri_signature_target_for_subject(
            subject_raw="0001",
            task="task",
            deriv_root=root,
            config=cfg,
            events_df=events_df,
            logger=logging.getLogger(__name__),
        )

    assert y_label == "fmri_signature.primary_targets.NPS"
    assert np.allclose(np.asarray(y, dtype=float), np.array([10.0, 20.0]))
    assert list(extra["pain_binary_coded"]) == [1.0, 0.0]
    assert list(extra["stimulus_temp"]) == [47.0, 44.0]


def _signature_alignment_config() -> "DotConfig":
    return DotConfig(
        {
            "machine_learning": {
                "fmri_signature": {
                    "method": "lss",
                    "contrast_name": "contrast",
                    "signature_name": "NPS",
                    "metric": "dot",
                    "normalization": "none",
                    "round_decimals": 3,
                    "timing_audit": {
                        "onset_offset_s": 3.0,
                        "onset_tolerance_s": 0.010,
                        "plateau_duration_s": 7.5,
                        "duration_tolerance_s": 0.010,
                    },
                }
            }
        }
    )


def _write_signature_table(root: Path, *, onsets, durations) -> None:
    sig_dir = root / "sub-0001" / "fmri" / "lss" / "task-task" / "contrast-contrast" / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run": ["run-01"] * len(onsets),
            "run_num": [1] * len(onsets),
            "trial_index": list(range(1, len(onsets) + 1)),
            "signature": ["NPS"] * len(onsets),
            "dot": [5.0, 9.0][: len(onsets)],
            "onset": list(onsets),
            "duration": list(durations),
        }
    ).to_csv(sig_dir / "trial_signature_expression.tsv", sep="\t", index=False)


def _eeg_events(onsets) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": [1] * len(onsets),
            "trial_number": list(range(1, len(onsets) + 1)),
            "onset": list(onsets),
            "duration": [0.001] * len(onsets),
        }
    )


def test_signature_alignment_rejects_trials_that_only_share_an_identifier() -> None:
    """A shared trial index is an assertion about pairing, not evidence of one.

    Both tables number trials by position within a run, so one row dropped on either
    side renumbers everything after it while every key still matches. Only the protocol
    offset between the EEG trigger and the modeled plateau can tell the two apart.
    """
    from eeg_pipeline.utils.data.fmri_signature_targets import (
        load_fmri_signature_target_for_subject,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Same run and trial ids on both sides, 200 s apart in time.
        _write_signature_table(root, onsets=[213.0, 243.0], durations=[7.5, 7.5])

        with pytest.raises(ValueError, match="protocol offset"):
            load_fmri_signature_target_for_subject(
                subject_raw="0001",
                task="task",
                deriv_root=root,
                config=_signature_alignment_config(),
                events_df=_eeg_events([10.0, 40.0]),
                logger=logging.getLogger(__name__),
            )


def test_signature_alignment_accepts_trials_that_meet_the_protocol_offset() -> None:
    from eeg_pipeline.utils.data.fmri_signature_targets import (
        load_fmri_signature_target_for_subject,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Within the 10 ms residual the protocol allows.
        _write_signature_table(root, onsets=[13.004, 43.008], durations=[7.5, 7.5])

        y, _label, _extra = load_fmri_signature_target_for_subject(
            subject_raw="0001",
            task="task",
            deriv_root=root,
            config=_signature_alignment_config(),
            events_df=_eeg_events([10.0, 40.0]),
            logger=logging.getLogger(__name__),
        )

    assert np.allclose(np.asarray(y, dtype=float), np.array([5.0, 9.0]))


def test_signature_alignment_validates_plateau_duration_separately() -> None:
    from eeg_pipeline.utils.data.fmri_signature_targets import (
        load_fmri_signature_target_for_subject,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Onsets are right; the second trial models a different interval.
        _write_signature_table(root, onsets=[13.0, 43.0], durations=[7.5, 5.0])

        with pytest.raises(ValueError, match="plateau duration"):
            load_fmri_signature_target_for_subject(
                subject_raw="0001",
                task="task",
                deriv_root=root,
                config=_signature_alignment_config(),
                events_df=_eeg_events([10.0, 40.0]),
                logger=logging.getLogger(__name__),
            )
