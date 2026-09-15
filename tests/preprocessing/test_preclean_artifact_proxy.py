"""The peripheral artifact proxy must be measured before EEG cleaning removes it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")

from eeg_pipeline.utils.data import preprocessing as preproc  # noqa: E402
from tests.utils.pipelines_test_utils import DotConfig  # noqa: E402

SFREQ = 500.0
CHANNELS = ["Fp1", "Fp2", "Cz"]
TRIAL_ONSETS = [10.0, 40.0, 70.0]
BURST_TRIAL = 1  # zero-based index of the trial carrying the artifact burst


def _config() -> DotConfig:
    return DotConfig(
        {
            "epochs": {"tmin": -1.0, "tmax": 12.0},
            "eeg": {"ecg_channels": []},
            "preprocessing": {
                "clean_events_qc": {
                    "enabled": True,
                    "ecg_coupling": {"enabled": False, "channels": []},
                    "peripheral_low_gamma": {
                        "enabled": True,
                        "output_column": "fp1_fp2_high_frequency_power",
                        "channels": ["Fp1", "Fp2"],
                        "band": [70.0, 95.0],
                        "window": [3.0, 10.5],
                    },
                }
            },
        }
    )


def _write_filtered_run(path: Path, *, seed: int = 0) -> None:
    """A quiet recording with one 80 Hz frontal burst inside one trial's plateau."""
    rng = np.random.default_rng(seed)
    duration_s = 100.0
    times = np.arange(0.0, duration_s, 1.0 / SFREQ)
    data = rng.normal(0.0, 1e-7, size=(len(CHANNELS), times.size))

    burst_start = TRIAL_ONSETS[BURST_TRIAL] + 3.0
    burst = (times >= burst_start) & (times < burst_start + 7.0)
    data[:2, burst] += 5e-5 * np.sin(2.0 * np.pi * 80.0 * times[burst])

    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(path, overwrite=True, verbose=False)


def _write_bids_events(bids_root: Path) -> None:
    events_dir = bids_root / "sub-0001" / "eeg"
    events_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "onset": TRIAL_ONSETS,
            "duration": [0.001] * len(TRIAL_ONSETS),
            "trial_type": ["stim"] * len(TRIAL_ONSETS),
            "trial_number": [1, 2, 3],
        }
    ).to_csv(events_dir / "sub-0001_task-task_run-1_events.tsv", sep="\t", index=False)


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    bids_root = tmp_path / "bids"
    deriv_root = tmp_path / "derivatives"
    _write_bids_events(bids_root)
    _write_filtered_run(
        deriv_root
        / "preprocessed"
        / "eeg"
        / "sub-0001"
        / "eeg"
        / "sub-0001_task-task_run-1_proc-filt_raw.fif"
    )
    return bids_root, deriv_root


def test_proxy_is_measured_on_the_filtered_continuous_recording(tmp_path: Path) -> None:
    bids_root, deriv_root = _prepare(tmp_path)

    out_path = preproc.write_preclean_artifact_proxy(
        subject="0001",
        task="task",
        bids_root=bids_root,
        deriv_root=deriv_root,
        config=_config(),
        conditions=["stim"],
    )

    table = pd.read_csv(out_path, sep="\t")
    assert list(table["trial_number"]) == [1, 2, 3]
    assert list(table["run_id"]) == [1, 1, 1]

    power = table["fp1_fp2_high_frequency_power"].to_numpy(dtype=float)
    others = np.delete(power, BURST_TRIAL)
    assert power[BURST_TRIAL] > 100.0 * others.max()


def test_qc_table_reads_the_stored_proxy_rather_than_the_clean_epochs(tmp_path: Path) -> None:
    """ICA and AutoReject can remove the very artifact this covariate must quantify."""
    bids_root, deriv_root = _prepare(tmp_path)
    config = _config()
    preproc.write_preclean_artifact_proxy(
        subject="0001",
        task="task",
        bids_root=bids_root,
        deriv_root=deriv_root,
        config=config,
        conditions=["stim"],
    )
    stored = pd.read_csv(
        preproc.preclean_artifact_proxy_path(deriv_root=deriv_root, subject="0001", task="task"),
        sep="\t",
    )

    # Epochs that have been scrubbed clean: the burst is simply gone from them.
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    times = np.arange(-1.0, 12.0, 1.0 / SFREQ)
    clean = mne.EpochsArray(
        np.zeros((3, len(CHANNELS), times.size)), info, tmin=-1.0, verbose=False
    )
    kept = pd.DataFrame({"run_id": [1, 1, 1], "trial_number": [1, 2, 3]})

    qc_table = preproc._compute_clean_events_qc_table(
        epochs=clean,
        qc_cfg=preproc.CleanEventsQCConfig.from_config(config),
        kept=kept,
        proxy_path=preproc.preclean_artifact_proxy_path(
            deriv_root=deriv_root, subject="0001", task="task"
        ),
        context="test",
    )

    np.testing.assert_allclose(
        qc_table["fp1_fp2_high_frequency_power"].to_numpy(dtype=float),
        stored["fp1_fp2_high_frequency_power"].to_numpy(dtype=float),
    )
    assert qc_table["fp1_fp2_high_frequency_power"].to_numpy(dtype=float).max() > 0.0


def test_proxy_follows_trial_identifiers_through_epoch_rejection(tmp_path: Path) -> None:
    bids_root, deriv_root = _prepare(tmp_path)
    config = _config()
    preproc.write_preclean_artifact_proxy(
        subject="0001",
        task="task",
        bids_root=bids_root,
        deriv_root=deriv_root,
        config=config,
        conditions=["stim"],
    )
    proxy_path = preproc.preclean_artifact_proxy_path(
        deriv_root=deriv_root, subject="0001", task="task"
    )
    stored = pd.read_csv(proxy_path, sep="\t")

    # Trial 1 was rejected; the covariate must follow trial 2 and 3, not positions 0 and 1.
    kept = pd.DataFrame({"run_id": [1, 1], "trial_number": [2, 3]})
    aligned = preproc._preclean_artifact_proxy_for_events(
        kept=kept,
        column="fp1_fp2_high_frequency_power",
        proxy_path=proxy_path,
        context="test",
    )

    np.testing.assert_allclose(
        aligned, stored["fp1_fp2_high_frequency_power"].to_numpy(dtype=float)[1:]
    )


def test_a_missing_proxy_is_an_error_not_a_recomputation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pre-cleaning artifact proxy"):
        preproc._preclean_artifact_proxy_for_events(
            kept=pd.DataFrame({"run_id": [1], "trial_number": [1]}),
            column="fp1_fp2_high_frequency_power",
            proxy_path=tmp_path / "absent.tsv",
            context="test",
        )


def test_an_unmeasured_retained_trial_is_never_imputed(tmp_path: Path) -> None:
    bids_root, deriv_root = _prepare(tmp_path)
    preproc.write_preclean_artifact_proxy(
        subject="0001",
        task="task",
        bids_root=bids_root,
        deriv_root=deriv_root,
        config=_config(),
        conditions=["stim"],
    )
    proxy_path = preproc.preclean_artifact_proxy_path(
        deriv_root=deriv_root, subject="0001", task="task"
    )

    with pytest.raises(ValueError, match="no pre-cleaning artifact proxy row"):
        preproc._preclean_artifact_proxy_for_events(
            kept=pd.DataFrame({"run_id": [1, 1], "trial_number": [3, 99]}),
            column="fp1_fp2_high_frequency_power",
            proxy_path=proxy_path,
            context="test",
        )
