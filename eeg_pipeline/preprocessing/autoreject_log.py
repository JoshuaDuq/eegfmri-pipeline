"""Persist AutoReject's channel-by-epoch verdict as a derivative.

MNE-BIDS-Pipeline fits AutoReject in ``_09_ptp_reject``, uses the reject log for one log
line and one report figure, and then discards it. That log is the only record of which
channel-in-trial samples are measured and which are spline estimates from neighbouring
electrodes, and the distinction matters downstream: an interpolated channel is by
construction a weighted sum of its neighbours, so any measure computed between it and
those neighbours -- coherence, wPLI, AEC, and anything built on a CSD transform -- is
partly determined by the interpolation rather than by the brain.

The log is reconstructed by fitting AutoReject with the same settings the pipeline passes
it, on the same pre-rejection epochs it reads. ``verify_log_describes_clean_epochs``
checks the reconstruction against the derivative it claims to describe, so a fit that
diverged from the pipeline's is an error rather than a silently wrong sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

#: Column names carried into the clean events table for each surviving trial.
INTERPOLATED_COLUMN = "n_channels_interpolated"
BAD_NOT_INTERPOLATED_COLUMN = "n_channels_bad_not_interpolated"

_EPOCH_INDEX_COLUMN = "epoch_index"
_BAD_EPOCH_COLUMN = "bad_epoch"

#: AutoReject's label encoding, preserved verbatim so the file needs no translation.
_GOOD = 0
_BAD = 1
_BAD_INTERPOLATED = 2


@dataclass(frozen=True)
class AutorejectLogSettings:
    """The AutoReject settings MNE-BIDS-Pipeline fits with."""

    n_interpolate: tuple[int, ...]
    random_state: int
    n_jobs: int

    @classmethod
    def from_config(cls, config: Any) -> "AutorejectLogSettings":
        n_interpolate = config.get("epochs.autoreject_n_interpolate")
        if not n_interpolate:
            raise ValueError(
                "epochs.autoreject_n_interpolate must be set to reconstruct the "
                "AutoReject log; it is the interpolation grid the pipeline fits with."
            )
        random_state = config.get("preprocessing.random_state")
        if random_state is None:
            raise ValueError(
                "preprocessing.random_state must be set to reconstruct the AutoReject "
                "log; without it the fit is not reproducible."
            )
        reject = config.get("epochs.reject")
        if reject != "autoreject_local":
            raise ValueError(
                "The AutoReject log describes a fit the pipeline performs only when "
                f"epochs.reject is 'autoreject_local'; it is {reject!r}. Disable "
                "preprocessing.autoreject_log or restore the rejection method."
            )
        return cls(
            n_interpolate=tuple(int(value) for value in n_interpolate),
            random_state=int(random_state),
            n_jobs=int(config.get("preprocessing.n_jobs", 1)),
        )


@dataclass(frozen=True)
class AutorejectLog:
    """AutoReject's verdict for every channel in every pre-rejection epoch."""

    ch_names: tuple[str, ...]
    #: (n_epochs, n_channels), values 0 good, 1 bad, 2 bad and interpolated.
    labels: np.ndarray
    #: (n_epochs,) True where the whole epoch was dropped.
    bad_epochs: np.ndarray
    #: The interpolation limit the cross-validation selected.
    n_interpolate: int
    #: The consensus fraction the cross-validation selected.
    consensus: float


def scored_channel_picks(epochs: mne.BaseEpochs) -> np.ndarray:
    """Return the channel indices AutoReject scores, in AutoReject's own order.

    AutoReject evaluates the data channels that are not marked bad. A channel bad for
    the whole recording is interpolated wholesale elsewhere, so it has no per-trial
    verdict to record.
    """
    return mne.pick_types(epochs.info, eeg=True, exclude="bads")


def compute_autoreject_log(
    epochs: mne.BaseEpochs,
    settings: AutorejectLogSettings,
) -> AutorejectLog:
    """Fit AutoReject on pre-rejection epochs and return its verdict."""
    import autoreject

    # The pipeline calls AutoReject with no ``picks``, and AutoReject's own default drops
    # ``info['bads']``. Passing explicit indices here would not: ``_picks_to_idx`` leaves
    # an integer array alone, so a channel bad for the whole recording would be scored,
    # inflating the per-epoch bad counts that the consensus cross-validation reads.
    picks = scored_channel_picks(epochs)
    if len(picks) == 0:
        raise ValueError("Cannot fit AutoReject: the epochs hold no EEG channels.")

    ar = autoreject.AutoReject(
        n_interpolate=np.array(settings.n_interpolate),
        random_state=settings.random_state,
        n_jobs=settings.n_jobs,
        verbose=False,
    )
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    # get_reject_log returns one column per channel in the recording and fills the
    # columns outside `picks` with NaN, so the EEG columns are selected here rather
    # than trusting the object's own channel names.
    labels = np.asarray(reject_log.labels, dtype=float)[:, picks]
    if not np.all(np.isfinite(labels)):
        raise ValueError("AutoReject returned a non-finite label for a picked channel.")

    return AutorejectLog(
        ch_names=tuple(epochs.ch_names[index] for index in picks),
        labels=labels.astype(int),
        bad_epochs=np.asarray(reject_log.bad_epochs, dtype=bool),
        n_interpolate=int(ar.n_interpolate_["eeg"]),
        consensus=float(ar.consensus_["eeg"]),
    )


def write_autoreject_log(log: AutorejectLog, path: Path) -> Path:
    """Write the label matrix as a TSV with a JSON sidecar for the fitted settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(log.labels, columns=list(log.ch_names))
    frame.insert(0, _BAD_EPOCH_COLUMN, log.bad_epochs.astype(int))
    frame.insert(0, _EPOCH_INDEX_COLUMN, range(len(frame)))
    frame.to_csv(path, sep="\t", index=False)

    _sidecar_path(path).write_text(
        json.dumps(
            {
                "Description": (
                    "AutoReject verdict per channel per pre-rejection epoch. "
                    "0 good, 1 bad and left in place, 2 bad and interpolated from "
                    "neighbouring electrodes across the whole epoch."
                ),
                "n_interpolate": log.n_interpolate,
                "consensus": log.consensus,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def read_autoreject_log(path: Path) -> AutorejectLog:
    """Read a log written by :func:`write_autoreject_log`."""
    path = Path(path)
    frame = pd.read_csv(path, sep="\t")
    sidecar = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))

    ch_names = [
        column for column in frame.columns if column not in {_EPOCH_INDEX_COLUMN, _BAD_EPOCH_COLUMN}
    ]
    return AutorejectLog(
        ch_names=tuple(ch_names),
        labels=frame[ch_names].to_numpy(dtype=int),
        bad_epochs=frame[_BAD_EPOCH_COLUMN].to_numpy(dtype=bool),
        n_interpolate=int(sidecar["n_interpolate"]),
        consensus=float(sidecar["consensus"]),
    )


def kept_epoch_counts(log: AutorejectLog) -> pd.DataFrame:
    """Per-trial repair counts for the epochs that survived rejection.

    One row per surviving epoch, in the order the clean derivative holds them, so the
    table concatenates directly onto the clean events rows.
    """
    kept = log.labels[~log.bad_epochs]
    return pd.DataFrame(
        {
            INTERPOLATED_COLUMN: (kept == _BAD_INTERPOLATED).sum(axis=1),
            BAD_NOT_INTERPOLATED_COLUMN: (kept == _BAD).sum(axis=1),
        }
    ).reset_index(drop=True)


def verify_log_describes_clean_epochs(
    log: AutorejectLog,
    clean_epochs: mne.BaseEpochs,
) -> None:
    """Fail unless the reconstructed log matches the derivative it annotates.

    The log is fitted separately from the pipeline's own AutoReject call, so agreement is
    checked rather than assumed: a reconstruction that kept a different set of epochs, or
    that saw a different montage, describes something other than the epochs on disk.
    """
    kept = int((~log.bad_epochs).sum())
    if kept != len(clean_epochs):
        raise ValueError(
            f"AutoReject log keeps {kept} epochs but the clean epochs file holds "
            f"{len(clean_epochs)}. The reconstructed fit does not describe this "
            "derivative."
        )

    clean_names = tuple(
        clean_epochs.ch_names[index] for index in scored_channel_picks(clean_epochs)
    )
    if clean_names != log.ch_names:
        raise ValueError(
            "AutoReject log channel names do not match the clean epochs: "
            f"{sorted(set(log.ch_names) ^ set(clean_names))} differ."
        )


def autoreject_log_path_for_epochs(epochs_path: Path) -> Path:
    """Return the log path that annotates a given epochs file, beside it."""
    epochs_path = Path(epochs_path)
    name = epochs_path.name
    for suffix in _EPOCHS_SUFFIXES:
        if name.endswith(suffix):
            return epochs_path.with_name(f"{name[: -len(suffix)]}_desc-autoreject_log.tsv")
    raise ValueError(f"Not an MNE epochs filename, cannot derive a log path: {name}")


def pre_rejection_epochs_path(
    clean_epochs_path: Path,
    spatial_filter: str | None = None,
) -> Path:
    """Return the epochs file AutoReject was fitted on, given the cleaned output.

    ``_09_ptp_reject`` reads ``processing=spatial_filter`` and writes a ``proc-clean``
    copy beside it, so the fit input carries whichever spatial filter ran before it --
    ``proc-ica`` for ICA, ``proc-ssp`` for SSP. Only when no spatial filter is configured
    does it read the epochs ``_07_make_epochs`` wrote, which carry no processing entity.

    Passing the wrong one is not a naming detail: the pre-ICA epochs still hold the
    artifacts ICA removed, so a fit on them rejects a different set of trials than the
    one that produced the derivative.
    """
    clean_epochs_path = Path(clean_epochs_path)
    name = clean_epochs_path.name
    fit_entity = f"_proc-{spatial_filter}" if spatial_filter else ""
    for suffix in ("_proc-cleaned_epo.fif", "_proc-clean_epo.fif", "_clean_epo.fif"):
        if name.endswith(suffix):
            return clean_epochs_path.with_name(f"{name[: -len(suffix)]}{fit_entity}_epo.fif")
    raise ValueError(f"Not a cleaned MNE epochs filename: {name}")


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


#: Ordered longest-first so `_proc-clean_epo.fif` is not truncated to `_proc-clean`.
_EPOCHS_SUFFIXES = (
    "_proc-cleaned_epo.fif",
    "_proc-clean_epo.fif",
    "_clean_epo.fif",
    "_epo.fif",
)


__all__ = [
    "BAD_NOT_INTERPOLATED_COLUMN",
    "INTERPOLATED_COLUMN",
    "AutorejectLog",
    "AutorejectLogSettings",
    "autoreject_log_path_for_epochs",
    "compute_autoreject_log",
    "kept_epoch_counts",
    "pre_rejection_epochs_path",
    "read_autoreject_log",
    "scored_channel_picks",
    "verify_log_describes_clean_epochs",
    "write_autoreject_log",
]
