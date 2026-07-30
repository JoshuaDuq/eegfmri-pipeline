"""Remove the ballistocardiogram at a given set of beats, and nowhere else.

Both methods take an array and return a corrected copy, so the identical call serves the
real correction and the sham control that measures what the procedure destroys when run at
beat times where no artifact sits.

Confinement matters as much as removal here: Analyzer's correction is already excellent at
the beats it marked (0.16% residual variance, 0 of 63 channels above null), so touching
those stretches can only make them worse.
"""

from __future__ import annotations

import numpy as np


def _epoch_bounds(beat_seconds, sfreq, window, n_times):
    pre = int(round(window[0] * sfreq))
    length = int(round((window[1] - window[0]) * sfreq))
    starts = np.round(np.asarray(beat_seconds, dtype=float) * sfreq).astype(int) + pre
    keep = (starts >= 0) & (starts + length <= n_times)
    return starts[keep], length


def _correct_aas(data_uv, beat_seconds, sfreq, window, n_neighbours):
    """Sliding-window average artifact subtraction (Allen et al. 1998).

    Each beat's template is the mean of its `n_neighbours` nearest epochs, which lets the
    template follow slow changes in the artifact rather than assuming one fixed shape.
    """
    starts, length = _epoch_bounds(beat_seconds, sfreq, window, data_uv.shape[1])
    if starts.size == 0:
        return data_uv.copy()
    index = starts[:, None] + np.arange(length)[None, :]
    epochs = data_uv[:, index]

    out = data_uv.copy()
    half = max(n_neighbours // 2, 1)
    for position in range(starts.size):
        lo = max(position - half, 0)
        hi = min(position + half + 1, starts.size)
        neighbours = np.delete(np.arange(lo, hi), np.where(np.arange(lo, hi) == position))
        if neighbours.size == 0:
            continue
        template = epochs[:, neighbours, :].mean(axis=1)
        out[:, index[position]] -= template
    return out


def _confine_to_epochs(original_uv, corrected_uv, beat_seconds, sfreq, window):
    """Keep the correction inside the beat epochs and restore the original elsewhere.

    `mne.preprocessing.apply_pca_obs` does not confine itself to the neighbourhoods of
    `qrs_times`. Measured on 120 s of noise carrying 8 beats, it changed all 120,000
    samples: up to 3.4 uV outside the epochs, on top of demeaning every channel over the
    whole recording. Splicing here is what makes the confinement guarantee hold, so the
    stretches Analyzer already corrected come back untouched.
    """
    starts, length = _epoch_bounds(beat_seconds, sfreq, window, original_uv.shape[1])
    out = original_uv.copy()
    if starts.size == 0:
        return out

    inside = np.zeros(original_uv.shape[1], dtype=bool)
    inside[(starts[:, None] + np.arange(length)[None, :]).ravel()] = True
    if (~inside).any():
        # Undo the whole-channel demeaning against the samples that should not have moved,
        # so the splice does not leave a step at every epoch boundary.
        offset = original_uv[:, ~inside].mean(axis=1) - corrected_uv[:, ~inside].mean(axis=1)
        corrected_uv = corrected_uv + offset[:, None]
    out[:, inside] = corrected_uv[:, inside]
    return out


def _correct_obs(data_uv, beat_seconds, sfreq, n_components, ch_names):
    """PCA optimal basis set (Niazy et al. 2005) via MNE."""
    import mne

    mne.set_log_level("ERROR")
    names = ch_names or [f"CH{i:03d}" for i in range(data_uv.shape[0])]
    info = mne.create_info(names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")
    mne.preprocessing.apply_pca_obs(
        raw,
        picks=names,
        qrs_times=np.asarray(beat_seconds, dtype=float),
        n_components=n_components,
        copy=False,
        verbose="ERROR",
    )
    return raw.get_data() * 1e6


def correct_beats(
    data_uv: np.ndarray,
    beat_seconds: np.ndarray,
    sfreq: float,
    *,
    method: str = "obs",
    n_components: int = 4,
    window: tuple[float, float] = (-0.3, 0.7),
    n_neighbours: int = 21,
    ch_names: list[str] | None = None,
) -> np.ndarray:
    """Corrected copy of `data_uv`, with the artifact removed at `beat_seconds`."""
    if method == "aas":
        return _correct_aas(data_uv, beat_seconds, sfreq, window, n_neighbours)
    if method == "obs":
        corrected = _correct_obs(data_uv, beat_seconds, sfreq, n_components, ch_names)
        return _confine_to_epochs(data_uv, corrected, beat_seconds, sfreq, window)
    raise ValueError(f"unknown method {method!r}; expected 'obs' or 'aas'")


def substitute_stretches(
    base_uv: np.ndarray,
    replacement_uv: np.ndarray,
    stretches: list[tuple[float, float]],
    sfreq: float,
) -> np.ndarray:
    """Copy named time ranges out of `replacement_uv` into `base_uv`.

    This is how Analyzer's correction is kept everywhere except the gaps.
    """
    if base_uv.shape != replacement_uv.shape:
        raise ValueError(
            f"shape mismatch: base {base_uv.shape} vs replacement {replacement_uv.shape}"
        )
    out = base_uv.copy()
    for start_s, end_s in stretches:
        lo = max(int(round(start_s * sfreq)), 0)
        hi = min(int(round(end_s * sfreq)), base_uv.shape[1])
        if hi > lo:
            out[:, lo:hi] = replacement_uv[:, lo:hi]
    return out
