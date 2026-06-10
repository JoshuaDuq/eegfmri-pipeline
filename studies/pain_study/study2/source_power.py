"""sLORETA source-power extraction helpers for Study 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import hilbert

from studies.pain_study.study2.validation import finite_number


@dataclass(frozen=True)
class SourcePowerExtraction:
    power_logratio: np.ndarray
    baseline_window_s: tuple[float, float]
    active_window_s: tuple[float, float]
    n_trials: int
    n_vertices: int


def build_surface_forward_model(
    info: Any,
    *,
    subject: str,
    subjects_dir: str,
    trans: str,
    bem: str,
    spacing: str,
    mindist_mm: float,
) -> Any:
    """Build a surface-oriented EEG forward solution for source reconstruction.

    The surface-normal orientation matches the loose-orientation inverse used by
    ``compute_sloreta_source_estimates`` with ``pick_ori="normal"``.
    """
    import mne

    src = mne.setup_source_space(
        subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    forward = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=finite_number(mindist_mm, "study2.source_modeling.forward_mindist_mm"),
        verbose=False,
    )
    return mne.convert_forward_solution(
        forward,
        surf_ori=True,
        use_cps=True,
        copy=False,
        verbose=False,
    )


def compute_baseline_noise_covariance(
    epochs: Any,
    *,
    baseline_window_s: tuple[float, float],
    method: str = "empirical",
) -> Any:
    """Estimate the noise covariance from the pre-stimulus baseline window."""
    import mne

    start = finite_number(baseline_window_s[0], "study2 noise covariance baseline start")
    stop = finite_number(baseline_window_s[1], "study2 noise covariance baseline stop")
    if stop <= start:
        raise ValueError("Study 2 noise covariance baseline stop must be > start.")
    return mne.compute_covariance(
        epochs,
        tmin=start,
        tmax=stop,
        method=method,
        verbose=False,
    )


def make_sloreta_inverse_operator(
    *,
    info: Any,
    forward: Any,
    noise_cov: Any,
    loose: float,
    depth: float,
) -> Any:
    """Build the single loose-orientation inverse operator (README Section 4)."""
    if noise_cov is None:
        raise ValueError("Study 2 sLORETA source estimation requires noise_cov.")

    loose_value = finite_number(loose, "study2.source_modeling.regularization.loose_orientation")
    depth_value = finite_number(depth, "study2.source_modeling.regularization.depth_weighting")
    if loose_value < 0.0 or loose_value > 1.0:
        raise ValueError("Study 2 sLORETA loose orientation must be in [0, 1].")
    if depth_value < 0.0 or depth_value > 1.0:
        raise ValueError("Study 2 sLORETA depth weighting must be in [0, 1].")

    from mne.minimum_norm import make_inverse_operator

    return make_inverse_operator(
        info,
        forward,
        noise_cov,
        loose=loose_value,
        depth=depth_value,
        verbose=False,
    )


def apply_sloreta_inverse(
    *,
    epochs: Any,
    inverse_operator: Any,
    snr: float,
    pick_ori: str | None = "normal",
) -> list[Any]:
    """Apply a prebuilt sLORETA inverse operator to band-filtered epochs."""
    snr_value = finite_number(snr, "study2.source_modeling.regularization.snr")
    if snr_value <= 0.0:
        raise ValueError("Study 2 sLORETA snr must be positive.")

    from mne.minimum_norm import apply_inverse_epochs

    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=1.0 / snr_value**2,
        method="sLORETA",
        pick_ori=pick_ori,
        verbose=False,
    )
    return list(stcs)


def make_surface_source_morph(
    *,
    reference_stc: Any,
    subject_from: str,
    subject_to: str,
    subjects_dir: str,
    spacing: str,
) -> Any:
    """Build a surface morph into the configured common source space."""
    subject_from_label = _non_empty_string(subject_from, name="subject_from")
    subject_to_label = _non_empty_string(subject_to, name="subject_to")
    subjects_dir_path = _non_empty_string(subjects_dir, name="subjects_dir")
    spacing_name = _non_empty_string(spacing, name="source morph spacing")

    import mne

    target_source_space = mne.setup_source_space(
        subject_to_label,
        spacing=spacing_name,
        subjects_dir=subjects_dir_path,
        add_dist=False,
        verbose=False,
    )
    return mne.compute_source_morph(
        reference_stc,
        subject_from=subject_from_label,
        subject_to=subject_to_label,
        subjects_dir=subjects_dir_path,
        spacing=None,
        src_to=target_source_space,
        verbose=False,
    )


def apply_source_morph(
    stcs: list[Any] | tuple[Any, ...],
    *,
    morph: Any,
) -> list[Any]:
    """Apply a precomputed surface morph to every trial source estimate."""
    if not stcs:
        raise ValueError("Study 2 source morph requires at least one STC.")
    if morph is None:
        raise ValueError("Study 2 source morph requires a morph object.")
    return [morph.apply(stc) for stc in stcs]


def compute_sloreta_source_estimates(
    *,
    epochs: Any,
    forward: Any,
    noise_cov: Any,
    snr: float,
    loose: float,
    depth: float,
    pick_ori: str | None = "normal",
) -> tuple[list[Any], Any]:
    """Build the inverse operator and apply it to the given epochs."""
    if noise_cov is None:
        raise ValueError("Study 2 sLORETA source estimation requires noise_cov.")
    inverse_operator = make_sloreta_inverse_operator(
        info=epochs.info,
        forward=forward,
        noise_cov=noise_cov,
        loose=loose,
        depth=depth,
    )
    stcs = apply_sloreta_inverse(
        epochs=epochs,
        inverse_operator=inverse_operator,
        snr=snr,
        pick_ori=pick_ori,
    )
    return stcs, inverse_operator


def compute_sloreta_hilbert_logratio_power(
    *,
    stcs: list[Any] | tuple[Any, ...],
    times: np.ndarray,
    baseline_window_s: tuple[float, float],
    active_window_s: tuple[float, float],
    epsilon: float = 1.0e-12,
    chunk_size_vertices: int = 16,
) -> SourcePowerExtraction:
    """Convert band-limited sLORETA time series into trial-by-vertex log-ratio power."""
    time_values = _validate_times(times)
    baseline_mask = _window_mask(time_values, baseline_window_s, name="baseline")
    active_mask = _window_mask(time_values, active_window_s, name="active")
    epsilon_value = finite_number(epsilon, "source-power epsilon")
    if epsilon_value <= 0.0:
        raise ValueError("Study 2 source-power epsilon must be positive.")

    chunk_size = _positive_integer(
        chunk_size_vertices,
        name="source-power vertex chunk size",
    )
    stc_data = _validate_stc_data(stcs, n_times=time_values.size)
    logratio = _compute_chunked_logratio_power(
        stc_data,
        baseline_mask=baseline_mask,
        active_mask=active_mask,
        epsilon=epsilon_value,
        chunk_size_vertices=chunk_size,
    )
    if not np.all(np.isfinite(logratio)):
        raise ValueError("Study 2 sLORETA source-power log-ratio contains non-finite values.")
    return SourcePowerExtraction(
        power_logratio=logratio,
        baseline_window_s=(float(baseline_window_s[0]), float(baseline_window_s[1])),
        active_window_s=(float(active_window_s[0]), float(active_window_s[1])),
        n_trials=int(logratio.shape[0]),
        n_vertices=int(logratio.shape[1]),
    )


def _validate_times(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Study 2 source-power times must be 1D, got shape {values.shape}.")
    if values.size < 2:
        raise ValueError("Study 2 source-power times must contain at least two samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Study 2 source-power times contain non-finite values.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Study 2 source-power times must be strictly increasing.")
    return values


def _window_mask(
    times: np.ndarray,
    window: tuple[float, float],
    *,
    name: str,
) -> np.ndarray:
    if len(window) != 2:
        raise ValueError(f"Study 2 source-power {name} window must have two values.")
    start = finite_number(window[0], f"{name} window start")
    stop = finite_number(window[1], f"{name} window stop")
    if stop < start:
        raise ValueError(f"Study 2 source-power {name} window stop must be >= start.")
    mask = (times >= start) & (times <= stop)
    if not np.any(mask):
        raise ValueError(f"Study 2 source-power {name} window contains no samples.")
    return mask


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Study 2 {name} must be a positive integer.")
    return value


def _non_empty_string(value: object, *, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Study 2 {name} must be a non-empty string.")
    return text


def _validate_stc_data(
    stcs: list[Any] | tuple[Any, ...],
    *,
    n_times: int,
) -> tuple[np.ndarray, ...]:
    if not stcs:
        raise ValueError("Study 2 sLORETA source-power extraction requires at least one STC.")
    stc_data: list[np.ndarray] = []
    n_vertices: int | None = None
    for index, stc in enumerate(stcs):
        data = np.asarray(getattr(stc, "data", None), dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Study 2 STC data must be 2D for trial {index}.")
        if data.shape[1] != n_times:
            raise ValueError("Study 2 STC data time dimension must match times.")
        if n_vertices is None:
            n_vertices = int(data.shape[0])
        elif data.shape[0] != n_vertices:
            raise ValueError("Study 2 STC data vertex counts must match across trials.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Study 2 STC data contains non-finite values.")
        stc_data.append(data)
    return tuple(stc_data)


def _compute_chunked_logratio_power(
    stc_data: tuple[np.ndarray, ...],
    *,
    baseline_mask: np.ndarray,
    active_mask: np.ndarray,
    epsilon: float,
    chunk_size_vertices: int,
) -> np.ndarray:
    n_trials = len(stc_data)
    n_vertices = int(stc_data[0].shape[0])
    logratio = np.empty((n_trials, n_vertices), dtype=float)
    for start in range(0, n_vertices, chunk_size_vertices):
        stop = min(start + chunk_size_vertices, n_vertices)
        chunk = np.stack([data[start:stop, :] for data in stc_data], axis=0)
        analytic_power = np.abs(hilbert(chunk, axis=2)) ** 2
        baseline_power = np.mean(analytic_power[:, :, baseline_mask], axis=2)
        active_power = np.mean(analytic_power[:, :, active_mask], axis=2)
        logratio[:, start:stop] = np.log(
            (active_power + epsilon) / (baseline_power + epsilon)
        )
    return logratio


__all__ = [
    "SourcePowerExtraction",
    "apply_source_morph",
    "apply_sloreta_inverse",
    "build_surface_forward_model",
    "compute_baseline_noise_covariance",
    "compute_sloreta_hilbert_logratio_power",
    "compute_sloreta_source_estimates",
    "make_surface_source_morph",
    "make_sloreta_inverse_operator",
]
