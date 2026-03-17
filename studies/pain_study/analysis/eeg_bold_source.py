"""Source-estimation helpers for pain EEG-BOLD coupling."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, List, Mapping, Tuple

import numpy as np

from eeg_pipeline.analysis.features.source_localization import (
    _compute_eloreta_source_estimates,
)

_PSD_REL_TOL = 1.0e-12
_MAX_CONDITION_NUMBER = 1.0e8


def _covariance_eigenvalue_bounds(cov: Any) -> Tuple[float, float]:
    data = np.asarray(cov["data"], dtype=float)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError("Covariance data must be a square matrix.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (data + data.T))
    return float(np.min(eigenvalues)), float(np.max(eigenvalues))


def _stabilize_covariance(
    *,
    cov: Any,
    label: str,
    logger: logging.Logger,
) -> Any:
    regularized = cov.copy()
    data = np.asarray(regularized["data"], dtype=float)
    symmetric = 0.5 * (data + data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(np.max(eigenvalues))
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError(f"{label} covariance has non-positive spectrum.")
    smallest = float(np.min(eigenvalues))
    psd_tolerance = max(float(np.finfo(float).eps), largest * _PSD_REL_TOL)
    if smallest < -psd_tolerance:
        raise ValueError(
            f"{label} covariance is not positive semidefinite "
            f"(min eigenvalue={smallest:.3e})."
        )
    clipped_eigenvalues = np.maximum(eigenvalues, 0.0)
    keep_mask = clipped_eigenvalues > psd_tolerance
    if not np.any(keep_mask):
        raise ValueError(f"{label} covariance has no positive spectrum above tolerance.")
    effective_eigenvalues = clipped_eigenvalues[keep_mask]
    smallest_effective = float(np.min(effective_eigenvalues))
    condition_number = largest / smallest_effective
    if not np.isfinite(condition_number) or condition_number > _MAX_CONDITION_NUMBER:
        raise ValueError(
            f"{label} covariance is too ill-conditioned for LCMV "
            f"(condition number={condition_number:.3e})."
        )
    conditioned = (eigenvectors * clipped_eigenvalues) @ eigenvectors.T
    regularized["data"] = conditioned
    regularized["eig"] = None
    regularized["eigvec"] = None
    regularized["method"] = f"{cov.get('method', 'empirical')}_validated_psd"
    if smallest < 0:
        logger.info(
            "%s covariance had roundoff-scale negative eigenvalues "
            "(min=%.3e, tolerance=%.3e); clipped to zero.",
            label,
            smallest,
            psd_tolerance,
        )
    min_eig, max_eig = _covariance_eigenvalue_bounds(regularized)
    logger.info(
        "%s covariance eigenvalue bounds after validation: min=%.3e max=%.3e "
        "(effective rank=%d, condition=%.3e)",
        label,
        min_eig,
        max_eig,
        int(np.sum(keep_mask)),
        condition_number,
    )
    return regularized


def _compute_band_covariance(
    *,
    epochs: Any,
    tmin: float | None,
    tmax: float | None,
    rank: str,
) -> Any:
    import mne

    return mne.compute_covariance(
        epochs,
        tmin=tmin,
        tmax=tmax,
        method="oas",
        rank=rank,
        verbose=False,
    )


def iterate_band_specific_lcmv_estimates(
    *,
    epochs: Any,
    fwd: Any,
    band: str,
    frequency_bands: Mapping[str, Tuple[float, float]],
    baseline_window: Tuple[float, float],
    reg: float,
    logger: logging.Logger,
) -> Iterable[Any]:
    from mne.beamformer import apply_lcmv_epochs, make_lcmv

    if band not in frequency_bands:
        raise ValueError(f"Unknown frequency band {band!r}.")
    fmin, fmax = float(frequency_bands[band][0]), float(frequency_bands[band][1])
    band_epochs = epochs.copy().filter(
        l_freq=fmin,
        h_freq=fmax,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        phase="zero",
        verbose=False,
    )
    data_cov = _compute_band_covariance(
        epochs=band_epochs,
        tmin=None,
        tmax=None,
        rank="info",
    )
    noise_cov = _compute_band_covariance(
        epochs=band_epochs,
        tmin=float(baseline_window[0]),
        tmax=float(baseline_window[1]),
        rank="info",
    )
    data_cov = _stabilize_covariance(
        cov=data_cov,
        label=f"{band} data",
        logger=logger,
    )
    noise_cov = _stabilize_covariance(
        cov=noise_cov,
        label=f"{band} noise",
        logger=logger,
    )
    filters = make_lcmv(
        band_epochs.info,
        fwd,
        data_cov,
        reg=reg,
        noise_cov=noise_cov,
        pick_ori="normal",
        weight_norm="unit-noise-gain",
        rank="info",
        verbose=False,
    )
    logger.info(
        "LCMV %s: streaming %d epochs with %d channels",
        band,
        len(band_epochs),
        len(band_epochs.ch_names),
    )
    return apply_lcmv_epochs(
        band_epochs,
        filters,
        return_generator=True,
        verbose=False,
    )


def compute_source_estimates(
    *,
    epochs: Any,
    fwd: Any,
    method: str,
    baseline_window: Tuple[float, float],
    reg: float,
    loose: float,
    depth: float,
    snr: float,
    logger: logging.Logger,
) -> List[Any]:
    method_name = str(method).strip().lower()
    if method_name == "lcmv":
        raise ValueError(
            "LCMV source estimates must be computed with iterate_band_specific_lcmv_estimates()."
        )
    if method_name == "eloreta":
        stcs, _ = _compute_eloreta_source_estimates(
            epochs=epochs,
            fwd=fwd,
            loose=loose,
            depth=depth,
            snr=snr,
            pick_ori="normal",
            logger=logger,
        )
        return stcs
    return _compute_minimum_norm_source_estimates(
        epochs=epochs,
        fwd=fwd,
        method=method_name,
        baseline_window=baseline_window,
        loose=loose,
        depth=depth,
        snr=snr,
        logger=logger,
    )


def _compute_minimum_norm_source_estimates(
    *,
    epochs: Any,
    fwd: Any,
    method: str,
    baseline_window: Tuple[float, float],
    loose: float,
    depth: float,
    snr: float,
    logger: logging.Logger,
) -> List[Any]:
    import mne
    from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator

    method_map = {
        "dspm": "dSPM",
        "wmne": "MNE",
    }
    if method not in method_map:
        raise ValueError(
            "Source method must be one of {'lcmv','eloreta','dspm','wmne'}."
        )

    baseline_epochs = epochs.copy().crop(
        tmin=float(baseline_window[0]),
        tmax=float(baseline_window[1]),
    )
    noise_cov = mne.compute_covariance(
        baseline_epochs,
        method="empirical",
        keep_sample_mean=False,
        verbose=False,
    )
    inv = make_inverse_operator(
        epochs.info,
        fwd,
        noise_cov,
        loose=loose,
        depth=depth,
        verbose=False,
    )
    lambda2 = 1.0 / float(snr) ** 2
    stcs = apply_inverse_epochs(
        epochs,
        inv,
        lambda2=lambda2,
        method=method_map[method],
        pick_ori="normal",
        verbose=False,
    )
    logger.info(
        "%s: %d epochs, %d sources",
        method_map[method],
        len(stcs),
        int(np.asarray(stcs[0].data).shape[0]),
    )
    return stcs


__all__ = ["iterate_band_specific_lcmv_estimates", "compute_source_estimates"]
