"""Reporting interval helpers for Study 2 scalar summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from studies.pain_study.study2.statistics import as_finite_1d


@dataclass(frozen=True)
class BootstrapMeanInterval:
    mean: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    n_resamples: int,
    random_state: int = 0,
    confidence_level: float = 0.95,
) -> BootstrapMeanInterval:
    data = as_finite_1d(values, name="bootstrap values")
    if data.size < 2:
        raise ValueError("Study 2 bootstrap mean interval requires at least two values.")
    if isinstance(n_resamples, bool):
        raise TypeError("Study 2 bootstrap n_resamples must be an integer.")
    numeric_resamples = float(n_resamples)
    if not np.isfinite(numeric_resamples) or not numeric_resamples.is_integer():
        raise ValueError("Study 2 bootstrap n_resamples must be an integer.")
    resample_count = int(numeric_resamples)
    if resample_count <= 0:
        raise ValueError("Study 2 bootstrap n_resamples must be positive.")
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("Study 2 bootstrap confidence_level must be in (0, 1).")

    observed = float(np.mean(data))
    rng = np.random.default_rng(random_state)
    bootstrap = np.asarray(
        [
            float(np.mean(data[rng.integers(0, data.size, size=data.size)]))
            for _ in range(resample_count)
        ],
        dtype=float,
    )
    ci_low, ci_high = _bca_interval(
        data=data,
        observed=observed,
        bootstrap=bootstrap,
        confidence_level=confidence_level,
    )
    return BootstrapMeanInterval(
        mean=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=resample_count,
        confidence_level=float(confidence_level),
    )


def _bca_interval(
    *,
    data: np.ndarray,
    observed: float,
    bootstrap: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    if float(np.std(data, ddof=0)) <= 0.0:
        return observed, observed

    proportion_less = float(np.mean(bootstrap < observed))
    proportion_less = min(max(proportion_less, 1.0 / (2.0 * len(bootstrap))), 1.0)
    proportion_less = min(proportion_less, 1.0 - 1.0 / (2.0 * len(bootstrap)))
    bias_correction = float(stats.norm.ppf(proportion_less))

    jackknife = np.asarray(
        [float(np.mean(np.delete(data, index))) for index in range(data.size)],
        dtype=float,
    )
    jackknife_mean = float(np.mean(jackknife))
    centered = jackknife_mean - jackknife
    acceleration_denominator = 6.0 * float(np.sum(centered**2) ** 1.5)
    acceleration = 0.0
    if acceleration_denominator > 0.0:
        acceleration = float(np.sum(centered**3) / acceleration_denominator)

    alpha = (1.0 - confidence_level) / 2.0
    quantiles = [
        _bca_quantile(alpha, bias_correction=bias_correction, acceleration=acceleration),
        _bca_quantile(1.0 - alpha, bias_correction=bias_correction, acceleration=acceleration),
    ]
    return (
        float(np.quantile(bootstrap, quantiles[0])),
        float(np.quantile(bootstrap, quantiles[1])),
    )


def _bca_quantile(alpha: float, *, bias_correction: float, acceleration: float) -> float:
    z_alpha = float(stats.norm.ppf(alpha))
    denominator = 1.0 - acceleration * (bias_correction + z_alpha)
    if denominator == 0.0:
        raise ValueError("Study 2 BCa interval has zero denominator.")
    adjusted = bias_correction + (bias_correction + z_alpha) / denominator
    return float(np.clip(stats.norm.cdf(adjusted), 0.0, 1.0))


__all__ = ["BootstrapMeanInterval", "bootstrap_mean_interval"]
