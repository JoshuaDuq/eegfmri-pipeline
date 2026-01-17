"""
Visualization Statistics
========================

Statistics for visualization and plotting, including diagnostic plots
for permutation distributions, cluster-mass histograms, and p-p plots.
"""

from __future__ import annotations

from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import json
import datetime

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import stats

from .fdr import fdr_bh
from .validation import (
    compute_fwer_bonferroni,
    compute_fwer_holm,
    compute_fwer_sidak,
)


# Constants
DEFAULT_CORRELATION_VMAX = 0.5
KS_CRITICAL_ALPHA_05 = 1.36
MIN_DEGREES_OF_FREEDOM = 1
MIN_SAMPLE_SIZE_FOR_QQ = 3
MIN_SAMPLE_SIZE_FOR_KDE = 2
SHAPIRO_WILK_MAX_SAMPLE_SIZE = 5000
MAX_OBSERVED_MASSES_TO_RETURN = 100
ARRAY_TRUNCATION_LENGTH = 100
KDE_PADDING_FACTOR = 0.1


###################################################################
# Core Visualization Statistics
###################################################################


def compute_kde_scale(
    data: pd.Series,
    hist_bins: int = 15,
    kde_points: int = 100,
) -> float:
    """Compute KDE scaling factor for overlay on histogram."""
    hist_counts, _ = np.histogram(data, bins=hist_bins)
    kde = gaussian_kde(data)
    data_range = np.linspace(data.min(), data.max(), kde_points)
    kde_vals = kde(data_range)
    if kde_vals.max() > 0:
        return hist_counts.max() / kde_vals.max()
    return 1.0


def compute_correlation_vmax(data: Union[np.ndarray, List[Dict]]) -> float:
    """Compute symmetric vmax for correlation heatmaps."""
    if isinstance(data, np.ndarray):
        finite_vals = data[np.isfinite(data)]
        if len(finite_vals) == 0:
            return DEFAULT_CORRELATION_VMAX
        return max(abs(np.min(finite_vals)), abs(np.max(finite_vals)))
    
    all_corrs = []
    for bd in data:
        correlations = bd['correlations']
        sig_mask = bd.get('significant_mask', np.ones(len(correlations), dtype=bool))
        sig_corrs = correlations[sig_mask]
        finite_sig = sig_corrs[np.isfinite(sig_corrs)]
        if len(finite_sig) > 0:
            all_corrs.extend(finite_sig)
    
    if not all_corrs:
        for bd in data:
            correlations = bd['correlations']
            finite_corrs = correlations[np.isfinite(correlations)]
            all_corrs.extend(finite_corrs)
    
    if not all_corrs:
        return DEFAULT_CORRELATION_VMAX
    
    return max(abs(np.min(all_corrs)), abs(np.max(all_corrs)))


###################################################################
# Diagnostic Plot Data Computation
###################################################################


def compute_permutation_distribution_data(
    null_distribution: np.ndarray,
    observed_statistic: float,
    n_bins: int = 50,
) -> Dict[str, Any]:
    """Compute data for permutation distribution diagnostic plot.
    
    Returns dict with histogram data, observed marker, and summary stats.
    """
    null = np.asarray(null_distribution).ravel()
    null = null[np.isfinite(null)]
    
    if len(null) == 0:
        return {"error": "Empty null distribution"}
    
    # Histogram
    counts, bin_edges = np.histogram(null, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # P-value
    n_extreme = np.sum(np.abs(null) >= np.abs(observed_statistic))
    p_value = (n_extreme + 1) / (len(null) + 1)
    
    percentiles = [2.5, 5, 95, 97.5]
    pct_values = {f"p{p}": float(np.percentile(null, p)) for p in percentiles}
    
    return {
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "observed": float(observed_statistic),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_median": float(np.median(null)),
        "n_permutations": len(null),
        **pct_values,
    }


def compute_cluster_mass_histogram_data(
    cluster_masses: np.ndarray,
    null_masses: np.ndarray,
    n_bins: int = 30,
) -> Dict[str, Any]:
    """Compute data for cluster-mass histogram diagnostic plot.
    
    Shows distribution of observed cluster masses vs null distribution.
    """
    obs = np.asarray(cluster_masses).ravel()
    null = np.asarray(null_masses).ravel()
    
    obs = obs[np.isfinite(obs)]
    null = null[np.isfinite(null)]
    
    if len(null) == 0:
        return {"error": "Empty null distribution"}
    
    all_masses = np.concatenate([obs, null])
    bin_edges = np.linspace(all_masses.min(), all_masses.max(), n_bins + 1)
    
    null_counts, _ = np.histogram(null, bins=bin_edges, density=True)
    if len(obs) > 0:
        obs_counts, _ = np.histogram(obs, bins=bin_edges, density=True)
    else:
        obs_counts = np.zeros(n_bins)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    threshold_95 = float(np.percentile(null, 95))
    threshold_99 = float(np.percentile(null, 99))
    n_sig_95 = int(np.sum(obs > threshold_95)) if len(obs) > 0 else 0
    n_sig_99 = int(np.sum(obs > threshold_99)) if len(obs) > 0 else 0
    
    return {
        "bin_centers": bin_centers.tolist(),
        "null_counts": null_counts.tolist(),
        "obs_counts": obs_counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
        "n_observed_clusters": len(obs),
        "n_significant_95": n_sig_95,
        "n_significant_99": n_sig_99,
        "observed_masses": (
            obs[:MAX_OBSERVED_MASSES_TO_RETURN].tolist()
            if len(obs) >= MAX_OBSERVED_MASSES_TO_RETURN
            else obs.tolist()
        ),
    }


def compute_pp_plot_data(
    p_values: np.ndarray,
    n_points: int = 100,
) -> Dict[str, Any]:
    """Compute P-P plot data for uniformity check under the null.
    
    Under a true null, p-values should be uniformly distributed.
    Deviations indicate either true effects or model misspecification.
    """
    p = np.asarray(p_values).ravel()
    p = p[np.isfinite(p) & (p >= 0) & (p <= 1)]
    
    if len(p) == 0:
        return {"error": "No valid p-values"}
    
    p_sorted = np.sort(p)
    n = len(p_sorted)
    expected = (np.arange(1, n + 1) - 0.5) / n
    
    ks_stat, ks_p = stats.kstest(p, 'uniform')
    ks_critical_value = KS_CRITICAL_ALPHA_05 / np.sqrt(n)
    
    if n > n_points:
        idx = np.linspace(0, n - 1, n_points, dtype=int)
        expected_plot = expected[idx]
        observed_plot = p_sorted[idx]
    else:
        expected_plot = expected
        observed_plot = p_sorted
    
    return {
        "expected": expected_plot.tolist(),
        "observed": observed_plot.tolist(),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "uniform_rejected": ks_p < 0.05,
        "n_pvalues": n,
        "ci_band_width": float(ks_critical_value),
        "proportion_below_05": float(np.mean(p < 0.05)),
        "proportion_below_01": float(np.mean(p < 0.01)),
        "expected_below_05": 0.05,
        "expected_below_01": 0.01,
    }


def compute_qq_plot_data(
    data: np.ndarray,
    distribution: str = "norm",
) -> Dict[str, Any]:
    """Compute Q-Q plot data for distribution assessment.
    
    Parameters
    ----------
    data : array
        Sample data
    distribution : str
        'norm' for normal, 't' for t-distribution
    """
    x = np.asarray(data).ravel()
    x = x[np.isfinite(x)]
    
    if len(x) < MIN_SAMPLE_SIZE_FOR_QQ:
        return {"error": "Insufficient data"}
    
    x_sorted = np.sort(x)
    n = len(x_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    
    if distribution == "t":
        degrees_of_freedom = max(n - 1, MIN_DEGREES_OF_FREEDOM)
        theoretical = stats.t.ppf(probs, degrees_of_freedom)
        dist_name = f"t(df={degrees_of_freedom})"
    else:
        theoretical = stats.norm.ppf(probs)
        dist_name = "Normal"
    
    slope, intercept, r_value, _, _ = stats.linregress(theoretical, x_sorted)
    
    if len(x) <= SHAPIRO_WILK_MAX_SAMPLE_SIZE:
        sw_stat, sw_p = stats.shapiro(x)
    else:
        sw_stat, sw_p = np.nan, np.nan
    
    return {
        "theoretical": theoretical.tolist(),
        "sample": x_sorted.tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "distribution": dist_name,
        "shapiro_w": float(sw_stat) if not np.isnan(sw_stat) else None,
        "shapiro_p": float(sw_p) if not np.isnan(sw_p) else None,
        "n": n,
    }


###################################################################
# Effect Size Distribution Visualization
###################################################################


def compute_effect_size_distribution_data(
    effect_sizes: np.ndarray,
    ci_lows: Optional[np.ndarray] = None,
    ci_highs: Optional[np.ndarray] = None,
    n_bins: int = 30,
) -> Dict[str, Any]:
    """Compute data for effect size distribution visualization.
    
    Includes histogram, summary statistics, and optional CI data.
    """
    es = np.asarray(effect_sizes).ravel()
    es = es[np.isfinite(es)]
    
    if len(es) == 0:
        return {"error": "No valid effect sizes"}
    
    counts, bin_edges = np.histogram(es, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    summary = {
        "mean": float(np.mean(es)),
        "median": float(np.median(es)),
        "std": float(np.std(es)),
        "min": float(np.min(es)),
        "max": float(np.max(es)),
        "n": len(es),
        "n_positive": int(np.sum(es > 0)),
        "n_negative": int(np.sum(es < 0)),
        "n_small": int(np.sum(np.abs(es) < 0.2)),
        "n_medium": int(np.sum((np.abs(es) >= 0.2) & (np.abs(es) < 0.8))),
        "n_large": int(np.sum(np.abs(es) >= 0.8)),
    }
    
    result = {
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "summary": summary,
    }
    
    if ci_lows is not None and ci_highs is not None:
        ci_l = np.asarray(ci_lows).ravel()
        ci_h = np.asarray(ci_highs).ravel()
        valid_ci = np.isfinite(ci_l) & np.isfinite(ci_h)
        
        result["ci_widths"] = (ci_h[valid_ci] - ci_l[valid_ci]).tolist()
        result["n_ci_exclude_zero"] = int(np.sum((ci_l > 0) | (ci_h < 0)))
    
    return result


###################################################################
# Bootstrap Distribution Visualization
###################################################################


def compute_bootstrap_distribution_data(
    bootstrap_samples: np.ndarray,
    observed_statistic: float,
    ci_level: float = 0.95,
    n_bins: int = 50,
) -> Dict[str, Any]:
    """Compute data for bootstrap distribution visualization.
    
    Shows bootstrap distribution with CI bounds and observed value.
    """
    boot = np.asarray(bootstrap_samples).ravel()
    boot = boot[np.isfinite(boot)]
    
    if len(boot) == 0:
        return {"error": "Empty bootstrap distribution"}
    
    counts, bin_edges = np.histogram(boot, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    alpha = 1 - ci_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    ci_low = float(np.percentile(boot, lower_percentile))
    ci_high = float(np.percentile(boot, upper_percentile))
    
    boot_mean = float(np.mean(boot))
    bias = boot_mean - observed_statistic
    
    return {
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "observed": float(observed_statistic),
        "boot_mean": boot_mean,
        "boot_std": float(np.std(boot)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci_level,
        "bias": float(bias),
        "n_bootstrap": len(boot),
    }


###################################################################
# Raincloud/Half-Violin Plot Data
###################################################################


def compute_raincloud_data(
    groups: Dict[str, np.ndarray],
    kde_points: int = 100,
) -> Dict[str, Any]:
    """Compute data for raincloud/half-violin plots.
    
    Returns KDE curves, raw points, and summary statistics per group.
    """
    result = {"groups": {}}
    
    for name, data in groups.items():
        x = np.asarray(data).ravel()
        x = x[np.isfinite(x)]
        
        if len(x) < MIN_SAMPLE_SIZE_FOR_KDE:
            result["groups"][name] = {"error": "Insufficient data"}
            continue
        
        try:
            kde = gaussian_kde(x)
            data_range = np.ptp(x)
            padding = KDE_PADDING_FACTOR * data_range
            x_min = x.min() - padding
            x_max = x.max() + padding
            x_range = np.linspace(x_min, x_max, kde_points)
            kde_vals = kde(x_range)
        except (np.linalg.LinAlgError, ValueError):
            x_range = np.array([])
            kde_vals = np.array([])
        
        result["groups"][name] = {
            "raw_points": x.tolist(),
            "kde_x": x_range.tolist(),
            "kde_y": kde_vals.tolist(),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": float(np.std(x)),
            "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "n": len(x),
        }
    
    return result


###################################################################
# Spaghetti Plot Data (Within-Subject)
###################################################################


def compute_spaghetti_plot_data(
    subject_data: Dict[str, Dict[str, float]],
    conditions: List[str],
) -> Dict[str, Any]:
    """Compute data for paired spaghetti plots showing individual variability.
    
    Parameters
    ----------
    subject_data : dict
        {subject_id: {condition: value, ...}, ...}
    conditions : list
        Ordered list of condition names
    """
    subjects = list(subject_data.keys())
    n_subjects = len(subjects)
    n_conditions = len(conditions)
    
    data_matrix = np.full((n_subjects, n_conditions), np.nan)
    for i, subj in enumerate(subjects):
        for j, cond in enumerate(conditions):
            if cond in subject_data[subj]:
                data_matrix[i, j] = subject_data[subj][cond]
    
    group_means = np.nanmean(data_matrix, axis=0)
    group_stds = np.nanstd(data_matrix, axis=0)
    n_per_condition = np.sum(~np.isnan(data_matrix), axis=0)
    group_sems = group_stds / np.sqrt(n_per_condition)
    
    trajectories = []
    for i, subj in enumerate(subjects):
        traj = {
            "subject": subj,
            "values": data_matrix[i, :].tolist(),
            "complete": not np.any(np.isnan(data_matrix[i, :])),
        }
        trajectories.append(traj)
    
    return {
        "conditions": conditions,
        "subjects": subjects,
        "trajectories": trajectories,
        "group_means": group_means.tolist(),
        "group_stds": group_stds.tolist(),
        "group_sems": group_sems.tolist(),
        "n_subjects": n_subjects,
        "n_complete": int(np.sum([t["complete"] for t in trajectories])),
    }


###################################################################
# Multiple Comparison Visualization
###################################################################


def compute_correction_comparison_data(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute data showing how different corrections shift significance.
    
    Compares raw, Bonferroni, Holm, Šidák, and FDR-BH corrections.
    """
    p = np.asarray(p_values).ravel()
    p = p[np.isfinite(p)]
    n = len(p)
    
    if n == 0:
        return {"error": "No valid p-values"}
    
    bonf_adj, bonf_reject, bonf_alpha = compute_fwer_bonferroni(p, alpha)
    holm_adj, holm_reject = compute_fwer_holm(p, alpha)
    sidak_adj, sidak_reject, sidak_alpha = compute_fwer_sidak(p, alpha)
    fdr_adj = fdr_bh(p, alpha)
    fdr_reject = fdr_adj < alpha
    
    results = {
        "n_tests": n,
        "alpha": alpha,
        "methods": {
            "raw": {
                "n_significant": int(np.sum(p < alpha)),
                "threshold": alpha,
                "adjusted_p": p.tolist(),
            },
            "bonferroni": {
                "n_significant": int(np.sum(bonf_reject)),
                "threshold": float(bonf_alpha),
                "adjusted_p": bonf_adj.tolist(),
            },
            "holm": {
                "n_significant": int(np.sum(holm_reject)),
                "threshold": float(alpha / n),
                "adjusted_p": holm_adj.tolist(),
            },
            "sidak": {
                "n_significant": int(np.sum(sidak_reject)),
                "threshold": float(sidak_alpha),
                "adjusted_p": sidak_adj.tolist(),
            },
            "fdr_bh": {
                "n_significant": int(np.sum(fdr_reject)),
                "threshold": alpha,  # FDR uses q-values
                "adjusted_p": fdr_adj.tolist(),
            },
        },
    }
    
    sorted_idx = np.argsort(p)
    results["sorted_p"] = p[sorted_idx].tolist()
    results["sorted_idx"] = sorted_idx.tolist()
    
    return results


###################################################################
# Provenance Block for Reproducibility
###################################################################


def create_provenance_block(
    sample_size: int,
    test_type: str,
    correction_method: str,
    n_permutations: Optional[int] = None,
    random_seed: Optional[int] = None,
    alpha: float = 0.05,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create provenance block for figure annotation.
    
    Returns dict suitable for embedding in figures or saving as JSON.
    """
    provenance = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sample_size": sample_size,
        "test_type": test_type,
        "correction_method": correction_method,
        "alpha": alpha,
    }
    
    if n_permutations is not None:
        provenance["n_permutations"] = n_permutations
    
    if random_seed is not None:
        provenance["random_seed"] = random_seed
    
    if additional_info:
        provenance.update(additional_info)
    
    return provenance


def format_provenance_text(provenance: Dict[str, Any], max_chars: int = 100) -> str:
    """Format provenance block as compact text for figure annotation."""
    parts = []
    
    if "sample_size" in provenance:
        parts.append(f"n={provenance['sample_size']}")
    if "test_type" in provenance:
        parts.append(provenance["test_type"])
    if "correction_method" in provenance:
        parts.append(f"corr={provenance['correction_method']}")
    if "n_permutations" in provenance:
        parts.append(f"perm={provenance['n_permutations']}")
    if "random_seed" in provenance:
        parts.append(f"seed={provenance['random_seed']}")
    
    text = " | ".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."
    
    return text


def save_stats_for_plot(
    stats_dict: Dict[str, Any],
    plot_path: Path,
    output_format: str = "json",
) -> Path:
    """Save statistics used for a plot alongside the figure.
    
    Creates a companion file with same name but .json/.csv extension.
    """
    plot_path = Path(plot_path)
    
    if output_format == "json":
        stats_path = plot_path.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2, default=str)
    elif output_format == "csv":
        stats_path = plot_path.with_suffix(".stats.csv")
        flat_dict = {
            k: str(v)[:ARRAY_TRUNCATION_LENGTH] if isinstance(v, (list, np.ndarray)) else v
            for k, v in stats_dict.items()
        }
        pd.DataFrame([flat_dict]).to_csv(stats_path, index=False)
    else:
        raise ValueError(f"Unknown format: {output_format}")
    
    return stats_path


