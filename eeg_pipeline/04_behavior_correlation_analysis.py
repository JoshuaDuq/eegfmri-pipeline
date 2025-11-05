from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    _load_features_and_targets,
    get_available_subjects,
    load_epochs_for_analysis,
    parse_subject_args,
)
from eeg_pipeline.utils.io_utils import (
    deriv_plots_path,
    deriv_stats_path,
    get_subject_logger,
    setup_matplotlib,
    ensure_dir,
)
from eeg_pipeline.plotting.plot_behavior import (
    plot_psychometrics,
    plot_power_roi_scatter,
    plot_power_behavior_correlation,
    plot_power_behavior_correlation_matrix,
    plot_significant_correlations_topomap,
    plot_behavioral_response_patterns,
    plot_power_spectrogram_with_behavior,
    plot_power_spectrogram_temperature_band,
    plot_behavior_modulated_connectivity,
    plot_top_behavioral_predictors,
    plot_time_frequency_correlation_heatmap,
    plot_power_behavior_evolution_across_trials,
)
from eeg_pipeline.analysis.behavior import (
    compute_power_roi_stats as _compute_power_roi_stats,
    compute_time_frequency_correlations as _compute_time_frequency_correlations,
    correlate_power_topomaps as _correlate_power_topomaps,
    correlate_connectivity_roi_summaries as _correlate_connectivity_roi_summaries,
    correlate_connectivity_heatmaps as _correlate_connectivity_heatmaps,
    export_all_significant_predictors as _export_all_significant_predictors,
    export_combined_power_corr_stats as _export_combined_power_corr_stats,
    apply_global_fdr as _apply_global_fdr,
)
from eeg_pipeline.analysis.group_aggregation import aggregate_behavior_correlations


###################################################################
# Configuration Constants
###################################################################

PARTIAL_COVARS_DEFAULT: List[str] = []
PLOT_SUBDIR = "04_behavior_correlation_analysis"


###################################################################
# Wrapper Functions: Power ROI Statistics
###################################################################

def compute_power_roi_stats(
    subject: str,
    deriv_root: Path,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    return _compute_power_roi_stats(
        subject, deriv_root, task, use_spearman, partial_covars, bootstrap, n_perm, rng
    )


def compute_time_frequency_correlations(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    *,
    use_spearman: bool = True,
) -> Optional[Path]:
    return _compute_time_frequency_correlations(subject, task, deriv_root, config, use_spearman=use_spearman)


###################################################################
# Wrapper Functions: Channel-Level Power Correlations
###################################################################

def correlate_power_topomaps(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    return _correlate_power_topomaps(subject, task, use_spearman, partial_covars, bootstrap, n_perm, rng)


###################################################################
# Wrapper Functions: Connectivity Correlations
###################################################################

def correlate_connectivity_roi_summaries(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    return _correlate_connectivity_roi_summaries(
        subject, task, use_spearman, partial_covars, bootstrap, n_perm, rng
    )


def correlate_connectivity_heatmaps(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
) -> None:
    return _correlate_connectivity_heatmaps(subject, task, use_spearman)


###################################################################
# Wrapper Functions: Export and Statistics
###################################################################

def export_all_significant_predictors(
    subject: str,
    alpha: float = 0.05,
    use_fdr: bool = True,
) -> None:
    return _export_all_significant_predictors(subject, alpha, use_fdr)


def export_combined_power_corr_stats(subject: str) -> None:
    return _export_combined_power_corr_stats(subject)


def apply_global_fdr(subject: str, alpha: float = 0.05) -> None:
    return _apply_global_fdr(subject, alpha)


###################################################################
# Subject Processing
###################################################################

def process_subject(
    subject: str,
    deriv_root: Path,
    task: str,
    config,
    correlation_method: str = "spearman",
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng_seed: int = 42,
) -> None:
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"=== Behavior-feature analyses: sub-{subject}, task-{task} ===")
    
    use_spearman = (correlation_method == "spearman")
    rng = np.random.default_rng(rng_seed)
    
    plot_psychometrics(subject, deriv_root, task, config)
    
    power_bands_to_use = config.get("features.frequency_bands", ["delta", "theta", "alpha", "beta", "gamma"])
    
    _temporal_df, pow_df, _, y, _ = _load_features_and_targets(subject, task, deriv_root, config)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    ensure_dir(plots_dir)
    plot_power_behavior_correlation(pow_df, y, power_bands_to_use, subject, plots_dir, logger, config=config)

    _temporal_df, pow_df, _, y, info = _load_features_and_targets(subject, task, deriv_root, config)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    ensure_dir(plots_dir)
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False, deriv_root=deriv_root, bids_root=config.bids_root, config=config
    )
    if epochs is None:
        aligned_events = None
    
    plot_power_behavior_correlation_matrix(pow_df, y, power_bands_to_use, subject, plots_dir, logger, config=config)
    plot_significant_correlations_topomap(pow_df, y, power_bands_to_use, info, subject, plots_dir, logger, config=config)
    plot_behavioral_response_patterns(y, aligned_events, subject, plots_dir, logger, config=config)
    plot_power_spectrogram_with_behavior(pow_df, y, power_bands_to_use, subject, plots_dir, logger, config=config)
    plot_power_spectrogram_temperature_band(pow_df, aligned_events, power_bands_to_use, subject, plots_dir, logger, config=config)
    plot_power_behavior_evolution_across_trials(subject, task, window_size=20, bands_to_plot=None)
    
    tf_data_path = compute_time_frequency_correlations(
        subject, task, deriv_root, config, use_spearman=use_spearman
    )
    if tf_data_path is not None:
        plot_time_frequency_correlation_heatmap(subject, task, data_path=tf_data_path)
    else:
        logger.warning("Skipping TF correlation heatmap due to missing analysis outputs")
    
    plot_behavior_modulated_connectivity(subject, task, y, plots_dir, logger, config=config)
    correlate_connectivity_heatmaps(subject, task, use_spearman=use_spearman)
    correlate_connectivity_roi_summaries(
        subject, task, use_spearman=use_spearman, partial_covars=partial_covars, bootstrap=bootstrap, n_perm=n_perm, rng=rng
    )

    compute_power_roi_stats(
        subject, deriv_root, task=task, use_spearman=use_spearman, partial_covars=partial_covars,
        bootstrap=bootstrap, n_perm=n_perm, rng=rng
    )
    
    stats_dir = deriv_stats_path(deriv_root, subject)
    rating_stats: Optional[pd.DataFrame] = None
    temp_stats: Optional[pd.DataFrame] = None
    rating_path = stats_dir / "corr_stats_pow_roi_vs_rating.tsv"
    temp_path = stats_dir / "corr_stats_pow_roi_vs_temp.tsv"
    
    if rating_path.exists():
        try:
            rating_stats = pd.read_csv(rating_path, sep="\t")
        except Exception as exc:
            logger.error(f"Failed to read ROI rating stats at {rating_path}: {exc}")
    
    if temp_path.exists():
        try:
            temp_stats = pd.read_csv(temp_path, sep="\t")
        except Exception as exc:
            logger.error(f"Failed to read ROI temperature stats at {temp_path}: {exc}")

    plot_power_roi_scatter(
        subject, deriv_root, task=task, use_spearman=use_spearman, partial_covars=partial_covars,
        do_temp=True, bootstrap_ci=bootstrap, rng=rng, rating_stats=rating_stats, temp_stats=temp_stats
    )

    correlate_power_topomaps(
        subject, task, use_spearman=use_spearman, partial_covars=partial_covars, bootstrap=bootstrap, n_perm=0, rng=rng
    )
    export_combined_power_corr_stats(subject)
    plot_top_behavioral_predictors(subject, task)
    export_all_significant_predictors(subject, alpha=0.05, use_fdr=True)
    apply_global_fdr(subject)


###################################################################
# Main Entry Point
###################################################################

def main(
    subjects=None,
    task=None,
    correlation_method=None,
    partial_covars=None,
    bootstrap=0,
    n_perm=0,
    do_group=False,
    group_only=False,
    rng_seed=42,
    all_subjects=False,
    pooling_strategy="within_subject_centered",
    cluster_bootstrap=0,
    group_subject_fixed_effects=True,
):
    config = load_settings()
    setup_matplotlib(config)
    
    deriv_root = Path(config.deriv_root)
    from eeg_pipeline.utils.io_utils import ensure_derivatives_dataset_description
    ensure_derivatives_dataset_description(deriv_root=deriv_root)
    
    if task is None:
        task = config.task
    if correlation_method is None:
        correlation_method = config.get("behavior_analysis.statistics.correlation_method", "spearman")
    if partial_covars is None:
        partial_covars = config.get("behavior_analysis.statistics.partial_covariates", [])
    if n_perm is None:
        n_perm = config.get("behavior_analysis.statistics.n_permutations", 0)
    
    if all_subjects:
        available = get_available_subjects(deriv_root=deriv_root, task=task, config=config)
        subjects = [
            s for s in available
            if (deriv_root / f"sub-{s}" / "eeg" / "features" / "features_eeg_direct.tsv").exists()
            and (deriv_root / f"sub-{s}" / "eeg" / "features" / "target_vas_ratings.tsv").exists()
        ]
        if not subjects:
            raise ValueError(f"No subjects with features found in {deriv_root}")
    elif not subjects:
        raise ValueError("No subjects specified. Use --group all|A,B,C, or --subject (can repeat), or --all-subjects.")
    
    if not group_only:
        for sub in subjects:
            process_subject(
                sub, deriv_root, task, config=config, correlation_method=correlation_method,
                partial_covars=partial_covars, bootstrap=bootstrap, n_perm=n_perm, rng_seed=rng_seed
            )
    
    if do_group or group_only:
        aggregate_behavior_correlations(
            subjects, task, deriv_root, pooling_strategy=pooling_strategy,
            cluster_bootstrap=int(cluster_bootstrap), subject_fixed_effects=bool(group_subject_fixed_effects), config=config
        )


###################################################################
# Command-Line Interface
###################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral psychometrics and EEG feature correlations")
    
    sel = parser.add_mutually_exclusive_group(required=False)
    sel.add_argument("--group", type=str, help="Group to process: 'all' or comma-separated subject labels")
    sel.add_argument("--subject", "-s", type=str, action="append", help="Subject label(s) without 'sub-' prefix")
    sel.add_argument("--all-subjects", action="store_true", help="Process all available subjects")
    
    parser.add_argument("--subjects", nargs="*", default=None, help="[Deprecated] Subject IDs list")
    parser.add_argument("--task", default=None, help="Task label (default from config)")
    parser.add_argument(
        "--correlation-method", choices=["spearman", "pearson"], default=None,
        help="Correlation method (default from config: behavior_analysis.statistics.correlation_method)"
    )
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap iterations (default: 0)")
    parser.add_argument("--n-perm", type=int, default=None, help="Number of permutations (default from config)")
    parser.add_argument("--rng-seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--pooling-strategy",
        choices=["within_subject_centered", "within_subject_zscored", "fisher_by_subject"],
        default="within_subject_centered",
        help="Group-level pooling strategy"
    )
    parser.add_argument("--cluster-bootstrap", type=int, default=0, help="Cluster bootstrap iterations (default: 0)")
    parser.add_argument(
        "--group-subject-fixed-effects", action="store_true", default=True,
        help="Use subject fixed effects in group analysis"
    )
    
    args = parser.parse_args()
    
    config = load_settings()
    deriv_root = Path(config.deriv_root)
    
    subjects = parse_subject_args(args, config, task=args.task, deriv_root=deriv_root)
    
    if not subjects:
        print("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        raise SystemExit(2)
    
    pooling_map = {
        "within_subject_centered": "within_subject_centered",
        "within_subject_zscored": "within_subject_zscored",
        "fisher_by_subject": "fisher_by_subject",
    }
    
    corr_method = args.correlation_method if hasattr(args, "correlation_method") and args.correlation_method else None
    bootstrap_val = args.bootstrap if args.bootstrap is not None else 0
    
    main(
        subjects, task=args.task, correlation_method=corr_method, partial_covars=None,
        bootstrap=bootstrap_val, n_perm=None, do_group=(len(subjects) > 1) if subjects else False,
        group_only=False, rng_seed=args.rng_seed, all_subjects=False,
        pooling_strategy=pooling_map[args.pooling_strategy], cluster_bootstrap=int(args.cluster_bootstrap),
        group_subject_fixed_effects=bool(getattr(args, "group_subject_fixed_effects", True))
    )
