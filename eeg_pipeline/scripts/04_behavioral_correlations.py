from __future__ import annotations

# Standard library
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import hashlib

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Third-party
import numpy as np
import pandas as pd
import mne

# Local - config and data
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

# Local - analysis
from eeg_pipeline.analysis.behavior import (
    apply_global_fdr,
    compute_power_roi_stats,
    correlate_connectivity_heatmaps,
    correlate_connectivity_roi_summaries,
    correlate_power_topomaps,
    export_all_significant_predictors,
    export_combined_power_corr_stats,
)
from eeg_pipeline.analysis.group_aggregation import aggregate_behavior_correlations

# Local - plotting
from eeg_pipeline.plotting.plot_behavioral import (
    plot_power_roi_scatter,
    plot_significant_correlations_topomap,
    plot_behavioral_response_patterns,
    plot_behavior_modulated_connectivity,
    plot_top_behavioral_predictors,
    plot_time_frequency_correlation_heatmap,
)


###################################################################
# Configuration Constants
###################################################################

PARTIAL_COVARS_DEFAULT: List[str] = []
PLOT_SUBDIR = "04_behavior_correlations"




###################################################################
# Subject Processing
###################################################################

def _get_subject_seed(base_seed: int, subject: str) -> int:
    """Generate a deterministic subject-specific seed from base seed and subject ID.
    
    Combines the base seed with a hash of the subject identifier to ensure
    each subject gets a unique but deterministic random sequence.
    """
    subject_bytes = subject.encode('utf-8')
    subject_hash = int(hashlib.md5(subject_bytes).hexdigest()[:8], 16) & 0x7FFFFFFF
    return (base_seed + subject_hash) % (2**31)


def _run_power_behavior_correlations(
    subject: str,
    deriv_root: Path,
    task: str,
    config,
    plots_dir: Path,
    logger: logging.Logger,
    epochs: Optional[mne.Epochs] = None,
) -> None:
    _, pow_df, _, y, info = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)
    power_bands = config.get("features.frequency_bands", ["delta", "theta", "alpha", "beta", "gamma"])
    
    plot_significant_correlations_topomap(
        pow_df, y, power_bands, info, subject, plots_dir, logger, config=config
    )


def _run_time_frequency_correlations(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    use_spearman: bool,
    logger: logging.Logger,
) -> None:
    stats_dir = deriv_stats_path(deriv_root, subject)
    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"
    behavior_config = config.get("behavior_analysis", {})
    heatmap_config = behavior_config.get("time_frequency_heatmap", {})
    roi_selection = heatmap_config.get("roi_selection")
    roi_suffix = f"_{roi_selection.lower()}" if roi_selection and roi_selection != "null" else ""
    tf_data_path = stats_dir / f"time_frequency_correlation_data{roi_suffix}{use_spearman_suffix}.npz"
    
    if tf_data_path.exists():
        plot_time_frequency_correlation_heatmap(subject, task, data_path=tf_data_path)
    else:
        logger.debug(f"TF correlation heatmap skipped: data file not found at {tf_data_path}")


def _run_connectivity_correlations(
    subject: str,
    task: str,
    y: pd.Series,
    plots_dir: Path,
    logger: logging.Logger,
    config,
    use_spearman: bool,
    partial_covars: Optional[List[str]],
    bootstrap: int,
    n_perm: int,
    rng: np.random.Generator,
) -> None:
    plot_behavior_modulated_connectivity(subject, task, y, plots_dir, logger, config=config)
    correlate_connectivity_heatmaps(subject, task, use_spearman=use_spearman)
    correlate_connectivity_roi_summaries(
        subject, task, use_spearman=use_spearman,
        partial_covars=partial_covars, bootstrap=bootstrap,
        n_perm=n_perm, rng=rng
    )


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
    skip_scatter: bool = False,
    no_plots: bool = False,
) -> None:
    if not subject or not isinstance(subject, str):
        raise ValueError(f"subject must be non-empty string, got: {subject}")
    if not task or not isinstance(task, str):
        raise ValueError(f"task must be non-empty string, got: {task}")
    
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"=== Behavior-feature analyses: sub-{subject}, task-{task} ===")
    
    use_spearman = (correlation_method == "spearman")
    subject_seed = _get_subject_seed(rng_seed, subject)
    rng = np.random.default_rng(subject_seed)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=PLOT_SUBDIR)
    ensure_dir(plots_dir)
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False, deriv_root=deriv_root, bids_root=config.bids_root, config=config
    )
    if epochs is None:
        aligned_events = None
    
    _run_power_behavior_correlations(subject, deriv_root, task, config, plots_dir, logger, epochs=epochs)
    
    _, pow_df, _, y, info = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)
    
    plot_behavioral_response_patterns(y, subject, plots_dir, logger, config=config)
    
    _run_time_frequency_correlations(subject, task, deriv_root, config, use_spearman, logger)
    _run_connectivity_correlations(
        subject, task, y, plots_dir, logger, config,
        use_spearman, partial_covars, bootstrap, n_perm, rng
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

    if no_plots:
        logger.info("Skipping all plotting operations (--no-plots enabled)")
    else:
        if not skip_scatter:
            plot_power_roi_scatter(
                subject, deriv_root, task=task, use_spearman=use_spearman, partial_covars=partial_covars,
                do_temp=True, bootstrap_ci=bootstrap, rng=rng, rating_stats=rating_stats, temp_stats=temp_stats,
                config=config
            )
        else:
            logger.info("Skipping scatter plots (overall and roi_scatters) as requested")

        correlate_power_topomaps(
            subject, task, use_spearman=use_spearman, partial_covars=partial_covars, bootstrap=bootstrap, n_perm=n_perm, rng=rng
        )
        export_combined_power_corr_stats(subject)
        plot_top_behavioral_predictors(subject, task)
    sig_alpha = float(config.get("statistics.sig_alpha", 0.05))
    export_all_significant_predictors(subject, alpha=sig_alpha, use_fdr=True)
    
    try:
        apply_global_fdr(subject)
        logger.info("Global FDR correction applied successfully across all analysis types")
    except Exception as e:
        error_msg = (
            f"CRITICAL: Failed to apply global FDR correction: {e}. "
            f"Statistical validity is compromised. Results without proper FDR correction are invalid. "
            f"This is a required step for valid statistical inference."
        )
        logger.error(error_msg)
        raise RuntimeError(
            "Global FDR correction failed. "
            "This is REQUIRED for valid statistical inference. "
            "Fix the error and rerun. Results without FDR correction are statistically invalid."
        ) from e


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
    group_subject_fixed_effects=False,
    skip_scatter=False,
    no_plots=False,
):
    config = load_settings()
    setup_matplotlib(config)
    
    deriv_root = Path(config.deriv_root)
    from eeg_pipeline.utils.io_utils import ensure_derivatives_dataset_description
    ensure_derivatives_dataset_description(deriv_root=deriv_root)
    
    task = task or config.task
    correlation_method = correlation_method or config.get("behavior_analysis.statistics.correlation_method", "spearman")
    if partial_covars is None:
        partial_covars = config.get("behavior_analysis.statistics.partial_covariates", [])
    if n_perm is None:
        n_perm = config.get("behavior_analysis.statistics.n_permutations", 0)
    
    if not subjects and not all_subjects:
        raise ValueError("No subjects specified. Use --group all|A,B,C, or --subject (can repeat), or --all-subjects.")
    
    if all_subjects:
        available = get_available_subjects(deriv_root=deriv_root, task=task, config=config)
        subjects = [
            s for s in available
            if (deriv_root / f"sub-{s}" / "eeg" / "features" / "features_eeg_direct.tsv").exists()
            and (deriv_root / f"sub-{s}" / "eeg" / "features" / "target_vas_ratings.tsv").exists()
        ]
        if not subjects:
            raise ValueError(f"No subjects with features found in {deriv_root}")
    
    if not group_only:
        for sub in subjects:
            process_subject(
                sub, deriv_root, task, config=config, correlation_method=correlation_method,
                partial_covars=partial_covars, bootstrap=bootstrap, n_perm=n_perm, rng_seed=rng_seed,
                skip_scatter=skip_scatter, no_plots=no_plots
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
        "--group-subject-fixed-effects", action="store_true", default=False,
        help="Use subject fixed effects in group analysis (default: disabled)"
    )
    parser.add_argument(
        "--skip-scatter", action="store_true", default=False,
        help="Skip generation of scatter plots (overall and roi_scatters directories)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting operations"
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
    
    n_perm_val = args.n_perm if args.n_perm is not None else None
    
    main(
        subjects, task=args.task, correlation_method=corr_method, partial_covars=None,
        bootstrap=bootstrap_val, n_perm=n_perm_val, do_group=(len(subjects) > 1) if subjects else False,
        group_only=False, rng_seed=args.rng_seed, all_subjects=False,
        pooling_strategy=pooling_map[args.pooling_strategy], cluster_bootstrap=int(args.cluster_bootstrap),
        group_subject_fixed_effects=bool(getattr(args, "group_subject_fixed_effects", False)),
        skip_scatter=args.skip_scatter, no_plots=args.no_plots
    )
