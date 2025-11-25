from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    get_subject_logger,
    setup_matplotlib,
    ensure_dir,
    ensure_derivatives_dataset_description,
    get_logger,
    write_tsv,
)
from eeg_pipeline.utils.data.loading import (
    _load_features_and_targets,
    load_epochs_for_analysis,
)
from eeg_pipeline.analysis.behavior.fdr_correction import apply_global_fdr
from eeg_pipeline.analysis.behavior.power_roi import compute_power_roi_stats
from eeg_pipeline.analysis.behavior.connectivity import (
    correlate_connectivity_heatmaps,
    correlate_connectivity_roi_summaries,
    _correlate_sliding_connectivity,
)
from eeg_pipeline.analysis.behavior.topomaps import correlate_power_topomaps
from eeg_pipeline.analysis.behavior.exports import (
    export_all_significant_predictors,
    export_combined_power_corr_stats,
)
from eeg_pipeline.analysis.behavior.temporal import (
    compute_time_frequency_correlations,
    compute_temporal_correlations_by_condition,
)
from eeg_pipeline.analysis.behavior.cluster_tests import _run_pain_nonpain_cluster_test
from eeg_pipeline.utils.analysis.reliability import (
    get_subject_seed,
    compute_feature_split_half_reliability,
    compute_tf_split_half_reliability,
)


###################################################################
# Behavior analysis helpers
###################################################################


def initialize_analysis_context(
    subject: str,
    task: Optional[str],
    config: Any,
    logger_name: str = "behavior_analysis",
) -> Tuple[Any, str, Path, Path, logging.Logger]:
    """
    Initialize analysis context for behavior analysis.
    
    Sets up configuration, task, logger, and directory paths for a subject's
    behavior analysis. This is a general utility that can be reused across
    different behavior analysis modules.
    
    Parameters
    ----------
    subject : str
        Subject identifier (without 'sub-' prefix)
    task : Optional[str]
        Task name. If None, uses config.task
    config : Any
        Configuration object. If None, loads settings from config file
    logger_name : str
        Logger name (default: "behavior_analysis")
        
    Returns
    -------
    Tuple[Any, str, Path, Path, logging.Logger]
        (config, task, deriv_root, stats_dir, logger) tuple
        
    Raises
    ------
    ValueError
        If subject is not provided
    """
    if not subject:
        raise ValueError("Subject must be provided")
    
    if config is None:
        config = load_settings()
    
    if task is None:
        task = config.task
    
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger(logger_name, subject, log_name, config=config)
    
    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    
    return config, task, deriv_root, stats_dir, logger


def process_subject(
    subject: str,
    deriv_root: Path,
    task: str,
    config,
    logger: Optional[logging.Logger] = None,
    correlation_method: str = "spearman",
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng_seed: int = 42,
    computations: Optional[List[str]] = None,
) -> None:
    if not subject or not task:
        raise ValueError(f"subject and task must be non-empty strings, got: subject={subject}, task={task}")
    
    if logger is None:
        logger = get_subject_logger("behavior_analysis", subject, config.get("output.log_file_name"), config=config)
    logger.info(f"=== Computing behavior correlations: sub-{subject}, task-{task} ===")
    
    use_spearman = (correlation_method == "spearman")
    subject_seed = get_subject_seed(rng_seed, subject)
    rng = np.random.default_rng(subject_seed)
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False, deriv_root=deriv_root, 
        bids_root=config.bids_root, config=config
    )
    
    _, pow_df, conn_df, y, _ = _load_features_and_targets(subject, task, deriv_root, config, epochs=epochs)  # info unused
    
    if y is None:
        logger.warning("No target variable found; skipping correlations")
        return
    
    if n_perm == 0:
        stats_n_perm = config.get("behavior_analysis.statistics.n_permutations", 1024)
        if stats_n_perm > 0:
            n_perm = int(stats_n_perm)
            logger.info("n_perm was 0; using behavior_analysis.statistics.n_permutations=%d", n_perm)
        else:
            cluster_perm = config.get("behavior_analysis.cluster_correction.n_permutations", 1024)
            n_perm = int(cluster_perm) if cluster_perm > 0 else 1024
            logger.info("n_perm was 0; using behavior_analysis.cluster_correction.n_permutations=%d", n_perm)
    
    all_computations = {
        "time_frequency": lambda: compute_time_frequency_correlations(subject, task, deriv_root, config, use_spearman, logger),
        "connectivity_heatmaps": lambda: correlate_connectivity_heatmaps(subject, task, use_spearman=use_spearman),
        "connectivity_roi": lambda: correlate_connectivity_roi_summaries(
            subject, task, use_spearman=use_spearman,
            partial_covars=partial_covars, bootstrap=bootstrap,
            n_perm=n_perm, rng=rng
        ),
        "sliding_connectivity": lambda: _correlate_sliding_connectivity(
            conn_df=conn_df,
            ratings=y,
            config=config,
            stats_dir=deriv_stats_path(deriv_root, subject),
            logger=logger,
            use_spearman=use_spearman,
        ),
        "power_roi": lambda: compute_power_roi_stats(
            subject, deriv_root, task=task, use_spearman=use_spearman, partial_covars=partial_covars,
            bootstrap=bootstrap, n_perm=n_perm, rng=rng
        ),
        "topomaps": lambda: correlate_power_topomaps(
            subject, task, use_spearman=use_spearman, partial_covars=partial_covars, 
            bootstrap=bootstrap, n_perm=n_perm, rng=rng
        ),
        "cluster_test": lambda: _run_pain_nonpain_cluster_test(subject, task, deriv_root, config, logger),
        "temporal_correlations": lambda: compute_temporal_correlations_by_condition(subject, task, deriv_root, config, use_spearman, logger),
        "exports": lambda: export_combined_power_corr_stats(subject),
    }
    
    computations_to_run = computations if computations is not None else list(all_computations.keys())
    
    if "time_frequency" in computations_to_run:
        logger.info("Computing time-frequency correlations...")
        all_computations["time_frequency"]()
    
    if "connectivity_heatmaps" in computations_to_run:
        logger.info("Computing connectivity correlations...")
        all_computations["connectivity_heatmaps"]()
    
    if "connectivity_roi" in computations_to_run:
        logger.info("Computing connectivity ROI summaries...")
        all_computations["connectivity_roi"]()
    
    if "sliding_connectivity" in computations_to_run:
        logger.info("Computing sliding connectivity behavior correlations...")
        all_computations["sliding_connectivity"]()
    
    if "power_roi" in computations_to_run:
        logger.info("Computing ROI statistics...")
        all_computations["power_roi"]()
    
    if "topomaps" in computations_to_run:
        logger.info("Computing topomap correlations...")
        all_computations["topomaps"]()
    
    if "cluster_test" in computations_to_run:
        try:
            logger.info("Computing pain vs. non-pain time-cluster test...")
            all_computations["cluster_test"]()
        except Exception as e:
            logger.warning("Pain vs. non-pain cluster test failed: %s", e)
    
    if "temporal_correlations" in computations_to_run:
        logger.info("Computing temporal correlation topomaps...")
        all_computations["temporal_correlations"]()
    
    if "exports" in computations_to_run:
        logger.info("Exporting combined correlation stats...")
        all_computations["exports"]()

    if computations is None or "reliability" in computations:
        reliability_records = []
        rel_boot = int(config.get("behavior_analysis.statistics.reliability_bootstrap", 100))
        try:
            pow_cols = [c for c in pow_df.columns if str(c).startswith("pow_")]
            if pow_cols:
                feature_matrix = pow_df[pow_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                y_array = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
                med, lo, hi = compute_feature_split_half_reliability(
                    feature_matrix, y_array, rel_boot, use_spearman, rng
                )
                reliability_records.append({
                    "metric": "power_corr_split_half",
                    "median": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_boot": rel_boot,
                })
        except Exception as exc:
            logger.warning("Power split-half reliability failed: %s", exc)

        try:
            if epochs is not None and aligned_events is not None:
                med_tf, lo_tf, hi_tf = compute_tf_split_half_reliability(
                    epochs, aligned_events, y, config, use_spearman, rel_boot, rng, logger
                )
                reliability_records.append({
                    "metric": "tf_corr_split_half",
                    "median": med_tf,
                    "ci_low": lo_tf,
                    "ci_high": hi_tf,
                    "n_boot": rel_boot,
                })
        except Exception as exc:
            logger.warning("TF split-half reliability failed: %s", exc)

        if reliability_records:
            stats_dir = deriv_stats_path(deriv_root, subject)
            ensure_dir(stats_dir)
            write_tsv(pd.DataFrame(reliability_records), stats_dir / "behavior_reliability.tsv")
    
    if computations is None or "fdr" in computations:
        try:
            logger.info("Applying global FDR correction...")
            apply_global_fdr(subject)
            logger.info("Global FDR correction applied successfully")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to apply global FDR correction: {e}", exc_info=True)
            raise RuntimeError("Global FDR correction failed. This is REQUIRED for valid statistical inference.") from e

        sig_alpha = float(config.get("statistics.sig_alpha"))
        logger.info("Exporting significant predictors (using global FDR where available)...")
        export_all_significant_predictors(subject, alpha=sig_alpha, use_fdr=True)
    
    logger.info(f"Completed behavior correlations for sub-{subject}")


###################################################################
# Batch processing
###################################################################


def compute_behavior_correlations_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    correlation_method: Optional[str] = None,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: Optional[int] = None,
    rng_seed: int = 42,
    logger: Optional[logging.Logger] = None,
    run_group_aggregation: bool = True,
) -> None:
    if not subjects:
        raise ValueError("No subjects specified. Use --group all|A,B,C, or --subject (can repeat), or --all-subjects.")
    
    if config is None:
        from eeg_pipeline.utils.config.loader import load_settings
        config = load_settings()
    
    if deriv_root is None:
        deriv_root = Path(config.deriv_root)
    
    setup_matplotlib(config)
    ensure_derivatives_dataset_description(deriv_root=deriv_root)
    
    task = task or config.get("project.task")
    correlation_method = correlation_method or config.get("behavior_analysis.statistics.correlation_method")
    partial_covars = partial_covars or config.get("behavior_analysis.statistics.partial_covariates")
    
    if n_perm is None or n_perm == 0:
        stats_n_perm = config.get("behavior_analysis.statistics.n_permutations", 1024)
        if stats_n_perm > 0:
            n_perm = int(stats_n_perm)
        else:
            cluster_perm = config.get("behavior_analysis.cluster_correction.n_permutations", 1024)
            n_perm = int(cluster_perm) if cluster_perm > 0 else 1024
            if logger is None:
                logger = get_logger(__name__)
            logger.info(
                "behavior_analysis.statistics.n_permutations is 0; "
                "falling back to cluster_correction.n_permutations=%d for permutation tests.",
                n_perm,
            )
    
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Starting behavior correlation computation: {len(subjects)} subject(s), task={task}")
    
    for idx, sub in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Processing sub-{sub}")
        process_subject(
            sub, deriv_root, task, config=config, correlation_method=correlation_method,
            partial_covars=partial_covars, bootstrap=bootstrap, n_perm=n_perm, rng_seed=rng_seed
        )
    
    if run_group_aggregation and len(subjects) >= 2:
        logger.info("Running group-level behavior aggregation...")
        from eeg_pipeline.analysis.group import aggregate_behavior_correlations
        aggregate_behavior_correlations(
            subjects, task, deriv_root, pooling_strategy="within_subject_centered", config=config
        )
    
    logger.info(f"Behavior correlation computation complete: {len(subjects)} subjects processed")

