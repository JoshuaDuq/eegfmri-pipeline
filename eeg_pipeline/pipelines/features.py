"""
Feature Extraction Pipeline.

Orchestrates EEG feature extraction from preprocessed epochs.

Features extracted:
- Power: Band power per channel
- Connectivity: wPLI, AEC, graph metrics
- Microstates: Coverage, duration, transitions
- Aperiodic: 1/f slope and offset
- Phase: ITPC, PAC
- Precomputed: ERD/ERS, spectral, temporal, complexity, ROI

Usage:
    # Single subject
    process_subject("0001", "thermalactive", deriv_root, config)

    # Multiple subjects
    extract_features_for_subjects(["0001", "0002"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mne
import numpy as np
import pandas as pd

# Feature categories
FEATURE_CATEGORIES = [
    "power",
    "connectivity",
    "microstates",
    "aperiodic",
    "itpc",
    "pac",
    "precomputed",
]


###################################################################
# Feature Extraction
###################################################################


def extract_all_features(
    epochs: mne.Epochs,
    aligned_events: pd.DataFrame,
    subject: str,
    task: str,
    config: Any,
    deriv_root: Path,
    logger: logging.Logger,
    fixed_templates: Optional[np.ndarray] = None,
    fixed_template_ch_names: Optional[List[str]] = None,
    feature_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract all EEG features for a subject.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs
    aligned_events : pd.DataFrame
        Event metadata aligned with epochs
    subject, task : str
        Subject and task identifiers
    config : Any
        Configuration object
    deriv_root : Path
        Derivatives directory root
    logger : logging.Logger
        Logger instance
    fixed_templates : np.ndarray, optional
        Fixed microstate templates
    fixed_template_ch_names : List[str], optional
        Channel names for fixed templates
    feature_categories : List[str], optional
        Which categories to extract. Default: all.

    Returns
    -------
    Dict[str, Any]
        Dictionary with all extracted features and metadata
    """
    from eeg_pipeline.utils.validation import validate_epochs
    from eeg_pipeline.utils.progress import PipelineProgress
    from eeg_pipeline.utils.analysis.tfr import (
        compute_tfr_for_subject,
        get_tfr_config,
        compute_adaptive_n_cycles,
        normalize_power_with_baseline,
        save_tfr_with_sidecar,
        resolve_tfr_workers,
    )
    from eeg_pipeline.utils.io.general import deriv_features_path
    from eeg_pipeline.analysis.features.power import extract_band_power_features
    from eeg_pipeline.analysis.features.connectivity import (
        extract_connectivity_features,
        compute_sliding_connectivity_features,
    )
    from eeg_pipeline.analysis.features.microstates import extract_microstate_features
    from eeg_pipeline.analysis.features.aperiodic import extract_aperiodic_features
    from eeg_pipeline.analysis.features.phase import (
        extract_itpc_features,
        extract_trialwise_itpc_features,
        compute_pac_comodulograms,
    )
    from eeg_pipeline.analysis.features.pipeline import extract_precomputed_features
    from eeg_pipeline.utils.data.features import load_group_microstate_templates

    # Configuration
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    expected_n_trials = len(epochs)

    # Validate
    validation = validate_epochs(epochs, config, logger=logger)
    if not validation.valid:
        logger.warning(f"Validation issues: {validation.issues}")
        if validation.critical:
            raise ValueError(f"Critical errors: {validation.critical}")

    # Determine categories
    categories = feature_categories if feature_categories else list(FEATURE_CATEGORIES)

    # Initialize results
    results: Dict[str, Any] = {
        "tfr": None,
        "baseline_df": None,
        "baseline_cols": [],
        "pow_df": None,
        "pow_cols": [],
        "conn_df": None,
        "conn_cols": [],
        "ms_df": None,
        "ms_cols": [],
        "ms_templates": None,
        "aper_df": None,
        "aper_cols": [],
        "aper_qc": None,
        "itpc_df": None,
        "itpc_cols": [],
        "itpc_trial_df": None,
        "itpc_trial_cols": [],
        "itpc_map": None,
        "itpc_freqs": None,
        "itpc_times": None,
        "itpc_ch_names": None,
        "pac_df": None,
        "pac_phase_freqs": None,
        "pac_amp_freqs": None,
        "pac_trials_df": None,
        "pac_time_df": None,
        "precomputed_df": None,
        "precomputed_cols": [],
    }

    # Compute TFR (always needed)
    tfr, baseline_df, baseline_cols, b_start, b_end = compute_tfr_for_subject(
        epochs, aligned_events, subject, task, config, deriv_root, logger
    )
    results["tfr"] = tfr
    results["baseline_df"] = baseline_df
    results["baseline_cols"] = baseline_cols

    # Complex TFR for phase-based metrics
    tfr_complex = None
    if any(c in categories for c in ["itpc", "pac"]):
        tfr_complex = _compute_complex_tfr(epochs, config, logger)

    # Progress tracking
    progress = PipelineProgress(total=len(categories), logger=logger, desc="Features")
    progress.start()

    # Power features
    if "power" in categories:
        progress.step(message="Extracting power features...")
        pow_df, pow_cols = extract_band_power_features(tfr, power_bands, config, logger)
        if pow_df is not None and not pow_df.empty:
            try:
                pow_df = normalize_power_with_baseline(pow_df, baseline_df, config, logger)
            except Exception as exc:
                logger.error(f"Baseline normalization failed: {exc}")
        if pow_df is not None and len(pow_df) != expected_n_trials:
            raise ValueError(f"Power length mismatch: {len(pow_df)} vs {expected_n_trials}")
        results["pow_df"] = pow_df
        results["pow_cols"] = pow_cols

    # Apply baseline to TFR
    logger.info("Applying baseline correction to TFR...")
    tfr.apply_baseline(baseline=(b_start, b_end), mode="logratio")
    tfr.comment = f"BASELINED:mode=logratio;win=({b_start:.3f},{b_end:.3f})"

    if config.get("feature_engineering.save_tfr_with_sidecar", False):
        tfr_out = deriv_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
        save_tfr_with_sidecar(tfr, tfr_out, (b_start, b_end), "logratio", logger, config)

    # Connectivity features
    if "connectivity" in categories:
        progress.step(message="Extracting connectivity features...")
        conn_df, conn_cols = extract_connectivity_features(
            epochs, subject, task, power_bands, deriv_root, config, logger
        )
        if conn_df is not None and len(conn_df) != expected_n_trials:
            raise ValueError(f"Connectivity length mismatch: {len(conn_df)} vs {expected_n_trials}")

        # Sliding connectivity
        sw_df, sw_cols, graph_df, graph_cols = compute_sliding_connectivity_features(epochs, config, logger)
        if sw_df is not None and not sw_df.empty:
            conn_df = pd.concat([conn_df, sw_df], axis=1) if conn_df is not None else sw_df
            conn_cols.extend(sw_cols)
        if graph_df is not None and not graph_df.empty:
            conn_df = pd.concat([conn_df, graph_df], axis=1) if conn_df is not None else graph_df
            conn_cols.extend(graph_cols)

        results["conn_df"] = conn_df
        results["conn_cols"] = conn_cols

    # Microstate features
    if "microstates" in categories:
        progress.step(message="Extracting microstate features...")
        use_fixed = bool(config.get("feature_engineering.microstates.use_fixed_templates", False))
        use_group = bool(config.get("feature_engineering.microstates.use_group_templates", False))

        if use_group and not use_fixed:
            group_templates, group_ch_names = load_group_microstate_templates(deriv_root, n_microstates, logger)
            if group_templates is not None:
                fixed_templates = group_templates
                fixed_template_ch_names = group_ch_names
                use_fixed = True
                logger.info("Using group-level microstate templates")

        if use_fixed and fixed_templates is not None:
            ms_df, ms_cols, ms_templates = extract_microstate_features(
                epochs, n_microstates, config, logger,
                fixed_templates=fixed_templates,
                fixed_template_ch_names=fixed_template_ch_names,
            )
            if ms_df is None or ms_df.empty:
                logger.warning("Fixed templates failed; falling back to subject-specific")
                ms_df, ms_cols, ms_templates = extract_microstate_features(
                    epochs, n_microstates, config, logger
                )
        else:
            ms_df, ms_cols, ms_templates = extract_microstate_features(
                epochs, n_microstates, config, logger
            )

        results["ms_df"] = ms_df
        results["ms_cols"] = ms_cols
        results["ms_templates"] = ms_templates

    # Aperiodic features
    if "aperiodic" in categories:
        progress.step(message="Extracting aperiodic features...")
        aper_df, aper_cols, aper_qc = extract_aperiodic_features(
            epochs, (b_start, b_end), power_bands, config, logger, events_df=aligned_events
        )
        results["aper_df"] = aper_df
        results["aper_cols"] = aper_cols
        results["aper_qc"] = aper_qc

    # ITPC features
    if "itpc" in categories:
        progress.step(message="Extracting ITPC features...")
        if tfr_complex is None:
            tfr_complex = _compute_complex_tfr(epochs, config, logger)

        itpc_df, itpc_cols, itpc_map, itpc_freqs, itpc_times, itpc_ch_names = extract_itpc_features(
            epochs, power_bands, config, logger, tfr_complex=tfr_complex
        )
        itpc_trial_df, itpc_trial_cols = extract_trialwise_itpc_features(
            epochs, power_bands, config, logger, tfr_complex=tfr_complex
        )

        results["itpc_df"] = itpc_df
        results["itpc_cols"] = itpc_cols
        results["itpc_trial_df"] = itpc_trial_df
        results["itpc_trial_cols"] = itpc_trial_cols
        results["itpc_map"] = itpc_map
        results["itpc_freqs"] = itpc_freqs
        results["itpc_times"] = itpc_times
        results["itpc_ch_names"] = itpc_ch_names

    # PAC features
    if "pac" in categories:
        progress.step(message="Computing PAC features...")
        if tfr_complex is None:
            tfr_complex = _compute_complex_tfr(epochs, config, logger)

        if tfr_complex is not None:
            freq_min, freq_max, n_freqs, *_ = get_tfr_config(config)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
            pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df = compute_pac_comodulograms(
                tfr_complex, freqs, tfr_complex.times, epochs.info, config, logger
            )
            results["pac_df"] = pac_df
            results["pac_phase_freqs"] = pac_phase_freqs
            results["pac_amp_freqs"] = pac_amp_freqs
            results["pac_trials_df"] = pac_trials_df
            results["pac_time_df"] = pac_time_df

    # Precomputed features (with condition column from aligned_events)
    if "precomputed" in categories:
        progress.step(message="Extracting precomputed features...")
        groups = config.get(
            "feature_engineering.precomputed_groups",
            ["erds", "spectral", "gfp", "roi", "temporal", "complexity", "ratios", "asymmetry"],
        )
        try:
            precomputed_result = extract_precomputed_features(
                epochs, power_bands, config, logger,
                feature_groups=groups,
                n_plateau_windows=int(config.get("feature_engineering.erds.n_temporal_windows", 5)),
                events_df=aligned_events,  # Pass events for condition column
            )
            results["precomputed_df"] = precomputed_result.get_combined_df()
            results["precomputed_cols"] = precomputed_result.get_all_columns()
            if results["precomputed_df"] is not None:
                n_cols = len(results['precomputed_cols'])
                condition_info = ""
                if precomputed_result.condition is not None:
                    condition_info = f" (pain={precomputed_result.n_pain}, nonpain={precomputed_result.n_nonpain})"
                logger.info(f"Extracted {n_cols} precomputed features{condition_info}")
        except Exception as exc:
            logger.error(f"Precomputed extraction failed: {exc}")

    progress.finish()
    return results


def _compute_complex_tfr(epochs: mne.Epochs, config: Any, logger: logging.Logger):
    """Compute complex TFR for phase-based metrics."""
    from eeg_pipeline.utils.analysis.tfr import (
        get_tfr_config,
        compute_adaptive_n_cycles,
        resolve_tfr_workers,
    )

    freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    workers = resolve_tfr_workers(int(config.get("tfr_topography_pipeline.tfr.workers", -1)))

    logger.info("Computing complex TFR for phase-based metrics...")
    try:
        return epochs.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            decim=tfr_decim,
            picks=tfr_picks,
            use_fft=True,
            return_itc=False,
            average=False,
            output="complex",
            n_jobs=workers,
        )
    except Exception as exc:
        logger.error(f"Complex TFR computation failed: {exc}")
        return None


###################################################################
# Subject Processing
###################################################################


def process_subject(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    fixed_templates_path: Optional[Path] = None,
    feature_categories: Optional[List[str]] = None,
) -> None:
    """Process a single subject for feature extraction."""
    from eeg_pipeline.utils.io.general import (
        get_subject_logger,
        setup_matplotlib,
        ensure_dir,
        deriv_features_path,
        deriv_plots_path,
        write_tsv,
        _load_events_df,
        _pick_target_column,
    )
    from eeg_pipeline.utils.data.loading import load_epochs_for_analysis
    from eeg_pipeline.utils.data.features import (
        align_feature_dataframes,
        export_fmri_regressors,
        build_plateau_features,
        save_all_features,
        save_microstate_templates,
        compute_group_microstate_templates,
        save_trial_alignment_manifest,
        save_dropped_trials_log,
    )
    from eeg_pipeline.plotting.behavioral import plot_regressor_distributions

    if not subject or not task:
        raise ValueError(f"subject and task required, got: {subject=}, {task=}")

    if logger is None:
        setup_matplotlib(config)
        logger = get_subject_logger(
            "feature_engineering",
            subject,
            config.get("logging.file_names.feature_engineering", "extract_features.log"),
            config=config,
        )

    logger.info(f"=== Feature extraction: sub-{subject}, task-{task} ===")

    features_dir = deriv_features_path(deriv_root, subject)
    ensure_dir(features_dir)

    # Load data
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False,
        deriv_root=deriv_root, logger=logger, config=config,
    )

    if epochs is None:
        logger.error(f"No cleaned epochs for sub-{subject}; skipping")
        return

    if aligned_events is None:
        logger.warning("No events available; skipping")
        return

    # Save dropped trials log
    original_events = _load_events_df(subject, task, bids_root=config.bids_root, config=config)
    if original_events is not None:
        save_dropped_trials_log(epochs, original_events, features_dir / "dropped_trials.tsv", logger)

    save_trial_alignment_manifest(aligned_events, epochs, features_dir / "trial_alignment.tsv", config, logger)

    # Get target column
    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    target_col = _pick_target_column(aligned_events, constants=constants)
    if target_col is None:
        logger.warning("No target column found; skipping")
        return

    y = pd.to_numeric(aligned_events[target_col], errors="coerce")

    # Load fixed templates if provided
    fixed_templates = None
    fixed_template_ch_names = None
    if fixed_templates_path and fixed_templates_path.exists():
        try:
            data = np.load(fixed_templates_path)
            fixed_templates = data["templates"]
            fixed_template_ch_names = data.get("ch_names")
            logger.info(f"Loaded fixed templates from {fixed_templates_path}")
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")

    # Extract features
    features = extract_all_features(
        epochs, aligned_events, subject, task, config, deriv_root, logger,
        fixed_templates=fixed_templates,
        fixed_template_ch_names=fixed_template_ch_names,
        feature_categories=feature_categories,
    )

    # Unpack results
    tfr = features["tfr"]
    pow_df = features["pow_df"]
    pow_cols = features["pow_cols"]
    baseline_df = features["baseline_df"]
    baseline_cols = features["baseline_cols"]
    conn_df = features["conn_df"]
    conn_cols = features["conn_cols"]
    ms_df = features["ms_df"]
    ms_cols = features["ms_cols"]
    ms_templates = features["ms_templates"]
    aper_df = features["aper_df"]
    aper_cols = features["aper_cols"]
    itpc_df = features.get("itpc_df")
    itpc_cols = features.get("itpc_cols", [])
    itpc_trial_df = features.get("itpc_trial_df")
    itpc_trial_cols = features.get("itpc_trial_cols", [])
    precomputed_df = features.get("precomputed_df")
    precomputed_cols = features.get("precomputed_cols", [])

    # Merge ITPC trial features into power
    if itpc_trial_df is not None and not itpc_trial_df.empty:
        pow_df = pd.concat([pow_df, itpc_trial_df], axis=1)
        pow_cols.extend(itpc_trial_cols)

    # Save microstate templates
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    save_microstate_templates(epochs, ms_templates, subject, n_microstates, deriv_root, logger)

    if config.get("feature_engineering.microstates.build_group_templates", False):
        group_templates, _ = compute_group_microstate_templates(deriv_root, n_microstates, logger)
        if group_templates is not None:
            logger.info("Updated group microstate templates")

    # Build plateau features and align
    logger.info("Building plateau features and aligning...")
    plateau_df, plateau_cols = build_plateau_features(
        pow_df, pow_cols, baseline_df, baseline_cols, tfr, power_bands, logger
    )

    (
        pow_df_aligned, baseline_df_aligned, conn_df_aligned,
        ms_df_aligned, aper_df_aligned, y_aligned, retention_stats
    ) = align_feature_dataframes(
        pow_df, baseline_df, conn_df, ms_df, aper_df, y, aligned_events, features_dir, logger, config
    )

    # Save all features
    combined_df = save_all_features(
        pow_df_aligned, pow_cols, baseline_df_aligned, baseline_cols,
        conn_df_aligned, conn_cols, ms_df_aligned, ms_cols,
        aper_df_aligned, aper_cols, itpc_df, itpc_cols,
        features.get("pac_df"), features.get("pac_trials_df"), features.get("pac_time_df"),
        features.get("aper_qc"), plateau_df, plateau_cols, y_aligned, features_dir, logger, config
    )

    # Save precomputed features
    if precomputed_df is not None and not precomputed_df.empty:
        write_tsv(precomputed_df, features_dir / "features_precomputed.tsv")
        write_tsv(pd.Series(precomputed_cols, name="feature").to_frame(), features_dir / "features_precomputed_columns.tsv")
        combined_df = pd.concat([combined_df, precomputed_df], axis=1)

    # Export fMRI regressors
    regressor_df = export_fmri_regressors(
        aligned_events, plateau_df, plateau_cols, ms_df_aligned,
        features.get("pac_trials_df"), aper_df_aligned, y_aligned,
        power_bands, subject, task, features_dir, config, logger,
    )
    if regressor_df is not None:
        plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="behavior")
        plot_regressor_distributions(regressor_df, subject, plots_dir, logger, config)

    # Summary
    n_trials = len(y_aligned)
    n_pow = pow_df_aligned.shape[1] if pow_df_aligned is not None else 0
    n_conn = conn_df_aligned.shape[1] if conn_df_aligned is not None and not conn_df_aligned.empty else 0
    n_ms = ms_df_aligned.shape[1] if ms_df_aligned is not None and not ms_df_aligned.empty else 0
    n_aper = aper_df_aligned.shape[1] if aper_df_aligned is not None and not aper_df_aligned.empty else 0
    n_precomp = precomputed_df.shape[1] if precomputed_df is not None and not precomputed_df.empty else 0
    n_total = combined_df.shape[1]

    logger.info(
        f"Done: sub-{subject}, trials={n_trials}, power={n_pow}, conn={n_conn}, "
        f"ms={n_ms}, aper={n_aper}, precomp={n_precomp}, total={n_total}"
    )


###################################################################
# Batch Processing
###################################################################


def extract_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    fixed_templates_path: Optional[Path] = None,
    run_group_aggregation: bool = True,
    feature_categories: Optional[List[str]] = None,
) -> None:
    """Extract features for multiple subjects."""
    from eeg_pipeline.utils.config.loader import load_settings
    from eeg_pipeline.utils.io.general import get_logger
    from eeg_pipeline.utils.progress import BatchProgress

    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        config = load_settings()

    if deriv_root is None:
        deriv_root = Path(config.deriv_root)

    task = task or config.get("project.task", "thermalactive")
    logger = get_logger(__name__)

    fixed_path = Path(fixed_templates_path) if fixed_templates_path else None

    with BatchProgress(subjects=subjects, logger=logger, desc="Feature Extraction") as batch:
        for subject in subjects:
            start_time = batch.start_subject(subject)
            try:
                process_subject(
                    subject, task, deriv_root, config,
                    fixed_templates_path=fixed_path,
                    feature_categories=feature_categories,
                )
                batch.finish_subject(subject, start_time)
            except Exception as exc:
                logger.error(f"Failed sub-{subject}: {exc}")
                batch.finish_subject(subject, start_time)

    if run_group_aggregation and len(subjects) >= 2:
        logger.info("Running group-level aggregation...")
        from eeg_pipeline.analysis.group import aggregate_feature_stats

        aggregate_feature_stats(subjects, task, deriv_root, config)
