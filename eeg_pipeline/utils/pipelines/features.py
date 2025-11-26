from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import load_epochs_for_analysis
from eeg_pipeline.utils.data.features import (
    align_feature_dataframes,
    export_fmri_regressors,
    build_plateau_features,
    save_all_features,
    save_microstate_templates,
    load_group_microstate_templates,
    compute_group_microstate_templates,
    save_trial_alignment_manifest,
    save_dropped_trials_log,
)
from eeg_pipeline.utils.io.general import (
    _load_events_df,
    _pick_target_column,
    deriv_features_path,
    deriv_plots_path,
    ensure_dir,
    get_subject_logger,
    setup_matplotlib,
    get_logger,
)
from eeg_pipeline.utils.analysis.tfr import (
    compute_adaptive_n_cycles,
    save_tfr_with_sidecar,
    get_tfr_config,
    compute_tfr_for_subject,
    normalize_power_with_baseline,
    resolve_tfr_workers,
)
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
from eeg_pipeline.plotting.behavioral import plot_regressor_distributions


###################################################################
# Feature extraction helpers
###################################################################


def extract_all_features(
    epochs: mne.Epochs,
    aligned_events: pd.DataFrame,
    subject: str,
    task: str,
    config,
    deriv_root: Path,
    logger: logging.Logger,
    fixed_templates: Optional[np.ndarray] = None,
    fixed_template_ch_names: Optional[List[str]] = None,
    feature_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    expected_n_trials = len(epochs)
    
    categories_to_extract = feature_categories if feature_categories is not None else [
        "power", "connectivity", "microstates", "aperiodic", "itpc", "pac"
    ]
    
    # 1. Compute Raw TFR and Baseline Features (always needed as prerequisite)
    tfr, baseline_df, baseline_cols, b_start, b_end = compute_tfr_for_subject(
        epochs, aligned_events, subject, task, config, deriv_root, logger
    )

    tfr_complex = None
    if "itpc" in categories_to_extract or "pac" in categories_to_extract:
        freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
        workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
        workers = resolve_tfr_workers(workers_default=workers_default)
        logger.info("Computing complex TFR for phase-based metrics...")
        try:
            tfr_complex = epochs.compute_tfr(
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
            logger.error("Failed to compute complex TFR for phase-based metrics: %s", exc)
            tfr_complex = None
    
    pow_df = None
    pow_cols = []
    conn_df = None
    conn_cols = []
    ms_df = None
    ms_cols = []
    ms_templates = None
    aper_df = None
    aper_cols = []
    aper_qc = None
    itpc_df = None
    itpc_cols = []
    itpc_trial_df = None
    itpc_trial_cols = []
    itpc_map = None
    itpc_freqs = None
    itpc_times = None
    itpc_ch_names = None
    pac_df = None
    pac_phase_freqs = None
    pac_amp_freqs = None
    pac_trials_df = None
    pac_time_df = None
    
    if "power" in categories_to_extract:
        logger.info("Extracting band power features from raw TFR and normalizing with baseline means (log(mean_bin / mean_baseline))...")
        pow_df_raw, pow_cols = extract_band_power_features(tfr, power_bands, config, logger)
        pow_df = pow_df_raw
        if pow_df_raw is not None and not pow_df_raw.empty:
            try:
                pow_df = normalize_power_with_baseline(pow_df_raw, baseline_df, config, logger)
            except Exception as exc:
                logger.error("Failed to normalize power with baseline; returning raw power features: %s", exc)
                pow_df = pow_df_raw
        if pow_df is not None and len(pow_df) != expected_n_trials:
            raise ValueError(
                f"Power feature extraction length mismatch: expected {expected_n_trials} "
                f"but got {len(pow_df)}"
            )
    
    # 4. Apply Baseline to TFR and Save (if configured)
    logger.info("Applying baseline correction to TFR object for saving...")
    tfr.apply_baseline(baseline=(b_start, b_end), mode="logratio")
    tfr.comment = f"BASELINED:mode=logratio;win=({b_start:.3f},{b_end:.3f})"
    
    save_cfg = config.get("feature_engineering.save_tfr_with_sidecar", False)
    if save_cfg:
        tfr_out = deriv_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
        save_tfr_with_sidecar(
            tfr, tfr_out, baseline_window=(b_start, b_end),
            mode="logratio", logger=logger, config=config
        )

    if "connectivity" in categories_to_extract:
        logger.info("Extracting connectivity features...")
        conn_df, conn_cols = extract_connectivity_features(
            epochs, subject, task, power_bands, deriv_root, config, logger
        )
        if conn_df is not None and len(conn_df) != expected_n_trials:
            raise ValueError(
                f"Connectivity feature extraction length mismatch: expected {expected_n_trials} "
                f"but got {len(conn_df)}"
            )
            
        if conn_df is None or conn_df.empty:
            logger.warning(
                "Connectivity dataframe is empty after computation; check band definitions or preprocessing."
            )

        logger.info("Computing sliding-window connectivity and graph metrics...")
        sw_edges_df, sw_edge_cols, graph_df, graph_cols = compute_sliding_connectivity_features(epochs, config, logger)
        if sw_edges_df is not None and not sw_edges_df.empty:
            conn_df = pd.concat([conn_df, sw_edges_df], axis=1) if conn_df is not None else sw_edges_df
            conn_cols.extend(sw_edge_cols)
        if graph_df is not None and not graph_df.empty:
            conn_df = pd.concat([conn_df, graph_df], axis=1) if conn_df is not None else graph_df
            conn_cols.extend(graph_cols)
    
    if "microstates" in categories_to_extract:
        logger.info("Extracting microstate features...")
        use_fixed_templates = bool(config.get("feature_engineering.microstates.use_fixed_templates", False))
        use_group_templates = bool(config.get("feature_engineering.microstates.use_group_templates", False))

        if use_group_templates and not use_fixed_templates:
            group_templates, group_ch_names = load_group_microstate_templates(deriv_root, n_microstates, logger)
            if group_templates is not None:
                fixed_templates = group_templates
                fixed_template_ch_names = group_ch_names
                use_fixed_templates = True
                logger.info("Using group-level microstate templates for subject-level extraction.")

        if use_fixed_templates:
            if fixed_templates is None:
                logger.error(
                    "feature_engineering.microstates.use_fixed_templates is True but no templates were provided. "
                    "Skipping microstate features to avoid misaligned state labels."
                )
            else:
                ms_df, ms_cols, ms_templates = extract_microstate_features(
                    epochs, n_microstates, config, logger,
                    fixed_templates=fixed_templates,
                    fixed_template_ch_names=fixed_template_ch_names,
                )
                if ms_df is None or (hasattr(ms_df, "empty") and ms_df.empty):
                    logger.warning("Fixed/group microstate templates failed; falling back to subject-specific clustering.")
                    ms_df, ms_cols, ms_templates = extract_microstate_features(
                        epochs, n_microstates, config, logger,
                        fixed_templates=None,
                        fixed_template_ch_names=None,
                    )
        else:
            ms_df, ms_cols, ms_templates = extract_microstate_features(
                epochs, n_microstates, config, logger,
                fixed_templates=None,
                fixed_template_ch_names=None,
            )

    if "aperiodic" in categories_to_extract:
        logger.info("Extracting aperiodic features...")
        aper_df, aper_cols, aper_qc = extract_aperiodic_features(
            epochs, (b_start, b_end), power_bands, config, logger, events_df=aligned_events
        )

    if "itpc" in categories_to_extract:
        if tfr_complex is None:
            freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
            n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
            workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
            workers = resolve_tfr_workers(workers_default=workers_default)
            logger.info("Computing complex TFR for ITPC...")
            try:
                tfr_complex = epochs.compute_tfr(
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
                logger.error("Failed to compute complex TFR for ITPC: %s", exc)
                tfr_complex = None
        
        logger.info("Extracting ITPC features...")
        itpc_df, itpc_cols, itpc_map, itpc_freqs, itpc_times, itpc_ch_names = extract_itpc_features(
            epochs, power_bands, config, logger, tfr_complex=tfr_complex
        )

        logger.info("Extracting trialwise ITPC features...")
        itpc_trial_df, itpc_trial_cols = extract_trialwise_itpc_features(
            epochs, power_bands, config, logger, tfr_complex=tfr_complex
        )

    if "pac" in categories_to_extract:
        if tfr_complex is None:
            freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
            n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
            workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
            workers = resolve_tfr_workers(workers_default=workers_default)
            logger.info("Computing complex TFR for PAC...")
            try:
                tfr_complex = epochs.compute_tfr(
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
                logger.error("Failed to compute complex TFR for PAC: %s", exc)
                tfr_complex = None
        
        logger.info("Computing PAC comodulograms...")
        if tfr_complex is not None:
            pac_times = tfr_complex.times
            pac_info = epochs.info
            freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        else:
            pac_times = epochs.times
            pac_info = epochs.info
            freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df = compute_pac_comodulograms(
            tfr_complex, freqs, pac_times, pac_info, config, logger
        )

    results = {
        "tfr": tfr,
        "baseline_df": baseline_df,
        "baseline_cols": baseline_cols,
        "pow_df": pow_df,
        "pow_cols": pow_cols,
        "conn_df": conn_df,
        "conn_cols": conn_cols,
        "ms_df": ms_df,
        "ms_cols": ms_cols,
        "ms_templates": ms_templates,
        "aper_df": aper_df,
        "aper_cols": aper_cols,
        "aper_qc": aper_qc,
        "itpc_df": itpc_df,
        "itpc_cols": itpc_cols,
        "itpc_trial_df": itpc_trial_df,
        "itpc_trial_cols": itpc_trial_cols,
        "itpc_map": itpc_map,
        "itpc_freqs": itpc_freqs,
        "itpc_times": itpc_times,
        "itpc_ch_names": itpc_ch_names,
        "pac_df": pac_df,
        "pac_phase_freqs": pac_phase_freqs,
        "pac_amp_freqs": pac_amp_freqs,
        "pac_trials_df": pac_trials_df,
        "pac_time_df": pac_time_df,
    }
    return results


def process_subject(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: Optional[logging.Logger] = None,
    fixed_templates_path: Optional[Path] = None,
    feature_categories: Optional[List[str]] = None,
) -> None:
    if not subject or not task:
        raise ValueError(f"subject and task must be non-empty strings, got: subject={subject}, task={task}")
    
    if logger is None:
        setup_matplotlib(config)
        log_file_name = config.get("logging.file_names.feature_engineering", "extract_features.log")
        logger = get_subject_logger("feature_engineering", subject, log_file_name, config=config)
    logger.info(f"=== Feature extraction: sub-{subject}, task-{task} ===")
    
    features_dir = deriv_features_path(deriv_root, subject)
    ensure_dir(features_dir)

    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False,
        deriv_root=deriv_root, logger=logger, config=config,
    )
    
    if epochs is None:
        logger.error(f"No cleaned epochs found for sub-{subject}; skipping")
        return
    
    if aligned_events is None:
        logger.warning("No events available for targets; skipping subject")
        return

    original_events_df = _load_events_df(subject, task, bids_root=config.bids_root, config=config)
    if original_events_df is None:
        logger.warning("Could not load original events TSV for dropped trials log; skipping drop log")
    else:
        drop_log_path = features_dir / "dropped_trials.tsv"
        save_dropped_trials_log(epochs, original_events_df, drop_log_path, logger)
    
    manifest_path = features_dir / "trial_alignment.tsv"
    save_trial_alignment_manifest(aligned_events, epochs, manifest_path, config, logger)

    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    target_col = _pick_target_column(aligned_events, constants=constants)
    if target_col is None:
        logger.warning("No suitable target column found in events; skipping")
        return
    
    y = pd.to_numeric(aligned_events[target_col], errors="coerce")
    
    fixed_templates = None
    fixed_template_ch_names: Optional[List[str]] = None
    if fixed_templates_path is not None:
        if fixed_templates_path.exists():
            try:
                data = np.load(fixed_templates_path)
                fixed_templates = data['templates']
                fixed_template_ch_names = data.get('ch_names')
                logger.info(f"Loaded fixed microstate templates from {fixed_templates_path}")
            except Exception as e:
                logger.error(f"Failed to load fixed templates: {e}")
        else:
            logger.warning(f"Fixed templates file not found: {fixed_templates_path}")

    features = extract_all_features(
        epochs, aligned_events, subject, task, config, deriv_root, logger,
        fixed_templates=fixed_templates,
        fixed_template_ch_names=fixed_template_ch_names,
        feature_categories=feature_categories,
    )
    
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

    if itpc_trial_df is not None and not itpc_trial_df.empty:
        pow_df = pd.concat([pow_df, itpc_trial_df], axis=1)
        pow_cols.extend(itpc_trial_cols)

    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    save_microstate_templates(epochs, ms_templates, subject, n_microstates, deriv_root, logger)
    if config.get("feature_engineering.microstates.build_group_templates", False):
        group_templates, _ = compute_group_microstate_templates(deriv_root, n_microstates, logger)
        if group_templates is not None:
            logger.info("Saved/updated group microstate templates.")

    logger.info("Building plateau features and aligning all feature dataframes...")
    plateau_df, plateau_cols = build_plateau_features(
        pow_df, pow_cols, baseline_df, baseline_cols, tfr, power_bands, logger
    )

    pow_df_aligned, baseline_df_aligned, conn_df_aligned, ms_df_aligned, aper_df_aligned, y_aligned, retention_stats = align_feature_dataframes(
        pow_df, baseline_df, conn_df, ms_df, aper_df, y, aligned_events, features_dir, logger, config
    )

    combined_df = save_all_features(
        pow_df_aligned, pow_cols, baseline_df_aligned, baseline_cols, conn_df_aligned, conn_cols,
        ms_df_aligned, ms_cols, aper_df_aligned, aper_cols, itpc_df, itpc_cols,
        features.get("pac_df"), features.get("pac_trials_df"), features.get("pac_time_df"),
        features.get("aper_qc"),
        plateau_df, plateau_cols, y_aligned, features_dir, logger, config
    )

    regressor_df = export_fmri_regressors(
        aligned_events,
        plateau_df,
        plateau_cols,
        ms_df_aligned,
        features.get("pac_trials_df"),
        aper_df_aligned,
        y_aligned,
        power_bands,
        subject,
        task,
        features_dir,
        config,
        logger,
    )
    if regressor_df is not None:
        plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="behavior")
        plot_regressor_distributions(regressor_df, subject, plots_dir, logger, config)
    
    n_trials = len(y_aligned)
    n_pow_features = pow_df_aligned.shape[1] if pow_df_aligned is not None else 0
    n_conn_features = conn_df_aligned.shape[1] if conn_df_aligned is not None and not conn_df_aligned.empty else 0
    n_ms_features = ms_df_aligned.shape[1] if ms_df_aligned is not None and not ms_df_aligned.empty else 0
    n_aper_features = aper_df_aligned.shape[1] if aper_df_aligned is not None and not aper_df_aligned.empty else 0
    n_itpc_entries = len(itpc_df) if itpc_df is not None else 0
    n_pac_entries = len(features.get("pac_df")) if features.get("pac_df") is not None else 0
    n_all_features = combined_df.shape[1]
    
    logger.info(
        f"Done: sub-{subject}, n_trials={n_trials}, n_direct_features={n_pow_features}, "
        f"n_conn_features={n_conn_features}, n_microstate_features={n_ms_features}, "
        f"n_aper_features={n_aper_features}, n_itpc_entries={n_itpc_entries}, "
        f"n_pac_entries={n_pac_entries}, "
        f"n_all_features={n_all_features}"
    )


###################################################################
# Batch processing
###################################################################


def extract_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    fixed_templates_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    run_group_aggregation: bool = True,
    feature_categories: Optional[List[str]] = None,
) -> None:
    if not subjects:
        raise ValueError("No subjects specified")
    
    if config is None:
        config = load_settings()
    
    if deriv_root is None:
        deriv_root = Path(config.deriv_root)
    
    task = task or config.get("project.task", "thermalactive")
    
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Starting feature extraction: {len(subjects)} subject(s), task={task}")
    
    fixed_path = fixed_templates_path if isinstance(fixed_templates_path, Path) else (
        Path(fixed_templates_path) if fixed_templates_path else None
    )
    
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Processing sub-{subject}")
        process_subject(subject, task, deriv_root, config, fixed_templates_path=fixed_path, feature_categories=feature_categories)
    
    if run_group_aggregation and len(subjects) >= 2:
        logger.info("Running group-level feature aggregation...")
        from eeg_pipeline.analysis.group import aggregate_feature_stats
        aggregate_feature_stats(subjects, task, deriv_root, config)
    
    logger.info(f"Feature extraction complete: {len(subjects)} subjects processed")
