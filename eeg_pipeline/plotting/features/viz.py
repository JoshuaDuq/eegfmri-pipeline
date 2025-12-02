"""
Feature visualization orchestration module.

High-level functions for visualizing all feature types including power, microstates,
connectivity, phase, and aperiodic components.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging
import re

import pandas as pd
import mne
import numpy as np

from ...utils.data.loading import load_epochs_for_analysis, load_feature_bundle_for_subject
from ...utils.io.general import deriv_plots_path, ensure_dir, deriv_stats_path
from ...utils.analysis.tfr import compute_tfr_for_visualization
from ...utils.analysis.windowing import sliding_window_centers
from ...utils.config.loader import get_frequency_band_names

from .power import (
    plot_power_distributions,
    plot_channel_power_heatmap,
    plot_power_time_courses,
    plot_power_spectral_density,
    plot_power_spectral_density_by_pain,
    plot_power_time_course_by_temperature,
)
from .microstates import (
    plot_microstate_templates,
    plot_microstate_coverage_by_pain,
    plot_microstate_temporal_evolution,
    plot_microstate_templates_by_pain,
    plot_microstate_templates_by_temperature,
    plot_microstate_gfp_colored_by_state,
    plot_microstate_gfp_by_temporal_bins,
    plot_microstate_pain_correlation_heatmap,
    plot_microstate_transition_network,
    plot_microstate_duration_distributions,
)
from .aperiodic import (
    plot_aperiodic_topomaps,
    plot_aperiodic_vs_pain,
    plot_aperiodic_r2_histogram,
    plot_aperiodic_residual_spectra,
    plot_aperiodic_run_trajectories,
)
from .connectivity import (
    plot_connectivity_circle_for_band,
    plot_connectivity_heatmap,
    plot_connectivity_network,
    plot_sliding_connectivity_trajectories,
    plot_sliding_degree_heatmap,
)
from .phase import (
    plot_itpc_heatmap,
    plot_itpc_topomaps,
    plot_itpc_behavior_scatter,
    plot_pac_comodulograms,
    plot_pac_behavior_scatter,
    plot_pac_time_ribbons,
)
from .cfc import (
    plot_pac_comodulogram,
    plot_mi_pac_topomaps,
    plot_phase_phase_coupling_matrix,
)
from .dynamics import (
    plot_autocorrelation_decay,
    plot_dfa_scaling,
    plot_mse_complexity_curves,
    plot_neural_timescale_comparison,
)
from .quality import (
    plot_feature_distribution_grid,
    plot_outlier_trials_heatmap,
    plot_snr_distribution,
    plot_missing_data_matrix,
    plot_quality_summary_dashboard,
)

from ...analysis.features.microstates import (
    compute_microstate_metric_correlations,
    compute_microstate_transition_stats,
    compute_microstate_duration_stats,
)
from ...analysis.features.phase import extract_itpc_features

from .power import (
    plot_trial_power_variability,
    plot_inter_band_spatial_power_correlation,
)


###################################################################
# Feature visualization helper
###################################################################


def visualize_subject_features(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
) -> None:
    """Visualize all feature types for a single subject.
    
    Args:
        subject: Subject identifier
        task: Task name
        config: Configuration object
        logger: Logger instance
    """
    logger.info(f"Visualizing features for sub-{subject}...")
    
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="features")
    ensure_dir(plots_dir)

    def _safe_plot(name: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"{name} failed: {exc}")
            return None

    pow_df, ms_df, conn_df, aper_df_loaded, pac_df, pac_trials_df, pac_time_df, itpc_df_long = load_feature_bundle_for_subject(
        subject, config.deriv_root, logger
    )
    
    if pow_df is None:
        logger.error(f"No features found for sub-{subject}")
        return

    if aper_df_loaded is not None:
        aper_df = aper_df_loaded
    else:
        aper_cols = [c for c in pow_df.columns if str(c).startswith("aper_")]
        aper_df = pow_df[aper_cols].copy() if aper_cols else pd.DataFrame()
        if aper_df.empty:
            logger.warning("Aperiodic features not found (features_aperiodic.tsv or aper_* columns). Aperiodic plots will be skipped.")
    
    power_bands = get_frequency_band_names(config)
    
    logger.info("Plotting power distributions...")
    _safe_plot("Power distributions", plot_power_distributions, pow_df, power_bands, subject, plots_dir, logger, config)
    
    logger.info("Plotting power heatmaps...")
    _safe_plot("Power heatmap", plot_channel_power_heatmap, pow_df, power_bands, subject, plots_dir, logger, config)
    
    logger.info("Plotting trial variability...")
    _safe_plot("Trial power variability", plot_trial_power_variability, pow_df, power_bands, subject, plots_dir, logger, config)

    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=False,
        deriv_root=config.deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )

    if epochs is not None:
        logger.info("Plotting aperiodic topomaps...")
        if not aper_df.empty:
            _safe_plot("Aperiodic topomaps", plot_aperiodic_topomaps, aper_df, aligned_events, epochs.info, subject, plots_dir, logger, config)
        else:
            logger.warning("Aperiodic features missing; skipping aperiodic topomaps.")
        
        if aligned_events is not None:
            logger.info("Plotting aperiodic slope vs pain...")
            if not aper_df.empty:
                _safe_plot("Aperiodic vs pain", plot_aperiodic_vs_pain, aper_df, aligned_events, subject, plots_dir, logger, config)
            else:
                logger.warning("Aperiodic features missing; skipping aperiodic slope vs pain plot.")
        _safe_plot("Aperiodic R2 histogram", plot_aperiodic_r2_histogram, subject, plots_dir, logger, config)
        _safe_plot("Aperiodic residual spectra", plot_aperiodic_residual_spectra, subject, plots_dir, logger, config)
        _safe_plot("Aperiodic run trajectories", plot_aperiodic_run_trajectories, subject, plots_dir, logger, config)

        itpc_trial_cols = [c for c in pow_df.columns if str(c).startswith("itpc_")]
        itpc_trial_df = pow_df[itpc_trial_cols].copy() if itpc_trial_cols else pd.DataFrame()

        if itpc_df_long is not None and not itpc_df_long.empty:
            logger.info("Plotting ITPC topomaps...")
            _safe_plot("ITPC topomaps", plot_itpc_topomaps, itpc_df_long, epochs.info, subject, plots_dir, logger, config)
        if not itpc_trial_df.empty and aligned_events is not None:
            logger.info("Plotting ITPC-rating scatter grid...")
            _safe_plot("ITPC-rating scatter", plot_itpc_behavior_scatter, itpc_trial_df, aligned_events, subject, plots_dir, logger, config)

    if pac_df is not None and not pac_df.empty:
        logger.info("Plotting PAC comodulograms...")
        _safe_plot("PAC comodulograms", plot_pac_comodulograms, pac_df, subject, plots_dir, logger, config)
        if pac_time_df is not None and not pac_time_df.empty:
            logger.info("Plotting time-resolved PAC ribbons...")
            _safe_plot("PAC time ribbons", plot_pac_time_ribbons, pac_time_df, subject, plots_dir, logger, config)
        if aligned_events is not None and pac_trials_df is not None and not pac_trials_df.empty:
            logger.info("Plotting PAC-behavior scatter overlays...")
            _safe_plot("PAC-behavior scatter", plot_pac_behavior_scatter, pac_trials_df, aligned_events, subject, plots_dir, logger, config)

    if conn_df is not None and epochs is not None:
        logger.info("Plotting connectivity circles...")
        for measure in ["wpli", "aec"]:
            for band in power_bands:
                _safe_plot(
                    f"Connectivity circle {measure}-{band}",
                    plot_connectivity_circle_for_band,
                    conn_df, epochs.info, subject, plots_dir, logger, config,
                    measure=measure, band=band
                )

        sw_labels = sorted(
            {match.group(1) for col in conn_df.columns for match in [re.match(r"^sw(\d+)corr_all_", col)] if match}
        )
        for lbl in sw_labels:
            prefix = f"sw{lbl}corr_all"
            _safe_plot("Sliding connectivity heatmap", plot_connectivity_heatmap, conn_df, epochs.info, subject, plots_dir, logger, config, prefix, events_df=aligned_events)
            _safe_plot("Sliding connectivity network", plot_connectivity_network, conn_df, epochs.info, subject, plots_dir, logger, config, prefix, events_df=aligned_events)

        if sw_labels:
            window_indices = [int(lbl) for lbl in sw_labels if lbl.isdigit()]
            if window_indices:
                window_centers = sliding_window_centers(config, max(window_indices) + 1)
                _safe_plot(
                    "Sliding connectivity trajectories",
                    plot_sliding_connectivity_trajectories,
                    conn_df,
                    window_indices,
                    window_centers,
                    aligned_events,
                    subject,
                    plots_dir,
                    logger,
                    config,
                )
                _safe_plot(
                    "Sliding degree heatmap",
                    plot_sliding_degree_heatmap,
                    conn_df,
                    window_indices,
                    window_centers,
                    subject,
                    plots_dir,
                    logger,
                    config,
                )

    if epochs is not None:
        tfr = _safe_plot("TFR for visualization", compute_tfr_for_visualization, epochs, config, logger)

        if tfr is not None:
            logger.info("Plotting power time courses...")
            _safe_plot("Power time courses", plot_power_time_courses, tfr, power_bands, subject, plots_dir, logger, config)
            
            logger.info("Plotting inter-band correlations...")
            _safe_plot("Inter-band spatial power correlation", plot_inter_band_spatial_power_correlation, tfr, subject, plots_dir, logger, config)
            
            logger.info("Plotting power spectral density...")
            _safe_plot("Power spectral density", plot_power_spectral_density, tfr, subject, plots_dir, logger, aligned_events, config)
            
            if aligned_events is not None:
                _safe_plot("PSD by pain", plot_power_spectral_density_by_pain, tfr, subject, plots_dir, logger, aligned_events, config)
                
                for band in power_bands:
                    _safe_plot(
                        f"Power time course by temperature ({band})",
                        plot_power_time_course_by_temperature,
                        tfr, subject, plots_dir, logger, aligned_events, band, config
                    )

        logger.info("Plotting ITPC (phase-locking) summaries...")
        itpc_out = _safe_plot("ITPC feature extraction", extract_itpc_features, epochs, power_bands, config, logger)
        if itpc_out is not None:
            itpc_df_vis, _, itpc_map_vis, itpc_freqs_vis, itpc_times_vis, _ = itpc_out
            _safe_plot("ITPC heatmap", plot_itpc_heatmap, itpc_map_vis, itpc_freqs_vis, itpc_times_vis, subject, plots_dir, logger, config)
            _safe_plot("ITPC topomaps (vis)", plot_itpc_topomaps, itpc_df_vis, epochs.info, subject, plots_dir, logger, config)
    
    if ms_df is not None and not ms_df.empty:
        logger.info("Plotting microstate features...")
        n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
        
        if aligned_events is not None and epochs is not None:
            stats_dir = deriv_stats_path(config.deriv_root, subject)
            template_path = stats_dir / f"microstates_templates_K{n_microstates}.npz"
            
            if template_path.exists():
                try:
                    data = np.load(template_path)
                    ms_templates = data['templates']
                    
                    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
                    info_eeg = mne.pick_info(epochs.info, picks)
                    
                    logger.info("Plotting microstate templates...")
                    _safe_plot("Microstate templates", plot_microstate_templates, ms_templates, info_eeg, subject, plots_dir, n_microstates, logger, config)
                    
                    logger.info("Plotting microstate templates by pain...")
                    _safe_plot("Microstate templates by pain", plot_microstate_templates_by_pain, epochs, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
                    
                    logger.info("Plotting microstate templates by temperature...")
                    _safe_plot("Microstate templates by temperature", plot_microstate_templates_by_temperature, epochs, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
                    
                    logger.info("Plotting microstate GFP colored by state...")
                    _safe_plot("Microstate GFP colored by state", plot_microstate_gfp_colored_by_state, epochs, ms_templates, aligned_events, subject, plots_dir, n_microstates, logger, config)
                    
                    logger.info("Plotting microstate GFP by temporal bins...")
                    _safe_plot("Microstate GFP by temporal bins", plot_microstate_gfp_by_temporal_bins, epochs, ms_templates, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
                    
                    logger.info("Plotting microstate coverage by pain...")
                    _safe_plot("Microstate coverage by pain", plot_microstate_coverage_by_pain, ms_df, aligned_events, subject, plots_dir, n_microstates, logger, config)
                    
                    logger.info("Computing and plotting microstate pain correlations...")
                    corr_result = _safe_plot("Microstate metric correlations", compute_microstate_metric_correlations, ms_df, aligned_events, config=config)
                    if corr_result is not None:
                        corr_df, pval_df = corr_result
                        _safe_plot("Microstate pain correlation heatmap", plot_microstate_pain_correlation_heatmap, corr_df, pval_df, subject, plots_dir, logger, config)
                    
                    logger.info("Plotting microstate temporal evolution...")
                    _safe_plot("Microstate temporal evolution", plot_microstate_temporal_evolution, epochs, ms_templates, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
                    
                    logger.info("Computing and plotting microstate transitions...")
                    trans_stats = _safe_plot("Microstate transition stats", compute_microstate_transition_stats, ms_df, aligned_events, n_states=n_microstates, config=config)
                    if trans_stats is not None:
                        _safe_plot("Microstate transition network", plot_microstate_transition_network, trans_stats, subject, plots_dir, logger, config)
                    
                    logger.info("Computing and plotting microstate durations...")
                    dur_stats = _safe_plot("Microstate duration stats", compute_microstate_duration_stats, ms_df, aligned_events, n_states=n_microstates, config=config)
                    if dur_stats is not None:
                        _safe_plot("Microstate duration distributions", plot_microstate_duration_distributions, dur_stats, subject, plots_dir, logger, config)
                except Exception as e:
                    logger.error(f"Error plotting microstates for sub-{subject}: {e}")
    
    # Advanced feature visualizations
    _visualize_cfc_features(pow_df, plots_dir, logger, config)
    _visualize_dynamics_features(pow_df, aligned_events, plots_dir, logger, config)
    _visualize_quality_features(pow_df, plots_dir, logger, config)
    
    logger.info(f"Feature visualizations saved to {plots_dir}")


###################################################################
# Advanced Feature Visualization Helpers
###################################################################


def _visualize_cfc_features(pow_df, plots_dir, logger, config):
    """Visualize cross-frequency coupling features if present."""
    if pow_df is None:
        return
    
    # MI-PAC columns
    mi_cols = [c for c in pow_df.columns if "mi_pac_" in c]
    if mi_cols:
        try:
            logger.info("Plotting MI-PAC topomaps...")
            save_path = plots_dir / "mi_pac_summary.png"
            plot_mi_pac_topomaps(pow_df, None, save_path)
        except Exception as e:
            logger.warning(f"MI-PAC plot failed: {e}")
    
    # PPC columns
    ppc_cols = [c for c in pow_df.columns if "ppc_" in c]
    if ppc_cols:
        try:
            logger.info("Plotting phase-phase coupling...")
            save_path = plots_dir / "phase_phase_coupling.png"
            plot_phase_phase_coupling_matrix(pow_df, save_path)
        except Exception as e:
            logger.warning(f"PPC plot failed: {e}")


def _visualize_dynamics_features(pow_df, events_df, plots_dir, logger, config):
    """Visualize temporal dynamics features if present."""
    if pow_df is None:
        return
    
    # Autocorrelation
    acf_cols = [c for c in pow_df.columns if "acf_" in c]
    if acf_cols:
        try:
            logger.info("Plotting autocorrelation features...")
            condition_col = "condition" if "condition" in pow_df.columns else None
            save_path = plots_dir / "autocorrelation_decay.png"
            plot_autocorrelation_decay(pow_df, save_path, by_condition=condition_col)
        except Exception as e:
            logger.warning(f"ACF plot failed: {e}")
    
    # DFA
    dfa_cols = [c for c in pow_df.columns if "dfa_" in c]
    if dfa_cols:
        try:
            logger.info("Plotting DFA scaling...")
            save_path = plots_dir / "dfa_scaling.png"
            plot_dfa_scaling(pow_df, save_path)
        except Exception as e:
            logger.warning(f"DFA plot failed: {e}")
    
    # MSE
    mse_cols = [c for c in pow_df.columns if "mse_" in c]
    if mse_cols:
        try:
            logger.info("Plotting MSE complexity curves...")
            condition_col = "condition" if "condition" in pow_df.columns else None
            save_path = plots_dir / "mse_complexity.png"
            plot_mse_complexity_curves(pow_df, save_path, by_condition=condition_col)
        except Exception as e:
            logger.warning(f"MSE plot failed: {e}")
    
    # Timescale comparison
    if acf_cols and dfa_cols:
        try:
            logger.info("Plotting neural timescale comparison...")
            condition_col = "condition" if "condition" in pow_df.columns else None
            save_path = plots_dir / "timescale_comparison.png"
            plot_neural_timescale_comparison(pow_df, save_path, by_condition=condition_col)
        except Exception as e:
            logger.warning(f"Timescale comparison failed: {e}")


def _visualize_quality_features(pow_df, plots_dir, logger, config):
    """Visualize feature quality metrics."""
    if pow_df is None or not isinstance(pow_df, pd.DataFrame) or pow_df.empty:
        return
    
    try:
        logger.info("Plotting feature distributions...")
        save_path = plots_dir / "feature_distributions.png"
        plot_feature_distribution_grid(pow_df, save_path, max_features=16)
    except Exception as e:
        logger.warning(f"Distribution grid failed: {e}")
    
    try:
        logger.info("Plotting outlier heatmap...")
        save_path = plots_dir / "outlier_trials.png"
        plot_outlier_trials_heatmap(pow_df, save_path)
    except Exception as e:
        logger.warning(f"Outlier heatmap failed: {e}")
    
    try:
        logger.info("Plotting missing data matrix...")
        save_path = plots_dir / "missing_data.png"
        plot_missing_data_matrix(pow_df, save_path)
    except Exception as e:
        logger.warning(f"Missing data plot failed: {e}")
    
    # SNR if available
    if "snr_db" in pow_df.columns:
        try:
            logger.info("Plotting SNR distribution...")
            save_path = plots_dir / "snr_distribution.png"
            plot_snr_distribution(pow_df, save_path)
        except Exception as e:
            logger.warning(f"SNR plot failed: {e}")


###################################################################
# Batch processing
###################################################################


def visualize_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Visualize features for multiple subjects.
    
    Args:
        subjects: List of subject identifiers
        task: Optional task name (defaults to config)
        deriv_root: Optional derivatives root path
        config: Optional configuration object
        logger: Optional logger instance
    """
    if not subjects:
        raise ValueError("No subjects specified")
    
    if config is None:
        from ...utils.config.loader import load_settings
        config = load_settings()
    
    from ...utils.io.general import setup_matplotlib, get_logger
    
    setup_matplotlib(config)
    
    task = task or config.get("project.task", "thermalactive")
    
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Starting feature visualization: {len(subjects)} subject(s), task={task}")
    
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        visualize_subject_features(subject, task, config, logger)
    
    logger.info("Feature visualization complete")
