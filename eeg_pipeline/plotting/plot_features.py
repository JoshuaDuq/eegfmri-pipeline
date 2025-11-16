from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import stats

from eeg_pipeline.utils.io_utils import (
    get_band_color as _get_band_color,
    unwrap_figure as _unwrap_figure,
    sanitize_label as _sanitize_label,
    build_footer,
    save_fig,
    ensure_dir,
    find_column_in_events,
    find_pain_column_in_events,
    find_temperature_column_in_events,
    find_column_in_metadata,
    find_pain_column_in_metadata,
    find_temperature_column_in_metadata,
    extract_plotting_constants,
    extract_eeg_picks,
    format_baseline_string as _format_baseline_string,
    log_if_present as _log_if_present,
    validate_picks as _validate_picks,
    validate_epochs_for_plotting,
)
from eeg_pipeline.utils.plotting_config import get_plot_config
from eeg_pipeline.utils.stats_utils import (
    compute_statistics_for_mask,
    compute_coverage_statistics,
    compute_band_spatial_correlation,
    fisher_z_transform_mean,
    compute_band_statistics_array,
    compute_consensus_labels,
    compute_inter_band_coupling_matrix,
    compute_group_channel_power_statistics,
    compute_group_band_statistics,
    compute_error_bars_from_arrays,
    compute_band_pair_correlation,
    compute_subject_band_correlation_matrix,
    compute_group_band_correlation_matrix,
    format_correlation_text,
)
from eeg_pipeline.utils.stats_utils import (
    compute_correlation_ci_fisher,
    compute_inter_band_correlation_statistics,
)
from eeg_pipeline.utils.data_loading import (
    validate_aligned_events_length,
    build_epoch_query_string,
    extract_band_channel_vectors,
)
from eeg_pipeline.utils.tfr_utils import extract_band_channel_means
from eeg_pipeline.utils.data_loading import (
    resolve_columns,
    get_aligned_events,
    align_events_with_policy,
    process_temperature_levels,
    select_epochs_by_value,
)
from eeg_pipeline.utils.tfr_utils import validate_baseline_indices
from eeg_pipeline.analysis.feature_extraction import (
    zscore_maps as _zscore_maps,
    compute_gfp as _compute_gfp,
    corr_maps as _corr_maps,
    label_timecourse as _label_timecourse,
    extract_templates_from_trials as _extract_templates_from_trials,
    MicrostateDurationStat,
    MicrostateTransitionStats,
)



###################################################################
# ERP plotting helpers
###################################################################

def _apply_baseline_correction(epochs: mne.Epochs, baseline_window: Tuple[float, float]) -> mne.Epochs:
    baseline_start = float(baseline_window[0])
    baseline_end = min(float(baseline_window[1]), 0.0)
    return epochs.copy().apply_baseline((baseline_start, baseline_end))




def _find_pain_column(epochs: mne.Epochs, config) -> Optional[str]:
    return find_pain_column_in_metadata(epochs, config)


def _find_temperature_column(epochs: mne.Epochs, config) -> Optional[str]:
    return find_temperature_column_in_metadata(epochs, config)


def _plot_evoked_with_agg_backend(evoked: mne.Evoked, picks: str, title: str):
    figure = evoked.plot(picks=picks, spatial_colors=True, show=False)
    figure.suptitle(title)
    return figure


def _save_erp_figure(
    figure, output_path: Path, config, baseline_str: str, method: str, logger
):
    plot_cfg = get_plot_config(config)
    footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=method)
    ensure_dir(output_path.parent)
    save_fig(
        _unwrap_figure(figure),
        output_path,
        logger=logger,
        footer=footer_text,
        tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )


###################################################################
# ERP plotting functions
###################################################################

def erp_contrast_pain(
    epochs: mne.Epochs,
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    pain_color: str,
    nonpain_color: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None,
    subject: Optional[str] = None
) -> None:
    if not validate_epochs_for_plotting(epochs, logger):
        return
    
    pain_column = _find_pain_column(epochs, config)
    if pain_column is None:
        _log_if_present(logger, "warning", "ERP pain contrast: No pain column found in metadata.")
        return

    epochs_baselined = _apply_baseline_correction(epochs, baseline_window)
    epochs_pain = select_epochs_by_value(epochs_baselined, pain_column, 1)
    epochs_nonpain = select_epochs_by_value(epochs_baselined, pain_column, 0)
    if len(epochs_pain) == 0 or len(epochs_nonpain) == 0:
        _log_if_present(logger, "warning", "ERP pain contrast: one of the groups has zero trials.")
        return

    evoked_pain = epochs_pain.average(picks=erp_picks)
    evoked_nonpain = epochs_nonpain.average(picks=erp_picks)
    colors = {"painful": pain_color, "non-painful": nonpain_color}
    
    subject_prefix = f"sub-{subject}_" if subject else ""
    baseline_str = _format_baseline_string(baseline_window)
    
    butterfly_name = erp_output_files.get("pain_butterfly", "erp_pain_binary_butterfly.png")
    butterfly_name = subject_prefix + butterfly_name.replace(".png", ".svg")
    evokeds_dict = {"painful": evoked_pain, "non-painful": evoked_nonpain}
    figure = _create_evoked_comparison_plot(evokeds_dict, erp_picks, None, colors)
    output_path = output_dir / butterfly_name
    _save_erp_figure(figure, output_path, config, baseline_str, erp_combine, logger)


def erp_by_temperature(
    epochs: mne.Epochs,
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None,
    subject: Optional[str] = None
) -> None:
    if not validate_epochs_for_plotting(epochs, logger):
        return
    
    temperature_column = _find_temperature_column(epochs, config)
    if temperature_column is None:
        _log_if_present(logger, "warning", "ERP by temperature: No temperature column found in metadata.")
        return

    epochs_baselined = _apply_baseline_correction(epochs, baseline_window)
    temperature_levels, temperature_labels, is_numeric = process_temperature_levels(
        epochs_baselined, temperature_column
    )
    evokeds_by_temperature: Dict[str, mne.Evoked] = {}
    for level in temperature_levels:
        query, label = build_epoch_query_string(
            temperature_column, level, is_numeric, temperature_labels
        )
        epochs_at_level = epochs_baselined[query]
        if len(epochs_at_level) > 0:
            evokeds_by_temperature[label] = epochs_at_level.average(picks=erp_picks)
    
    if not evokeds_by_temperature:
        _log_if_present(logger, "warning", "ERP by temperature: No evokeds computed.")
        return
    
    subject_prefix = f"sub-{subject}_" if subject else ""
    baseline_str = _format_baseline_string(baseline_window)
    template_name = erp_output_files.get(
        "temp_butterfly_template",
        "erp_by_temperature_butterfly_{label}.png"
    )
    
    for label, evoked in evokeds_by_temperature.items():
        figure = _plot_evoked_with_agg_backend(evoked, erp_picks, f"ERP - Temperature {label}")
        safe_label = _sanitize_label(label)
        output_name = template_name.format(label=safe_label)
        output_name = subject_prefix + output_name.replace(".png", ".svg")
        output_path = output_dir / output_name
        plot_cfg = get_plot_config(config)
        footer_text = build_footer(
            "erp_complete", config=config, baseline=baseline_str, method=erp_combine
        )
        ensure_dir(output_path.parent)
        save_fig(
            figure,
            output_path,
            logger=logger,
            footer=footer_text,
            tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
        )
    
    if len(evokeds_by_temperature) >= 2:
        butterfly_name = erp_output_files.get("temp_butterfly", "erp_by_temperature_butterfly.png")
        butterfly_name = subject_prefix + butterfly_name.replace(".png", ".svg")
        
        figure = _plot_compare_evokeds_silent(
            evokeds_by_temperature, erp_picks, None
        )
        _save_erp_figure(
            figure, output_dir / butterfly_name, config, baseline_str, erp_combine, logger
        )


def group_erp_contrast_pain(
    all_epochs: List[mne.Epochs],
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    pain_color: str,
    nonpain_color: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None
) -> None:
    if not all_epochs:
        return
    
    pain_evokeds: List[mne.Evoked] = []
    nonpain_evokeds: List[mne.Evoked] = []
    for epochs in all_epochs:
        if not validate_epochs_for_plotting(epochs, logger):
            continue
        pain_column = _find_pain_column(epochs, config)
        if pain_column is None:
            continue
        epochs_pain = select_epochs_by_value(epochs, pain_column, 1)
        epochs_nonpain = select_epochs_by_value(epochs, pain_column, 0)
        if len(epochs_pain) > 0:
            pain_evokeds.append(epochs_pain.average(picks=erp_picks))
        if len(epochs_nonpain) > 0:
            nonpain_evokeds.append(epochs_nonpain.average(picks=erp_picks))
    
    if not pain_evokeds or not nonpain_evokeds:
        _log_if_present(logger, "warning", "Group ERP pain contrast: insufficient data across subjects")
        return
    
    grand_average_pain = mne.grand_average(pain_evokeds, interpolate_bads=True)
    grand_average_nonpain = mne.grand_average(nonpain_evokeds, interpolate_bads=True)
    colors = {"painful": pain_color, "non-painful": nonpain_color}
    plot_configs = [
        (erp_combine, erp_output_files.get("pain_gfp", "erp_pain_binary_gfp.png")),
        (None, erp_output_files.get("pain_butterfly", "erp_pain_binary_butterfly.png")),
    ]
    baseline_str = _format_baseline_string(baseline_window)
    
    for combine_method, output_name in plot_configs:
        evokeds_dict = {"painful": grand_average_pain, "non-painful": grand_average_nonpain}
        figure = _create_evoked_comparison_plot(evokeds_dict, erp_picks, combine_method, colors)
        figure = _unwrap_figure(figure)
        figure.suptitle(
            f"Group ERP: Pain vs Non-Pain (N={len(pain_evokeds)} subjects)",
            fontsize=14,
            fontweight='bold'
        )
        group_output_name = "group_" + output_name
        output_path = output_dir / group_output_name
        plot_cfg = get_plot_config(config)
        footer_text = build_footer(
            "erp_complete", config=config, baseline=baseline_str, method=erp_combine
        )
        ensure_dir(output_path.parent)
        save_fig(
            figure,
            output_path,
            logger=logger,
            footer=footer_text,
            tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
        )


###################################################################
# Group ERP Helpers
###################################################################

def _collect_evokeds_by_temperature(all_epochs, config, erp_picks):
    evokeds_by_temperature: Dict[str, List[mne.Evoked]] = {}
    for epochs in all_epochs:
        temperature_column = _find_temperature_column(epochs, config)
        if temperature_column is None:
            continue
        temperature_levels, temperature_labels, is_numeric = (
            process_temperature_levels(epochs, temperature_column)
        )
        for level in temperature_levels:
            query, label = build_epoch_query_string(
                temperature_column, level, is_numeric, temperature_labels
            )
            epochs_at_level = epochs[query]
            if len(epochs_at_level) > 0:
                evoked = epochs_at_level.average(picks=erp_picks)
                evokeds_by_temperature.setdefault(label, []).append(evoked)
    return evokeds_by_temperature


def _save_group_erp_individual_temperature_plots(
    grand_averages, output_dir, erp_picks, erp_combine,
    erp_output_files, baseline_str, config, logger
):
    for label, evoked in grand_averages.items():
        figure = _plot_evoked_with_agg_backend(
            evoked, erp_picks, f"Group ERP - Temperature {label}"
        )
        safe_label = _sanitize_label(label)
        template_name = erp_output_files.get(
            "temp_butterfly_template",
            "erp_by_temperature_butterfly_{label}.png"
        )
        output_name = "group_" + template_name.format(label=safe_label)
        output_path = output_dir / output_name
        plot_cfg = get_plot_config(config)
        footer_text = build_footer(
            "erp_complete", config=config, baseline=baseline_str, method=erp_combine
        )
        ensure_dir(output_path.parent)
        save_fig(
            figure, output_path, logger=logger, footer=footer_text,
            tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
        )


def _plot_compare_evokeds_silent(evokeds, picks, combine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.set_log_level('ERROR')
        return mne.viz.plot_compare_evokeds(
            evokeds, picks=picks, combine=combine, show=False
        )


def _create_evoked_comparison_plot(evokeds_dict, picks, combine_method, colors):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.set_log_level('ERROR')
        return mne.viz.plot_compare_evokeds(
            evokeds_dict, picks=picks, combine=combine_method, show=False, colors=colors
        )


def _save_group_erp_comparison_plot(
    grand_averages, evokeds_by_temperature, output_dir,
    erp_picks, erp_combine, erp_output_files, baseline_str, config, logger,
    combine_method, output_key, title_template
):
    figure = _plot_compare_evokeds_silent(grand_averages, erp_picks, combine_method)
    unwrapped_figure = _unwrap_figure(figure)
    subject_info = ", ".join([
        f"{label}: N={len(evokeds_by_temperature[label])}"
        for label in grand_averages.keys()
    ])
    unwrapped_figure.suptitle(title_template.format(subject_info), fontsize=14, fontweight='bold')
    plot_cfg = get_plot_config(config)
    footer_text = build_footer(
        "erp_complete", config=config, baseline=baseline_str, method=erp_combine
    )
    output_name = "group_" + erp_output_files.get(output_key, f"erp_by_temperature_{output_key}.png")
    output_path = output_dir / output_name
    ensure_dir(output_path.parent)
    save_fig(
        unwrapped_figure, output_path, logger=logger, footer=footer_text,
        tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )


def _save_group_erp_comparison_plots(
    grand_averages, evokeds_by_temperature, output_dir,
    erp_picks, erp_combine, erp_output_files, baseline_str, config, logger
):
    _save_group_erp_comparison_plot(
        grand_averages, evokeds_by_temperature, output_dir,
        erp_picks, erp_combine, erp_output_files, baseline_str, config, logger,
        None, "temp_butterfly", "Group ERP by Temperature (Butterfly) — {}"
    )
    _save_group_erp_comparison_plot(
        grand_averages, evokeds_by_temperature, output_dir,
        erp_picks, erp_combine, erp_output_files, baseline_str, config, logger,
        erp_combine, "temp_gfp", "Group ERP by Temperature ({} subjects)"
    )


def group_erp_by_temperature(
    all_epochs: List[mne.Epochs],
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None
) -> None:
    if not all_epochs:
        return
    
    evokeds_by_temperature = _collect_evokeds_by_temperature(
        all_epochs, config, erp_picks
    )
    
    if not evokeds_by_temperature:
        _log_if_present(logger, "warning", "Group ERP by temperature: No evokeds computed across subjects")
        return
    
    grand_averages = {
        label: mne.grand_average(evokeds, interpolate_bads=True)
        for label, evokeds in evokeds_by_temperature.items()
        if evokeds
    }
    
    baseline_str = _format_baseline_string(baseline_window)
    
    _save_group_erp_individual_temperature_plots(
        grand_averages, output_dir, erp_picks, erp_combine,
        erp_output_files, baseline_str, config, logger
    )
    
    if len(grand_averages) >= 2:
        _save_group_erp_comparison_plots(
            grand_averages, evokeds_by_temperature, output_dir,
            erp_picks, erp_combine, erp_output_files, baseline_str, config, logger
        )


###################################################################
# Microstate plotting
###################################################################

def _create_state_letters(n_states: int) -> List[str]:
    return [chr(65 + i) for i in range(n_states)]


def _setup_template_grid(n_states: int, plot_cfg) -> Tuple[plt.Figure, List[plt.Axes]]:
    n_cols = min(4, n_states)
    n_rows = int(np.ceil(n_states / n_cols))
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    width_per_state = plot_cfg.figure_sizes.get("microstate_width_per_state", 3.6)
    height_per_state = plot_cfg.figure_sizes.get("microstate_height_per_state", 3.2)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(width_per_state * n_cols, height_per_state * n_rows)
    )
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    return fig, axes




def plot_microstate_templates(templates, info, subject, save_dir, n_states, logger, config):
    if templates is None or len(templates) == 0:
        _log_if_present(logger, "warning", "No templates to plot")
        return
    
    plot_cfg = get_plot_config(config)
    fig, axes = _setup_template_grid(n_states, plot_cfg)
    state_letters = _create_state_letters(n_states)
    
    for i in range(n_states):
        mne.viz.plot_topomap(templates[i], info, axes=axes[i], show=False, contours=6, cmap="RdBu_r")
        axes[i].set_title(f"State {state_letters[i]}", fontsize=plot_cfg.font.title)
    
    for j in range(n_states, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Microstate Templates (K={n_states})", fontsize=plot_cfg.font.figure_title)
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    tight_rect = microstate_config.get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.tight_layout(rect=tight_rect)
    save_fig(fig, save_dir / f"sub-{subject}_microstate_templates", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi, 
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate templates")




def plot_microstate_coverage_by_pain(ms_df, events_df, subject, save_dir, n_states, logger, config):
    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for coverage by pain plot")
        return
    
    pain_col, _, _ = resolve_columns(events_df, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    if len(ms_df) != len(events_df):
        raise ValueError(
            f"Microstate dataframe ({len(ms_df)} rows) and events "
            f"({len(events_df)} rows) length mismatch for subject {subject}"
        )
    
    pain_values = pd.to_numeric(events_df[pain_col], errors="coerce")
    valid_mask = pain_values.notna()
    nonpain_mask = valid_mask & (pain_values == 0)
    pain_mask = valid_mask & (pain_values == 1)
    state_letters = _create_state_letters(n_states)
    means_nonpain, means_pain, sems_nonpain, sems_pain = [], [], [], []
    
    for state_idx in range(n_states):
        coverage_column = f"ms_coverage_{state_idx}"
        if coverage_column not in ms_df.columns:
            means_nonpain.append(0.0)
            means_pain.append(0.0)
            sems_nonpain.append(0.0)
            sems_pain.append(0.0)
            continue
        
        coverage_values = pd.to_numeric(ms_df[coverage_column], errors="coerce")
        mean_np, mean_p, sem_np, sem_p = compute_coverage_statistics(
            coverage_values, nonpain_mask, pain_mask
        )
        means_nonpain.append(mean_np)
        means_pain.append(mean_p)
        sems_nonpain.append(sem_np)
        sems_pain.append(sem_p)
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    x_positions = np.arange(n_states)
    fig, ax = plt.subplots(figsize=fig_size)
    
    nonpain_color = plot_cfg.get_color("nonpain", plot_type="features")
    pain_color = plot_cfg.get_color("pain", plot_type="features")
    
    ax.bar(
        x_positions - plot_cfg.style.bar.width/2, means_nonpain, plot_cfg.style.bar.width, 
        yerr=sems_nonpain, label='Non-pain', color=nonpain_color, 
        alpha=plot_cfg.style.bar.alpha, capsize=plot_cfg.style.bar.capsize
    )
    ax.bar(
        x_positions + plot_cfg.style.bar.width/2, means_pain, plot_cfg.style.bar.width, 
        yerr=sems_pain, label='Pain', color=pain_color, 
        alpha=plot_cfg.style.bar.alpha, capsize=plot_cfg.style.bar.capsize
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(state_letters)
    ax.set_xlabel("Microstate", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Coverage (fraction of time)", fontsize=plot_cfg.font.ylabel)
    ax.set_title("Microstate Coverage by Pain Condition", fontsize=plot_cfg.font.figure_title)
    ax.legend(fontsize=plot_cfg.font.title)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, axis='y')
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_coverage_by_pain",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate coverage by pain condition")




def _prepare_pvalue_matrix(pval_df, metric_labels, state_labels, corr_matrix_shape):
    if pval_df is not None and not pval_df.empty:
        return pval_df.reindex(index=metric_labels, columns=state_labels).to_numpy(dtype=float)
    return np.full_like(corr_matrix_shape, np.nan)


def plot_microstate_pain_correlation_heatmap(corr_df, pval_df, subject, save_dir, logger, config):
    if corr_df is None or corr_df.empty:
        _log_if_present(logger, "warning", "No microstate correlation data provided; skipping heatmap")
        return

    plot_cfg = get_plot_config(config)
    features_config = plot_cfg.plot_type_configs.get("features", {})
    correlation_config = features_config.get("correlation", {})

    metric_labels = list(corr_df.index)
    state_labels = list(corr_df.columns)
    n_states = len(state_labels)
    corr_matrix = corr_df.to_numpy(dtype=float)
    p_matrix = _prepare_pvalue_matrix(pval_df, metric_labels, state_labels, corr_matrix)

    width_per_col = plot_cfg.figure_sizes.get("microstate_width_per_column", 1.2)
    height_per_row = plot_cfg.figure_sizes.get("microstate_height_per_row", 1.0)

    fig, ax = plt.subplots(
        figsize=(
            max(6, n_states * width_per_col),
            max(5, len(metric_labels) * height_per_row)
        )
    )
    
    vmin = correlation_config.get("vmin", -0.6)
    vmax = correlation_config.get("vmax", 0.6)
    threshold_text = correlation_config.get("threshold_text", 0.4)
    
    im = ax.imshow(
        corr_matrix, cmap="RdBu_r",
        vmin=vmin, vmax=vmax, aspect="auto"
    )
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(state_labels)
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Microstate", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Metric", fontsize=plot_cfg.font.ylabel)
    ax.set_title("Microstate-Pain Rating Correlations (Spearman r)", fontsize=plot_cfg.font.figure_title)

    for i, metric in enumerate(metric_labels):
        for j, state in enumerate(state_labels):
            value = corr_matrix[i, j]
            if not np.isfinite(value):
                continue
            
            text_color = "white" if abs(value) > threshold_text else "black"
            text = format_correlation_text(value, p_matrix[i, j])
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=plot_cfg.font.medium)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spearman r", fontsize=plot_cfg.font.title)
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_pain_correlation",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate-pain correlation heatmap")



def plot_microstate_temporal_evolution(
    epochs, templates, events_df, subject, task, save_dir, n_states, logger, config
):
    if templates is None or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for temporal evolution plot")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        return
    
    epoch_data = epochs.get_data()[:, picks, :]
    times = epochs.times
    pain_values = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    nonpain_mask = (pain_values == 0).to_numpy()
    pain_mask = (pain_values == 1).to_numpy()
    state_probabilities_nonpain = np.zeros((n_states, len(times)))
    state_probabilities_pain = np.zeros((n_states, len(times)))
    
    for trial_idx in range(len(epoch_data)):
        epoch = epoch_data[trial_idx]
        state_labels, _ = _label_timecourse(epoch, templates)
        
        if nonpain_mask[trial_idx]:
            for time_idx, state_idx in enumerate(state_labels):
                state_probabilities_nonpain[state_idx, time_idx] += 1
        elif pain_mask[trial_idx]:
            for time_idx, state_idx in enumerate(state_labels):
                state_probabilities_pain[state_idx, time_idx] += 1
    
    n_nonpain_trials = max(1, nonpain_mask.sum())
    n_pain_trials = max(1, pain_mask.sum())
    state_probabilities_nonpain /= n_nonpain_trials
    state_probabilities_pain /= n_pain_trials
    
    plot_cfg = get_plot_config(config)
    state_letters = _create_state_letters(n_states)
    colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for state_idx in range(n_states):
        axes[0].plot(
            times, state_probabilities_nonpain[state_idx],
            label=f"State {state_letters[state_idx]}",
            color=colors[state_idx], linewidth=plot_cfg.style.line.width_thick
        )
        axes[1].plot(
            times, state_probabilities_pain[state_idx],
            label=f"State {state_letters[state_idx]}",
            color=colors[state_idx], linewidth=plot_cfg.style.line.width_thick
        )
    
    axes[0].axvline(stimulus_start_time, color='k', linestyle='--',
                    linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    axes[1].axvline(stimulus_start_time, color='k', linestyle='--',
                    linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    axes[0].set_ylabel("Probability", fontsize=plot_cfg.font.ylabel)
    axes[0].set_title("Non-pain Trials", fontsize=plot_cfg.font.large)
    axes[0].legend(loc='upper right', fontsize=plot_cfg.font.medium, ncol=n_states)
    axes[0].grid(True, alpha=plot_cfg.style.alpha_grid)
    axes[0].set_ylim([0, 1])
    axes[1].set_ylabel("Probability", fontsize=plot_cfg.font.ylabel)
    axes[1].set_xlabel("Time (s)", fontsize=plot_cfg.font.label)
    axes[1].set_title("Pain Trials", fontsize=plot_cfg.font.large)
    axes[1].legend(loc='upper right', fontsize=plot_cfg.font.medium, ncol=n_states)
    axes[1].grid(True, alpha=plot_cfg.style.alpha_grid)
    axes[1].set_ylim([0, 1])
    
    tight_rect = microstate_config.get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.suptitle("Temporal Evolution of Microstate Probabilities", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=tight_rect)
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_temporal_evolution",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate temporal evolution")


def plot_microstate_templates_by_pain(
    epochs, events_df, subject, task, save_dir, n_states, logger, config
):
    if events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing events for pain-specific templates")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        _log_if_present(logger, "warning", "No EEG channels for pain-specific templates")
        return
    
    pain_vals = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    nonpain_mask = (pain_vals == 0).to_numpy()
    pain_mask = (pain_vals == 1).to_numpy()
    
    plot_cfg = get_plot_config(config)
    min_trials_for_templates = plot_cfg.validation.get("min_trials_for_templates", 5)
    if (nonpain_mask.sum() < min_trials_for_templates or
            pain_mask.sum() < min_trials_for_templates):
        _log_if_present(logger, "warning", "Insufficient trials for pain-specific templates")
        return
    
    X = epochs.get_data()[:, picks, :]
    sfreq = float(epochs.info["sfreq"])
    templates_nonpain = _extract_templates_from_trials(
        X[nonpain_mask], sfreq, n_states, config
    )
    templates_pain = _extract_templates_from_trials(
        X[pain_mask], sfreq, n_states, config
    )
    
    if templates_nonpain is None or templates_pain is None:
        _log_if_present(logger, "warning", "Could not compute pain-specific templates")
        return
    
    plot_cfg = get_plot_config(config)
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    width_per_state = plot_cfg.figure_sizes.get("microstate_width_per_state", 3.6)
    height_templates = plot_cfg.figure_sizes.get("microstate_height_templates", 7.0)
    fig, axes = plt.subplots(
        2, n_states,
        figsize=(width_per_state * n_states, height_templates)
    )
    if n_states == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_states):
        mne.viz.plot_topomap(
            templates_nonpain[i], info_eeg, axes=axes[0, i],
            show=False, contours=6, cmap="RdBu_r"
        )
        axes[0, i].set_title(f"State {state_letters[i]}", fontsize=10)
        mne.viz.plot_topomap(
            templates_pain[i], info_eeg, axes=axes[1, i],
            show=False, contours=6, cmap="RdBu_r"
        )
    
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    label_x_offset = microstate_config.get("label_offset_x", -0.3)
    label_y_position = microstate_config.get("label_y_position", 0.5)
    axes[0, 0].text(
        label_x_offset, label_y_position, "Non-pain",
        transform=axes[0, 0].transAxes,
        fontsize=plot_cfg.font.large, rotation=90, va='center', weight='bold'
    )
    axes[1, 0].text(
        label_x_offset, label_y_position, "Pain",
        transform=axes[1, 0].transAxes,
        fontsize=plot_cfg.font.large, rotation=90, va='center', weight='bold'
    )
    
    plot_cfg = get_plot_config(config)
    tight_rect = plot_cfg.plot_type_configs.get("microstate", {}).get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.suptitle(f"Microstate Templates by Pain Condition (K={n_states})", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=tight_rect)
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_templates_by_pain",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate templates by pain condition")


def plot_microstate_templates_by_temperature(
    epochs, events_df, subject, task, save_dir, n_states, logger, config
):
    if events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing events for temperature-specific templates")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    temp_col = find_temperature_column_in_events(aligned_events)
    if temp_col is None:
        _log_if_present(logger, "warning", "No temperature column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        _log_if_present(logger, "warning", "No EEG channels for temperature-specific templates")
        return
    
    temps = pd.to_numeric(aligned_events[temp_col], errors="coerce")
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < 2:
        _log_if_present(logger, "warning", "Insufficient temperature levels for comparison")
        return
    
    X = epochs.get_data()[:, picks, :]
    sfreq = float(epochs.info["sfreq"])
    templates_by_temp = {}
    
    for temp in unique_temps:
        temp_mask = (temps == temp).to_numpy()
        plot_cfg = get_plot_config(config)
        min_trials_for_templates = plot_cfg.validation.get("min_trials_for_templates", 5)
        if temp_mask.sum() < min_trials_for_templates:
            continue
        templates = _extract_templates_from_trials(X[temp_mask], sfreq, n_states, config)
        if templates is not None:
            templates_by_temp[temp] = templates
    
    if len(templates_by_temp) < 2:
        _log_if_present(logger, "warning", "Could not compute templates for multiple temperatures")
        return
    
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    plot_cfg = get_plot_config(config)
    sorted_temps = sorted(templates_by_temp.keys())
    n_temps = len(sorted_temps)
    width_per_state = plot_cfg.figure_sizes.get("microstate_width_per_state", 3.6)
    height_per_state = plot_cfg.figure_sizes.get("microstate_height_per_state", 3.2)
    fig, axes = plt.subplots(
        n_temps, n_states,
        figsize=(width_per_state * n_states, height_per_state * n_temps)
    )
    if n_temps == 1 or n_states == 1:
        axes = axes.reshape(n_temps, n_states)
    
    for row_idx, temp in enumerate(sorted_temps):
        templates = templates_by_temp[temp]
        for col_idx in range(n_states):
            mne.viz.plot_topomap(
                templates[col_idx], info_eeg, axes=axes[row_idx, col_idx],
                show=False, contours=6, cmap="RdBu_r"
            )
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    f"State {state_letters[col_idx]}", fontsize=10
                )
        microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
        temp_label_x = microstate_config.get("temperature_label_x", -0.35)
        label_y_position = microstate_config.get("label_y_position", 0.5)
        axes[row_idx, 0].text(
            temp_label_x, label_y_position, f"{temp:.1f}°C",
            transform=axes[row_idx, 0].transAxes,
            fontsize=plot_cfg.font.large, rotation=90, va='center', weight='bold'
        )
    
    plot_cfg = get_plot_config(config)
    tight_rect = plot_cfg.plot_type_configs.get("microstate", {}).get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.suptitle(f"Microstate Templates by Temperature (K={n_states})", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=tight_rect)
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_templates_by_temperature",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate templates by temperature")


def _compute_gfp_and_labels_for_condition(epoch_data_condition, templates, time_mask):
    gfp_all_trials = []
    labels_all_trials = []
    for epoch in epoch_data_condition:
        gfp = _compute_gfp(epoch)
        state_labels, _ = _label_timecourse(epoch, templates)
        gfp_all_trials.append(gfp[time_mask])
        labels_all_trials.append(state_labels[time_mask])
    return gfp_all_trials, labels_all_trials


def _plot_topomap_row(fig, gs, templates, info_eeg, state_letters, colors, n_states):
    for state_idx in range(n_states):
        ax_topo = fig.add_subplot(gs[0, state_idx])
        mne.viz.plot_topomap(
            templates[state_idx], info_eeg, axes=ax_topo,
            show=False, contours=6, cmap="RdBu_r"
        )
        ax_topo.set_title(
            f"State {state_letters[state_idx]}", fontsize=11, weight='bold', color=colors[state_idx]
        )


def _plot_gfp_sequence(ax_gfp, times, gfp_mean, labels_consensus, colors, plateau_start, plot_cfg):
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    for time_idx in range(len(times) - 1):
        state = labels_consensus[time_idx]
        ax_gfp.fill_between(
            [times[time_idx], times[time_idx + 1]], 0, gfp_mean[time_idx],
            color=colors[state], alpha=plot_cfg.style.alpha_fill, linewidth=0
        )
    ax_gfp.plot(times, gfp_mean, 'k-', linewidth=plot_cfg.style.line.width_thick, alpha=plot_cfg.style.line.alpha_standard)
    ax_gfp.axvline(stimulus_start_time, color='gray', linestyle='--',
                   linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    ax_gfp.axvline(plateau_start, color='gray', linestyle=':',
                   linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    ax_gfp.set_xlabel("Time (s)", fontsize=plot_cfg.font.large)
    ax_gfp.set_ylabel("GFP (μV)", fontsize=plot_cfg.font.large)
    ax_gfp.set_xlim([times[0], times[-1]])
    ax_gfp.grid(True, alpha=plot_cfg.style.alpha_grid, axis='y')


def _validate_gfp_plotting_inputs(epochs, templates, events_df, logger) -> bool:
    if templates is None or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for GFP microstate plot")
        return False
    return True




def _prepare_gfp_plotting_data(epochs, events_df, config, logger):
    pain_col, _, _ = resolve_columns(events_df, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return None, None, None, None, None
    
    aligned_events = align_events_with_policy(events_df, epochs, config=config, logger=logger)
    if not validate_aligned_events_length(aligned_events, epochs, logger):
        return None, None, None, None, None
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        return None, None, None, None, None
    
    pain_values = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    epoch_data = epochs.get_data()[:, picks, :]
    times = epochs.times
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
    plateau_end = float(plateau_window[1])
    stimulus_mask = (times >= 0.0) & (times <= plateau_end)
    
    if not stimulus_mask.any():
        _log_if_present(logger, "warning", "No timepoints in stimulus window")
        return None, None, None, None, None
    
    times_stimulus = times[stimulus_mask]
    nonpain_mask = (pain_values == 0).to_numpy()
    pain_mask = (pain_values == 1).to_numpy()
    
    n_nonpain_trials = nonpain_mask.sum()
    n_pain_trials = pain_mask.sum()
    plot_cfg = get_plot_config(config)
    min_trials_for_comparison = plot_cfg.validation.get("min_trials_for_comparison", 1)
    if n_nonpain_trials < min_trials_for_comparison or n_pain_trials < min_trials_for_comparison:
        _log_if_present(logger, "warning", "Insufficient trials for pain comparison")
        return None, None, None, None, None
    
    return epoch_data, times_stimulus, nonpain_mask, pain_mask, plateau_window


def _create_gfp_sequence_figure(
    templates, info_eeg, state_letters, colors, n_states,
    times_stimulus, gfp_mean, labels_consensus, plateau_start, plot_cfg
):
    fig = plt.figure(figsize=(14, 6))
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    grid_height_ratio_topomap = microstate_config.get("grid_height_ratio_topomap", 1.0)
    grid_height_ratio_gfp = microstate_config.get("grid_height_ratio_gfp", 1.5)
    grid_hspace = microstate_config.get("grid_hspace", 0.15)
    grid_wspace = microstate_config.get("grid_wspace", 0.3)
    gs = fig.add_gridspec(
        2, n_states,
        height_ratios=[grid_height_ratio_topomap, grid_height_ratio_gfp],
        hspace=grid_hspace, wspace=grid_wspace
    )
    
    _plot_topomap_row(fig, gs, templates, info_eeg, state_letters, colors, n_states)
    
    ax_gfp = fig.add_subplot(gs[1, :])
    _plot_gfp_sequence(ax_gfp, times_stimulus, gfp_mean, labels_consensus, colors, plateau_start, plot_cfg)
    
    return fig


def plot_microstate_gfp_colored_by_state(
    epochs, templates, events_df, subject, save_dir, n_states, logger, config
):
    if not _validate_gfp_plotting_inputs(epochs, templates, events_df, logger):
        return
    
    result = _prepare_gfp_plotting_data(epochs, events_df, config, logger)
    if result[0] is None:
        return
    
    epoch_data, times_stimulus, nonpain_mask, pain_mask, plateau_window = result
    
    picks = extract_eeg_picks(epochs)
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    plateau_start = float(plateau_window[0])
    plateau_end = float(plateau_window[1])
    plot_cfg = get_plot_config(config)
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    gfp_scale_factor = microstate_config.get("gfp_scale_factor", 1e6)
    min_trials_for_comparison = plot_cfg.validation.get("min_trials_for_comparison", 1)
    
    times = epochs.times
    stimulus_mask = (times >= stimulus_start_time) & (times <= plateau_end)
    
    for condition_mask, condition_label in [(nonpain_mask, "nonpain"), (pain_mask, "pain")]:
        if condition_mask.sum() < min_trials_for_comparison:
            continue
        
        epoch_data_condition = epoch_data[condition_mask]
        gfp_all_trials, labels_all_trials = _compute_gfp_and_labels_for_condition(
            epoch_data_condition, templates, stimulus_mask
        )
        
        gfp_mean = np.mean(gfp_all_trials, axis=0) * gfp_scale_factor
        labels_consensus = compute_consensus_labels(labels_all_trials, len(times_stimulus))
        
        fig = _create_gfp_sequence_figure(
            templates, info_eeg, state_letters, colors, n_states,
            times_stimulus, gfp_mean, labels_consensus, plateau_start, plot_cfg
        )
        
        condition_title = "Non-pain" if condition_label == "nonpain" else "Pain"
        fig.suptitle(
            f"Microstate Sequence - {condition_title} Trials (Stimulus Period)",
            fontsize=13,
            weight='bold'
        )
        plot_cfg = get_plot_config(config)
        tight_rect = plot_cfg.plot_type_configs.get("microstate", {}).get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
        plt.tight_layout(rect=tight_rect)
        save_fig(
            fig,
            save_dir / f"sub-{subject}_microstate_gfp_sequence_{condition_label}",
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
        )
        plt.close(fig)
    
    _log_if_present(logger, "info", "Saved microstate GFP sequence plots")


def _parse_temporal_bin_config(bin_config, config=None) -> Optional[Tuple[float, float, str]]:
    if isinstance(bin_config, dict):
        if config is not None:
            plot_cfg = get_plot_config(config)
            microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
            stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
        else:
            stimulus_start_time = 0.0
        start_time = float(bin_config.get("start", stimulus_start_time))
        end_time = float(bin_config.get("end", stimulus_start_time))
        label = str(bin_config.get("label", "unknown"))
        return start_time, end_time, label
    
    if isinstance(bin_config, (list, tuple)) and len(bin_config) >= 3:
        start_time = float(bin_config[0])
        end_time = float(bin_config[1])
        label = str(bin_config[2])
        return start_time, end_time, label
    
    return None


def _compute_gfp_and_labels_for_bin(epoch_data_condition, templates, bin_mask):
    gfp_all_trials = []
    labels_all_trials = []
    for epoch in epoch_data_condition:
        gfp = _compute_gfp(epoch)
        state_labels, _ = _label_timecourse(epoch, templates)
        gfp_all_trials.append(gfp[bin_mask])
        labels_all_trials.append(state_labels[bin_mask])
    return gfp_all_trials, labels_all_trials




def _plot_gfp_sequence_for_bin(
    templates, info_eeg, times_bin, gfp_mean, labels_consensus,
    state_letters, colors, n_states, condition_title, temporal_label, plot_cfg
):
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    grid_height_ratio_topomap = microstate_config.get("grid_height_ratio_topomap", 1.0)
    grid_height_ratio_gfp = microstate_config.get("grid_height_ratio_gfp", 1.5)
    grid_hspace = microstate_config.get("grid_hspace", 0.15)
    grid_wspace = microstate_config.get("grid_wspace", 0.3)
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(
        2, n_states,
        height_ratios=[grid_height_ratio_topomap, grid_height_ratio_gfp],
        hspace=grid_hspace, wspace=grid_wspace
    )
    
    _plot_topomap_row(fig, gs, templates, info_eeg, state_letters, colors, n_states)
    
    ax_gfp = fig.add_subplot(gs[1, :])
    plateau_start = times_bin[0] if times_bin[0] > 0 else None
    _plot_gfp_sequence(ax_gfp, times_bin, gfp_mean, labels_consensus, colors, plateau_start, plot_cfg)
    if times_bin[0] <= stimulus_start_time <= times_bin[-1]:
        ax_gfp.axvline(
            stimulus_start_time, color='gray', linestyle='--',
            linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim
        )
    
    fig.suptitle(
        f"Microstate Sequence - {condition_title} Trials ({temporal_label.capitalize()} Period)",
        fontsize=13, weight='bold'
    )
    return fig


def plot_microstate_gfp_by_temporal_bins(
    epochs, templates, events_df, subject, task, save_dir, n_states, logger, config
):
    if templates is None or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for temporal bin GFP plots")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        return
    
    pain_values = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    epoch_data = epochs.get_data()[:, picks, :]
    times = epochs.times
    nonpain_mask = (pain_values == 0).to_numpy()
    pain_mask = (pain_values == 1).to_numpy()
    
    if nonpain_mask.sum() < 1 or pain_mask.sum() < 1:
        _log_if_present(logger, "warning", "Insufficient trials for pain comparison")
        return
    
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    temporal_bins = config.get("feature_engineering.features.temporal_bins", [])
    
    for bin_config in temporal_bins:
        bin_params = _parse_temporal_bin_config(bin_config, config)
        if bin_params is None:
            _log_if_present(logger, "warning", f"Invalid temporal bin configuration: {bin_config}; skipping")
            continue
        
        t_start, t_end, temporal_label = bin_params
        bin_mask = (times >= t_start) & (times <= t_end)
        if not bin_mask.any():
            _log_if_present(logger, "warning", f"No timepoints in {temporal_label} bin")
            continue
        
        times_bin = times[bin_mask]
        
        for condition_mask, condition_label in [(nonpain_mask, "nonpain"), (pain_mask, "pain")]:
            if condition_mask.sum() < 1:
                continue
            
            epoch_data_condition = epoch_data[condition_mask]
            gfp_all, labels_all = _compute_gfp_and_labels_for_bin(
                epoch_data_condition, templates, bin_mask
            )
            plot_cfg = get_plot_config(config)
            microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
            gfp_scale_factor = microstate_config.get("gfp_scale_factor", 1e6)
            gfp_mean = np.mean(gfp_all, axis=0) * gfp_scale_factor
            labels_consensus = compute_consensus_labels(labels_all, len(times_bin))
            
            condition_title = "Non-pain" if condition_label == "nonpain" else "Pain"
            fig = _plot_gfp_sequence_for_bin(
                templates, info_eeg, times_bin, gfp_mean, labels_consensus,
                state_letters, colors, n_states, condition_title, temporal_label, plot_cfg
            )
            
            tight_rect = microstate_config.get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
            plt.tight_layout(rect=tight_rect)
            output_path = save_dir / f"sub-{subject}_microstate_gfp_sequence_{condition_label}_{temporal_label}"
            save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                     bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
            plt.close(fig)
    
    _log_if_present(logger, "info", "Saved microstate GFP sequence plots by temporal bins")


def plot_microstate_transition_network(transitions: MicrostateTransitionStats, subject, save_dir, logger, config):
    if transitions is None:
        _log_if_present(logger, "warning", "No microstate transition data provided; skipping plot")
        return

    state_labels = transitions.state_labels
    n_states = len(state_labels)
    if n_states == 0:
        _log_if_present(logger, "warning", "Empty transition matrices; skipping plot")
        return

    trans_nonpain = transitions.nonpain
    trans_pain = transitions.pain
    vmax = max(float(np.max(trans_nonpain)), float(np.max(trans_pain)), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in zip(axes, [trans_nonpain, trans_pain], ["Non-pain", "Pain"]):
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(state_labels)
        ax.set_yticklabels(state_labels)
        ax.set_xlabel("To State", fontsize=10)
        ax.set_ylabel("From State", fontsize=10)
        ax.set_title(f"{title} Transitions", fontsize=11)
        for i in range(n_states):
            for j in range(n_states):
                value = matrix[i, j]
                if value <= 0:
                    continue
                text_color = "white" if value > 0.5 * vmax else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        plt.colorbar(im, ax=ax, label="Probability", shrink=0.8)

    plot_cfg = get_plot_config(config)
    plt.suptitle("Microstate Transition Probabilities by Condition", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_transitions",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate transition network")



def plot_microstate_duration_distributions(duration_stats: List[MicrostateDurationStat], subject, save_dir, logger, config):
    if not duration_stats:
        _log_if_present(logger, "warning", "No microstate duration statistics provided; skipping violin plot")
        return

    n_states = len(duration_stats)
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for ax, stat in zip(axes, duration_stats):
        nonpain_data = stat.nonpain
        pain_data = stat.pain
        if nonpain_data.size == 0 and pain_data.size == 0:
            ax.set_visible(False)
            continue

        data_to_plot: List[np.ndarray] = []
        labels: List[str] = []
        colors_to_use: List[str] = []

        if nonpain_data.size:
            data_to_plot.append(nonpain_data)
            labels.append("Non-pain")
            colors_to_use.append("steelblue")
        if pain_data.size:
            data_to_plot.append(pain_data)
            labels.append("Pain")
            colors_to_use.append("orangered")

        if data_to_plot:
            parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmeans=True, showmedians=True)
            for pc, color in zip(parts['bodies'], colors_to_use):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Duration (s)", fontsize=10)
        ax.set_title(f"State {stat.state}", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        if nonpain_data.size and pain_data.size and np.isfinite(stat.p_value):
            if stat.p_value < 0.05:
                y_min, y_max = ax.get_ylim()
                ax.plot([0, 1], [y_max * 0.95, y_max * 0.95], 'k-', linewidth=1)
                sig_text = "**" if stat.p_value < 0.01 else "*"
                ax.text(0.5, y_max * 0.97, sig_text, ha='center', fontsize=12)

    if n_states > 0:
        axes[-1].set_xlabel("Condition", fontsize=10)

    plot_cfg = get_plot_config(config)
    plt.suptitle("Microstate Duration Distributions by Pain Condition", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_duration_distributions",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate duration distributions")


###################################################################
# Power-related plotting
###################################################################

def _setup_subplot_grid(n_items: int, n_cols: int = 2) -> Tuple[plt.Figure, List[plt.Axes]]:
    n_rows = (n_items + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    if n_items == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    return fig, axes


def plot_power_distributions(pow_df, bands, subject, save_dir, logger, config):
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {})
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    n_bands = len(bands)
    fig, axes = _setup_subplot_grid(n_bands)
    
    for i, band in enumerate(bands):
        band = str(band)
        band_cols = [col for col in pow_df.columns if str(col).startswith(f'{power_prefix}{band}_')]
        if not band_cols:
            logger.warning(f"No columns found for band '{band}'")
            continue
        
        band_data = pow_df[band_cols].values.flatten()
        band_data = band_data[~np.isnan(band_data)]
        if len(band_data) == 0:
            logger.warning(f"No valid data for band '{band}'")
            continue
        
        parts = axes[i].violinplot(
            [band_data], positions=[1], showmeans=True, showmedians=True
        )
        band_color = _get_band_color(band, config)
        for pc in parts['bodies']:
            pc.set_facecolor(band_color)
            pc.set_alpha(plot_cfg.style.alpha_violin_body)
        
        axes[i].axhline(y=0, color=plot_cfg.style.colors.red, linestyle='--', 
                       alpha=plot_cfg.style.line.alpha_reference, label='Baseline')
        axes[i].set_title(f'{band.capitalize()} Power Distribution\n(All channels, all trials)',
                         fontsize=plot_cfg.font.title)
        axes[i].set_ylabel('log10(power/baseline)', fontsize=plot_cfg.font.ylabel)
        axes[i].set_xticks([])
        axes[i].grid(True, alpha=plot_cfg.style.alpha_grid)
        
        mean_val = np.mean(band_data)
        std_val = np.std(band_data)
        median_val = np.median(band_data)
        stats_text = (
            f'μ={mean_val:.3f}\nσ={std_val:.3f}\n'
            f'Mdn={median_val:.3f}\nn={len(band_data)}'
        )
        axes[i].text(
            0.7, 0.95, stats_text, transform=axes[i].transAxes,
            verticalalignment='top', fontsize=plot_cfg.font.small,
            bbox=dict(boxstyle='round', facecolor='white', alpha=plot_cfg.style.alpha_text_box)
        )
    
    for j in range(len(bands), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f'sub-{subject}_power_distributions_per_band',
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    logger.info("Saved power distributions")


def plot_channel_power_heatmap(pow_df, bands, subject, save_dir, logger, config):
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {})
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    band_means = []
    channel_names = []
    valid_bands = []
    
    for band in bands:
        band = str(band)
        band_cols = [col for col in pow_df.columns if str(col).startswith(f'{power_prefix}{band}_')]
        if band_cols:
            band_data = pow_df[band_cols].mean(axis=0)
            band_means.append(band_data.values)
            valid_bands.append(band)
            if not channel_names:
                channel_names = [col.replace(f'{power_prefix}{band}_', '') for col in band_cols]
    
    if not band_means:
        logger.warning("No valid band data for heatmap")
        return
    
    plot_cfg = get_plot_config(config)
    features_config = plot_cfg.plot_type_configs.get("features", {})
    power_config = features_config.get("power", {})
    
    heatmap_data = np.array(band_means)
    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    vmin = power_config.get("vmin")
    vmax = power_config.get("vmax")
    if vmin is None or vmax is None:
        data_min = np.nanmin(heatmap_data)
        data_max = np.nanmax(heatmap_data)
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=plot_cfg.font.small)
    ax.set_yticks(range(len(valid_bands)))
    ax.set_yticklabels([b.capitalize() for b in valid_bands], fontsize=plot_cfg.font.medium)
    ax.set_title('Mean Power per Channel and Band\nlog10(power/baseline)', fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel('Channel', fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
    
    heatmap_threshold = features_config.get("heatmap_text_threshold", 200)
    if len(channel_names) * len(valid_bands) <= heatmap_threshold:
        std_threshold = np.std(heatmap_data)
        for i in range(len(valid_bands)):
            for j in range(len(channel_names)):
                value = heatmap_data[i, j]
                text = f'{value:.2f}'
                color = 'white' if abs(value) > std_threshold else 'black'
                fontsize = max(6, min(10, heatmap_threshold/len(channel_names)))
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=fontsize)
    
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f'sub-{subject}_channel_power_heatmap',
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    logger.info("Saved channel power heatmap")


def plot_power_time_courses(tfr_raw, bands, subject, save_dir, logger, config):
    times = tfr_raw.times
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    
    for band in bands:
        if band not in features_freq_bands:
            logger.warning(f"Band '{band}' not in config; skipping time course.")
            continue
        
        fmin, fmax = features_freq_bands[band]
        freq_mask = (tfr_raw.freqs >= fmin) & (tfr_raw.freqs <= fmax)
        if not freq_mask.any():
            logger.warning(f"No frequencies found for {band} band ({fmin}-{fmax} Hz)")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        band_power_log = tfr_raw.data[:, :, freq_mask, :].mean(axis=(0, 1, 2))
        ax.plot(times, band_power_log, linewidth=2, color=_get_band_color(band, config))
        
        b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline)
        bs = max(float(times.min()), float(b_start))
        be = min(float(times.max()), float(b_end))
        if be > bs:
            ax.axvspan(bs, be, alpha=0.2, color='gray', label='Baseline')
        ax.axvspan(0, times[-1], alpha=0.2, color='orange', label='Stimulus')
        
        plot_cfg = get_plot_config(config)
        ax.set_ylabel('log10(power/baseline)', fontsize=plot_cfg.font.ylabel)
        ax.set_xlabel('Time (s)', fontsize=plot_cfg.font.label)
        ax.set_title(f'{band.capitalize()} Band Power Time Course', fontsize=plot_cfg.font.title)
        ax.grid(True, alpha=plot_cfg.style.alpha_grid)
        ax.legend(fontsize=plot_cfg.font.small)
        
        plt.tight_layout()
        output_path = save_dir / f'sub-{subject}_power_time_course_{band}'
        save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                 bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
        plt.close(fig)
        logger.info(f"Saved {band} power time course")


###################################################################
# Power Spectral Density Plotting
###################################################################

def _validate_epochs_tfr(tfr, function_name: str, logger) -> bool:
    """Validate that TFR is EpochsTFR (4D) and raise if AverageTFR (3D)."""
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        if isinstance(tfr, mne.time_frequency.AverageTFR):
            error_msg = (
                f"{function_name} requires EpochsTFR (4D: n_epochs, n_channels, n_freqs, n_times), "
                f"but received AverageTFR (3D: n_channels, n_freqs, n_times). "
                f"Cannot split by epochs/conditions with averaged data."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        else:
            error_msg = (
                f"{function_name} requires EpochsTFR, but received {type(tfr).__name__}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
    
    if len(tfr.data.shape) != 4:
        error_msg = (
            f"{function_name} requires 4D TFR data (n_epochs, n_channels, n_freqs, n_times), "
            f"but received shape {tfr.data.shape}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True


def _plot_psd_by_temperature(tfr_win, temps, subject, save_dir, logger, config):
    _validate_epochs_tfr(tfr_win, "_plot_psd_by_temperature", logger)
    
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < 2:
        return False
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    temp_colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(unique_temps)))
    
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        if temp_mask.sum() < 1:
            continue
        data_temp = tfr_win.data[temp_mask]
        psd_avg = data_temp.mean(axis=(0, 1, 3))
        if len(psd_avg) != len(tfr_win.freqs):
            logger.warning(f"Frequency dimension mismatch: {len(psd_avg)} vs {len(tfr_win.freqs)}")
            continue
        ax.plot(
            tfr_win.freqs, psd_avg,
            color=temp_colors[idx], linewidth=1.5,
            label=f'{temp:.0f}°C', alpha=0.9
        )
    
    plot_cfg = get_plot_config(config)
    ax.axhline(0, color=plot_cfg.style.colors.gray, 
               linewidth=plot_cfg.style.line.width_standard, 
               alpha=plot_cfg.style.line.alpha_dim, linestyle='--')
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Power spectral density (log10 ratio to baseline)", fontsize=plot_cfg.font.ylabel)
    ax.legend(loc='upper left', fontsize=plot_cfg.font.title, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_temperature'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved PSD by temperature")
    return True


def _plot_psd_overall(tfr, tfr_win, subject, save_dir, logger, config):
    _validate_epochs_tfr(tfr_win, "_plot_psd_overall", logger)
    
    data = tfr_win.data.mean(axis=(0, 3))
    psd_avg = data.mean(axis=0)
    psd_sem = data.std(axis=0) / np.sqrt(data.shape[0])
    
    fig, ax = plt.subplots(figsize=(4.0, 2.5), constrained_layout=True)
    ax.plot(tfr.freqs, psd_avg, color="0.2", linewidth=1.0)
    ax.fill_between(
        tfr.freqs,
        psd_avg - 1.96 * psd_sem,
        psd_avg + 1.96 * psd_sem,
        color="0.4", alpha=0.15, linewidth=0
    )
    ax.axhline(0, color="0.7", linewidth=0.5, alpha=0.6)
    
    default_freq_bands = {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    }
    freq_bands = config.get("time_frequency_analysis.bands", default_freq_bands)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    
    for band, (fmin, fmax) in features_freq_bands.items():
        if fmin < tfr.freqs.max():
            fmax_clipped = min(fmax, tfr.freqs.max())
            ax.axvspan(fmin, fmax_clipped, alpha=0.08, color="0.5", linewidth=0)
            mid = (fmin + fmax_clipped) / 2
            if mid < tfr.freqs.max():
                y_max = ax.get_ylim()[1]
                ax.text(
                    mid, y_max * 0.95, band[0].upper(),
                    fontsize=7, ha="center", va="top", color="0.4"
                )
    
    plot_cfg = get_plot_config(config)
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.medium)
    ax.set_ylabel("log10(power/baseline)", fontsize=plot_cfg.font.medium)
    ax.tick_params(labelsize=plot_cfg.font.small)
    sns.despine(ax=ax, trim=True)
    
    output_path = save_dir / f'sub-{subject}_power_spectral_density'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved PSD")


def _get_plateau_window(config) -> List[float]:
    if config:
        return config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
    return [3.0, 10.5]


def _crop_tfr_to_plateau(tfr, plateau_window: List[float], logger) -> Optional[Any]:
    times = np.asarray(tfr.times)
    plateau_start = float(plateau_window[0])
    plateau_end = float(plateau_window[1])
    tmin = max(times.min(), plateau_start)
    tmax = min(times.max(), plateau_end)
    
    if tmax <= tmin:
        logger.warning("Invalid plateau window; skipping PSD")
        return None
    
    return tfr.copy().crop(tmin, tmax)


def plot_power_spectral_density(tfr, subject, save_dir, logger, events_df=None, config=None):
    _validate_epochs_tfr(tfr, "plot_power_spectral_density", logger)
    
    plateau_window = _get_plateau_window(config)
    tfr_win = _crop_tfr_to_plateau(tfr, plateau_window, logger)
    
    if tfr_win is None:
        return
    
    if events_df is not None and not events_df.empty:
        temp_col = find_temperature_column_in_events(events_df)
        if temp_col is not None:
            temps = pd.to_numeric(events_df[temp_col], errors="coerce")
            if len(tfr_win) != len(temps):
                raise ValueError(
                    f"TFR window ({len(tfr_win)} epochs) and events "
                    f"({len(temps)} rows) length mismatch for subject {subject}"
                )
            if _plot_psd_by_temperature(tfr_win, temps, subject, save_dir, logger, config):
                return
    
    _plot_psd_overall(tfr, tfr_win, subject, save_dir, logger, config)


def plot_power_spectral_density_by_pain(tfr, subject, save_dir, logger, events_df=None, config=None):
    _validate_epochs_tfr(tfr, "plot_power_spectral_density_by_pain", logger)
    
    if events_df is None or events_df.empty:
        logger.warning("No events for PSD by pain")
        return
    
    pain_col = find_pain_column_in_events(events_df)
    if pain_col is None:
        logger.warning("No pain binary column found")
        return
    
    plateau_window = _get_plateau_window(config)
    tfr_win = _crop_tfr_to_plateau(tfr, plateau_window, logger)
    
    if tfr_win is None:
        return
    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    
    if len(tfr_win) != len(pain_vals):
        raise ValueError(
            f"TFR window ({len(tfr_win)} epochs) and events "
            f"({len(pain_vals)} rows) length mismatch for subject {subject}"
        )
    
    nonpain_mask = (pain_vals == 0).to_numpy()
    pain_mask = (pain_vals == 1).to_numpy()
    
    if nonpain_mask.sum() < 1 or pain_mask.sum() < 1:
        logger.warning("Insufficient trials for pain comparison")
        return
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    for mask, label, color in [
        (nonpain_mask, 'Non-pain', 'steelblue'),
        (pain_mask, 'Pain', 'orangered')
    ]:
        if mask.sum() < 1:
            continue
        data_cond = tfr_win.data[mask]
        psd_avg = data_cond.mean(axis=(0, 1, 3))
        if len(psd_avg) != len(tfr_win.freqs):
            logger.warning(f"Frequency dimension mismatch for {label}")
            continue
        ax.plot(
            tfr_win.freqs, psd_avg,
            color=color, linewidth=1.5, label=label, alpha=0.9
        )
    
    plot_cfg = get_plot_config(config)
    ax.axhline(0, color=plot_cfg.style.colors.gray, 
               linewidth=plot_cfg.style.line.width_standard, 
               alpha=plot_cfg.style.line.alpha_dim, linestyle='--')
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Power spectral density (log10 ratio to baseline)", fontsize=plot_cfg.font.ylabel)
    ax.legend(loc='upper left', fontsize=plot_cfg.font.title, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_pain'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved PSD by pain condition")


def _validate_temperature_data(tfr, events_df, subject, logger):
    from eeg_pipeline.utils.io_utils import find_temperature_column_in_events
    from eeg_pipeline.utils.data_loading import extract_temperature_series, compute_aligned_data_length
    
    if events_df is None or events_df.empty:
        logger.warning(f"No events for temperature time course for subject {subject}")
        return None
    
    temp_col = find_temperature_column_in_events(events_df)
    if temp_col is None:
        logger.warning(f"No temperature column found for subject {subject}")
        return None
    
    n = compute_aligned_data_length(tfr, events_df)
    temps = extract_temperature_series(tfr, events_df, temp_col, n)
    if temps is None or temps.isna().all():
        logger.warning(f"No valid temperature data for subject {subject}")
        return None
    
    return temps


def _get_band_frequency_mask(tfr, band, config, logger):
    from eeg_pipeline.utils.tfr_utils import get_bands_for_tfr, freq_mask
    
    bands = get_bands_for_tfr(tfr=tfr, config=config)
    if band not in bands:
        logger.warning(f"Band '{band}' not found in frequency bands")
        return None
    
    fmin, fmax = bands[band]
    mask = freq_mask(tfr.freqs, fmin, fmax)
    if not mask.any():
        logger.warning(f"No frequencies found for {band} band ({fmin}-{fmax} Hz)")
        return None
    
    return mask


def plot_power_time_course_by_temperature(tfr, subject, save_dir, logger, events_df=None, band='alpha', config=None):
    _validate_epochs_tfr(tfr, "plot_power_time_course_by_temperature", logger)
    
    band = str(band)
    temps = _validate_temperature_data(tfr, events_df, subject, logger)
    if temps is None:
        return
    
    freq_mask = _get_band_frequency_mask(tfr, band, config, logger)
    if freq_mask is None:
        return
    
    unique_temps = sorted(temps.dropna().unique())
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(unique_temps)))
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        if temp_mask.sum() < 1:
            continue
        
        band_power = tfr.data[temp_mask][:, :, freq_mask, :].mean(axis=(0, 1, 2))
        ax.plot(
            tfr.times, band_power,
            color=colors[idx], linewidth=1.8, alpha=0.9,
            label=f"{temp:.0f}°C"
        )
    
    plot_cfg = get_plot_config(config)
    ax.axhline(0, color=plot_cfg.style.colors.gray, 
               linewidth=plot_cfg.style.line.width_standard, 
               alpha=plot_cfg.style.line.alpha_dim, linestyle='--')
    ax.set_xlabel("Time (s)", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel(f"{band.capitalize()} power (log10 ratio)", fontsize=plot_cfg.font.ylabel)
    ax.legend(loc='upper left', fontsize=plot_cfg.font.title, frameon=False)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    output_path = save_dir / f'sub-{subject}_time_course_by_temperature_{band}'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved band time course by temperature")


__all__ = [
    # ERP
    "erp_contrast_pain",
    "erp_by_temperature",
    "group_erp_contrast_pain",
    "group_erp_by_temperature",
    # microstate
    "plot_microstate_templates",
    "plot_microstate_coverage_by_pain",
    "plot_microstate_pain_correlation_heatmap",
    "plot_microstate_temporal_evolution",
    "plot_microstate_templates_by_pain",
    "plot_microstate_templates_by_temperature",
    "plot_microstate_gfp_colored_by_state",
    "plot_microstate_gfp_by_temporal_bins",
    "plot_microstate_transition_network",
    "plot_microstate_duration_distributions",
    # power
    "plot_power_distributions",
    "plot_channel_power_heatmap",
    "plot_power_time_courses",
    "plot_power_spectral_density",
    "plot_power_spectral_density_by_pain",
    "plot_power_time_course_by_temperature",
    "plot_group_power_plots",
    "plot_group_band_power_time_courses",
]

###################################################################
# Additional power plots moved from 02_feature_extraction
###################################################################

def plot_trial_power_variability(pow_df, bands, subject, save_dir, logger, config):
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {})
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    n_bands = len(bands)
    fig, axes = plt.subplots(n_bands, 1, figsize=(12, 3 * n_bands))
    if n_bands == 1:
        axes = [axes]
    
    for i, band in enumerate(bands):
        band_str = str(band)
        band_cols = [col for col in pow_df.columns if str(col).startswith(f'{power_prefix}{band_str}_')]
        if not band_cols:
            continue
        
        band_power_trials = pow_df[band_cols].mean(axis=1)
        trial_numbers = range(1, len(band_power_trials) + 1)
        band_color = _get_band_color(band_str, config)
        axes[i].plot(
            trial_numbers, band_power_trials, 'o-',
            alpha=0.7, linewidth=1, color=band_color
        )
        
        mean_power = band_power_trials.mean()
        std_power = band_power_trials.std()
        coefficient_of_variation = (
            std_power / abs(mean_power) if abs(mean_power) > 1e-10 else np.nan
        )
        
        axes[i].axhline(
            mean_power, color='red', linestyle='--', alpha=0.8,
            label=f'Mean = {mean_power:.3f}'
        )
        axes[i].fill_between(
            trial_numbers, mean_power - std_power, mean_power + std_power,
            alpha=0.2, color='red', label=f'±1 SD = ±{std_power:.3f}'
        )
        axes[i].set_ylabel(f'{band_str.capitalize()}\nlog10(power/baseline)')
        axes[i].set_title(
            f'{band_str.capitalize()} Band Power Variability (CV = {coefficient_of_variation:.3f})'
        )
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plot_cfg = get_plot_config(config)
    axes[-1].set_xlabel('Trial Number', fontsize=plot_cfg.font.label)
    plt.tight_layout()
    save_fig(
        fig, save_dir / f'sub-{subject}_trial_power_variability',
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    logger.info("Saved trial power variability")








def plot_inter_band_spatial_power_correlation(tfr, subject, save_dir, logger, config):
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    band_names = list(features_freq_bands.keys())
    n_bands = len(band_names)
    
    times = np.asarray(tfr.times)
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
    plateau_start = float(plateau_window[0])
    plateau_end = float(plateau_window[1])
    tmin_clip = float(max(times.min(), plateau_start))
    tmax_clip = float(min(times.max(), plateau_end))
    
    if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
        logger.warning(
            f"Skipping inter-band spatial power correlation: invalid plateau within data range "
            f"(requested [{plateau_start}, {plateau_end}] s, "
            f"available [{times.min():.2f}, {times.max():.2f}] s)"
        )
        return
    
    tfr_windowed = tfr.copy().crop(tmin_clip, tmax_clip)
    tfr_avg = tfr_windowed.average()
    coupling_matrix = compute_inter_band_coupling_matrix(
        tfr_avg,
        band_names,
        features_freq_bands,
        extract_band_channel_means
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(coupling_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_bands))
    ax.set_yticks(range(n_bands))
    ax.set_xticklabels([band.capitalize() for band in band_names], rotation=45, ha='right')
    ax.set_yticklabels([band.capitalize() for band in band_names])
    
    for i in range(n_bands):
        for j in range(n_bands):
            value = coupling_matrix[i, j]
            text_color = "black" if abs(value) < 0.5 else "white"
            ax.text(
                j, i, f'{value:.2f}',
                ha="center", va="center", color=text_color
            )
    
    plot_cfg = get_plot_config(config)
    ax.set_title('Inter Band Spatial Power Correlation', fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation (r)', fontsize=plot_cfg.font.title)
    
    plt.tight_layout()
    output_path = save_dir / f'sub-{subject}_inter_band_spatial_power_correlation'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info("Saved inter-band spatial power correlation")


###################################################################
# Group-level power plotting (migrated from 02_feature_extraction)
###################################################################

def _collect_channel_names_by_band(subj_pow: dict, bands: list, config=None) -> dict:
    plot_cfg = get_plot_config(config) if config else None
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {}) if plot_cfg else {}
    power_prefix = behavioral_config.get("power_prefix", "pow_") if plot_cfg else "pow_"
    band_channels = {}
    for band in bands:
        band_str = str(band)
        channel_union = set()
        for _, df in subj_pow.items():
            cols = [c for c in df.columns if str(c).startswith(f"{power_prefix}{band_str}_")]
            channel_union.update([str(c).replace(f"{power_prefix}{band_str}_", "") for c in cols])
        band_channels[band_str] = sorted(channel_union)
    return band_channels




def _plot_group_channel_power_heatmap(heatmap_data: np.ndarray, channels: list, bands: list, output_path: Path, config):
    if heatmap_data.size == 0:
        return
    plot_cfg = get_plot_config(config)
    features_config = plot_cfg.plot_type_configs.get("features", {})
    correlation_config = features_config.get("correlation", {})
    
    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    vmin = correlation_config.get("vmin", -0.6)
    vmax = correlation_config.get("vmax", 0.6)
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=plot_cfg.font.small)
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([b.capitalize() for b in bands], fontsize=plot_cfg.font.medium)
    ax.set_title("Group Mean Power per Channel and Band\nlog10(power/baseline)", fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel("Channel", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Frequency Band", fontsize=plot_cfg.font.ylabel)
    plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
    plt.tight_layout()
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)


def _collect_band_power_records(subj_pow: dict, bands: list, config=None) -> list:
    plot_cfg = get_plot_config(config) if config else None
    behavioral_config = plot_cfg.plot_type_configs.get("behavioral", {}) if plot_cfg else {}
    power_prefix = behavioral_config.get("power_prefix", "pow_") if plot_cfg else "pow_"
    records = []
    for band in bands:
        band_str = str(band)
        for subject, df in subj_pow.items():
            cols = [c for c in df.columns if str(c).startswith(f"{power_prefix}{band_str}_")]
            if not cols:
                continue
            values = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            records.append({
                "subject": subject,
                "band": band_str,
                "mean_power": float(np.mean(values))
            })
    return records






def _plot_subject_scatter(ax, df, bands_present, config=None, rng: Optional[np.random.Generator] = None):
    if rng is None:
        if config is None:
            rng_seed = 42
        else:
            rng_seed = config.get("random.seed", 42)
        rng = np.random.default_rng(rng_seed)
    
    jitter_range = 0.2
    if config is not None:
        jitter_range = config.get("behavior_analysis.group_aggregation.jitter_range", 0.2)
    
    for i, band in enumerate(bands_present):
        values = df[df["band"] == band]["mean_power"].to_numpy(dtype=float)
        jitter = (rng.random(len(values)) - 0.5) * jitter_range
        ax.scatter(
            np.full_like(values, i, dtype=float) + jitter,
            values, color='k', s=12, alpha=0.6
        )


def _plot_group_band_power_summary(
    bands_present: list, means: list, ci_lower: list, ci_upper: list, n_subjects: list,
    df: pd.DataFrame, output_path: Path, stats_path: Path, config, logger
):
    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(bands_present))
    
    ax.bar(x_positions, means, color='steelblue', alpha=0.8)
    yerr = compute_error_bars_from_arrays(means, ci_lower, ci_upper)
    ax.errorbar(x_positions, means, yerr=yerr, fmt='none', ecolor='k', capsize=3)
    _plot_subject_scatter(ax, df, bands_present, config=config)
    
    plot_cfg = get_plot_config(config)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([b.capitalize() for b in bands_present], fontsize=plot_cfg.font.medium)
    ax.set_ylabel('Mean log10(power/baseline) across subjects', fontsize=plot_cfg.font.ylabel)
    ax.set_title('Group Band Power Summary (subject means, 95% CI)', fontsize=plot_cfg.font.figure_title)
    ax.axhline(0, color=plot_cfg.style.colors.black, linewidth=plot_cfg.style.line.width_standard)
    
    plt.tight_layout()
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    output_df = pd.DataFrame({
        "band": bands_present,
        "group_mean": means,
        "ci_low": ci_lower,
        "ci_high": ci_upper,
        "n_subjects": n_subjects
    })
    output_df.to_csv(stats_path, sep="\t", index=False)
    logger.info("Saved group band power distributions and stats.")










def _plot_group_inter_band_correlation(
    group_correlation: np.ndarray, band_names: list, output_path: Path, config
):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(group_correlation, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(band_names)))
    ax.set_yticks(range(len(band_names)))
    ax.set_xticklabels([b.capitalize() for b in band_names], rotation=45, ha='right')
    ax.set_yticklabels([b.capitalize() for b in band_names])
    for i in range(len(band_names)):
        for j in range(len(band_names)):
            if np.isfinite(group_correlation[i, j]):
                value = group_correlation[i, j]
                text_color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f"{value:.2f}", ha='center', va='center', color=text_color)
    plot_cfg = get_plot_config(config)
    ax.set_title('Group Inter Band Spatial Power Correlation', fontsize=plot_cfg.font.figure_title)
    ax.set_xlabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel('Frequency Band', fontsize=plot_cfg.font.ylabel)
    cbar = plt.colorbar(im, ax=ax, label='Correlation (r)')
    cbar.set_label('Correlation (r)', fontsize=plot_cfg.font.title)
    plt.tight_layout()
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)




def plot_group_power_plots(subj_pow: dict, bands: list, gplots, gstats, config, logger) -> None:
    band_channels = _collect_channel_names_by_band(subj_pow, bands, config)
    all_channels = sorted(set().union(*band_channels.values())) if band_channels else []
    
    heatmap_rows, statistics_rows = compute_group_channel_power_statistics(
        subj_pow, bands, all_channels
    )
    heatmap_data = np.vstack(heatmap_rows) if heatmap_rows else np.zeros((0, 0))
    
    _plot_group_channel_power_heatmap(
        heatmap_data, all_channels, bands,
        gplots / "group_channel_power_heatmap", config
    )
    pd.DataFrame(statistics_rows).to_csv(
        gstats / "group_channel_power_means.tsv", sep="\t", index=False
    )
    logger.info("Saved group channel power heatmap and stats.")
    
    records = _collect_band_power_records(subj_pow, bands, config)
    df_records = pd.DataFrame(records)
    if not df_records.empty:
        bands_present, means, ci_lower, ci_upper, n_subjects = compute_group_band_statistics(
            df_records, bands
        )
        _plot_group_band_power_summary(
            bands_present, means, ci_lower, ci_upper, n_subjects, df_records,
            gplots / "group_power_distributions_per_band_across_subjects",
            gstats / "group_band_power_subject_means.tsv",
            config, logger
        )
    
    default_freq_bands = {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    }
    freq_bands = config.get("time_frequency_analysis.bands", default_freq_bands)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    band_names = list(features_freq_bands.keys())
    n_bands = len(band_names)
    per_subject_correlations = []
    
    for _, df in subj_pow.items():
        band_vectors = extract_band_channel_vectors(df, band_names)
        if len(band_vectors) < 2:
            continue
        correlation_matrix = compute_subject_band_correlation_matrix(band_vectors, band_names)
        per_subject_correlations.append(correlation_matrix)
    
    if len(per_subject_correlations) >= 2:
        group_correlation = compute_group_band_correlation_matrix(
            per_subject_correlations, n_bands
        )
        _plot_group_inter_band_correlation(
            group_correlation, band_names,
            gplots / "group_inter_band_spatial_power_correlation", config
        )
        correlation_statistics = compute_inter_band_correlation_statistics(
            per_subject_correlations, band_names
        )
        if correlation_statistics:
            pd.DataFrame(correlation_statistics).to_csv(
                gstats / "group_inter_band_correlation.tsv", sep="\t", index=False
            )
            logger.info("Saved group inter-band correlation heatmap and stats.")


def _add_baseline_region_to_axis(ax, times: np.ndarray, config):
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    min_baseline_samples = int(config.get("tfr_topography_pipeline.min_baseline_samples", 5))
    b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline, min_samples=min_baseline_samples)
    baseline_start = max(float(times.min()), float(b_start))
    baseline_end = min(float(times.max()), float(b_end))
    if baseline_end > baseline_start:
        ax.axvspan(baseline_start, baseline_end, alpha=0.1, color='gray')
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)


def _plot_group_band_time_course(
    valid_bands: list, band_data: dict, times: np.ndarray,
    ylabel: str, title: str, output_path: Path, config, logger
):
    nrows = len(valid_bands)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    
    for i, band in enumerate(valid_bands):
        ax = axes[i]
        series_list = band_data.get(band, [])
        if len(series_list) < 2:
            continue
        array = np.vstack(series_list)
        mean_values = np.nanmean(array, axis=0)
        se = np.nanstd(array, axis=0, ddof=1) / np.sqrt(array.shape[0])
        ci = 1.96 * se
        band_color = _get_band_color(band, config)
        ax.plot(times, mean_values, color=band_color, label=str(band))
        ax.fill_between(times, mean_values - ci, mean_values + ci, color=band_color, alpha=0.2)
        _add_baseline_region_to_axis(ax, times, config)
        ax.set_title(f"{band.capitalize()} (group mean ±95% CI)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    plot_cfg = get_plot_config(config)
    axes[-1].set_xlabel("Time (s)", fontsize=plot_cfg.font.label)
    fig.suptitle(title, fontsize=plot_cfg.font.figure_title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    logger.info(f"Saved {title.lower()}")


def plot_group_band_power_time_courses(valid_bands: list, band_tc: dict, band_tc_pct: dict, tref: np.ndarray, gplots, config, logger) -> None:
    _plot_group_band_time_course(
        valid_bands, band_tc, tref,
        "log10(power/baseline)", "Group Band Power Time Courses",
        gplots / "group_band_power_time_courses", config, logger
    )
    _plot_group_band_time_course(
        valid_bands, band_tc_pct, tref,
        "Percent change from baseline (%)",
        "Group Band Power Time Courses (percent change, ratio-domain averaging)",
        gplots / "group_band_power_time_courses_percent_change", config, logger
    )


