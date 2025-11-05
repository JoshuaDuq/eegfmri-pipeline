from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
)
from eeg_pipeline.utils.io_utils import save_fig, ensure_dir
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

def _constants_from_config(config):
    return {
        "FIG_DPI": int(config.get("output.fig_dpi", 300)) if config else 300,
        "SAVE_FORMATS": list(config.get("output.save_formats", ["png"])) if config else ["png"],
        "output.bbox_inches": config.get("output.bbox_inches", "tight") if config else "tight",
        "output.pad_inches": float(config.get("output.pad_inches", 0.02)) if config else 0.02,
    }


###################################################################
# ERP plotting helpers
###################################################################

def _apply_baseline_local(epochs: mne.Epochs, baseline_window: Tuple[float, float]) -> mne.Epochs:
    baseline_start = float(baseline_window[0])
    baseline_end = float(baseline_window[1])
    return epochs.copy().apply_baseline((baseline_start, min(baseline_end, 0.0)))




def _build_epoch_query_and_label_local(column: str, level: any, is_numeric: bool, labels: Dict) -> Tuple[str, str]:
    if is_numeric:
        return f"{column} == {level}", labels[level]
    escaped_level = str(level).replace('"', '\\"')
    query = f'{column} == "{escaped_level}"'
    return query, str(level)


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
    logger: Optional[object] = None
) -> None:
    if epochs.metadata is None:
        if logger:
            logger.warning("ERP pain contrast: epochs.metadata is missing.")
        return
    pain_columns = config.get("event_columns.pain_binary", []) if config else []
    pain_column = next((col for col in pain_columns if col in epochs.metadata.columns), None) if pain_columns else None
    if pain_column is None:
        if logger:
            logger.warning("ERP pain contrast: No pain column found in metadata.")
        return

    epochs_baselined = _apply_baseline_local(epochs, baseline_window)
    epochs_pain = select_epochs_by_value(epochs_baselined, pain_column, 1)
    epochs_nonpain = select_epochs_by_value(epochs_baselined, pain_column, 0)
    if len(epochs_pain) == 0 or len(epochs_nonpain) == 0:
        if logger:
            logger.warning("ERP pain contrast: one of the groups has zero trials.")
        return

    evoked_pain = epochs_pain.average(picks=erp_picks)
    evoked_nonpain = epochs_nonpain.average(picks=erp_picks)
    colors = {"painful": pain_color, "non-painful": nonpain_color}
    plot_configs = [
        (erp_combine, erp_output_files.get("pain_gfp", "erp_pain_binary_gfp.png")),
        (None, erp_output_files.get("pain_butterfly", "erp_pain_binary_butterfly.png")),
    ]
    baseline_str = f"[{baseline_window[0]:.2f}, {baseline_window[1]:.2f}]"
    for combine_method, output_name in plot_configs:
        figure = mne.viz.plot_compare_evokeds(
            {"painful": evoked_pain, "non-painful": evoked_nonpain},
            picks=erp_picks,
            combine=combine_method,
            show=False,
            colors=colors,
        )
        footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
        ensure_dir((output_dir / output_name).parent)
        save_fig(_unwrap_figure(figure), output_dir / output_name, logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})


def erp_by_temperature(
    epochs: mne.Epochs,
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None
) -> None:
    if epochs.metadata is None:
        if logger:
            logger.warning("ERP by temperature: epochs.metadata is missing.")
        return
    temperature_columns = config.get("event_columns.temperature", []) if config else []
    temperature_column = next((col for col in temperature_columns if col in epochs.metadata.columns), None) if temperature_columns else None
    if temperature_column is None:
        if logger:
            logger.warning("ERP by temperature: No temperature column found in metadata.")
        return

    epochs_baselined = _apply_baseline_local(epochs, baseline_window)
    temperature_levels, temperature_labels, is_numeric = process_temperature_levels(epochs_baselined, temperature_column)
    evokeds_by_temperature: Dict[str, mne.Evoked] = {}
    for level in temperature_levels:
        query, label = _build_epoch_query_and_label_local(temperature_column, level, is_numeric, temperature_labels)
        epochs_at_level = epochs_baselined[query]
        if len(epochs_at_level) > 0:
            evokeds_by_temperature[label] = epochs_at_level.average(picks=erp_picks)
    if not evokeds_by_temperature:
        if logger:
            logger.warning("ERP by temperature: No evokeds computed.")
        return
    baseline_str = f"[{baseline_window[0]:.2f}, {baseline_window[1]:.2f}]"
    for label, evoked in evokeds_by_temperature.items():
        figure = evoked.plot(picks=erp_picks, spatial_colors=True, show=False)
        figure.suptitle(f"ERP - Temperature {label}")
        safe_label = _sanitize_label(label)
        output_name = erp_output_files.get("temp_butterfly_template", "erp_by_temperature_butterfly_{label}.png").format(label=safe_label)
        footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
        ensure_dir((output_dir / output_name).parent)
        save_fig(figure, output_dir / output_name, logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})
    if len(evokeds_by_temperature) >= 2:
        figure = mne.viz.plot_compare_evokeds(evokeds_by_temperature, picks=erp_picks, combine=None, show=False)
        footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
        ensure_dir((output_dir / erp_output_files.get("temp_butterfly", "erp_by_temperature_butterfly.png")).parent)
        save_fig(_unwrap_figure(figure), output_dir / erp_output_files.get("temp_butterfly", "erp_by_temperature_butterfly.png"), logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})
        figure = mne.viz.plot_compare_evokeds(evokeds_by_temperature, picks=erp_picks, combine=erp_combine, show=False)
        footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
        ensure_dir((output_dir / erp_output_files.get("temp_gfp", "erp_by_temperature_gfp.png")).parent)
        save_fig(_unwrap_figure(figure), output_dir / erp_output_files.get("temp_gfp", "erp_by_temperature_gfp.png"), logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})


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
        if epochs.metadata is None:
            continue
        pain_columns = config.get("event_columns.pain_binary", []) if config else []
        pain_column = next((col for col in pain_columns if col in epochs.metadata.columns), None) if pain_columns else None
        if pain_column is None:
            continue
        epochs_pain = select_epochs_by_value(epochs, pain_column, 1)
        epochs_nonpain = select_epochs_by_value(epochs, pain_column, 0)
        if len(epochs_pain) > 0:
            pain_evokeds.append(epochs_pain.average(picks=erp_picks))
        if len(epochs_nonpain) > 0:
            nonpain_evokeds.append(epochs_nonpain.average(picks=erp_picks))
    if not pain_evokeds or not nonpain_evokeds:
        if logger:
            logger.warning("Group ERP pain contrast: insufficient data across subjects")
        return
    grand_average_pain = mne.grand_average(pain_evokeds, interpolate_bads=True)
    grand_average_nonpain = mne.grand_average(nonpain_evokeds, interpolate_bads=True)
    colors = {"painful": pain_color, "non-painful": nonpain_color}
    plot_configs = [
        (erp_combine, erp_output_files.get("pain_gfp", "erp_pain_binary_gfp.png")),
        (None, erp_output_files.get("pain_butterfly", "erp_pain_binary_butterfly.png")),
    ]
    baseline_str = f"[{baseline_window[0]:.2f}, {baseline_window[1]:.2f}]"
    for combine_method, output_name in plot_configs:
        figure = mne.viz.plot_compare_evokeds(
            {"painful": grand_average_pain, "non-painful": grand_average_nonpain},
            picks=erp_picks,
            combine=combine_method,
            show=False,
            colors=colors,
        )
        unwrapped_figure = _unwrap_figure(figure)
        unwrapped_figure.suptitle(
            f"Group ERP: Pain vs Non-Pain (N={len(pain_evokeds)} subjects)", fontsize=14, fontweight='bold'
        )
        footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
        ensure_dir((output_dir / ("group_" + output_name)).parent)
        save_fig(unwrapped_figure, output_dir / ("group_" + output_name), logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})


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
    evokeds_by_temperature: Dict[str, List[mne.Evoked]] = {}
    for epochs in all_epochs:
        if epochs.metadata is None:
            continue
        temperature_columns = config.get("event_columns.temperature", []) if config else []
        temperature_column = next((col for col in temperature_columns if col in epochs.metadata.columns), None) if temperature_columns else None
        if temperature_column is None:
            continue
        temperature_levels, temperature_labels, is_numeric = process_temperature_levels(epochs, temperature_column)
        for level in temperature_levels:
            query, label = _build_epoch_query_and_label_local(temperature_column, level, is_numeric, temperature_labels)
            epochs_at_level = epochs[query]
            if len(epochs_at_level) > 0:
                evokeds_by_temperature.setdefault(label, []).append(epochs_at_level.average(picks=erp_picks))
    if not evokeds_by_temperature:
        if logger:
            logger.warning("Group ERP by temperature: No evokeds computed across subjects")
        return
    grand_averages = {label: mne.grand_average(evokeds, interpolate_bads=True) for label, evokeds in evokeds_by_temperature.items() if evokeds}
    baseline_str = f"[{baseline_window[0]:.2f}, {baseline_window[1]:.2f}]"
    for label, evoked in grand_averages.items():
        figure = evoked.plot(picks=erp_picks, spatial_colors=True, show=False)
        figure.suptitle(f"Group ERP - Temperature {label}")
        safe_label = _sanitize_label(label)
        output_name = "group_" + erp_output_files.get("temp_butterfly_template", "erp_by_temperature_butterfly_{label}.png").format(label=safe_label)
        footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
        ensure_dir((output_dir / output_name).parent)
        save_fig(figure, output_dir / output_name, logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})
    if len(grand_averages) >= 2:
        figure = mne.viz.plot_compare_evokeds(grand_averages, picks=erp_picks, combine=None, show=False)
        unwrapped_figure = _unwrap_figure(figure)
        subject_counts = ", ".join([f"{label}: N={len(evokeds_by_temperature[label])}" for label in grand_averages.keys()])
        unwrapped_figure.suptitle(f"Group ERP by Temperature (Butterfly) — {subject_counts}", fontsize=14, fontweight='bold')
        footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
        ensure_dir((output_dir / ("group_" + erp_output_files.get("temp_butterfly", "erp_by_temperature_butterfly.png"))).parent)
        save_fig(unwrapped_figure, output_dir / ("group_" + erp_output_files.get("temp_butterfly", "erp_by_temperature_butterfly.png")), logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})
    figure = mne.viz.plot_compare_evokeds(grand_averages, picks=erp_picks, combine=erp_combine, show=False)
    unwrapped_figure = _unwrap_figure(figure)
    subject_info = ", ".join([f"{label}: N={len(evokeds)}" for label, evokeds in evokeds_by_temperature.items()])
    unwrapped_figure.suptitle(f"Group ERP by Temperature ({subject_info} subjects)", fontsize=14, fontweight='bold')
    footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=erp_combine)
    ensure_dir((output_dir / ("group_" + erp_output_files.get("temp_gfp", "erp_by_temperature_gfp.png"))).parent)
    save_fig(unwrapped_figure, output_dir / ("group_" + erp_output_files.get("temp_gfp", "erp_by_temperature_gfp.png")), logger=logger, footer=footer_text, tight_layout_rect=[0, 0.03, 1, 1], constants={"FIG_DPI": config.get("output.fig_dpi", 300) if config else 300})


###################################################################
# Microstate plotting
###################################################################

def plot_microstate_templates(templates, info, subject, save_dir, n_states, logger, config):
    if templates is None or len(templates) == 0:
        logger.warning("No templates to plot")
        return
    n_cols = min(4, n_states)
    n_rows = int(np.ceil(n_states / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.2 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    state_letters = [chr(65 + i) for i in range(n_states)]
    for i in range(n_states):
        mne.viz.plot_topomap(templates[i], info, axes=axes[i], show=False, contours=6, cmap="RdBu_r")
        axes[i].set_title(f"State {state_letters[i]}", fontsize=10)
    for j in range(n_states, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"Microstate Templates (K={n_states})", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, save_dir / f"sub-{subject}_microstate_templates", constants=_constants_from_config(config))
    plt.close(fig)
    logger.info("Saved microstate templates")


def plot_microstate_coverage_by_pain(ms_df, events_df, subject, save_dir, n_states, logger, config):
    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        logger.warning("Missing data for coverage by pain plot")
        return
    pain_col, _, _ = resolve_columns(events_df, config=config)
    if pain_col is None:
        logger.warning("No pain binary column found")
        return
    if len(ms_df) != len(events_df):
        raise ValueError(
            f"Microstate dataframe ({len(ms_df)} rows) and events ({len(events_df)} rows) length mismatch for subject {subject}"
        )
    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    valid_mask = pain_vals.notna()
    state_letters = [chr(65 + i) for i in range(n_states)]
    means_nonpain, means_pain, sems_nonpain, sems_pain = [], [], [], []
    for i in range(n_states):
        col = f"ms_coverage_{i}"
        if col not in ms_df.columns:
            means_nonpain.append(0); means_pain.append(0); sems_nonpain.append(0); sems_pain.append(0)
            continue
        cov_vals = pd.to_numeric(ms_df[col], errors="coerce")
        nonpain_mask = valid_mask & (pain_vals == 0)
        pain_mask = valid_mask & (pain_vals == 1)
        nonpain_data = cov_vals[nonpain_mask].to_numpy(); nonpain_data = nonpain_data[np.isfinite(nonpain_data)]
        pain_data = cov_vals[pain_mask].to_numpy(); pain_data = pain_data[np.isfinite(pain_data)]
        means_nonpain.append(np.mean(nonpain_data) if len(nonpain_data) > 0 else 0)
        means_pain.append(np.mean(pain_data) if len(pain_data) > 0 else 0)
        sems_nonpain.append(np.std(nonpain_data) / np.sqrt(len(nonpain_data)) if len(nonpain_data) > 1 else 0)
        sems_pain.append(np.std(pain_data) / np.sqrt(len(pain_data)) if len(pain_data) > 1 else 0)
    x_pos = np.arange(n_states); width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos - width/2, means_nonpain, width, yerr=sems_nonpain, label='Non-pain', color='steelblue', alpha=0.8, capsize=4)
    ax.bar(x_pos + width/2, means_pain, width, yerr=sems_pain, label='Pain', color='orangered', alpha=0.8, capsize=4)
    ax.set_xticks(x_pos); ax.set_xticklabels(state_letters)
    ax.set_xlabel("Microstate", fontsize=11); ax.set_ylabel("Coverage (fraction of time)", fontsize=11)
    ax.set_title("Microstate Coverage by Pain Condition", fontsize=12); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); save_fig(fig, save_dir / f"sub-{subject}_microstate_coverage_by_pain", constants=_constants_from_config(config)); plt.close(fig)
    logger.info("Saved microstate coverage by pain condition")


def plot_microstate_pain_correlation_heatmap(corr_df, pval_df, subject, save_dir, logger, config):
    if corr_df is None or corr_df.empty:
        logger.warning("No microstate correlation data provided; skipping heatmap")
        return

    metric_labels = list(corr_df.index)
    state_labels = list(corr_df.columns)
    n_states = len(state_labels)
    corr_matrix = corr_df.to_numpy(dtype=float)
    p_matrix = (
        pval_df.reindex(index=metric_labels, columns=state_labels).to_numpy(dtype=float)
        if pval_df is not None and not pval_df.empty
        else np.full_like(corr_matrix, np.nan)
    )

    fig, ax = plt.subplots(
        figsize=(max(6, n_states * 1.2), max(5, len(metric_labels) * 1.0))
    )
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(state_labels)
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Microstate", fontsize=11)
    ax.set_ylabel("Metric", fontsize=11)
    ax.set_title("Microstate-Pain Rating Correlations (Spearman r)", fontsize=12)

    for i, metric in enumerate(metric_labels):
        for j, state in enumerate(state_labels):
            value = corr_matrix[i, j]
            if np.isfinite(value):
                text_color = "white" if abs(value) > 0.4 else "black"
                text = f"{value:.2f}"
                p_val = p_matrix[i, j]
                if np.isfinite(p_val) and p_val < 0.05:
                    text += "*"
                if np.isfinite(p_val) and p_val < 0.01:
                    text += "*"
                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spearman r", fontsize=10)
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_pain_correlation",
        constants=_constants_from_config(config),
    )
    plt.close(fig)
    logger.info("Saved microstate-pain correlation heatmap")



def plot_microstate_temporal_evolution(epochs, templates, events_df, subject, task, save_dir, n_states, logger, config):
    if templates is None or events_df is None or events_df.empty:
        logger.warning("Missing data for temporal evolution plot"); return
    aligned_events = get_aligned_events(epochs, subject, task, strict=True, config=config, logger=logger)
    if aligned_events is None:
        logger.error("Alignment failed for plotting function"); return
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        logger.warning("No pain binary column found"); return
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0: return
    X = epochs.get_data()[:, picks, :]; times = epochs.times
    pain_vals = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    nonpain_mask = (pain_vals == 0).to_numpy(); pain_mask = (pain_vals == 1).to_numpy()
    state_probs_nonpain = np.zeros((n_states, len(times))); state_probs_pain = np.zeros((n_states, len(times)))
    for trial_idx in range(len(X)):
        ep = X[trial_idx]; labels, _ = _label_timecourse(ep, templates)
        if nonpain_mask[trial_idx]:
            for t_idx in range(len(times)): state_probs_nonpain[labels[t_idx], t_idx] += 1
        elif pain_mask[trial_idx]:
            for t_idx in range(len(times)): state_probs_pain[labels[t_idx], t_idx] += 1
    state_probs_nonpain /= max(1, nonpain_mask.sum()); state_probs_pain /= max(1, pain_mask.sum())
    state_letters = [chr(65 + i) for i in range(n_states)]; colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i in range(n_states):
        axes[0].plot(times, state_probs_nonpain[i], label=f"State {state_letters[i]}", color=colors[i], linewidth=1.5)
        axes[1].plot(times, state_probs_pain[i], label=f"State {state_letters[i]}", color=colors[i], linewidth=1.5)
    axes[0].axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5); axes[1].axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_ylabel("Probability", fontsize=10); axes[0].set_title("Non-pain Trials", fontsize=11); axes[0].legend(loc='upper right', fontsize=8, ncol=n_states); axes[0].grid(True, alpha=0.3); axes[0].set_ylim([0, 1])
    axes[1].set_ylabel("Probability", fontsize=10); axes[1].set_xlabel("Time (s)", fontsize=10); axes[1].set_title("Pain Trials", fontsize=11); axes[1].legend(loc='upper right', fontsize=8, ncol=n_states); axes[1].grid(True, alpha=0.3); axes[1].set_ylim([0, 1])
    plt.suptitle("Temporal Evolution of Microstate Probabilities", fontsize=12); plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, save_dir / f"sub-{subject}_microstate_temporal_evolution", constants=_constants_from_config(config)); plt.close(fig); logger.info("Saved microstate temporal evolution")


def plot_microstate_templates_by_pain(epochs, events_df, subject, task, save_dir, n_states, logger, config):
    if events_df is None or events_df.empty:
        logger.warning("Missing events for pain-specific templates"); return
    aligned_events = get_aligned_events(epochs, subject, task, strict=True, config=config, logger=logger)
    if aligned_events is None:
        logger.error("Alignment failed for plotting function"); return
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        logger.warning("No pain binary column found"); return
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels for pain-specific templates"); return
    pain_vals = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    nonpain_mask = (pain_vals == 0).to_numpy(); pain_mask = (pain_vals == 1).to_numpy()
    if nonpain_mask.sum() < 5 or pain_mask.sum() < 5:
        logger.warning("Insufficient trials for pain-specific templates"); return
    X = epochs.get_data()[:, picks, :]; sfreq = float(epochs.info["sfreq"])\
    
    templates_nonpain = _extract_templates_from_trials(X[nonpain_mask], sfreq, n_states, config)
    templates_pain = _extract_templates_from_trials(X[pain_mask], sfreq, n_states, config)
    if templates_nonpain is None or templates_pain is None:
        logger.warning("Could not compute pain-specific templates"); return
    info_eeg = mne.pick_info(epochs.info, picks); state_letters = [chr(65 + i) for i in range(n_states)]
    fig, axes = plt.subplots(2, n_states, figsize=(3.6 * n_states, 7))
    if n_states == 1: axes = axes.reshape(2, 1)
    for i in range(n_states):
        mne.viz.plot_topomap(templates_nonpain[i], info_eeg, axes=axes[0, i], show=False, contours=6, cmap="RdBu_r"); axes[0, i].set_title(f"State {state_letters[i]}", fontsize=10)
        mne.viz.plot_topomap(templates_pain[i], info_eeg, axes=axes[1, i], show=False, contours=6, cmap="RdBu_r")
    axes[0, 0].text(-0.3, 0.5, "Non-pain", transform=axes[0, 0].transAxes, fontsize=11, rotation=90, va='center', weight='bold')
    axes[1, 0].text(-0.3, 0.5, "Pain", transform=axes[1, 0].transAxes, fontsize=11, rotation=90, va='center', weight='bold')
    plt.suptitle(f"Microstate Templates by Pain Condition (K={n_states})", fontsize=12); plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, save_dir / f"sub-{subject}_microstate_templates_by_pain", constants=_constants_from_config(config)); plt.close(fig); logger.info("Saved microstate templates by pain condition")


def plot_microstate_templates_by_temperature(epochs, events_df, subject, task, save_dir, n_states, logger, config):
    if events_df is None or events_df.empty:
        logger.warning("Missing events for temperature-specific templates"); return
    aligned_events = get_aligned_events(epochs, subject, task, strict=True, config=config, logger=logger)
    if aligned_events is None:
        logger.error("Alignment failed for plotting function"); return
    temp_col = None
    for c in ["stimulus_temp", "temperature", "thermode_temperature"]:
        if c in aligned_events.columns: temp_col = c; break
    if temp_col is None:
        logger.warning("No temperature column found"); return
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels for temperature-specific templates"); return
    temps = pd.to_numeric(aligned_events[temp_col], errors="coerce")
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < 2:
        logger.warning("Insufficient temperature levels for comparison"); return
    X = epochs.get_data()[:, picks, :]; sfreq = float(epochs.info["sfreq"])
    templates_by_temp = {}
    for temp in unique_temps:
        temp_mask = (temps == temp).to_numpy()
        if temp_mask.sum() < 5: continue
        templates = _extract_templates_from_trials(X[temp_mask], sfreq, n_states, config)
        if templates is not None: templates_by_temp[temp] = templates
    if len(templates_by_temp) < 2:
        logger.warning("Could not compute templates for multiple temperatures"); return
    info_eeg = mne.pick_info(epochs.info, picks); state_letters = [chr(65 + i) for i in range(n_states)]
    sorted_temps = sorted(templates_by_temp.keys()); n_temps = len(sorted_temps)
    fig, axes = plt.subplots(n_temps, n_states, figsize=(3.6 * n_states, 3.2 * n_temps))
    if n_temps == 1 or n_states == 1: axes = axes.reshape(n_temps, n_states)
    for row_idx, temp in enumerate(sorted_temps):
        templates = templates_by_temp[temp]
        for col_idx in range(n_states):
            mne.viz.plot_topomap(templates[col_idx], info_eeg, axes=axes[row_idx, col_idx], show=False, contours=6, cmap="RdBu_r")
            if row_idx == 0: axes[row_idx, col_idx].set_title(f"State {state_letters[col_idx]}", fontsize=10)
        axes[row_idx, 0].text(-0.35, 0.5, f"{temp:.1f}°C", transform=axes[row_idx, 0].transAxes, fontsize=11, rotation=90, va='center', weight='bold')
    plt.suptitle(f"Microstate Templates by Temperature (K={n_states})", fontsize=12); plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, save_dir / f"sub-{subject}_microstate_templates_by_temperature", constants=_constants_from_config(config)); plt.close(fig); logger.info("Saved microstate templates by temperature")


def plot_microstate_gfp_colored_by_state(epochs, templates, events_df, subject, save_dir, n_states, logger, config):
    if templates is None or events_df is None or events_df.empty:
        logger.warning("Missing data for GFP microstate plot"); return
    pain_col, _, _ = resolve_columns(events_df, config=config)
    if pain_col is None:
        logger.warning("No pain binary column found"); return
    aligned_events = align_events_with_policy(events_df, epochs, config=config, logger=logger)
    if aligned_events is None:
        logger.error("Alignment failed for plotting function: aligned_events is None"); return
    if len(aligned_events) != len(epochs):
        logger.error(f"Alignment failed: events ({len(aligned_events)}) != epochs ({len(epochs)})"); return
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0: return
    pain_vals = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    X = epochs.get_data()[:, picks, :]; times = epochs.times
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5]); plateau_end = float(plateau_window[1])
    stim_mask = (times >= 0.0) & (times <= plateau_end)
    if not stim_mask.any():
        logger.warning("No timepoints in stimulus window"); return
    times_stim = times[stim_mask]
    nonpain_mask = (pain_vals == 0).to_numpy(); pain_mask = (pain_vals == 1).to_numpy()
    if nonpain_mask.sum() < 1 or pain_mask.sum() < 1:
        logger.warning("Insufficient trials for pain comparison"); return
    info_eeg = mne.pick_info(epochs.info, picks); state_letters = [chr(65 + i) for i in range(n_states)]; colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    for mask, cond_label in [(nonpain_mask, "nonpain"), (pain_mask, "pain")]:
        if mask.sum() < 1: continue
        X_cond = X[mask]; gfp_all = []; labels_all = []
        for ep in X_cond:
            gfp = _compute_gfp(ep); labels, _ = _label_timecourse(ep, templates)
            gfp_all.append(gfp[stim_mask]); labels_all.append(labels[stim_mask])
        gfp_mean = np.mean(gfp_all, axis=0) * 1e6
        labels_consensus = np.array([np.bincount([labels_all[trial][t] for trial in range(len(labels_all))]).argmax() for t in range(len(times_stim))])
        fig = plt.figure(figsize=(14, 6)); gs = fig.add_gridspec(2, n_states, height_ratios=[1, 1.5], hspace=0.15, wspace=0.3)
        for i in range(n_states):
            ax_topo = fig.add_subplot(gs[0, i])
            mne.viz.plot_topomap(templates[i], info_eeg, axes=ax_topo, show=False, contours=6, cmap="RdBu_r")
            ax_topo.set_title(f"State {state_letters[i]}", fontsize=11, weight='bold', color=colors[i])
        ax_gfp = fig.add_subplot(gs[1, :])
        for t_idx in range(len(times_stim) - 1):
            state = labels_consensus[t_idx]
            ax_gfp.fill_between([times_stim[t_idx], times_stim[t_idx + 1]], 0, gfp_mean[t_idx], color=colors[state], alpha=0.6, linewidth=0)
        ax_gfp.plot(times_stim, gfp_mean, 'k-', linewidth=1.5, alpha=0.8)
        ax_gfp.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5]); plateau_start = float(plateau_window[0])
        ax_gfp.axvline(plateau_start, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax_gfp.set_xlabel("Time (s)", fontsize=11); ax_gfp.set_ylabel("GFP (μV)", fontsize=11); ax_gfp.set_xlim([times_stim[0], times_stim[-1]]); ax_gfp.grid(True, alpha=0.3, axis='y')
        cond_title = "Non-pain" if cond_label == "nonpain" else "Pain"
        fig.suptitle(f"Microstate Sequence - {cond_title} Trials (Stimulus Period)", fontsize=13, weight='bold')
        plt.tight_layout(rect=[0, 0.02, 1, 0.96]); save_fig(fig, save_dir / f"sub-{subject}_microstate_gfp_sequence_{cond_label}", constants=_constants_from_config(config)); plt.close(fig)
    logger.info("Saved microstate GFP sequence plots")


def plot_microstate_gfp_by_temporal_bins(epochs, templates, events_df, subject, task, save_dir, n_states, logger, config):
    if templates is None or events_df is None or events_df.empty:
        logger.warning("Missing data for temporal bin GFP plots"); return
    aligned_events = get_aligned_events(epochs, subject, task, strict=True, config=config, logger=logger)
    if aligned_events is None:
        logger.error("Alignment failed for plotting function"); return
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        logger.warning("No pain binary column found"); return
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0: return
    pain_vals = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    X = epochs.get_data()[:, picks, :]; times = epochs.times
    nonpain_mask = (pain_vals == 0).to_numpy(); pain_mask = (pain_vals == 1).to_numpy()
    if nonpain_mask.sum() < 1 or pain_mask.sum() < 1:
        logger.warning("Insufficient trials for pain comparison"); return
    info_eeg = mne.pick_info(epochs.info, picks); state_letters = [chr(65 + i) for i in range(n_states)]; colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    bins = config.get("feature_engineering.features.temporal_bins", [])
    for bin_config in bins:
        if isinstance(bin_config, dict): t_start = float(bin_config.get("start", 0.0)); t_end = float(bin_config.get("end", 0.0)); t_label = str(bin_config.get("label", "unknown"))
        elif isinstance(bin_config, (list, tuple)) and len(bin_config) >= 3: t_start = float(bin_config[0]); t_end = float(bin_config[1]); t_label = str(bin_config[2])
        else: logger.warning(f"Invalid temporal bin configuration: {bin_config}; skipping"); continue
        bin_mask = (times >= t_start) & (times <= t_end)
        if not bin_mask.any(): logger.warning(f"No timepoints in {t_label} bin"); continue
        times_bin = times[bin_mask]
        for mask, cond_label in [(nonpain_mask, "nonpain"), (pain_mask, "pain")]:
            if mask.sum() < 1: continue
            X_cond = X[mask]; gfp_all = []; labels_all = []
            for ep in X_cond:
                gfp = _compute_gfp(ep); labels, _ = _label_timecourse(ep, templates)
                gfp_all.append(gfp[bin_mask]); labels_all.append(labels[bin_mask])
            gfp_mean = np.mean(gfp_all, axis=0) * 1e6
            labels_consensus = np.array([np.bincount([labels_all[trial][t] for trial in range(len(labels_all))]).argmax() for t in range(len(times_bin))])
            fig = plt.figure(figsize=(14, 6)); gs = fig.add_gridspec(2, n_states, height_ratios=[1, 1.5], hspace=0.15, wspace=0.3)
            for i in range(n_states):
                ax_topo = fig.add_subplot(gs[0, i])
                mne.viz.plot_topomap(templates[i], info_eeg, axes=ax_topo, show=False, contours=6, cmap="RdBu_r")
                ax_topo.set_title(f"State {state_letters[i]}", fontsize=11, weight='bold', color=colors[i])
            ax_gfp = fig.add_subplot(gs[1, :])
            for t_idx in range(len(times_bin) - 1):
                state = labels_consensus[t_idx]
                ax_gfp.fill_between([times_bin[t_idx], times_bin[t_idx + 1]], 0, gfp_mean[t_idx], color=colors[state], alpha=0.6, linewidth=0)
            ax_gfp.plot(times_bin, gfp_mean, 'k-', linewidth=1.5, alpha=0.8)
            if 0 >= times_bin[0] and 0 <= times_bin[-1]: ax_gfp.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax_gfp.set_xlabel("Time (s)", fontsize=11); ax_gfp.set_ylabel("GFP (μV)", fontsize=11); ax_gfp.set_xlim([times_bin[0], times_bin[-1]]); ax_gfp.grid(True, alpha=0.3, axis='y')
            cond_title = "Non-pain" if cond_label == "nonpain" else "Pain"
            fig.suptitle(f"Microstate Sequence - {cond_title} Trials ({t_label.capitalize()} Period)", fontsize=13, weight='bold')
            plt.tight_layout(rect=[0, 0.02, 1, 0.96]); save_fig(fig, save_dir / f"sub-{subject}_microstate_gfp_sequence_{cond_label}_{t_label}", constants=_constants_from_config(config)); plt.close(fig)
    logger.info("Saved microstate GFP sequence plots by temporal bins")


def plot_microstate_transition_network(transitions: MicrostateTransitionStats, subject, save_dir, logger, config):
    if transitions is None:
        logger.warning("No microstate transition data provided; skipping plot")
        return

    state_labels = transitions.state_labels
    n_states = len(state_labels)
    if n_states == 0:
        logger.warning("Empty transition matrices; skipping plot")
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

    plt.suptitle("Microstate Transition Probabilities by Condition", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_transitions",
        constants=_constants_from_config(config),
    )
    plt.close(fig)
    logger.info("Saved microstate transition network")



def plot_microstate_duration_distributions(duration_stats: List[MicrostateDurationStat], subject, save_dir, logger, config):
    if not duration_stats:
        logger.warning("No microstate duration statistics provided; skipping violin plot")
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

    plt.suptitle("Microstate Duration Distributions by Pain Condition", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_duration_distributions",
        constants=_constants_from_config(config),
    )
    plt.close(fig)
    logger.info("Saved microstate duration distributions")


###################################################################
# Power-related plotting
###################################################################



###################################################################
# Power-related plotting
###################################################################

def plot_power_distributions(pow_df, bands, subject, save_dir, logger, config):
    n_bands = len(bands); cols = 2; rows = (n_bands + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    axes = [axes] if n_bands == 1 else axes.reshape(1, -1) if rows == 1 else axes.flatten()
    for i, band in enumerate(bands):
        band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
        if not band_cols:
            logger.warning(f"No columns found for band '{band}'"); continue
        band_data = pow_df[band_cols].values.flatten(); band_data = band_data[~np.isnan(band_data)]
        if len(band_data) == 0:
            logger.warning(f"No valid data for band '{band}'"); continue
        parts = axes[i].violinplot([band_data], positions=[1], showmeans=True, showmedians=True)
        band_color = _get_band_color(band, config)
        for pc in parts['bodies']: pc.set_facecolor(band_color); pc.set_alpha(0.7)
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
        axes[i].set_title(f'{band.capitalize()} Power Distribution\n(All channels, all trials)')
        axes[i].set_ylabel('log10(power/baseline)'); axes[i].set_xticks([]); axes[i].grid(True, alpha=0.3)
        mean_val = np.mean(band_data); std_val = np.std(band_data); median_val = np.median(band_data)
        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nMdn={median_val:.3f}\nn={len(band_data)}'
        axes[i].text(0.7, 0.95, stats_text, transform=axes[i].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    for j in range(len(bands), len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); save_fig(fig, save_dir / f'sub-{subject}_power_distributions_per_band', constants=_constants_from_config(config)); plt.close(fig)
    logger.info("Saved power distributions")


def plot_channel_power_heatmap(pow_df, bands, subject, save_dir, logger, config):
    band_means = []; channel_names = []; valid_bands = []
    for band in bands:
        band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
        if band_cols:
            band_data = pow_df[band_cols].mean(axis=0)
            band_means.append(band_data.values); valid_bands.append(band)
            if not channel_names: channel_names = [col.replace(f'pow_{band}_', '') for col in band_cols]
    if not band_means:
        logger.warning("No valid band data for heatmap"); return
    heatmap_data = np.array(band_means)
    fig, ax = plt.subplots(figsize=(max(12, len(channel_names)*0.4), max(6, len(valid_bands)*0.8)))
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(channel_names))); ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.set_yticks(range(len(valid_bands))); ax.set_yticklabels([b.capitalize() for b in valid_bands])
    ax.set_title(f'Mean Power per Channel and Band\nlog10(power/baseline)')
    ax.set_xlabel('Channel'); ax.set_ylabel('Frequency Band')
    plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
    if len(channel_names) * len(valid_bands) <= 200:
        for i in range(len(valid_bands)):
            for j in range(len(channel_names)):
                text = f'{heatmap_data[i,j]:.2f}'; color = 'white' if abs(heatmap_data[i,j]) > np.std(heatmap_data) else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=max(6, min(10, 200/len(channel_names))))
    plt.tight_layout(); save_fig(fig, save_dir / f'sub-{subject}_channel_power_heatmap', constants=_constants_from_config(config)); plt.close(fig)
    logger.info("Saved channel power heatmap")


def plot_power_time_courses(tfr_raw, bands, subject, save_dir, logger, config):
    times = tfr_raw.times
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    for band in bands:
        if band not in features_freq_bands:
            logger.warning(f"Band '{band}' not in config; skipping time course."); continue
        fmin, fmax = features_freq_bands[band]
        freq_mask = (tfr_raw.freqs >= fmin) & (tfr_raw.freqs <= fmax)
        if not freq_mask.any():
            logger.warning(f"No frequencies found for {band} band ({fmin}-{fmax} Hz)"); continue
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        band_power_log = tfr_raw.data[:, :, freq_mask, :].mean(axis=(0, 1, 2))
        ax.plot(times, band_power_log, linewidth=2, color=_get_band_color(band, config))
        b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline)
        bs = max(float(times.min()), float(b_start)); be = min(float(times.max()), float(b_end))
        if be > bs: ax.axvspan(bs, be, alpha=0.2, color='gray', label='Baseline')
        ax.axvspan(0, times[-1], alpha=0.2, color='orange', label='Stimulus')
        ax.set_ylabel(f'log10(power/baseline)'); ax.set_xlabel('Time (s)'); ax.set_title(f'{band.capitalize()} Band Power Time Course'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); save_fig(fig, save_dir / f'sub-{subject}_power_time_course_{band}', constants=_constants_from_config(config)); plt.close(fig)
        logger.info(f"Saved {band} power time course")


def plot_power_spectral_density(tfr, subject, save_dir, logger, events_df=None, config=None):
    times = np.asarray(tfr.times)
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5]) if config else [3.0, 10.5]
    plateau_start = float(plateau_window[0]); plateau_end = float(plateau_window[1])
    tmin, tmax = max(times.min(), plateau_start), min(times.max(), plateau_end)
    if tmax <= tmin: logger.warning(f"Invalid plateau window; skipping PSD"); return
    tfr_win = tfr.copy().crop(tmin, tmax)
    if events_df is not None and not events_df.empty:
        temp_col = None
        for c in ["stimulus_temp", "temperature", "thermode_temperature"]:
            if c in events_df.columns: temp_col = c; break
        if temp_col is not None:
            temps = pd.to_numeric(events_df[temp_col], errors="coerce")
            if len(tfr_win) != len(temps):
                raise ValueError(f"TFR window ({len(tfr_win)} epochs) and events ({len(temps)} rows) length mismatch for subject {subject}")
            unique_temps = sorted(temps.dropna().unique())
            if len(unique_temps) >= 2:
                fig, ax = plt.subplots(figsize=(9, 6))
                temp_colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(unique_temps)))
                for idx, temp in enumerate(unique_temps):
                    temp_mask = (temps == temp).to_numpy()
                    if temp_mask.sum() < 1: continue
                    data_temp = tfr_win.data[temp_mask]; psd_avg = data_temp.mean(axis=(0, 1, 3))
                    if len(psd_avg) != len(tfr_win.freqs): logger.warning(f"Frequency dimension mismatch: {len(psd_avg)} vs {len(tfr_win.freqs)}"); continue
                    ax.plot(tfr_win.freqs, psd_avg, color=temp_colors[idx], linewidth=1.5, label=f'{temp:.0f}°C', alpha=0.9)
                ax.axhline(0, color='gray', linewidth=1.0, alpha=0.4, linestyle='--')
                ax.set_xlabel("Frequency (Hz)", fontsize=12); ax.set_ylabel("Power spectral density (log10 ratio to baseline)", fontsize=12)
                ax.legend(loc='upper left', fontsize=10, frameon=False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
                plt.tight_layout(); save_fig(fig, save_dir / f'sub-{subject}_power_spectral_density_by_temperature', constants=_constants_from_config(config)); plt.close(fig); logger.info("Saved PSD by temperature"); return
    data = tfr_win.data.mean(axis=(0, 3)); psd_avg = data.mean(axis=0); psd_sem = data.std(axis=0) / np.sqrt(data.shape[0])
    fig, ax = plt.subplots(figsize=(4.0, 2.5), constrained_layout=True)
    ax.plot(tfr.freqs, psd_avg, color="0.2", linewidth=1.0)
    ax.fill_between(tfr.freqs, psd_avg - 1.96*psd_sem, psd_avg + 1.96*psd_sem, color="0.4", alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.7", linewidth=0.5, alpha=0.6)
    freq_bands = config.get("time_frequency_analysis.bands", {"delta": [1.0, 3.9], "theta": [4.0, 7.9], "alpha": [8.0, 12.9], "beta": [13.0, 30.0], "gamma": [30.1, 80.0], })
    FEATURES_FREQ_BANDS = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    for band, (fmin, fmax) in FEATURES_FREQ_BANDS.items():
        if fmin < tfr.freqs.max():
            ax.axvspan(fmin, min(fmax, tfr.freqs.max()), alpha=0.08, color="0.5", linewidth=0)
            mid = (fmin + min(fmax, tfr.freqs.max())) / 2
            if mid < tfr.freqs.max(): ax.text(mid, ax.get_ylim()[1]*0.95, band[0].upper(), fontsize=7, ha="center", va="top", color="0.4")
    ax.set_xlabel("Frequency (Hz)", fontsize=9); ax.set_ylabel("log10(power/baseline)", fontsize=9); ax.tick_params(labelsize=8); sns.despine(ax=ax, trim=True)
    save_fig(fig, save_dir / f'sub-{subject}_power_spectral_density', constants=_constants_from_config(config)); plt.close(fig); logger.info("Saved PSD")


def plot_power_spectral_density_by_pain(tfr, subject, save_dir, logger, events_df=None, config=None):
    if events_df is None or events_df.empty:
        logger.warning("No events for PSD by pain"); return
    pain_col = None
    for c in ["pain_binary_coded", "pain_binary", "pain"]:
        if c in events_df.columns: pain_col = c; break
    if pain_col is None: logger.warning("No pain binary column found"); return
    times = np.asarray(tfr.times)
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5]) if config else [3.0, 10.5]
    plateau_start = float(plateau_window[0]); plateau_end = float(plateau_window[1])
    tmin, tmax = max(times.min(), plateau_start), min(times.max(), plateau_end)
    if tmax <= tmin: logger.warning(f"Invalid plateau window; skipping PSD by pain"); return
    tfr_win = tfr.copy().crop(tmin, tmax)
    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    if len(tfr_win) != len(pain_vals):
        raise ValueError(f"TFR window ({len(tfr_win)} epochs) and events ({len(pain_vals)} rows) length mismatch for subject {subject}")
    nonpain_mask = (pain_vals == 0).to_numpy(); pain_mask = (pain_vals == 1).to_numpy()
    if nonpain_mask.sum() < 1 or pain_mask.sum() < 1: logger.warning("Insufficient trials for pain comparison"); return
    fig, ax = plt.subplots(figsize=(9, 6))
    for mask, label, color in [(nonpain_mask, 'Non-pain', 'steelblue'), (pain_mask, 'Pain', 'orangered')]:
        if mask.sum() < 1: continue
        data_cond = tfr_win.data[mask]; psd_avg = data_cond.mean(axis=(0, 1, 3))
        if len(psd_avg) != len(tfr_win.freqs): logger.warning(f"Frequency dimension mismatch for {label}"); continue
        ax.plot(tfr_win.freqs, psd_avg, color=color, linewidth=1.5, label=label, alpha=0.9)
    ax.axhline(0, color='gray', linewidth=1.0, alpha=0.4, linestyle='--')
    ax.set_xlabel("Frequency (Hz)", fontsize=12); ax.set_ylabel("Power spectral density (log10 ratio to baseline)", fontsize=12)
    ax.legend(loc='upper left', fontsize=10, frameon=False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
    plt.tight_layout(); save_fig(fig, save_dir / f'sub-{subject}_power_spectral_density_by_pain', constants=_constants_from_config(config)); plt.close(fig); logger.info("Saved PSD by pain condition")


def plot_power_time_course_by_temperature(tfr, subject, save_dir, logger, events_df=None, band='alpha', config=None):
    if events_df is None or events_df.empty:
        logger.warning("No events for time course by temperature"); return
    temp_col = None
    for c in ["stimulus_temp", "temperature", "thermode_temperature"]:
        if c in events_df.columns: temp_col = c; break
    if temp_col is None: logger.warning("No temperature column found"); return
    temps = pd.to_numeric(events_df[temp_col], errors="coerce")
    if len(tfr) != len(temps):
        raise ValueError(f"TFR ({len(tfr)} epochs) and events ({len(temps)} rows) length mismatch for subject {subject}")
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < 2: logger.warning("Insufficient temperature levels for time course plot"); return
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands if config else {}
    if band not in features_freq_bands: logger.warning(f"Band {band} not in configured bands"); return
    fmin, fmax = features_freq_bands[band]; freq_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    if not freq_mask.any(): logger.warning(f"No frequencies found for {band} band"); return
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(unique_temps)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        if temp_mask.sum() < 1: continue
        band_power = tfr.data[temp_mask][:, :, freq_mask, :].mean(axis=(0, 1, 2))
        ax.plot(tfr.times, band_power, color=colors[idx], linewidth=1.8, alpha=0.9, label=f"{temp:.0f}°C")
    ax.axhline(0, color='gray', linewidth=1.0, alpha=0.4, linestyle='--')
    ax.set_xlabel("Time (s)", fontsize=12); ax.set_ylabel(f"{band.capitalize()} power (log10 ratio)", fontsize=12)
    ax.legend(loc='upper left', fontsize=10, frameon=False); ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
    plt.tight_layout(); save_fig(fig, save_dir / f'sub-{subject}_time_course_by_temperature_{band}', constants=_constants_from_config(config)); plt.close(fig); logger.info("Saved band time course by temperature")


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
    n_bands = len(bands)
    fig, axes = plt.subplots(n_bands, 1, figsize=(12, 3*n_bands))
    if n_bands == 1:
        axes = [axes]
    for i, band in enumerate(bands):
        band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
        if not band_cols:
            continue
        band_power_trials = pow_df[band_cols].mean(axis=1)
        trial_nums = range(1, len(band_power_trials) + 1)
        axes[i].plot(trial_nums, band_power_trials, 'o-', alpha=0.7, linewidth=1, color=_get_band_color(band, config))
        mean_power = band_power_trials.mean()
        axes[i].axhline(mean_power, color='red', linestyle='--', alpha=0.8, label=f'Mean = {mean_power:.3f}')
        std_power = band_power_trials.std()
        cv_power = std_power / abs(mean_power) if abs(mean_power) > 1e-10 else np.nan
        axes[i].fill_between(trial_nums, mean_power - std_power, mean_power + std_power, alpha=0.2, color='red', label=f'±1 SD = ±{std_power:.3f}')
        axes[i].set_ylabel(f'{band.capitalize()}\nlog10(power/baseline)')
        axes[i].set_title(f'{band.capitalize()} Band Power Variability (CV = {cv_power:.3f})')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    axes[-1].set_xlabel('Trial Number')
    plt.tight_layout()
    save_fig(fig, save_dir / f'sub-{subject}_trial_power_variability', constants=_constants_from_config(config))
    plt.close(fig)
    logger.info("Saved trial power variability")


def plot_inter_band_spatial_power_correlation(tfr, subject, save_dir, logger, config):
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    band_names = list(features_freq_bands.keys())
    n_bands = len(band_names)
    coupling_matrix = np.zeros((n_bands, n_bands))
    times = np.asarray(tfr.times)
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
    plateau_start = float(plateau_window[0])
    plateau_end = float(plateau_window[1])
    tmin_clip = float(max(times.min(), plateau_start))
    tmax_clip = float(min(times.max(), plateau_end))
    if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
        logger.warning(
            f"Skipping inter-band spatial power correlation: invalid plateau within data range "
            f"(requested [{plateau_start}, {plateau_end}] s, available [{times.min():.2f}, {times.max():.2f}] s)"
        )
        return
    tfr_windowed = tfr.copy().crop(tmin_clip, tmax_clip)
    tfr_avg = tfr_windowed.average()
    for i, band1 in enumerate(band_names):
        fmin1, fmax1 = features_freq_bands[band1]
        freq_mask1 = (tfr_avg.freqs >= fmin1) & (tfr_avg.freqs <= fmax1)
        if not freq_mask1.any():
            continue
        for j, band2 in enumerate(band_names):
            if i == j:
                coupling_matrix[i, j] = 1.0
                continue
            fmin2, fmax2 = features_freq_bands[band2]
            freq_mask2 = (tfr_avg.freqs >= fmin2) & (tfr_avg.freqs <= fmax2)
            if not freq_mask2.any():
                continue
            band1_channels = tfr_avg.data[:, freq_mask1, :].mean(axis=(1, 2))
            band2_channels = tfr_avg.data[:, freq_mask2, :].mean(axis=(1, 2))
            if len(band1_channels) > 1 and len(band2_channels) > 1:
                correlation = np.corrcoef(band1_channels, band2_channels)[0, 1]
                coupling_matrix[i, j] = correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(coupling_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_bands)); ax.set_yticks(range(n_bands))
    ax.set_xticklabels([band.capitalize() for band in band_names], rotation=45, ha='right')
    ax.set_yticklabels([band.capitalize() for band in band_names])
    for i in range(n_bands):
        for j in range(n_bands):
            ax.text(j, i, f'{coupling_matrix[i, j]:.2f}', ha="center", va="center", color=("black" if abs(coupling_matrix[i, j]) < 0.5 else "white"))
    ax.set_title('Inter Band Spatial Power Correlation')
    ax.set_xlabel('Frequency Band'); ax.set_ylabel('Frequency Band')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('Correlation (r)')
    plt.tight_layout()
    save_fig(fig, save_dir / f'sub-{subject}_inter_band_spatial_power_correlation', constants=_constants_from_config(config))
    plt.close(fig)
    logger.info("Saved inter-band spatial power correlation")


###################################################################
# Group-level power plotting (migrated from 02_feature_extraction)
###################################################################

def plot_group_power_plots(subj_pow: dict, bands: list, gplots, gstats, config, logger) -> None:
    import pandas as pd
    import numpy as np
    # Heatmap of mean power per channel across subjects
    band_channels = {}
    for b in bands:
        ch_union = set()
        for _, df in subj_pow.items():
            cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
            ch_union.update([c.replace(f"pow_{b}_", "") for c in cols])
        band_channels[b] = sorted(ch_union)
    all_ch_union = sorted(set().union(*band_channels.values())) if band_channels else []
    heat_rows = []
    stats_rows = []
    for b in bands:
        subj_means_per_ch = []
        for _, df in subj_pow.items():
            vals = []
            for ch in all_ch_union:
                col = f"pow_{b}_{ch}"
                if col in df.columns:
                    vals.append(float(pd.to_numeric(df[col], errors="coerce").mean()))
                else:
                    vals.append(np.nan)
            subj_means_per_ch.append(vals)
        arr = np.asarray(subj_means_per_ch, dtype=float)
        mean_across_subj = np.nanmean(arr, axis=0)
        heat_rows.append(mean_across_subj)
        n_eff = np.sum(np.isfinite(arr), axis=0)
        std_across_subj = np.nanstd(arr, axis=0, ddof=1)
        for j, ch in enumerate(all_ch_union):
            stats_rows.append({"band": b, "channel": ch, "mean": float(mean_across_subj[j]) if np.isfinite(mean_across_subj[j]) else np.nan, "std": float(std_across_subj[j]) if np.isfinite(std_across_subj[j]) else np.nan, "n_subjects": int(n_eff[j])})
    heat = np.vstack(heat_rows) if heat_rows else np.zeros((0, 0))
    if heat.size > 0:
        fig, ax = plt.subplots(figsize=(max(12, len(all_ch_union) * 0.4), max(6, len(bands) * 0.8)))
        im = ax.imshow(heat, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(all_ch_union)))
        ax.set_xticklabels(all_ch_union, rotation=45, ha='right')
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels([b.capitalize() for b in bands])
        ax.set_title("Group Mean Power per Channel and Band\nlog10(power/baseline)")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Frequency Band")
        plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
        plt.tight_layout()
        save_fig(fig, gplots / "group_channel_power_heatmap", constants=_constants_from_config(config))
    pd.DataFrame(stats_rows).to_csv(gstats / "group_channel_power_means.tsv", sep="\t", index=False)
    logger.info("Saved group channel power heatmap and stats.")

    # Band power summary with subject means and 95% CI
    recs = []
    for b in bands:
        for s, df in subj_pow.items():
            cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
            if not cols:
                continue
            vals = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            recs.append({"subject": s, "band": b, "mean_power": float(np.mean(vals))})
    dfm = pd.DataFrame(recs)
    if not dfm.empty:
        bands_present = [b for b in bands if b in set(dfm["band"])]
        means = []
        ci_l = []
        ci_h = []
        ns = []
        for b in bands_present:
            v = dfm[dfm["band"] == b]["mean_power"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            mu = float(np.mean(v)) if v.size else np.nan
            se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            delta = 1.96 * se if np.isfinite(se) else np.nan
            means.append(mu); ci_l.append(mu - delta if np.isfinite(delta) else np.nan); ci_h.append(mu + delta if np.isfinite(delta) else np.nan); ns.append(len(v))
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(bands_present))
        ax.bar(x, means, color='steelblue', alpha=0.8)
        yerr = np.array([[mu - lo if np.isfinite(mu) and np.isfinite(lo) else 0 for lo, mu in zip(ci_l, means)], [hi - mu if np.isfinite(mu) and np.isfinite(hi) else 0 for hi, mu in zip(ci_h, means)]])
        ax.errorbar(x, means, yerr=yerr, fmt='none', ecolor='k', capsize=3)
        for i, b in enumerate(bands_present):
            vals = dfm[dfm["band"] == b]["mean_power"].to_numpy(dtype=float)
            jitter = (np.random.rand(len(vals)) - 0.5) * 0.2
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, color='k', s=12, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels([bp.capitalize() for bp in bands_present])
        ax.set_ylabel('Mean log10(power/baseline) across subjects'); ax.set_title('Group Band Power Summary (subject means, 95% CI)'); ax.axhline(0, color='k', linewidth=0.8)
        plt.tight_layout(); save_fig(fig, gplots / "group_power_distributions_per_band_across_subjects", constants=_constants_from_config(config))
        out = pd.DataFrame({"band": bands_present, "group_mean": means, "ci_low": ci_l, "ci_high": ci_h, "n_subjects": ns})
        out.to_csv(gstats / "group_band_power_subject_means.tsv", sep="\t", index=False)
        logger.info("Saved group band power distributions and stats.")

    # Inter-band spatial correlation across subjects
    freq_bands = config.get("time_frequency_analysis.bands", {"delta": [1.0, 3.9], "theta": [4.0, 7.9], "alpha": [8.0, 12.9], "beta": [13.0, 30.0], "gamma": [30.1, 80.0], })
    FEATURES_FREQ_BANDS = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    band_names = list(FEATURES_FREQ_BANDS.keys()); m = len(band_names)
    per_subject_corrs = []
    for _, df in subj_pow.items():
        band_vecs = {}
        for b in band_names:
            cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
            if not cols:
                continue
            ser = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
            ch_means = {c.replace(f"pow_{b}_", ""): float(v) for c, v in ser.items() if np.isfinite(v)}
            if ch_means: band_vecs[b] = ch_means
        if len(band_vecs) < 2: continue
        corr_mat = np.eye(m, dtype=float)
        for i, bi in enumerate(band_names):
            for j, bj in enumerate(band_names):
                if j <= i: continue
                di = band_vecs.get(bi); dj = band_vecs.get(bj)
                if di is None or dj is None: corr = np.nan
                else:
                    common = sorted(set(di.keys()) & set(dj.keys()))
                    if len(common) < 2: corr = np.nan
                    else:
                        vi = np.array([di[ch] for ch in common], dtype=float); vj = np.array([dj[ch] for ch in common], dtype=float)
                        corr = float(np.corrcoef(vi, vj)[0, 1]) if np.std(vi) >= 1e-12 and np.std(vj) >= 1e-12 else np.nan
                corr_mat[i, j] = corr; corr_mat[j, i] = corr
        per_subject_corrs.append(corr_mat)
    if len(per_subject_corrs) >= 2:
        arr = np.stack(per_subject_corrs, axis=0); group_corr = np.eye(m, dtype=float)
        for i in range(m):
            for j in range(m):
                if i == j: group_corr[i, j] = 1.0; continue
                rvals = arr[:, i, j]; rvals = rvals[np.isfinite(rvals)]
                if rvals.size == 0: group_corr[i, j] = np.nan
                else:
                    z = np.arctanh(np.clip(rvals, -0.999999, 0.999999)); zbar = float(np.mean(z)); group_corr[i, j] = float(np.tanh(zbar))
        fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(group_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(m)); ax.set_yticks(range(m)); ax.set_xticklabels([b.capitalize() for b in band_names], rotation=45, ha='right'); ax.set_yticklabels([b.capitalize() for b in band_names])
        for i in range(m):
            for j in range(m):
                if np.isfinite(group_corr[i, j]): ax.text(j, i, f"{group_corr[i, j]:.2f}", ha='center', va='center', color=('white' if abs(group_corr[i, j]) > 0.5 else 'black'))
        ax.set_title('Group Inter Band Spatial Power Correlation'); ax.set_xlabel('Frequency Band'); ax.set_ylabel('Frequency Band'); plt.colorbar(im, ax=ax, label='Correlation (r)'); plt.tight_layout(); save_fig(fig, gplots / "group_inter_band_spatial_power_correlation", constants=_constants_from_config(config))
        rows = []
        for i in range(m):
            for j in range(i + 1, m):
                rvals = np.array([cm[i, j] for cm in per_subject_corrs], dtype=float); rvals = rvals[np.isfinite(rvals)]
                if rvals.size == 0: continue
                z = np.arctanh(np.clip(rvals, -0.999999, 0.999999)); zbar = float(np.mean(z)); se = float(np.std(z, ddof=1) / np.sqrt(len(z))) if len(z) > 1 else np.nan
                ci_l = float(np.tanh(zbar - 1.96 * se)) if np.isfinite(se) else np.nan; ci_h = float(np.tanh(zbar + 1.96 * se)) if np.isfinite(se) else np.nan
                rows.append({"band_i": band_names[i], "band_j": band_names[j], "r_group": float(np.tanh(zbar)), "r_ci_low": ci_l, "r_ci_high": ci_h, "n_subjects": int(len(rvals))})
        if rows:
            pd.DataFrame(rows).to_csv(gstats / "group_inter_band_correlation.tsv", sep="\t", index=False); logger.info("Saved group inter-band correlation heatmap and stats.")


def plot_group_band_power_time_courses(valid_bands: list, band_tc: dict, band_tc_pct: dict, tref: np.ndarray, gplots, config, logger) -> None:
    nrows = len(valid_bands)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    for i, b in enumerate(valid_bands):
        ax = axes[i]
        series_list = band_tc.get(b, [])
        arr = np.vstack(series_list)
        mu = np.nanmean(arr, axis=0)
        se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ci = 1.96 * se
        ax.plot(tref, mu, color=_get_band_color(b, config), label=f"{b}")
        ax.fill_between(tref, mu - ci, mu + ci, color=_get_band_color(b, config), alpha=0.2)
        tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
        min_baseline_samples = int(config.get("tfr_topography_pipeline.min_baseline_samples", 5))
        b_start, b_end, _ = validate_baseline_indices(tref, tfr_baseline, min_samples=min_baseline_samples)
        bs = max(float(tref.min()), float(b_start))
        be = min(float(tref.max()), float(b_end))
        if be > bs:
            ax.axvspan(bs, be, alpha=0.1, color='gray')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(f"{b.capitalize()} (group mean ±95% CI)")
        ax.set_ylabel("log10(power/baseline)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Group Band Power Time Courses")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, gplots / "group_band_power_time_courses", constants=_constants_from_config(config))
    plt.close(fig)
    logger.info("Saved group band power time courses.")

    fig2, axes2 = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes2 = [axes2]
    for i, b in enumerate(valid_bands):
        ax2 = axes2[i]
        series_list_pct = band_tc_pct.get(b, [])
        if len(series_list_pct) < 2:
            continue
        arrp = np.vstack(series_list_pct)
        mu = np.nanmean(arrp, axis=0)
        se = np.nanstd(arrp, axis=0, ddof=1) / np.sqrt(arrp.shape[0])
        ci = 1.96 * se
        ax2.plot(tref, mu, color=_get_band_color(b, config), label=f"{b}")
        ax2.fill_between(tref, mu - ci, mu + ci, color=_get_band_color(b, config), alpha=0.2)
        tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
        min_baseline_samples = int(config.get("tfr_topography_pipeline.min_baseline_samples", 5))
        b_start, b_end, _ = validate_baseline_indices(tref, tfr_baseline, min_samples=min_baseline_samples)
        bs = max(float(tref.min()), float(b_start))
        be = min(float(tref.max()), float(b_end))
        if be > bs:
            ax2.axvspan(bs, be, alpha=0.1, color='gray')
        ax2.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax2.set_title(f"{b.capitalize()} (group mean ±95% CI)")
        ax2.set_ylabel("Percent change from baseline (%)")
        ax2.grid(True, alpha=0.3)
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Group Band Power Time Courses (percent change, ratio-domain averaging)")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig2, gplots / "group_band_power_time_courses_percent_change", constants=_constants_from_config(config))
    plt.close(fig2)
    logger.info("Saved group band power time courses (percent change).")


