"""ERP waveform visualizations.

This module provides functions for plotting event-related potentials (ERPs),
including butterfly plots, ROI-based waveforms, and condition contrasts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.style import use_style, get_color
from eeg_pipeline.utils.analysis.channels import build_roi_map, pick_eeg_channels
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions


def plot_butterfly_erp(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    conditions: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Plot butterfly ERP for all trials or specific conditions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs data.
    subject : str
        Subject ID for labeling and filenames.
    save_dir : Path
        Directory to save plots.
    config : Any
        Pipeline configuration.
    logger : logging.Logger
        Logger instance.
    conditions : dict, optional
        Conditions to plot separately (name -> query).
        If None, plots all trials.
    """
    saved_paths = []
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    with use_style(context="paper"):
        # 1. Overall butterfly (all trials)
        evoked = epochs.average()
        fig = evoked.plot(
            picks="eeg",
            spatial_colors=True,
            gfp=True,
            show=False,
            window_title=f"sub-{subject} Butterfly ERP (All Trials)"
        )
        fig.suptitle(f"sub-{subject}: Butterfly ERP (All Trials)", fontsize=14)
        
        path = save_dir / f"sub-{subject}_erp_butterfly_all.{primary_ext}"
        save_fig(
            fig,
            path,
            logger=logger,
            formats=plot_cfg.formats,
            dpi=plot_cfg.savefig_dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        saved_paths.append(path)
        
        # 2. Condition-specific butterflies
        if conditions:
            for cond_name, query in conditions.items():
                try:
                    cond_epochs = epochs[query]
                    if len(cond_epochs) == 0:
                        continue
                        
                    evoked_cond = cond_epochs.average()
                    fig = evoked_cond.plot(
                        picks="eeg",
                        spatial_colors=True,
                        gfp=True,
                        show=False,
                        window_title=f"sub-{subject} Butterfly ERP ({cond_name})"
                    )
                    fig.suptitle(f"sub-{subject}: Butterfly ERP ({cond_name})", fontsize=14)
                    
                    path = save_dir / f"sub-{subject}_erp_butterfly_{cond_name}.{primary_ext}"
                    save_fig(
                        fig,
                        path,
                        logger=logger,
                        formats=plot_cfg.formats,
                        dpi=plot_cfg.savefig_dpi,
                        bbox_inches=plot_cfg.bbox_inches,
                        pad_inches=plot_cfg.pad_inches,
                    )
                    saved_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to plot butterfly for condition {cond_name}: {e}")

    return saved_paths


def plot_roi_erp(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    conditions: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Plot ROI-based ERP waveforms with shaded error bars.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs data.
    subject : str
    save_dir : Path
    config : Any
    logger : logging.Logger
    conditions : dict, optional
        Mapping of display name -> metadata query string (pandas/MNE syntax).
    """
    saved_paths = []
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    roi_defs = get_roi_definitions(config)
    if not roi_defs:
        logger.warning("No ROI definitions found in config; skipping ROI ERP plot.")
        return saved_paths

    picks, ch_names = pick_eeg_channels(epochs)
    roi_map = build_roi_map(ch_names, roi_defs)
    
    if not roi_map:
        logger.warning("No ROIs matched available channels.")
        return saved_paths

    time_vec = epochs.times * 1000  # Convert to ms
    
    # Filter ROIs based on config
    from eeg_pipeline.utils.config.loader import get_config_value
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    has_specific = any(r.lower() != "all" for r in comp_rois)
    if comp_rois and has_specific:
        filtered_roi_map = {}
        for r in comp_rois:
            if r.lower() == "all": continue
            if r in roi_map:
                filtered_roi_map[r] = roi_map[r]
        roi_map = filtered_roi_map

    with use_style(context="paper"):
        for roi_name, indices in roi_map.items():
            if not indices:
                continue
                
            fig, ax = plt.subplots(figsize=(8, 5))
            
            if conditions:
                for cond_name, query in conditions.items():
                    try:
                        cond_data = epochs[query].get_data(picks=indices)
                        if cond_data.size == 0:
                            continue
                            
                        # Mean across channels, then across trials
                        roi_avg_per_trial = np.mean(cond_data, axis=1) # (trials, times)
                        mean_waveform = np.mean(roi_avg_per_trial, axis=0) * 1e6 # Convert to uV
                        sem_waveform = (np.std(roi_avg_per_trial, axis=0) / np.sqrt(len(cond_data))) * 1e6
                        
                        color = get_color(cond_name, default=None)
                        ax.plot(time_vec, mean_waveform, label=cond_name, color=color, linewidth=2)
                        ax.fill_between(
                            time_vec,
                            mean_waveform - sem_waveform,
                            mean_waveform + sem_waveform,
                            alpha=0.2,
                            color=color
                        )
                    except Exception as e:
                        logger.warning(f"Failed to plot ROI {roi_name} for condition {cond_name}: {e}")
            else:
                # All trials
                data = epochs.get_data(picks=indices)
                roi_avg_per_trial = np.mean(data, axis=1)
                mean_waveform = np.mean(roi_avg_per_trial, axis=0) * 1e6
                sem_waveform = (np.std(roi_avg_per_trial, axis=0) / np.sqrt(len(data))) * 1e6
                
                ax.plot(time_vec, mean_waveform, label="All Trials", color="#333333", linewidth=2)
                ax.fill_between(
                    time_vec,
                    mean_waveform - sem_waveform,
                    mean_waveform + sem_waveform,
                    alpha=0.2,
                    color="#333333"
                )

            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            ax.axhline(0, color="black", linestyle="-", alpha=0.3)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (μV)")
            ax.set_title(f"sub-{subject}: {roi_name} ERP")
            ax.legend()
            
            # Use baseline window from config if available
            erp_cfg = config.get("feature_engineering.erp", {})
            baseline = erp_cfg.get("baseline_window", [-0.2, 0.0])
            ax.axvspan(baseline[0]*1000, baseline[1]*1000, color="gray", alpha=0.1, label="Baseline")
            
            path = save_dir / f"sub-{subject}_erp_roi_{roi_name.lower()}.{primary_ext}"
            save_fig(
                fig,
                path,
                logger=logger,
                formats=plot_cfg.formats,
                dpi=plot_cfg.savefig_dpi,
                bbox_inches=plot_cfg.bbox_inches,
                pad_inches=plot_cfg.pad_inches,
            )
            saved_paths.append(path)
            
    return saved_paths


def plot_erp_contrast(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    cond_a: Optional[str] = None,
    cond_b: Optional[str] = None,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
) -> List[Path]:
    """Plot ERP contrast (A - B) for all channels."""
    saved_paths = []
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"

    if cond_a is None or cond_b is None or label_a is None or label_b is None:
        from eeg_pipeline.utils.analysis.events import resolve_comparison_spec

        def _query_for_value(col: str, value: Any) -> str:
            import numpy as np
            import pandas as pd

            col_expr = f"`{col}`"
            try:
                v_num = pd.to_numeric(str(value), errors="coerce")
                if not np.isnan(v_num):
                    if float(v_num).is_integer():
                        return f"{col_expr} == {int(v_num)}"
                    return f"{col_expr} == {float(v_num)}"
            except Exception:
                pass
            return f"{col_expr} == {repr(str(value))}"

        meta = getattr(epochs, "metadata", None)
        spec = resolve_comparison_spec(meta, config, require_enabled=False) if meta is not None else None
        if spec is None:
            logger.warning("No configured comparison found in metadata; skipping ERP contrast.")
            return saved_paths
        col, v1, v2, auto_label1, auto_label2 = spec
        cond_a = _query_for_value(col, v2)
        cond_b = _query_for_value(col, v1)
        label_a = auto_label2
        label_b = auto_label1
    
    try:
        evoked_a = epochs[cond_a].average()
        evoked_b = epochs[cond_b].average()
        
        diff = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
        
        with use_style(context="paper"):
            fig = diff.plot(
                picks="eeg",
                spatial_colors=True,
                gfp=True,
                show=False,
                window_title=f"sub-{subject} ERP Contrast ({label_a} - {label_b})"
            )
            fig.suptitle(f"sub-{subject}: ERP Contrast ({label_a} - {label_b})", fontsize=14)
            
            path = save_dir / f"sub-{subject}_erp_contrast_{label_a.lower()}_{label_b.lower()}.{primary_ext}"
            save_fig(
                fig,
                path,
                logger=logger,
                formats=plot_cfg.formats,
                dpi=plot_cfg.savefig_dpi,
                bbox_inches=plot_cfg.bbox_inches,
                pad_inches=plot_cfg.pad_inches,
            )
            saved_paths.append(path)
            
    except Exception as e:
        logger.warning(f"Failed to plot ERP contrast: {e}")
        
    return saved_paths
