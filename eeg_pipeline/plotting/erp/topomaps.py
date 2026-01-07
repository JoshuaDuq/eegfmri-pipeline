"""ERP topomap visualizations.

This module provides functions for plotting the spatial distribution of ERP
amplitudes at specific time windows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.style import use_style


def plot_erp_topomaps(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    conditions: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Plot ERP topomaps for defined components.
    
    Parameters
    ----------
    epochs : mne.Epochs
    subject : str
    save_dir : Path
    config : Any
    logger : logging.Logger
    conditions : dict, optional
    """
    saved_paths = []
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    erp_cfg = config.get("feature_engineering.erp", {})
    components = erp_cfg.get("components", [])
    if not components:
        logger.warning("No ERP components defined in config; using defaults for topomaps.")
        components = [
            {"name": "N2", "start": 0.15, "end": 0.25},
            {"name": "P2", "start": 0.3, "end": 0.45}
        ]
        
    times = [ (c["start"] + c["end"]) / 2 for c in components ]
    time_names = [ c["name"] for c in components ]
    
    with use_style(context="paper"):
        # 1. Overall topomaps (all trials)
        evoked = epochs.average()
        fig = evoked.plot_topomap(
            times=times,
            average=0.05, # +/- 25ms window around center
            show=False,
            #title=f"sub-{subject} ERP Topomaps (All Trials)"
        )
        for ax in fig.axes:
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=8)
        # fig.suptitle is tricky with mne.plot_topomap as it returns a Figure object 
        # but the layout is already tight.
        
        path = save_dir / f"sub-{subject}_erp_topomaps_all.{primary_ext}"
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
        
        # 2. Condition-specific topomaps
        if conditions:
            for cond_name, query in conditions.items():
                try:
                    cond_epochs = epochs[query]
                    if len(cond_epochs) == 0:
                        continue
                        
                    evoked_cond = cond_epochs.average()
                    fig = evoked_cond.plot_topomap(
                        times=times,
                        average=0.05,
                        show=False,
                    )
                    for ax in fig.axes:
                        if ax.get_title():
                            ax.set_title(ax.get_title(), fontsize=8)
                    
                    path = save_dir / f"sub-{subject}_erp_topomaps_{cond_name}.{primary_ext}"
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
                    logger.warning(f"Failed to plot topomap for condition {cond_name}: {e}")

        # 3. Contrast topomaps
        if conditions and len(conditions) >= 2:
            # Assume first two are the primary contrast (e.g. condition A vs condition B)
            keys = list(conditions.keys())
            try:
                evoked_a = epochs[conditions[keys[0]]].average()
                evoked_b = epochs[conditions[keys[1]]].average()
                diff = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
                
                fig = diff.plot_topomap(
                    times=times,
                    average=0.05,
                    show=False,
                )
                for ax in fig.axes:
                    if ax.get_title():
                        ax.set_title(ax.get_title(), fontsize=8)
                
                path = save_dir / f"sub-{subject}_erp_topomaps_contrast.{primary_ext}"
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
                logger.warning(f"Failed to plot contrast topomaps: {e}")

    return saved_paths
