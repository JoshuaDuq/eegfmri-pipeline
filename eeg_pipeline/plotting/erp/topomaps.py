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
        # fig.suptitle is tricky with mne.plot_topomap as it returns a Figure object 
        # but the layout is already tight.
        
        path = save_dir / f"sub-{subject}_erp_topomaps_all.png"
        fig.savefig(path)
        plt.close(fig)
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
                    
                    path = save_dir / f"sub-{subject}_erp_topomaps_{cond_name}.png"
                    fig.savefig(path)
                    plt.close(fig)
                    saved_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to plot topomap for condition {cond_name}: {e}")

        # 3. Contrast topomaps
        if conditions and len(conditions) >= 2:
            # Assume first two are the primary contrast (e.g. Pain vs NoPain)
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
                
                path = save_dir / f"sub-{subject}_erp_topomaps_contrast.png"
                fig.savefig(path)
                plt.close(fig)
                saved_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to plot contrast topomaps: {e}")

    return saved_paths
