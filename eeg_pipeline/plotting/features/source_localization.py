"""
Source Localization Visualization
=================================

Plotting functions for 3D source localization brain maps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import mne
from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names

def plot_source_stc_3d(
    subject: str,
    stc_files: List[Path],
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    subjects_dir: Optional[str] = None
) -> None:
    """Plot 3D brain maps from saved STC files."""
    if not stc_files:
        logger.info("No STC files provided for source plotting.")
        return

    if subjects_dir is None:
        subjects_dir = get_config_value(config, "feature_engineering.sourcelocalization.subjects_dir", None)
    
    if subjects_dir is None:
        logger.warning("subjects_dir is not configured. 3D source plotting may fail if fsaverage or subject MRI is missing.")

    out_dir = save_dir / "3d_brains"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Plotting 3D brains for {len(stc_files)} STC combinations...")

    for stc_path in stc_files:
        try:
            # We assume the STC path stem has a specific naming convention we set during feature computation
            # E.g., sub-0000_task-..._cond-..._band-..._lcmv
            stc_name = stc_path.name
            stc = mne.read_source_estimate(str(stc_path))
            
            # Using PyVista 3D Engine for Brain plotting
            brain = stc.plot(
                subject=subject,
                subjects_dir=subjects_dir,
                hemi="both",
                views=["lateral", "medial"],
                background="white",
                time_viewer=False,
                show_traces=False,
                size=(1200, 600),
                cortex="bone",
            )
            
            image_name = stc_path.name.replace("-lh.stc", "").replace("-rh.stc", "").replace("-vl.stc", "")
            save_path = out_dir / f"{image_name}_3d.png"
            
            brain.save_image(str(save_path))
            brain.close()
            
            if logger:
                logger.debug(f"Saved 3D source plot: {save_path.name}")
                
        except Exception as exc:
            if logger:
                logger.error(f"Failed to plot STC 3D brain for {stc_path.name}: {exc}")

