"""
HTML Report Generation
======================

Generates interactive HTML dashboards using MNE Report.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import mne
import matplotlib.pyplot as plt

def generate_feature_report(ctx: Any, saved_files: Dict[str, Path]) -> Path:
    """
    Generate an HTML report aggregating all visualization outputs.
    
    Args:
        ctx: FeaturePlotContext
        saved_files: Dictionary mapping plot names to file paths
        
    Returns:
        Path to the generated HTML report
    """
    logger = getattr(ctx, "logger", logging.getLogger(__name__))
    logger.info("Generating HTML Report...")
    
    rep = mne.Report(
        title=f"Feature Analysis: sub-{ctx.subject}",
        verbose=False
    )
    
    # 1. Add Summary Info
    # We can add raw html or text
    summary_html = f"""
    <h3>Analysis Context</h3>
    <ul>
        <li><b>Subject:</b> {ctx.subject}</li>
        <li><b>Trials:</b> {ctx.n_trials}</li>
        <li><b>Feature Directory:</b> {ctx.features_dir}</li>
    </ul>
    """
    rep.add_html(summary_html, title="Experiment Info", section="Overview")
    
    # 2. Add Plots
    # logical order
    sections_order = [
        "Summary",
        "Quality",
        "Power",
        "Connectivity",
        "Aperiodic",
        "Complexity",
        "ERDS",
        "ITPC",
        "PAC",
        "ERP",
    ]
    
    for name, path in saved_files.items():
        if not path.exists():
            continue
            
        # Infer section from name
        name_lower = name.lower()
        section = "Other"
        for candidate in sections_order:
            if candidate.lower() in name_lower:
                section = candidate
                break
        
        # Add to report
        # MNE Report handles image paths directly
        # Clean title
        title = name.replace("_", " ").title()
        
        try:
            rep.add_figure(
                fig=path, 
                title=title,
                section=section,
                tags=(section.lower(),)
            )
        except Exception as e:
            logger.warning(f"Failed to add {name} to report: {e}")
            
    # 3. Save
    out_path = ctx.plots_dir.parent / f"sub-{ctx.subject}_feature_report.html"
    rep.save(out_path, overwrite=True, open_browser=False)
    
    logger.info(f"Report saved to: {out_path}")
    return out_path
