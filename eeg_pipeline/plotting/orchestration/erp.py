"""ERP visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for event-related potential (ERP) visualizations.
Plot primitives live in `eeg_pipeline.plotting.erp.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Any

from joblib import Parallel, delayed

from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.infra.paths import deriv_plots_path, ensure_dir, resolve_deriv_root
from eeg_pipeline.utils.parallel import get_n_jobs
from eeg_pipeline.utils.analysis.events import resolve_comparison_spec

from eeg_pipeline.plotting.erp.waveform import (
    plot_butterfly_erp,
    plot_roi_erp,
    plot_erp_contrast,
)
from eeg_pipeline.plotting.erp.topomaps import plot_erp_topomaps


def visualize_subject_erp(
    subject: str,
    task: str,
    config: Any,
    logger: logging.Logger,
    plots: Optional[List[str]] = None,
    deriv_root: Optional[Path] = None,
) -> None:
    """Visualize ERPs for a single subject."""
    logger.info(f"Visualizing ERP for sub-{subject}...")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    plots_dir = deriv_plots_path(effective_deriv_root, subject, subdir="erp")
    ensure_dir(plots_dir)

    # 1. Load Data
    epochs, events_df = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=True,
        deriv_root=effective_deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )

    if epochs is None:
        logger.error(f"Failed to load epochs for sub-{subject}")
        return

    # 2. Define Conditions for Contrast
    def _condition_key(label: str) -> str:
        key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
        if key in {"non_pain", "nopain", "no_pain"}:
            return "nonpain"
        return key or "condition"

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

    conditions = None
    contrast_spec = resolve_comparison_spec(events_df, config, require_enabled=False) if events_df is not None else None
    if contrast_spec is not None:
        col, v1, v2, label1, label2 = contrast_spec
        candidates = {
            _condition_key(label1): _query_for_value(col, v1),
            _condition_key(label2): _query_for_value(col, v2),
        }
        available_conditions = {}
        for name, query in candidates.items():
            try:
                if len(epochs[query]) > 0:
                    available_conditions[name] = query
            except Exception:
                continue
        conditions = available_conditions if available_conditions else None

    # 3. Execute Selected Plots
    plots_to_run = plots if plots is not None else [
        "butterfly",
        "roi",
        "contrast",
        "topomaps",
    ]

    if "butterfly" in plots_to_run:
        logger.info("Plotting butterfly ERPs...")
        plot_butterfly_erp(epochs, subject, plots_dir, config, logger, conditions=conditions)

    if "roi" in plots_to_run:
        logger.info("Plotting ROI ERPs...")
        plot_roi_erp(epochs, subject, plots_dir, config, logger, conditions=conditions)

    if "contrast" in plots_to_run and contrast_spec is not None:
        logger.info("Plotting ERP contrasts...")
        col, v1, v2, label1, label2 = contrast_spec
        plot_erp_contrast(
            epochs,
            subject,
            plots_dir,
            config,
            logger,
            cond_a=_query_for_value(col, v2),
            cond_b=_query_for_value(col, v1),
            label_a=label2,
            label_b=label1,
        )

    if "topomaps" in plots_to_run:
        logger.info("Plotting ERP topomaps...")
        plot_erp_topomaps(epochs, subject, plots_dir, config, logger, conditions=conditions)

    logger.info(f"ERP visualizations saved to {plots_dir}")


def _visualize_single_subject(
    subject: str,
    task: str,
    config: Any,
    plots: Optional[List[str]],
    deriv_root: Path,
) -> Optional[str]:
    """Worker function for parallel ERP visualization."""
    from eeg_pipeline.plotting.io.figures import setup_matplotlib
    from eeg_pipeline.infra.logging import get_logger

    setup_matplotlib(config)
    logger = get_logger(__name__)

    try:
        visualize_subject_erp(
            subject,
            task,
            config,
            logger,
            plots=plots,
            deriv_root=deriv_root,
        )
        return subject
    except Exception as e:
        logger.error(f"Failed to visualize sub-{subject}: {e}")
        return None


def visualize_erp_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
    plots: Optional[List[str]] = None,
    n_jobs: Optional[int] = None,
) -> None:
    """Batch process ERP visualizations for multiple subjects."""
    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        from eeg_pipeline.utils.config.loader import load_config
        config = load_config()

    from eeg_pipeline.plotting.io.figures import setup_matplotlib
    from eeg_pipeline.infra.logging import get_logger

    setup_matplotlib(config)
    task = task or config.get("project.task", "thermalactive")
    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    if n_jobs is None:
        n_jobs = get_n_jobs(config, config_path="erp_analysis.n_jobs")

    if n_jobs == 1 or len(subjects) == 1:
        logger.info(f"Starting ERP visualization: {len(subjects)} subject(s), task={task} [sequential]")
        for idx, subject in enumerate(subjects, 1):
            logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
            visualize_subject_erp(
                subject,
                task,
                config,
                logger,
                plots=plots,
                deriv_root=effective_deriv_root,
            )
    else:
        logger.info(f"Starting ERP visualization: {len(subjects)} subject(s), task={task} [parallel, n_jobs={n_jobs}]")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_visualize_single_subject)(
                subject,
                task,
                config,
                plots,
                effective_deriv_root,
            )
            for subject in subjects
        )
        successful = [r for r in results if r is not None]
        logger.info(f"Completed {len(successful)}/{len(subjects)} subjects")

    logger.info("ERP visualization complete")


__all__ = [
    "visualize_subject_erp",
    "visualize_erp_for_subjects",
]
