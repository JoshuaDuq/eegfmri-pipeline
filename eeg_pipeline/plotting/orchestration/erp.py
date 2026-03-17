"""ERP visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for event-related potential (ERP) visualizations.
Plot primitives live in `eeg_pipeline.plotting.erp.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

import pandas as pd
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


def _normalize_condition_key(label: str) -> str:
    """Normalize condition label to a valid key format."""
    key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    return key or "condition"


def _build_epoch_query(column: str, value: Any) -> str:
    """Build an epoch query string for a column and value.
    
    Parameters
    ----------
    column : str
        Column name to query.
    value : Any
        Value to match. Handles numeric and string values.
    
    Returns
    -------
    str
        Query string in format `column` == value.
    """
    column_expr = f"`{column}`"
    numeric_value = pd.to_numeric(str(value), errors="coerce")
    if pd.notna(numeric_value):
        float_value = float(numeric_value)
        if float_value.is_integer():
            return f"{column_expr} == {int(float_value)}"
        return f"{column_expr} == {float_value}"
    return f"{column_expr} == {repr(str(value))}"


def _resolve_available_conditions(
    epochs: Any,
    events_df: Optional[pd.DataFrame],
    config: Any,
    logger: logging.Logger,
) -> Optional[Dict[str, str]]:
    """Resolve available ERP conditions from comparison specification.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object to query.
    events_df : pd.DataFrame, optional
        Events dataframe. If None, returns None.
    config : Any
        Configuration object.
    logger : logging.Logger
        Logger instance for warnings and debug messages.
    
    Returns
    -------
    dict[str, str] or None
        Dictionary mapping condition names to query strings, or None if no
        valid conditions found.
    """
    if events_df is None:
        return None
    
    contrast_spec = resolve_comparison_spec(
        events_df, config, require_enabled=False
    )
    if contrast_spec is None:
        return None
    
    column, value1, value2, label1, label2 = contrast_spec
    
    if column not in events_df.columns:
        logger.error(
            f"Comparison column '{column}' not found in events_df. "
            f"Available: {list(events_df.columns)}"
        )
        return None
    
    condition_queries = {
        _normalize_condition_key(label1): _build_epoch_query(column, value1),
        _normalize_condition_key(label2): _build_epoch_query(column, value2),
    }
    
    metadata = getattr(epochs, "metadata", None)
    column_in_metadata = metadata is not None and column in metadata.columns
    
    if not column_in_metadata:
        if metadata is None:
            metadata = pd.DataFrame(index=range(len(epochs)))
        metadata[column] = events_df[column].values
        epochs.metadata = metadata
    
    available_conditions = {}
    for condition_name, query in condition_queries.items():
        try:
            matched_epochs = epochs[query]
            if len(matched_epochs) > 0:
                available_conditions[condition_name] = query
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Condition '{condition_name}' query '{query}' failed: {e}")
            continue
    
    if not available_conditions:
        logger.error(
            f"No valid conditions found. Queries: {list(condition_queries.values())}"
        )
        return None
    
    return available_conditions


def _execute_erp_plots(
    epochs: Any,
    subject: str,
    plots_dir: Path,
    config: Any,
    logger: logging.Logger,
    plots_to_run: List[str],
    conditions: Optional[Dict[str, str]],
    contrast_spec: Optional[tuple],
) -> None:
    """Execute selected ERP plot functions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object to plot.
    subject : str
        Subject identifier.
    plots_dir : Path
        Directory to save plots.
    config : Any
        Configuration object.
    logger : logging.Logger
        Logger instance.
    plots_to_run : list[str]
        List of plot names to execute.
    conditions : dict[str, str] or None
        Condition name to query mapping.
    contrast_spec : tuple or None
        Contrast specification (column, value1, value2, label1, label2).
    """
    if "butterfly" in plots_to_run:
        logger.info("Plotting butterfly ERPs...")
        plot_butterfly_erp(
            epochs, subject, plots_dir, config, logger, conditions=conditions
        )
    
    if "roi" in plots_to_run:
        logger.info("Plotting ROI ERPs...")
        plot_roi_erp(
            epochs, subject, plots_dir, config, logger, conditions=conditions
        )
    
    if "contrast" in plots_to_run and contrast_spec is not None:
        if conditions is None or len(conditions) < 2:
            logger.warning(
                "Cannot plot ERP contrast: insufficient validated conditions. "
                f"Required: 2, Available: {len(conditions) if conditions else 0}"
            )
            return
        
        logger.info("Plotting ERP contrasts...")
        column, value1, value2, label1, label2 = contrast_spec
        
        # Map contrast spec labels to normalized condition keys
        norm_label1 = _normalize_condition_key(label1)
        norm_label2 = _normalize_condition_key(label2)
        
        # Use validated queries from conditions dict
        if norm_label1 not in conditions or norm_label2 not in conditions:
            logger.warning(
                f"Cannot plot ERP contrast: conditions not found in validated set. "
                f"Required: {norm_label1}, {norm_label2}. "
                f"Available: {list(conditions.keys())}"
            )
            return
        
        plot_erp_contrast(
            epochs,
            subject,
            plots_dir,
            config,
            logger,
            cond_a=conditions[norm_label2],
            cond_b=conditions[norm_label1],
            label_a=label2,
            label_b=label1,
        )


def visualize_subject_erp(
    subject: str,
    task: str,
    config: Any,
    logger: logging.Logger,
    plots: Optional[List[str]] = None,
    deriv_root: Optional[Path] = None,
) -> None:
    """Visualize ERPs for a single subject.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    task : str
        Task name.
    config : Any
        Configuration object.
    logger : logging.Logger
        Logger instance.
    plots : list[str], optional
        List of plot names to generate. If None, generates all plots.
    deriv_root : Path, optional
        Derived data root directory. If None, resolved from config.
    """
    logger.info(f"Visualizing ERP for sub-{subject}...")
    
    effective_deriv_root = resolve_deriv_root(
        deriv_root=deriv_root, config=config
    )
    plots_dir = deriv_plots_path(effective_deriv_root, subject, subdir="erp")
    ensure_dir(plots_dir)
    
    epochs, events_df = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=True,
        deriv_root=effective_deriv_root,
        config=config,
        logger=logger,
    )
    
    if epochs is None:
        logger.error(f"Failed to load epochs for sub-{subject}")
        return
    
    conditions = _resolve_available_conditions(epochs, events_df, config, logger)
    contrast_spec = (
        resolve_comparison_spec(events_df, config, require_enabled=False)
        if events_df is not None
        else None
    )
    
    plots_to_run = plots or ["butterfly", "roi", "contrast"]
    
    _execute_erp_plots(
        epochs,
        subject,
        plots_dir,
        config,
        logger,
        plots_to_run,
        conditions,
        contrast_spec,
    )
    
    logger.info(f"ERP visualizations saved to {plots_dir}")


def _visualize_single_subject(
    subject: str,
    task: str,
    config: Any,
    plots: Optional[List[str]],
    deriv_root: Path,
) -> Optional[str]:
    """Worker function for parallel ERP visualization.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    task : str
        Task name.
    config : Any
        Configuration object.
    plots : list[str], optional
        List of plot names to generate.
    deriv_root : Path
        Derived data root directory.
    
    Returns
    -------
    str or None
        Subject identifier if successful, None otherwise.
    """
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


def _process_subjects_sequentially(
    subjects: List[str],
    task: str,
    config: Any,
    logger: logging.Logger,
    plots: Optional[List[str]],
    deriv_root: Path,
) -> None:
    """Process subjects sequentially."""
    logger.info(
        f"Starting ERP visualization: {len(subjects)} subject(s), "
        f"task={task} [sequential]"
    )
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        visualize_subject_erp(
            subject,
            task,
            config,
            logger,
            plots=plots,
            deriv_root=deriv_root,
        )


def _process_subjects_parallel(
    subjects: List[str],
    task: str,
    config: Any,
    logger: logging.Logger,
    plots: Optional[List[str]],
    deriv_root: Path,
    n_jobs: int,
) -> None:
    """Process subjects in parallel."""
    logger.info(
        f"Starting ERP visualization: {len(subjects)} subject(s), "
        f"task={task} [parallel, n_jobs={n_jobs}]"
    )
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_visualize_single_subject)(
            subject,
            task,
            config,
            plots,
            deriv_root,
        )
        for subject in subjects
    )
    successful = [result for result in results if result is not None]
    logger.info(f"Completed {len(successful)}/{len(subjects)} subjects")


def visualize_erp_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
    plots: Optional[List[str]] = None,
    n_jobs: Optional[int] = None,
) -> None:
    """Batch process ERP visualizations for multiple subjects.
    
    Parameters
    ----------
    subjects : list[str]
        List of subject identifiers to process.
    task : str, optional
        Task name. If None, resolved from config.
    deriv_root : Path, optional
        Derived data root directory. If None, resolved from config.
    config : Any, optional
        Configuration object. If None, loaded from default location.
    logger : logging.Logger, optional
        Logger instance. If None, created for this module.
    plots : list[str], optional
        List of plot names to generate.
    n_jobs : int, optional
        Number of parallel jobs. If None, resolved from config.
    """
    if not subjects:
        raise ValueError("No subjects specified")
    
    if config is None:
        from eeg_pipeline.utils.config.loader import load_config
        config = load_config()
    
    from eeg_pipeline.plotting.io.figures import setup_matplotlib
    from eeg_pipeline.infra.logging import get_logger
    
    setup_matplotlib(config)
    task = task or config.get("project.task", "task")
    effective_deriv_root = resolve_deriv_root(
        deriv_root=deriv_root, config=config
    )
    
    logger = logger or get_logger(__name__)
    n_jobs = n_jobs or get_n_jobs(config, config_path="erp_analysis.n_jobs")
    
    use_parallel = n_jobs > 1 and len(subjects) > 1
    if use_parallel:
        _process_subjects_parallel(
            subjects, task, config, logger, plots, effective_deriv_root, n_jobs
        )
    else:
        _process_subjects_sequentially(
            subjects, task, config, logger, plots, effective_deriv_root
        )
    
    logger.info("ERP visualization complete")


__all__ = [
    "visualize_subject_erp",
    "visualize_erp_for_subjects",
]
