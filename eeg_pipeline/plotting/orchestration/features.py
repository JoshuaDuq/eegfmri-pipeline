"""Feature visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for descriptive feature visualizations.
Plot primitives and plot registries live in `eeg_pipeline.plotting.features.*`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.infra.paths import (
    deriv_features_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    resolve_deriv_root,
)
from eeg_pipeline.plotting.io.figures import setup_matplotlib

# Import plotters for side-effects: registers plot functions into VisualizationRegistry.
# This import must not call back into this module at import time.
from eeg_pipeline.plotting.features import registrations as _feature_plotters  # noqa: F401
from eeg_pipeline.plotting.features.context import (
    FeaturePlotContext,
    VisualizationManager,
    VisualizationRegistry,
)
from eeg_pipeline.domain.features.naming import NamingSchema


# Known feature types for manifest generation
# These must match the categories registered in VisualizationRegistry
_KNOWN_FEATURE_TYPES = [
    "aperiodic",
    "bursts",
    "complexity",
    "connectivity",
    "erds",
    "erp",
    "itpc",
    "pac",
    "power",
    "ratios",
    "spectral",
    "asymmetry",
    "microstates",
    "temporal",
]

_PLOT_FILE_EXTENSIONS = ["png", "svg", "pdf"]


def _parse_plotter_tokens(tokens: List[str]) -> Dict[str, Set[str]]:
    """Parse plotter tokens into category-to-names mapping.
    
    Tokens should be in format "category.name" (e.g., "power.spectral").
    Invalid tokens are silently skipped.
    
    Parameters
    ----------
    tokens : list of str
        Plotter tokens in "category.name" format.
        
    Returns
    -------
    dict mapping category names to sets of plotter names.
    """
    selected_by_category: Dict[str, Set[str]] = {}
    
    for token in tokens:
        token = str(token).strip()
        if "." not in token:
            continue
        
        category, _, name = token.partition(".")
        category = category.strip()
        name = name.strip()
        
        if category and name:
            selected_by_category.setdefault(category, set()).add(name)
    
    return selected_by_category


def _get_filtered_plotters(category: str, wanted_names: Set[str]) -> List[Tuple[str, Any]]:
    """Get plotters for a category filtered by wanted names.
    
    Parameters
    ----------
    category : str
        Category name.
    wanted_names : set of str
        Names of plotters to include.
        
    Returns
    -------
    list of (name, function) tuples for matching plotters.
    """
    all_plotters = VisualizationRegistry.get_plotters(category)
    return [(name, func) for name, func in all_plotters if name in wanted_names]


def _run_visualizations(
    manager: VisualizationManager,
    visualize_categories: Optional[List[str]],
    selected_by_category: Dict[str, Set[str]],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Run visualizations based on category and plotter selections.
    
    Parameters
    ----------
    manager : VisualizationManager
        Manager instance to run visualizations.
    visualize_categories : optional list of str
        Specific categories to visualize. If None, all categories are used.
    selected_by_category : dict
        Mapping of category names to sets of wanted plotter names.
    logger : logging.Logger
        Logger instance.
        
    Returns
    -------
    dict mapping plot names to file paths.
    """
    has_plotter_filter = bool(selected_by_category)
    
    if visualize_categories is not None:
        logger.info(f"Visualizing specific categories: {', '.join(visualize_categories)}")
        categories_to_run = visualize_categories
    elif has_plotter_filter:
        categories_to_run = list(selected_by_category.keys())
    else:
        return manager.run_all()
    
    for category in categories_to_run:
        if has_plotter_filter:
            wanted_names = selected_by_category.get(category)
            if not wanted_names:
                continue
            plotters = _get_filtered_plotters(category, wanted_names)
            manager.run_category(category, plotters=plotters)
        else:
            manager.run_category(category)
    
    return manager.saved_plots


def _create_plot_context(
    subject: str,
    deriv_root: Path,
    config: Optional[Any],
    logger: logging.Logger,
    epochs_info: Optional[Any],
    aligned_events: Optional[Any],
    epochs: Optional[Any],
    tfr: Optional[Any],
    plot_name_patterns: Optional[List[str]] = None,
) -> FeaturePlotContext:
    """Create and initialize feature plot context.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    deriv_root : Path
        Root directory for derivatives.
    config : optional
        Configuration object.
    logger : logging.Logger
        Logger instance.
    epochs_info : optional
        Epochs info object.
    aligned_events : optional
        Aligned events DataFrame.
    epochs : optional
        Epochs object.
    tfr : optional
        Time-frequency representation.
        
    Returns
    -------
    Initialized FeaturePlotContext.
    """
    features_dir = deriv_features_path(deriv_root, subject)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir="features")
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    
    context = FeaturePlotContext(
        subject=subject,
        plots_dir=plots_dir,
        features_dir=features_dir,
        config=config,
        logger=logger,
        plot_name_patterns=plot_name_patterns,
        stats_dir=stats_dir,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        tfr=tfr,
    )
    context.load_data()
    
    return context


def _validate_context_has_data(context: FeaturePlotContext, subject: str, logger: logging.Logger) -> bool:
    """Validate that context has sufficient data for visualization.
    
    Parameters
    ----------
    context : FeaturePlotContext
        Context to validate.
    subject : str
        Subject identifier for logging.
    logger : logging.Logger
        Logger instance.
        
    Returns
    -------
    True if context has sufficient data, False otherwise.
    """
    has_feature_data = context.power_df is not None or context.connectivity_df is not None
    
    if not has_feature_data:
        logger.warning(f"No feature data found for subject {subject}")
        if context.epochs is None:
            logger.warning("No epochs data found either")
            return False
    
    return True


def visualize_features(
    subject: str,
    deriv_root: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    epochs_info: Optional[Any] = None,
    aligned_events: Optional[Any] = None,
    epochs: Optional[Any] = None,
    tfr: Optional[Any] = None,
    visualize_categories: Optional[List[str]] = None,
    feature_plotters: Optional[List[str]] = None,
    plot_name_patterns: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Generate descriptive feature visualizations using registered plotters.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    deriv_root : Path
        Root directory for derivatives.
    config : optional
        Configuration object.
    logger : optional logging.Logger
        Logger instance. If None, creates a new logger.
    epochs_info : optional
        Epochs info object.
    aligned_events : optional
        Aligned events DataFrame.
    epochs : optional
        Epochs object.
    tfr : optional
        Time-frequency representation.
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        If None, all registered categories are visualized.
    feature_plotters : optional list of str
        Specific plotters to use in "category.name" format.
        If None, all plotters for selected categories are used.
        
    Returns
    -------
    dict mapping plot names to file paths.
    """
    if logger is None:
        logger = get_logger(__name__)
    
    context = _create_plot_context(
        subject=subject,
        deriv_root=deriv_root,
        config=config,
        logger=logger,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        tfr=tfr,
        plot_name_patterns=plot_name_patterns,
    )
    
    if not _validate_context_has_data(context, subject, logger):
        return {}
    
    manager = VisualizationManager(context)
    
    selected_by_category = _parse_plotter_tokens(feature_plotters) if feature_plotters else {}
    
    saved_plots = _run_visualizations(
        manager=manager,
        visualize_categories=visualize_categories,
        selected_by_category=selected_by_category,
        logger=logger,
    )
    
    _save_plot_manifest(
        plots_dir=context.plots_dir,
        subject=subject,
        logger=logger,
    )
    
    return saved_plots


def _detect_feature_type_from_path(path: Path) -> str:
    """Detect feature type from plot file path.
    
    Parameters
    ----------
    path : Path
        Path to plot file.
        
    Returns
    -------
    Feature type string, or "unknown" if not detected.
    """
    for feature_type in _KNOWN_FEATURE_TYPES:
        if feature_type in path.parts:
            return feature_type
    return "unknown"


def _collect_plot_files(plots_dir: Path) -> List[Path]:
    """Collect all plot files from directory.
    
    Parameters
    ----------
    plots_dir : Path
        Directory containing plot files.
        
    Returns
    -------
    Sorted list of plot file paths.
    """
    plot_files = [
        path
        for extension in _PLOT_FILE_EXTENSIONS
        for path in plots_dir.rglob(f"*.{extension}")
    ]
    return sorted(plot_files)


def _create_plot_entry(path: Path, plots_dir: Path) -> Dict[str, str]:
    """Create manifest entry for a single plot file.
    
    Parameters
    ----------
    path : Path
        Path to plot file.
    plots_dir : Path
        Base plots directory for relative path calculation.
        
    Returns
    -------
    Dictionary with plot metadata.
    """
    return {
        "name": path.stem,
        "path": str(path.relative_to(plots_dir)),
        "feature_type": _detect_feature_type_from_path(path),
        "format": path.suffix[1:],
    }


def _save_plot_manifest(
    plots_dir: Path,
    subject: str,
    logger: logging.Logger,
) -> None:
    """Save plot manifest JSON file listing all generated plots.
    
    Parameters
    ----------
    plots_dir : Path
        Directory containing plot files.
    subject : str
        Subject identifier.
    logger : logging.Logger
        Logger instance.
    """
    plot_files = _collect_plot_files(plots_dir)
    
    manifest = {
        "subject": subject,
        "generated_at": datetime.now().isoformat(),
        "plots_directory": str(plots_dir),
        "total_plots": len(plot_files),
        "plots": [_create_plot_entry(path, plots_dir) for path in plot_files],
    }
    
    manifest_path = plots_dir / f"sub-{subject}_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    logger.info(f"Saved plot manifest ({len(plot_files)} plots)")


def _load_features_power_df(
    *,
    features_dir: Path,
    read_table: Any,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Load the per-subject power feature table(s) with deterministic logic."""
    if not features_dir.exists():
        return None

    search_dirs = [features_dir / "power", features_dir]
    search_dirs = [d for d in search_dirs if d.is_dir()]
    if not search_dirs:
        return None

    exts = [".parquet", ".tsv"]

    suffix_paths: List[Path] = []
    for d in search_dirs:
        for ext in exts:
            suffix_paths.extend(sorted(d.glob(f"features_power_*{ext}")))
    suffix_paths = [p for p in suffix_paths if p.stem != "features_power"]

    paths: List[Path] = []
    if suffix_paths:
        seen: Set[Path] = set()
        for p in suffix_paths:
            if p not in seen:
                seen.add(p)
                paths.append(p)
    else:
        for d in search_dirs:
            for ext in exts:
                p = d / f"features_power{ext}"
                if p.exists():
                    paths = [p]
                    break
            if paths:
                break

    if not paths:
        return None

    frames: List[pd.DataFrame] = []
    base_len: Optional[int] = None

    for path in paths:
        try:
            df = read_table(path)
        except Exception as exc:  # pragma: no cover - best-effort file read
            logger.warning("Failed to read %s: %s", path, exc)
            continue
        if df is None or df.empty:
            continue
        df = df.reset_index(drop=True)
        if base_len is None:
            base_len = len(df)
        elif len(df) != base_len:
            logger.warning(
                "Skipping %s (rows=%d) due to length mismatch (expected %d)",
                path.name,
                len(df),
                base_len,
            )
            continue
        frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, axis=1)
    if combined.columns.duplicated().any():
        combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def _load_config_if_needed(config: Any) -> Any:
    """Load configuration if not provided.
    
    Parameters
    ----------
    config : optional
        Configuration object. If None, loads default config.
        
    Returns
    -------
    Configuration object.
    """
    if config is not None:
        return config
    from eeg_pipeline.utils.config.loader import load_config
    return load_config()


def _resolve_task(task: Optional[str], config: Any) -> str:
    """Resolve task name from parameter or config.
    
    Parameters
    ----------
    task : optional str
        Task name. If None, uses config default.
    config : Any
        Configuration object.
        
    Returns
    -------
    Task name string.
    """
    if task is not None:
        return task
    return config.get("project.task", "thermalactive")


def _log_visualization_start(
    subjects: List[str],
    task: str,
    visualize_categories: Optional[List[str]],
    logger: logging.Logger,
) -> None:
    """Log start of visualization process.
    
    Parameters
    ----------
    subjects : list of str
        Subject identifiers.
    task : str
        Task name.
    visualize_categories : optional list of str
        Categories being visualized.
    logger : logging.Logger
        Logger instance.
    """
    category_suffix = f" ({', '.join(visualize_categories)})" if visualize_categories else ""
    logger.info(
        f"Starting feature visualization{category_suffix}: "
        f"{len(subjects)} subject(s), task={task}"
    )


def _visualize_single_subject(
    subject: str,
    task: str,
    effective_deriv_root: Path,
    config: Any,
    logger: logging.Logger,
    visualize_categories: Optional[List[str]],
    feature_plotters: Optional[List[str]],
    plot_name_patterns: Optional[List[str]],
    subject_index: int,
    total_subjects: int,
) -> None:
    """Visualize features for a single subject.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    task : str
        Task name.
    effective_deriv_root : Path
        Resolved derivatives root directory.
    config : Any
        Configuration object.
    logger : logging.Logger
        Logger instance.
    visualize_categories : optional list of str
        Categories to visualize.
    feature_plotters : optional list of str
        Specific plotters to use.
    subject_index : int
        Current subject index (1-based).
    total_subjects : int
        Total number of subjects.
    """
    logger.info(f"[{subject_index}/{total_subjects}] Visualizing sub-{subject}")
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject=subject,
        task=task,
        align="strict",
        preload=False,
        deriv_root=effective_deriv_root,
        config=config,
        logger=logger,
    )
    
    epochs_info = epochs.info if epochs else None
    
    visualize_features(
        subject=subject,
        deriv_root=effective_deriv_root,
        config=config,
        logger=logger,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        visualize_categories=visualize_categories,
        feature_plotters=feature_plotters,
        plot_name_patterns=plot_name_patterns,
    )


def visualize_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
    visualize_categories: Optional[List[str]] = None,
    feature_plotters: Optional[List[str]] = None,
    plot_name_patterns: Optional[List[str]] = None,
) -> None:
    """Visualize features for multiple subjects.
    
    Parameters
    ----------
    subjects : list of str
        Subject identifiers.
    task : optional str
        Task name. If None, uses config default.
    deriv_root : optional Path
        Root directory for derivatives. If None, resolves from config.
    config : optional
        Configuration object. If None, loads default config.
    logger : optional logging.Logger
        Logger instance. If None, creates a new logger.
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        If None, all registered categories are visualized.
    feature_plotters : optional list of str
        Specific plotters to use in "category.name" format.
        If None, all plotters for selected categories are used.
    """
    if not subjects:
        raise ValueError("No subjects specified")
    
    config = _load_config_if_needed(config)
    setup_matplotlib(config)
    
    task = _resolve_task(task, config)
    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    
    if logger is None:
        logger = get_logger(__name__)
    
    _log_visualization_start(subjects, task, visualize_categories, logger)
    
    for index, subject in enumerate(subjects, 1):
        _visualize_single_subject(
            subject=subject,
            task=task,
            effective_deriv_root=effective_deriv_root,
            config=config,
            logger=logger,
            visualize_categories=visualize_categories,
            feature_plotters=feature_plotters,
            plot_name_patterns=plot_name_patterns,
            subject_index=index,
            total_subjects=len(subjects),
        )
    
    logger.info("Feature visualization complete")


def visualize_band_power_topomaps_for_group(
    *,
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Compute group-aggregate band power topomaps.

    Aggregation is equal-weight per subject: for each subject, compute a mean
    feature vector across trials, then average those vectors across subjects.
    Outputs are written under `sub-group/eeg/plots/features/power/`.
    """
    if not subjects:
        raise ValueError("No subjects specified")

    config = _load_config_if_needed(config)
    setup_matplotlib(config)
    task = _resolve_task(task, config)
    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    from eeg_pipeline.infra.tsv import read_table
    from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names
    from eeg_pipeline.plotting.features.power import (
        plot_band_power_topomaps,
        plot_band_power_topomaps_window_contrast,
        plot_band_power_topomaps_group_condition_contrast,
    )
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask

    # Window(s) to plot must match NamingSchema "segment" tokens in features_power columns.
    topomap_windows = get_config_value(config, "plotting.plots.features.power.topomap_windows", None)
    if topomap_windows:
        if isinstance(topomap_windows, str):
            windows_list = [w.strip() for w in topomap_windows.split() if w.strip()]
        elif isinstance(topomap_windows, list):
            windows_list = [str(w).strip() for w in topomap_windows if str(w).strip()]
        else:
            windows_list = []
    else:
        windows_list = []

    if not windows_list:
        raise ValueError(
            "band_power_topomaps requires plotting.plots.features.power.topomap_windows to be set."
        )

    bands = get_frequency_band_names(config)
    if not bands:
        raise ValueError("No frequency bands resolved for plotting.")

    per_subject_means: List[pd.Series] = []
    per_subject_cond1_means: List[pd.Series] = []
    per_subject_cond2_means: List[pd.Series] = []
    infos: List[Any] = []
    cond_label1: Optional[str] = None
    cond_label2: Optional[str] = None

    for subject in subjects:
        features_dir = deriv_features_path(effective_deriv_root, subject)
        power_df = _load_features_power_df(features_dir=features_dir, read_table=read_table, logger=logger)
        if power_df is None or power_df.empty:
            logger.warning("Group topomaps: missing power features for sub-%s; skipping", subject)
            continue

        epochs, events_df = load_epochs_for_analysis(
            subject=subject,
            task=task,
            align="strict",
            preload=False,
            deriv_root=effective_deriv_root,
            config=config,
            logger=logger,
        )
        if epochs is None:
            logger.warning("Group topomaps: missing epochs for sub-%s; skipping", subject)
            continue

        subject_mean = power_df.mean(numeric_only=True)
        if subject_mean.empty:
            logger.warning("Group topomaps: no numeric power columns for sub-%s; skipping", subject)
            continue

        per_subject_means.append(subject_mean)
        infos.append(epochs.info)

        # Optional: group-level condition comparison (paired across subjects).
        # We compute per-subject means within each condition and then run paired tests
        # across subjects for the contrast maps.
        compare_columns = bool(get_config_value(config, "plotting.comparisons.compare_columns", False))
        if compare_columns:
            if events_df is None or events_df.empty:
                logger.warning(
                    "Group topomaps: compare_columns requested but events_df is missing for sub-%s; skipping condition comparison for this subject",
                    subject,
                )
            elif len(events_df) != len(power_df):
                logger.warning(
                    "Group topomaps: compare_columns requested but length mismatch for sub-%s (events=%d, power=%d); skipping condition comparison for this subject",
                    subject,
                    len(events_df),
                    len(power_df),
                )
            else:
                comp = extract_comparison_mask(events_df, config, require_enabled=False)
                if comp is None:
                    logger.warning(
                        "Group topomaps: compare_columns requested but could not resolve comparison masks for sub-%s; skipping condition comparison for this subject",
                        subject,
                    )
                else:
                    mask1, mask2, label1, label2 = comp
                    if cond_label1 is None:
                        cond_label1 = label1
                        cond_label2 = label2
                    n = len(power_df)
                    m1 = np.asarray(mask1[:n], dtype=bool)
                    m2 = np.asarray(mask2[:n], dtype=bool)
                    if int(m1.sum()) == 0 or int(m2.sum()) == 0:
                        logger.warning(
                            "Group topomaps: sub-%s has no trials for one or both conditions (%s=%d, %s=%d); skipping condition comparison for this subject",
                            subject,
                            label1,
                            int(m1.sum()),
                            label2,
                            int(m2.sum()),
                        )
                    else:
                        cond1_mean = power_df[m1].mean(numeric_only=True)
                        cond2_mean = power_df[m2].mean(numeric_only=True)
                        if cond1_mean.empty or cond2_mean.empty:
                            logger.warning(
                                "Group topomaps: sub-%s condition means empty; skipping condition comparison for this subject",
                                subject,
                            )
                        else:
                            per_subject_cond1_means.append(cond1_mean)
                            per_subject_cond2_means.append(cond2_mean)

    if len(per_subject_means) < 2:
        raise ValueError("Group topomaps require at least 2 valid subjects with epochs and power features.")

    # Choose a reference montage/channel set from the first valid subject.
    # We intentionally do NOT require strict channel intersection across subjects,
    # because that can drop otherwise-valid electrodes and change the topomap.
    ref_info = infos[0]
    allowed_channels: Set[str] = set(ref_info.ch_names)

    union_cols: Set[str] = set()
    channels_in_cols: Set[str] = set()
    for mean_series in per_subject_means:
        for col in mean_series.index:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("scope") != "ch":
                continue
            if parsed.get("segment") not in windows_list:
                continue
            if parsed.get("band") not in bands:
                continue
            ch = parsed.get("identifier")
            if ch not in allowed_channels:
                continue
            union_cols.add(str(col))
            if ch:
                channels_in_cols.add(str(ch))

    if not union_cols:
        raise ValueError("Group topomaps: no band×channel power features found across subjects.")

    ordered_cols = sorted(union_cols)
    group_df = pd.DataFrame([s.reindex(ordered_cols) for s in per_subject_means], columns=ordered_cols)

    # Drop columns that are entirely missing across subjects.
    all_nan_cols = [c for c in group_df.columns if group_df[c].isna().all()]
    if all_nan_cols:
        logger.warning(
            "Group topomaps: dropping %d columns with no data across subjects.",
            len(all_nan_cols),
        )
        group_df = group_df.drop(columns=all_nan_cols)

    if group_df.empty or group_df.shape[1] == 0:
        raise ValueError("Group topomaps: no usable columns remain after alignment.")

    # Build a reference Info restricted to the channels that appear in the selected columns.
    import mne

    ordered_channels = [ch for ch in ref_info.ch_names if ch in channels_in_cols]
    if not ordered_channels:
        raise ValueError("Group topomaps: no channels available after alignment to reference montage.")
    picks = mne.pick_channels(ref_info.ch_names, include=ordered_channels, ordered=True)
    ref_info_common = mne.pick_info(ref_info, picks)

    plots_dir = deriv_plots_path(effective_deriv_root, "group", subdir="features")
    power_plots_dir = plots_dir / "power"
    ensure_dir(power_plots_dir)

    for window in windows_list:
        plot_band_power_topomaps(
            pow_df=group_df,
            epochs_info=ref_info_common,
            bands=bands,
            subject="group",
            save_dir=power_plots_dir,
            logger=logger,
            config=config,
            segment=window,
            events_df=None,
            sample_unit="subjects",
        )

    # Group-level condition comparison (means per condition + paired contrast).
    compare_columns = bool(get_config_value(config, "plotting.comparisons.compare_columns", False))
    if (
        compare_columns
        and cond_label1 is not None
        and cond_label2 is not None
        and len(per_subject_cond1_means) >= 2
        and len(per_subject_cond2_means) >= 2
        and len(per_subject_cond1_means) == len(per_subject_cond2_means)
    ):
        # Reuse the same feature selection logic as the primary group plot, but based on the
        # condition-specific per-subject summaries.
        union_cols_cond: Set[str] = set()
        channels_in_cols_cond: Set[str] = set()
        for mean_series in list(per_subject_cond1_means) + list(per_subject_cond2_means):
            for col in mean_series.index:
                parsed = NamingSchema.parse(str(col))
                if not parsed.get("valid"):
                    continue
                if parsed.get("group") != "power":
                    continue
                if parsed.get("scope") != "ch":
                    continue
                if parsed.get("segment") not in windows_list:
                    continue
                if parsed.get("band") not in bands:
                    continue
                ch = parsed.get("identifier")
                if ch not in allowed_channels:
                    continue
                union_cols_cond.add(str(col))
                if ch:
                    channels_in_cols_cond.add(str(ch))

        if union_cols_cond:
            ordered_cols_cond = sorted(union_cols_cond)
            cond1_df = pd.DataFrame(
                [s.reindex(ordered_cols_cond) for s in per_subject_cond1_means],
                columns=ordered_cols_cond,
            )
            cond2_df = pd.DataFrame(
                [s.reindex(ordered_cols_cond) for s in per_subject_cond2_means],
                columns=ordered_cols_cond,
            )

            # Drop columns that are entirely missing across both conditions.
            combined = pd.concat([cond1_df, cond2_df], axis=0, ignore_index=True)
            all_nan_cols = [c for c in ordered_cols_cond if c in combined.columns and combined[c].isna().all()]
            if all_nan_cols:
                cond1_df = cond1_df.drop(columns=all_nan_cols)
                cond2_df = cond2_df.drop(columns=all_nan_cols)

            for window in windows_list:
                plot_band_power_topomaps(
                    pow_df=cond1_df,
                    epochs_info=ref_info_common,
                    bands=bands,
                    subject="group",
                    save_dir=power_plots_dir,
                    logger=logger,
                    config=config,
                    segment=window,
                    events_df=None,
                    sample_unit="subjects",
                    label_suffix=cond_label1,
                )
                plot_band_power_topomaps(
                    pow_df=cond2_df,
                    epochs_info=ref_info_common,
                    bands=bands,
                    subject="group",
                    save_dir=power_plots_dir,
                    logger=logger,
                    config=config,
                    segment=window,
                    events_df=None,
                    sample_unit="subjects",
                    label_suffix=cond_label2,
                )
                plot_band_power_topomaps_group_condition_contrast(
                    pow_df_condition1=cond1_df,
                    pow_df_condition2=cond2_df,
                    epochs_info=ref_info_common,
                    bands=bands,
                    subject="group",
                    save_dir=power_plots_dir,
                    logger=logger,
                    config=config,
                    segment=window,
                    label1=cond_label1,
                    label2=cond_label2,
                    sample_unit="subjects",
                )
        else:
            logger.warning("Group topomaps: compare_columns requested but no usable condition columns were found.")
    elif compare_columns:
        logger.warning(
            "Group topomaps: compare_columns requested but insufficient paired subjects were available (cond1=%d, cond2=%d).",
            len(per_subject_cond1_means),
            len(per_subject_cond2_means),
        )

    compare_windows = bool(get_config_value(config, "plotting.comparisons.compare_windows", True))
    if compare_windows and len(windows_list) == 2:
        window1, window2 = windows_list[0], windows_list[1]
        plot_band_power_topomaps_window_contrast(
            pow_df=group_df,
            epochs_info=ref_info_common,
            bands=bands,
            subject="group",
            save_dir=power_plots_dir,
            logger=logger,
            config=config,
            window1=window1,
            window2=window2,
        )

    _save_plot_manifest(plots_dir=plots_dir, subject="group", logger=logger)


__all__ = [
    "visualize_features",
    "visualize_features_for_subjects",
    "visualize_band_power_topomaps_for_group",
]
