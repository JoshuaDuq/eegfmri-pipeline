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
    feature_categories_to_load: Optional[Set[str]] = None,
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
        feature_categories_to_load=feature_categories_to_load,
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
    has_feature_data = any(
        df is not None
        for df in (
            context.power_df,
            context.connectivity_df,
            context.aperiodic_df,
            context.erp_df,
            context.erds_df,
            context.bursts_df,
            context.quality_df,
            context.spectral_df,
            context.ratios_df,
            context.asymmetry_df,
            context.microstates_df,
            context.pac_df,
            context.complexity_df,
            context.itpc_df,
            context.temporal_df,
            context.sourcelocalization_df,
        )
    )
    has_source_estimates = (
        any(context.features_dir.glob("sourcelocalization/*/source_estimates/*.stc"))
        or any(context.features_dir.glob("sourcelocalization/*/source_estimates/*/*.stc"))
    )
    
    if not has_feature_data and not has_source_estimates:
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
    
    selected_by_category = _parse_plotter_tokens(feature_plotters) if feature_plotters else {}

    categories_to_load: Optional[Set[str]] = None
    if visualize_categories is not None:
        categories_to_load = {str(cat).strip() for cat in visualize_categories if str(cat).strip()}
    elif selected_by_category:
        categories_to_load = set(selected_by_category.keys())

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
        feature_categories_to_load=categories_to_load,
    )
    
    if not _validate_context_has_data(context, subject, logger):
        return {}
    
    manager = VisualizationManager(context)
    
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
    return config.get("project.task", "task")


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


def _resolve_power_roi_names(
    *,
    config: Any,
    rois: Dict[str, Any],
) -> List[str]:
    """Resolve ROI names for power plotting from comparison config."""
    from eeg_pipeline.utils.config.loader import get_config_value

    configured_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if isinstance(configured_rois, (list, tuple)) and configured_rois:
        roi_names: List[str] = []
        for roi_name in configured_rois:
            roi_name = str(roi_name).strip()
            if not roi_name:
                continue
            if roi_name.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
                continue
            if roi_name in rois:
                roi_names.append(roi_name)
        return roi_names

    if not rois:
        return ["all"]
    return ["all", *list(rois.keys())]


def _extract_power_channels(columns: List[Any]) -> List[str]:
    """Extract channel identifiers from NamingSchema power channel columns."""
    channels: Set[str] = set()
    for col in columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "power":
            continue
        if parsed.get("scope") != "ch":
            continue
        identifier = str(parsed.get("identifier") or "").strip()
        if identifier:
            channels.add(identifier)
    return sorted(channels)


def _select_power_columns(
    power_df: pd.DataFrame,
    *,
    segment: str,
    band: str,
    roi_channels: Optional[List[str]] = None,
) -> List[str]:
    """Select power columns matching a segment, band, and optional ROI channel set."""
    channel_set = set(roi_channels) if roi_channels is not None else None
    columns: List[str] = []

    for col in power_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "power":
            continue
        if str(parsed.get("segment") or "") != str(segment):
            continue
        if str(parsed.get("band") or "") != str(band):
            continue
        identifier = str(parsed.get("identifier") or "")
        if channel_set is not None and identifier and identifier not in channel_set:
            continue
        columns.append(str(col))

    return columns


def _resolve_group_conditions(
    *,
    events_df: pd.DataFrame,
    config: Any,
    allow_single_value: bool,
) -> List[Tuple[str, np.ndarray]]:
    """Resolve configured comparison conditions into (label, mask) pairs."""
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.utils.config.loader import get_config_value, require_config_value

    comparison_column = str(require_config_value(config, "plotting.comparisons.comparison_column")).strip()
    if comparison_column == "":
        raise ValueError("plotting.comparisons.comparison_column must be a non-empty string.")
    if comparison_column not in events_df.columns:
        raise ValueError(
            f"Comparison column '{comparison_column}' not found in events DataFrame."
        )

    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)

    if not isinstance(values_spec, (list, tuple)):
        raise ValueError(
            "plotting.comparisons.comparison_values must be a list/tuple."
        )

    if allow_single_value:
        if len(values_spec) < 1:
            raise ValueError(
                "At least one comparison value is required in plotting.comparisons.comparison_values."
            )
    elif len(values_spec) < 2:
        raise ValueError(
            "At least two comparison values are required for condition comparisons."
        )

    if len(values_spec) == 1:
        if not allow_single_value:
            raise ValueError("Single-value condition selection is not allowed for this plot.")

        value = values_spec[0]
        if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 1 and str(labels_spec[0]).strip():
            label = str(labels_spec[0]).strip()
        else:
            label = str(value)

        column_values = events_df[comparison_column]
        try:
            numeric_value = float(value)
            mask = (pd.to_numeric(column_values, errors="coerce") == numeric_value).values
        except (ValueError, TypeError):
            value_str = str(value).strip().lower()
            mask = (column_values.astype(str).str.strip().str.lower() == value_str).values

        mask_bool = np.asarray(mask, dtype=bool)
        if int(mask_bool.sum()) == 0:
            raise ValueError(
                f"No events matched comparison value {value!r} in column {comparison_column!r}."
            )
        return [(label, mask_bool)]

    if len(values_spec) == 2:
        comparison = extract_comparison_mask(events_df, config, require_enabled=True)
        if comparison is None:
            raise ValueError("Could not resolve two-condition comparison masks from events.")
        mask1, mask2, label1, label2 = comparison
        mask1 = np.asarray(mask1, dtype=bool)
        mask2 = np.asarray(mask2, dtype=bool)
        return [(str(label1), mask1), (str(label2), mask2)]

    multi_group = extract_multi_group_masks(events_df, config, require_enabled=True)
    if multi_group is None:
        raise ValueError("Could not resolve multi-condition masks from events.")
    masks_dict, labels = multi_group

    conditions: List[Tuple[str, np.ndarray]] = []
    for label in labels:
        mask = masks_dict.get(label)
        if mask is None:
            mask_bool = np.zeros(len(events_df), dtype=bool)
        else:
            mask_bool = np.asarray(mask, dtype=bool)
        conditions.append((str(label), mask_bool))

    if len(conditions) < 2:
        raise ValueError(
            "Multi-condition comparison requires at least two groups with matching events."
        )
    return conditions


def _parse_comparison_windows(config: Any) -> List[str]:
    """Resolve configured comparison windows."""
    from eeg_pipeline.utils.config.loader import require_config_value

    segments = require_config_value(config, "plotting.comparisons.comparison_windows")
    if not isinstance(segments, (list, tuple)) or len(segments) < 2:
        raise ValueError(
            "plotting.comparisons.comparison_windows must be a list/tuple with at least 2 window names."
        )
    parsed_segments = [str(segment).strip() for segment in segments if str(segment).strip()]
    if len(parsed_segments) < 2:
        raise ValueError(
            "plotting.comparisons.comparison_windows must contain at least 2 non-empty window names."
        )
    return parsed_segments


def _is_finite_value(value: Any) -> bool:
    """Return True when value can be interpreted as a finite float."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _compute_subject_condition_psd(
    *,
    tfr_epochs: Any,
    mask: np.ndarray,
    active_window: Tuple[float, float],
    baseline_window: Tuple[float, float],
    logger: logging.Logger,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute one subject-level PSD vector for a selected condition."""
    from eeg_pipeline.utils.analysis.tfr import apply_baseline_and_crop

    n_epochs = min(len(tfr_epochs), len(mask))
    if n_epochs <= 0:
        return None

    mask_bool = np.asarray(mask[:n_epochs], dtype=bool)
    if int(mask_bool.sum()) == 0:
        return None

    tfr_condition = tfr_epochs[mask_bool]
    if len(tfr_condition) == 0:
        return None

    tfr_condition_avg = tfr_condition.average()
    apply_baseline_and_crop(
        tfr_condition_avg,
        baseline=baseline_window,
        mode="logratio",
        logger=logger,
    )

    times = np.asarray(tfr_condition_avg.times)
    tmin = max(float(times.min()), float(active_window[0]))
    tmax = min(float(times.max()), float(active_window[1]))
    if tmax <= tmin:
        return None

    tfr_window = tfr_condition_avg.copy().crop(tmin=tmin, tmax=tmax)
    freqs = np.asarray(tfr_window.freqs, dtype=float)
    psd_vector = np.asarray(tfr_window.data.mean(axis=(0, 2)), dtype=float)

    if freqs.ndim != 1 or psd_vector.ndim != 1 or len(freqs) != len(psd_vector):
        return None
    if len(freqs) == 0:
        return None
    return freqs, psd_vector


def visualize_power_by_condition_for_group(
    *,
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Render group-level power condition plots using subject-level summaries."""
    if not subjects:
        raise ValueError("No subjects specified")

    config = _load_config_if_needed(config)
    setup_matplotlib(config)
    task = _resolve_task(task, config)
    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    from eeg_pipeline.infra.tsv import read_table
    from eeg_pipeline.plotting.features.roi import get_roi_channels, get_roi_definitions
    from eeg_pipeline.plotting.features.utils import (
        plot_multi_window_comparison,
        plot_paired_comparison,
    )
    from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names, require_config_value
    from eeg_pipeline.utils.formatting import sanitize_label

    compare_windows = bool(get_config_value(config, "plotting.comparisons.compare_windows", True))
    compare_columns = bool(get_config_value(config, "plotting.comparisons.compare_columns", False))
    if not compare_windows and not compare_columns:
        raise ValueError(
            "Group power_by_condition requires plotting.comparisons.compare_windows or compare_columns."
        )

    segments = _parse_comparison_windows(config) if compare_windows else []
    comparison_segment = ""
    if compare_columns:
        comparison_segment = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        if comparison_segment == "":
            raise ValueError("plotting.comparisons.comparison_segment must be a non-empty string.")

    bands = list(get_frequency_band_names(config) or [])
    if not bands:
        raise ValueError("No frequency bands resolved for plotting.")

    rois = get_roi_definitions(config)
    roi_names = _resolve_power_roi_names(config=config, rois=rois)

    window_subject_values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    column_subject_values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    column_labels: Optional[List[str]] = None

    for subject in subjects:
        features_dir = deriv_features_path(effective_deriv_root, subject)
        power_df = _load_features_power_df(features_dir=features_dir, read_table=read_table, logger=logger)
        if power_df is None or power_df.empty:
            logger.warning("Group power_by_condition: missing power features for sub-%s; skipping", subject)
            continue

        subject_channels = _extract_power_channels(list(power_df.columns))

        if compare_windows:
            for roi_name in roi_names:
                if roi_name == "all":
                    roi_channels: Optional[List[str]] = None
                else:
                    roi_channels = get_roi_channels(rois.get(roi_name, []), subject_channels)
                    if not roi_channels:
                        continue

                for band in bands:
                    for segment in segments:
                        columns = _select_power_columns(
                            power_df,
                            segment=segment,
                            band=band,
                            roi_channels=roi_channels,
                        )
                        if not columns:
                            continue

                        series = power_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                        mean_value = float(np.nanmean(series.values))
                        if not np.isfinite(mean_value):
                            continue

                        window_subject_values.setdefault(roi_name, {}).setdefault(band, {}).setdefault(subject, {})[
                            segment
                        ] = mean_value

        if compare_columns:
            epochs, events_df = load_epochs_for_analysis(
                subject=subject,
                task=task,
                align="strict",
                preload=False,
                deriv_root=effective_deriv_root,
                config=config,
                logger=logger,
            )
            if epochs is None or events_df is None or events_df.empty:
                logger.warning(
                    "Group power_by_condition: missing epochs/events for sub-%s; skipping condition comparison",
                    subject,
                )
                continue
            if len(events_df) != len(power_df):
                logger.warning(
                    "Group power_by_condition: length mismatch for sub-%s (events=%d, power=%d); skipping condition comparison",
                    subject,
                    len(events_df),
                    len(power_df),
                )
                continue

            try:
                conditions = _resolve_group_conditions(
                    events_df=events_df,
                    config=config,
                    allow_single_value=False,
                )
            except ValueError as exc:
                logger.warning(
                    "Group power_by_condition: could not resolve conditions for sub-%s; skipping (%s)",
                    subject,
                    exc,
                )
                continue
            labels = [label for label, _ in conditions]
            if column_labels is None:
                column_labels = labels
            elif labels != column_labels:
                raise ValueError(
                    "Inconsistent condition labels across subjects in group power_by_condition."
                )

            for roi_name in roi_names:
                if roi_name == "all":
                    roi_channels = None
                else:
                    roi_channels = get_roi_channels(rois.get(roi_name, []), subject_channels)
                    if not roi_channels:
                        continue

                for band in bands:
                    columns = _select_power_columns(
                        power_df,
                        segment=comparison_segment,
                        band=band,
                        roi_channels=roi_channels,
                    )
                    if not columns:
                        continue

                    series = power_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    n_rows = len(series)
                    for label, mask in conditions:
                        n = min(n_rows, len(mask))
                        if n <= 0:
                            continue
                        mask_bool = np.asarray(mask[:n], dtype=bool)
                        if int(mask_bool.sum()) == 0:
                            continue
                        values = series.iloc[:n][mask_bool].dropna().values
                        mean_value = float(np.nanmean(values))
                        if not np.isfinite(mean_value):
                            continue
                        column_subject_values.setdefault(roi_name, {}).setdefault(band, {}).setdefault(subject, {})[
                            label
                        ] = mean_value

    plots_dir = deriv_plots_path(effective_deriv_root, "group", subdir="features")
    power_plots_dir = plots_dir / "power"
    ensure_dir(power_plots_dir)

    rendered_plots = 0

    if compare_windows:
        for roi_name in roi_names:
            by_band = window_subject_values.get(roi_name, {})
            if not by_band:
                continue

            roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
            suffix = f"_roi-{roi_safe}" if roi_safe else ""

            if len(segments) == 2:
                paired_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                seg1, seg2 = segments[0], segments[1]

                for band in bands:
                    per_subject = by_band.get(band, {})
                    values1: List[float] = []
                    values2: List[float] = []
                    for seg_values in per_subject.values():
                        value1 = seg_values.get(seg1)
                        value2 = seg_values.get(seg2)
                        if not _is_finite_value(value1) or not _is_finite_value(value2):
                            continue
                        values1.append(float(value1))
                        values2.append(float(value2))

                    if len(values1) >= 2:
                        paired_data[band] = (np.asarray(values1, dtype=float), np.asarray(values2, dtype=float))

                if paired_data:
                    save_path = power_plots_dir / f"sub-group_power_by_condition{suffix}_window"
                    plot_paired_comparison(
                        data_by_band=paired_data,
                        subject="group",
                        save_path=save_path,
                        feature_label="Band Power",
                        config=config,
                        logger=logger,
                        label1=seg1.capitalize(),
                        label2=seg2.capitalize(),
                        roi_name=roi_name,
                        sample_unit="subjects",
                    )
                    rendered_plots += 1
            else:
                multi_segment_data: Dict[str, Dict[str, np.ndarray]] = {}
                for band in bands:
                    per_subject = by_band.get(band, {})
                    complete_subjects = [
                        subject_id
                        for subject_id, seg_values in per_subject.items()
                        if all(_is_finite_value(seg_values.get(segment)) for segment in segments)
                    ]
                    if len(complete_subjects) < 2:
                        continue

                    segment_arrays = {
                        segment: np.asarray(
                            [float(per_subject[subject_id][segment]) for subject_id in complete_subjects],
                            dtype=float,
                        )
                        for segment in segments
                    }
                    multi_segment_data[band] = segment_arrays

                if multi_segment_data:
                    save_path = power_plots_dir / f"sub-group_power_by_condition{suffix}_multiwindow"
                    plot_multi_window_comparison(
                        data_by_band=multi_segment_data,
                        subject="group",
                        save_path=save_path,
                        feature_label="Band Power",
                        segments=segments,
                        config=config,
                        logger=logger,
                        roi_name=roi_name,
                        sample_unit="subjects",
                        comparison_dimension_name="windows",
                    )
                    rendered_plots += 1

    if compare_columns:
        if column_labels is None:
            raise ValueError(
                "Group power_by_condition could not resolve configured condition masks for any subject."
            )

        for roi_name in roi_names:
            by_band = column_subject_values.get(roi_name, {})
            if not by_band:
                continue

            roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
            suffix = f"_roi-{roi_safe}" if roi_safe else ""

            if len(column_labels) == 2:
                label1, label2 = column_labels[0], column_labels[1]
                paired_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

                for band in bands:
                    per_subject = by_band.get(band, {})
                    values1: List[float] = []
                    values2: List[float] = []
                    for condition_values in per_subject.values():
                        value1 = condition_values.get(label1)
                        value2 = condition_values.get(label2)
                        if not _is_finite_value(value1) or not _is_finite_value(value2):
                            continue
                        values1.append(float(value1))
                        values2.append(float(value2))

                    if len(values1) >= 2:
                        paired_data[band] = (np.asarray(values1, dtype=float), np.asarray(values2, dtype=float))

                if paired_data:
                    save_path = power_plots_dir / f"sub-group_power_by_condition{suffix}_column"
                    plot_paired_comparison(
                        data_by_band=paired_data,
                        subject="group",
                        save_path=save_path,
                        feature_label="Band Power",
                        config=config,
                        logger=logger,
                        label1=label1,
                        label2=label2,
                        roi_name=roi_name,
                        sample_unit="subjects",
                    )
                    rendered_plots += 1
            else:
                multi_condition_data: Dict[str, Dict[str, np.ndarray]] = {}
                for band in bands:
                    per_subject = by_band.get(band, {})
                    complete_subjects = [
                        subject_id
                        for subject_id, condition_values in per_subject.items()
                        if all(_is_finite_value(condition_values.get(label)) for label in column_labels)
                    ]
                    if len(complete_subjects) < 2:
                        continue

                    condition_arrays = {
                        label: np.asarray(
                            [float(per_subject[subject_id][label]) for subject_id in complete_subjects],
                            dtype=float,
                        )
                        for label in column_labels
                    }
                    multi_condition_data[band] = condition_arrays

                if multi_condition_data:
                    save_path = power_plots_dir / f"sub-group_power_by_condition{suffix}_multigroup"
                    plot_multi_window_comparison(
                        data_by_band=multi_condition_data,
                        subject="group",
                        save_path=save_path,
                        feature_label="Band Power",
                        segments=column_labels,
                        config=config,
                        logger=logger,
                        roi_name=roi_name,
                        sample_unit="subjects",
                        comparison_dimension_name="conditions",
                    )
                    rendered_plots += 1

    if rendered_plots == 0:
        raise ValueError(
            "Group power_by_condition generated no plots. Check available power features and comparison configuration."
        )

    _save_plot_manifest(plots_dir=plots_dir, subject="group", logger=logger)


def visualize_power_spectral_density_for_group(
    *,
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Render group-level PSD curves using subject-level condition summaries."""
    if not subjects:
        raise ValueError("No subjects specified")

    config = _load_config_if_needed(config)
    setup_matplotlib(config)
    task = _resolve_task(task, config)
    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    from eeg_pipeline.plotting.config import get_plot_config
    from eeg_pipeline.plotting.features.roi import get_roi_channels, get_roi_definitions
    from eeg_pipeline.plotting.io.figures import get_band_color, save_fig
    from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization
    from eeg_pipeline.utils.config.loader import get_frequency_bands, require_config_value
    from eeg_pipeline.utils.formatting import sanitize_label
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    active_window_config = require_config_value(config, "time_frequency_analysis.active_window")
    baseline_window_config = require_config_value(config, "time_frequency_analysis.baseline_window")
    if not isinstance(active_window_config, (list, tuple)) or len(active_window_config) < 2:
        raise ValueError("time_frequency_analysis.active_window must be a list/tuple with two values.")
    if not isinstance(baseline_window_config, (list, tuple)) or len(baseline_window_config) < 2:
        raise ValueError("time_frequency_analysis.baseline_window must be a list/tuple with two values.")

    active_window = (float(active_window_config[0]), float(active_window_config[1]))
    baseline_window = (float(baseline_window_config[0]), float(baseline_window_config[1]))

    rois = get_roi_definitions(config)
    roi_names = _resolve_power_roi_names(config=config, rois=rois)
    frequency_bands = get_frequency_bands(config)
    plot_cfg = get_plot_config(config)

    roi_reference_freqs: Dict[str, np.ndarray] = {}
    roi_subject_conditions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    condition_labels: Optional[List[str]] = None

    for subject in subjects:
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
            logger.warning("Group PSD: missing epochs for sub-%s; skipping", subject)
            continue
        if events_df is None or events_df.empty:
            logger.warning("Group PSD: missing events for sub-%s; skipping", subject)
            continue
        if len(epochs) != len(events_df):
            logger.warning(
                "Group PSD: length mismatch for sub-%s (epochs=%d, events=%d); skipping",
                subject,
                len(epochs),
                len(events_df),
            )
            continue

        try:
            conditions = _resolve_group_conditions(
                events_df=events_df,
                config=config,
                allow_single_value=True,
            )
        except ValueError as exc:
            logger.warning(
                "Group PSD: could not resolve conditions for sub-%s; skipping (%s)",
                subject,
                exc,
            )
            continue
        labels = [label for label, _ in conditions]
        if condition_labels is None:
            condition_labels = labels
        elif labels != condition_labels:
            raise ValueError("Inconsistent condition labels across subjects in group PSD.")

        tfr = compute_tfr_for_visualization(epochs, config, logger)
        subject_channels = list(tfr.ch_names)

        for roi_name in roi_names:
            if roi_name == "all":
                roi_channels = subject_channels
            else:
                roi_channels = get_roi_channels(rois.get(roi_name, []), subject_channels)
                if not roi_channels:
                    continue

            tfr_roi = tfr.copy().pick_channels(roi_channels)
            per_condition_vectors: Dict[str, np.ndarray] = {}

            for label, mask in conditions:
                psd = _compute_subject_condition_psd(
                    tfr_epochs=tfr_roi,
                    mask=mask,
                    active_window=active_window,
                    baseline_window=baseline_window,
                    logger=logger,
                )
                if psd is None:
                    continue

                freqs, psd_vector = psd
                reference_freqs = roi_reference_freqs.get(roi_name)
                if reference_freqs is None:
                    roi_reference_freqs[roi_name] = freqs
                    aligned_vector = psd_vector
                elif len(freqs) == len(reference_freqs) and np.allclose(freqs, reference_freqs, rtol=1e-6, atol=1e-8):
                    aligned_vector = psd_vector
                else:
                    aligned_vector = np.interp(reference_freqs, freqs, psd_vector, left=np.nan, right=np.nan)

                if int(np.isfinite(aligned_vector).sum()) == 0:
                    continue
                per_condition_vectors[label] = aligned_vector.astype(float, copy=False)

            if per_condition_vectors:
                roi_subject_conditions.setdefault(roi_name, {})[subject] = per_condition_vectors

    if condition_labels is None:
        raise ValueError("Group PSD could not resolve configured conditions for any subject.")

    plots_dir = deriv_plots_path(effective_deriv_root, "group", subdir="features")
    power_plots_dir = plots_dir / "power"
    ensure_dir(power_plots_dir)

    rendered_plots = 0
    condition_colors = plt.cm.Set2(np.linspace(0.2, 0.8, len(condition_labels)))

    for roi_name in roi_names:
        subject_condition_map = roi_subject_conditions.get(roi_name, {})
        if not subject_condition_map:
            continue

        complete_subjects = [
            subject_id
            for subject_id, label_map in subject_condition_map.items()
            if all(label in label_map for label in condition_labels)
        ]
        if len(complete_subjects) < 2:
            logger.warning(
                "Group PSD: ROI %s has fewer than 2 complete subjects; skipping",
                roi_name,
            )
            continue

        freqs = roi_reference_freqs.get(roi_name)
        if freqs is None or len(freqs) == 0:
            continue

        fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
        fig, ax = plt.subplots(figsize=fig_size)
        plotted = False

        for idx, label in enumerate(condition_labels):
            stacked = np.vstack([subject_condition_map[subject_id][label] for subject_id in complete_subjects])
            mean_psd = np.nanmean(stacked, axis=0)

            if stacked.shape[0] >= 2:
                sem_psd = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                ci_lower = mean_psd - 1.96 * sem_psd
                ci_upper = mean_psd + 1.96 * sem_psd
                ax.fill_between(freqs, ci_lower, ci_upper, color=condition_colors[idx], alpha=0.15, linewidth=0)

            ax.plot(
                freqs,
                mean_psd,
                color=condition_colors[idx],
                linewidth=2.0,
                label=f"{label} (n={len(complete_subjects)})",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        for band_name, bounds in frequency_bands.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                continue
            band_min = float(bounds[0])
            band_max = float(bounds[1])
            if band_max <= float(freqs.min()) or band_min >= float(freqs.max()):
                continue

            span_start = max(band_min, float(freqs.min()))
            span_end = min(band_max, float(freqs.max()))
            ax.axvspan(
                span_start,
                span_end,
                color=get_band_color(str(band_name), config),
                alpha=0.08,
                zorder=0,
            )

            mid_frequency = 0.5 * (span_start + span_end)
            y_max = ax.get_ylim()[1]
            ax.text(
                mid_frequency,
                y_max * 0.95,
                str(band_name).upper(),
                fontsize=7,
                ha="center",
                va="top",
                color="0.4",
            )

        roi_display = "All Channels" if roi_name == "all" else roi_name.replace("_", " ").title()
        ax.set_title(
            f"Power Spectral Density (Group) | ROI: {roi_display}",
            fontsize=plot_cfg.font.title,
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xticks([2, 4, 8, 16, 32, 64])
        ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.medium)
        ax.set_ylabel(r"$\log_{10}$(power/baseline)", fontsize=plot_cfg.font.medium)
        ax.tick_params(labelsize=plot_cfg.font.small)
        ax.legend(frameon=False, fontsize=plot_cfg.font.small)
        sns.despine(ax=ax, trim=True)

        fig.text(
            0.5,
            0.01,
            (
                f"n={len(complete_subjects)} subjects | "
                f"active window=[{active_window[0]:.3f}, {active_window[1]:.3f}] s | "
                f"baseline=[{baseline_window[0]:.3f}, {baseline_window[1]:.3f}] s"
            ),
            ha="center",
            va="bottom",
            fontsize=plot_cfg.font.small,
            color="gray",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        roi_suffix = f"_roi-{roi_safe}" if roi_safe else ""
        save_fig(
            fig,
            power_plots_dir / f"sub-group_power_spectral_density{roi_suffix}",
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            config=config,
        )
        plt.close(fig)
        rendered_plots += 1

    if rendered_plots == 0:
        raise ValueError(
            "Group power_spectral_density generated no plots. "
            "Check epochs availability and comparison configuration."
        )

    _save_plot_manifest(plots_dir=plots_dir, subject="group", logger=logger)


__all__ = [
    "visualize_features",
    "visualize_features_for_subjects",
    "visualize_band_power_topomaps_for_group",
    "visualize_power_by_condition_for_group",
    "visualize_power_spectral_density_for_group",
]
