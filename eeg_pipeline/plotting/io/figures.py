"""
Plotting utilities for figure saving, topomaps, and visualization.

This module provides functions for saving figures, configuring matplotlib,
and creating topomap visualizations.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.utils.config.loader import get_nested_value, load_config


def build_footer(template_name: str, config: Dict[str, Any], **kwargs) -> str:
    """Build footer string from template configuration.
    
    Args:
        template_name: Name of the footer template to use.
        config: Configuration dictionary containing footer templates.
        **kwargs: Variables to format into the template.
        
    Returns:
        Formatted footer string.
        
    Raises:
        ValueError: If config is None or template_name is not found.
    """
    if config is None:
        raise ValueError("config is required for build_footer")
    templates = config.get("visualization.footer_templates", {})
    if template_name not in templates:
        available = list(templates.keys())
        raise ValueError(
            f"Footer template '{template_name}' not found in config. "
            f"Available templates: {available}"
        )
    template = templates[template_name]
    return template.format(**kwargs)


def unwrap_figure(obj: Any) -> Any:
    """Extract figure from list if wrapped, otherwise return as-is.
    
    Args:
        obj: Figure object or list containing a figure.
        
    Returns:
        The figure object.
    """
    return obj[0] if isinstance(obj, list) else obj


def _format_baseline_string(config: Dict[str, Any]) -> str:
    """Format baseline window string from configuration."""
    baseline_window = tuple(config.get("time_frequency_analysis.baseline_window"))
    start_time = float(baseline_window[0])
    end_time = float(baseline_window[1])
    return f"Baseline: [{start_time:.2f}, {end_time:.2f}] s"


def get_behavior_footer(
    config: Dict[str, Any],
    *,
    inference: Optional[str] = None,
    alpha: Optional[float] = None,
) -> str:
    """Build footer string for behavioral analysis plots.
    
    Args:
        config: Configuration dictionary.
        inference: Optional inference method name.
        alpha: Optional significance level.
        
    Returns:
        Formatted footer string with baseline and significance information.
    """
    baseline_str = _format_baseline_string(config)
    
    if inference is None:
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha")
        return f"{baseline_str} | Significance: BH-FDR α={fdr_alpha}"

    if alpha is not None and np.isfinite(alpha):
        alpha_str = f" (α={float(alpha):.3g})"
        return f"{baseline_str} | {inference}{alpha_str}"
    
    return f"{baseline_str} | {inference}"


def get_band_color(band: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Get color for frequency band from configuration.
    
    Args:
        band: Frequency band name.
        config: Optional configuration dictionary.
        
    Returns:
        Color string for the band.
    """
    from eeg_pipeline.plotting.core.colors import get_band_color as _get_band_color

    return _get_band_color(band, config)


def logratio_to_pct(logratio: float | np.ndarray) -> float | np.ndarray:
    """Convert log-ratio to percentage change.
    
    Args:
        logratio: Log-ratio value(s).
        
    Returns:
        Percentage change value(s).
    """
    logratio_array = np.asarray(logratio, dtype=float)
    return (np.power(10.0, logratio_array) - 1.0) * 100.0


def pct_to_logratio(percentage: float | np.ndarray) -> float | np.ndarray:
    """Convert percentage change to log-ratio.
    
    Args:
        percentage: Percentage change value(s).
        
    Returns:
        Log-ratio value(s).
    """
    percentage_array = np.asarray(percentage, dtype=float)
    min_value = 1e-9
    clipped = np.clip(1.0 + (percentage_array / 100.0), min_value, None)
    return np.log10(clipped)


def get_viz_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get visualization parameters from configuration.
    
    Args:
        config: Optional configuration dictionary. If None, attempts to load default.
        
    Returns:
        Dictionary of visualization parameters.
        
    Raises:
        ValueError: If config cannot be loaded.
    """
    if config is None:
        try:
            config = load_config()
        except Exception as e:
            raise ValueError(
                f"config is required for get_viz_params and could not be loaded: {e}"
            ) from e

    default_sig_mask_params = {
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgecolor": "g",
        "linewidth": 0.8,
        "markersize": 3,
    }

    topomap_config = config.get("plotting.plots.topomap", {})

    return {
        "topo_contours": topomap_config.get("contours"),
        "topo_cmap": topomap_config.get("colormap"),
        "colorbar_fraction": topomap_config.get("colorbar_fraction"),
        "colorbar_pad": topomap_config.get("colorbar_pad"),
        "diff_annotation_enabled": topomap_config.get("diff_annotation_enabled"),
        "annotate_descriptive_topo": topomap_config.get("annotate_descriptive"),
        "sig_mask_params": topomap_config.get("sig_mask_params", default_sig_mask_params),
    }


def _add_descriptive_annotation(fig: plt.Figure) -> None:
    """Add descriptive annotation to figure if not already present."""
    annotation_text = "Descriptive topomap; see stats for inference (FDR/cluster)"
    annotation_key = "_descriptive_note_added"
    
    if not getattr(fig, annotation_key, False):
        fig.text(
            0.02,
            0.02,
            annotation_text,
            fontsize=7,
            ha="left",
            va="bottom",
            alpha=0.7,
        )
        setattr(fig, annotation_key, True)


def plot_topomap_on_ax(
    ax: plt.Axes,
    data: np.ndarray,
    info: mne.Info,
    mask: Optional[np.ndarray] = None,
    mask_params: Optional[Dict[str, Any]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot topomap on given axes.
    
    Args:
        ax: Matplotlib axes to plot on.
        data: Data array for topomap.
        info: MNE Info object with channel positions.
        mask: Optional mask array.
        mask_params: Optional mask parameters.
        vmin: Optional minimum value for colormap.
        vmax: Optional maximum value for colormap.
        config: Optional configuration dictionary.
    """
    viz_params = get_viz_params(config)
    vlim = (vmin, vmax) if vmin is not None and vmax is not None else None
    
    mne.viz.plot_topomap(
        data,
        info,
        axes=ax,
        show=False,
        mask=mask,
        mask_params=mask_params or {},
        sensors=True,
        contours=viz_params["topo_contours"],
        cmap=viz_params["topo_cmap"],
        vlim=vlim,
    )
    
    if viz_params["annotate_descriptive_topo"] and hasattr(ax, "figure"):
        _add_descriptive_annotation(ax.figure)


def _get_robust_vlim_defaults(config: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Get default parameters for robust symmetric value limits."""
    defaults = {
        "q_low": 0.01,
        "q_high": 0.99,
        "cap": 0.25,
        "min_v": 1e-6,
        "adaptive_multiplier": 2.0,
    }
    
    if config is not None:
        vlim_config = config.get("visualization.robust_vlim", {})
        for key, default_value in defaults.items():
            if key in vlim_config:
                defaults[key] = float(vlim_config[key])
    
    return defaults


def _flatten_arrays(arrs: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Flatten array(s) into a single 1D array."""
    if isinstance(arrs, (list, tuple)):
        arrays = [np.asarray(a).ravel() for a in arrs if a is not None]
        if not arrays:
            return np.array([])
        return np.concatenate(arrays)
    return np.asarray(arrs).ravel()


def _compute_robust_symmetric_limit(
    flat_data: np.ndarray,
    q_low: float,
    q_high: float,
    cap: float,
    min_v: float,
    adaptive_multiplier: float,
) -> float:
    """Compute robust symmetric value limit from flattened data."""
    if flat_data.size == 0:
        return cap
    
    finite_data = flat_data[np.isfinite(flat_data)]
    if finite_data.size == 0:
        return cap
    
    low_quantile = np.nanquantile(finite_data, q_low)
    high_quantile = np.nanquantile(finite_data, q_high)
    max_abs_value = float(max(abs(low_quantile), abs(high_quantile)))
    
    if not np.isfinite(max_abs_value) or max_abs_value <= 0:
        return min_v
    
    scaled_value = max_abs_value * adaptive_multiplier
    return min(scaled_value, float(cap))


def robust_sym_vlim(
    arrs: np.ndarray | list[np.ndarray],
    q_low: Optional[float] = None,
    q_high: Optional[float] = None,
    cap: Optional[float] = None,
    min_v: Optional[float] = None,
    adaptive_multiplier: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute robust symmetric value limits for visualization.
    
    Args:
        arrs: Array or list of arrays to compute limits from.
        q_low: Lower quantile (default from config).
        q_high: Upper quantile (default from config).
        cap: Maximum value cap (default from config).
        min_v: Minimum value (default from config).
        adaptive_multiplier: Multiplier for computed value (default from config).
        config: Optional configuration dictionary.
        
    Returns:
        Symmetric value limit for visualization.
    """
    defaults = _get_robust_vlim_defaults(config)
    
    quantile_low = q_low if q_low is not None else defaults["q_low"]
    quantile_high = q_high if q_high is not None else defaults["q_high"]
    value_cap = cap if cap is not None else defaults["cap"]
    min_value = min_v if min_v is not None else defaults["min_v"]
    multiplier = adaptive_multiplier if adaptive_multiplier is not None else defaults["adaptive_multiplier"]
    
    flat_data = _flatten_arrays(arrs)
    
    return _compute_robust_symmetric_limit(
        flat_data, quantile_low, quantile_high, value_cap, min_value, multiplier
    )


def _configure_matplotlib_backend() -> None:
    """Configure matplotlib to use non-interactive backend."""
    import matplotlib
    
    backend_key = "_backend_set_for_pipeline"
    if not getattr(matplotlib, backend_key, False):
        try:
            matplotlib.use("Agg", force=False)
            setattr(matplotlib, backend_key, True)
        except Exception:
            pass


def setup_matplotlib(config: Optional[Dict[str, Any]] = None) -> None:
    """Configure matplotlib and seaborn for pipeline plotting.
    
    Sets up non-interactive backend, theme, and default parameters.
    
    Args:
        config: Optional configuration dictionary for DPI settings.
    """
    _configure_matplotlib_backend()
    
    default_dpi = 300
    dpi = get_nested_value(config, "plotting.defaults.savefig_dpi", default_dpi) if config else default_dpi

    sns.set_theme(context="paper", style="white", font_scale=1.05)
    
    rc_params = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "savefig.dpi": dpi,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    }
    plt.rcParams.update(rc_params)


def extract_eeg_picks(
    epochs_or_info: Any,
    exclude_bads: bool = True,
) -> np.ndarray:
    """Extract EEG channel picks from epochs or info object.
    
    Args:
        epochs_or_info: MNE epochs object or Info object.
        exclude_bads: Whether to exclude bad channels.
        
    Returns:
        Array of channel indices.
    """
    if hasattr(epochs_or_info, "info"):
        info = epochs_or_info.info
    else:
        info = epochs_or_info

    exclude_channels = "bads" if exclude_bads else []
    return mne.pick_types(
        info, eeg=True, meg=False, eog=False, stim=False, exclude=exclude_channels
    )


def log_if_present(logger: Optional[logging.Logger], level: str, message: str) -> None:
    """Log message if logger is provided.
    
    Args:
        logger: Optional logger instance.
        level: Log level name (e.g., "info", "warning").
        message: Message to log.
    """
    if logger is not None:
        getattr(logger, level)(message)


def get_default_config() -> Dict[str, Any]:
    """Load and return default configuration.
    
    Returns:
        Configuration dictionary.
    """
    return load_config()


def _prepare_figure_footer(
    footer: Optional[str],
    footer_template_name: Optional[str],
    footer_kwargs: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Prepare figure footer from template or use provided footer.
    
    Args:
        footer: Optional pre-formatted footer string.
        footer_template_name: Optional template name to use.
        footer_kwargs: Optional keyword arguments for template.
        
    Returns:
        Footer string or None.
    """
    if footer is not None:
        return footer

    if footer_template_name is None:
        return None

    try:
        config = load_config()
        return build_footer(footer_template_name, config, **(footer_kwargs or {}))
    except (KeyError, ValueError, AttributeError):
        return None


def _should_add_footer(footer: Optional[str]) -> bool:
    """Determine if footer should be added to figure.
    
    Args:
        footer: Footer string to check.
        
    Returns:
        True if footer should be added, False otherwise.
    """
    if footer is None:
        return False

    footer_env_var = "FIG_FOOTER_OFF"
    footer_env_value = os.getenv(footer_env_var, "0").lower()
    disabled_values = {"1", "true", "yes"}
    return footer_env_value not in disabled_values


def _apply_tight_layout(fig: plt.Figure, rect: Optional[Tuple[float, float, float, float]]) -> None:
    """Apply tight layout to figure with fallback.
    
    Args:
        fig: Figure to adjust.
        rect: Optional rectangle tuple for tight_layout.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            if rect is not None:
                fig.tight_layout(rect=rect)
            else:
                fig.tight_layout()
        except RuntimeError:
            fallback_bottom = 0.06
            fig.subplots_adjust(bottom=fallback_bottom)


def _prepare_figure_layout(
    fig: plt.Figure,
    footer: Optional[str],
    tight_layout_rect: Optional[Tuple[float, float, float, float]],
) -> None:
    if _should_add_footer(footer):
        fig.text(0.01, 0.01, footer, fontsize=8, alpha=0.8)
        rect = tight_layout_rect or [0, 0.03, 1, 1]
        _apply_tight_layout(fig, rect)
    elif tight_layout_rect is not None:
        _apply_tight_layout(fig, tight_layout_rect)
    else:
        _apply_tight_layout(fig, None)


def _save_figure_with_fallback(
    fig: plt.Figure,
    output_path: Path,
    dpi: int,
    bbox_inches: str,
    pad_inches: float,
) -> bool:
    """Save figure with fallback to Agg backend if needed.
    
    Args:
        fig: Figure to save.
        output_path: Path to save figure.
        dpi: DPI for saved figure.
        bbox_inches: Bounding box setting.
        pad_inches: Padding in inches.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        return True
    except AttributeError as error:
        error_message = str(error)
        if "copy_from_bbox" not in error_message:
            raise
        
        try:
            return _save_figure_with_agg_backend(fig, output_path, dpi, bbox_inches, pad_inches)
        except Exception as fallback_error:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to save figure {output_path} with fallback backend: {fallback_error}"
            )
            return False
    except Exception as error:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to save figure {output_path}: {error}")
        return False


def _save_figure_with_agg_backend(
    fig: plt.Figure,
    output_path: Path,
    dpi: int,
    bbox_inches: str,
    pad_inches: float,
) -> bool:
    """Save figure using Agg backend.
    
    Args:
        fig: Figure to save.
        output_path: Path to save figure.
        dpi: DPI for saved figure.
        bbox_inches: Bounding box setting.
        pad_inches: Padding in inches.
        
    Returns:
        True if successful, False otherwise.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    original_canvas = fig.canvas

    try:
        fig.canvas = FigureCanvasAgg(fig)
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        return True
    finally:
        fig.canvas = original_canvas


def _save_figure_to_formats(
    fig: plt.Figure,
    base_path: Path,
    formats: Tuple[str, ...],
    dpi: int,
    bbox_inches: str,
    pad_inches: float,
) -> List[Path]:
    """Save figure to multiple formats.
    
    Args:
        fig: Figure to save.
        base_path: Base path without extension.
        formats: Tuple of format extensions.
        dpi: DPI for saved figure.
        bbox_inches: Bounding box setting.
        pad_inches: Padding in inches.
        
    Returns:
        List of successfully saved file paths.
    """
    saved_paths = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for extension in formats:
            output_path = base_path.with_suffix(f".{extension}")
            if _save_figure_with_fallback(fig, output_path, dpi, bbox_inches, pad_inches):
                saved_paths.append(output_path)

    return saved_paths


def save_fig(
    fig: plt.Figure,
    path: str | Path,
    logger: Optional[logging.Logger] = None,
    footer: Optional[str] = None,
    formats: Optional[Tuple[str, ...]] = None,
    dpi: Optional[int] = None,
    bbox_inches: Optional[str] = None,
    pad_inches: Optional[float] = None,
    tight_layout_rect: Optional[Tuple[float, float, float, float]] = None,
    footer_template_name: Optional[str] = None,
    footer_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Save figure to file(s) in specified format(s).
    
    Args:
        fig: Figure to save.
        path: Output path (extension will be replaced based on formats).
        logger: Optional logger for status messages.
        footer: Optional pre-formatted footer string.
        formats: Optional tuple of format extensions (default from config).
        dpi: Optional DPI (default from config).
        bbox_inches: Optional bounding box setting (default from config).
        pad_inches: Optional padding in inches (default from config).
        tight_layout_rect: Optional rectangle tuple for tight_layout.
        footer_template_name: Optional footer template name.
        footer_kwargs: Optional keyword arguments for footer template.
    """
    output_path = Path(path)
    ensure_dir(output_path.parent)
    base_path = output_path.with_suffix("") if output_path.suffix else output_path

    default_formats = ("png",)
    default_dpi = 150
    default_bbox = "tight"
    default_pad = 0.1

    save_formats = formats or default_formats
    save_dpi = dpi if dpi is not None else default_dpi
    save_bbox = bbox_inches or default_bbox
    save_pad = pad_inches if pad_inches is not None else default_pad

    prepared_footer = _prepare_figure_footer(footer, footer_template_name, footer_kwargs)
    _prepare_figure_layout(fig, prepared_footer, tight_layout_rect)

    saved_paths = _save_figure_to_formats(fig, base_path, save_formats, save_dpi, save_bbox, save_pad)

    plt.close(fig)

    if logger is not None:
        filenames = ", ".join(saved_path.name for saved_path in saved_paths)
        logger.info(f"  Saved: {filenames}")


__all__ = [
    "build_footer",
    "unwrap_figure",
    "get_behavior_footer",
    "get_band_color",
    "logratio_to_pct",
    "pct_to_logratio",
    "get_viz_params",
    "plot_topomap_on_ax",
    "robust_sym_vlim",
    "setup_matplotlib",
    "extract_eeg_picks",
    "log_if_present",
    "get_default_config",
    "save_fig",
]
