"""
Plotting utilities for figure saving, topomaps, and visualization.

This module provides functions for saving figures, configuring matplotlib,
and creating topomap visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import os
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from eeg_pipeline.utils.config.loader import get_nested_value, load_settings

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.io.formatting import sanitize_label, format_baseline_string


_PLOT_DEFAULT_DPI = None
_PLOT_DEFAULT_FORMATS = None
_PLOT_DEFAULT_BBOX = None
_PLOT_DEFAULT_PAD = None


@dataclass
class SaveFigConfig:
    """Configuration for saving figures (dpi, formats, etc.)."""

    dpi: int = 300
    formats: Tuple[str, ...] = ("png",)
    bbox_inches: str = "tight"
    pad_inches: float = 0.02

    @classmethod
    def from_constants(cls, constants: Dict[str, Any]) -> "SaveFigConfig":
        if "FIG_DPI" not in constants:
            raise ValueError("FIG_DPI not found in constants")
        if "SAVE_FORMATS" not in constants:
            raise ValueError("SAVE_FORMATS not found in constants")
        if "output.bbox_inches" not in constants:
            raise ValueError("output.bbox_inches not found in constants")
        if "output.pad_inches" not in constants:
            raise ValueError("output.pad_inches not found in constants")

        return cls(
            dpi=constants["FIG_DPI"],
            formats=tuple(constants["SAVE_FORMATS"]),
            bbox_inches=constants["output.bbox_inches"],
            pad_inches=float(constants["output.pad_inches"]),
        )

    @classmethod
    def get_defaults(cls) -> "SaveFigConfig":
        return cls()


PlotConfig = SaveFigConfig


def _get_plot_constants(constants=None):
    global _PLOT_DEFAULT_DPI, _PLOT_DEFAULT_FORMATS, _PLOT_DEFAULT_BBOX, _PLOT_DEFAULT_PAD
    if _PLOT_DEFAULT_DPI is None:
        if constants is None:
            raise ValueError("constants is required for _get_plot_constants")
        plot_config = SaveFigConfig.from_constants(constants)
        _PLOT_DEFAULT_DPI = plot_config.dpi
        _PLOT_DEFAULT_FORMATS = plot_config.formats
        _PLOT_DEFAULT_BBOX = plot_config.bbox_inches
        _PLOT_DEFAULT_PAD = plot_config.pad_inches


def build_footer(template_name: str, config, **kwargs) -> str:
    if config is None:
        raise ValueError("config is required for build_footer")
    templates = config.get("visualization.footer_templates", {})
    if template_name not in templates:
        raise ValueError(
            f"Footer template '{template_name}' not found in config. Available: {list(templates.keys())}"
        )
    template = templates[template_name]
    return template.format(**kwargs)


def unwrap_figure(obj):
    return obj[0] if isinstance(obj, list) else obj


def get_behavior_footer(config) -> str:
    bwin = tuple(config.get("time_frequency_analysis.baseline_window"))
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha")
    return f"Baseline: [{float(bwin[0]):.2f}, {float(bwin[1]):.2f}] s | Significance: BH-FDR α={fdr_alpha}"


def get_band_color(band: str, config=None) -> str:
    from eeg_pipeline.plotting.core.colors import get_band_color as _get_band_color

    return _get_band_color(band, config)


def logratio_to_pct(v):
    v_arr = np.asarray(v, dtype=float)
    return (np.power(10.0, v_arr) - 1.0) * 100.0


def pct_to_logratio(p):
    p_arr = np.asarray(p, dtype=float)
    return np.log10(np.clip(1.0 + (p_arr / 100.0), 1e-9, None))


def get_viz_params(config=None):
    if config is None:
        try:
            config = load_settings()
        except Exception:
            pass

    if config is None:
        raise ValueError("config is required for get_viz_params")

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


def plot_topomap_on_ax(ax, data, info, mask=None, mask_params=None, vmin=None, vmax=None, config=None):
    viz_params = get_viz_params(config)
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
        vlim=(vmin, vmax) if vmin is not None and vmax is not None else None,
    )
    if viz_params["annotate_descriptive_topo"] and hasattr(ax, "figure"):
        fig = ax.figure
        if not getattr(fig, "_descriptive_note_added", False):
            fig.text(
                0.02,
                0.02,
                "Descriptive topomap; see stats for inference (FDR/cluster)",
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.7,
            )
            setattr(fig, "_descriptive_note_added", True)


def robust_sym_vlim(
    arrs: "np.ndarray | list[np.ndarray]",
    q_low: Optional[float] = None,
    q_high: Optional[float] = None,
    cap: Optional[float] = None,
    min_v: Optional[float] = None,
    adaptive_multiplier: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    defaults = {
        "q_low": 0.01,
        "q_high": 0.99,
        "cap": 0.25,
        "min_v": 1e-6,
        "adaptive_multiplier": 2.0,
    }

    if config is None:
        config = load_settings()

    if config is not None:
        vlim_config = config.get("visualization.robust_vlim", {})
        defaults.update({k: float(vlim_config.get(k, v)) for k, v in defaults.items()})

    q_low = q_low or defaults["q_low"]
    q_high = q_high or defaults["q_high"]
    cap = cap or defaults["cap"]
    min_v = min_v or defaults["min_v"]
    adaptive_multiplier = adaptive_multiplier or defaults["adaptive_multiplier"]

    if isinstance(arrs, (list, tuple)):
        flat = np.concatenate([np.asarray(a).ravel() for a in arrs if a is not None])
    else:
        flat = np.asarray(arrs).ravel()

    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return cap

    lo = np.nanquantile(flat, q_low)
    hi = np.nanquantile(flat, q_high)
    v = float(max(abs(lo), abs(hi)))

    if not np.isfinite(v) or v <= 0:
        v = min_v
    else:
        v = v * adaptive_multiplier

    return min(v, float(cap))


def setup_matplotlib(config: Optional[Dict[str, Any]] = None) -> None:
    import matplotlib

    _backend_set = getattr(matplotlib, "_backend_set_for_pipeline", False)
    if not _backend_set:
        try:
            matplotlib.use("Agg", force=False)
            matplotlib._backend_set_for_pipeline = True
        except Exception:
            pass

    dpi = 300
    if config is not None:
        dpi = get_nested_value(config, "plotting.defaults.savefig_dpi", 300)

    sns.set_theme(context="paper", style="white", font_scale=1.05)
    plt.rcParams.update(
        {
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
    )


def extract_plotting_constants(config, save_formats: Optional[List[str]] = None):
    if config is None:
        raise ValueError("config is required for extract_plotting_constants")

    return {
        "FIG_DPI": int(config.get("plotting.defaults.savefig_dpi", 300)),
        "SAVE_FORMATS": save_formats or list(config.get("plotting.defaults.formats", ["svg"])),
        "output.bbox_inches": config.get("plotting.defaults.bbox_inches", "tight"),
        "output.pad_inches": float(config.get("plotting.defaults.pad_inches", 0.02)),
    }


def extract_eeg_picks(epochs_or_info, exclude_bads: bool = True):
    if hasattr(epochs_or_info, "info"):
        info = epochs_or_info.info
    else:
        info = epochs_or_info

    exclude = "bads" if exclude_bads else []
    return mne.pick_types(info, eeg=True, meg=False, eog=False, stim=False, exclude=exclude)


def log_if_present(logger, level: str, message: str):
    if logger:
        getattr(logger, level)(message)


def validate_picks(picks, logger):
    if len(picks) == 0:
        log_if_present(logger, "warning", "No valid EEG channels found")
        return False
    return True


def get_default_config():
    return load_settings()


def _prepare_figure_footer(
    footer: Optional[str],
    footer_template_name: Optional[str],
    footer_kwargs: Optional[dict],
    constants,
) -> Optional[str]:
    if footer is not None:
        return footer

    if footer_template_name is None or constants is None:
        return None

    try:
        cfg = load_settings()
        return build_footer(footer_template_name, cfg, **(footer_kwargs or {}))
    except (KeyError, ValueError, AttributeError):
        return None


def _should_add_footer(footer: Optional[str]) -> bool:
    if footer is None:
        return False

    footer_env = os.getenv("FIG_FOOTER_OFF", "0").lower()
    return footer_env not in {"1", "true", "yes"}


def _apply_tight_layout(fig: plt.Figure, rect: Optional[Tuple[float, float, float, float]]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            if rect is not None:
                fig.tight_layout(rect=rect)
            else:
                fig.tight_layout()
        except RuntimeError:
            fig.subplots_adjust(bottom=0.06)


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
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        return True
    except AttributeError as e:
        if "copy_from_bbox" not in str(e):
            raise
        try:
            return _save_figure_with_agg_backend(fig, output_path, dpi, bbox_inches, pad_inches)
        except Exception as e2:
            logging.getLogger(__name__).warning(
                f"Failed to save figure {output_path} with fallback backend: {e2}"
            )
            return False
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save figure {output_path}: {e}")
        return False


def _save_figure_with_agg_backend(
    fig: plt.Figure,
    output_path: Path,
    dpi: int,
    bbox_inches: str,
    pad_inches: float,
) -> bool:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    old_canvas = fig.canvas

    try:
        fig.canvas = FigureCanvasAgg(fig)
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        return True
    finally:
        fig.canvas = old_canvas


def _save_figure_to_formats(
    fig: plt.Figure,
    base_path: Path,
    formats: Tuple[str, ...],
    dpi: int,
    bbox_inches: str,
    pad_inches: float,
) -> List[Path]:
    saved_paths = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in formats:
            output_path = base_path.with_suffix(f".{ext}")
            if _save_figure_with_fallback(fig, output_path, dpi, bbox_inches, pad_inches):
                saved_paths.append(output_path)

    return saved_paths


def save_fig(
    fig: plt.Figure,
    path,
    logger=None,
    footer: Optional[str] = None,
    formats: Optional[Tuple[str, ...]] = None,
    dpi: Optional[int] = None,
    bbox_inches: Optional[str] = None,
    pad_inches: Optional[float] = None,
    tight_layout_rect: Optional[Tuple[float, float, float, float]] = None,
    constants=None,
    footer_template_name: Optional[str] = None,
    footer_kwargs: Optional[dict] = None,
):
    path = Path(path)
    ensure_dir(path.parent)
    base_path = path.with_suffix("") if path.suffix else path

    if constants is not None:
        _get_plot_constants(constants=constants)

    formats = formats or _PLOT_DEFAULT_FORMATS or ("png",)
    dpi = dpi if dpi is not None else (_PLOT_DEFAULT_DPI or 150)
    bbox_inches = bbox_inches or (_PLOT_DEFAULT_BBOX or "tight")
    pad_inches = pad_inches if pad_inches is not None else (_PLOT_DEFAULT_PAD or 0.1)

    footer = _prepare_figure_footer(footer, footer_template_name, footer_kwargs, constants)
    _prepare_figure_layout(fig, footer, tight_layout_rect)

    saved_paths = _save_figure_to_formats(fig, base_path, formats, dpi, bbox_inches, pad_inches)

    plt.close(fig)

    if logger is not None:
        filenames = ", ".join(p.name for p in saved_paths)
        logger.info(f"  Saved: {filenames}")


__all__ = [
    "SaveFigConfig",
    "PlotConfig",
    "_get_plot_constants",
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
    "extract_plotting_constants",
    "extract_eeg_picks",
    "log_if_present",
    "validate_picks",
    "get_default_config",
    "_prepare_figure_footer",
    "_should_add_footer",
    "_apply_tight_layout",
    "_prepare_figure_layout",
    "_save_figure_with_fallback",
    "_save_figure_with_agg_backend",
    "_save_figure_to_formats",
    "save_fig",
]
