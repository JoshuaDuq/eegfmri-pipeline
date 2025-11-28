"""
Plotting configuration utilities.

Loads and manages plotting configuration from YAML config files with
comprehensive defaults and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


###################################################################
# Configuration Dataclasses
###################################################################


@dataclass
class FontConfig:
    """Font size configuration."""
    small: int = 7
    medium: int = 8
    large: int = 9
    title: int = 10
    annotation: int = 4
    label: int = 8
    ylabel: int = 10
    suptitle: int = 11
    figure_title: int = 12
    family: str = "sans-serif"
    weight: str = "normal"


@dataclass
class ScatterStyle:
    """Scatter plot styling."""
    marker_size_small: int = 3
    marker_size_large: int = 30
    marker_size_default: int = 30
    alpha: float = 0.7
    edgecolor: str = "white"
    edgewidth: float = 0.3


@dataclass
class BarStyle:
    """Bar plot styling."""
    alpha: float = 0.8
    width: float = 0.6
    capsize: int = 4
    capsize_large: int = 3


@dataclass
class LineStyle:
    """Line plot styling."""
    width_thin: float = 0.5
    width_standard: float = 0.75
    width_thick: float = 1.0
    width_bold: float = 1.5
    alpha_standard: float = 0.8
    alpha_dim: float = 0.5
    alpha_zero: float = 0.3
    alpha_fit: float = 0.8
    alpha_diagonal: float = 0.5
    alpha_reference: float = 0.5
    regression_width: float = 1.5
    residual_width: float = 1.0
    qq_width: float = 0.4


@dataclass
class HistogramStyle:
    """Histogram styling."""
    bins: int = 30
    bins_behavioral: int = 15
    bins_residual: int = 20
    bins_tfr: int = 50
    edgecolor: str = "white"
    edgewidth: float = 0.5
    alpha: float = 0.7
    alpha_residual: float = 0.75
    alpha_tfr: float = 0.8


@dataclass
class ColorPalette:
    """Color palette configuration."""
    gray: str = "#555555"
    light_gray: str = "#999999"
    black: str = "k"
    red: str = "#d62728"
    blue: str = "#1f77b4"
    significant: str = "#C42847"
    nonsignificant: str = "#666666"
    pain: str = "crimson"
    nonpain: str = "navy"
    network_node: str = "#87CEEB"


@dataclass
class PlotStyleConfig:
    """Complete styling configuration."""
    scatter: ScatterStyle = field(default_factory=ScatterStyle)
    bar: BarStyle = field(default_factory=BarStyle)
    line: LineStyle = field(default_factory=LineStyle)
    histogram: HistogramStyle = field(default_factory=HistogramStyle)
    kde_points: int = 100
    kde_color: str = "darkblue"
    kde_linewidth: float = 1.5
    kde_alpha: float = 0.8
    errorbar_markersize: int = 4
    errorbar_capsize: int = 2
    errorbar_capsize_large: int = 3
    colors: ColorPalette = field(default_factory=ColorPalette)
    alpha_grid: float = 0.3
    alpha_fill: float = 0.6
    alpha_text_box: float = 0.8
    alpha_violin_body: float = 0.6
    alpha_ridge_fill: float = 0.6
    alpha_ci: float = 0.1
    alpha_ci_line: float = 0.5


@dataclass
class TextPosition:
    """Text positioning configuration."""
    stats_x: float = 0.05
    stats_y: float = 0.95
    p_value_x: float = 0.98
    p_value_y: float = 0.98
    bootstrap_x: float = 0.02
    bootstrap_y: float = 0.98
    channel_annotation_x: float = 0.02
    channel_annotation_y: float = 0.94
    title_y: float = 0.975
    residual_qc_title_y: float = 1.02


@dataclass
class PlotConfig:
    """Complete plotting configuration.
    
    All values should come from eeg_config.yaml. This dataclass
    does not define defaults - use from_config() or get_defaults()
    to create instances from config file.
    """
    dpi: int
    savefig_dpi: int
    formats: Tuple[str, ...]
    bbox_inches: str
    pad_inches: float
    
    font: FontConfig
    style: PlotStyleConfig
    text_position: TextPosition
    
    figure_sizes: Dict[str, Tuple[float, float]]
    plot_type_configs: Dict[str, Dict[str, Any]]
    validation: Dict[str, int]
    layout_rects: Dict[str, List[float]]
    gridspec_params: Dict[str, Any]
    
    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "PlotConfig":
        """Load configuration from config dictionary."""
        if config is None:
            return cls.get_defaults()
        
        plotting = config.get("plotting", {})
        if not plotting:
            return cls.get_defaults()
        
        defaults = plotting.get("defaults", {})
        styling = plotting.get("styling", {})
        figure_sizes_dict = plotting.get("figure_sizes", {})
        plots_config = plotting.get("plots", {})
        plot_types_legacy = plotting.get("plot_types", {})
        validation_dict = plotting.get("validation", {})
        
        plot_types = dict(plot_types_legacy)
        plot_types.update(plots_config)
        
        layout = defaults.get("layout", {})
        layout_rects = {
            "tight_rect": layout.get("tight_rect", [0, 0.03, 1, 1]),
            "tight_rect_microstate": layout.get("tight_rect_microstate", [0, 0.02, 1, 0.96]),
        }
        gridspec_config = layout.get("gridspec", {})
        gridspec_params = {
            "width_ratios": gridspec_config.get("width_ratios", [4, 1]),
            "height_ratios": gridspec_config.get("height_ratios", [1, 4]),
            "hspace": gridspec_config.get("hspace", 0.15),
            "wspace": gridspec_config.get("wspace", 0.15),
            "left": gridspec_config.get("left", 0.1),
            "right": gridspec_config.get("right", 0.95),
            "top": gridspec_config.get("top", 0.80),
            "bottom": gridspec_config.get("bottom", 0.12),
        }
        
        figure_sizes = {}
        for name, size in figure_sizes_dict.items():
            if isinstance(size, list) and len(size) == 2:
                figure_sizes[name] = tuple(size)
            elif isinstance(size, dict):
                for sub_name, sub_size in size.items():
                    if isinstance(sub_size, (int, float)):
                        figure_sizes[f"{name}_{sub_name}"] = (sub_size, sub_size)
                    elif isinstance(sub_size, list) and len(sub_size) == 2:
                        figure_sizes[f"{name}_{sub_name}"] = tuple(sub_size)
        
        font_dict = defaults.get("font", {})
        font_sizes = font_dict.get("sizes", {})
        font = FontConfig(
            small=font_sizes.get("small", 7),
            medium=font_sizes.get("medium", 8),
            large=font_sizes.get("large", 9),
            title=font_sizes.get("title", 10),
            annotation=font_sizes.get("annotation", 4),
            label=font_sizes.get("label", 8),
            ylabel=font_sizes.get("ylabel", 10),
            suptitle=font_sizes.get("suptitle", 11),
            figure_title=font_sizes.get("figure_title", 12),
            family=font_dict.get("family", "sans-serif"),
            weight=font_dict.get("weight", "normal"),
        )
        
        scatter_dict = styling.get("scatter", {})
        marker_sizes = scatter_dict.get("marker_size", {})
        scatter = ScatterStyle(
            marker_size_small=marker_sizes.get("small", 3),
            marker_size_large=marker_sizes.get("large", 30),
            marker_size_default=marker_sizes.get("default", 30),
            alpha=scatter_dict.get("alpha", 0.7),
            edgecolor=scatter_dict.get("edgecolor", "white"),
            edgewidth=scatter_dict.get("edgewidth", 0.3),
        )
        
        bar_dict = styling.get("bar", {})
        bar = BarStyle(
            alpha=bar_dict.get("alpha", 0.8),
            width=bar_dict.get("width", 0.6),
            capsize=bar_dict.get("capsize", 4),
            capsize_large=bar_dict.get("capsize_large", 3),
        )
        
        line_dict = styling.get("line", {})
        line_widths = line_dict.get("width", {})
        line_alphas = line_dict.get("alpha", {})
        line = LineStyle(
            width_thin=line_widths.get("thin", 0.5),
            width_standard=line_widths.get("standard", 0.75),
            width_thick=line_widths.get("thick", 1.0),
            width_bold=line_widths.get("bold", 1.5),
            alpha_standard=line_alphas.get("standard", 0.8),
            alpha_dim=line_alphas.get("dim", 0.5),
            alpha_zero=line_alphas.get("zero_line", 0.3),
            alpha_fit=line_alphas.get("fit_line", 0.8),
            alpha_diagonal=line_alphas.get("diagonal", 0.5),
            alpha_reference=line_alphas.get("reference", 0.5),
            regression_width=line_dict.get("regression_width", 1.5),
            residual_width=line_dict.get("residual_width", 1.0),
            qq_width=line_dict.get("qq_width", 0.4),
        )
        
        hist_dict = styling.get("histogram", {})
        histogram = HistogramStyle(
            bins=hist_dict.get("bins", 30),
            bins_behavioral=hist_dict.get("bins_behavioral", 15),
            bins_residual=hist_dict.get("bins_residual", 20),
            bins_tfr=hist_dict.get("bins_tfr", 50),
            edgecolor=hist_dict.get("edgecolor", "white"),
            edgewidth=hist_dict.get("edgewidth", 0.5),
            alpha=hist_dict.get("alpha", 0.7),
            alpha_residual=hist_dict.get("alpha_residual", 0.75),
            alpha_tfr=hist_dict.get("alpha_tfr", 0.8),
        )
        
        kde_dict = styling.get("kde", {})
        colors_dict = styling.get("colors", {})
        colors = ColorPalette(
            gray=colors_dict.get("gray", "#555555"),
            light_gray=colors_dict.get("light_gray", "#999999"),
            black=colors_dict.get("black", "k"),
            red=colors_dict.get("red", "#d62728"),
            blue=colors_dict.get("blue", "#1f77b4"),
            significant=colors_dict.get("significant", "#C42847"),
            nonsignificant=colors_dict.get("nonsignificant", "#666666"),
            pain=colors_dict.get("pain", "crimson"),
            nonpain=colors_dict.get("nonpain", "navy"),
            network_node=colors_dict.get("network_node", "#87CEEB"),
        )
        
        alpha_dict = styling.get("alpha", {})
        style = PlotStyleConfig(
            scatter=scatter,
            bar=bar,
            line=line,
            histogram=histogram,
            kde_points=kde_dict.get("points", 100),
            kde_color=kde_dict.get("color", "darkblue"),
            kde_linewidth=kde_dict.get("linewidth", 1.5),
            kde_alpha=kde_dict.get("alpha", 0.8),
            errorbar_markersize=styling.get("errorbar", {}).get("markersize", 4),
            errorbar_capsize=styling.get("errorbar", {}).get("capsize", 2),
            errorbar_capsize_large=styling.get("errorbar", {}).get("capsize_large", 3),
            colors=colors,
            alpha_grid=alpha_dict.get("grid", 0.3),
            alpha_fill=alpha_dict.get("fill", 0.6),
            alpha_text_box=alpha_dict.get("text_box", 0.8),
            alpha_violin_body=alpha_dict.get("violin_body", 0.6),
            alpha_ridge_fill=alpha_dict.get("ridge_fill", 0.6),
            alpha_ci=alpha_dict.get("ci", 0.1),
            alpha_ci_line=alpha_dict.get("ci_line", 0.5),
        )
        
        text_pos_dict = styling.get("text_position", {})
        text_position = TextPosition(
            stats_x=text_pos_dict.get("stats_x", 0.05),
            stats_y=text_pos_dict.get("stats_y", 0.95),
            p_value_x=text_pos_dict.get("p_value_x", 0.98),
            p_value_y=text_pos_dict.get("p_value_y", 0.98),
            bootstrap_x=text_pos_dict.get("bootstrap_x", 0.02),
            bootstrap_y=text_pos_dict.get("bootstrap_y", 0.98),
            channel_annotation_x=text_pos_dict.get("channel_annotation_x", 0.02),
            channel_annotation_y=text_pos_dict.get("channel_annotation_y", 0.94),
            title_y=text_pos_dict.get("title_y", 0.975),
            residual_qc_title_y=text_pos_dict.get("residual_qc_title_y", 1.02),
        )
        
        return cls(
            dpi=defaults.get("dpi", 300),
            savefig_dpi=defaults.get("savefig_dpi", 600),
            formats=tuple(defaults.get("formats", ["png", "svg"])),
            bbox_inches=defaults.get("bbox_inches", "tight"),
            pad_inches=defaults.get("pad_inches", 0.02),
            font=font,
            style=style,
            text_position=text_position,
            figure_sizes=figure_sizes,
            plot_type_configs=plot_types,
            validation=validation_dict,
            layout_rects=layout_rects,
            gridspec_params=gridspec_params,
        )
    
    @classmethod
    def get_defaults(cls) -> "PlotConfig":
        """Get default configuration by loading from config file.
        
        This ensures eeg_config.yaml is the single source of truth.
        If config file cannot be loaded, raises an error to prevent
        silent fallback to hardcoded values.
        """
        try:
            from ..utils.config.loader import load_config
            config = load_config()
            return cls.from_config({"plotting": config.get("plotting", {})})
        except Exception as e:
            logger.error(
                f"Failed to load plotting config from eeg_config.yaml: {e}. "
                "Please ensure eeg_config.yaml exists and contains a 'plotting' section."
            )
            raise RuntimeError(
                "Cannot load plotting configuration. "
                "eeg_config.yaml must be the single source of truth. "
                f"Error: {e}"
            ) from e
    
    def _get_plot_type_override(self, plot_type: Optional[str], key: str) -> Optional[Any]:
        if plot_type and plot_type in self.plot_type_configs:
            return self.plot_type_configs[plot_type].get(key)
        return None
    
    def get_figure_size(self, size_name: str, plot_type: Optional[str] = None) -> Tuple[float, float]:
        override = self._get_plot_type_override(plot_type, "figure_size")
        if override:
            size_name = override
        
        return self.figure_sizes.get(size_name, self.figure_sizes.get("standard", (10.0, 8.0)))
    
    def get_scatter_marker_size(self, plot_type: Optional[str] = None) -> int:
        override = self._get_plot_type_override(plot_type, "scatter_marker_size")
        if override == "small":
            return self.style.scatter.marker_size_small
        if override == "large":
            return self.style.scatter.marker_size_large
        return self.style.scatter.marker_size_default
    
    def get_line_width(self, width_name: str, plot_type: Optional[str] = None) -> float:
        override = self._get_plot_type_override(plot_type, "line_width")
        if override:
            width_name = override
        
        width_map = {
            "thin": self.style.line.width_thin,
            "standard": self.style.line.width_standard,
            "thick": self.style.line.width_thick,
            "bold": self.style.line.width_bold,
        }
        return width_map.get(width_name, self.style.line.width_standard)
    
    def get_color(self, color_name: str, plot_type: Optional[str] = None) -> str:
        if plot_type and plot_type in self.plot_type_configs:
            colors = self.plot_type_configs[plot_type].get("colors", {})
            if color_name in colors:
                return colors[color_name]
        
        color_map = {
            "gray": self.style.colors.gray,
            "light_gray": self.style.colors.light_gray,
            "black": self.style.colors.black,
            "red": self.style.colors.red,
            "blue": self.style.colors.blue,
            "significant": self.style.colors.significant,
            "nonsignificant": self.style.colors.nonsignificant,
            "pain": self.style.colors.pain,
            "nonpain": self.style.colors.nonpain,
            "network_node": self.style.colors.network_node,
        }
        return color_map.get(color_name, self.style.colors.gray)
    
    def get_histogram_bins(self, plot_type: Optional[str] = None) -> int:
        """Get histogram bins based on plot type.
        
        Args:
            plot_type: Plot type (e.g., "behavioral", "tfr")
        
        Returns:
            Number of bins
        """
        if plot_type == "behavioral":
            return self.style.histogram.bins_behavioral
        elif plot_type == "tfr":
            return self.style.histogram.bins_tfr
        elif plot_type == "residual":
            return self.style.histogram.bins_residual
        return self.style.histogram.bins
    
    def get_layout_rect(self, rect_name: str = "tight_rect") -> list:
        """Get layout rectangle from stored config.
        
        Args:
            rect_name: Name of rectangle ("tight_rect" or "tight_rect_microstate")
        
        Returns:
            List [left, bottom, right, top]
        """
        return self.layout_rects.get(rect_name, [0, 0.03, 1, 1] if rect_name == "tight_rect" else [0, 0.02, 1, 0.96])
    
    def get_gridspec_params(self) -> Dict[str, Any]:
        """Get gridspec layout parameters from stored config.
        
        Returns:
            Dictionary with gridspec parameters (width_ratios, height_ratios, hspace, wspace, left, right, top, bottom)
        """
        return self.gridspec_params.copy()
    
    def get_behavioral_config(self) -> Dict[str, Any]:
        """Get behavioral plotting configuration.
        
        Returns behavioral-specific configuration dictionary from plot_type_configs.
        This method centralizes behavioral config access to break circular dependencies.
        
        Returns:
            Dictionary containing behavioral plot configuration parameters.
            Returns empty dict if "behavioral" key is not present in plot_type_configs.
        """
        return self.plot_type_configs.get("behavioral", {})


###################################################################
# Config Loading
###################################################################


_plot_config_cache: Optional[PlotConfig] = None


def get_plot_config(config: Optional[Any] = None) -> PlotConfig:
    """Get plotting configuration, with caching.
    
    Args:
        config: ConfigDict or dict from load_config(). If None, uses defaults.
    
    Returns:
        PlotConfig instance
    
    Examples:
        >>> from eeg_pipeline.utils.config.loader import load_config
        >>> config = load_config()
        >>> plot_cfg = get_plot_config(config)
        >>> fig_size = plot_cfg.get_figure_size("square", plot_type="decoding")
    """
    global _plot_config_cache
    
    if config is None:
        if _plot_config_cache is None:
            _plot_config_cache = PlotConfig.get_defaults()
        return _plot_config_cache
    
    config_dict = {}
    
    if isinstance(config, dict):
        config_dict = config
    elif hasattr(config, "get"):
        try:
            plotting_dict = config.get("plotting", {})
            if isinstance(plotting_dict, dict):
                config_dict = {"plotting": plotting_dict}
            else:
                config_dict = {}
        except (TypeError, AttributeError):
            logger.warning("Could not extract plotting config, using defaults")
            return PlotConfig.get_defaults()
    else:
        try:
            if hasattr(config, "__dict__"):
                config_dict = dict(config)
            elif hasattr(config, "items"):
                config_dict = dict(config)
        except (TypeError, AttributeError):
            logger.warning("Could not convert config to dict, using defaults")
            return PlotConfig.get_defaults()
    
    _plot_config_cache = PlotConfig.from_config(config_dict)
    return _plot_config_cache


def reset_plot_config_cache() -> None:
    """Reset the global plot config cache. Useful for testing."""
    global _plot_config_cache
    _plot_config_cache = None

