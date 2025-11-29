from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Union, Iterable, Dict, Any
from dataclasses import dataclass
import functools
import logging
import os
import glob
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
import json
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from ..analysis.stats import fdr_bh, fdr_bh_reject

try:
    from ..config.loader import load_settings, ConfigDict, get_config_value, get_constants
except ImportError:
    load_settings = None
    ConfigDict = dict
    def get_config_value(config, key, default):
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        if isinstance(config, dict):
            keys = key.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        return default
    get_constants = None

EEGConfig = ConfigDict



def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    df.to_csv(path, sep="\t", index=index)


def find_first(pattern: str) -> Optional[Path]:
    candidates = sorted(glob.glob(pattern))
    return Path(candidates[0]) if candidates else None


def _resolve_log_dir(config: Optional[EEGConfig] = None) -> Optional[Path]:
    if config is None:
        return None
    
    log_dir = config.get("output.log_dir")
    if log_dir:
        return Path(log_dir)
    
    deriv_root = config.get("paths.deriv_root")
    if deriv_root:
        return Path(deriv_root) / "logs"
    
    return None


# Track which loggers have been configured to prevent duplicate handlers
_configured_loggers: set = set()


def _configure_logger_handlers(
    logger: logging.Logger,
    log_file_path: Optional[Path] = None
) -> None:
    """
    Configure handlers for a logger, preventing duplicates.
    
    This function:
    1. Tracks configured loggers to prevent duplicate handler setup
    2. Disables propagation to prevent duplicate messages from parent loggers
    3. Uses a consistent format across all loggers
    """
    # Skip if already configured
    if logger.name in _configured_loggers:
        return
    
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    # Disable propagation to parent loggers to prevent duplicate messages
    logger.propagate = False
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file_path is not None:
        ensure_dir(log_file_path.parent)
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(logging.INFO)
    _configured_loggers.add(logger.name)


def get_logger(
    name: str,
    log_file_path: Optional[Path] = None
) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Ensures no duplicate handlers are added and propagation is disabled.
    """
    logger = logging.getLogger(name)
    _configure_logger_handlers(logger, log_file_path)
    return logger


def get_module_logger(logger: Optional[logging.Logger] = None, module_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a module, using provided logger if available.
    
    This is the preferred way to get a logger within functions that accept
    an optional logger parameter.
    """
    if logger is not None:
        return logger
    return get_logger(module_name or __name__)


def log_and_raise_error(logger: logging.Logger, error_msg: str, exception_class=ValueError) -> None:
    logger.error(error_msg)
    raise exception_class(error_msg)


def reset_logging() -> None:
    """
    Reset the logging configuration.
    
    Clears all configured loggers and their handlers. Useful for testing
    or when reconfiguring logging at runtime.
    """
    global _configured_loggers
    for name in list(_configured_loggers):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    _configured_loggers.clear()


def _get_log_file_path(logger_name: str, log_file_name: Optional[str], config: Optional[EEGConfig]) -> Optional[Path]:
    if not log_file_name and not config:
        return None
    log_dir = _resolve_log_dir(config)
    if not log_dir:
        return None
    filename = log_file_name or f"{logger_name}.log"
    return log_dir / filename


def get_subject_logger(
    module_name: str,
    subject: str,
    log_file_name: Optional[str] = None,
    config: Optional[EEGConfig] = None
) -> logging.Logger:
    logger_name = f"{module_name}.sub-{subject}"
    log_file_path = _get_log_file_path(logger_name, log_file_name, config)
    return get_logger(logger_name, log_file_path)


def get_group_logger(
    module_name: str,
    log_file_name: Optional[str] = None,
    config: Optional[EEGConfig] = None
) -> logging.Logger:
    logger_name = f"{module_name}.group"
    log_file_path = _get_log_file_path(logger_name, log_file_name, config)
    return get_logger(logger_name, log_file_path)


def get_pipeline_logger(
    module_name: Optional[str] = None,
    config: Optional[EEGConfig] = None
) -> logging.Logger:
    logger_name = module_name or "eeg_pipeline"
    log_file_path = _get_log_file_path(logger_name, None, config) if config else None
    return get_logger(logger_name, log_file_path)


def setup_logger(
    config: Optional[EEGConfig] = None,
    subject: Optional[str] = None
) -> logging.Logger:
    if subject:
        return get_subject_logger("pipeline", subject, config=config)
    return get_pipeline_logger(config=config)


###################################################################
# Data Validation
###################################################################

def validate_epochs_for_plotting(epochs: mne.Epochs, logger: Optional[logging.Logger] = None) -> bool:
    if epochs is None:
        if logger:
            logger.warning("Epochs object is None")
        return False
    if len(epochs) == 0:
        if logger:
            logger.warning("Epochs object is empty")
        return False
    return True


def require_epochs_tfr(tfr, context_msg: str, logger: Optional[logging.Logger] = None) -> bool:
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        if logger:
            logger.warning(f"{context_msg} requires EpochsTFR; skipping.")
        return False
    return True


def detect_data_format(
    data: np.ndarray, 
    data_format: Optional[str] = None, 
    percent_threshold: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    if data_format is not None:
        return data_format == "percent"
    
    if percent_threshold is None:
        if config is None and load_settings is not None:
            config = load_settings()
        
        # Use config value or fallback
        # Ideally this constant should be in config.io.constants
        if config:
             percent_threshold = float(get_config_value(config, "io.constants.percent_threshold", 5.0))
        else:
             percent_threshold = 5.0
    
    data_finite = data[np.isfinite(data)]
    if data_finite.size == 0:
        return False
    data_abs_max = float(np.nanmax(np.abs(data_finite)))
    return data_abs_max > percent_threshold


def format_baseline_window_string(baseline_used: Tuple[float, float]) -> str:
    b_start, b_end = baseline_used
    return f"bl{abs(b_start):.1f}to{abs(b_end):.2f}"


###################################################################
# Typed Return Values
###################################################################

@dataclass
class EEGDatasetResult:
    epochs: Optional[mne.Epochs]
    events: Optional[pd.DataFrame]
    
    @property
    def empty(self) -> bool:
        return self.epochs is None or self.events is None or len(self.events) == 0


###################################################################
# Caching
###################################################################


###################################################################
# BIDS-Centric Path Helpers
###################################################################

def _normalize_subject_label(subject: str) -> str:
    if subject.startswith("sub-"):
        return subject
    return f"sub-{subject}"


def bids_sub_eeg_path(bids_root: Path, subject: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return Path(bids_root) / subject_label / "eeg"


def bids_events_path(bids_root: Path, subject: str, task: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return bids_sub_eeg_path(bids_root, subject) / f"{subject_label}_task-{task}_events.tsv"


def deriv_sub_eeg_path(deriv_root: Path, subject: str) -> Path:
    subject_label = _normalize_subject_label(subject)
    return Path(deriv_root) / subject_label / "eeg"


def deriv_features_path(deriv_root: Path, subject: str) -> Path:
    return deriv_sub_eeg_path(deriv_root, subject) / "features"


def deriv_stats_path(deriv_root: Path, subject: str) -> Path:
    return deriv_sub_eeg_path(deriv_root, subject) / "stats"


def deriv_plots_path(deriv_root: Path, subject: str, subdir: Optional[str] = None) -> Path:
    plots_dir = deriv_sub_eeg_path(deriv_root, subject) / "plots"
    if subdir:
        return plots_dir / subdir
    return plots_dir


def deriv_group_eeg_path(deriv_root: Path) -> Path:
    return Path(deriv_root) / "group" / "eeg"


def deriv_group_stats_path(deriv_root: Path) -> Path:
    return deriv_group_eeg_path(deriv_root) / "stats"


def deriv_group_plots_path(deriv_root: Path, subdir: Optional[str] = None) -> Path:
    plots_root = deriv_group_eeg_path(deriv_root) / "plots"
    if subdir:
        return plots_root / subdir
    return plots_root


def find_connectivity_features_path(deriv_root: Path, subject: str) -> Path:
    sub = _normalize_subject_label(subject)
    parquet_path = Path(deriv_root) / sub / "eeg" / "connectivity_features.parquet"
    if parquet_path.exists():
        return parquet_path
    return (
        Path(deriv_root)
        / sub
        / "eeg"
        / "features"
        / "features_connectivity.tsv"
    )


def _resolve_deriv_root(
    deriv_root: Optional[Path],
    config: Optional[EEGConfig] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> Path:
    if deriv_root is not None:
        return Path(deriv_root)
    
    if config is not None:
        try:
            deriv_path = config.deriv_root
            if deriv_path is not None:
                return Path(deriv_path)
        except AttributeError:
            pass
    
    if constants is not None and "DERIV_ROOT" in constants:
        return Path(constants["DERIV_ROOT"])
    
    raise ValueError(
        "Either deriv_root, config, or constants must be provided to resolve derivatives root"
    )


def _normalize_subject_id(subject: str) -> str:
    if subject.startswith("sub-"):
        return subject.replace("sub-", "")
    return subject


def _check_clean_tokens(filename: str) -> bool:
    clean_tokens = ("proc-clean", "proc-cleaned", "clean")
    return any(token in filename for token in clean_tokens)


def _search_standard_bids_paths(
    root: Path,
    subject_clean: str,
    task: str,
) -> Optional[Path]:
    bids_path = BIDSPath(
        subject=subject_clean,
        task=task,
        datatype="eeg",
        processing="clean",
        suffix="epo",
        extension=".fif",
        root=root,
        check=False,
    )
    if bids_path.fpath and bids_path.fpath.exists():
        return bids_path.fpath
    
    standard_paths = [
        root / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_proc-clean_epo.fif",
        root / "preprocessed" / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_proc-clean_epo.fif",
    ]
    
    for path in standard_paths:
        if path.exists():
            return path
    
    return None


def _search_directory_for_epochs(
    directory: Path,
    subject_clean: str,
    task: str,
    prefer_clean: bool = True,
) -> Optional[Path]:
    if not directory.exists():
        return None
    
    pattern = f"sub-{subject_clean}_task-{task}*epo.fif"
    candidates = sorted(directory.glob(pattern))
    
    if not candidates:
        return None
    
    if prefer_clean:
        for candidate in candidates:
            if _check_clean_tokens(candidate.name):
                return candidate
    
    return candidates[0]


def _find_clean_epochs_path(
    subject: str,
    task: str,
    deriv_root: Optional[Path] = None,
    constants: Optional[Dict[str, Any]] = None,
    config: Optional[EEGConfig] = None,
) -> Optional[Path]:
    root = _resolve_deriv_root(deriv_root, config, constants)
    subject_clean = _normalize_subject_id(subject)
    
    standard_path = _search_standard_bids_paths(root, subject_clean, task)
    if standard_path:
        return standard_path
    
    search_directories = [
        (root / f"sub-{subject_clean}" / "eeg", True),
        (root / "preprocessed" / f"sub-{subject_clean}" / "eeg", True),
        (root / f"sub-{subject_clean}", False),
        (root / "preprocessed", True),
    ]
    
    for directory, prefer_clean in search_directories:
        found_path = _search_directory_for_epochs(directory, subject_clean, task, prefer_clean)
        if found_path:
            return found_path
    
    return None


def _resolve_bids_root(
    bids_root: Optional[Path],
    constants: Optional[Dict[str, Any]],
    config: Optional[EEGConfig],
) -> Path:
    if bids_root is not None:
        return Path(bids_root)
    
    if config is not None:
        return Path(config.bids_root)
    
    if constants is not None:
        return Path(constants["BIDS_ROOT"])
    
    raise ValueError("Either bids_root, constants, or config must be provided to resolve BIDS root")


def _load_events_df(
    subject: str,
    task: str,
    bids_root: Optional[Path] = None,
    constants=None,
    config: Optional[EEGConfig] = None,
) -> pd.DataFrame | None:
    root = _resolve_bids_root(bids_root, constants, config)
    subject_clean = _normalize_subject_id(subject)
    
    events_path = _find_events_path(root, subject_clean, task)
    if events_path is None or not events_path.exists():
        return None
    
    return pd.read_csv(events_path, sep="\t")


def _find_events_path(bids_root: Path, subject_clean: str, task: str) -> Optional[Path]:
    bids_path = BIDSPath(
        subject=subject_clean,
        task=task,
        datatype="eeg",
        suffix="events",
        extension=".tsv",
        root=bids_root,
        check=False,
    )
    
    if bids_path.fpath is not None:
        return bids_path.fpath
    
    fallback_path = bids_root / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_events.tsv"
    return fallback_path




###################################################################
# Event Column Validation
###################################################################


def reconstruct_kept_indices(dropped_trials_path: Path, n_events: int) -> np.ndarray:
    if not dropped_trials_path.exists():
        return np.arange(n_events)
    
    dropped_df = pd.read_csv(dropped_trials_path, sep="\t")
    if "original_index" not in dropped_df.columns or len(dropped_df) == 0:
        return np.arange(n_events)
    
    dropped_indices_raw = pd.to_numeric(dropped_df["original_index"], errors="coerce").dropna()
    if len(dropped_indices_raw) == 0:
        return np.arange(n_events)
    
    dropped_indices = set(dropped_indices_raw.astype(int).tolist())
    kept_indices = np.array([i for i in range(n_events) if i not in dropped_indices])
    return kept_indices


def get_pain_window(constants=None, config: Optional[EEGConfig] = None) -> Tuple[float, float]:
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window")
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to get_pain_window")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError(
            "PLATEAU_WINDOW not found in constants. "
            "Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)"
        )
    
    return constants["PLATEAU_WINDOW"]


WINDOW_PAIN = get_pain_window


def _pick_target_column(df: pd.DataFrame, constants: Dict[str, Any]) -> Optional[str]:
    if constants is None:
        raise ValueError("constants is required for _pick_target_column")
    
    target_columns = tuple(constants.get("TARGET_COLUMNS", ()))
    for col in target_columns:
        if col in df.columns:
            return col
    
    for col in df.columns:
        col_lower = str(col).lower()
        if ("vas" in col_lower or "rating" in col_lower) and pd.api.types.is_numeric_dtype(df[col]):
            return col
    
    return None




def ensure_derivatives_dataset_description(deriv_root: Optional[Path] = None, constants=None, config=None) -> None:
    root = _resolve_deriv_root(deriv_root, config, constants)
    
    desc_path = root / "dataset_description.json"
    if desc_path.exists():
        return
    
    meta = {
        "Name": "EEG Pipeline Derivatives",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "EEG_fMRI_Analysis Pipeline",
                "Version": "unknown",
                "Description": "Custom EEG analysis (ERP, TFR, features, decoding)",
            }
        ],
    }
    ensure_dir(root)
    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _handle_alignment_error(msg: str, strict: bool, logger: Optional[logging.Logger]) -> None:
    if strict:
        raise ValueError(msg)
    if logger:
        logger.warning(msg)


def ensure_aligned_lengths(
    *arrays_or_series,
    context: str = "",
    strict: Optional[bool] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    strict = strict if strict is not None else True
    
    non_null = [obj for obj in arrays_or_series if obj is not None]
    if len(non_null) < 2:
        return
    
    lengths = {len(obj) for obj in non_null}
    if len(lengths) > 1:
        msg = f"{context}: Length mismatch detected: {[len(obj) for obj in non_null]}"
        _handle_alignment_error(msg, strict, logger)
        return
    
    ref_index = None
    for obj in non_null:
        idx = getattr(obj, "index", None)
        if idx is None:
            continue
        if ref_index is None:
            ref_index = idx
        elif len(idx) != len(ref_index) or not idx.equals(ref_index):
            msg = f"{context}: Index misalignment detected; align before analysis"
            _handle_alignment_error(msg, strict, logger)
            return



###################################################################
# Figure Saving
###################################################################

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
        from ..config.loader import load_settings
        cfg = load_settings()
        return build_footer(footer_template_name, cfg, **(footer_kwargs or {}))
    except (ImportError, KeyError, ValueError, AttributeError):
        return None


def _should_add_footer(footer: Optional[str]) -> bool:
    if footer is None:
        return False
    
    footer_env = os.getenv("FIG_FOOTER_OFF", "0").lower()
    return footer_env not in {"1", "true", "yes"}


def _apply_tight_layout(fig: plt.Figure, rect: Optional[Tuple[float, float, float, float]]) -> None:
    import warnings
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
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches
        )
        return True
    except AttributeError as e:
        if 'copy_from_bbox' not in str(e):
            raise
        try:
            return _save_figure_with_agg_backend(fig, output_path, dpi, bbox_inches, pad_inches)
        except Exception as e2:
            logging.getLogger(__name__).warning(f"Failed to save figure {output_path} with fallback backend: {e2}")
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
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches
        )
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
    
    import warnings
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
    
    formats = formats or _PLOT_DEFAULT_FORMATS
    dpi = dpi if dpi is not None else _PLOT_DEFAULT_DPI
    bbox_inches = bbox_inches or _PLOT_DEFAULT_BBOX
    pad_inches = pad_inches if pad_inches is not None else _PLOT_DEFAULT_PAD

    footer = _prepare_figure_footer(footer, footer_template_name, footer_kwargs, constants)
    _prepare_figure_layout(fig, footer, tight_layout_rect)

    saved_paths = _save_figure_to_formats(fig, base_path, formats, dpi, bbox_inches, pad_inches)
    
    plt.close(fig)
    
    if logger is not None:
        filenames = ", ".join(p.name for p in saved_paths)
        logger.info(f"  Saved: {filenames}")


###################################################################
# Plot Config and Footer Helpers
###################################################################

_PLOT_DEFAULT_DPI = None
_PLOT_DEFAULT_FORMATS = None
_PLOT_DEFAULT_BBOX = None
_PLOT_DEFAULT_PAD = None


@dataclass
class PlotConfig:
    dpi: int = 300
    formats: Tuple[str, ...] = ("png",)
    bbox_inches: str = "tight"
    pad_inches: float = 0.02
    
    @classmethod
    def from_constants(cls, constants: Dict[str, Any]) -> "PlotConfig":
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
    def get_defaults(cls) -> "PlotConfig":
        return cls()


def _get_plot_constants(constants=None):
    global _PLOT_DEFAULT_DPI, _PLOT_DEFAULT_FORMATS, _PLOT_DEFAULT_BBOX, _PLOT_DEFAULT_PAD
    if _PLOT_DEFAULT_DPI is None:
        if constants is None:
            raise ValueError("constants is required for _get_plot_constants")
        plot_config = PlotConfig.from_constants(constants)
        _PLOT_DEFAULT_DPI = plot_config.dpi
        _PLOT_DEFAULT_FORMATS = plot_config.formats
        _PLOT_DEFAULT_BBOX = plot_config.bbox_inches
        _PLOT_DEFAULT_PAD = plot_config.pad_inches


def build_footer(template_name: str, config, **kwargs) -> str:
    if config is None:
        raise ValueError("config is required for build_footer")
    templates = config.get("visualization.footer_templates", {})
    if template_name not in templates:
        raise ValueError(f"Footer template '{template_name}' not found in config. Available: {list(templates.keys())}")
    template = templates[template_name]
    return template.format(**kwargs)


def unwrap_figure(obj):
    return obj[0] if isinstance(obj, list) else obj


def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(label))


def get_behavior_footer(config) -> str:
    bwin = tuple(config.get("time_frequency_analysis.baseline_window"))
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha")
    return f"Baseline: [{float(bwin[0]):.2f}, {float(bwin[1]):.2f}] s | Significance: BH-FDR α={fdr_alpha}"


def get_band_color(band: str, config=None) -> str:
    if config is None:
        raise ValueError("config is required for get_band_color")
    colors = config.get("visualization.band_colors")
    if colors is None:
        raise ValueError("visualization.band_colors not found in config")
    if band in colors:
        return str(colors[band])
    raise ValueError(f"Band '{band}' not found in visualization.band_colors config")


def logratio_to_pct(v):
    v_arr = np.asarray(v, dtype=float)
    return (np.power(10.0, v_arr) - 1.0) * 100.0


def pct_to_logratio(p):
    p_arr = np.asarray(p, dtype=float)
    return np.log10(np.clip(1.0 + (p_arr / 100.0), 1e-9, None))


###################################################################
# Topomap Plotting Helpers and Formatting
###################################################################

def get_viz_params(config=None):
    if config is None:
        try:
            from ..config.loader import load_settings
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
    
    topomap_config = config.get("plotting.topomap") or config.get("plotting.plots.topomap", {})
    
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
        data, info, axes=ax, show=False, mask=mask,
        mask_params=mask_params or {}, sensors=True, contours=viz_params["topo_contours"],
        cmap=viz_params["topo_cmap"], vlim=(vmin, vmax) if vmin is not None and vmax is not None else None
    )
    if viz_params["annotate_descriptive_topo"] and hasattr(ax, 'figure'):
        fig = ax.figure
        if not getattr(fig, '_descriptive_note_added', False):
            fig.text(0.02, 0.02, "Descriptive topomap; see stats for inference (FDR/cluster)",
                     fontsize=7, ha='left', va='bottom', alpha=0.7)
            setattr(fig, '_descriptive_note_added', True)


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
        "adaptive_multiplier": 2.0
    }
    
    if config is None and load_settings is not None:
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


###################################################################
# Matplotlib Setup
###################################################################

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
        from ..config.loader import get_nested_value
        dpi = get_nested_value(config, "visualization.dpi", 300)

    sns.set_theme(context="paper", style="white", font_scale=1.05)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "savefig.dpi": dpi,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    })


###################################################################
# Column Finding Utilities
###################################################################

@functools.lru_cache(maxsize=None)
def _get_io_constants(config=None):
    if get_constants is not None:
        return get_constants("io", config)
    
    if config is None:
        if load_settings is not None:
            config = load_settings()
    
    if config is None:
        raise ValueError("Config is required. Cannot load IO constants without config.")
    
    constants = config.get("io.constants")
    if constants is None:
        raise ValueError("io.constants not found in config.")
    
    return {
        "temperature_column_names": constants["temperature_column_names"],
        "pain_column_names": constants["pain_column_names"],
    }


def _get_config_key_for_column_type(column_type: Optional[str]) -> Optional[str]:
    config_key_map = {
        "pain": "event_columns.pain_binary",
        "temperature": "event_columns.temperature"
    }
    return config_key_map.get(column_type) if column_type else None

def _find_column_in_dataframe(df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    return next((col for col in column_names if col in df.columns), None)


def _get_column_names_for_type(column_type: str, config: Optional[Any] = None) -> Optional[List[str]]:
    constants = _get_io_constants(config)
    column_map = {
        "pain": constants["pain_column_names"],
        "temperature": constants["temperature_column_names"]
    }
    return column_map.get(column_type)


def find_column_in_events(events_df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    return _find_column_in_dataframe(events_df, column_names)


def find_pain_column_in_events(events_df: pd.DataFrame, config: Optional[Any] = None) -> Optional[str]:
    column_names = _get_column_names_for_type("pain", config)
    if not column_names:
        return None
    return _find_column_in_dataframe(events_df, column_names)


def find_temperature_column_in_events(events_df: pd.DataFrame, config: Optional[Any] = None) -> Optional[str]:
    column_names = _get_column_names_for_type("temperature", config)
    if not column_names:
        return None
    return _find_column_in_dataframe(events_df, column_names)


def find_column_in_metadata(epochs: mne.Epochs, config_key: str, config) -> Optional[str]:
    if not hasattr(epochs, "metadata") or epochs.metadata is None:
        return None
    column_names = config.get(config_key)
    if not column_names:
        return None
    return _find_column_in_dataframe(epochs.metadata, column_names)


def find_pain_column_in_metadata(epochs: mne.Epochs, config) -> Optional[str]:
    config_key = _get_config_key_for_column_type("pain")
    if not config_key:
        return None
    return find_column_in_metadata(epochs, config_key, config)


def find_temperature_column_in_metadata(epochs: mne.Epochs, config) -> Optional[str]:
    config_key = _get_config_key_for_column_type("temperature")
    if not config_key:
        return None
    return find_column_in_metadata(epochs, config_key, config)


def get_column_from_config(
    config: Any,
    column_key: str,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    if config is None:
        raise ValueError("config is required")
    
    columns = config.get(column_key)
    if columns is None:
        raise ValueError(f"{column_key} not found in config")
    if not columns:
        return None
    
    if events_df is not None:
        return _find_column_in_dataframe(events_df, columns)
    
    return columns[0] if columns else None


def get_pain_column_from_config(config, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    return get_column_from_config(config, "event_columns.pain_binary", events_df)


def get_temperature_column_from_config(config, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    return get_column_from_config(config, "event_columns.temperature", events_df)


###################################################################
# Plotting Constants Extraction
###################################################################

def extract_plotting_constants(config, save_formats: Optional[List[str]] = None):
    if config is None:
        raise ValueError("config is required for extract_plotting_constants")
    
    return {
        "FIG_DPI": int(config.get("output.fig_dpi")),
        "SAVE_FORMATS": save_formats or list(config.get("output.save_formats")),
        "output.bbox_inches": config.get("output.bbox_inches"),
        "output.pad_inches": float(config.get("output.pad_inches")),
    }


###################################################################
# EEG Channel Selection Utilities
###################################################################

def extract_eeg_picks(epochs_or_info, exclude_bads: bool = True):
    if hasattr(epochs_or_info, "info"):
        info = epochs_or_info.info
    else:
        info = epochs_or_info
    
    exclude = "bads" if exclude_bads else []
    return mne.pick_types(
        info, eeg=True, meg=False, eog=False, stim=False, exclude=exclude
    )


###################################################################
# Plotting Helper Utilities
###################################################################

def format_baseline_string(baseline_window: Tuple[float, float]) -> str:
    return f"[{baseline_window[0]:.2f}, {baseline_window[1]:.2f}]"


def log_if_present(logger, level: str, message: str):
    if logger:
        getattr(logger, level)(message)


def validate_picks(picks, logger):
    if len(picks) == 0:
        log_if_present(logger, "warning", "No valid EEG channels found")
        return False
    return True


def get_default_logger() -> logging.Logger:
    return logging.getLogger(__name__)


def get_default_config():
    if load_settings is None:
        raise ImportError("load_settings not available")
    return load_settings()


###################################################################
# Filename Parsing Utilities
###################################################################

def parse_analysis_type_from_filename(filename: str) -> str:
    if filename.startswith("corr_stats_pow_roi"):
        return "pow_roi"
    if filename.startswith("corr_stats_conn_roi_summary"):
        return "conn_roi_summary"
    if filename.startswith("corr_stats_edges"):
        return "conn_edges"
    return "other"


def parse_target_from_filename(filename: str) -> str:
    if "_vs_" not in filename:
        return ""
    return filename.split("_vs_", 1)[1].split(".", 1)[0]


def parse_measure_band_from_filename(analysis_type: str, filename: str) -> str:
    prefixes = {
        "conn_edges": "corr_stats_edges_",
        "conn_roi_summary": "corr_stats_conn_roi_summary_"
    }
    prefix = prefixes.get(analysis_type)
    if prefix and filename.startswith(prefix):
        return filename[len(prefix):].split("_vs_", 1)[0]
    return ""


def format_band_range(band: str, freq_bands: Dict[str, List[float]]) -> str:
    if not band or not freq_bands:
        return ""
    
    band_rng = freq_bands.get(band)
    if not band_rng or len(band_rng) < 2:
        return ""
    
    band_range_tuple = tuple(band_rng)
    return f"{band_range_tuple[0]:g}–{band_range_tuple[1]:g} Hz"


def build_partial_covars_string(covariates_df: Optional[pd.DataFrame]) -> str:
    if covariates_df is None or covariates_df.empty:
        return ""
    return ",".join(covariates_df.columns.tolist())


def format_band_label(band: str, config) -> str:
    if not band:
        return ""
    
    freq_bands = config.get("time_frequency_analysis.bands", {})
    band_range = freq_bands.get(band)
    if band_range:
        band_range = tuple(band_range)
        return f"{band} ({band_range[0]:g}\u2013{band_range[1]:g} Hz)"
    return band


###################################################################
# Formatting Utilities
###################################################################

def get_correlation_type_labels(correlation_type: str) -> Tuple[str, str]:
    labels = {
        "rating": ("Behavior", "behavior"),
        "temperature": ("Temperature", "temperature")
    }
    return labels.get(correlation_type, ("Temperature", "temperature"))


def format_temperature_label(val: Union[float, str]) -> str:
    try:
        return f"{float(val):.1f}".replace(".", "p")
    except (ValueError, TypeError):
        return sanitize_label(str(val))


def extract_subject_id_from_path(path: Path) -> Optional[str]:
    import re
    path_str = str(path)
    match = re.search(r'sub-(\d+)', path_str)
    return match.group(1) if match else None


def write_group_trial_counts(
    subjects: List[str],
    output_dir: Path,
    counts_file_name: str,
    pain_counts: Optional[List[Tuple[int, int]]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if pain_counts is None:
        pain_counts = [(0, 0)] * len(subjects)
    
    if len(subjects) != len(pain_counts):
        raise ValueError(f"Length mismatch: {len(subjects)} subjects but {len(pain_counts)} count tuples")
    
    rows = []
    for subject, (n_pain, n_nonpain) in zip(subjects, pain_counts):
        rows.append({
            "subject": subject,
            "n_pain": n_pain,
            "n_nonpain": n_nonpain,
            "n_total": n_pain + n_nonpain
        })
    
    if not rows:
        return
    
    counts_df = pd.DataFrame(rows)
    totals = counts_df[["n_pain", "n_nonpain", "n_total"]].sum()
    total_row = {
        "subject": "TOTAL",
        **{key: int(value) for key, value in totals.to_dict().items()}
    }
    counts_df = pd.concat([counts_df, pd.DataFrame([total_row])], ignore_index=True)
    
    ensure_dir(output_dir)
    output_path = output_dir / counts_file_name
    counts_df.to_csv(output_path, sep="\t", index=False)
    if logger:
        logger.info(f"Saved counts: {output_path}")


def format_channel_list_for_display(
    channels: List[str], 
    max_channels: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    if max_channels is None:
        if config is None and load_settings is not None:
            config = load_settings()
        max_channels = int(config.get("plotting.behavioral.max_channels_to_display", 10)) if config else 10
    
    displayed = channels[:max_channels]
    return "Channels: " + ", ".join(displayed)


def format_roi_description(roi_channels: Optional[List[str]]) -> str:
    if not roi_channels:
        return "Overall"
    if len(roi_channels) == 1:
        return f"Channel: {roi_channels[0]}"
    return f"ROI: {len(roi_channels)} channels"


def get_residual_labels(method_code: str, target_type: str) -> Tuple[str, str]:
    ranked_suffix = " (ranked)" if method_code == "spearman" else ""
    x_label = f"Partial residuals{ranked_suffix} of log10(power/baseline)"
    
    target_labels = {
        "rating": f"Partial residuals{ranked_suffix} of rating",
        "temperature": f"Partial residuals{ranked_suffix} of temperature (°C)"
    }
    y_label = target_labels.get(target_type, f"Partial residuals{ranked_suffix} of {target_type}")
    
    return x_label, y_label


def get_target_labels(target_type: str) -> Tuple[str, str]:
    target_labels = {
        "rating": "Rating",
        "temperature": "Temperature (°C)"
    }
    y_label = target_labels.get(target_type, target_type)
    return "log10(power/baseline [-5–0 s])", y_label


def get_temporal_xlabel(time_label: str) -> str:
    return f"log10(power/baseline) [{time_label} window]"


def format_time_suffix(time_label: Optional[str]) -> str:
    if time_label:
        return f" ({time_label})"
    return " (plateau)"


# Statistical Utilities (imported for backward compatibility)
###################################################################

# fdr_bh and fdr_bh_reject are re-exported from analysis.stats


__all__ = [
    "_find_clean_epochs_path",
    "_load_events_df",
    "_pick_target_column",
    "EEGDatasetResult",
    "PlotConfig",
    "ensure_derivatives_dataset_description",
    "ensure_dir",
    "ensure_aligned_lengths",
    "bids_sub_eeg_path",
    "bids_events_path",
    "deriv_sub_eeg_path",
    "deriv_features_path",
    "deriv_stats_path",
    "deriv_plots_path",
    "deriv_group_eeg_path",
    "deriv_group_stats_path",
    "deriv_group_plots_path",
    "find_connectivity_features_path",
    "save_fig",
    "fdr_bh",
    "fdr_bh_reject",
    "get_pipeline_logger",
    "get_subject_logger",
    "get_group_logger",
    "setup_logger",
    "reconstruct_kept_indices",
    "WINDOW_PAIN",
    "_get_plot_constants",
    "build_footer",
    "unwrap_figure",
    "sanitize_label",
    "get_viz_params",
    "plot_topomap_on_ax",
    "robust_sym_vlim",
    "get_behavior_footer",
    "get_band_color",
    "logratio_to_pct",
    "pct_to_logratio",
    "setup_matplotlib",
    "find_column_in_events",
    "find_pain_column_in_events",
    "find_temperature_column_in_events",
    "find_column_in_metadata",
    "find_pain_column_in_metadata",
    "find_temperature_column_in_metadata",
    "get_column_from_config",
    "get_pain_column_from_config",
    "get_temperature_column_from_config",
    "extract_plotting_constants",
    "extract_eeg_picks",
    "format_baseline_string",
    "log_if_present",
    "validate_picks",
    "get_default_logger",
    "get_default_config",
    "parse_analysis_type_from_filename",
    "parse_target_from_filename",
    "parse_measure_band_from_filename",
    "build_partial_covars_string",
    "format_band_label",
    # Formatting utilities
    "format_band_range",
    "get_correlation_type_labels",
    "format_channel_list_for_display",
    "extract_subject_id_from_path",
    "write_group_trial_counts",
    "format_temperature_label",
    "format_roi_description",
    "get_residual_labels",
    "get_target_labels",
    "get_temporal_xlabel",
    "format_time_suffix",
    "validate_epochs_for_plotting",
    "require_epochs_tfr",
    "detect_data_format",
    "format_baseline_window_string",
    # File validation and building utilities
    "validate_predictor_file",
    "build_file_updates_dict",
    "build_predictor_column_mapping",
    "build_predictor_name",
    "build_connectivity_heatmap_records",
    "build_meta_for_row",
]


###################################################################
# File Validation and Building Utilities
###################################################################

def validate_predictor_file(df: pd.DataFrame, predictor_type: str, target: str, logger) -> bool:
    if predictor_type == "Channel":
        if "channel" not in df.columns or "band" not in df.columns:
            logger.debug(
                f"Skipping combined file for target '{target}' - missing required columns "
                f"(expected 'channel' and 'band')"
            )
            return False
    return True

def build_file_updates_dict(
    file_references: List[Tuple[Path, int]],
    q_array: np.ndarray,
    rejections_array: np.ndarray,
    p_array: np.ndarray,
) -> Dict[Path, List[Tuple[int, float, bool, float]]]:
    from ..analysis.stats import _safe_float
    
    file_updates: Dict[Path, List[Tuple[int, float, bool, float]]] = {}
    
    for index, (file_path, row_index) in enumerate(file_references):
        update_item = (
            row_index,
            _safe_float(q_array[index]),
            bool(rejections_array[index]),
            _safe_float(p_array[index]),
        )
        file_updates.setdefault(file_path, []).append(update_item)
    
    return file_updates

def build_predictor_column_mapping(predictor_type: str) -> Dict[str, str]:
    base_cols = {
        "predictor": "predictor",
        "band": "band",
        "r": "r",
        "p": "p",
        "n": "n",
        "predictor_type": "type",
        "target": "target",
    }
    region_col = "roi" if predictor_type == "ROI" else "channel"
    base_cols[region_col] = "region"
    return base_cols

def build_predictor_name(df: pd.DataFrame, predictor_type: str) -> pd.Series:
    region_col = "roi" if predictor_type == "ROI" else "channel"
    return df[region_col] + " (" + df["band"] + ")"

def build_connectivity_heatmap_records(
    n_nodes: int,
    node_names: List[str],
    correlation_matrix: np.ndarray,
    p_value_matrix: np.ndarray,
    rejection_map: Dict[Tuple[int, int], bool],
    critical_value: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            pair_key = (i, j)
            records.append({
                "node_i": node_names[i],
                "node_j": node_names[j],
                "r": correlation_matrix[i, j],
                "p": p_value_matrix[i, j],
                "fdr_reject": rejection_map.get(pair_key, False),
                "fdr_crit_p": critical_value,
            })
    return records

def _get_df_value(df: pd.DataFrame, col: str, row_idx: int, default: str = "") -> str:
    return df.get(col, pd.Series([default] * len(df))).iloc[row_idx]


def build_meta_for_row(
    df: pd.DataFrame,
    row_idx: int,
    filename: str,
    analysis_type: str,
    target: str,
    measure_band: str,
    p_source: str,
) -> Dict[str, Any]:
    meta = {
        "source_file": filename,
        "analysis_type": analysis_type,
        "target": target,
        "measure_band": measure_band,
        "row_index": int(row_idx),
    }
    if p_source:
        meta["p_used_source"] = p_source

    try:
        if analysis_type == "pow_roi":
            roi = _get_df_value(df, "roi", row_idx)
            band = _get_df_value(df, "band", row_idx)
            meta.update({"roi": roi, "band": band})
            meta["test_label"] = f"pow_{band}_ROI {roi} vs {target}"
        elif analysis_type == "conn_roi_summary":
            roi_i = _get_df_value(df, "roi_i", row_idx)
            roi_j = _get_df_value(df, "roi_j", row_idx)
            meta.update({"roi_i": roi_i, "roi_j": roi_j})
            meta["test_label"] = f"conn_{measure_band}_ROI {roi_i}-{roi_j} vs {target}"
        elif analysis_type == "pow_channel":
            channel = _get_df_value(df, "channel", row_idx)
            band = _get_df_value(df, "band", row_idx)
            meta.update({"channel": channel, "band": band})
            meta["test_label"] = f"pow_{band}_Channel {channel} vs {target}"
        elif analysis_type == "conn_edges":
            node_i = _get_df_value(df, "node_i", row_idx)
            node_j = _get_df_value(df, "node_j", row_idx)
            meta.update({"node_i": node_i, "node_j": node_j})
            meta["test_label"] = f"conn_{measure_band}_Edge {node_i}-{node_j} vs {target}"
    except (KeyError, IndexError):
        pass
    
    return meta
