from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Union, Iterable, Dict, Any
from dataclasses import dataclass
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
from .stats_utils import fdr_bh, fdr_bh_reject

try:
    from .config_loader import load_settings, ConfigDict
except ImportError:
    load_settings = None
    ConfigDict = dict

EEGConfig = ConfigDict



def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_first(pattern: str) -> Optional[Path]:
    candidates = sorted(glob.glob(pattern))
    return Path(candidates[0]) if candidates else None


def get_subject_logger(
    module_name: str,
    subject: str,
    log_file_name: str = None,
    config = None
) -> logging.Logger:
    """Get a logger for subject-level processing."""
    logger_name = f"{module_name}.sub-{subject}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    return logger


def get_group_logger(
    module_name: str,
    log_file_name: str = None,
    config = None
) -> logging.Logger:
    """Get a logger for group-level processing."""
    logger_name = f"{module_name}.group"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    return logger


def get_pipeline_logger(
    module_name: str = None,
    config = None
) -> logging.Logger:
    """Get a logger for pipeline-level processing."""
    logger_name = module_name or "eeg_pipeline"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    return logger


def setup_logger(config = None, subject: str = None) -> logging.Logger:
    """Setup a logger (legacy compatibility function)."""
    if subject:
        return get_subject_logger("pipeline", subject, config=config)
    return get_pipeline_logger(config=config)


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

def bids_sub_eeg_path(bids_root: Path, subject: str) -> Path:
    subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    return Path(bids_root) / subject_label / "eeg"


def bids_events_path(bids_root: Path, subject: str, task: str) -> Path:
    subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    return bids_sub_eeg_path(bids_root, subject) / f"{subject_label}_task-{task}_events.tsv"


def deriv_sub_eeg_path(deriv_root: Path, subject: str) -> Path:
    subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
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
    sub = f"sub-{subject}" if not subject.startswith("sub-") else subject
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


def _find_clean_epochs_path(
    subject: str,
    task: str,
    deriv_root: Optional[Path] = None,
    constants=None,
    config: Optional[EEGConfig] = None,
) -> Path | None:
    if deriv_root is None:
        if config is not None:
            root = config.deriv_root
        elif constants is not None:
            root = Path(constants["DERIV_ROOT"])
        else:
            raise ValueError("Either deriv_root, constants, or config must be provided to _find_clean_epochs_path")
    else:
        root = Path(deriv_root)

    subject_clean = subject.replace("sub-", "") if subject.startswith("sub-") else subject

    bp = BIDSPath(
        subject=subject_clean,
        task=task,
        datatype="eeg",
        processing="clean",
        suffix="epo",
        extension=".fif",
        root=root,
        check=False,
    )
    p1 = bp.fpath
    if p1 and p1.exists():
        return p1

    p2 = root / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_proc-clean_epo.fif"
    if p2.exists():
        return p2

    subj_eeg_dir = root / f"sub-{subject_clean}" / "eeg"
    if subj_eeg_dir.exists():
        cands = sorted(subj_eeg_dir.glob(f"sub-{subject_clean}_task-{task}*epo.fif"))
        for c in cands:
            if any(tok in c.name for tok in ("proc-clean", "proc-cleaned", "clean")):
                return c
        if cands:
            return cands[0]

    subj_dir = root / f"sub-{subject_clean}"
    if subj_dir.exists():
        for c in sorted(subj_dir.rglob(f"sub-{subject_clean}_task-{task}*epo.fif")):
            return c
    return None


def _load_events_df(
    subject: str,
    task: str,
    bids_root: Optional[Path] = None,
    constants=None,
    config: Optional[EEGConfig] = None,
) -> pd.DataFrame | None:
    if bids_root is None:
        if config is not None:
            root = config.bids_root
        elif constants is not None:
            root = Path(constants["BIDS_ROOT"])
        else:
            raise ValueError("Either bids_root, constants, or config must be provided to _load_events_df")
    else:
        root = Path(bids_root)
    
    subject_clean = subject.replace("sub-", "") if subject.startswith("sub-") else subject
    
    ebp = BIDSPath(
        subject=subject_clean,
        task=task,
        datatype="eeg",
        suffix="events",
        extension=".tsv",
        root=root,
        check=False,
    )
    p = ebp.fpath
    if p is None:
        p = root / f"sub-{subject_clean}" / "eeg" / f"sub-{subject_clean}_task-{task}_events.tsv"
    
    if not p.exists():
        return None
    
    return pd.read_csv(p, sep="\t")




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


def WINDOW_PAIN(constants=None, config: Optional[EEGConfig] = None) -> Tuple[float, float]:
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to WINDOW_PAIN")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError("PLATEAU_WINDOW not found in constants. Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)")
    
    return constants["PLATEAU_WINDOW"]


def _pick_target_column(df: pd.DataFrame, constants) -> Optional[str]:
    if constants is None:
        raise ValueError("constants is required for _pick_target_column")
    target_columns = tuple(constants.get("TARGET_COLUMNS", ()))
    for c in target_columns:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if ("vas" in cl or "rating" in cl) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def ensure_derivatives_dataset_description(deriv_root: Optional[Path] = None, constants=None, config=None) -> None:
    if deriv_root is None:
        if config is not None:
            root = config.deriv_root
        elif constants is not None:
            root = Path(constants["DERIV_ROOT"])
        else:
            raise ValueError("Either deriv_root, config, or constants must be provided to ensure_derivatives_dataset_description")
    else:
        root = Path(deriv_root)
    
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


def WINDOW_PAIN(constants=None, config: Optional[EEGConfig] = None) -> Tuple[float, float]:
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to WINDOW_PAIN")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError("PLATEAU_WINDOW not found in constants. Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)")
    
    return constants["PLATEAU_WINDOW"]


def _pick_target_column(df: pd.DataFrame, constants) -> Optional[str]:
    if constants is None:
        raise ValueError("constants is required for _pick_target_column")
    target_columns = tuple(constants.get("TARGET_COLUMNS", ()))
    for c in target_columns:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if ("vas" in cl or "rating" in cl) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def ensure_derivatives_dataset_description(deriv_root: Optional[Path] = None, constants=None, config=None) -> None:
    if deriv_root is None:
        if config is not None:
            root = config.deriv_root
        elif constants is not None:
            root = Path(constants["DERIV_ROOT"])
        else:
            raise ValueError("Either deriv_root, config, or constants must be provided to ensure_derivatives_dataset_description")
    else:
        root = Path(deriv_root)
    
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


def WINDOW_PAIN(constants=None, config: Optional[EEGConfig] = None) -> Tuple[float, float]:
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to WINDOW_PAIN")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError("PLATEAU_WINDOW not found in constants. Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)")
    
    return constants["PLATEAU_WINDOW"]


def _pick_target_column(df: pd.DataFrame, constants) -> Optional[str]:
    if constants is None:
        raise ValueError("constants is required for _pick_target_column")
    target_columns = tuple(constants.get("TARGET_COLUMNS", ()))
    for c in target_columns:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if ("vas" in cl or "rating" in cl) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def ensure_derivatives_dataset_description(deriv_root: Optional[Path] = None, constants=None, config=None) -> None:
    if deriv_root is None:
        if config is not None:
            root = config.deriv_root
        elif constants is not None:
            root = Path(constants["DERIV_ROOT"])
        else:
            raise ValueError("Either deriv_root, config, or constants must be provided to ensure_derivatives_dataset_description")
    else:
        root = Path(deriv_root)
    
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


def WINDOW_PAIN(constants=None, config: Optional[EEGConfig] = None) -> Tuple[float, float]:
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to WINDOW_PAIN")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError("PLATEAU_WINDOW not found in constants. Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)")
    
    return constants["PLATEAU_WINDOW"]


def _pick_target_column(df: pd.DataFrame, constants) -> Optional[str]:
    if constants is None:
        raise ValueError("constants is required for _pick_target_column")
    target_columns = tuple(constants.get("TARGET_COLUMNS", ()))
    for c in target_columns:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if ("vas" in cl or "rating" in cl) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def ensure_derivatives_dataset_description(deriv_root: Optional[Path] = None, constants=None, config=None) -> None:
    if deriv_root is None:
        if config is not None:
            root = config.deriv_root
        elif constants is not None:
            root = Path(constants["DERIV_ROOT"])
        else:
            raise ValueError("Either deriv_root, config, or constants must be provided to ensure_derivatives_dataset_description")
    else:
        root = Path(deriv_root)
    
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


def WINDOW_PAIN(constants=None, config: Optional[EEGConfig] = None) -> Tuple[float, float]:
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to WINDOW_PAIN")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError("PLATEAU_WINDOW not found in constants. Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)")
    
    return constants["PLATEAU_WINDOW"]


def _pick_target_column(df: pd.DataFrame, constants) -> Optional[str]:
    if constants is None:
        raise ValueError("constants is required for _pick_target_column")
    target_columns = tuple(constants.get("TARGET_COLUMNS", ()))
    for c in target_columns:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if ("vas" in cl or "rating" in cl) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def ensure_derivatives_dataset_description(deriv_root: Optional[Path] = None, constants=None, config=None) -> None:
    if deriv_root is None:
        if config is not None:
            root = config.deriv_root
        elif constants is not None:
            root = Path(constants["DERIV_ROOT"])
        else:
            raise ValueError("Either deriv_root, config, or constants must be provided to ensure_derivatives_dataset_description")
    else:
        root = Path(deriv_root)
    
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

def ensure_aligned_lengths(
    *arrays_or_series,
    context: str = "",
    strict: Optional[bool] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    if strict is None:
        strict = True
    
    non_null = [obj for obj in arrays_or_series if obj is not None]
    if len(non_null) < 2:
        return
    
    lengths = {len(obj) for obj in non_null}
    if len(lengths) > 1:
        msg = f"{context}: Length mismatch detected: {[len(obj) for obj in non_null]}"
        if strict:
            raise ValueError(msg)
        if logger:
            logger.warning(msg)
    
    ref_index = None
    for obj in non_null:
        idx = getattr(obj, "index", None)
        if idx is None:
            continue
        if ref_index is None:
            ref_index = idx
        elif len(idx) != len(ref_index) or not idx.equals(ref_index):
            msg = f"{context}: Index misalignment detected; align before analysis"
            if strict:
                raise ValueError(msg)
            if logger:
                logger.warning(msg)



###################################################################
# Figure Saving
###################################################################

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
    if path.suffix:
        base_path = path.with_suffix("")
    else:
        base_path = path
    
    if constants is not None:
        _get_plot_constants(constants=constants)
    
    formats = formats or _PLOT_DEFAULT_FORMATS
    dpi = dpi if dpi is not None else _PLOT_DEFAULT_DPI
    bbox_inches = bbox_inches or _PLOT_DEFAULT_BBOX
    pad_inches = pad_inches if pad_inches is not None else _PLOT_DEFAULT_PAD

    if footer is None and footer_template_name is not None and constants is not None:
        try:
            from .config_loader import load_settings
            cfg = load_settings()
            footer = build_footer(footer_template_name, cfg, **(footer_kwargs or {}))
        except Exception:
            footer = None
    should_add_footer = footer is not None and os.getenv("FIG_FOOTER_OFF", "0").lower() not in {"1", "true", "yes"}

    if should_add_footer:
        fig.text(0.01, 0.01, footer, fontsize=8, alpha=0.8)
        rect = tight_layout_rect or [0, 0.03, 1, 1]
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
            try:
                fig.tight_layout(rect=rect)
            except RuntimeError:
                fig.subplots_adjust(bottom=0.06)
    elif tight_layout_rect is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
            try:
                fig.tight_layout(rect=tight_layout_rect)
            except RuntimeError:
                fig.subplots_adjust(bottom=0.06)
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
            try:
                fig.tight_layout()
            except RuntimeError:
                fig.subplots_adjust(bottom=0.06)

    saved_paths = []
    try:
        for ext in formats:
            output_path = base_path.with_suffix(f".{ext}")
            fig.savefig(
                output_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches
            )
            saved_paths.append(output_path)
    finally:
        plt.close(fig)
    if logger is not None:
        paths_str = ", ".join(str(p) for p in saved_paths)
        logger.info(f"Saved: {paths_str}")


###################################################################
# Plot Config and Footer Helpers
###################################################################

_PLOT_DEFAULT_DPI = None
_PLOT_DEFAULT_FORMATS = None
_PLOT_DEFAULT_BBOX = None
_PLOT_DEFAULT_PAD = None


def _get_plot_constants(constants=None):
    global _PLOT_DEFAULT_DPI, _PLOT_DEFAULT_FORMATS, _PLOT_DEFAULT_BBOX, _PLOT_DEFAULT_PAD
    if _PLOT_DEFAULT_DPI is None:
        if constants is None:
            raise ValueError("constants is required for _get_plot_constants")
        _PLOT_DEFAULT_DPI = constants.get("FIG_DPI", 300)
        _PLOT_DEFAULT_FORMATS = tuple(constants.get("SAVE_FORMATS", ["png"]))
        _PLOT_DEFAULT_BBOX = constants.get("output.bbox_inches", "tight")
        _PLOT_DEFAULT_PAD = float(constants.get("output.pad_inches", 0.02))


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
    bwin = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    return f"Baseline: [{float(bwin[0]):.2f}, {float(bwin[1]):.2f}] s | Significance: BH-FDR α={fdr_alpha}"


def get_band_color(band: str, config=None) -> str:
    if config is not None:
        colors = config.get("visualization.band_colors", {})
        if band in colors:
            return str(colors[band])
    fallback = {"delta": "#4169e1", "theta": "purple", "alpha": "green", "beta": "orange", "gamma": "red"}
    return fallback.get(band, "#1f77b4")


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
            from .config_loader import load_settings
            config = load_settings()
        except Exception:
            config = None
    return {
        "topo_contours": config.get("time_frequency_analysis.topo_contours", 6) if config else 6,
        "topo_cmap": config.get("time_frequency_analysis.topo_cmap", "RdBu_r") if config else "RdBu_r",
        "colorbar_fraction": config.get("time_frequency_analysis.colorbar_fraction", 0.03) if config else 0.03,
        "colorbar_pad": config.get("time_frequency_analysis.colorbar_pad", 0.02) if config else 0.02,
        "diff_annotation_enabled": config.get("time_frequency_analysis.diff_annotation_enabled", True) if config else True,
        "annotate_descriptive_topo": config.get("time_frequency_analysis.annotate_descriptive_topo", False) if config else False,
        "sig_mask_params": config.get("time_frequency_analysis.sig_mask_params", {
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "k",
            "linewidth": 1.0,
            "markersize": 5,
        }) if config else {
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "k",
            "linewidth": 1.0,
            "markersize": 5,
        },
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


def format_p_value(p):
    try:
        p_arr = np.asarray(p, dtype=float)
    except Exception:
        return "p=nan"
    if p_arr.size == 0:
        return "p=nan"
    p_val = float(np.nanmin(p_arr)) if p_arr.size > 1 else float(p_arr)
    if not np.isfinite(p_val):
        return "p=nan"
    return "p<.001" if p_val < 1e-3 else f"p={p_val:.3f}"


def format_cluster_ann(p, k=None, mass=None, config=None):
    parts = [format_p_value(p)]
    if config is None:
        try:
            from .config_loader import load_settings
            config = load_settings()
        except Exception:
            config = None
    report_size = config.get("statistics.cluster_report_size", False) if config else False
    report_mass = config.get("statistics.cluster_report_mass", False) if config else False
    if report_size and isinstance(k, (int, np.integer)) and k and k > 0:
        parts.append(f"k={int(k)}")
    if report_mass and mass is not None and np.isfinite(mass):
        parts.append(f"mass={float(mass):.1f}")
    return "; ".join(parts)


def format_fdr_ann(q_min: Optional[float], k_rej: Optional[int], alpha: float = 0.05) -> str:
    parts = []
    if q_min is not None and np.isfinite(q_min):
        parts.append(f"FDR q={q_min:.3f}" if q_min >= 1e-3 else "FDR q<.001")
    if k_rej is not None and isinstance(k_rej, (int, np.integer)) and int(k_rej) > 0:
        parts.append(f"k={int(k_rej)}")
    return "; ".join(parts) if parts else ""


def robust_sym_vlim(
    arrs: "np.ndarray | list[np.ndarray]",
    q_low: float = 0.02,
    q_high: float = 0.98,
    cap: float = 0.25,
    min_v: float = 1e-6,
) -> float:
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
        from .config_loader import get_nested_value
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
# Statistical Utilities (imported for backward compatibility)
###################################################################

# fdr_bh and fdr_bh_reject are re-exported from stats_utils


__all__ = [
    "_find_clean_epochs_path",
    "_load_events_df",
    "_pick_target_column",
    "EEGDatasetResult",
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
    "format_p_value",
    "format_cluster_ann",
    "format_fdr_ann",
    "robust_sym_vlim",
    "get_behavior_footer",
    "get_band_color",
    "logratio_to_pct",
    "pct_to_logratio",
    "setup_matplotlib",
]
