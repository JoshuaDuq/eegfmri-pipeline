from __future__ import annotations

import functools
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import mne
import numpy as np
import pandas as pd

from .config_loader import load_settings


###################################################################
# Constants Loading
###################################################################

def _load_tfr_constants(config=None):
    if config is None:
        config = load_settings()
    
    if config is None:
        raise ValueError("Config is required. Cannot load TFR constants without config.")
    
    constants = config.get("time_frequency_analysis.constants")
    if constants is None:
        raise ValueError("time_frequency_analysis.constants not found in config.")
    
    return {
        "baseline_sentinel": constants["baseline_sentinel"],
        "min_log_value": constants["min_log_value"],
        "min_divisor": constants["min_divisor"],
        "min_samples_for_stats": constants["min_samples_for_stats"],
        "default_min_cycles": constants["default_min_cycles"],
        "default_min_cycles_check": constants["default_min_cycles_check"],
        "low_freq_threshold": constants["low_freq_threshold"],
        "low_cycles_warning_threshold": constants["low_cycles_warning_threshold"],
        "freq_threshold_for_time_res": constants["freq_threshold_for_time_res"],
        "default_time_res_threshold_low_freq": constants["default_time_res_threshold_low_freq"],
        "default_time_res_threshold_high_freq": constants["default_time_res_threshold_high_freq"],
        "default_cycles_factor": constants["default_cycles_factor"],
        "mad_scaling_factor": constants["mad_scaling_factor"],
    }


@functools.lru_cache(maxsize=None)
def _get_tfr_constants_cached():
    return _load_tfr_constants(None)


def _get_tfr_constants(config=None):
    if config is None:
        return _get_tfr_constants_cached()
    return _load_tfr_constants(config)


###################################################################
# Channel Extraction & Finding
###################################################################

def extract_eeg_channels(epochs: mne.Epochs) -> List[str]:
    return [
        ch for ch in epochs.info["ch_names"]
        if epochs.get_channel_types(picks=[ch])[0] == "eeg"
    ]

def find_common_eeg_channels(epochs_dict: Dict[str, Any]) -> List[str]:
    if not epochs_dict:
        return []
    
    eeg_channel_sets = [
        set(extract_eeg_channels(epochs))
        for epochs in epochs_dict.values()
    ]
    
    if not eeg_channel_sets:
        return []
    
    if len(eeg_channel_sets) == 1:
        return sorted(list(eeg_channel_sets[0]))
    
    return sorted(list(set.intersection(*eeg_channel_sets)))

def find_common_channels_across_subjects(
    subject_epochs_map: Dict[str, mne.Epochs],
    subjects: List[str]
) -> List[str]:
    if not subjects:
        return []
    
    channel_sets = [
        set(extract_eeg_channels(subject_epochs_map[subj]))
        for subj in subjects
    ]
    
    if len(channel_sets) == 1:
        return sorted(list(channel_sets[0]))
    
    return sorted(list(set.intersection(*channel_sets)))

def find_common_channels_train_test(
    train_subjects: List[str],
    test_subject: str,
    subj_to_epochs: Dict[str, mne.Epochs]
) -> List[str]:
    train_channel_sets = [
        set(extract_eeg_channels(subj_to_epochs[s]))
        for s in train_subjects
    ]
    
    if len(train_channel_sets) == 1:
        common_train = sorted(list(train_channel_sets[0]))
    else:
        common_train = sorted(list(set.intersection(*train_channel_sets)))
    
    test_channels = set(extract_eeg_channels(subj_to_epochs[test_subject]))
    return sorted([ch for ch in common_train if ch in test_channels])

def prepare_bands_data_for_topomap(bands_to_df: Dict[str, pd.DataFrame], 
                                    fdr_alpha: float) -> List[Dict]:
    if not bands_to_df:
        return []
    
    bands_data = []
    for band, df_band in bands_to_df.items():
        if df_band.empty or "channel" not in df_band.columns:
            continue
        
        channels = df_band["channel"].astype(str).tolist()
        correlations = df_band["r_group"].to_numpy()
        p_values = df_band["p_group"].to_numpy()
        sig_mask = np.isfinite(p_values) & (p_values < fdr_alpha)
        bands_data.append({
            "band": band,
            "channels": channels,
            "correlations": correlations,
            "p_values": p_values,
            "significant_mask": sig_mask,
        })
    return bands_data


###################################################################
# ROI Channel Operations
###################################################################

def canonicalize_ch_name(ch: str) -> str:
    cleaned = ch.strip()
    try:
        cleaned = re.sub(r"^(EEG[ \-_]*)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"[-/]", cleaned)[0]
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"(Ref|LE|RE|M1|M2|A1|A2|AVG|AVE)$", "", cleaned, flags=re.IGNORECASE)
        return cleaned
    except (re.error, TypeError, AttributeError) as e:
        logging.getLogger(__name__).warning(f"Error canonicalizing channel name '{ch}': {e}; returning original")
        return ch


def get_rois(config) -> Dict[str, List[str]]:
    if config is None:
        raise ValueError("config is required for get_rois")
    rois = config.get("time_frequency_analysis.rois")
    if rois is None:
        raise ValueError("time_frequency_analysis.rois not found in config")
    return dict(rois)


def find_roi_channels(info: mne.Info, patterns: List[str]) -> List[str]:
    channel_names = info["ch_names"]
    canon_map = {ch: canonicalize_ch_name(ch) for ch in channel_names}
    matched_channels = set()
    
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for ch_name in channel_names:
            canon_name = canon_map.get(ch_name, ch_name)
            if regex.match(ch_name) or regex.match(canon_name):
                matched_channels.add(ch_name)
    
    ordered_channels = []
    seen = set()
    for ch_name in channel_names:
        if ch_name in matched_channels and ch_name not in seen:
            ordered_channels.append(ch_name)
            seen.add(ch_name)
    
    return ordered_channels


def build_rois_from_info(info: mne.Info, config=None) -> Dict[str, List[str]]:
    rois = {}
    roi_defs = get_rois(config)
    for roi, pats in roi_defs.items():
        chans = find_roi_channels(info, pats)
        if chans:
            rois[roi] = chans
    return rois


def extract_hemisphere_from_node(node_name: str) -> Optional[str]:
    if not node_name:
        return None
    
    hemisphere_tokens = ("LH", "RH")
    tokens = node_name.split("_")
    for token in tokens:
        if token in hemisphere_tokens:
            return token
    
    hemisphere_pattern = r"(?:^|_)(LH|RH)_([A-Za-z]+)"
    match = re.search(hemisphere_pattern, node_name)
    if match:
        return match.group(1)
    
    return None


def extract_system_from_node(node_name: str) -> Optional[str]:
    if not node_name:
        return None
    
    brain_systems = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}
    tokens = node_name.split("_")
    for token in tokens:
        if token in brain_systems:
            return token
    
    hemisphere_pattern = r"(?:^|_)(LH|RH)_([A-Za-z]+)"
    match = re.search(hemisphere_pattern, node_name)
    if match:
        candidate_system = match.group(2)
        if candidate_system in brain_systems:
            return candidate_system
    
    return None


def build_roi_name(system: str, hemisphere: Optional[str], hemisphere_split: bool) -> str:
    if not hemisphere_split or not hemisphere:
        return system
    return f"{system}_{hemisphere}"


def build_atlas_rois_from_nodes(
    node_list: List[str],
    hemisphere_split: bool = True,
) -> Dict[str, List[str]]:
    if not node_list:
        return {}
    
    roi_nodes: Dict[str, List[str]] = {}
    for node_name in node_list:
        if not node_name:
            continue
        
        system = extract_system_from_node(node_name)
        if system is None:
            continue
        
        hemisphere = extract_hemisphere_from_node(node_name)
        roi_name = build_roi_name(system, hemisphere, hemisphere_split)
        roi_nodes.setdefault(roi_name, []).append(node_name)
    
    return roi_nodes


def get_summary_type(roi_i: str, roi_j: str) -> str:
    return "within" if roi_i == roi_j else "between"


def build_summary_map_from_roi_nodes(
    roi_map: Dict[str, List[str]],
    prefix: str,
    column_names: List[str],
) -> Dict[Tuple[str, str], List[str]]:
    if not roi_map or not column_names:
        return {}
    
    summary_map: Dict[Tuple[str, str], List[str]] = {}
    prefix_with_sep = prefix + "_"
    
    for column_name in column_names:
        if not column_name.startswith(prefix_with_sep):
            continue
        
        node_pair_str = column_name.split(prefix_with_sep, 1)[-1]
        if "__" not in node_pair_str:
            continue
        
        node_a, node_b = node_pair_str.split("__", 1)
        roi_a = _find_roi_for_node(roi_map, node_a)
        roi_b = _find_roi_for_node(roi_map, node_b)
        
        if roi_a is not None and roi_b is not None:
            roi_pair = (roi_a, roi_b)
            summary_map.setdefault(roi_pair, []).append(column_name)
    
    return summary_map


def _find_roi_for_node(roi_map: Dict[str, List[str]], node: str) -> Optional[str]:
    for roi_name, nodes in roi_map.items():
        if node in nodes:
            return roi_name
    return None


def extract_node_pair_from_column(
    column_name: str,
    prefix: str,
) -> Optional[Tuple[str, str]]:
    if not column_name.startswith(prefix + "_"):
        return None
    
    node_pair_str = column_name.split(prefix + "_", 1)[-1]
    if "__" not in node_pair_str:
        return None
    
    parts = node_pair_str.split("__", 1)
    if len(parts) != 2:
        return None
    
    return (parts[0], parts[1])


###################################################################
# TFR Parameter Computation
###################################################################

def compute_adaptive_n_cycles(
    freqs: Union[np.ndarray, list],
    cycles_factor: Optional[float] = None,
    min_cycles: Optional[float] = None,
    max_cycles: Optional[float] = None,
    config: Optional[Any] = None
) -> np.ndarray:
    constants = _get_tfr_constants(config)
    if cycles_factor is None:
        cycles_factor = constants["default_cycles_factor"]
    if min_cycles is None:
        min_cycles = _get_config_float(
            config, "time_frequency_analysis.tfr.min_cycles", constants["default_min_cycles"]
        )
    
    freqs = np.asarray(freqs, dtype=float)
    base_cycles = freqs / cycles_factor
    n_cycles = np.maximum(base_cycles, min_cycles)
    
    if max_cycles is not None:
        n_cycles = np.minimum(n_cycles, max_cycles)
    
    return n_cycles


def _get_config_float(config: Optional[Any], key: str, default: float) -> float:
    if config is None:
        return default
    return float(config.get(key, default))


def log_tfr_resolution(
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None
) -> None:
    logger = _get_logger(logger)
    constants = _get_tfr_constants(config)

    time_res = n_cycles / freqs
    freq_res = freqs / n_cycles

    logger.info("TFR Resolution Summary:")
    logger.info(f"  Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    logger.info(f"  n_cycles range: {n_cycles.min():.1f} - {n_cycles.max():.1f}")
    logger.info(f"  Time resolution: {time_res.min():.3f} - {time_res.max():.3f} s")
    logger.info(f"  Frequency resolution: {freq_res.min():.2f} - {freq_res.max():.2f} Hz")

    low_freq_mask = freqs <= constants["low_freq_threshold"]
    if np.any(low_freq_mask):
        low_cycles = n_cycles[low_freq_mask]
        if np.any(low_cycles < constants["low_cycles_warning_threshold"]):
            logger.warning(f"Low n_cycles detected in theta/alpha: min={low_cycles.min():.1f}")
        else:
            logger.info(f"Good low-frequency resolution: min n_cycles={low_cycles.min():.1f}")


def validate_tfr_parameters(
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None
) -> bool:
    logger = _get_logger(logger)
    issues = []
    
    constants = _get_tfr_constants(config)
    min_cycles_check = _get_config_float(
        config, "time_frequency_analysis.tfr.min_cycles_check", constants["default_min_cycles_check"]
    )
    if np.any(n_cycles < min_cycles_check):
        issues.append(f"n_cycles too low: min={n_cycles.min():.1f}")
    
    nyquist = sfreq / 2
    if np.any(freqs >= nyquist):
        issues.append(f"Frequencies above Nyquist: max_freq={freqs.max():.1f}, Nyquist={nyquist:.1f}")
    
    max_time_res = np.max(n_cycles / freqs)
    min_freq = np.min(freqs)
    time_res_threshold = _get_time_res_threshold(min_freq, config)
    
    if max_time_res > time_res_threshold:
        issues.append(
            f"Excessive time resolution: max={max_time_res:.1f}s "
            f"(threshold={time_res_threshold:.1f}s for min_freq={min_freq:.1f}Hz)"
        )

    if issues:
        for issue in issues:
            logger.warning(f"TFR parameter issue: {issue}")
        return False

    logger.info("TFR parameters validation passed")
    return True


def _get_time_res_threshold(min_freq: float, config: Optional[Any]) -> float:
    constants = _get_tfr_constants(config)
    if min_freq <= constants["freq_threshold_for_time_res"]:
        return _get_config_float(
            config, "time_frequency_analysis.tfr.time_res_threshold_low_freq",
            constants["default_time_res_threshold_low_freq"]
        )
    return _get_config_float(
        config, "time_frequency_analysis.tfr.time_res_threshold_high_freq",
        constants["default_time_res_threshold_high_freq"]
    )


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is None:
        return logging.getLogger(__name__)
    return logger


def resolve_tfr_workers(workers_default: int = -1) -> int:
    raw = os.getenv("EEG_TFR_WORKERS")
    if not raw or raw.strip().lower() in {"auto", ""}:
        return workers_default

    try:
        return max(1, int(raw))
    except ValueError:
        logger = _get_logger(None)
        logger.warning(f"EEG_TFR_WORKERS={raw} invalid; using {workers_default}")
        return workers_default


def get_bands_for_tfr(
    tfr=None,
    max_freq_available: Optional[float] = None,
    band_bounds: Optional[Dict[str, Tuple[float, Optional[float]]]] = None,
    config=None,
) -> Dict[str, Tuple[float, float]]:
    if band_bounds is None:
        config = config or load_settings()
        from .config_loader import get_frequency_bands
        
        config_bands = get_frequency_bands(config)
        if not config_bands:
            config_bands = config.get("tfr_topography_pipeline.bands", {})
        if not config_bands:
            config_bands = config.get("time_frequency_analysis.bands", {})
        
        if not config_bands:
            raise ValueError("No frequency bands found in config. Check time_frequency_analysis.bands or tfr_topography_pipeline.bands")
        
        band_bounds = {
            k: (v[0], v[1] if v[1] is not None else None)
            for k, v in dict(config_bands).items()
        }

    max_freq = max_freq_available
    if max_freq is None:
        if tfr is not None:
            max_freq = float(np.max(tfr.freqs))
        else:
            config = config or load_settings()
            max_freq = float(config.get("time_frequency_analysis.tfr.freq_max"))
    
    standard_bands = ["delta", "theta", "alpha", "beta"]
    bands = {k: v for k, v in band_bounds.items() if k in standard_bands}

    gamma_lower, gamma_upper = band_bounds.get("gamma", (None, None))
    if gamma_lower is None or gamma_upper is None:
        config = config or load_settings()
        from .config_loader import get_frequency_bands
        config_bands = get_frequency_bands(config)
        if "gamma" not in config_bands:
            raise ValueError("Gamma band not found in config frequency bands")
        default_gamma = config_bands["gamma"]
        gamma_lower = gamma_lower if gamma_lower is not None else default_gamma[0]
        gamma_upper = gamma_upper if gamma_upper is not None else default_gamma[1]
    
    bands["gamma"] = (gamma_lower, min(gamma_upper or max_freq, max_freq))
    return bands


###################################################################
# TFR I/O with Unit Standardization
###################################################################

def read_tfr_average_with_logratio(
    tfr_path: Union[str, "os.PathLike[str]"],
    baseline_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    min_baseline_samples: int = 5,
    config: Optional[Any] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    logger = _get_logger(logger)
    
    tfr_obj = _load_tfr_from_path(tfr_path, logger)
    if tfr_obj is None:
        return None
    
    if _has_baseline_sentinel(tfr_obj, config=config):
        logger.info(
            f"TFR at {tfr_path} already has baseline correction marker; "
            f"skipping baseline application to prevent double correction."
        )
        return tfr_obj

    mode_detected, detected_by_sidecar = _detect_baseline_mode(tfr_obj, tfr_path, logger)

    if mode_detected is None:
        return _apply_baseline_if_needed(
            tfr_obj, baseline_window, min_baseline_samples, logger, tfr_path
        )

    if mode_detected == "logratio":
        return tfr_obj
    
    if mode_detected == "ratio":
        return _convert_ratio_to_logratio(tfr_obj, tfr_path, logger, detected_by_sidecar, config=config)
    
    if mode_detected == "percent":
        return _convert_percent_to_logratio(tfr_obj, tfr_path, logger, detected_by_sidecar, config=config)

    return None


def _load_tfr_from_path(
    tfr_path: Union[str, "os.PathLike[str]"],
    logger: logging.Logger
) -> Optional["mne.time_frequency.AverageTFR"]:
    try:
        read = getattr(mne.time_frequency, "read_tfrs", None)
        if read is None:
            logger.warning(f"read_tfrs unavailable for {tfr_path}")
            return None

        tfrs = read(str(tfr_path))
        if tfrs is None:
            logger.warning(f"read_tfrs returned None for {tfr_path}")
            return None
            
        if not isinstance(tfrs, list):
            tfrs = [tfrs]
            
        if len(tfrs) == 0:
            logger.warning(f"No TFRs found in {tfr_path}")
            return None

        tfr_obj = tfrs[0]
        data = getattr(tfr_obj, "data", None)
        if data is not None and getattr(data, "ndim", 0) == 4:
            tfr_obj = tfr_obj.average()
        
        return tfr_obj
    except (OSError, ValueError) as e:
        logger.warning(f"Error reading TFR from {tfr_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading TFR from {tfr_path}: {e}")
        raise


def _has_baseline_sentinel(tfr_obj: Any, config: Optional[Any] = None) -> bool:
    constants = _get_tfr_constants(config)
    comment = getattr(tfr_obj, "comment", None)
    return isinstance(comment, str) and constants["baseline_sentinel"] in comment


def _convert_ratio_to_logratio(
    tfr_obj: Any,
    tfr_path: Union[str, "os.PathLike[str]"],
    logger: logging.Logger,
    verified: bool,
    config: Optional[Any] = None
) -> Optional["mne.time_frequency.AverageTFR"]:
    if not verified:
        logger.error(
            f"{tfr_path}: detected 'ratio' but sidecar verification failed. "
            f"Refusing conversion to prevent invalid results."
        )
        return None
    
    constants = _get_tfr_constants(config)
    tfr_obj.data = np.log10(np.maximum(tfr_obj.data, constants["min_log_value"]))
    if hasattr(tfr_obj, "comment"):
        tfr_obj.comment = (tfr_obj.comment or "") + " | converted ratio->log10ratio"
    return tfr_obj


def _convert_percent_to_logratio(
    tfr_obj: Any,
    tfr_path: Union[str, "os.PathLike[str]"],
    logger: logging.Logger,
    verified: bool,
    config: Optional[Any] = None
) -> Optional["mne.time_frequency.AverageTFR"]:
    if not verified:
        logger.error(
            f"{tfr_path}: detected 'percent' but sidecar verification failed. "
            f"Refusing conversion to prevent invalid results."
        )
        return None
    
    constants = _get_tfr_constants(config)
    ratio = 1.0 + (tfr_obj.data / 100.0)
    tfr_obj.data = np.log10(np.clip(ratio, constants["min_log_value"], np.inf))
    if hasattr(tfr_obj, "comment"):
        tfr_obj.comment = (tfr_obj.comment or "") + " | converted percent->log10ratio"
    return tfr_obj


def _detect_baseline_mode(
    tfr_obj: Any,
    tfr_path: Union[str, "os.PathLike[str]"],
    logger: logging.Logger,
) -> Tuple[Optional[str], bool]:
    sidecar_path = str(tfr_path).rsplit(".", 1)[0] + ".json"
    
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        if not bool(metadata.get("baseline_applied", False)):
            return None, False
        
        mode_detected = str(metadata.get("baseline_mode", "")).strip().lower() or None
        if mode_detected is None:
            return None, False
        
        if mode_detected not in {"logratio", "ratio", "percent"}:
            logger.warning(
                f"Unsupported baseline mode '{mode_detected}' "
                f"from sidecar for {tfr_path}; skipping."
            )
            return None, False
        
        comment = getattr(tfr_obj, "comment", "") or str(tfr_path)
        logger.info(f"{comment}: Sidecar found. Baseline mode={mode_detected}")
        return mode_detected, True
        
    except FileNotFoundError:
        logger.warning(
            f"Missing TFR sidecar for {tfr_path}. "
            f"Cannot determine baseline units from sidecar. "
            f"Will apply fallback baseline correction."
        )
        return None, False
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed reading TFR sidecar for {tfr_path}: {exc}")
        return None, False


def _apply_baseline_if_needed(
    tfr_obj: Any,
    baseline_window: Tuple[float, float],
    min_baseline_samples: int,
    logger: logging.Logger,
    tfr_path: Union[str, "os.PathLike[str]"],
) -> Optional["mne.time_frequency.AverageTFR"]:
    times = np.asarray(tfr_obj.times)
    
    try:
        baseline_start, baseline_end, baseline_indices = validate_baseline_indices(
            times, baseline_window, min_samples=min_baseline_samples, logger=logger
        )
    except ValueError as e:
        logger.warning(f"Baseline validation failed for {tfr_path}: {e}")
        return None
    
    tfr_obj.apply_baseline(baseline=(baseline_start, baseline_end), mode="logratio")
    if hasattr(tfr_obj, "comment"):
        comment = tfr_obj.comment or ""
        tfr_obj.comment = f"{comment} | baseline(logratio) applied"
    
    return tfr_obj


def save_tfr_with_sidecar(
    tfr: Union["mne.time_frequency.EpochsTFR", "mne.time_frequency.AverageTFR"],
    out_path: Union[str, "os.PathLike[str]"],
    baseline_window: Tuple[float, float],
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    logger = _get_logger(logger)
    
    if mode is None:
        config = config or load_settings()
        mode = str(config.get("tfr_processing.baseline_mode"))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tfr.save(str(path), overwrite=True)
    
    sidecar = {
        "baseline_applied": True,
        "baseline_mode": str(mode),
        "units": ("log10ratio" if str(mode).lower() == "logratio" else str(mode)),
        "baseline_window": [float(baseline_window[0]), float(baseline_window[1])],
        "created_by": "tfr_utils.save_tfr_with_sidecar",
        "comment": getattr(tfr, "comment", None),
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    
    logger.info(f"Saved TFR and sidecar: {path} (+ .json)")


###################################################################
# TFR Baseline Operations
###################################################################

def validate_baseline_window(
    times: np.ndarray,
    baseline: Tuple[float, float],
    min_samples: int = 5,
    config=None,
) -> Tuple[float, float, np.ndarray]:
    b_start, b_end = baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)
    
    if b_end > 0:
        raise ValueError(f"Baseline window must end at or before 0 s, got [{b_start}, {b_end}]")
    
    if b_start >= b_end:
        raise ValueError(
            f"Baseline window start ({b_start}) must be < end ({b_end}). "
            f"Invalid baseline window configuration."
        )
    
    mask = (times >= b_start) & (times < b_end)
    n_samples = int(mask.sum())
    
    if n_samples < min_samples:
        msg = (
            f"Baseline window [{b_start:.3f}, {b_end:.3f}] s has {n_samples} samples; "
            f"at least {min_samples} required"
        )
        logger = _get_logger(None)
        logger.error(msg)
        raise ValueError(msg)
    
    return b_start, b_end, mask


def validate_baseline_indices(
    times: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]],
    min_samples: int = 5,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, np.ndarray]:
    b_start, b_end = baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)

    if b_end > 0:
        raise ValueError("Baseline window must end at or before 0 s (stimulus onset)")
    
    if b_start >= b_end:
        raise ValueError(
            f"Baseline window start ({b_start}) must be < end ({b_end}). "
            f"Invalid baseline window configuration."
        )

    baseline_mask = (times >= b_start) & (times < b_end)
    idx = np.where(baseline_mask)[0]

    if len(idx) < min_samples:
        raise ValueError(
            f"Baseline window contains only {len(idx)} samples "
            f"(minimum {min_samples} required)"
        )

    if logger is not None:
        timespan = (float(times[idx[0]]), float(times[idx[-1]]))
        logger.info(
            f"Baseline indices: window [{b_start:.3f}, {b_end:.3f}] s maps to "
            f"indices [{idx[0]}, {idx[-1]}] with actual timespan [{timespan[0]:.3f}, {timespan[1]:.3f}] s "
            f"(n_samples={len(idx)})"
        )

    return b_start, b_end, idx


def validate_baseline_window_for_times(
    times: np.ndarray,
    baseline_window: Optional[Tuple[float, float]] = None,
    min_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> Tuple[float, float, np.ndarray]:
    config = config or load_settings()
    
    if baseline_window is None:
        baseline_window = tuple(config.get("time_frequency_analysis.baseline_window"))
    
    if min_samples is None:
        min_samples = int(config.get("tfr_topography_pipeline.min_baseline_samples"))
    
    return validate_baseline_indices(times, baseline_window, min_samples=min_samples, logger=logger)


def validate_plateau_window_for_times(
    times: np.ndarray,
    plateau_window: Optional[Tuple[float, float]] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> Tuple[float, float, np.ndarray]:
    config = config or load_settings()
    
    if plateau_window is None:
        plateau_window = tuple(config.get("time_frequency_analysis.plateau_window"))
    
    plateau_start, plateau_end = plateau_window
    
    if plateau_start >= plateau_end:
        raise ValueError(
            f"Plateau window invalid: start ({plateau_start}) must be < end ({plateau_end})"
        )
    
    if plateau_start < 0:
        raise ValueError(
            f"Plateau window must start at or after stimulus onset (0 s), got start={plateau_start}"
        )
    
    mask = (times >= plateau_start) & (times < plateau_end)
    idx = np.where(mask)[0]
    
    if len(idx) < 1:
        raise ValueError(
            f"Plateau window [{plateau_start:.3f}, {plateau_end:.3f}] s contains no samples "
            f"for time range [{times.min():.3f}, {times.max():.3f}] s"
        )
    
    if logger is not None:
        actual_tmin = float(times[idx[0]])
        actual_tmax = float(times[idx[-1]])
        logger.info(
            f"Plateau window [{plateau_start:.3f}, {plateau_end:.3f}] s maps to "
            f"indices [{idx[0]}, {idx[-1]}] with actual timespan [{actual_tmin:.3f}, {actual_tmax:.3f}] s "
            f"(n_samples={len(idx)})"
        )
    
    return plateau_start, plateau_end, idx


def _check_baseline_already_applied(
    tfr_obj: Any,
    force: bool,
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> bool:
    if force:
        return False
    
    constants = _get_tfr_constants(config)
    comment = getattr(tfr_obj, "comment", None)
    if not isinstance(comment, str) or constants["baseline_sentinel"] not in comment:
        return False
    
    logger.warning(
        f"Detected baseline-corrected TFR by sentinel '{constants['baseline_sentinel']}' in comment; "
        f"skipping re-application to prevent double-baselining. "
        f"Use force=True to override."
    )
    return True


def _clip_baseline_window(
    baseline_start: float,
    baseline_end: float,
    times: np.ndarray,
    logger: logging.Logger,
) -> Tuple[float, float]:
    time_min = float(times[0])
    time_max = float(times[-1])
    
    baseline_start_clipped = max(baseline_start, time_min)
    baseline_end_clipped = min(baseline_end, time_max)
    
    if baseline_end_clipped > 0:
        baseline_end_clipped = min(0.0, time_max)
        logger.warning(f"Clipping baseline end to 0.0 (was {baseline_end})")
    
    if baseline_start_clipped != baseline_start or baseline_end_clipped != baseline_end:
        logger.info(
            f"Clipped baseline window from [{baseline_start}, {baseline_end}] to "
            f"[{baseline_start_clipped}, {baseline_end_clipped}] to fit data range."
        )
    
    return baseline_start_clipped, baseline_end_clipped


def apply_baseline_safe(
    tfr_obj: Any,
    baseline: Tuple[Optional[float], Optional[float]],
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> bool:
    logger = _get_logger(logger)
    config = config or load_settings()

    if mode is None:
        mode = str(config.get("tfr_processing.baseline_mode"))

    if _check_baseline_already_applied(tfr_obj, force, logger):
        return True

    if isinstance(tfr_obj, mne.time_frequency.AverageTFR):
        logger.warning("Applying baseline to AverageTFR (averaged) — prefer per-epoch baseline")

    times = np.asarray(tfr_obj.times)
    
    if min_samples is None:
        min_samples = int(config.get("time_frequency_analysis.min_baseline_samples"))
    
    baseline_start = float(times.min()) if baseline[0] is None else float(baseline[0])
    baseline_end = 0.0 if baseline[1] is None else float(baseline[1])
    
    if baseline_end > 0:
        error_msg = (
            f"Baseline window must end at or before 0 s (pre-stimulus), got [{baseline_start}, {baseline_end}]. "
            f"Post-stimulus baseline windows are invalid and indicate a configuration error."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    baseline_start_clipped, baseline_end_clipped = _clip_baseline_window(
        baseline_start, baseline_end, times, logger
    )
    
    _validate_baseline_samples(
        baseline_start_clipped, baseline_end_clipped, times, min_samples, logger
    )
    
    tfr_obj.apply_baseline(baseline=(baseline_start_clipped, baseline_end_clipped), mode=mode)
    _add_baseline_comment(tfr_obj, mode, baseline_start_clipped, baseline_end_clipped, config=config)
    logger.info(f"Applied baseline {(baseline_start_clipped, baseline_end_clipped)} with mode='{mode}'.")
    return True


def _validate_baseline_samples(
    baseline_start_clipped: float,
    baseline_end_clipped: float,
    times: np.ndarray,
    min_samples: int,
    logger: logging.Logger
) -> None:
    times_array = np.asarray(times)
    baseline_mask = (times_array >= baseline_start_clipped) & (times_array < baseline_end_clipped)
    n_samples = int(baseline_mask.sum())
    
    if baseline_start_clipped >= baseline_end_clipped or n_samples < max(1, min_samples):
        time_min_available = float(times[0])
        time_max_available = float(times[-1])
        error_msg = (
            f"Baseline window [{baseline_start_clipped}, {baseline_end_clipped}] "
            f"invalid/insufficient for available times "
            f"[{time_min_available}, {time_max_available}] (samples={n_samples}); "
            f"at least {min_samples} required"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _add_baseline_comment(
    tfr_obj: Any,
    mode: str,
    baseline_start_clipped: float,
    baseline_end_clipped: float,
    config: Optional[Any] = None
) -> None:
    constants = _get_tfr_constants(config)
    previous_comment = getattr(tfr_obj, "comment", "")
    baseline_tag = (
        f"{constants['baseline_sentinel']}mode={mode};"
        f"win=({baseline_start_clipped:.3f},{baseline_end_clipped:.3f})"
    )
    tfr_obj.comment = f"{previous_comment} | {baseline_tag}" if previous_comment else baseline_tag


def log_baseline_qc(
    tfr_obj,
    baseline: Tuple[Optional[float], Optional[float]],
    min_samples: int = 5,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
):
    logger = _get_logger(logger)
    constants = _get_tfr_constants(config)

    times = np.asarray(tfr_obj.times)
    b_start, b_end, idx = validate_baseline_indices(times, baseline, min_samples=min_samples, logger=logger)
    
    base = tfr_obj.data[:, :, :, idx]
    temporal_std = np.nanstd(base, axis=-1)
    med_temporal_std = float(np.nanmedian(temporal_std))
    epoch_means = np.nanmean(base, axis=(1, 2, 3))
    med = float(np.nanmedian(epoch_means))
    mad = float(np.nanmedian(np.abs(epoch_means - med)))
    divisor = abs(med) if abs(med) > constants["min_divisor"] else constants["min_divisor"]
    rcv = float(constants["mad_scaling_factor"] * mad / divisor)
    n_time = int(len(idx))
    
    msg = (
        f"Baseline QC: n_time={n_time}, median_temporal_std={med_temporal_std:.3g}, "
        f"epoch_MAD={mad:.3g}, RCV={rcv:.3g}"
    )
    logger.info(msg)
    
    if not np.isfinite(med_temporal_std) or not np.isfinite(rcv):
        logger.warning("Baseline QC: non-finite metrics detected; baseline may be unstable.")


def apply_baseline_and_crop(
    tfr_obj,
    baseline: Tuple[Optional[float], Optional[float]],
    crop_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force_baseline: bool = False,
    min_samples: Optional[int] = None,
    config=None,
) -> Tuple[float, float]:
    logger = _get_logger(logger)

    baseline_applied = apply_baseline_safe(
        tfr_obj,
        baseline=baseline,
        mode=mode,
        logger=logger,
        force=force_baseline,
        min_samples=min_samples,
        config=config,
    )

    baseline_used = _extract_baseline_from_comment(tfr_obj, baseline) if baseline_applied else baseline

    if crop_window is not None:
        _apply_crop_window(tfr_obj, crop_window, logger)
    
    return baseline_used


def _extract_baseline_from_comment(
    tfr_obj: Any,
    default_baseline: Tuple[Optional[float], Optional[float]]
) -> Tuple[float, float]:
    if not hasattr(tfr_obj, "comment"):
        return default_baseline
    
    comment = str(tfr_obj.comment)
    match = re.search(r"BASELINED:.*?win=\(([^,]+),([^)]+)\)", comment)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    
    return default_baseline


def _apply_crop_window(
    tfr_obj: Any,
    crop_window: Tuple[Optional[float], Optional[float]],
    logger: logging.Logger
) -> None:
    times = np.asarray(tfr_obj.times)
    tmin_req, tmax_req = crop_window
    tmin_avail, tmax_avail = float(times[0]), float(times[-1])
    
    tmin_req = float(times.min()) if tmin_req is None else float(tmin_req)
    tmax_req = float(times.max()) if tmax_req is None else float(tmax_req)
    
    tmin_clip = max(tmin_req, tmin_avail)
    tmax_clip = min(tmax_req, tmax_avail)
    
    if tmin_clip > tmax_clip:
        logger.warning(
            f"Requested crop window [{tmin_req}, {tmax_req}] invalid for available times "
            f"[{tmin_avail}, {tmax_avail}]; using full range."
        )
        tmin_clip, tmax_clip = tmin_avail, tmax_avail
    elif tmin_clip != tmin_req or tmax_clip != tmax_req:
        logger.info(
            f"Clipped crop window from [{tmin_req}, {tmax_req}] to "
            f"[{tmin_clip}, {tmax_clip}] to fit data range."
        )
    
    tfr_obj.crop(tmin=tmin_clip, tmax=tmax_clip)


###################################################################
# TFR Computation and Processing
###################################################################

def run_tfr_morlet(
    epochs,
    freqs,
    n_cycles,
    *,
    decim: int = 4,
    picks: Union[str, list] = "eeg",
    workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
):
    if workers is None:
        config = config or load_settings()
        workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
        workers = resolve_tfr_workers(workers_default=workers_default)
    
    return mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=decim,
        n_jobs=workers,
        picks=picks,
        verbose=False
    )


###################################################################
# TFR Data Extraction and Masking
###################################################################

def average_tfr_band(tfr_avg, fmin: float, fmax: float, tmin: float, tmax: float):
    freqs = np.asarray(tfr_avg.freqs)
    times = np.asarray(tfr_avg.times)
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    t_mask = (times >= tmin) & (times < tmax)
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    sel = tfr_avg.data[:, f_mask, :][:, :, t_mask]
    return sel.mean(axis=(1, 2))


def extract_epochwise_channel_values(tfr_epochs, fmin: float, fmax: float, tmin: float, tmax: float):
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    fmask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    tmask = (times >= float(tmin)) & (times < float(tmax))
    if fmask.sum() == 0 or tmask.sum() == 0:
        return None
    data = np.asarray(tfr_epochs.data)[:, :, fmask, :][:, :, :, tmask]
    return data.mean(axis=(2, 3))


def effective_plateau_window(
    times: np.ndarray,
    requested: Tuple[float, float]
) -> Tuple[Optional[float], Optional[float], np.ndarray]:
    t_arr = np.asarray(times)
    tmin_req, tmax_req = float(requested[0]), float(requested[1])
    tmin_eff = float(max(t_arr.min(), tmin_req))
    tmax_eff = float(min(t_arr.max(), tmax_req))
    if not np.isfinite(tmin_eff) or not np.isfinite(tmax_eff) or tmax_eff <= tmin_eff:
        return None, None, np.zeros_like(t_arr, dtype=bool)
    tmask = (t_arr >= tmin_eff) & (t_arr < tmax_eff)
    return tmin_eff, tmax_eff, tmask


def band_time_masks(
    freqs: np.ndarray,
    times: np.ndarray,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    f_arr = np.asarray(freqs)
    t_arr = np.asarray(times)
    fmask = (f_arr >= float(fmin)) & (f_arr <= float(fmax))
    tmask = (t_arr >= float(tmin)) & (t_arr < float(tmax))
    return fmask, tmask


def time_mask(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return (times >= tmin) & (times < tmax)


def freq_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= fmax)


def find_tfr_path(subject: str, task: str, deriv_root: Path) -> Optional[Path]:
    primary_path = deriv_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
    if primary_path.exists():
        return primary_path
    
    eeg_dir = deriv_root / f"sub-{subject}" / "eeg"
    if eeg_dir.exists():
        candidates = sorted(eeg_dir.glob(f"sub-{subject}_task-{task}*_epo-tfr.h5"))
        if candidates:
            return candidates[0]
    
    subj_dir = deriv_root / f"sub-{subject}"
    if subj_dir.exists():
        candidates = sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*_epo-tfr.h5"))
        if candidates:
            return candidates[0]
    
    return None


###################################################################
# Subject-Level TFR Computation
###################################################################

def compute_subject_tfr(
    subject: str,
    task: str,
    freq_min: float,
    freq_max: float,
    n_freqs: int,
    n_cycles_factor: float,
    tfr_decim: int,
    tfr_picks: Union[str, list],
    workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional["mne.time_frequency.EpochsTFR"], Optional[pd.DataFrame]]:
    from .data_loading import load_epochs_for_analysis
    
    config = load_settings()
    epochs, events_df = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=config.deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )
    
    if epochs is None:
        if logger:
            logger.error(f"No cleaned epochs for sub-{subject}, task-{task}")
        return None, None
    
    if events_df is None:
        if logger:
            logger.warning("Events missing; contrasts will be skipped for this subject.")
    else:
        _validate_trial_indices(events_df, logger)
    
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    power = run_tfr_morlet(
        epochs, freqs, n_cycles,
        decim=tfr_decim, picks=tfr_picks, workers=workers, logger=logger
    )
    
    return power, events_df


def _validate_trial_indices(events_df: pd.DataFrame, logger: Optional[logging.Logger]) -> None:
    trial_cols = ["trial", "trial_number", "trial_index"]
    trial_col = next((c for c in trial_cols if c in events_df.columns), None)
    if trial_col is None:
        return
    
    trial_vals = pd.to_numeric(events_df[trial_col], errors="coerce")
    run_col = next((c for c in ["run_id", "run", "run_number"] if c in events_df.columns), None)
    
    if run_col is not None:
        _validate_multi_run_trials(events_df, trial_col, run_col, logger)
    else:
        _validate_single_run_trials(trial_vals, trial_col, logger)


def _validate_multi_run_trials(
    events_df: pd.DataFrame,
    trial_col: str,
    run_col: str,
    logger: Optional[logging.Logger]
) -> None:
    grouped = events_df.groupby(run_col)[trial_col]
    within_run_dups = grouped.apply(
        lambda x: pd.to_numeric(x, errors="coerce").duplicated().sum()
    ).sum()
    
    if within_run_dups > 0:
        error_msg = (
            f"CRITICAL: Trial index column '{trial_col}' has {within_run_dups} duplicates within runs. "
            f"This indicates a serious alignment issue."
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    if logger:
        logger.debug(
            f"Trial indices validated: {len(events_df)} trials across {events_df[run_col].nunique()} runs "
            f"(trial numbers reset per run, which is expected)."
        )


def _validate_single_run_trials(
    trial_vals: pd.Series,
    trial_col: str,
    logger: Optional[logging.Logger]
) -> None:
    if trial_vals.duplicated().any():
        n_dup = trial_vals.duplicated().sum()
        error_msg = (
            f"CRITICAL: Trial index column '{trial_col}' has {n_dup} duplicates. "
            f"This indicates a serious alignment issue where multiple epochs map to the same trial."
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not trial_vals.is_monotonic_increasing and not trial_vals.is_monotonic_decreasing:
        n_gaps = (trial_vals.diff() != 1).sum() - 1
        if logger:
            logger.debug(
                f"Trial index column '{trial_col}' is non-monotonic ({n_gaps} gaps). "
                f"This is expected when trials were dropped during preprocessing."
            )


###################################################################
# Group ROI Helpers
###################################################################

def collect_group_temperatures(
    events_by_subj: List[Optional[pd.DataFrame]],
    temperature_columns: List[str],
) -> List[float]:
    temps = set()
    for ev in events_by_subj:
        if ev is None:
            continue
        tcol = None
        for c in temperature_columns:
            if c in ev.columns:
                tcol = c
                break
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1).dropna().unique()
        for v in vals:
            temps.add(float(v))
    return sorted(temps)


def avg_alltrials_to_avg_tfr(
    power: "mne.time_frequency.EpochsTFR",
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> "mne.time_frequency.AverageTFR":
    tfr_all = power.copy()
    apply_baseline_and_crop(tfr_all, baseline=baseline, mode="logratio", logger=logger)
    return tfr_all.average()


def avg_by_mask_to_avg_tfr(
    power: "mne.time_frequency.EpochsTFR",
    mask: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    t = power.copy()[mask]
    apply_baseline_and_crop(t, baseline=baseline, mode="logratio", logger=logger)
    return t.average()


def align_avg_tfrs(
    tfr_list: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Info], Optional[np.ndarray]]:
    if not tfr_list:
        return None, None
    tfr_list = [t for t in tfr_list if t is not None]
    if not tfr_list:
        return None, None
    base = tfr_list[0]
    base_times = np.asarray(base.times)
    base_freqs = np.asarray(base.freqs)
    base_chs = list(base.info["ch_names"])
    keep: List[Tuple[str, "mne.time_frequency.AverageTFR"]] = [("S0", base)]
    for i, tfr in enumerate(tfr_list[1:], start=1):
        ok = np.allclose(tfr.times, base_times) and np.allclose(tfr.freqs, base_freqs)
        if not ok:
            if logger:
                logger.warning(f"Skipping subject {i}: times/freqs mismatch for group alignment")
            continue
        keep.append((f"S{i}", tfr))
    if len(keep) == 0:
        return None, None
    ch_sets = [set(t.info["ch_names"]) for _, t in keep]
    common = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    if len(common) == 0:
        if logger:
            logger.warning("No common channels across subjects; cannot align")
        return None, None
    arrs = []
    for tag, t in keep:
        idxs = [t.info["ch_names"].index(ch) for ch in common]
        arrs.append(np.asarray(t.data)[idxs, :, :])
    data = np.stack(arrs, axis=0)
    pick_inds = [base_chs.index(ch) for ch in common]
    info_common = mne.pick_info(base.info, pick_inds)
    return info_common, data


def align_paired_avg_tfrs(
    list_a: List["mne.time_frequency.AverageTFR"],
    list_b: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Info], Optional[np.ndarray], Optional[np.ndarray]]:
    if not list_a or not list_b:
        return None, None, None
    n = min(len(list_a), len(list_b))
    pairs = []
    for i in range(n):
        a = list_a[i]
        b = list_b[i]
        if a is None or b is None:
            continue
        pairs.append((a, b))
    if not pairs:
        return None, None, None
    base_a, base_b = pairs[0]
    base_times = np.asarray(base_a.times)
    base_freqs = np.asarray(base_a.freqs)
    keep: list[tuple["mne.time_frequency.AverageTFR", "mne.time_frequency.AverageTFR"]] = []
    for idx, (a, b) in enumerate(pairs):
        ok = (
            np.allclose(a.times, base_times) and np.allclose(a.freqs, base_freqs)
            and np.allclose(b.times, base_times) and np.allclose(b.freqs, base_freqs)
        )
        if not ok:
            if logger:
                logger.warning(f"Skipping subject pair {idx}: times/freqs mismatch for paired alignment")
            continue
        keep.append((a, b))
    if not keep:
        return None, None, None
    per_pair_common = []
    for a, b in keep:
        per_pair_common.append(set(a.info["ch_names"]) & set(b.info["ch_names"]))
    common = list(sorted(set.intersection(*per_pair_common))) if per_pair_common else []
    if len(common) == 0:
        if logger:
            logger.warning("Paired alignment: no common channels across retained pairs")
        return None, None, None
    
    data_a = []
    data_b = []
    for a, b in keep:
        idx_a = [a.info["ch_names"].index(ch) for ch in common]
        idx_b = [b.info["ch_names"].index(ch) for ch in common]
        data_a.append(np.asarray(a.data)[idx_a, :, :])
        data_b.append(np.asarray(b.data)[idx_b, :, :])
    data_a_arr = np.stack(data_a, axis=0)
    data_b_arr = np.stack(data_b, axis=0)
    pick_inds = [base_a.info["ch_names"].index(ch) for ch in common]
    info_common = mne.pick_info(base_a.info, pick_inds)
    return info_common, data_a_arr, data_b_arr


###################################################################
# Time Window Utilities
###################################################################


def clip_time_range(times: np.ndarray, tmin_req: float, tmax_req: float) -> Optional[Tuple[float, float]]:
    tmin_clip = float(max(times.min(), tmin_req))
    tmax_clip = float(min(times.max(), tmax_req))
    
    if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
        return None
    
    return tmin_clip, tmax_clip


def create_time_windows_fixed_size(tmin_start: float, tmax_clip: float, window_size_ms: float, config: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    window_size_s = window_size_ms / 1000.0
    window_starts = np.arange(tmin_start, tmax_clip, window_size_s)
    window_ends = window_starts + window_size_s
    window_ends = np.minimum(window_ends, tmax_clip)
    
    valid_mask = window_starts < tmax_clip
    window_starts = window_starts[valid_mask]
    window_ends = window_ends[valid_mask]
    
    return window_starts, window_ends


def create_time_windows_fixed_count(tmin_clip: float, tmax_clip: float, window_count: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(tmin_clip, tmax_clip, window_count + 1)
    window_starts = edges[:-1]
    window_ends = edges[1:]
    return window_starts, window_ends


def create_time_mask_strict(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    tmask = (times >= float(tmin)) & (times < float(tmax))
    
    if not np.any(tmask):
        msg = f"Time window [{tmin}, {tmax}] outside data range [{times.min():.2f}, {times.max():.2f}]"
        raise ValueError(msg)
    
    return tmask


def create_time_mask_loose(times: np.ndarray, tmin: float, tmax: float, logger: Optional[logging.Logger] = None) -> np.ndarray:
    tmask = (times >= float(tmin)) & (times < float(tmax))
    
    if not np.any(tmask):
        msg = f"Time window [{tmin}, {tmax}] outside data range [{times.min():.2f}, {times.max():.2f}]"
        if logger:
            logger.warning(f"{msg}; using entire time span")
        else:
            logging.getLogger(__name__).warning(f"{msg}; using entire time span")
        tmask = np.ones_like(times, dtype=bool)
    
    return tmask


def clip_time_window(
    time_window: Tuple[float, float],
    tfr_times: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[float, float]]:
    from .stats_utils import _safe_float
    
    tmin_req = _safe_float(time_window[0])
    tmax_req = _safe_float(time_window[1])
    tmin_avail = float(tfr_times[0])
    tmax_avail = float(tfr_times[-1])
    tmin_clip = max(tmin_req, tmin_avail)
    tmax_clip = min(tmax_req, tmax_avail)
    
    if tmin_clip > tmax_clip:
        if logger:
            logger.warning(
                f"Requested window [{tmin_req}, {tmax_req}] outside TFR range "
                f"[{tmin_avail}, {tmax_avail}]; aborting TF correlation computation"
            )
        return None
    
    return (tmin_clip, tmax_clip)


###################################################################
# TFR Time Series Extraction Utilities
###################################################################

def extract_band_series_from_tfr(
    tfr: "mne.time_frequency.AverageTFR",
    fmin: float,
    fmax: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if tfr is None or fmin >= fmax:
        return None, None
    
    freq_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    if freq_mask.sum() == 0:
        return None, None
    
    series_logr = np.nanmean(tfr.data[:, freq_mask, :], axis=(0, 1))
    ratio_data = np.power(10.0, tfr.data[:, freq_mask, :])
    series_ratio = np.nanmean(ratio_data, axis=(0, 1))
    
    return series_logr, series_ratio


def interpolate_to_reference_times(
    values: np.ndarray,
    times: np.ndarray,
    reference_times: np.ndarray,
    config: Optional[Any] = None
) -> np.ndarray:
    if values.size == 0 or times.size == 0 or reference_times.size == 0:
        return np.full_like(reference_times, np.nan)
    
    constants = _get_tfr_constants(config)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < constants["min_samples_for_stats"]:
        return np.full_like(reference_times, np.nan)
    
    if finite_mask.sum() < len(values):
        values = np.interp(times, times[finite_mask], values[finite_mask])
    
    return np.interp(reference_times, times, values)


def extract_band_time_courses(
    tfr_list: List["mne.time_frequency.AverageTFR"],
    bands: List[str],
    freq_bands: Dict[str, Tuple[float, float]],
    tmin: float,
    tmax: float,
    config: Optional[Any] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], np.ndarray]:
    if not tfr_list or not bands or not freq_bands:
        return {}, {}, np.array([])
    
    reference = tfr_list[0]
    ref_mask = time_mask(reference.times, tmin, tmax)
    reference_times = reference.times[ref_mask]
    
    band_timecourses_logr = {band: [] for band in bands}
    band_timecourses_pct = {band: [] for band in bands}
    
    for tfr in tfr_list:
        for band in bands:
            if band not in freq_bands:
                continue
            
            fmin, fmax = freq_bands[band]
            series_logr, series_ratio = extract_band_series_from_tfr(tfr, fmin, fmax)
            
            if series_logr is None:
                continue
            
            time_mask_subj = time_mask(tfr.times, tmin, tmax)
            constants = _get_tfr_constants(config)
            if time_mask_subj.sum() < constants["min_samples_for_stats"]:
                continue
            
            times_subj = tfr.times[time_mask_subj]
            values_logr = series_logr[time_mask_subj]
            values_ratio = series_ratio[time_mask_subj]
            
            if not np.any(np.isfinite(values_logr)) and not np.any(np.isfinite(values_ratio)):
                continue
            
            values_logr_ref = interpolate_to_reference_times(values_logr, times_subj, reference_times, config=config)
            values_ratio_ref = interpolate_to_reference_times(values_ratio, times_subj, reference_times, config=config)
            
            band_timecourses_logr[band].append(values_logr_ref)
            values_pct = 100.0 * (values_ratio_ref - 1.0)
            band_timecourses_pct[band].append(values_pct)
    
    return band_timecourses_logr, band_timecourses_pct, reference_times


###################################################################
# TFR Object Extraction Utilities
###################################################################

def extract_trial_band_power(tfr_epochs, fmin: float, fmax: float, tmin: float, tmax: float) -> Optional[np.ndarray]:
    if not isinstance(tfr_epochs, mne.time_frequency.EpochsTFR):
        return None
    
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    f_mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    t_mask = (times >= float(tmin)) & (times < float(tmax))
    
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    
    sel = np.asarray(tfr_epochs.data)[:, :, f_mask, :][:, :, :, t_mask]
    return sel.mean(axis=(2, 3))


def extract_band_channel_means(tfr_avg, freq_mask: np.ndarray) -> np.ndarray:
    return tfr_avg.data[:, freq_mask, :].mean(axis=(1, 2))


def build_roi_channel_mask(ch_names: List[str], roi_channels: List[str]) -> np.ndarray:
    return np.array([ch in roi_channels for ch in ch_names], dtype=bool)


def extract_significant_roi_channels(ch_names: List[str], mask_vec: np.ndarray, sig_mask: np.ndarray) -> Tuple[List[int], List[str]]:
    roi_sig_indices = [i for i in range(len(ch_names)) if mask_vec[i] and sig_mask[i]]
    roi_sig_chs = [ch_names[i] for i in roi_sig_indices]
    return roi_sig_indices, roi_sig_chs


def extract_roi_from_tfr(avg_tfr, roi: str, roi_map: Optional[Dict[str, List[str]]], config) -> Optional[Any]:
    import mne
    
    if roi_map is not None:
        chs_all = roi_map.get(roi)
        if chs_all is not None:
            subj_chs = avg_tfr.info['ch_names']
            canon_subj = {canonicalize_ch_name(ch).upper(): ch for ch in subj_chs}
            want = {canonicalize_ch_name(ch).upper() for ch in chs_all}
            chs = [canon_subj[canonicalize_ch_name(ch).upper()] for ch in subj_chs if canonicalize_ch_name(ch).upper() in want]
            if len(chs) > 0:
                picks = mne.pick_channels(subj_chs, include=chs, exclude=[])
                roi_tfr = avg_tfr.copy()
                roi_tfr.data = np.nanmean(np.asarray(avg_tfr.data)[picks, :, :], axis=0, keepdims=True)
                roi_tfr.info = mne.create_info([f"ROI:{roi}"], sfreq=avg_tfr.info['sfreq'], ch_types='eeg')
                return roi_tfr
    
    roi_defs = get_rois(config)
    pats = roi_defs.get(roi, [])
    chs = find_roi_channels(avg_tfr.info, pats)
    if len(chs) > 0:
        picks = mne.pick_channels(avg_tfr.info['ch_names'], include=chs, exclude=[])
        roi_tfr = avg_tfr.copy()
        roi_tfr.data = np.nanmean(np.asarray(avg_tfr.data)[picks, :, :], axis=0, keepdims=True)
        roi_tfr.info = mne.create_info([f"ROI:{roi}"], sfreq=avg_tfr.info['sfreq'], ch_types='eeg')
        return roi_tfr
    return None


def extract_roi_contrast_data(power: np.ndarray, ev: pd.DataFrame, roi: str, roi_map: Optional[Dict[str, List[str]]], config) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    from .io_utils import get_pain_column_from_config
    from .tfr_utils import avg_by_mask_to_avg_tfr
    import mne
    
    if power is None or ev is None:
        return None
    
    pain_col = get_pain_column_from_config(config, ev) if config else None
    if pain_col is None or pain_col not in ev.columns:
        return None
    
    pain_vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int)
    pain_mask = pain_vals == 1
    non_mask = pain_vals == 0
    
    if not (pain_mask.any() and non_mask.any()):
        return None
    
    a_p = avg_by_mask_to_avg_tfr(power, pain_mask)
    a_n = avg_by_mask_to_avg_tfr(power, non_mask)
    
    if a_p is None or a_n is None:
        return None
    
    r_p = extract_roi_from_tfr(a_p, roi, roi_map, config)
    r_n = extract_roi_from_tfr(a_n, roi, roi_map, config)
    
    if r_p is not None and r_n is not None:
        return r_p.data[0], r_n.data[0]
    
    return None


def extract_tfr_object(tfr: Any):
    if tfr is None or (isinstance(tfr, list) and len(tfr) == 0):
        return None
    return tfr[0] if isinstance(tfr, list) else tfr


def extract_band_power(
    tfr_data: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    time_mask_array: np.ndarray,
) -> Optional[np.ndarray]:
    freq_mask_indices = freq_mask(freqs, fmin, fmax)
    if not np.any(freq_mask_indices):
        return None
    
    band_power = tfr_data[:, :, freq_mask_indices, :][:, :, :, time_mask_array].mean(axis=(2, 3))
    return band_power


def process_temporal_bin(
    tfr_data: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    channel_names: List[str],
    band: str,
    fmin: float,
    fmax: float,
    time_start: float,
    time_end: float,
    time_label: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    logger = _get_logger(logger)
    
    time_mask_array = time_mask(times, time_start, time_end)
    if not np.any(time_mask_array):
        logger.warning(
            f"No time points in bin {time_label} ({time_start}-{time_end}s) for band '{band}'"
        )
        return None
    
    band_power = extract_band_power(tfr_data, freqs, fmin, fmax, time_mask_array)
    if band_power is None:
        return None
    
    column_names = [f"pow_{band}_{ch}_{time_label}" for ch in channel_names]
    return band_power, column_names


def create_tfr_subset(tfr, n: int):
    return tfr.copy()[:n]


def apply_baseline_and_average(
    tfr,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None
):
    tfr_copy = tfr.copy()
    baseline_used = apply_baseline_and_crop(
        tfr_copy, baseline=baseline, mode="logratio", logger=logger
    )
    
    if isinstance(tfr_copy, mne.time_frequency.EpochsTFR):
        tfr_avg = tfr_copy.average()
    else:
        tfr_avg = tfr_copy
    
    return tfr_avg, baseline_used


__all__ = [
    "canonicalize_ch_name",
    "find_roi_channels",
    "build_rois_from_info",
    "get_rois",
    "compute_adaptive_n_cycles",
    "log_tfr_resolution",
    "validate_tfr_parameters",
    "resolve_tfr_workers",
    "get_bands_for_tfr",
    "read_tfr_average_with_logratio",
    "save_tfr_with_sidecar",
    "validate_baseline_window",
    "validate_baseline_indices",
    "apply_baseline_safe",
    "apply_baseline_and_crop",
    "log_baseline_qc",
    "run_tfr_morlet",
    "average_tfr_band",
    "extract_epochwise_channel_values",
    "effective_plateau_window",
    "band_time_masks",
    "time_mask",
    "freq_mask",
    "compute_subject_tfr",
    "collect_group_temperatures",
    "avg_alltrials_to_avg_tfr",
    "avg_by_mask_to_avg_tfr",
    "align_avg_tfrs",
    "align_paired_avg_tfrs",
    "clip_time_window",
    "clip_time_range",
    "create_time_windows_fixed_size",
    "create_time_windows_fixed_count",
    "create_time_mask_strict",
    "create_time_mask_loose",
    # TFR data extraction utilities
    "extract_trial_band_power",
    "extract_band_channel_means",
    # ROI processing utilities
    "build_roi_channel_mask",
    "extract_significant_roi_channels",
    "extract_roi_from_tfr",
    "extract_roi_contrast_data",
    # TFR time series extraction utilities
    "extract_band_series_from_tfr",
    "interpolate_to_reference_times",
    "extract_band_time_courses",
    # TFR object extraction utilities
    "extract_tfr_object",
    "extract_band_power",
    "process_temporal_bin",
    # TFR manipulation utilities
    "create_tfr_subset",
    "apply_baseline_and_average",
]
