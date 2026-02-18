"""
Feature Extraction Pipeline (Canonical)
========================================

Single source of truth for feature extraction orchestration.

Usage:
    # Single subject
    pipeline = FeaturePipeline(config=config)
    pipeline.process_subject("0001", "thermalactive")

    # Multiple subjects
    pipeline.run_batch(["0001", "0002"])
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.analysis.features.api import (
    extract_all_features,
    extract_precomputed_features,
)
from eeg_pipeline.analysis.features.preparation import precompute_data
from eeg_pipeline.analysis.features.results import (
    ExtractionResult,
    FeatureExtractionResult,
    FeatureSet,
)
from eeg_pipeline.analysis.features.selection import resolve_feature_categories
from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES
from eeg_pipeline.infra.paths import (
    _load_events_df,
    deriv_features_path,
    deriv_stats_path,
    ensure_dir,
)
from eeg_pipeline.infra.tsv import write_parquet
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.tfr import compute_complex_tfr, compute_tfr_morlet
from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.features import align_feature_dataframes
from eeg_pipeline.utils.data.feature_io import (
    save_all_features,
    save_dropped_trials_log,
)


_ACCUMULATOR_EXTRAS = ["baseline", "pac_time"]
_FEATURE_ACCUMULATOR_KEYS = list(FEATURE_CATEGORIES) + _ACCUMULATOR_EXTRAS

_TFR_CATEGORIES = {"power", "itpc", "pac"}
_PRECOMPUTE_CATEGORIES = {
    "connectivity",
    "erds",
    "ratios",
    "asymmetry",
    "pac",
    "complexity",
    "bursts",
}


def _resolve_time_ranges(explicit_windows: Optional[List[Dict[str, Any]]], tmin: Optional[float], tmax: Optional[float]) -> List[Dict[str, Any]]:
    """Resolve time ranges from explicit windows or single range parameters."""
    if explicit_windows:
        return list(explicit_windows)
    return [{"name": None, "tmin": tmin, "tmax": tmax}]


def _calculate_total_steps(n_ranges: int) -> int:
    """Calculate total progress steps: load + 3 per time range (extract, align, save)."""
    return 1 + (n_ranges * 3)


def _load_fixed_templates(
    templates_path: Optional[Path],
    logger: Any,
) -> tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[str]]]:
    """Load fixed templates from file if path exists."""
    if templates_path is None or not templates_path.exists():
        return None, None, None

    def _decode_optional_text_array(values: Optional[np.ndarray]) -> Optional[List[str]]:
        if values is None:
            return None
        decoded: List[str] = []
        for value in np.asarray(values).tolist():
            if isinstance(value, (bytes, np.bytes_)):
                decoded.append(value.decode("utf-8", errors="ignore"))
            else:
                decoded.append(str(value))
        return decoded

    try:
        with np.load(templates_path, allow_pickle=False) as data:
            if "templates" not in data:
                logger.warning("Fixed templates file missing 'templates': %s", templates_path)
                return None, None, None
            templates = np.asarray(data["templates"], dtype=float)
            raw_ch_names = data["ch_names"] if "ch_names" in data else None
            raw_labels = None
            for label_key in ("labels", "template_labels", "state_labels", "class_labels"):
                if label_key in data:
                    raw_labels = data[label_key]
                    break
    except Exception as exc:
        logger.warning("Failed loading fixed templates from %s: %s", templates_path, exc)
        return None, None, None

    ch_names = _decode_optional_text_array(raw_ch_names)
    labels = _decode_optional_text_array(raw_labels)

    logger.info("Loaded fixed templates from %s", templates_path)
    return templates, ch_names, labels


def _precompute_tfr_if_needed(
    epochs: "mne.Epochs",
    time_ranges: List[Dict[str, Any]],
    feature_categories: List[str],
    config: Any,
    logger: Any,
) -> Optional[Any]:
    """Pre-compute TFR on full epochs if multiple ranges and TFR categories requested."""
    needs_tfr = len(time_ranges) > 1 and any(cat in feature_categories for cat in _TFR_CATEGORIES)
    if not needs_tfr:
        return None
    
    logger.info("Pre-computing TFR on full epochs for multi-range extraction...")
    return compute_tfr_morlet(epochs, config, logger=logger)


def _precompute_complex_tfr_if_needed(
    epochs: "mne.Epochs",
    time_ranges: List[Dict[str, Any]],
    feature_categories: List[str],
    config: Any,
    logger: Any,
) -> Optional[Any]:
    """Pre-compute complex TFR once for multi-range extraction.

    This avoids recomputing (and reallocating) a full-length complex TFR for each
    time range when ITPC is requested, or when PAC explicitly uses TFR-based mode.
    """
    if len(time_ranges) <= 1:
        return None

    needs_itpc = "itpc" in feature_categories or "phase" in feature_categories
    needs_pac_complex = False
    if "pac" in feature_categories and hasattr(config, "get"):
        pac_cfg = config.get("feature_engineering.pac", {}) or {}
        pac_source = str(pac_cfg.get("source", "precomputed")).strip().lower()
        needs_pac_complex = pac_source != "precomputed"

    if not (needs_itpc or needs_pac_complex):
        return None

    from eeg_pipeline.analysis.features.preparation import (
        _apply_spatial_transform,
        _get_spatial_transform_type,
    )

    itpc_transform = (
        _get_spatial_transform_type(config, feature_family="itpc")
        if needs_itpc
        else "none"
    )
    pac_transform = (
        _get_spatial_transform_type(config, feature_family="pac")
        if needs_pac_complex
        else "none"
    )
    if needs_itpc and needs_pac_complex and itpc_transform != pac_transform:
        logger.warning(
            "Skipping shared complex TFR precompute: ITPC requires spatial_transform='%s' "
            "but PAC requires '%s'. Each family will compute its own complex TFR.",
            itpc_transform,
            pac_transform,
        )
        return None

    phase_transform = itpc_transform if needs_itpc else pac_transform

    epochs_for_complex = epochs
    if phase_transform in {"csd", "laplacian"}:
        epochs_for_complex = epochs.copy().pick_types(
            eeg=True, meg=False, eog=False, stim=False, exclude="bads"
        )
        epochs_for_complex = _apply_spatial_transform(
            epochs_for_complex, phase_transform, config, logger
        )

    logger.info("Pre-computing complex TFR on full epochs for multi-range extraction...")
    complex_tfr = compute_complex_tfr(epochs_for_complex, config, logger)
    if complex_tfr is not None:
        try:
            setattr(complex_tfr, "_spatial_transform", str(phase_transform))
        except Exception:
            pass
    return complex_tfr


def _precompute_intermediates_if_needed(
    epochs: "mne.Epochs",
    time_ranges: List[Dict[str, Any]],
    feature_categories: List[str],
    bands: Optional[List[str]],
    config: Any,
    logger: Any,
) -> Optional[PrecomputedData]:
    """Pre-compute shared intermediates if multiple ranges and precompute categories requested."""
    needs_precompute = len(time_ranges) > 1 and any(
        cat in feature_categories for cat in _PRECOMPUTE_CATEGORIES
    )
    if not needs_precompute:
        return None
    
    logger.info("Pre-computing shared intermediates on full epochs for multi-range extraction...")
    resolved_bands = bands or get_frequency_band_names(config)
    windows_spec = TimeWindowSpec(
        times=epochs.times,
        config=config,
        sampling_rate=float(epochs.info["sfreq"]),
        logger=logger,
        explicit_windows=time_ranges,
    )
    return precompute_data(
        epochs,
        resolved_bands,
        config,
        logger,
        compute_psd_data=True,
        windows_spec=windows_spec,
    )


def _create_feature_accumulator() -> Dict[str, List[pd.DataFrame]]:
    """Create accumulator dictionary for merging features from multiple time ranges."""
    return {key: [] for key in _FEATURE_ACCUMULATOR_KEYS}


def _unpack_feature_results(features: FeatureExtractionResult) -> Dict[str, Any]:
    """Extract all feature DataFrames and column lists from extraction result."""
    return {
        "pow_df": features.pow_df,
        "pow_cols": features.pow_cols,
        "baseline_df": features.baseline_df,
        "baseline_cols": features.baseline_cols,
        "conn_df": features.conn_df,
        "conn_cols": features.conn_cols,
        "dconn_df": features.dconn_df,
        "dconn_cols": features.dconn_cols,
        "source_df": features.source_df,
        "source_cols": features.source_cols,
        "aper_df": features.aper_df,
        "aper_cols": features.aper_cols,
        "erp_df": features.erp_df,
        "erp_cols": features.erp_cols,
        "itpc_df": features.phase_df,
        "itpc_cols": features.phase_cols,
        "itpc_trial_df": features.itpc_trial_df,
        "itpc_trial_cols": features.itpc_trial_cols,
        "pac_df": features.pac_df,
        "pac_trials_df": features.pac_trials_df,
        "pac_time_df": features.pac_time_df,
        "comp_df": features.comp_df,
        "comp_cols": features.comp_cols,
        "bursts_df": features.bursts_df,
        "bursts_cols": features.bursts_cols,
        "spectral_df": features.spectral_df,
        "spectral_cols": features.spectral_cols,
        "erds_df": features.erds_df,
        "erds_cols": features.erds_cols,
        "ratios_df": features.ratios_df,
        "ratios_cols": features.ratios_cols,
        "asymmetry_df": features.asymmetry_df,
        "asymmetry_cols": features.asymmetry_cols,
        "quality_df": features.quality_df,
        "quality_cols": features.quality_cols,
        "microstates_df": getattr(features, "microstates_df", None),
        "microstates_cols": getattr(features, "microstates_cols", []),
        "aper_qc": features.aper_qc,
    }


def _build_extra_blocks(unpacked: Dict[str, Any], features: FeatureExtractionResult) -> Dict[str, pd.DataFrame]:
    """Build extra blocks dictionary for alignment, filtering out None/empty DataFrames."""
    extra_blocks = {
        "itpc": unpacked["itpc_df"],
        "itpc_trial": unpacked["itpc_trial_df"],
        # Align PAC at the per-trial level when available.
        "pac_trials": unpacked["pac_trials_df"] if unpacked["pac_trials_df"] is not None else unpacked["pac_df"],
        "pac_time": unpacked["pac_time_df"],
        "complexity": unpacked["comp_df"],
        "spectral": unpacked["spectral_df"],
        "erp": unpacked["erp_df"],
        "bursts": unpacked["bursts_df"],
        "erds": unpacked["erds_df"],
        "dconn": unpacked["dconn_df"],
        "source": unpacked["source_df"],
        "ratios": features.ratios_df,
        "asymmetry": features.asymmetry_df,
        "microstates": unpacked.get("microstates_df"),
        "quality": features.quality_df,
    }
    return {
        k: v
        for k, v in extra_blocks.items()
        if v is not None and not getattr(v, "empty", False)
    }


def _update_from_aligned_extra(
    unpacked: Dict[str, Any],
    features: FeatureExtractionResult,
    extra_blocks: Dict[str, pd.DataFrame],
) -> None:
    """Update unpacked dict and features object with filtered extra blocks."""
    unpacked["itpc_df"] = extra_blocks.get("itpc", unpacked["itpc_df"])
    unpacked["itpc_trial_df"] = extra_blocks.get("itpc_trial", unpacked["itpc_trial_df"])
    # Backward-compatibility: accept legacy key "pac" if present.
    aligned_pac_trials = extra_blocks.get("pac_trials", extra_blocks.get("pac"))
    if aligned_pac_trials is not None:
        unpacked["pac_trials_df"] = aligned_pac_trials
    unpacked["pac_df"] = extra_blocks.get("pac", unpacked["pac_df"])
    unpacked["pac_time_df"] = extra_blocks.get("pac_time", unpacked["pac_time_df"])
    unpacked["comp_df"] = extra_blocks.get("complexity", unpacked["comp_df"])
    unpacked["spectral_df"] = extra_blocks.get("spectral", unpacked["spectral_df"])
    unpacked["erp_df"] = extra_blocks.get("erp", unpacked["erp_df"])
    unpacked["bursts_df"] = extra_blocks.get("bursts", unpacked["bursts_df"])
    unpacked["erds_df"] = extra_blocks.get("erds", unpacked["erds_df"])
    unpacked["dconn_df"] = extra_blocks.get("dconn", unpacked["dconn_df"])
    unpacked["source_df"] = extra_blocks.get("source", unpacked["source_df"])
    unpacked["microstates_df"] = extra_blocks.get("microstates", unpacked.get("microstates_df"))
    features.ratios_df = extra_blocks.get("ratios", features.ratios_df)
    features.asymmetry_df = extra_blocks.get("asymmetry", features.asymmetry_df)
    features.quality_df = extra_blocks.get("quality", features.quality_df)
    if hasattr(features, "microstates_df"):
        features.microstates_df = extra_blocks.get("microstates", features.microstates_df)


def _build_feature_qc(features: FeatureExtractionResult, ctx: FeatureContext) -> Dict[str, Any]:
    """Build feature QC dictionary from extraction result and context."""
    qc: Dict[str, Any] = {}
    if features.aper_qc is not None:
        qc["aperiodic"] = features.aper_qc
    if ctx.precomputed is not None and hasattr(ctx.precomputed, "qc") and ctx.precomputed.qc is not None:
        qc["precomputed_intermediates"] = asdict(ctx.precomputed.qc)
    return qc


def _accumulate_features(
    accumulated: Dict[str, List[pd.DataFrame]],
    unpacked: Dict[str, Any],
    features: FeatureExtractionResult,
    aligned: Dict[str, pd.DataFrame],
) -> None:
    """Accumulate aligned features for later merging."""
    feature_mapping = {
        "power": aligned.get("pow_df_aligned"),
        "baseline": aligned.get("baseline_df_aligned"),
        "connectivity": aligned.get("conn_df_aligned"),
        "directedconnectivity": unpacked["dconn_df"],
        "sourcelocalization": unpacked["source_df"],
        "aperiodic": aligned.get("aper_df_aligned"),
        "erp": unpacked["erp_df"],
        "itpc": unpacked["itpc_df"],
        "pac": unpacked["pac_trials_df"] if unpacked["pac_trials_df"] is not None else unpacked["pac_df"],
        "pac_time": unpacked["pac_time_df"],
        "complexity": unpacked["comp_df"],
        "bursts": unpacked["bursts_df"],
        "spectral": unpacked["spectral_df"],
        "erds": unpacked["erds_df"],
        "ratios": features.ratios_df,
        "asymmetry": features.asymmetry_df,
        "microstates": unpacked.get("microstates_df"),
        "quality": features.quality_df,
    }
    
    for key, df in feature_mapping.items():
        if key not in accumulated:
            accumulated[key] = []
        if df is not None and not df.empty:
            accumulated[key].append(df)


def _get_df_cols(df: Optional[pd.DataFrame]) -> int:
    """Get number of columns from DataFrame, returning 0 if None or empty."""
    return df.shape[1] if df is not None and not df.empty else 0


def _merge_dataframes(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Merge DataFrames by concatenating columns, avoiding duplicates."""
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        return None
    if len(valid_dfs) == 1:
        return valid_dfs[0]

    common_attrs = dict(getattr(valid_dfs[0], "attrs", {}) or {})
    for df in valid_dfs[1:]:
        attrs = dict(getattr(df, "attrs", {}) or {})
        for key in list(common_attrs.keys()):
            if attrs.get(key) != common_attrs[key]:
                common_attrs.pop(key, None)

    merged = pd.concat(valid_dfs, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    if common_attrs:
        merged.attrs.update(common_attrs)
    return merged


def _save_merged_features(
    accumulated: Dict[str, List[pd.DataFrame]],
    features_dir: Path,
    config: Any,
    logger: Any,
) -> None:
    """Merge and save accumulated features from multiple time ranges."""
    from eeg_pipeline.utils.config.loader import get_config_value

    also_save_csv = bool(
        get_config_value(config, "feature_engineering.output.also_save_csv", False)
    )

    feature_file_mapping = {
        "power": ("features_power.parquet", ["power", "baseline"]),
        "connectivity": ("features_connectivity.parquet", ["connectivity"]),
        "directedconnectivity": ("features_directedconnectivity.parquet", ["directedconnectivity"]),
        "sourcelocalization": ("features_sourcelocalization.parquet", ["sourcelocalization"]),
        "aperiodic": ("features_aperiodic.parquet", ["aperiodic"]),
        "erp": ("features_erp.parquet", ["erp"]),
        "itpc": ("features_itpc.parquet", ["itpc"]),
        "pac": ("features_pac_trials.parquet", ["pac"]),
        "complexity": ("features_complexity.parquet", ["complexity"]),
        "bursts": ("features_bursts.parquet", ["bursts"]),
        "spectral": ("features_spectral.parquet", ["spectral"]),
        "erds": ("features_erds.parquet", ["erds"]),
        "ratios": ("features_ratios.parquet", ["ratios"]),
        "asymmetry": ("features_asymmetry.parquet", ["asymmetry"]),
        "microstates": ("features_microstates.parquet", ["microstates"]),
        "quality": ("features_quality.parquet", ["quality"]),
    }

    for filename, keys in feature_file_mapping.values():
        dfs_to_merge = []
        for key in keys:
            dfs_to_merge.extend(accumulated.get(key, []))

        merged_df = _merge_dataframes(dfs_to_merge)
        if merged_df is not None:
            from eeg_pipeline.utils.data.feature_io import _get_folder_for_feature
            from eeg_pipeline.domain.features.naming import generate_manifest

            base_name = filename.replace(".parquet", "")
            folder = _get_folder_for_feature(base_name, config)
            save_path = features_dir / folder / filename
            write_parquet(merged_df, save_path)
            if also_save_csv:
                from eeg_pipeline.infra.tsv import write_csv

                csv_path = save_path.with_suffix(".csv")
                write_csv(merged_df, csv_path, index=False)
                logger.info("Also saved merged %s as CSV: %s", base_name, csv_path)
            subject_str = (
                features_dir.parts[-3].replace("sub-", "")
                if len(features_dir.parts) > 3
                else "unknown"
            )
            metadata_dir = save_path.parent / "metadata"
            ensure_dir(metadata_dir)
            meta_path = metadata_dir / filename.replace(".parquet", ".json")
            df_attrs = getattr(merged_df, "attrs", None) or {}
            manifest = generate_manifest(
                feature_columns=list(merged_df.columns),
                config=config,
                subject=subject_str,
                task=config.get("project.task") if config is not None else None,
                qc=None,
                df_attrs=dict(df_attrs),
            )
            meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            feature_name = filename.replace("features_", "").replace(".parquet", "")
            logger.info(
                "Saved merged %s features: %d columns",
                feature_name,
                int(merged_df.shape[1]),
            )


def _save_extraction_config(
    config: Dict[str, Any],
    features_dir: Path,
    suffix: Optional[str],
    logger: Any,
    feature_categories: Optional[List[str]] = None,
    pipeline_config: Optional[Any] = None,
) -> None:
    """Save extraction configuration to JSON file in each feature category's metadata folder."""
    from eeg_pipeline.utils.data.feature_io import _get_folder_for_feature
    
    config_name = f"extraction_config_{suffix}.json" if suffix else "extraction_config.json"
    categories = feature_categories or config.get("feature_categories", [])
    
    saved_to = []
    for category in categories:
        folder = _get_folder_for_feature(f"features_{category}", pipeline_config)
        if folder:
            category_metadata_dir = features_dir / folder / "metadata"
            ensure_dir(category_metadata_dir)
            save_path = category_metadata_dir / config_name
            save_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            saved_to.append(str(save_path))
    
    if saved_to:
        logger.info("Saved extraction config to %d feature category folders", len(saved_to))


def _collect_trial_table_feature_tables(
    *,
    direct_df: Optional[pd.DataFrame],
    conn_df_aligned: Optional[pd.DataFrame],
    aper_df_aligned: Optional[pd.DataFrame],
    unpacked: Dict[str, Any],
    features: FeatureExtractionResult,
) -> List[tuple[str, Optional[pd.DataFrame]]]:
    """Collect trialwise feature tables for canonical trial-table export."""
    itpc_trial_or_standard = unpacked.get("itpc_trial_df")
    if itpc_trial_or_standard is None or getattr(itpc_trial_or_standard, "empty", False):
        itpc_trial_or_standard = unpacked.get("itpc_df")

    pac_trial_or_standard = unpacked.get("pac_trials_df")
    if pac_trial_or_standard is None or getattr(pac_trial_or_standard, "empty", False):
        pac_trial_or_standard = unpacked.get("pac_df")

    return [
        ("power", direct_df),
        ("connectivity", conn_df_aligned),
        ("aperiodic", aper_df_aligned),
        ("directedconnectivity", unpacked.get("dconn_df")),
        ("sourcelocalization", unpacked.get("source_df")),
        ("erp", unpacked.get("erp_df")),
        ("itpc", itpc_trial_or_standard),
        ("pac", pac_trial_or_standard),
        ("complexity", unpacked.get("comp_df")),
        ("bursts", unpacked.get("bursts_df")),
        ("quality", getattr(features, "quality_df", None)),
        ("erds", unpacked.get("erds_df")),
        ("spectral", unpacked.get("spectral_df")),
        ("ratios", getattr(features, "ratios_df", None)),
        ("asymmetry", getattr(features, "asymmetry_df", None)),
        ("microstates", unpacked.get("microstates_df")),
        ("temporal", getattr(features, "temporal_df", None)),
    ]


def _save_canonical_trial_table_artifact(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
    aligned_events: pd.DataFrame,
    feature_tables: List[tuple[str, Optional[pd.DataFrame]]],
    config: Any,
    logger: Any,
    suffix: Optional[str] = None,
) -> Optional[Path]:
    """Save canonical `stats/trial_table/all/trials_all*.parquet` from extracted features."""
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.data.trial_table import (
        build_trial_table_contract,
        combine_feature_tables,
        save_trial_table,
    )

    if aligned_events is None or aligned_events.empty:
        logger.warning("Skipping canonical trial table export: aligned events are unavailable.")
        return None

    events = aligned_events.reset_index(drop=True)
    n_trials = len(events)
    trialwise_tables: List[tuple[str, Optional[pd.DataFrame]]] = []
    for name, df in feature_tables:
        if df is None or getattr(df, "empty", False):
            continue
        if len(df) != n_trials:
            logger.info(
                "Skipping non-trialwise block for canonical trial table export: %s (%d vs %d rows)",
                name,
                len(df),
                n_trials,
            )
            continue
        trialwise_tables.append((name, df.reset_index(drop=True)))

    combined_features = combine_feature_tables(trialwise_tables)
    if combined_features is not None and not combined_features.empty:
        df_trials = pd.concat([events, combined_features.reset_index(drop=True)], axis=1)
    else:
        df_trials = events.copy()

    metadata: Dict[str, Any] = {
        "subject": subject,
        "task": task,
        "n_trials": int(len(df_trials)),
        "n_columns": int(df_trials.shape[1]),
        "producer": "feature_extraction",
        "contract": build_trial_table_contract(
            df_trials,
            n_events_rows=int(len(events)),
            n_feature_rows=int(len(combined_features)) if combined_features is not None else 0,
            feature_signature=None,
        ),
    }

    stats_dir = deriv_stats_path(deriv_root, subject)
    out_dir = stats_dir / "trial_table" / "all"
    ensure_dir(out_dir)
    stem = "trials_all"
    if suffix:
        stem = f"{stem}_{suffix}"
    out_path = out_dir / f"{stem}.parquet"

    class _TableWrapper:
        def __init__(self, df, metadata):
            self.df = df
            self.metadata = metadata

    save_trial_table(_TableWrapper(df_trials, metadata), out_path, format="parquet")

    also_save_csv = bool(get_config_value(config, "feature_engineering.output.also_save_csv", False))
    if also_save_csv:
        from eeg_pipeline.infra.tsv import write_csv

        csv_path = out_dir / f"{stem}.csv"
        write_csv(df_trials, csv_path, index=False)

    meta_path = out_dir / f"{stem}.metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved canonical trial table artifact: %s", out_path)
    return out_path


class FeaturePipeline(PipelineBase):
    """Pipeline for EEG feature extraction."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="feature_extraction", config=config)

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        from eeg_pipeline.cli.common import ProgressReporter

        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")

        feature_categories = resolve_feature_categories(
            self.config, kwargs.get("feature_categories")
        )
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)

        self.logger.info("=== Feature extraction: sub-%s, task-%s ===", subject, task)
        self.logger.info(
            "Categories (%d): %s", len(feature_categories), ", ".join(feature_categories)
        )
        progress.subject_start(f"sub-{subject}")

        features_dir = deriv_features_path(self.deriv_root, subject)
        ensure_dir(features_dir)
        setup_matplotlib(self.config)

        explicit_windows = kwargs.get("time_ranges")
        time_ranges = _resolve_time_ranges(explicit_windows, kwargs.get("tmin"), kwargs.get("tmax"))
        total_steps = _calculate_total_steps(len(time_ranges))
        current_step = 0
        if len(time_ranges) > 1:
            self.logger.info(
                "Time ranges (%d): %s",
                len(time_ranges),
                ", ".join(tr.get("name", "unnamed") or "unnamed" for tr in time_ranges),
            )

        current_step += 1
        progress.step("Loading epochs", current=current_step, total=total_steps)
        self.logger.info("Loading cleaned epochs...")
        epochs, aligned_events = load_epochs_for_analysis(
            subject,
            task,
            align="strict",
            preload=True,
            deriv_root=self.deriv_root,
            logger=self.logger,
            config=self.config,
        )

        if epochs is None:
            self.logger.error("No cleaned epochs for sub-%s; skipping", subject)
            progress.error("no_epochs", f"No cleaned epochs for sub-{subject}")
            return

        if aligned_events is None:
            self.logger.warning("No events available; skipping")
            progress.error("no_events", "No aligned events")
            return

        try:
            n_trials = len(epochs)
        except Exception:
            data_obj = getattr(epochs, "data", None)
            n_trials = int(data_obj.shape[0]) if hasattr(data_obj, "shape") and len(data_obj.shape) > 0 else 0
        n_channels = int(len(getattr(epochs, "ch_names", []) or []))
        sfreq = float(getattr(epochs, "info", {}).get("sfreq", np.nan))
        times = np.asarray(getattr(epochs, "times", np.array([], dtype=float)), dtype=float)
        t_start = float(times[0]) if times.size else np.nan
        t_end = float(times[-1]) if times.size else np.nan
        self.logger.info(
            "Epochs loaded: %d trials, %d channels, %.0f Hz, %.3f\u2013%.3fs",
            int(n_trials), int(n_channels), sfreq, t_start, t_end,
        )

        original_events = _load_events_df(
            subject,
            task,
            bids_root=self.config.bids_root,
            config=self.config,
            prefer_clean=False,
        )
        if original_events is not None:
            n_dropped = len(original_events) - int(n_trials)
            save_dropped_trials_log(
                epochs, original_events, features_dir / "metadata" / "dropped_trials.tsv", self.logger
            )
            if n_dropped > 0:
                self.logger.info(
                    "Dropped trials: %d/%d (%.0f%% retained)",
                    n_dropped, len(original_events),
                    100 * int(n_trials) / len(original_events),
                )

        target_columns = list(self.config.get("event_columns.rating", []) or [])
        target_col = pick_target_column(aligned_events, target_columns=target_columns)
        if target_col is None:
            self.logger.warning("No target column found; skipping")
            return

        y = pd.to_numeric(aligned_events[target_col], errors="coerce")
        n_valid = int(y.notna().sum())
        self.logger.info(
            "Target column: '%s' (%d/%d valid values)", target_col, n_valid, len(y)
        )

        fixed_templates_path = kwargs.get("fixed_templates_path")
        fixed_template_labels = None
        loaded_templates = _load_fixed_templates(fixed_templates_path, self.logger)
        if isinstance(loaded_templates, tuple) and len(loaded_templates) >= 2:
            fixed_templates = loaded_templates[0]
            fixed_template_ch_names = loaded_templates[1]
            if len(loaded_templates) >= 3:
                fixed_template_labels = loaded_templates[2]
        else:
            fixed_templates, fixed_template_ch_names = None, None

        tfr_full = _precompute_tfr_if_needed(
            epochs, time_ranges, feature_categories, self.config, self.logger
        )

        tfr_complex_full = _precompute_complex_tfr_if_needed(
            epochs, time_ranges, feature_categories, self.config, self.logger
        )

        precomputed_full = _precompute_intermediates_if_needed(
            epochs,
            time_ranges,
            feature_categories,
            kwargs.get("bands"),
            self.config,
            self.logger,
        )
        if precomputed_full is not None:
            if len(aligned_events) == int(precomputed_full.data.shape[0]):
                precomputed_full.metadata = aligned_events.reset_index(drop=True).copy()
                if "condition" in aligned_events.columns:
                    precomputed_full.condition_labels = aligned_events["condition"].to_numpy()
                elif "trial_type" in aligned_events.columns:
                    precomputed_full.condition_labels = aligned_events["trial_type"].to_numpy()
            else:
                self.logger.warning(
                    "Precomputed intermediates: aligned_events length (%d) != n_epochs (%d); skipping metadata.",
                    len(aligned_events),
                    int(precomputed_full.data.shape[0]),
                )

        accumulated_features = _create_feature_accumulator()
        accumulated_y = None

        for tr_spec in time_ranges:
            name = tr_spec.get("name")
            tmin = tr_spec.get("tmin")
            tmax = tr_spec.get("tmax")

            if tmin is not None and tmax is not None and tmin > tmax:
                self.logger.warning(
                    "Time range '%s' has tmin (%s) > tmax (%s). Swapping values.",
                    name, tmin, tmax,
                )
                tmin, tmax = tmax, tmin

            suffix = name
            range_info = f"range '{name}'" if name else "default range"
            self.logger.info("--- Processing %s (%.3f to %.3fs) ---", range_info, tmin, tmax)

            spatial_modes = kwargs.get("spatial_modes") or self.config.get(
                "feature_engineering.spatial_modes", ["roi", "channels", "global"]
            )
            ctx = FeatureContext(
                subject=subject,
                task=task,
                config=self.config,
                deriv_root=self.deriv_root,
                logger=self.logger,
                epochs=epochs,
                aligned_events=aligned_events,
                fixed_templates=fixed_templates,
                fixed_template_ch_names=fixed_template_ch_names,
                fixed_template_labels=fixed_template_labels,
                feature_categories=feature_categories,
                bands=kwargs.get("bands"),
                spatial_modes=spatial_modes,
                tmin=tmin,
                tmax=tmax,
                name=name,
                aggregation_method=kwargs.get("aggregation_method", "mean"),
                tfr=tfr_full,
                tfr_complex=tfr_complex_full,
                precomputed=precomputed_full,
                explicit_windows=explicit_windows,
                train_mask=kwargs.get("train_mask"),
                analysis_mode=kwargs.get(
                    "analysis_mode",
                    self.config.get("feature_engineering.analysis_mode", "group_stats"),
                ),
            )
            ctx.progress = progress
            ctx.total_steps = total_steps
            ctx.current_step = current_step

            current_step += 1
            progress.step(
                f"Extracting features ({name or 'full'})",
                current=current_step,
                total=total_steps,
            )
            features = extract_all_features(ctx)

            unpacked = _unpack_feature_results(features)

            current_step += 1
            progress.step(
                f"Aligning features ({name or 'full'})",
                current=current_step,
                total=total_steps,
            )
            critical_features = ["target"]
            if "power" in ctx.feature_categories:
                critical_features.extend(["power", "baseline"])

            extra_blocks = _build_extra_blocks(unpacked, features)

            (
                pow_df_aligned,
                baseline_df_aligned,
                conn_df_aligned,
                aper_df_aligned,
                y_aligned,
                retention_stats,
            ) = align_feature_dataframes(
                unpacked["pow_df"],
                unpacked["baseline_df"],
                unpacked["conn_df"],
                unpacked["aper_df"],
                y,
                aligned_events,
                features_dir,
                self.logger,
                self.config,
                critical_features=critical_features,
                extra_blocks=extra_blocks,
                requested_categories=ctx.feature_categories,
            )

            if retention_stats is None:
                self.logger.error("Feature alignment failed for %s; skipping save", range_info)
                continue

            n_retained = retention_stats.get("n_retained", len(y_aligned) if y_aligned is not None else 0)
            n_original = retention_stats.get("n_original", len(y))
            if n_retained < n_original:
                self.logger.info(
                    "Alignment: %d/%d trials retained (%.0f%%)",
                    n_retained, n_original, 100 * n_retained / max(n_original, 1),
                )

            extra_blocks = retention_stats.get("extra_blocks", {})
            _update_from_aligned_extra(unpacked, features, extra_blocks)

            feature_qc = _build_feature_qc(features, ctx)

            current_step += 1
            progress.step(
                f"Saving features ({name or 'full'})",
                current=current_step,
                total=total_steps,
            )

            combined_df = save_all_features(
                pow_df=pow_df_aligned,
                pow_cols=unpacked["pow_cols"],
                baseline_df=baseline_df_aligned,
                baseline_cols=unpacked["baseline_cols"],
                conn_df=conn_df_aligned,
                conn_cols=unpacked["conn_cols"],
                aper_df=aper_df_aligned,
                aper_cols=unpacked["aper_cols"],
                erp_df=unpacked["erp_df"],
                erp_cols=unpacked["erp_cols"],
                itpc_df=unpacked["itpc_df"],
                itpc_cols=unpacked["itpc_cols"],
                pac_df=unpacked["pac_df"],
                pac_trials_df=unpacked["pac_trials_df"],
                pac_time_df=unpacked["pac_time_df"],
                aper_qc=features.aper_qc,
                y=y_aligned,
                features_dir=features_dir,
                logger=self.logger,
                config=self.config,
                comp_df=unpacked["comp_df"],
                comp_cols=unpacked["comp_cols"],
                bursts_df=unpacked["bursts_df"],
                bursts_cols=unpacked["bursts_cols"],
                spectral_df=unpacked["spectral_df"],
                spectral_cols=unpacked["spectral_cols"],
                erds_df=unpacked["erds_df"],
                erds_cols=unpacked["erds_cols"],
                ratios_df=features.ratios_df,
                ratios_cols=features.ratios_cols,
                asymmetry_df=features.asymmetry_df,
                asymmetry_cols=features.asymmetry_cols,
                microstates_df=unpacked.get("microstates_df"),
                microstates_cols=unpacked.get("microstates_cols"),
                quality_df=features.quality_df,
                quality_cols=features.quality_cols,
                dconn_df=unpacked["dconn_df"],
                dconn_cols=unpacked["dconn_cols"],
                source_df=unpacked["source_df"],
                source_cols=unpacked["source_cols"],
                feature_qc=feature_qc or None,
                suffix=suffix,
            )

            trial_table_suffix = suffix if len(time_ranges) > 1 else None
            trial_feature_tables = _collect_trial_table_feature_tables(
                direct_df=combined_df,
                conn_df_aligned=conn_df_aligned,
                aper_df_aligned=aper_df_aligned,
                unpacked=unpacked,
                features=features,
            )
            _save_canonical_trial_table_artifact(
                deriv_root=self.deriv_root,
                subject=subject,
                task=task,
                aligned_events=aligned_events,
                feature_tables=trial_feature_tables,
                config=self.config,
                logger=self.logger,
                suffix=trial_table_suffix,
            )

            if len(time_ranges) > 1:
                aligned_dict = {
                    "pow_df_aligned": pow_df_aligned,
                    "baseline_df_aligned": baseline_df_aligned,
                    "conn_df_aligned": conn_df_aligned,
                    "aper_df_aligned": aper_df_aligned,
                }
                _accumulate_features(accumulated_features, unpacked, features, aligned_dict)
                if accumulated_y is None and y_aligned is not None:
                    accumulated_y = y_aligned

            n_trials = len(y_aligned)
            n_total = _get_df_cols(combined_df)

            extraction_config = {
                "cli_command": kwargs.get("cli_command"),
                "name": name,
                "spatial_modes": ctx.spatial_modes,
                "analysis_mode": str(ctx.analysis_mode or "").strip() or "group_stats",
                "connectivity_granularity": str(
                    self.config.get("feature_engineering.connectivity.granularity", "trial")
                ).strip(),
                "aggregation_method": ctx.aggregation_method,
                "tmin": ctx.tmin,
                "tmax": ctx.tmax,
                "bands": ctx.bands,
                "feature_categories": feature_categories,
                "n_trials": n_trials,
                "subject": subject,
                "task": task,
            }
            _save_extraction_config(extraction_config, features_dir, suffix, self.logger, feature_categories, pipeline_config=self.config)

            self.logger.info(
                "Saved %s: %d total columns \u00d7 %d trials",
                range_info, n_total, n_trials,
            )

            del ctx, features, unpacked, extra_blocks, combined_df
            del pow_df_aligned, baseline_df_aligned, conn_df_aligned, aper_df_aligned
            gc.collect()

        if len(time_ranges) > 1:
            self.logger.info(
                "Merging features from all time ranges into consolidated default files..."
            )
            _save_merged_features(accumulated_features, features_dir, self.config, self.logger)


            merged_extraction_config = {
                "cli_command": kwargs.get("cli_command"),
                "merged": True,
                "time_ranges": [tr.get("name") for tr in time_ranges],
                "spatial_modes": kwargs.get("spatial_modes")
                or self.config.get("feature_engineering.spatial_modes", ["roi", "global"]),
                "analysis_mode": str(
                    kwargs.get(
                        "analysis_mode",
                        self.config.get("feature_engineering.analysis_mode", "group_stats"),
                    )
                    or "group_stats"
                ).strip(),
                "connectivity_granularity": str(
                    self.config.get("feature_engineering.connectivity.granularity", "trial")
                ).strip(),
                "aggregation_method": kwargs.get("aggregation_method", "mean"),
                "feature_categories": feature_categories,
                "n_trials": len(accumulated_y) if accumulated_y is not None else 0,
                "subject": subject,
                "task": task,
            }
            _save_extraction_config(
                merged_extraction_config, features_dir, None, self.logger, feature_categories,
                pipeline_config=self.config,
            )
            self.logger.info("Saved merged extraction config")

        progress.subject_done(f"sub-{subject}", success=True)


def process_subject(
    subject: str, task: Optional[str] = None, config: Optional[Any] = None, **kwargs: Any
) -> None:
    """Process a single subject through the feature extraction pipeline."""
    pipeline = FeaturePipeline(config=config)
    pipeline.process_subject(subject, task=task, **kwargs)


def extract_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Extract features for multiple subjects."""
    pipeline = FeaturePipeline(config=config)
    return pipeline.run_batch(subjects, task=task, **kwargs)


__all__ = [
    "FeaturePipeline",
    "process_subject",
    "extract_features_for_subjects",
    "extract_all_features",
    "extract_precomputed_features",
    "FeatureExtractionResult",
    "ExtractionResult",
    "FeatureSet",
]
