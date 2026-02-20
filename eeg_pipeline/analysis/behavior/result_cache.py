from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value


class BehaviorResultCache:
    """In-memory cache for expensive computations to avoid repeated disk I/O and processing."""

    def __init__(
        self,
        *,
        feature_column_prefixes: Sequence[str],
        ensure_dir_fn: Callable[[Path], None],
        trial_table_suffix_from_context_fn: Callable[[Any], str],
        trial_table_output_dir_fn: Callable[[Any, bool], Path],
        find_trial_table_path_fn: Callable[[Path, Optional[List[str]]], Optional[Path]],
        validate_trial_table_contract_metadata_fn: Callable[[Any, Path, pd.DataFrame], None],
        filter_feature_cols_by_band_fn: Callable[[List[str], Any], List[str]],
        filter_feature_cols_for_computation_fn: Callable[[List[str], str, Any], List[str]],
        filter_feature_cols_by_provenance_fn: Callable[[List[str], Any, Optional[str]], List[str]],
        infer_feature_type_fn: Callable[[str, Any], str],
        infer_feature_band_fn: Callable[[str, Any], str],
    ):
        self._feature_column_prefixes = tuple(feature_column_prefixes)
        self._ensure_dir_fn = ensure_dir_fn
        self._trial_table_suffix_from_context_fn = trial_table_suffix_from_context_fn
        self._trial_table_output_dir_fn = trial_table_output_dir_fn
        self._find_trial_table_path_fn = find_trial_table_path_fn
        self._validate_trial_table_contract_metadata_fn = validate_trial_table_contract_metadata_fn
        self._filter_feature_cols_by_band_fn = filter_feature_cols_by_band_fn
        self._filter_feature_cols_for_computation_fn = filter_feature_cols_for_computation_fn
        self._filter_feature_cols_by_provenance_fn = filter_feature_cols_by_provenance_fn
        self._infer_feature_type_fn = infer_feature_type_fn
        self._infer_feature_band_fn = infer_feature_band_fn

        self._trial_table_df: Optional[pd.DataFrame] = None
        self._trial_table_path: Optional[Path] = None
        self._feature_cols: Dict[str, List[str]] = {}
        self._filtered_feature_cols: Dict[Tuple[str, ...], List[str]] = {}
        self._discovered_files: Dict[str, List[Path]] = {}
        self._fdr_results: Optional[Dict[str, Any]] = None
        self._feature_types: Dict[str, str] = {}
        self._feature_bands: Dict[str, str] = {}
        self._manifest: Optional[Dict[str, Any]] = None
        self._manifest_loaded: bool = False
        self._stats_subfolders: Dict[Tuple[str, str, bool], Path] = {}

    def get_stats_subfolder(
        self,
        stats_dir: Path,
        kind: str,
        overwrite: bool,
        *,
        ensure: bool,
    ) -> Path:
        """Return stable stats subfolder for this run (per kind)."""
        key = (str(stats_dir.resolve()), str(kind), bool(overwrite))
        if overwrite:
            path = stats_dir / kind
        else:
            if key in self._stats_subfolders:
                path = self._stats_subfolders[key]
            else:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = stats_dir / f"{kind}_{timestamp}"
                self._stats_subfolders[key] = path
        if ensure:
            self._ensure_dir_fn(path)
        return path

    def get_trial_table(self, ctx: Any) -> Optional[pd.DataFrame]:
        """Get cached trial table or load from disk."""
        if self._trial_table_df is not None:
            return self._trial_table_df

        suffix = self._trial_table_suffix_from_context_fn(ctx)
        fname = f"trials{suffix}"
        expected_dir = self._trial_table_output_dir_fn(ctx, False)
        fmt = str(get_config_value(ctx.config, "behavior_analysis.trial_table.format", "tsv")).strip().lower()
        preferred_ext = ".parquet" if fmt == "parquet" else ".tsv"
        trial_table_path = expected_dir / f"{fname}{preferred_ext}"
        if not trial_table_path.exists():
            alt_ext = ".tsv" if preferred_ext == ".parquet" else ".parquet"
            alt_path = expected_dir / f"{fname}{alt_ext}"
            if alt_path.exists():
                trial_table_path = alt_path

        if not trial_table_path.exists():
            feature_files = ctx.selected_feature_files or ctx.feature_categories or None
            found = self._find_trial_table_path_fn(ctx.stats_dir, feature_files)
            if found is None:
                return None
            trial_table_path = found

        from eeg_pipeline.infra.tsv import read_table

        self._trial_table_df = read_table(trial_table_path)
        self._validate_trial_table_contract_metadata_fn(ctx, trial_table_path, self._trial_table_df)
        self._trial_table_path = trial_table_path
        return self._trial_table_df

    def get_feature_cols(self, df: pd.DataFrame, ctx: Any) -> List[str]:
        """Get cached feature columns or compute from DataFrame."""
        _ = ctx
        cache_key = id(df)
        if cache_key not in self._feature_cols:
            self._feature_cols[cache_key] = [c for c in df.columns if str(c).startswith(self._feature_column_prefixes)]
        return self._feature_cols[cache_key]

    def get_filtered_feature_cols(
        self,
        feature_cols: List[str],
        ctx: Any,
        computation_name: Optional[str] = None,
    ) -> List[str]:
        """Get cached filtered feature columns."""
        bands_key = tuple(sorted(ctx.selected_bands)) if ctx.selected_bands else None
        cache_key = (id(feature_cols), bands_key, computation_name)

        if cache_key not in self._filtered_feature_cols:
            filtered = feature_cols.copy()
            if ctx.selected_bands:
                filtered = self._filter_feature_cols_by_band_fn(filtered, ctx)
            if computation_name:
                filtered = self._filter_feature_cols_for_computation_fn(filtered, computation_name, ctx)
            filtered = self._filter_feature_cols_by_provenance_fn(filtered, ctx, computation_name)
            self._filtered_feature_cols[cache_key] = filtered

        return self._filtered_feature_cols[cache_key]

    def get_discovered_files(self, ctx: Any, patterns: List[str]) -> List[Path]:
        """Get cached discovered files matching patterns."""
        cache_key = "_".join(sorted(patterns))
        if cache_key not in self._discovered_files:
            files: List[Path] = []
            for pat in patterns:
                files.extend(sorted(ctx.stats_dir.rglob(pat)))
            self._discovered_files[cache_key] = sorted({p.resolve() for p in files if p.exists()})
        return self._discovered_files[cache_key]

    def set_fdr_results(self, results: Dict[str, Any]) -> None:
        self._fdr_results = results

    def get_fdr_results(self) -> Optional[Dict[str, Any]]:
        return self._fdr_results

    def load_manifest(self, ctx: Any) -> Optional[Dict[str, Any]]:
        """Load feature manifest from disk if available."""
        if self._manifest_loaded:
            return self._manifest

        self._manifest_loaded = True
        from eeg_pipeline.infra.paths import deriv_features_path

        features_dir = deriv_features_path(ctx.deriv_root, ctx.subject)
        manifest_patterns = ["*_manifest.json", "*_features_manifest.json"]

        for pattern in manifest_patterns:
            for manifest_path in features_dir.glob(pattern):
                if manifest_path.exists():
                    text = manifest_path.read_text(encoding="utf-8")
                    try:
                        self._manifest = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid feature manifest JSON: {manifest_path}") from exc

                    self._populate_from_manifest()
                    ctx.logger.info("Loaded feature manifest: %s", manifest_path.name)
                    return self._manifest

        return None

    def _populate_from_manifest(self) -> None:
        if self._manifest is None:
            return

        features = self._manifest.get("features", [])
        for entry in features:
            name = entry.get("name")
            if not name:
                continue
            group = entry.get("group", "unknown")
            band = entry.get("band", "broadband")
            self._feature_types[name] = group
            self._feature_bands[name] = band if band and band != "unknown" else "broadband"

    def get_feature_type(self, feature: str, config: Any) -> str:
        if feature in self._feature_types:
            return self._feature_types[feature]
        self._feature_types[feature] = self._infer_feature_type_fn(feature, config)
        return self._feature_types[feature]

    def get_feature_band(self, feature: str, config: Any) -> str:
        if feature in self._feature_bands:
            return self._feature_bands[feature]
        self._feature_bands[feature] = self._infer_feature_band_fn(feature, config)
        return self._feature_bands[feature]

    def clear_feature_types(self) -> None:
        self._feature_types.clear()
        self._feature_bands.clear()

    def clear(self) -> None:
        self._trial_table_df = None
        self._trial_table_path = None
        self._feature_cols.clear()
        self._filtered_feature_cols.clear()
        self._discovered_files.clear()
        self._fdr_results = None
        self._feature_types.clear()
        self._feature_bands.clear()
        self._manifest = None
        self._manifest_loaded = False
        self._stats_subfolders.clear()
