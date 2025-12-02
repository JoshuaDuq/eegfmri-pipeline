"""
Unified Feature-Behavior Visualization
======================================

Orchestrates organized visualizations showing how EEG features relate to behavior.
Uses existing plotting utilities - no duplicate implementations.

Output Organization:
    plots/behavior/
    ├── scatter/           - Feature vs behavior scatter plots
    ├── topomaps/          - Spatial correlation topomaps
    ├── heatmaps/          - Correlation summary heatmaps
    ├── forest/            - Effect size forest plots
    ├── distributions/     - Feature distributions by condition
    └── summary/           - Dashboard figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.utils.io.general import (
    deriv_stats_path, deriv_features_path, deriv_plots_path,
    ensure_dir, read_tsv, get_band_color,
)

# Feature type color scheme
FEATURE_TYPE_COLORS = {
    "power": "#3B82F6",
    "erds": "#10B981",
    "connectivity": "#8B5CF6",
    "aperiodic": "#F59E0B",
    "microstate": "#EC4899",
    "itpc": "#06B6D4",
    "pac": "#EF4444",
    "complexity": "#6366F1",
    "spectral": "#14B8A6",
    "temporal": "#F97316",
    "unknown": "#6B7280",
}

PLOT_SUBDIRS = ["scatter", "topomaps", "heatmaps", "forest", "distributions", "summary"]


@dataclass
class PlotContext:
    """Context for plotting operations."""
    subject: str
    plots_dir: Path
    stats_dir: Path
    features_dir: Path
    config: Any
    logger: logging.Logger
    
    def subdir(self, name: str) -> Path:
        """Get or create subdirectory."""
        path = self.plots_dir / name
        ensure_dir(path)
        return path


def setup_behavior_plot_dirs(plots_dir: Path) -> Dict[str, Path]:
    """Create organized subdirectory structure."""
    dirs = {}
    for subdir in PLOT_SUBDIRS:
        path = plots_dir / subdir
        ensure_dir(path)
        dirs[subdir] = path
    return dirs


def _load_stats_file(stats_dir: Path, patterns: List[str]) -> Optional[pd.DataFrame]:
    """Load first matching stats file."""
    for pattern in patterns:
        files = list(stats_dir.glob(pattern))
        if files:
            try:
                return read_tsv(files[0])
            except Exception:
                continue
    return None


def _load_feature_file(features_dir: Path, patterns: List[str]) -> Optional[pd.DataFrame]:
    """Load first matching feature file."""
    for pattern in patterns:
        path = features_dir / pattern
        if path.exists():
            try:
                return read_tsv(path)
            except Exception:
                continue
    return None


def _load_all_feature_files(features_dir: Path, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    """Load and combine all available feature files.
    
    Ensures all dataframes have aligned indices before combining.
    
    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]
        Combined feature dataframe and mapping of column -> source_file_type
    """
    from eeg_pipeline.analysis.behavior.feature_correlator import FEATURE_FILES
    
    all_dfs = []
    loaded_files = []
    column_to_source: Dict[str, str] = {}  # Track which file each column came from
    
    for name, filename in FEATURE_FILES.items():
        path = features_dir / filename
        if path.exists():
            try:
                df = read_tsv(path)
                if df is not None and not df.empty:
                    # Track source file for each column
                    for col in df.columns:
                        column_to_source[col] = name
                    
                    all_dfs.append(df)
                    loaded_files.append(name)
                    logger.debug(f"Loaded {name}: {len(df.columns)} features, {len(df)} rows, index: {df.index[:3].tolist() if len(df) > 0 else 'empty'}")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
                continue
    
    if not all_dfs:
        logger.warning("No feature files found")
        return None, None
    
    # Align all dataframes to common index
    if len(all_dfs) > 1:
        # Find common index across all dataframes
        common_idx = all_dfs[0].index
        for df in all_dfs[1:]:
            common_idx = common_idx.intersection(df.index)
        
        if len(common_idx) > 0:
            # Align all dataframes to common index
            for i, df in enumerate(all_dfs):
                all_dfs[i] = df.loc[common_idx]
            logger.info(f"Aligned {len(all_dfs)} feature files to {len(common_idx)} common indices")
        else:
            # If no common index, try to align by position (assume same order)
            min_len = min(len(df) for df in all_dfs)
            if min_len > 0:
                for i, df in enumerate(all_dfs):
                    all_dfs[i] = df.iloc[:min_len]
                logger.info(f"Aligned {len(all_dfs)} feature files by position: {min_len} rows")
            else:
                logger.warning("No common indices or valid length for feature file alignment")
                return None, None
    
    # Combine all dataframes
    try:
        combined_df = pd.concat(all_dfs, axis=1)
        logger.info(f"Combined {len(loaded_files)} feature files: {len(combined_df.columns)} total features, {len(combined_df)} rows")
        return combined_df, column_to_source
    except Exception as e:
        logger.error(f"Failed to combine feature files: {e}", exc_info=True)
        return None, None


def _normalize_stats_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize column names for forest plot compatibility.
    
    Maps common column names to the expected format:
    - channel/feature/roi -> identifier
    - r -> correlation
    
    Returns None if required columns cannot be found.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    
    df = df.copy()
    
    # Map identifier column - check if already exists first
    if "identifier" not in df.columns:
        for col in ["channel", "feature", "roi", "name", "region", "precomputed"]:
            if col in df.columns:
                df["identifier"] = df[col]
                break
        else:
            return None  # Cannot proceed without identifier
    
    # Map correlation column
    if "correlation" not in df.columns:
        if "r" in df.columns:
            df["correlation"] = df["r"]
        elif "corr" in df.columns:
            df["correlation"] = df["corr"]
        else:
            return None  # Cannot proceed without correlation
    
    # Map p-value columns (optional)
    if "p" in df.columns and "p_value" not in df.columns:
        df["p_value"] = df["p"]
    if "q" in df.columns and "p_fdr" not in df.columns:
        df["p_fdr"] = df["q"]
    if "band" not in df.columns:
        df["band"] = "N/A"
    
    return df


def _has_forest_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame has required columns for forest plot."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    required = {"identifier", "correlation"}
    return required.issubset(df.columns)


# =============================================================================
# Main Orchestration
# =============================================================================


def visualize_feature_behavior_correlations(
    subject: str,
    deriv_root: Path,
    config: Any,
    logger: logging.Logger,
    *,
    targets: Optional[pd.Series] = None,
    temperature: Optional[pd.Series] = None,
    pain_condition: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    """Create organized feature-behavior visualizations.
    
    This function orchestrates creation of:
    - Forest plots of top correlations (using effect_sizes module)
    - Scatter plots by band (using builders module)
    - Condition comparisons (using distributions module)
    - Summary heatmaps
    
    Parameters
    ----------
    subject : str
        Subject ID
    deriv_root : Path
        Derivatives root directory
    config : Any
        Configuration object
    logger : Logger
        Logger instance
        
    Returns
    -------
    Dict[str, Path]
        Mapping of plot types to saved file paths
    """
    # Import existing plot functions (avoid circular imports at module level)
    from .effect_sizes import (
        plot_correlation_forest, plot_condition_effect_sizes, plot_temperature_mediation
    )
    from .distributions import plot_feature_by_condition, plot_feature_correlation_matrix
    from .summary import plot_analysis_dashboard
    
    logger.info(f"Creating feature-behavior visualizations for sub-{subject}...")
    
    # Setup paths
    plots_dir = deriv_plots_path(deriv_root, subject, subdir="behavior")
    stats_dir = deriv_stats_path(deriv_root, subject)
    features_dir = deriv_features_path(deriv_root, subject)
    
    dirs = setup_behavior_plot_dirs(plots_dir)
    saved_files: Dict[str, Path] = {}
    
    ctx = PlotContext(
        subject=subject,
        plots_dir=plots_dir,
        stats_dir=stats_dir,
        features_dir=features_dir,
        config=config,
        logger=logger,
    )
    
    # Load data
    rating_stats_raw = _load_stats_file(stats_dir, [
        "corr_stats_all_features_vs_rating.tsv",
        "corr_stats_precomputed_vs_rating*.tsv",
        "corr_stats_pow_roi_vs_rating.tsv",
        "corr_stats_pow_combined_vs_rating.tsv",
        "corr_stats_all_pain_spearman.tsv",  # fallback
    ])
    rating_stats = None
    if rating_stats_raw is not None:
        rating_stats = _normalize_stats_columns(rating_stats_raw)
        if rating_stats is None:
            logger.warning("Could not normalize rating stats columns")
    
    temp_stats_raw = _load_stats_file(stats_dir, [
        "corr_stats_all_features_vs_temperature.tsv",
        "corr_stats_precomputed_vs_temp*.tsv",
        "corr_stats_pow_roi_vs_temp.tsv",
        "corr_stats_all_vs_temp_pain_spearman.tsv",  # fallback
        "corr_stats_all_vs_temp_nonpain_spearman.tsv",
    ])
    temp_stats = None
    if temp_stats_raw is not None:
        temp_stats = _normalize_stats_columns(temp_stats_raw)
        if temp_stats is None:
            logger.warning("Could not normalize temperature stats columns")
    
    # Load targets and temperature if not provided
    if targets is None or temperature is None:
        from eeg_pipeline.utils.data.loading import _load_features_and_targets, load_epochs_for_analysis
        from eeg_pipeline.utils.io.general import _pick_first_column
        
        task = config.get("project.task", "thermalactive")
        _, _, _, y_loaded, _ = _load_features_and_targets(subject, task, deriv_root, config)
        
        if targets is None and y_loaded is not None:
            targets = pd.to_numeric(y_loaded, errors="coerce")
            logger.info(f"Loaded targets: {len(targets)} values")
        
        if temperature is None:
            epochs, aligned_events = load_epochs_for_analysis(
                subject, task, align="strict", preload=False,
                deriv_root=deriv_root, bids_root=config.bids_root, config=config, logger=logger
            )
            if aligned_events is not None:
                temp_cols = config.get("event_columns.temperature", [])
                temp_col = _pick_first_column(aligned_events, temp_cols)
                if temp_col:
                    temperature = pd.to_numeric(aligned_events[temp_col], errors="coerce")
                    logger.info(f"Loaded temperature: {len(temperature)} values")
    
    # Load ALL available feature files (not just 2)
    feature_df, column_to_source = _load_all_feature_files(features_dir, logger)
    
    # Align features with targets/temperature indices
    if feature_df is not None and not feature_df.empty:
        # Get reference index from targets if available, otherwise use feature_df index
        ref_index = None
        if targets is not None and len(targets) > 0:
            ref_index = targets.index
        elif temperature is not None and len(temperature) > 0:
            ref_index = temperature.index
        
        if ref_index is not None:
            # Align feature_df to reference index
            common_idx = feature_df.index.intersection(ref_index)
            if len(common_idx) > 0:
                feature_df = feature_df.loc[common_idx]
                if targets is not None:
                    targets = targets.loc[common_idx]
                if temperature is not None:
                    temperature = temperature.loc[common_idx]
                logger.info(f"Aligned features and targets: {len(common_idx)} common indices")
            else:
                logger.warning(f"No common indices between features ({len(feature_df)}) and targets/temperature")
                # Try to align by position if indices don't match
                min_len = min(len(feature_df), len(targets) if targets is not None else len(temperature) if temperature is not None else len(feature_df))
                if min_len > 0:
                    feature_df = feature_df.iloc[:min_len]
                    if targets is not None:
                        targets = targets.iloc[:min_len]
                    if temperature is not None:
                        temperature = temperature.iloc[:min_len]
                    logger.info(f"Aligned by position: {min_len} rows")
        
        # Convert numeric columns, keeping non-numeric as-is
        for col in feature_df.columns:
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
            except (ValueError, TypeError):
                # Keep non-numeric columns as-is
                pass
    
    # ═══════════════════════════════════════════════════════════════
    # 1. Forest Plots by Feature Type (using effect_sizes module)
    # ═══════════════════════════════════════════════════════════════
    
    def _create_forest_plots_by_type(stats_df: pd.DataFrame, target_name: str, saved_files: Dict[str, Path]):
        """Create separate forest plots for each feature type."""
        if stats_df is None or not _has_forest_columns(stats_df):
            return
        
        from eeg_pipeline.analysis.behavior.feature_correlator import classify_feature
        
        # Classify features by type
        # For stats files, we don't have source file info, so use column name matching
        feature_types: Dict[str, List[int]] = {}
        for idx, row in stats_df.iterrows():
            # Get feature identifier
            identifier = None
            for col in ["identifier", "channel", "feature", "roi", "name"]:
                if col in row.index and pd.notna(row[col]):
                    identifier = str(row[col])
                    break
            
            if identifier is None:
                continue
            
            # Try to match identifier to a column in feature_df to get source
            source_type = None
            if feature_df is not None and identifier in feature_df.columns:
                source_type = column_to_source.get(identifier) if column_to_source else None
            
            # Classify feature type and subtype (with source info if available)
            ftype, subtype, _ = classify_feature(identifier, source_file_type=source_type, include_subtype=True)
            key = (ftype, subtype)
            if key not in feature_types:
                feature_types[key] = []
            feature_types[key].append(idx)
        
        logger.info(f"Creating forest plots by feature type/subtype for {target_name}...")
        logger.info(f"Found {len(feature_types)} feature groups: {list(feature_types.keys())}")
        
        # Create plot for each feature type/subtype combination
        for (ftype, subtype), indices in feature_types.items():
            if len(indices) == 0:
                continue
            
            type_stats = stats_df.iloc[indices].copy()
            if type_stats.empty:
                continue
            
            try:
                # Create subdirectory structure: type/subtype
                if subtype and subtype != "unknown":
                    type_dir = ctx.subdir("forest") / ftype / subtype
                else:
                    type_dir = ctx.subdir("forest") / ftype
                ensure_dir(type_dir)
                
                # Create filename with type and subtype
                if subtype and subtype != "unknown":
                    filename = f"sub-{subject}_{target_name}_correlations_{ftype}_{subtype}.png"
                    title = f"{ftype.title()}-{subtype.title()}-{target_name.title()} Correlations"
                else:
                    filename = f"sub-{subject}_{target_name}_correlations_{ftype}.png"
                    title = f"{ftype.title()}-{target_name.title()} Correlations"
                
                path = type_dir / filename
                max_feat = min(50, len(type_stats)) if len(type_stats) > 30 else len(type_stats)
                
                plot_correlation_forest(type_stats, path, title=title, 
                                      max_features=max_feat, config=config)
                saved_files[f"forest_{target_name}_{ftype}_{subtype}"] = path
                logger.info(f"Created {ftype}/{subtype} forest plot: {len(type_stats)} features")
            except Exception as e:
                logger.warning(f"Forest plot failed for {ftype}/{subtype}: {e}", exc_info=True)
    
    if rating_stats is not None:
        _create_forest_plots_by_type(rating_stats, "rating", saved_files)
    
    if temp_stats is not None:
        _create_forest_plots_by_type(temp_stats, "temperature", saved_files)
    
    # ═══════════════════════════════════════════════════════════════
    # 2. Feature Correlation Matrix by Type (using distributions module)
    # ═══════════════════════════════════════════════════════════════
    
    def _create_correlation_matrices_by_type(feature_df: pd.DataFrame, saved_files: Dict[str, Path]):
        """Create separate correlation matrices for each feature type."""
        if feature_df is None or feature_df.empty:
            return
        
        from eeg_pipeline.analysis.behavior.feature_correlator import classify_feature
        
        # Group features by type and subtype (using source file information if available)
        feature_groups: Dict[Tuple[str, str], List[str]] = {}  # (type, subtype) -> columns
        for col in feature_df.columns:
            source_type = column_to_source.get(col) if column_to_source else None
            ftype, subtype, _ = classify_feature(col, source_file_type=source_type, include_subtype=True)
            key = (ftype, subtype)
            if key not in feature_groups:
                feature_groups[key] = []
            feature_groups[key].append(col)
        
        logger.info(f"Creating correlation matrices by feature type/subtype...")
        logger.info(f"Found {len(feature_groups)} feature groups: {list(feature_groups.keys())}")
        
        # Create correlation matrix for each feature type/subtype combination
        for (ftype, subtype), cols in feature_groups.items():
            if len(cols) < 2:
                continue
            
            try:
                type_df = feature_df[cols].copy()
                
                # Ensure we have valid numeric data
                numeric_cols = type_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    logger.debug(f"Skipping {ftype}/{subtype}: insufficient numeric columns ({len(numeric_cols)})")
                    continue
                
                # Limit to top features by variance (only if we have many)
                if len(numeric_cols) > 50:
                    # Compute variance with proper NaN handling
                    variances = {}
                    for col in numeric_cols:
                        col_data = type_df[col].dropna()
                        if len(col_data) > 2:  # Need at least 3 values for variance
                            var_val = col_data.var()
                            if pd.notna(var_val) and var_val > 1e-10:
                                variances[col] = var_val
                    
                    if len(variances) >= 2:
                        top_cols = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:50]
                        type_df = type_df[[c[0] for c in top_cols]]
                    else:
                        logger.debug(f"Skipping {ftype}/{subtype}: insufficient valid variance data")
                        continue
                
                # Create subdirectory structure: type/subtype
                if subtype and subtype != "unknown":
                    type_dir = ctx.subdir("heatmaps") / ftype / subtype
                else:
                    type_dir = ctx.subdir("heatmaps") / ftype
                ensure_dir(type_dir)
                
                # Create filename with type and subtype
                if subtype and subtype != "unknown":
                    filename = f"sub-{subject}_feature_correlations_{ftype}_{subtype}.png"
                else:
                    filename = f"sub-{subject}_feature_correlations_{ftype}.png"
                
                path = type_dir / filename
                plot_feature_correlation_matrix(type_df, path, max_features=50, config=config)
                saved_files[f"correlation_matrix_{ftype}_{subtype}"] = path
                logger.info(f"Created {ftype}/{subtype} correlation matrix: {len(type_df.columns)} features")
            except Exception as e:
                logger.warning(f"Correlation matrix failed for {ftype}/{subtype}: {e}", exc_info=True)
    
    if feature_df is not None and not feature_df.empty:
        _create_correlation_matrices_by_type(feature_df, saved_files)
    
    # ═══════════════════════════════════════════════════════════════
    # 3. Condition Comparison by Feature Type (pain vs non-pain)
    # ═══════════════════════════════════════════════════════════════
    
    def _create_condition_comparisons_by_type(feature_df: pd.DataFrame, pain_condition: np.ndarray, 
                                            saved_files: Dict[str, Path], comparison_type: str = "condition"):
        """Create separate condition comparison plots for each feature type."""
        if feature_df is None or feature_df.empty or pain_condition is None:
            return
        
        # Align pain_condition with feature_df index
        if len(pain_condition) != len(feature_df):
            min_len = min(len(pain_condition), len(feature_df))
            if min_len > 0:
                feature_df = feature_df.iloc[:min_len]
                pain_condition = pain_condition[:min_len]
                logger.info(f"Aligned pain_condition with features: {min_len} rows")
            else:
                logger.warning("Cannot align pain_condition with features: no valid length")
                return
        
        from eeg_pipeline.analysis.behavior.feature_correlator import classify_feature
        
        # Group features by type and subtype (using source file information)
        feature_groups: Dict[Tuple[str, str], List[str]] = {}  # (type, subtype) -> columns
        for col in feature_df.columns:
            source_type = column_to_source.get(col) if column_to_source else None
            ftype, subtype, _ = classify_feature(col, source_file_type=source_type, include_subtype=True)
            key = (ftype, subtype)
            if key not in feature_groups:
                feature_groups[key] = []
            feature_groups[key].append(col)
        
        logger.info(f"Creating {comparison_type} comparison plots by feature type/subtype...")
        
        # Create condition comparison for each feature type/subtype combination
        for (ftype, subtype), cols in feature_groups.items():
            if len(cols) == 0:
                continue
            
            try:
                type_df = feature_df[cols].copy()
                
                # Ensure we have valid numeric data
                numeric_cols = type_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    logger.debug(f"Skipping {ftype}/{subtype} condition comparison: no numeric columns")
                    continue
                
                # Get top features for this type
                top_features = []
                if rating_stats is not None and "r" in rating_stats.columns:
                    id_col = next((c for c in ["feature", "identifier", "channel", "roi", "name"] 
                                  if c in rating_stats.columns), None)
                    if id_col:
                        # Try to match stats identifiers to feature columns
                        # First, try exact matches
                        stats_ids = rating_stats[id_col].astype(str).str.lower()
                        for col in cols:
                            col_lower = str(col).lower()
                            if col_lower in stats_ids.values:
                                top_features.append(col)
                        
                        # If not enough exact matches, try partial matching
                        if len(top_features) < 10:
                            for col in cols[:50]:  # Limit search to avoid performance issues
                                col_lower = str(col).lower()
                                # Check if any part of the column name matches a stats identifier
                                matching_stats = stats_ids[stats_ids.str.contains(col_lower, na=False, regex=False)]
                                if not matching_stats.empty:
                                    # Get correlation values for matching stats
                                    matching_indices = matching_stats.index
                                    matching_r = rating_stats.loc[matching_indices, "r"].abs()
                                    if matching_r.max() > 0:  # At least one non-zero correlation
                                        top_features.append(col)
                        
                        # If we found matches, sort by correlation strength
                        if top_features:
                            # Get correlation values for matched features
                            feature_scores = {}
                            for col in top_features:
                                col_lower = str(col).lower()
                                matching = stats_ids[stats_ids.str.contains(col_lower, na=False, regex=False)]
                                if not matching.empty:
                                    max_r = rating_stats.loc[matching.index, "r"].abs().max()
                                    feature_scores[col] = max_r
                            
                            # Sort by correlation strength and take top 20
                            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:20]
                            top_features = [f[0] for f in top_features]
                
                if not top_features:
                    # Get top 20 features by variance for this type (more lenient)
                    numeric_cols = type_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # Compute variance with NaN handling
                        variances = {}
                        for col in numeric_cols:
                            col_data = type_df[col].dropna()
                            if len(col_data) > 2:  # Need at least 3 values for variance
                                var_val = col_data.var()
                                if pd.notna(var_val) and var_val > 1e-10:
                                    variances[col] = var_val
                        
                        if variances:
                            top_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:20]
                            top_features = [f[0] for f in top_features]
                
                if top_features:
                    # Create subdirectory structure: type/subtype
                    if subtype and subtype != "unknown":
                        type_dir = ctx.subdir("distributions") / ftype / subtype
                    else:
                        type_dir = ctx.subdir("distributions") / ftype
                    ensure_dir(type_dir)
                    
                    # Create filename with type and subtype
                    if subtype and subtype != "unknown":
                        filename = f"sub-{subject}_features_by_{comparison_type}_{ftype}_{subtype}.png"
                    else:
                        filename = f"sub-{subject}_features_by_{comparison_type}_{ftype}.png"
                    
                    path = type_dir / filename
                    plot_feature_by_condition(type_df, top_features, pain_condition, path, config=config)
                    saved_files[f"{comparison_type}_comparison_{ftype}_{subtype}"] = path
                    logger.info(f"Created {ftype}/{subtype} {comparison_type} comparison: {len(top_features)} features")
            except Exception as e:
                logger.warning(f"{comparison_type} comparison failed for {ftype}/{subtype}: {e}", exc_info=True)
    
    # Pain vs non-pain condition comparison
    if feature_df is not None and pain_condition is not None:
        _create_condition_comparisons_by_type(feature_df, pain_condition, saved_files, "condition")
    
    # ═══════════════════════════════════════════════════════════════
    # 3b. Rating-based Distribution Comparison (high vs low rating)
    # ═══════════════════════════════════════════════════════════════
    
    def _create_rating_distributions_by_type(feature_df: pd.DataFrame, targets: pd.Series, 
                                           saved_files: Dict[str, Path]):
        """Create separate distribution plots comparing high vs low rating features."""
        if feature_df is None or feature_df.empty or targets is None or len(targets) < 10:
            return
        
        from eeg_pipeline.analysis.behavior.feature_correlator import classify_feature
        
        # Create binary rating condition (high vs low) - use aligned targets
        median_rating = targets_aligned.median()
        rating_condition = (targets_aligned >= median_rating).astype(int).values  # 1 = high, 0 = low
        
        # Group features by type and subtype
        feature_groups: Dict[Tuple[str, str], List[str]] = {}  # (type, subtype) -> columns
        for col in feature_df.columns:
            source_type = column_to_source.get(col) if column_to_source else None
            ftype, subtype, _ = classify_feature(col, source_file_type=source_type, include_subtype=True)
            key = (ftype, subtype)
            if key not in feature_groups:
                feature_groups[key] = []
            feature_groups[key].append(col)
        
        logger.info(f"Creating rating-based distribution plots by feature type/subtype...")
        
        # Create rating comparison for each feature type/subtype combination
        for (ftype, subtype), cols in feature_groups.items():
            if len(cols) == 0:
                continue
            
            try:
                type_df = feature_df[cols].copy()
                
                # Ensure we have valid numeric data and aligned with targets
                numeric_cols = type_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    logger.debug(f"Skipping {ftype}/{subtype} rating distribution: no numeric columns")
                    continue
                
                # Align type_df with targets index
                if not type_df.index.equals(targets.index):
                    common_idx = type_df.index.intersection(targets.index)
                    if len(common_idx) > 0:
                        type_df = type_df.loc[common_idx]
                        targets_aligned = targets.loc[common_idx]
                    else:
                        logger.debug(f"Skipping {ftype}/{subtype} rating distribution: no common indices with targets")
                        continue
                else:
                    targets_aligned = targets
                
                # Get top features for this type by rating correlation
                top_features = []
                if rating_stats is not None and "r" in rating_stats.columns:
                    id_col = next((c for c in ["feature", "identifier", "channel", "roi", "name"] 
                                  if c in rating_stats.columns), None)
                    if id_col:
                        # Try exact matches first
                        stats_ids = rating_stats[id_col].astype(str).str.lower()
                        for col in cols:
                            col_lower = str(col).lower()
                            if col_lower in stats_ids.values:
                                top_features.append(col)
                        
                        # If not enough, try partial matching
                        if len(top_features) < 10:
                            for col in cols[:50]:
                                col_lower = str(col).lower()
                                matching = stats_ids[stats_ids.str.contains(col_lower, na=False, regex=False)]
                                if not matching.empty:
                                    max_r = rating_stats.loc[matching.index, "r"].abs().max()
                                    if max_r > 0:
                                        top_features.append(col)
                        
                        # Sort by correlation and take top 20
                        if top_features:
                            feature_scores = {}
                            for col in top_features:
                                col_lower = str(col).lower()
                                matching = stats_ids[stats_ids.str.contains(col_lower, na=False, regex=False)]
                                if not matching.empty:
                                    max_r = rating_stats.loc[matching.index, "r"].abs().max()
                                    feature_scores[col] = max_r
                            
                            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:20]
                            top_features = [f[0] for f in top_features]
                
                if not top_features:
                    # Get top 20 features by variance for this type (more lenient)
                    numeric_cols = type_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        variances = {}
                        for col in numeric_cols:
                            col_data = type_df[col].dropna()
                            if len(col_data) > 2:
                                var_val = col_data.var()
                                if pd.notna(var_val) and var_val > 1e-10:
                                    variances[col] = var_val
                        
                        if variances:
                            top_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:20]
                            top_features = [f[0] for f in top_features]
                
                if top_features:
                    # Create subdirectory structure: type/subtype
                    if subtype and subtype != "unknown":
                        type_dir = ctx.subdir("distributions") / ftype / subtype
                    else:
                        type_dir = ctx.subdir("distributions") / ftype
                    ensure_dir(type_dir)
                    
                    # Create filename with type and subtype
                    if subtype and subtype != "unknown":
                        filename = f"sub-{subject}_features_by_rating_{ftype}_{subtype}.png"
                    else:
                        filename = f"sub-{subject}_features_by_rating_{ftype}.png"
                    
                    path = type_dir / filename
                    plot_feature_by_condition(type_df, top_features, rating_condition, path, config=config)
                    saved_files[f"rating_comparison_{ftype}_{subtype}"] = path
                    logger.info(f"Created {ftype}/{subtype} rating distribution: {len(top_features)} features")
            except Exception as e:
                logger.warning(f"Rating distribution failed for {ftype}/{subtype}: {e}", exc_info=True)
    
    # Rating-based distribution comparison
    if feature_df is not None and targets is not None:
        _create_rating_distributions_by_type(feature_df, targets, saved_files)
    
    # ═══════════════════════════════════════════════════════════════
    # 4. Summary Dashboard (using summary module)
    # ═══════════════════════════════════════════════════════════════
    
    logger.info("Creating analysis dashboard...")
    try:
        path = ctx.subdir("summary") / f"sub-{subject}_analysis_dashboard"
        plot_analysis_dashboard(
            subject=subject,
            stats_df=rating_stats,
            power_df=feature_df,
            targets=targets,
            save_path=path,
            temperature=temperature,
            config=config,
            logger=logger,
        )
        saved_files["dashboard"] = path
    except Exception as e:
        logger.warning(f"Dashboard failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 5. Condition Effect Sizes (Pain vs Non-Pain)
    # ═══════════════════════════════════════════════════════════════
    
    logger.info("Creating condition effect size visualizations...")
    effect_size_file = stats_dir / "effect_sizes_all_pain_vs_nonpain.tsv"
    if effect_size_file.exists():
        try:
            effect_df = read_tsv(effect_size_file)
            if effect_df is not None and not effect_df.empty:
                path = ctx.subdir("forest") / f"sub-{subject}_condition_effect_sizes.png"
                plot_condition_effect_sizes(effect_df, path, config=config)
                saved_files["condition_effect_sizes"] = path
                logger.info(f"Created condition effect sizes plot: {len(effect_df)} features")
        except Exception as e:
            logger.warning(f"Condition effect sizes plot failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 6. Temperature Mediation Analysis
    # ═══════════════════════════════════════════════════════════════
    
    logger.info("Creating temperature mediation visualizations...")
    for method in ["spearman", "pearson"]:
        partial_file = stats_dir / f"partial_corr_controlling_temp_{method}.tsv"
        if partial_file.exists():
            try:
                partial_df = read_tsv(partial_file)
                if partial_df is not None and not partial_df.empty:
                    path = ctx.subdir("forest") / f"sub-{subject}_temperature_mediation_{method}.png"
                    plot_temperature_mediation(partial_df, path, config=config)
                    saved_files[f"temperature_mediation_{method}"] = path
                    logger.info(f"Created temperature mediation plot ({method}): {len(partial_df)} features")
                    break  # Only need one method
            except Exception as e:
                logger.warning(f"Temperature mediation plot failed ({method}): {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 7. Summary Heatmap by Feature Type (one per type)
    # ═══════════════════════════════════════════════════════════════
    
    def _create_summary_heatmaps_by_type(stats_df: pd.DataFrame, target_name: str, saved_files: Dict[str, Path]):
        """Create separate summary heatmaps for each feature type."""
        if stats_df is None or stats_df.empty:
            return
        
        from eeg_pipeline.analysis.behavior.feature_correlator import classify_feature
        
        # Classify features by type
        feature_types: Dict[str, List[int]] = {}
        for idx, row in stats_df.iterrows():
            identifier = None
            for col in ["identifier", "channel", "feature", "roi", "name"]:
                if col in row.index and pd.notna(row[col]):
                    identifier = str(row[col])
                    break
            
            if identifier is None:
                continue
            
            ftype, subtype, _ = classify_feature(identifier, include_subtype=True)
            key = (ftype, subtype)
            if key not in feature_types:
                feature_types[key] = []
            feature_types[key].append(idx)
        
        logger.info(f"Creating summary heatmaps by feature type/subtype for {target_name}...")
        
        # Create heatmap for each feature type/subtype combination
        for (ftype, subtype), indices in feature_types.items():
            if len(indices) == 0:
                continue
            
            type_stats = stats_df.iloc[indices].copy()
            if type_stats.empty:
                continue
            
            try:
                # Create subdirectory structure: type/subtype
                if subtype and subtype != "unknown":
                    type_dir = ctx.subdir("heatmaps") / ftype / subtype
                else:
                    type_dir = ctx.subdir("heatmaps") / ftype
                ensure_dir(type_dir)
                
                # Create filename with type and subtype
                if subtype and subtype != "unknown":
                    filename = f"sub-{subject}_{target_name}_correlation_summary_{ftype}_{subtype}.png"
                else:
                    filename = f"sub-{subject}_{target_name}_correlation_summary_{ftype}.png"
                
                path = type_dir / filename
                _plot_type_band_heatmap(type_stats, path, config, logger)
                saved_files[f"summary_heatmap_{target_name}_{ftype}_{subtype}"] = path
                logger.info(f"Created {ftype}/{subtype} summary heatmap: {len(type_stats)} features")
            except Exception as e:
                logger.warning(f"Summary heatmap failed for {ftype}/{subtype}: {e}", exc_info=True)
    
    if rating_stats is not None and not rating_stats.empty:
        _create_summary_heatmaps_by_type(rating_stats, "rating", saved_files)
    
    if temp_stats is not None and not temp_stats.empty:
        _create_summary_heatmaps_by_type(temp_stats, "temperature", saved_files)
    
    logger.info(f"Saved {len(saved_files)} visualization files to {plots_dir}")
    return saved_files


def _plot_type_band_heatmap(
    stats_df: pd.DataFrame,
    save_path: Path,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Create heatmap of mean correlations by feature type and band."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from eeg_pipeline.utils.io.general import save_fig
    from eeg_pipeline.plotting.config import get_plot_config
    
    plot_cfg = get_plot_config(config)
    
    if "band" not in stats_df.columns:
        stats_df = stats_df.copy()
        stats_df["band"] = "N/A"
    
    if "feature_type" not in stats_df.columns:
        # Infer feature type from column names
        stats_df = stats_df.copy()
        
        def _infer_type(row):
            for col in ["feature", "identifier", "channel"]:
                if col in row.index:
                    name = str(row[col]).lower()
                    if name.startswith("pow_"): return "power"
                    if name.startswith("erds_"): return "erds"
                    if name.startswith("ms_"): return "microstate"
                    if name.startswith("aper_"): return "aperiodic"
                    if name.startswith("itpc_"): return "itpc"
                    if name.startswith("wpli_") or name.startswith("plv_"): return "connectivity"
            return "other"
        
        stats_df["feature_type"] = stats_df.apply(_infer_type, axis=1)
    
    # Pivot
    try:
        pivot = stats_df.groupby(["feature_type", "band"])["r"].mean().unstack(fill_value=0)
    except Exception:
        logger.warning("Cannot create pivot for heatmap")
        return
    
    if pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean r", "shrink": 0.8},
        linewidths=0.5,
    )
    
    ax.set_xlabel("Frequency Band", fontsize=11)
    ax.set_ylabel("Feature Type", fontsize=11)
    ax.set_title("Mean Correlation with Behavior by Feature Type × Band",
                fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi)
    plt.close(fig)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "visualize_feature_behavior_correlations",
    "setup_behavior_plot_dirs",
    "PlotContext",
    "FEATURE_TYPE_COLORS",
]
