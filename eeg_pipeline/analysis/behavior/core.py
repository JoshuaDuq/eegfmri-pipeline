"""
Core structures and utilities for behavioral correlation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd

from eeg_pipeline.utils.validation import validate_epochs, validate_features, validate_targets


###################################################################
# Constants
###################################################################

# Minimum samples for valid correlations
MIN_SAMPLES_CHANNEL = 10       # Per-channel analysis
MIN_SAMPLES_ROI = 20           # Per-ROI analysis (more stringent)
MIN_SAMPLES_DEFAULT = 5        # Fallback
MIN_SAMPLES_EDGE = 30          # Connectivity edge analysis
MIN_SAMPLES_TEMPORAL = 15      # Temporal window analysis

# Trial count thresholds
MIN_TRIALS_PER_CONDITION = 15  # For condition-wise analysis
MIN_EPOCHS_FOR_TFR = 20        # For time-frequency analysis
MIN_CHANNELS_FOR_ADJACENCY = 2 # For cluster correction

# Statistical thresholds
DEFAULT_ALPHA = 0.05           # Significance level
DEFAULT_FDR_ALPHA = 0.05       # FDR correction alpha
CLUSTER_FORMING_ALPHA = 0.05   # Cluster-forming threshold

# Bootstrap/permutation defaults
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_PERMUTATIONS = 100
MIN_BOOTSTRAP_SAMPLES = 100

# Numerical stability
EPSILON_CORR = 1e-10           # For correlation stability
MAX_CORRELATION = 0.9999       # Cap for Fisher z-transform

# Correlation methods
CORRELATION_METHODS = ("spearman", "pearson")


###################################################################
# Result Structures
###################################################################


class ComputationStatus(Enum):
    """Status of a computation."""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class CorrelationRecord:
    """Standard record for a single correlation result."""
    identifier: str  # Channel, ROI, or feature name
    band: str  # Frequency band
    correlation: float  # Correlation coefficient
    p_value: float  # Raw p-value
    n_valid: int  # Number of valid samples
    method: str  # "spearman" or "pearson"
    
    # Optional fields
    ci_low: float = np.nan
    ci_high: float = np.nan
    p_perm: float = np.nan  # Permutation p-value
    q_value: float = np.nan  # FDR-corrected p-value
    
    # Partial correlation fields
    r_partial: float = np.nan
    p_partial: float = np.nan
    n_partial: int = 0
    p_partial_perm: float = np.nan
    
    # Temperature partial correlation fields
    r_partial_temp: float = np.nan
    p_partial_temp: float = np.nan
    n_partial_temp: int = 0
    p_partial_temp_perm: float = np.nan
    
    # Context
    identifier_type: str = "channel"  # "channel", "roi", "feature", "roi_pair"
    analysis_type: str = "power"  # "power", "connectivity", "aperiodic", etc.
    
    # Extra fields (stored as dict for flexibility)
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        record = {
            self.identifier_type: self.identifier,
            "band": self.band,
            "r": self.correlation,
            "p": self.p_value,
            "n": self.n_valid,
            "method": self.method,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_perm": self.p_perm,
            "q": self.q_value,
            "analysis": self.analysis_type,
        }
        
        # Add partial correlation fields if they're not NaN/default
        if np.isfinite(self.r_partial):
            record["r_partial"] = self.r_partial
        if np.isfinite(self.p_partial):
            record["p_partial"] = self.p_partial
        if self.n_partial > 0:
            record["n_partial"] = self.n_partial
        if np.isfinite(self.p_partial_perm):
            record["p_partial_perm"] = self.p_partial_perm
        
        # Add temperature partial fields if present
        if np.isfinite(self.r_partial_temp):
            record["r_partial_given_temp"] = self.r_partial_temp
        if np.isfinite(self.p_partial_temp):
            record["p_partial_given_temp"] = self.p_partial_temp
        if self.n_partial_temp > 0:
            record["n_partial_given_temp"] = self.n_partial_temp
        if np.isfinite(self.p_partial_temp_perm):
            record["p_partial_given_temp_perm"] = self.p_partial_temp_perm
        
        # Add extra fields
        record.update(self.extra_fields)
        
        return record
    
    @property
    def is_significant(self) -> bool:
        """Check if significant at default alpha."""
        return self.p_value < DEFAULT_ALPHA


@dataclass
class ComputationResult:
    """Result of a computation."""
    name: str
    status: ComputationStatus
    records: List[CorrelationRecord] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def n_significant(self) -> int:
        """Count of significant correlations."""
        return sum(1 for r in self.records if r.is_significant)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to DataFrame."""
        if self.dataframe is not None:
            return self.dataframe
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.records])


###################################################################
# Context
###################################################################


@dataclass
class BehaviorContext:
    """Shared context for behavior analysis with lazy-loaded data."""
    # Identity
    subject: str
    task: str
    
    # Configuration
    config: Any
    logger: Any
    deriv_root: Path
    stats_dir: Path
    
    # Analysis settings
    use_spearman: bool = True
    bootstrap: int = 0
    n_perm: int = 100
    rng: Optional[np.random.Generator] = None
    partial_covars: Optional[List[str]] = None
    
    epochs: Any = None
    epochs_info: Any = None
    aligned_events: Optional[pd.DataFrame] = None
    
    power_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    microstates_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    precomputed_df: Optional[pd.DataFrame] = None
    
    # Target and covariates
    targets: Optional[pd.Series] = None
    temperature: Optional[pd.Series] = None
    temperature_column: Optional[str] = None
    covariates_df: Optional[pd.DataFrame] = None
    covariates_without_temp_df: Optional[pd.DataFrame] = None
    
    # Results storage
    results: Dict[str, ComputationResult] = field(default_factory=dict)
    
    # Tracking
    _data_loaded: bool = False
    
    @property
    def method(self) -> str:
        """Correlation method name."""
        return "spearman" if self.use_spearman else "pearson"
    
    @property
    def n_trials(self) -> int:
        """Number of trials."""
        if self.targets is not None:
            return len(self.targets)
        if self.power_df is not None:
            return len(self.power_df)
        return 0
    
    @property
    def power_bands(self) -> List[str]:
        """Configured power bands."""
        return self.config.get(
            "power.bands_to_use", 
            ["delta", "theta", "alpha", "beta", "gamma"]
        )
    
    @property
    def min_samples_channel(self) -> int:
        """Minimum samples for channel-level analysis."""
        return int(self.config.get(
            "behavior_analysis.statistics.min_samples_channel",
            MIN_SAMPLES_CHANNEL
        ))
    
    @property
    def min_samples_roi(self) -> int:
        """Minimum samples for ROI-level analysis."""
        return int(self.config.get(
            "behavior_analysis.statistics.min_samples_roi",
            MIN_SAMPLES_ROI
        ))
    
    def load_data(self) -> bool:
        """Load all features and targets once. Returns True if successful."""
        if self._data_loaded:
            return self.targets is not None
        
        from eeg_pipeline.utils.data.loading import (
            _load_features_and_targets,
            load_epochs_for_analysis,
            extract_temperature_data,
            build_covariate_matrix,
            build_covariates_without_temp,
        )
        from eeg_pipeline.utils.io.general import (
            deriv_features_path,
            read_tsv,
        )
        
        self.logger.info("Loading ALL data once (epochs, features, targets, covariates)...")
        
        try:
            # 1. Load epochs and aligned events
            self.epochs, self.aligned_events = load_epochs_for_analysis(
                self.subject, self.task, 
                align="strict", preload=False,
                deriv_root=self.deriv_root,
                bids_root=self.config.bids_root,
                config=self.config,
            )
            
            if self.epochs is None:
                self.logger.warning("No epochs found")
                self._data_loaded = True
                return False
            
            # Validate epochs
            epoch_validation = validate_epochs(self.epochs, self.config, logger=self.logger)
            if not epoch_validation.valid:
                self.logger.warning(f"Epoch validation issues: {epoch_validation.issues}")
                if epoch_validation.critical:
                    self.logger.error(f"Critical epoch issues: {epoch_validation.critical}")
                    self._data_loaded = True
                    return False
            
            # Store epochs info for channel locations
            self.epochs_info = self.epochs.info
            
            # 2. Load main features and targets
            _, self.power_df, self.connectivity_df, self.targets, _ = _load_features_and_targets(
                self.subject, self.task, self.deriv_root, self.config,
                epochs=self.epochs,
            )
            
            # 3. Load temperature data
            if self.aligned_events is not None:
                self.temperature, self.temperature_column = extract_temperature_data(
                    self.aligned_events, self.config
                )
            
            # 4. Build covariates
            if self.aligned_events is not None:
                self.covariates_df = build_covariate_matrix(
                    self.aligned_events, self.partial_covars, self.config
                )
                self.covariates_without_temp_df = build_covariates_without_temp(
                    self.covariates_df, self.temperature_column
                )
            
            # 5. Load additional feature files
            features_dir = deriv_features_path(self.deriv_root, self.subject)
            
            # Microstates
            ms_path = features_dir / "features_microstates.tsv"
            if ms_path.exists():
                self.microstates_df = read_tsv(ms_path)
            
            # Aperiodic
            aper_path = features_dir / "features_aperiodic.tsv"
            if not aper_path.exists():
                # Try alternative name
                aper_path = features_dir / "features_eeg_direct.tsv"
            if aper_path.exists():
                df = read_tsv(aper_path)
                # Extract only aperiodic columns if mixed file
                aper_cols = [c for c in df.columns if str(c).startswith("aper_")]
                if aper_cols:
                    self.aperiodic_df = df[aper_cols]
            
            # Precomputed features (ERD/ERS, spectral, complexity, etc.)
            precomp_path = features_dir / "features_precomputed.tsv"
            if precomp_path.exists():
                self.precomputed_df = read_tsv(precomp_path)
            
            self._data_loaded = True
            
            if self.targets is None:
                self.logger.warning("No target variable found")
                return False
            
            # Validate targets
            target_validation = validate_targets(self.targets, logger=self.logger)
            if not target_validation.valid:
                self.logger.warning(f"Target validation issues: {target_validation.issues}")
            
            # Validate power features if available
            if self.power_df is not None and not self.power_df.empty:
                power_validation = validate_features(self.power_df, logger=self.logger)
                if not power_validation.valid:
                    self.logger.warning(f"Power feature validation issues: {power_validation.issues}")
            
            # Log summary
            n_features = sum([
                len(self.power_df.columns) if self.power_df is not None else 0,
                len(self.connectivity_df.columns) if self.connectivity_df is not None else 0,
                len(self.microstates_df.columns) if self.microstates_df is not None else 0,
                len(self.precomputed_df.columns) if self.precomputed_df is not None else 0,
            ])
            
            self.logger.info(
                f"Loaded ALL data: {self.n_trials} trials, ~{n_features} features, "
                f"temp={self.temperature is not None}"
            )
            return True
            
        except Exception as exc:
            self.logger.error(f"Failed to load data: {exc}")
            self._data_loaded = True
            return False
    
    @property
    def has_temperature(self) -> bool:
        """Check if temperature data is available."""
        return self.temperature is not None and len(self.temperature) > 0
    
    @property 
    def has_covariates(self) -> bool:
        """Check if covariates are available."""
        return self.covariates_df is not None and not self.covariates_df.empty
    
    def add_result(self, name: str, result: ComputationResult) -> None:
        """Add a computation result."""
        self.results[name] = result
        self.logger.info(
            f"  → {name}: {result.status.name}, "
            f"{len(result.records)} records, "
            f"{result.n_significant} significant"
        )
    
    def get_all_records(self) -> List[CorrelationRecord]:
        """Get all correlation records from all computations."""
        all_records = []
        for result in self.results.values():
            all_records.extend(result.records)
        return all_records
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """Get combined DataFrame of all results."""
        dfs = []
        for name, result in self.results.items():
            df = result.to_dataframe()
            if not df.empty:
                df["computation"] = name
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)


###################################################################
# Utility Functions
###################################################################


def build_correlation_record(
    identifier: str,
    band: str,
    r: float,
    p: float,
    n: int,
    method: str = "spearman",
    *,
    ci_low: float = np.nan,
    ci_high: float = np.nan,
    p_perm: float = np.nan,
    r_partial: float = np.nan,
    p_partial: float = np.nan,
    n_partial: int = 0,
    p_partial_perm: float = np.nan,
    r_partial_temp: float = np.nan,
    p_partial_temp: float = np.nan,
    n_partial_temp: int = 0,
    p_partial_temp_perm: float = np.nan,
    identifier_type: str = "channel",
    analysis_type: str = "power",
    **extra_fields,
) -> CorrelationRecord:
    """Build a standardized correlation record.
    
    This is the unified function for creating correlation records.
    All modules should use this function instead of creating dicts directly.
    """
    return CorrelationRecord(
        identifier=identifier,
        band=band,
        correlation=float(r) if np.isfinite(r) else np.nan,
        p_value=float(p) if np.isfinite(p) else np.nan,
        n_valid=int(n),
        method=method,
        ci_low=float(ci_low) if np.isfinite(ci_low) else np.nan,
        ci_high=float(ci_high) if np.isfinite(ci_high) else np.nan,
        p_perm=float(p_perm) if np.isfinite(p_perm) else np.nan,
        r_partial=float(r_partial) if np.isfinite(r_partial) else np.nan,
        p_partial=float(p_partial) if np.isfinite(p_partial) else np.nan,
        n_partial=int(n_partial),
        p_partial_perm=float(p_partial_perm) if np.isfinite(p_partial_perm) else np.nan,
        r_partial_temp=float(r_partial_temp) if np.isfinite(r_partial_temp) else np.nan,
        p_partial_temp=float(p_partial_temp) if np.isfinite(p_partial_temp) else np.nan,
        n_partial_temp=int(n_partial_temp),
        p_partial_temp_perm=float(p_partial_temp_perm) if np.isfinite(p_partial_temp_perm) else np.nan,
        identifier_type=identifier_type,
        analysis_type=analysis_type,
        extra_fields=extra_fields,
    )


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> Tuple[float, float, int]:
    """Compute correlation with validation. Returns (r, p, n_valid).
    
    This is the standardized correlation computation function.
    All modules should use this instead of direct scipy calls or
    compute_correlation from utils.analysis.stats.
    
    Args:
        x: First array
        y: Second array (must match length of x)
        method: "spearman" or "pearson"
        min_samples: Minimum number of valid samples required
        
    Returns:
        Tuple of (correlation, p_value, n_valid)
        Returns (nan, nan, n_valid) if insufficient samples or error
    """
    from scipy import stats
    
    # Convert to arrays if needed
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Check length match
    if x.size != y.size:
        return np.nan, np.nan, 0
    
    # Align and filter NaN
    mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(np.sum(mask))
    
    if n_valid < min_samples:
        return np.nan, np.nan, n_valid
    
    if n_valid == 0:
        return np.nan, np.nan, 0
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Check for constant arrays (correlation undefined)
    if np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return np.nan, np.nan, n_valid
    
    try:
        if method == "spearman":
            r, p = stats.spearmanr(x_clean, y_clean, nan_policy="omit")
        elif method == "pearson":
            r, p = stats.pearsonr(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")
        
        # Ensure finite values
        r = float(r) if np.isfinite(r) else np.nan
        p = float(p) if np.isfinite(p) else np.nan
        
        return r, p, n_valid
    except (ValueError, RuntimeWarning, FloatingPointError, TypeError) as e:
        # Return NaN on any error
        return np.nan, np.nan, n_valid


def align_to_valid(
    *arrays: np.ndarray,
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> Tuple[List[np.ndarray], int]:
    """Align arrays to common valid indices. Returns (aligned_arrays, n_valid)."""
    if not arrays:
        return [], 0
    
    # Find common valid mask
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    
    n_valid = np.sum(mask)
    if n_valid < min_samples:
        return [np.array([]) for _ in arrays], 0
    
    return [arr[mask] for arr in arrays], int(n_valid)


def correlate_features_loop(
    feature_df: pd.DataFrame,
    target_values: Union[pd.Series, np.ndarray],
    method: str = "spearman",
    min_samples: int = MIN_SAMPLES_DEFAULT,
    logger: Optional[Any] = None,
    *,
    condition_mask: Optional[np.ndarray] = None,
    identifier_type: str = "feature",
    analysis_type: str = "unknown",
    feature_classifier: Optional[Any] = None,
) -> Tuple[List[CorrelationRecord], pd.DataFrame]:
    """Generic function to correlate all features in a DataFrame with target values.
    
    This is the unified function for iterating over features and computing correlations.
    All modules should use this instead of custom loops.
    
    Args:
        feature_df: DataFrame with features as columns
        target_values: Target variable (Series or array)
        method: "spearman" or "pearson"
        min_samples: Minimum samples required
        logger: Optional logger
        condition_mask: Optional boolean mask to filter rows (for condition-specific analysis)
        identifier_type: Type of identifier for records ("feature", "channel", "roi", etc.)
        analysis_type: Type of analysis for records
        feature_classifier: Optional function to classify feature names (col_name) -> (type, metadata)
        
    Returns:
        Tuple of (list of CorrelationRecord, DataFrame of results)
    """
    if feature_df.empty:
        return [], pd.DataFrame()
    
    # Convert target to array
    if isinstance(target_values, pd.Series):
        target_arr = target_values.values
    else:
        target_arr = np.asarray(target_values)
    
    # Apply condition mask if provided
    if condition_mask is not None:
        feature_df = feature_df.iloc[condition_mask]
        target_arr = target_arr[condition_mask]
        n_condition = int(condition_mask.sum())
        if n_condition < min_samples:
            if logger:
                logger.warning(f"Insufficient trials ({n_condition}) for correlation")
            return [], pd.DataFrame()
    
    # Align lengths
    n_features = len(feature_df)
    n_targets = len(target_arr)
    
    if n_features != n_targets:
        if logger:
            logger.warning(f"Length mismatch: features={n_features}, targets={n_targets}")
        n_use = min(n_features, n_targets)
        feature_df = feature_df.iloc[:n_use]
        target_arr = target_arr[:n_use]
    
    records: List[CorrelationRecord] = []
    
    if logger:
        n_cols = len(feature_df.columns)
        context = f"{analysis_type} features" if analysis_type != "unknown" else "features"
        logger.info(f"Correlating {n_cols} {context}...")
    
    for col in feature_df.columns:
        feature_values = feature_df[col].to_numpy()
        
        # Classify feature if classifier provided
        if feature_classifier:
            feature_type, metadata = feature_classifier(col)
            identifier = metadata.get("identifier", metadata.get("channel", col))
            band = metadata.get("band", "N/A")
        else:
            feature_type = analysis_type
            identifier = col
            band = "N/A"
        
        # Compute correlation
        r, p, n = safe_correlation(feature_values, target_arr, method, min_samples)
        
        if not np.isfinite(r):
            continue
        
        record = build_correlation_record(
            identifier=identifier,
            band=band,
            r=r,
            p=p,
            n=n,
            method=method,
            identifier_type=identifier_type,
            analysis_type=feature_type,
        )
        records.append(record)
    
    n_sig = sum(1 for r in records if r.is_significant)
    if logger:
        logger.info(f"  {len(records)} features, {n_sig} significant (p<0.05)")
    
    if not records:
        return [], pd.DataFrame()
    
    results_df = pd.DataFrame([r.to_dict() for r in records])
    return records, results_df


def save_correlation_results(
    records: Union[List[CorrelationRecord], pd.DataFrame],
    output_path: Path,
    apply_fdr: bool = True,
    config: Optional[Any] = None,
    logger: Optional[Any] = None,
    use_permutation_p: bool = True,
    add_fdr_reject: bool = False,
) -> None:
    """Save correlation results to TSV with optional FDR correction.
    
    This is the unified function for saving correlation results.
    All modules should use this instead of manually applying FDR and writing.
    
    Args:
        records: List of CorrelationRecord objects or DataFrame
        output_path: Path to save TSV file
        apply_fdr: Whether to apply FDR correction
        config: Optional config for FDR alpha
        logger: Optional logger
        use_permutation_p: If True, use p_perm if available, otherwise use p
        add_fdr_reject: If True, add fdr_reject and fdr_crit_p columns (like apply_fdr_correction_and_save)
    """
    from eeg_pipeline.utils.io.general import write_tsv
    from eeg_pipeline.utils.analysis.stats import fdr_bh, fdr_bh_reject, get_fdr_alpha_from_config
    
    # Convert records to DataFrame if needed
    if isinstance(records, list):
        if not records:
            if logger:
                logger.warning(f"No records to save to {output_path}")
            return
        df = pd.DataFrame([r.to_dict() for r in records])
    else:
        df = records.copy()
    
    if df.empty:
        if logger:
            logger.warning(f"Empty DataFrame, not saving to {output_path}")
        return
    
    # Apply FDR if requested
    if apply_fdr and "p" in df.columns:
        # Select p-values based on preference
        if use_permutation_p and "p_perm" in df.columns:
            p_values = df["p_perm"].to_numpy(dtype=float)
        else:
            p_values = df["p"].to_numpy(dtype=float)
        
        valid_mask = np.isfinite(p_values)
        
        if np.any(valid_mask):
            alpha = get_fdr_alpha_from_config(config) if config else 0.05
            q_values = np.full(len(df), np.nan, dtype=float)
            q_values[valid_mask] = fdr_bh(p_values[valid_mask], alpha=alpha, config=config)
            df["q"] = q_values
            
            # Add rejection columns if requested
            if add_fdr_reject:
                rejections, critical_p = fdr_bh_reject(p_values[valid_mask], alpha=alpha)
                df["fdr_reject"] = False
                df.loc[valid_mask, "fdr_reject"] = rejections
                df["fdr_crit_p"] = critical_p
                
                if logger:
                    logger.info(
                        f"Applying per-analysis-type FDR correction (alpha={alpha}) to {len(df)} tests. "
                        f"Note: This controls FDR within this analysis type only. "
                        f"For global FDR across all analysis types, use apply_global_fdr() separately."
                    )
        else:
            df["q"] = np.nan
            if add_fdr_reject:
                df["fdr_reject"] = False
                df["fdr_crit_p"] = np.nan
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    write_tsv(df, output_path)
    
    if logger:
        n_sig = int((df["q"] < 0.05).sum()) if "q" in df.columns else 0
        logger.info(f"Saved {len(df)} correlation results to {output_path.name} ({n_sig} significant)")

