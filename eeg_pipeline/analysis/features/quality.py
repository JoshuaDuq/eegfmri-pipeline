"""
Feature Quality Assessment
===========================

Quality metrics for evaluating extracted features:
- Finite fraction (missing data)
- Variance (constant features)
- Outlier detection
- Reliability estimates

These metrics help identify unreliable features before ML modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne
from scipy import stats


@dataclass
class FeatureQuality:
    """Quality metadata for a single feature."""
    
    name: str
    finite_fraction: float  # Fraction of epochs with finite values
    variance: float  # Feature variance (0 = constant)
    n_outliers: int  # Number of statistical outliers
    outlier_fraction: float  # Fraction of outliers
    skewness: float  # Distribution skewness
    kurtosis: float  # Distribution kurtosis
    reliability: Optional[float] = None  # Split-half reliability if computed
    
    # Quality flags
    is_constant: bool = False  # Variance near zero
    is_sparse: bool = False  # Too many missing values
    has_outliers: bool = False  # Significant outlier count
    
    def is_usable(
        self,
        min_finite: float = 0.8,
        max_outlier_frac: float = 0.1,
        min_variance: float = 1e-10,
    ) -> bool:
        """Check if feature passes quality thresholds."""
        return (
            self.finite_fraction >= min_finite
            and self.outlier_fraction <= max_outlier_frac
            and self.variance >= min_variance
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FeatureQualityReport:
    """Quality report for a set of features."""
    
    features: Dict[str, FeatureQuality] = field(default_factory=dict)
    n_total: int = 0
    n_usable: int = 0
    n_constant: int = 0
    n_sparse: int = 0
    n_outlier_heavy: int = 0
    
    def add(self, fq: FeatureQuality) -> None:
        """Add a feature quality record."""
        self.features[fq.name] = fq
        self.n_total += 1
        if fq.is_usable():
            self.n_usable += 1
        if fq.is_constant:
            self.n_constant += 1
        if fq.is_sparse:
            self.n_sparse += 1
        if fq.has_outliers:
            self.n_outlier_heavy += 1
    
    def get_usable_features(self) -> List[str]:
        """Get list of usable feature names."""
        return [name for name, fq in self.features.items() if fq.is_usable()]
    
    def get_problematic_features(self) -> Dict[str, List[str]]:
        """Get features grouped by problem type."""
        return {
            "constant": [n for n, fq in self.features.items() if fq.is_constant],
            "sparse": [n for n, fq in self.features.items() if fq.is_sparse],
            "outlier_heavy": [n for n, fq in self.features.items() if fq.has_outliers],
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        records = [fq.to_dict() for fq in self.features.values()]
        return pd.DataFrame(records)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "n_total": self.n_total,
            "n_usable": self.n_usable,
            "usable_fraction": self.n_usable / self.n_total if self.n_total > 0 else 0.0,
            "n_constant": self.n_constant,
            "n_sparse": self.n_sparse,
            "n_outlier_heavy": self.n_outlier_heavy,
        }


def compute_feature_quality(
    df: pd.DataFrame,
    *,
    outlier_z_threshold: float = 3.0,
    min_finite_fraction: float = 0.8,
    min_variance: float = 1e-10,
    max_outlier_fraction: float = 0.1,
    exclude_columns: Optional[List[str]] = None,
) -> FeatureQualityReport:
    """
    Compute quality metrics for all features in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (epochs x features)
    outlier_z_threshold : float
        Z-score threshold for outlier detection
    min_finite_fraction : float
        Minimum fraction of finite values for a usable feature
    min_variance : float
        Minimum variance for non-constant feature
    max_outlier_fraction : float
        Maximum outlier fraction for usable feature
    exclude_columns : Optional[List[str]]
        Columns to exclude from quality assessment (e.g., "condition")
    
    Returns
    -------
    FeatureQualityReport
        Quality report for all features
    """
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject"]
    
    report = FeatureQualityReport()
    
    for col in df.columns:
        if col in exclude_columns:
            continue
        
        values = df[col].to_numpy(dtype=float)
        n_total = len(values)
        
        # Finite fraction
        finite_mask = np.isfinite(values)
        n_finite = np.sum(finite_mask)
        finite_fraction = n_finite / n_total if n_total > 0 else 0.0
        
        # Get finite values for further analysis
        finite_values = values[finite_mask]
        
        if len(finite_values) < 2:
            # Not enough data
            fq = FeatureQuality(
                name=col,
                finite_fraction=finite_fraction,
                variance=0.0,
                n_outliers=0,
                outlier_fraction=0.0,
                skewness=np.nan,
                kurtosis=np.nan,
                is_constant=True,
                is_sparse=True,
                has_outliers=False,
            )
            report.add(fq)
            continue
        
        # Variance
        variance = float(np.var(finite_values, ddof=1))
        is_constant = variance < min_variance
        
        # Outlier detection (Z-score based)
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values, ddof=1)
        
        if std_val > min_variance:
            z_scores = np.abs((finite_values - mean_val) / std_val)
            n_outliers = int(np.sum(z_scores > outlier_z_threshold))
        else:
            n_outliers = 0
        
        outlier_fraction = n_outliers / n_finite if n_finite > 0 else 0.0
        
        # Distribution stats
        skewness = float(stats.skew(finite_values, nan_policy='omit'))
        kurtosis = float(stats.kurtosis(finite_values, nan_policy='omit'))
        
        # Quality flags
        is_sparse = finite_fraction < min_finite_fraction
        has_outliers = outlier_fraction > max_outlier_fraction
        
        fq = FeatureQuality(
            name=col,
            finite_fraction=finite_fraction,
            variance=variance,
            n_outliers=n_outliers,
            outlier_fraction=outlier_fraction,
            skewness=skewness,
            kurtosis=kurtosis,
            is_constant=is_constant,
            is_sparse=is_sparse,
            has_outliers=has_outliers,
        )
        report.add(fq)
    
    return report


def filter_quality_features(
    df: pd.DataFrame,
    report: FeatureQualityReport,
    *,
    keep_metadata_cols: bool = True,
    metadata_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only usable features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original feature DataFrame
    report : FeatureQualityReport
        Quality report from compute_feature_quality
    keep_metadata_cols : bool
        Keep metadata columns like "condition"
    metadata_cols : Optional[List[str]]
        Specific metadata columns to keep
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only usable features
    """
    if metadata_cols is None:
        metadata_cols = ["condition", "epoch", "trial", "subject"]
    
    usable = report.get_usable_features()
    
    if keep_metadata_cols:
        cols_to_keep = [c for c in metadata_cols if c in df.columns] + usable
    else:
        cols_to_keep = usable
    
    return df[cols_to_keep].copy()


def identify_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    *,
    exclude_columns: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Identify highly correlated feature pairs.
    
    Useful for removing redundant features before ML modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    threshold : float
        Correlation threshold for flagging pairs
    exclude_columns : Optional[List[str]]
        Columns to exclude
    
    Returns
    -------
    List[Tuple[str, str, float]]
        List of (feature1, feature2, correlation) tuples
    """
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject"]
    
    # Get numeric columns only
    numeric_cols = [c for c in df.columns 
                   if c not in exclude_columns 
                   and pd.api.types.is_numeric_dtype(df[c])]
    
    if len(numeric_cols) < 2:
        return []
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find pairs above threshold
    correlated_pairs = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:  # Upper triangle only
                corr = corr_matrix.loc[col1, col2]
                if np.isfinite(corr) and abs(corr) >= threshold:
                    correlated_pairs.append((col1, col2, float(corr)))
    
    # Sort by correlation magnitude
    correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return correlated_pairs


def remove_redundant_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    *,
    prefer_shorter_names: bool = True,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove redundant (highly correlated) features.
    
    For each correlated pair, keeps one feature based on preference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    threshold : float
        Correlation threshold
    prefer_shorter_names : bool
        When removing one of a pair, prefer keeping shorter names
    exclude_columns : Optional[List[str]]
        Columns to never remove
    
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Filtered DataFrame and list of removed columns
    """
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject"]
    
    correlated = identify_correlated_features(df, threshold, exclude_columns=exclude_columns)
    
    to_remove = set()
    for col1, col2, _ in correlated:
        if col1 in to_remove or col2 in to_remove:
            continue
        
        # Decide which to remove
        if prefer_shorter_names:
            remove = col2 if len(col1) <= len(col2) else col1
        else:
            remove = col2
        
        to_remove.add(remove)
    
    removed = list(to_remove)
    keep_cols = [c for c in df.columns if c not in to_remove]
    
    return df[keep_cols].copy(), removed


###################################################################
# Distribution Validation
###################################################################


def validate_feature_distributions(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    normality_alpha: float = 0.05,
    outlier_z: float = 3.0,
) -> Dict[str, Dict[str, Any]]:
    """Check feature distributions for issues.
    
    Returns dict with per-feature diagnostics:
    - is_normal: Shapiro-Wilk test result
    - n_outliers: Count of outliers (|z| > threshold)
    - has_floor: >10% at minimum value
    - has_ceiling: >10% at maximum value
    - skewness, kurtosis: Distribution shape
    """
    from scipy import stats as scipy_stats
    
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject", "index"]
    
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_columns]
    results = {}
    
    for col in numeric_cols:
        values = df[col].dropna().values
        n = len(values)
        
        if n < 5:
            results[col] = {"valid": False, "error": "insufficient_data", "n": n}
            continue
        
        result = {"valid": True, "n": n}
        
        # Normality test (Shapiro-Wilk for n < 5000, else D'Agostino-Pearson)
        try:
            if n < 5000:
                stat, p = scipy_stats.shapiro(values[:5000] if n > 5000 else values)
            else:
                stat, p = scipy_stats.normaltest(values)
            result["is_normal"] = p > normality_alpha
            result["normality_p"] = float(p)
        except (ValueError, RuntimeError):
            result["is_normal"] = None
            result["normality_p"] = np.nan
        
        # Outliers
        z_scores = np.abs((values - np.mean(values)) / (np.std(values) + 1e-12))
        result["n_outliers"] = int(np.sum(z_scores > outlier_z))
        result["outlier_fraction"] = float(result["n_outliers"] / n)
        
        # Floor/ceiling effects
        val_min, val_max = np.min(values), np.max(values)
        result["has_floor"] = float(np.mean(values == val_min)) > 0.1
        result["has_ceiling"] = float(np.mean(values == val_max)) > 0.1
        
        # Shape statistics
        result["skewness"] = float(scipy_stats.skew(values))
        result["kurtosis"] = float(scipy_stats.kurtosis(values))
        result["mean"] = float(np.mean(values))
        result["std"] = float(np.std(values))
        result["median"] = float(np.median(values))
        
        results[col] = result
    
    return results


def compute_feature_icc(
    df: pd.DataFrame,
    feature_cols: List[str],
    subject_col: str = "subject",
    icc_type: str = "ICC(1,1)",
) -> pd.DataFrame:
    """Compute ICC for feature reliability across trials within subjects."""
    from eeg_pipeline.analysis.behavior.mixed_effects import compute_icc
    
    results = []
    
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        
        try:
            icc, (ci_low, ci_high) = compute_icc(df, feature, subject_col, icc_type)
            results.append({
                "feature": feature,
                "icc": icc,
                "icc_ci_low": ci_low,
                "icc_ci_high": ci_high,
                "icc_type": icc_type,
            })
        except (ValueError, KeyError):
            results.append({
                "feature": feature,
                "icc": np.nan,
                "icc_ci_low": np.nan,
                "icc_ci_high": np.nan,
                "icc_type": icc_type,
            })
    
    return pd.DataFrame(results)


def identify_problematic_trials(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    z_threshold: float = 3.0,
    max_nan_fraction: float = 0.5,
) -> np.ndarray:
    """Identify trials with outlier feature values or excessive missing data.
    
    Returns boolean array where True = problematic trial.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in ["epoch", "trial", "subject", "index"]]
    
    n_trials = len(df)
    problematic = np.zeros(n_trials, dtype=bool)
    
    # Check for excessive NaNs
    nan_fraction = df[feature_cols].isna().mean(axis=1).values
    problematic |= nan_fraction > max_nan_fraction
    
    # Check for outliers in each feature
    for col in feature_cols:
        if col not in df.columns:
            continue
        values = df[col].values
        valid = np.isfinite(values)
        if np.sum(valid) < 5:
            continue
        
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std < 1e-12:
            continue
        
        z_scores = np.abs((values - mean) / std)
        problematic |= (z_scores > z_threshold) & valid
    
    return problematic


def compute_trial_quality_metrics(
    epochs: "mne.Epochs",
    config: Any,
    logger: Any,
) -> pd.DataFrame:
    """Compute per-trial quality metrics."""
    import mne
    from eeg_pipeline.analysis.features.core import pick_eeg_channels
    
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        return pd.DataFrame()
    
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]
    
    records = []
    
    for ep_idx in range(n_epochs):
        ep_data = data[ep_idx]
        record = {"epoch": ep_idx}
        
        # Overall variance
        record["variance"] = float(np.var(ep_data))
        
        # SNR estimate (signal = low-freq, noise = high-freq)
        try:
            low_freq = mne.filter.filter_data(ep_data, sfreq, 1, 30, verbose=False)
            high_freq = mne.filter.filter_data(ep_data, sfreq, 50, None, verbose=False)
            signal_power = np.mean(low_freq ** 2)
            noise_power = np.mean(high_freq ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
            record["snr_db"] = float(snr)
        except (ValueError, RuntimeError):
            record["snr_db"] = np.nan
        
        # Muscle artifact indicator (high gamma power ratio)
        try:
            total_power = np.mean(ep_data ** 2)
            muscle = mne.filter.filter_data(ep_data, sfreq, 30, None, verbose=False)
            muscle_power = np.mean(muscle ** 2)
            record["muscle_ratio"] = float(muscle_power / (total_power + 1e-12))
        except (ValueError, RuntimeError):
            record["muscle_ratio"] = np.nan
        
        # Peak-to-peak amplitude (artifact indicator)
        record["ptp_amplitude"] = float(np.max(ep_data) - np.min(ep_data))
        
        # Fraction of finite values
        record["finite_fraction"] = float(np.mean(np.isfinite(ep_data)))
        
        records.append(record)
    
    return pd.DataFrame(records)


def generate_quality_report(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    subject_col: str = "subject",
) -> Dict[str, Any]:
    """Generate comprehensive quality report for feature DataFrame."""
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in ["epoch", "trial", "subject", "index", "condition"]]
    
    report = {
        "n_trials": len(df),
        "n_features": len(feature_cols),
        "n_subjects": df[subject_col].nunique() if subject_col in df.columns else 1,
    }
    
    # Missing data summary
    missing = df[feature_cols].isna().sum()
    report["missing_data"] = {
        "total_missing": int(missing.sum()),
        "features_with_missing": int((missing > 0).sum()),
        "worst_feature": missing.idxmax() if missing.max() > 0 else None,
        "worst_missing_count": int(missing.max()),
    }
    
    # Distribution validation
    report["distribution_issues"] = validate_feature_distributions(df, exclude_columns=[subject_col])
    
    # Problem features summary
    issues = report["distribution_issues"]
    report["summary"] = {
        "non_normal_features": sum(1 for f in issues.values() if f.get("is_normal") == False),
        "features_with_outliers": sum(1 for f in issues.values() if f.get("n_outliers", 0) > 0),
        "features_with_floor": sum(1 for f in issues.values() if f.get("has_floor")),
        "features_with_ceiling": sum(1 for f in issues.values() if f.get("has_ceiling")),
    }
    
    # Problematic trials
    problematic = identify_problematic_trials(df, feature_cols)
    report["problematic_trials"] = {
        "count": int(np.sum(problematic)),
        "fraction": float(np.mean(problematic)),
        "indices": list(np.where(problematic)[0]),
    }
    
    return report

