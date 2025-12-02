"""
Feature Distribution Visualizations.

Violin plots, raincloud plots, and distribution summaries for
EEG features and behavioral variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.general import save_fig, ensure_dir, get_band_color


def plot_feature_distributions(
    feature_df: pd.DataFrame,
    save_path: Path,
    *,
    feature_cols: Optional[List[str]] = None,
    max_features: int = 20,
    title: str = "Feature Distributions",
    figsize: Tuple[float, float] = (14, 10),
    config: Any = None,
) -> plt.Figure:
    """Create violin plots for multiple features.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame with features as columns
    save_path : Path
        Output path
    feature_cols : list, optional
        Columns to plot (default: auto-select)
    max_features : int
        Maximum features to show
    
    Returns
    -------
    matplotlib.Figure
    """
    if feature_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No feature data", ha="center", va="center")
        return fig
    
    # Select features
    if feature_cols is None:
        # Auto-select numeric columns, excluding identifiers
        exclude_cols = {"subject", "epoch", "trial", "condition", "time", "index"}
        feature_cols = [
            c for c in feature_df.columns 
            if c.lower() not in exclude_cols and pd.api.types.is_numeric_dtype(feature_df[c])
        ]
    
    if not feature_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No numeric features found", ha="center", va="center")
        return fig
    
    # Limit and sort by variance
    variances = feature_df[feature_cols].var()
    top_features = variances.nlargest(max_features).index.tolist()
    
    # Prepare data for violin plot
    plot_data = feature_df[top_features].melt(var_name="Feature", value_name="Value")
    
    # Create figure
    plot_cfg = get_plot_config(config)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Violin plot with individual points
    sns.violinplot(
        data=plot_data,
        x="Feature",
        y="Value",
        hue="Feature",
        ax=ax,
        inner="box",
        cut=0,
        density_norm="width",
        palette="Set2",
        legend=False,
    )
    
    # Styling
    tick_labels = ax.get_xticklabels()
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Value (z-scored recommended)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_raincloud(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: Path,
    *,
    hue_col: Optional[str] = None,
    title: str = "Raincloud Plot",
    orient: str = "h",
    figsize: Tuple[float, float] = (10, 6),
    config: Any = None,
) -> plt.Figure:
    """Create raincloud plot (half-violin + jittered strip + boxplot).
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with x and y columns
    x_col : str
        Categorical variable
    y_col : str
        Numeric variable
    hue_col : str, optional
        Additional grouping variable
    orient : str
        "h" for horizontal, "v" for vertical
    """
    if data.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    plot_cfg = get_plot_config(config)
    
    # Half violin
    if orient == "h":
        sns.violinplot(
            data=data,
            x=y_col,
            y=x_col,
            hue=hue_col,
            ax=ax,
            inner=None,
            split=True if hue_col else False,
            cut=0,
            palette="Set2",
            linewidth=1,
        )
        
        # Boxplot overlay
        sns.boxplot(
            data=data,
            x=y_col,
            y=x_col,
            hue=hue_col,
            ax=ax,
            width=0.15,
            showfliers=False,
            boxprops=dict(alpha=0.5),
            whiskerprops=dict(alpha=0.5),
            medianprops=dict(color="red", linewidth=1.5),
        )
        
        # Strip plot for individual points
        sns.stripplot(
            data=data,
            x=y_col,
            y=x_col,
            hue=hue_col,
            ax=ax,
            size=2,
            alpha=0.4,
            jitter=True,
            dodge=True if hue_col else False,
        )
    else:
        sns.violinplot(
            data=data,
            y=y_col,
            x=x_col,
            hue=hue_col,
            ax=ax,
            inner=None,
            split=True if hue_col else False,
            cut=0,
            palette="Set2",
        )
        
        sns.boxplot(
            data=data,
            y=y_col,
            x=x_col,
            hue=hue_col,
            ax=ax,
            width=0.15,
            showfliers=False,
            boxprops=dict(alpha=0.5),
        )
        
        sns.stripplot(
            data=data,
            y=y_col,
            x=x_col,
            hue=hue_col,
            ax=ax,
            size=2,
            alpha=0.4,
            jitter=True,
        )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    if hue_col:
        ax.legend(title=hue_col, loc="best", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_behavioral_summary(
    targets: pd.Series,
    temperature: Optional[pd.Series],
    condition: Optional[pd.Series],
    save_path: Path,
    *,
    subject: str = "",
    title: str = "Behavioral Summary",
    figsize: Tuple[float, float] = (12, 8),
    config: Any = None,
) -> plt.Figure:
    """Create summary plot of behavioral variables.
    
    Shows distributions and relationships between rating, temperature,
    and condition.
    """
    plot_cfg = get_plot_config(config)
    
    has_temp = temperature is not None and not temperature.empty
    has_cond = condition is not None and not condition.empty
    
    n_cols = 2 if has_temp else 1
    n_rows = 2 if has_cond else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    # Rating distribution
    ax = axes[0, 0]
    ax.hist(targets.dropna(), bins=25, color="#3B82F6", alpha=0.7, edgecolor="white")
    ax.axvline(targets.mean(), color="red", linestyle="--", label=f"Mean: {targets.mean():.2f}")
    ax.set_xlabel("Pain Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Temperature distribution / rating vs temp
    if has_temp:
        ax = axes[0, 1]
        valid = targets.notna() & temperature.notna()
        ax.scatter(temperature[valid], targets[valid], alpha=0.5, s=15, c="#22C55E")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Pain Rating")
        ax.set_title("Rating vs Temperature")
        ax.grid(alpha=0.3)
        
        # Add regression line
        if valid.sum() > 5:
            from scipy import stats
            slope, intercept, r, p, _ = stats.linregress(temperature[valid], targets[valid])
            x_line = np.array([temperature[valid].min(), temperature[valid].max()])
            ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, alpha=0.8)
            ax.text(0.05, 0.95, f"r = {r:.2f}, p = {p:.3f}", transform=ax.transAxes,
                   fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Condition-wise rating
    if has_cond:
        ax = axes[1, 0]
        cond_data = pd.DataFrame({"Rating": targets, "Condition": condition})
        sns.violinplot(data=cond_data, x="Condition", y="Rating", ax=ax, palette=["navy", "crimson"])
        ax.set_title("Rating by Condition")
        ax.grid(axis="y", alpha=0.3)
        
        if has_temp:
            ax = axes[1, 1]
            cond_data["Temperature"] = temperature
            for cond_val, color in zip(condition.unique(), ["navy", "crimson"]):
                mask = (condition == cond_val) & targets.notna() & temperature.notna()
                ax.scatter(temperature[mask], targets[mask], alpha=0.5, s=15, 
                          c=color, label=str(cond_val))
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Pain Rating")
            ax.set_title("Rating vs Temp by Condition")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    suptitle = f"{title}"
    if subject:
        suptitle += f" — sub-{subject}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_feature_by_condition(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    condition: Union[str, np.ndarray],
    save_path: Path,
    *,
    max_features: int = 12,
    title: str = "Features by Condition",
    figsize: Tuple[float, float] = (14, 10),
    config: Any = None,
) -> plt.Figure:
    """Compare feature distributions across conditions.
    
    Creates paired violin plots showing feature differences between conditions.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature data
    feature_cols : List[str]
        List of feature columns to plot
    condition : str or np.ndarray
        Either a column name in feature_df, or a numpy array of condition labels
        (0/1 for binary conditions, or categorical labels)
    """
    if feature_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Handle condition: if it's an array, add it as a column
    if isinstance(condition, np.ndarray):
        condition_col = "_condition"
        # Align condition array with dataframe index
        if len(condition) == len(feature_df):
            feature_df = feature_df.copy()
            feature_df[condition_col] = condition
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Condition length mismatch: {len(condition)} vs {len(feature_df)}", ha="center", va="center")
            save_fig(fig, save_path)
            plt.close(fig)
            return fig
    elif isinstance(condition, str):
        condition_col = condition
        if condition_col not in feature_df.columns:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Condition column '{condition_col}' not found", ha="center", va="center")
            save_fig(fig, save_path)
            plt.close(fig)
            return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Invalid condition type", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Validate feature columns
    if not feature_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No features provided", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Filter to features that exist in dataframe
    feature_cols = [c for c in feature_cols if c in feature_df.columns]
    if not feature_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid features found", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Limit features
    feature_cols = feature_cols[:max_features]
    n_features = len(feature_cols)
    
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    conditions = feature_df[condition_col].unique()
    # Map binary conditions to labels
    if len(conditions) == 2 and all(c in [0, 1] for c in conditions):
        condition_labels = {0: "Low/Non-Pain", 1: "High/Pain"}
        feature_df = feature_df.copy()
        feature_df[condition_col] = feature_df[condition_col].map(condition_labels)
        conditions = feature_df[condition_col].unique()
    
    palette = ["navy", "crimson"] if len(conditions) == 2 else "Set2"
    
    for idx, feat in enumerate(feature_cols):
        ax = axes[idx]
        plot_data = feature_df[[feat, condition_col]].dropna()
        
        if plot_data.empty:
            continue
        
        sns.violinplot(
            data=plot_data,
            x=condition_col,
            y=feat,
            ax=ax,
            hue=condition_col,
            palette=palette,
            inner="quartile",
            legend=False,
        )
        
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.3)
        
        # Add significance indicator
        if len(conditions) == 2:
            from scipy import stats
            g1 = plot_data[plot_data[condition_col] == conditions[0]][feat]
            g2 = plot_data[plot_data[condition_col] == conditions[1]][feat]
            if len(g1) > 3 and len(g2) > 3:
                try:
                    _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    if sig:
                        y_max = plot_data[feat].max()
                        ax.text(0.5, 0.98, sig, transform=ax.transAxes, ha="center", va="top",
                               fontsize=12, fontweight="bold")
                except Exception:
                    pass
    
    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_feature_correlation_matrix(
    feature_df: pd.DataFrame,
    save_path: Path,
    *,
    feature_cols: Optional[List[str]] = None,
    max_features: int = 30,
    method: str = "spearman",
    cluster: bool = True,
    title: str = "Feature Correlation Matrix",
    figsize: Tuple[float, float] = (12, 10),
    config: Any = None,
) -> plt.Figure:
    """Create clustered heatmap of feature correlations.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        Features as columns
    method : str
        Correlation method ("spearman" or "pearson")
    cluster : bool
        Whether to apply hierarchical clustering
    """
    if feature_df is None or feature_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Select features
    if feature_cols is None:
        exclude_cols = {"subject", "epoch", "trial", "condition", "time", "index"}
        feature_cols = [
            c for c in feature_df.columns 
            if c.lower() not in exclude_cols and pd.api.types.is_numeric_dtype(feature_df[c])
        ]
    
    feature_cols = list(feature_cols)[:max_features]
    
    if len(feature_cols) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Need at least 2 features", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Prepare data: handle NaN/Inf more leniently
    plot_df = feature_df[feature_cols].copy()
    
    # Replace Inf with NaN
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    
    # Filter out constant columns (zero variance) BEFORE dropping rows
    # This is important - some features might be constant, which causes issues
    valid_cols = []
    for col in feature_cols:
        if col in plot_df.columns:
            col_data = plot_df[col].dropna()
            if len(col_data) > 1 and col_data.nunique() > 1:
                # Check variance on non-NaN data
                var_val = col_data.var()
                if pd.notna(var_val) and var_val > 1e-10:
                    valid_cols.append(col)
    
    if len(valid_cols) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Insufficient valid features: {len(valid_cols)} (need ≥2)", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    plot_df = plot_df[valid_cols]
    
    # For correlation, use pairwise deletion (more lenient than listwise)
    # This allows features with different missing patterns to still be correlated
    # Compute correlation matrix with pairwise deletion
    try:
        corr_matrix = plot_df.corr(method=method, min_periods=3)  # Need at least 3 overlapping values
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Correlation computation failed: {e}", ha="center", va="center", fontsize=9)
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # Check if we have enough valid correlations
    if corr_matrix.isna().all().all():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid correlations (insufficient overlapping data)", ha="center", va="center")
        save_fig(fig, save_path)
        plt.close(fig)
        return fig
    
    # For clustering, we need complete data, so filter to features with enough complete cases
    # But first, try with pairwise correlations and fill NaN with 0 for visualization
    corr_matrix_filled = corr_matrix.fillna(0)
    
    # Check for Inf in correlation matrix
    if np.isinf(corr_matrix_filled.values).any():
        corr_matrix_filled = corr_matrix_filled.replace([np.inf, -np.inf], 0)
    
    # Use filled matrix for clustering, but warn if many NaN
    n_missing = corr_matrix.isna().sum().sum()
    n_total = len(corr_matrix) * len(corr_matrix.columns)
    if n_missing > n_total * 0.5:  # More than 50% missing
        # Too much missing data for reliable clustering - use simple heatmap
        cluster = False
        corr_matrix = corr_matrix_filled
    else:
        corr_matrix = corr_matrix_filled
    
    # Create clustered heatmap with error handling
    if cluster and len(corr_matrix) >= 2:
        try:
            # For clustering, we need complete data - check if we can get enough
            # Try to find features with sufficient complete cases
            complete_cases = plot_df.dropna()
            if len(complete_cases) >= 3 and len(complete_cases.columns) >= 2:
                # Recompute correlation on complete cases for clustering
                corr_for_cluster = complete_cases.corr(method=method)
                if not corr_for_cluster.isna().all().all():
                    g = sns.clustermap(
                        corr_for_cluster,
                        cmap="RdBu_r",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        figsize=figsize,
                        dendrogram_ratio=0.15,
                        cbar_pos=(0.02, 0.8, 0.03, 0.15),
                        linewidths=0.5,
                        annot=len(valid_cols) <= 15,
                        fmt=".2f",
                        annot_kws={"fontsize": 6},
                        method='ward',
                        metric='euclidean',
                    )
                    g.fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
                    save_fig(g.fig, save_path)
                    plt.close(g.fig)
                    return g.fig
            # If not enough complete cases, fall through to simple heatmap
            cluster = False
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as cluster_error:
            # If clustering fails, fall back to simple heatmap
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Clustering failed for {save_path.name}, using simple heatmap: {cluster_error}")
            cluster = False  # Fall through to non-clustered version
    if not cluster:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            square=True,
            linewidths=0.5,
            annot=len(valid_cols) <= 15,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            cbar_kws={"label": f"{method.title()} correlation", "shrink": 0.8},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, save_path)
        plt.close(fig)
        return fig


def plot_top_predictors_summary(
    stats_df: pd.DataFrame,
    save_path: Path,
    *,
    r_col: str = "r",
    p_col: str = "p",
    feature_col: str = "feature",
    band_col: str = "band",
    top_n: int = 25,
    alpha: float = 0.05,
    title: str = "Top Brain-Behavior Predictors",
    figsize: Tuple[float, float] = (12, 10),
    config: Any = None,
) -> plt.Figure:
    """Create comprehensive summary of top predictors.
    
    Combines horizontal bar chart with effect sizes and significance.
    """
    if stats_df.empty or r_col not in stats_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No stats data", ha="center", va="center")
        return fig
    
    df = stats_df.copy()
    df["abs_r"] = df[r_col].abs()
    df = df.dropna(subset=[r_col]).nlargest(top_n, "abs_r")
    
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid correlations", ha="center", va="center")
        return fig
    
    plot_cfg = get_plot_config(config)
    fig, ax = plt.subplots(figsize=figsize)
    
    n_features = len(df)
    y_pos = np.arange(n_features)[::-1]
    
    # Determine significance
    if p_col in df.columns:
        q_col = "q" if "q" in df.columns else None
        if q_col:
            sig_mask = df[q_col] < alpha
        else:
            sig_mask = df[p_col] < alpha
    else:
        sig_mask = np.ones(n_features, dtype=bool)
    
    # Color by band if available
    if band_col in df.columns:
        bands = df[band_col].unique()
        band_colors = {}
        for band in bands:
            try:
                band_colors[band] = get_band_color(band, config)
            except Exception:
                band_colors[band] = "#666666"
        colors = [band_colors.get(b, "#666666") for b in df[band_col]]
    else:
        colors = [plot_cfg.get_color("significant") if s else plot_cfg.get_color("nonsignificant") 
                  for s in sig_mask]
    
    # Create bars
    r_values = df[r_col].values
    bars = ax.barh(y_pos, r_values, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    
    # Add significance markers
    for i, (r, sig) in enumerate(zip(r_values, sig_mask)):
        marker = "★" if sig else ""
        x_pos = r + 0.02 if r >= 0 else r - 0.02
        ha = "left" if r >= 0 else "right"
        ax.text(x_pos, y_pos[i], f"{r:.3f} {marker}", va="center", ha=ha, fontsize=8)
    
    # Labels
    if feature_col in df.columns:
        labels = df[feature_col].values
        if band_col in df.columns:
            labels = [f"{f} ({b})" for f, b in zip(df[feature_col], df[band_col])]
    else:
        labels = [f"Feature {i}" for i in range(n_features)]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Correlation (r)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    
    # Legend for bands
    if band_col in df.columns:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=band_colors[b], label=b.title()) for b in bands]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8, title="Frequency Band")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig
