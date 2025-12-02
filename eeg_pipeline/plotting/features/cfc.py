"""
Cross-Frequency Coupling Visualization
========================================

Comodulograms, PAC topomaps, and phase-phase coupling plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.general import save_fig, ensure_dir

if TYPE_CHECKING:
    import mne


def plot_pac_comodulogram(
    pac_matrix: np.ndarray,
    phase_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    save_path: Path,
    *,
    title: str = "Phase-Amplitude Coupling",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (8, 6),
    vmin: float = None,
    vmax: float = None,
) -> plt.Figure:
    """Create comodulogram heatmap of PAC values.
    
    Parameters
    ----------
    pac_matrix : np.ndarray
        2D array of PAC values (phase_freqs x amp_freqs)
    phase_freqs : np.ndarray
        Phase frequency values
    amp_freqs : np.ndarray
        Amplitude frequency values
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if vmax is None:
        vmax = np.nanmax(pac_matrix)
    if vmin is None:
        vmin = 0
    
    im = ax.imshow(
        pac_matrix.T, origin="lower", aspect="auto",
        extent=[phase_freqs[0], phase_freqs[-1], amp_freqs[0], amp_freqs[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax
    )
    
    ax.set_xlabel("Phase Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Amplitude Frequency (Hz)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Modulation Index", fontsize=9)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_pac_comodulogram_grid(
    pac_data: Dict[str, np.ndarray],
    phase_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    save_path: Path,
    *,
    n_cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4, 3),
    cmap: str = "viridis",
) -> plt.Figure:
    """Grid of comodulograms for multiple conditions/channels."""
    n_plots = len(pac_data)
    if n_plots == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    # Find global vmax
    vmax = max(np.nanmax(m) for m in pac_data.values())
    
    for idx, (label, pac_matrix) in enumerate(pac_data.items()):
        ax = axes[idx]
        
        im = ax.imshow(
            pac_matrix.T, origin="lower", aspect="auto",
            extent=[phase_freqs[0], phase_freqs[-1], amp_freqs[0], amp_freqs[-1]],
            cmap=cmap, vmin=0, vmax=vmax
        )
        
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Phase (Hz)", fontsize=8)
        ax.set_ylabel("Amp (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide unused
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MI", fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_mi_pac_topomaps(
    mi_df: pd.DataFrame,
    info: "mne.Info",
    save_path: Path,
    *,
    mi_col_pattern: str = "mi_pac_",
    figsize: Tuple[float, float] = (12, 4),
    cmap: str = "Reds",
) -> plt.Figure:
    """Plot topomaps of modulation index values."""
    import mne
    
    # Find MI columns
    mi_cols = [c for c in mi_df.columns if mi_col_pattern in c and "_mean" in c]
    
    if not mi_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No MI columns found", ha="center", va="center")
        return fig
    
    n_plots = len(mi_cols)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0], figsize[1]))
    if n_plots == 1:
        axes = [axes]
    
    # Get channel positions
    ch_names = info["ch_names"]
    
    for idx, col in enumerate(mi_cols):
        ax = axes[idx]
        
        # Extract mean MI per channel (if available) or use global mean
        values = mi_df[col].values
        mean_mi = np.nanmean(values)
        
        # For now, show as bar (topomaps require per-channel data)
        ax.bar([0], [mean_mi], color="#C42847", width=0.5)
        ax.set_ylim(0, max(0.1, mean_mi * 1.5))
        
        # Clean up label
        label = col.replace(mi_col_pattern, "").replace("_mean", "")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("MI" if idx == 0 else "", fontsize=9)
    
    fig.suptitle("Modulation Index by Band Pair", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_phase_phase_coupling_matrix(
    ppc_df: pd.DataFrame,
    save_path: Path,
    *,
    ppc_col_pattern: str = "ppc_",
    title: str = "Phase-Phase Coupling",
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "RdYlBu_r",
) -> plt.Figure:
    """Plot phase-phase coupling values as matrix/bars."""
    ppc_cols = [c for c in ppc_df.columns if ppc_col_pattern in c and "_mean" in c]
    
    if not ppc_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No PPC columns found", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract mean values
    labels = []
    values = []
    
    for col in ppc_cols:
        label = col.replace(ppc_col_pattern, "").replace("_mean", "")
        labels.append(label)
        values.append(np.nanmean(ppc_df[col].values))
    
    y_pos = np.arange(len(labels))
    colors = plt.cm.RdYlBu_r(Normalize(vmin=0, vmax=1)(values))
    
    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Phase Locking Value", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Reference line
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_cfc_behavior_correlation(
    cfc_df: pd.DataFrame,
    behavior: np.ndarray,
    save_path: Path,
    *,
    cfc_col: str = "mi_pac_theta_gamma_mean",
    behavior_label: str = "Rating",
    title: str = "CFC-Behavior Correlation",
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Scatter plot of CFC measure vs behavior."""
    from scipy import stats
    
    if cfc_col not in cfc_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Column {cfc_col} not found", ha="center", va="center")
        return fig
    
    x = cfc_df[cfc_col].values
    y = np.asarray(behavior).flatten()
    
    # Align lengths
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]
    
    valid = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[valid], y[valid]
    
    if len(x_clean) < 5:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x_clean, y_clean, c="#3B82F6", alpha=0.6, s=40, edgecolors="white", linewidths=0.5)
    
    # Regression
    r, p = stats.spearmanr(x_clean, y_clean)
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"r = {r:.3f}, p = {p:.3f}")
    
    ax.set_xlabel(cfc_col.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel(behavior_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig

