import os
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from mne.time_frequency import AverageTFR
from nilearn.plotting import plot_connectome as nilearn_plot_connectome, plot_markers

from . import utils


###################################################################
# Group-Level Feature Plots
###################################################################

def group_plot_features(out_path, task):
    report = mne.Report()
    report.add_section("Peak alpha power and COG", level=1)

    features = pd.read_csv(
        os.path.join(out_path, f"task-{task}_features_frame.tsv"), sep="\t"
    )

    fig = plt.figure(figsize=(10, 5))
    sns.violinplot(
        features["alpha_cog_global"],
        label="Global",
    )
    sns.swarmplot(
        features["alpha_cog_global"],
        label="Global",
    )
    plt.figure(figsize=(10, 5))
    sns.histplot(
        features["alpha_peak_somato"],
        bins=20,
        kde=True,
        color="red",
        label="Somatosensory",
    )

    plt.figure(figsize=(10, 5))
    sns.histplot(
        features["alpha_peak_somato"],
        bins=20,
        kde=True,
        color="red",
        label="Somatosensory",
    )
    plt.title("Alpha peak frequency distribution at global and somatosensory channels")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Density")
    plt.legend()


###################################################################
# TFR Plotting
###################################################################

def plot_tfr_nice_spectro(
    tfr: AverageTFR,
    chans: List[str],
    cmap: str = "RdBu_r",
    time_range: Tuple[Optional[float], Optional[float]] = (None, None),
    freq_range: Optional[Tuple[float, float]] = None,
    vlim: Tuple[float, float] = (-3, 3),
    figsize: Tuple[float, float] = (2, 1),
    nameout: str = "",
    title: str = "",
    cbar: bool = True,
    fontsize: int = 8,
    save_path: Optional[str] = None,
    cbar_label: str = "Z-score",
    y_ticks: Optional[List[float]] = None,
    x_ticks: Optional[List[float]] = None,
    hide_xlabel: bool = False,
    hide_ylabel: bool = False,
    remove_xticks: bool = False,
    remove_yticks: bool = False,
    remove_xticklabels: bool = False,
    remove_yticklabels: bool = False,
    extension: str = "svg",
    dpi: int = 1200,
    bbox_inches: str = "tight",
    transparent: bool = True,
    show=False
):
    fig, ax = plt.subplots(figsize=figsize)

    tfr.plot(
        chans,
        baseline=None,
        show=False,
        vlim=vlim,
        tmin=time_range[0],
        tmax=time_range[1],
        axes=ax,
        cmap=cmap,
        title=None,
        colorbar=cbar,
        combine="mean",
    )

    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if time_range[0] is not None or time_range[1] is not None:
        ax.set_xlim(time_range)
    
    if freq_range is not None:
        ax.set_ylim(freq_range)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    if cbar:
        if len(fig.axes) > 1:
            cbar_ax = fig.axes[1]
            cbar_ax.set_ylabel(cbar_label, fontsize=fontsize)
            cbar_ax.tick_params(labelsize=fontsize - 2)
    
    if hide_ylabel:
        ax.set_ylabel("")

    if remove_yticks:
        ax.set_yticks([])

    if hide_xlabel:
        ax.set_xlabel("")

    if remove_xticks:
        ax.set_xticks([])
    
    if remove_xticklabels:
        ax.set_xticklabels([])

    if remove_yticklabels:
        ax.set_yticklabels([])
    
    if save_path is not None:
        full_path = os.path.join(
            save_path,
            f"{nameout}_spectro.{extension}",
        )
        fig.savefig(
            full_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
        )
    if not show:    
        plt.close(fig)


def plot_tfr_nice_topo(
    tfr: AverageTFR,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    title: str = "",
    cmap: str = "RdBu_r",
    vlim: Tuple[float, float] = (-1.5, 1.5),
    figsize: Tuple[float, float] = (1.5, 1.5),
    nameout: str = "",
    cbar: bool = True,
    fontsize: int = 10,
    cbar_label: str = "Z-score",
    save_path: Optional[str] = None,
    extension: str = "png",
    dpi: int = 800,
    bbox_inches: str = "tight",
    transparent: bool = True,
    show: bool = False
):
    avg_power_topo = (
        tfr.copy()
        .crop(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
        .get_data()
        .mean(axis=(1, 2))
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im, _ = mne.viz.plot_topomap(
        avg_power_topo,
        tfr.info,
        vlim=vlim,
        cmap=cmap,
        contours=False,
        axes=ax,
        show=False,
    )

    ax.set_title(title, fontsize=fontsize)

    if cbar:
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.04, ax.get_position().height])
        clb = fig.colorbar(im, cax=cax)
        clb.set_label(cbar_label, fontsize=fontsize - 2)
        clb.ax.tick_params(labelsize=fontsize - 3)

    if save_path is not None:
        full_path = os.path.join(
            save_path,
            f"{nameout}_topo.{extension}",
        )
        fig.savefig(
            full_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
        )
    
    if not show:
        plt.close(fig)


###################################################################
# PSD Plotting
###################################################################

def plot_psd(psd, save_path, title="PSD"):
    fig = psd.plot(show=False)
    fig.savefig(save_path)
    plt.close(fig)


def plot_psd_with_bands(psd, freq_bands, save_path, title="PSD with Frequency Bands", alpha_peak=None, alpha_cog=None):
    dat = psd.get_data().mean(axis=0)
    plt.figure()
    plt.plot(psd.freqs, dat, label="Average PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(title)
    
    if alpha_peak is not None:
        peak_idx = np.where(psd.freqs == alpha_peak)[0]
        if len(peak_idx) > 0:
            plt.scatter(
                alpha_peak,
                dat[peak_idx[0]],
                color="r",
                linestyle="--",
                label="Peak alpha",
            )
    
    if alpha_cog is not None:
        cog_idx = np.argmin(np.abs(psd.freqs - alpha_cog))
        plt.scatter(
            alpha_cog,
            dat[cog_idx],
            color="g",
            linestyle="--",
            label="COG alpha",
        )

    band_colors = ["red", "blue", "green", "yellow", "purple"]
    for band, color in zip(freq_bands, band_colors[:len(freq_bands)]):
        freq_range = (psd.freqs >= freq_bands[band][0]) & (
            psd.freqs <= freq_bands[band][1]
        )
        plt.fill_between(
            psd.freqs[freq_range], dat[freq_range], color=color, alpha=0.5, label=band
        )

    plt.legend()
    plt.savefig(save_path)
    plt.close("all")


###################################################################
# Connectivity Plotting
###################################################################

def plot_connectivity_matrix(connectivity_matrix, labels, save_path, title="Connectivity Matrix", cmap="viridis"):
    plt.figure(figsize=(20, 20))
    plt.imshow(connectivity_matrix, cmap=cmap)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close("all")


def plot_connectome(connectivity_matrix, coords, threshold=None, save_path=None, title="Connectome"):
    if threshold is not None:
        connectivity_matrix = connectivity_matrix.copy()
        connectivity_matrix[connectivity_matrix < threshold] = 0
    
    nilearn_plot_connectome(
        connectivity_matrix,
        coords,
        node_color="auto",
        edge_threshold=threshold,
        title=title,
        output_file=save_path,
    )


###################################################################
# Graph Measures Plotting
###################################################################

def plot_graph_measures(measures, coords, save_path, title_prefix=""):
    plot_markers(
        measures,
        coords,
        title=title_prefix,
    )
    plt.savefig(save_path)
    plt.close("all")

