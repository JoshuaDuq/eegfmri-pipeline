from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.core.utils import get_font_sizes
from eeg_pipeline.plotting.core.colorbars import create_difference_colorbar
from eeg_pipeline.plotting.core.annotations import get_sig_marker_text
from eeg_pipeline.plotting.behavioral.temporal import (
    _add_correlation_roi_annotations,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    ensure_dir,
    fdr_bh_reject,
    get_viz_params,
    plot_topomap_on_ax,
    robust_sym_vlim,
    save_fig,
    get_behavior_footer as _get_behavior_footer,
    get_default_logger as _get_default_logger,
    log_if_present as _log_if_present,
)
from eeg_pipeline.utils.analysis.stats import (
    get_eeg_adjacency,
    compute_cluster_masses_1d,
    compute_cluster_pvalues_1d,
)


###################################################################
# Helper Functions
###################################################################


def _bh_reject(p_vals: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    p_flat = p_vals[np.isfinite(p_vals)].ravel()
    m = len(p_flat)
    if m == 0:
        return np.zeros_like(p_vals, dtype=bool), np.nan
    order = np.argsort(p_flat)
    ranked = p_flat[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    rejects = ranked <= thresh
    max_idx = np.where(rejects)[0]
    crit = thresh[max_idx.max()] if max_idx.size else np.nan
    mask = np.zeros_like(p_vals, dtype=bool)
    if max_idx.size:
        mask_flat = np.zeros(m, dtype=bool)
        mask_flat[order] = rejects
        mask[np.isfinite(p_vals)] = mask_flat
    return mask, crit


###################################################################
# Group Temporal Aggregation
###################################################################


def _aggregate_temporal_across_subjects(
    subjects: List[str],
    deriv_root: Path,
    filename_base: str,
    condition_keys: List[str],
    use_spearman: bool,
    logger: logging.Logger,
    config: Any,
) -> Optional[Tuple[dict, Any, List[str], Path]]:
    suffix = "_spearman" if use_spearman else "_pearson"
    aggregated: dict = {}
    info_ref = None
    ch_names_ref: Optional[List[str]] = None
    stats_group_dir = Path(deriv_root) / "group" / "eeg" / "stats"
    ensure_dir(stats_group_dir)

    for subj in subjects:
        stats_dir = deriv_stats_path(deriv_root, subj)
        npz_path = stats_dir / f"{filename_base}{suffix}.npz"
        if not npz_path.exists():
            continue
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as exc:
            _log_if_present(logger, "warning", f"Failed to load {npz_path}: {exc}")
            continue

        if not condition_keys:
            derived_keys = [k for k in data.keys() if isinstance(k, str) and k.startswith("temp_")]
            if derived_keys:
                condition_keys = derived_keys

        if info_ref is None:
            info_ref = data.get("info", None)
            if isinstance(info_ref, np.ndarray) and info_ref.dtype == object:
                info_ref = info_ref.item()
            ch_names_ref = data.get("ch_names", None)
            if isinstance(ch_names_ref, np.ndarray):
                ch_names_ref = ch_names_ref.tolist()

        for cond in condition_keys:
            if cond not in data:
                continue
            cond_dict = data[cond].item()
            corr = cond_dict["correlations"]
            z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
            aggregated.setdefault(cond, []).append(z)
            aggregated.setdefault(f"{cond}_meta", cond_dict)

    if not aggregated:
        return None

    n_perm = int(config.get("behavior_analysis.statistics.group_temporal_n_perm", 2000))
    alpha = float(config.get("behavior_analysis.statistics.fdr_alpha", config.get("statistics.sig_alpha", 0.05)))

    out = {}
    group_npz = {}
    for cond in condition_keys:
        z_list = aggregated.get(cond, [])
        if not z_list:
            continue
        z_stack = np.stack(z_list, axis=0)
        n_subj = z_stack.shape[0]
        mean_z = np.nanmean(z_stack, axis=0)

        z_flat = mean_z.ravel()
        shape = mean_z.shape
        z_subject_flat = z_stack.reshape(n_subj, -1)
        rng = np.random.default_rng(42)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n_subj))
        perm_means = (signs @ z_subject_flat) / float(n_subj)
        perm_means = perm_means.reshape(n_perm, *shape)
        obs = mean_z
        with np.errstate(invalid="ignore"):
            p_perm = (np.sum(np.abs(perm_means) >= np.abs(obs), axis=0) + 1) / float(n_perm + 1)

        finite_mask = np.isfinite(p_perm)
        mask_fdr = np.zeros_like(p_perm, dtype=bool)
        q_vals = np.full_like(p_perm, np.nan, dtype=float)
        if finite_mask.any():
            p_flat = p_perm[finite_mask].ravel()
            order = np.argsort(p_flat)
            p_sorted = p_flat[order]
            m = len(p_sorted)
            q_sorted = p_sorted * m / (np.arange(1, m + 1))
            q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
            q_flat = np.empty_like(p_flat)
            q_flat[order] = q_sorted
            q_vals[finite_mask] = q_flat
            mask_fdr = (q_vals < alpha) & finite_mask

        mean_r = np.tanh(mean_z)
        adjacency = None
        if info_ref is not None:
            adjacency, _, _ = get_eeg_adjacency(info_ref, logger=logger)
        cluster_labels = np.zeros_like(mean_r, dtype=int)
        cluster_sig = np.zeros_like(mean_r, dtype=bool)
        p_cluster = np.full_like(mean_r, np.nan, dtype=float)
        if adjacency is not None and perm_means is not None:
            n_bands, n_windows, n_ch = mean_r.shape
            for b in range(n_bands):
                for w in range(n_windows):
                    obs_corr = mean_r[b, w]
                    obs_p = p_perm[b, w]
                    labels_obs, masses_obs = compute_cluster_masses_1d(
                        obs_corr,
                        obs_p,
                        alpha,
                        {i: i for i in range(n_ch)},
                        np.arange(n_ch),
                        adjacency,
                    )
                    perm_max = []
                    for pm in perm_means[:, b, w, :]:
                        _, masses_perm = compute_cluster_masses_1d(
                            pm,
                            obs_p,
                            alpha,
                            {i: i for i in range(n_ch)},
                            np.arange(n_ch),
                            adjacency,
                        )
                        max_mass = max(masses_perm.values()) if masses_perm else 0.0
                        perm_max.append(max_mass)
                    if masses_obs:
                        labels, pvals_clusters, records = compute_cluster_pvalues_1d(
                            labels_obs, masses_obs, perm_max, alpha
                        )
                        cluster_labels[b, w] = labels
                        p_cluster[b, w] = pvals_clusters
                        cluster_sig[b, w] = pvals_clusters < alpha

        meta = aggregated.get(f"{cond}_meta", {})
        out[cond] = {
            "correlations": mean_r,
            "p_values": p_perm,
            "q_values": q_vals,
            "fdr_mask": mask_fdr,
            "p_cluster": p_cluster,
            "cluster_labels": cluster_labels,
            "cluster_sig_mask": cluster_sig,
            "n_subjects": n_subj,
            "band_names": meta.get("band_names"),
            "band_ranges": meta.get("band_ranges"),
            "window_starts": meta.get("window_starts"),
            "window_ends": meta.get("window_ends"),
        }
        group_npz[cond] = out[cond]

    if group_npz:
        npz_path = stats_group_dir / f"group_{filename_base}{suffix}.npz"
        np.savez_compressed(npz_path, **group_npz, ch_names=np.asarray(ch_names_ref), info=info_ref)
        _log_if_present(logger, "info", f"Saved group temporal NPZ to {npz_path}")

    return out, info_ref, ch_names_ref, stats_group_dir


###################################################################
# Group Temporal Topomap Rendering
###################################################################


def _render_group_temporal_topomaps(
    group_results: dict,
    info,
    ch_names: List[str],
    condition_labels: List[str],
    plots_dir: Path,
    title_prefix: str,
    use_spearman: bool,
    config: Any,
    logger: logging.Logger,
) -> None:
    if not group_results:
        _log_if_present(logger, "warning", "No group temporal data to plot.")
        return

    viz_params = get_viz_params(config)
    font_sizes = get_font_sizes()
    sig_text = get_sig_marker_text(config)

    if info is not None and ch_names is not None:
        if len(ch_names) != len(info["ch_names"]) or set(ch_names) != set(info["ch_names"]):
            picks = mne.pick_channels(info["ch_names"], include=ch_names, exclude=[])
            info = mne.pick_info(info, picks)

    plot_cfg = get_plot_config(config) if config else None
    ensure_dir(plots_dir)

    band_names = None
    window_starts = None
    window_ends = None
    for cond_key in group_results:
        meta = group_results[cond_key]
        band_names = meta.get("band_names")
        window_starts = meta.get("window_starts")
        window_ends = meta.get("window_ends")
        break
    if band_names is None or window_starts is None or window_ends is None:
        _log_if_present(logger, "warning", "Missing metadata for group temporal plots.")
        return

    n_windows = len(window_starts)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    topomap_config = tfr_config.get("topomap", {})
    tfr_specific = topomap_config.get("tfr_specific", {})
    hspace = tfr_specific.get("hspace", 0.25)
    wspace = tfr_specific.get("wspace", 1.2)

    for band_idx, band_name in enumerate(band_names):
        all_corr = []
        for cond_key in condition_labels:
            result = group_results.get(cond_key)
            if result is None:
                continue
            corr = result["correlations"][band_idx]
            all_corr.extend([c for c in corr.flatten() if np.isfinite(c)])
        vabs_corr = robust_sym_vlim(all_corr) if all_corr else 0.6

        fig, axes = plt.subplots(
            len(condition_labels), n_windows,
            figsize=(4.0 * n_windows, 3.2 * len(condition_labels)),
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace},
        )

        for row_idx, cond_key in enumerate(condition_labels):
            result = group_results.get(cond_key)
            if result is None:
                continue
            correlations = result["correlations"][band_idx]
            p_vals = result["p_values"][band_idx]
            fdr_mask_all = result.get("fdr_mask")
            p_fdr_mask = fdr_mask_all[band_idx] if fdr_mask_all is not None else np.zeros_like(correlations, dtype=bool)

            if row_idx == 0:
                axes[row_idx, 0].set_ylabel(f"{cond_key}\n{band_name}", fontsize=font_sizes["ylabel"], labelpad=10)
            else:
                axes[row_idx, 0].set_ylabel(cond_key, fontsize=font_sizes["ylabel"], labelpad=10)

            for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
                if row_idx == 0:
                    axes[row_idx, col].set_title(f"{tmin_win:.2f}s", fontsize=font_sizes["title"], pad=12, y=1.07)

                corr_data = correlations[col, :]
                p_uncorr = p_vals[col, :]
                mask = p_fdr_mask[col, :] if p_fdr_mask is not None else np.zeros_like(corr_data, dtype=bool)

                plot_topomap_on_ax(
                    axes[row_idx, col], corr_data, info,
                    vmin=-vabs_corr, vmax=+vabs_corr,
                    mask=mask,
                    mask_params=dict(marker="o", markerfacecolor="green", markeredgecolor="green", markersize=4),
                    config=config
                )

                sig_unc = (p_uncorr < 0.05) & np.isfinite(p_uncorr) & ~mask
                if sig_unc.any():
                    try:
                        from mne.channels.layout import _find_topomap_coords
                        pos = _find_topomap_coords(info, picks=None)
                        axes[row_idx, col].plot(
                            pos[sig_unc, 0], pos[sig_unc, 1],
                            "o", markerfacecolor="white", markeredgecolor="black",
                            markersize=4, markeredgewidth=1, zorder=10
                        )
                    except Exception:
                        pass

                _add_correlation_roi_annotations(
                    axes[row_idx, col], corr_data, p_uncorr, p_uncorr, info, config=config, fdr_alpha=0.05
                )

        create_difference_colorbar(
            fig, axes, vabs_corr, viz_params["topo_cmap"],
            label="Correlation coefficient"
        )
        method_name = "Spearman" if use_spearman else "Pearson"
        fig.suptitle(
            f"{title_prefix} ({band_name})\n{method_name} correlation, vlim ±{vabs_corr:.2f}{sig_text}\n",
            fontsize=font_sizes["suptitle"], y=0.995
        )
        filename = f"group_temporal_correlations_{title_prefix.replace(' ', '_').lower()}_{band_name}.png"
        save_fig(fig, plots_dir / filename, formats=config.get("output.save_formats", ["svg"]), bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)


###################################################################
# Main Group Temporal Topomap Function
###################################################################


def plot_group_temporal_topomaps(
    subjects: List[str],
    deriv_root: Path,
    plots_dir: Path,
    config,
    logger: logging.Logger,
    use_spearman: bool = True,
    condition: str = "pain",
) -> None:
    if condition == "pain":
        filename_base = "temporal_correlations_by_pain"
        condition_keys = ["pain", "non_pain"]
        title_prefix = "Temporal correlations by pain (group)"
    elif condition == "temperature":
        filename_base = "temporal_correlations_by_temperature"
        condition_keys = []
        title_prefix = "Temporal correlations by temperature (group)"
    else:
        logger.warning(f"Unknown condition '{condition}' for group temporal plots.")
        return

    deriv_root = Path(deriv_root)
    agg_result = _aggregate_temporal_across_subjects(
        subjects, deriv_root, filename_base, condition_keys, use_spearman, logger, config
    )
    if agg_result is None:
        logger.warning("No temporal data aggregated across subjects.")
        return

    group_results, info, ch_names, stats_group_dir = agg_result
    if condition == "temperature":
        condition_labels = sorted([k for k in group_results.keys() if k.startswith("temp_")])
    else:
        condition_labels = condition_keys

    plot_dir = plots_dir / "behavior"
    ensure_dir(plot_dir)

    if condition_labels and group_results:
        rows = []
        for cond_key in condition_labels:
            res = group_results.get(cond_key)
            if res is None:
                continue
            band_names = res.get("band_names") or []
            window_starts = res.get("window_starts") or []
            window_ends = res.get("window_ends") or []
            q_vals = res.get("q_values")
            p_vals = res.get("p_values")
            corr = res.get("correlations")
            n_subj = res.get("n_subjects", 0)
            for b_idx, band in enumerate(band_names):
                for w_idx, (t_start, t_end) in enumerate(zip(window_starts, window_ends)):
                    for ch_idx, ch in enumerate(ch_names or []):
                        p_val = p_vals[b_idx, w_idx, ch_idx] if p_vals is not None else np.nan
                        q_val = q_vals[b_idx, w_idx, ch_idx] if q_vals is not None else np.nan
                        fdr_reject = bool(np.isfinite(q_val) and q_val < float(config.get('behavior_analysis', {}).get('statistics', {}).get('fdr_alpha', config.get('statistics.sig_alpha', 0.05))))
                        rows.append({
                            "condition": cond_key,
                            "band": band,
                            "time_start": float(t_start),
                            "time_end": float(t_end),
                            "channel": ch,
                            "r_mean": float(corr[b_idx, w_idx, ch_idx]) if corr is not None else np.nan,
                            "p_perm": float(p_val) if np.isfinite(p_val) else np.nan,
                            "q_fdr": float(q_val) if np.isfinite(q_val) else np.nan,
                            "fdr_reject": fdr_reject,
                            "n_subjects": int(n_subj),
                        })
        if rows:
            out_tsv = stats_group_dir / f"group_corr_stats_temporal_{condition}.tsv"
            ensure_dir(out_tsv.parent)
            pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
            _log_if_present(logger, "info", f"Wrote group temporal stats to {out_tsv}")
            try:
                df = pd.read_csv(out_tsv, sep="\t")
                if "p_perm" in df.columns:
                    p_vals = pd.to_numeric(df["p_perm"], errors="coerce")
                    mask = np.isfinite(p_vals)
                    alpha = float(config.get("behavior_analysis", {}).get("statistics", {}).get("fdr_alpha", config.get("statistics.sig_alpha", 0.05)))
                    rejections, crit_p = fdr_bh_reject(p_vals[mask].to_numpy(), alpha=alpha)
                    df.loc[mask, "q_fdr_global"] = np.nan
                    df.loc[mask, "fdr_reject_global"] = False
                    if rejections is not None:
                        df.loc[mask, "fdr_reject_global"] = rejections
                        sorted_idx = p_vals[mask].sort_values().index
                        m = len(sorted_idx)
                        ranks = np.arange(1, m + 1, dtype=float)
                        q_vals = (p_vals.loc[sorted_idx] * m / ranks).cummin().clip(upper=1.0)
                        df.loc[sorted_idx, "q_fdr_global"] = q_vals
                    df["fdr_crit_p_global"] = crit_p
                    df.to_csv(out_tsv, sep="\t", index=False)
            except Exception as exc:
                _log_if_present(logger, "warning", f"Failed global FDR for group temporal stats: {exc}")

    _render_group_temporal_topomaps(
        group_results=group_results,
        info=info,
        ch_names=ch_names,
        condition_labels=condition_labels,
        plots_dir=plot_dir,
        title_prefix=title_prefix,
        use_spearman=use_spearman,
        config=config,
        logger=logger,
    )

