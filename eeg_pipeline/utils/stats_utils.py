from __future__ import annotations

from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import mne
from scipy import stats
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test


###################################################################
# FDR (Benjamini–Hochberg)
###################################################################

def fdr_bh(pvals: Iterable[float], alpha: float = 0.05) -> np.ndarray:
    pvals_arr = np.asarray(list(pvals), dtype=float)
    qvals = np.full_like(pvals_arr, np.nan, dtype=float)

    valid_mask = np.isfinite(pvals_arr)
    if not np.any(valid_mask):
        return qvals

    pv = pvals_arr[valid_mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size

    denom = np.arange(1, n + 1, dtype=float)
    adjusted = ranked * n / denom
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted

    qvals[valid_mask] = restored
    return qvals


def fdr_bh_reject(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), np.nan
    
    valid_mask = np.isfinite(p)
    if not np.any(valid_mask):
        return np.zeros_like(p, dtype=bool), np.nan
    
    p_valid = p[valid_mask]
    order = np.argsort(p_valid)
    ranked = np.arange(1, len(p_valid) + 1)
    thresh = (ranked / len(p_valid)) * alpha
    passed = p_valid[order] <= thresh
    
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool), np.nan
    
    k_max = np.max(np.where(passed)[0])
    crit = float(p_valid[order][k_max])
    
    reject = np.zeros_like(p, dtype=bool)
    reject[valid_mask] = p_valid <= crit
    
    return reject, crit


###################################################################
# EEG Adjacency and Cluster Utilities
###################################################################

def get_eeg_adjacency(info, restrict_picks=None):
    from scipy.spatial import distance_matrix
    
    eeg_picks_all = mne.pick_types(info, eeg=True, exclude=[])
    if len(eeg_picks_all) == 0:
        return None, None, None
    
    if restrict_picks is not None:
        restrict_picks = np.asarray(restrict_picks, dtype=int)
        eeg_picks = np.array([p for p in eeg_picks_all if p in set(restrict_picks)], dtype=int)
    else:
        eeg_picks = np.asarray(eeg_picks_all, dtype=int)
    
    if eeg_picks.size == 0:
        return None, None, None
    
    info_eeg = mne.pick_info(info, sel=eeg_picks.tolist())
    
    try:
        adjacency, ch_names = mne.channels.find_ch_adjacency(info_eeg, ch_type="eeg")
    except Exception as e:
        print(f"Warning: Delaunay adjacency failed ({e.__class__.__name__}), using distance-based fallback")
        
        pos = np.array([ch['loc'][:3] for ch in info_eeg['chs']])
        if np.all(np.isnan(pos)) or np.allclose(pos, 0):
            print("Warning: Invalid channel positions, returning None adjacency")
            return None, eeg_picks, info_eeg
        
        dists = distance_matrix(pos, pos)
        k_neighbors = min(3, len(pos) - 1)
        adjacency = np.zeros((len(pos), len(pos)), dtype=bool)
        
        for i in range(len(pos)):
            nearest = np.argsort(dists[i])[1:k_neighbors+1]
            adjacency[i, nearest] = True
            adjacency[nearest, i] = True
        
        from scipy import sparse
        adjacency = sparse.csr_matrix(adjacency)
        ch_names = [ch['ch_name'] for ch in info_eeg['chs']]
        
    return adjacency, eeg_picks, info_eeg


def build_full_mask_from_eeg(sig_mask_eeg: np.ndarray, n_ch_total: int, eeg_picks: np.ndarray) -> np.ndarray:
    full = np.zeros(n_ch_total, dtype=bool)
    full[eeg_picks] = sig_mask_eeg.astype(bool)
    return full


def cluster_mask_from_clusters(
    clusters: list,
    p_vals: np.ndarray,
    n_features: int,
    alpha: float,
) -> np.ndarray:
    mask = np.zeros(n_features, dtype=bool)
    for cl, p in zip(clusters, p_vals):
        if float(p) <= float(alpha):
            if isinstance(cl, np.ndarray) and cl.dtype == bool and cl.shape[0] == n_features:
                mask |= cl
            else:
                mask[np.asarray(cl)] = True
    return mask


def cluster_test_two_sample_arrays(
    Xa: np.ndarray,
    Xb: np.ndarray,
    info: mne.Info,
    alpha: float = 0.05,
    paired: bool = False,
    n_permutations: int = 1024,
    restrict_picks: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    adjacency, eeg_picks, info_eeg = get_eeg_adjacency(info, restrict_picks=restrict_picks)
    if eeg_picks is None or info_eeg is None:
        return None, None, None, None
    
    A = np.asarray(Xa)[:, eeg_picks]
    B = np.asarray(Xb)[:, eeg_picks]
    
    if paired and A.shape[0] == B.shape[0]:
        Xd = A - B
        T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
            Xd,
            n_permutations=int(n_permutations),
            adjacency=adjacency,
            tail=0,
            out_type="mask",
            n_jobs=1,
        )
    else:
        T_obs, clusters, p_values, _ = permutation_cluster_test(
            [A, B],
            n_permutations=int(n_permutations),
            adjacency=adjacency,
            tail=0,
            out_type="mask",
            n_jobs=1,
        )
    
    sig_eeg = cluster_mask_from_clusters(clusters, p_values, n_features=A.shape[1], alpha=alpha)
    sig_full = build_full_mask_from_eeg(sig_eeg, n_ch_total=len(info["ch_names"]), eeg_picks=eeg_picks)
    
    p_of_largest: Optional[float] = None
    k_of_largest: Optional[int] = None
    mass_of_largest: Optional[float] = None
    
    if len(clusters) > 0:
        masses = []
        for cl in clusters:
            if isinstance(cl, np.ndarray) and cl.dtype == bool and cl.shape[0] == T_obs.shape[0]:
                idx = np.where(cl)[0]
            else:
                idx = np.asarray(cl)
            if idx.size == 0:
                masses.append(0.0)
            else:
                masses.append(float(np.nansum(np.abs(T_obs[idx]))))
        
        if len(masses) > 0:
            k = int(np.nanargmax(np.asarray(masses)))
            cl_k = clusters[k]
            if isinstance(cl_k, np.ndarray) and cl_k.dtype == bool and cl_k.shape[0] == T_obs.shape[0]:
                idx_k = np.where(cl_k)[0]
            else:
                idx_k = np.asarray(cl_k)
            p_of_largest = float(p_values[k]) if np.isfinite(p_values[k]) else None
            k_of_largest = int(idx_k.size)
            mass_of_largest = float(np.nansum(np.abs(T_obs[idx_k]))) if idx_k.size > 0 else 0.0
    
    return sig_full, p_of_largest, k_of_largest, mass_of_largest


def cluster_test_epochs(
    tfr_epochs: "mne.time_frequency.EpochsTFR",
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    paired: bool = False,
    alpha: float = 0.05,
    n_permutations: int = 1024,
    restrict_picks: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    fmask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    tmask = (times >= float(tmin)) & (times < float(tmax))
    if fmask.sum() == 0 or tmask.sum() == 0:
        return None, None, None, None
    data = np.asarray(tfr_epochs.data)[:, :, fmask, :][:, :, :, tmask]
    X = data.mean(axis=(2, 3))
    
    Xa = np.asarray(X)[np.asarray(mask_a, dtype=bool), :]
    Xb = np.asarray(X)[np.asarray(mask_b, dtype=bool), :]
    if Xa.shape[0] < 2 or Xb.shape[0] < 2:
        return None, None, None, None
    
    return cluster_test_two_sample_arrays(
        Xa, Xb, tfr_epochs.info,
        alpha=alpha, paired=paired, n_permutations=n_permutations,
        restrict_picks=restrict_picks,
    )


def fdr_bh_mask(p_vals: np.ndarray, alpha: float = 0.05) -> Optional[np.ndarray]:
    p_vals = np.asarray(p_vals, dtype=float)
    if p_vals.ndim != 1 or p_vals.size == 0:
        return None
    finite = np.isfinite(p_vals)
    if not np.any(finite):
        return np.zeros_like(p_vals, dtype=bool)
    rej, _ = fdr_bh_reject(p_vals[finite], alpha=float(alpha))
    mask = np.zeros_like(p_vals, dtype=bool)
    mask[finite] = rej.astype(bool)
    return mask


def fdr_bh_values(p_vals: np.ndarray, alpha: float = 0.05) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p_vals = np.asarray(p_vals, dtype=float)
    if p_vals.ndim != 1 or p_vals.size == 0:
        return None, None
    finite = np.isfinite(p_vals)
    if not np.any(finite):
        return np.zeros_like(p_vals, dtype=bool), np.full_like(p_vals, np.nan, dtype=float)
    rej, _ = fdr_bh_reject(p_vals[finite], alpha=float(alpha))
    q_vals = fdr_bh(p_vals[finite], alpha=float(alpha))
    reject_mask = np.zeros_like(p_vals, dtype=bool)
    q = np.full_like(p_vals, np.nan, dtype=float)
    reject_mask[finite] = rej.astype(bool)
    q[finite] = q_vals.astype(float)
    return reject_mask, q


###################################################################
# Correlation Utilities and Fisher Aggregation
###################################################################

def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, p.size + 1, dtype=float)
    p_sorted = p[order]
    q_raw = p_sorted * p.size / ranks
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def fisher_aggregate(rs: List[float]) -> Tuple[float, float, float, int]:
    vals = np.array([r for r in rs if np.isfinite(r)])
    vals = np.clip(vals, -0.999999, 0.999999)
    n = vals.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    z = np.arctanh(vals)
    mean_z = _safe_float(np.mean(z))
    sd_z = _safe_float(np.std(z, ddof=1))
    se = sd_z / np.sqrt(n) if sd_z > 0 else np.nan
    if np.isnan(se) or se == 0:
        return _safe_float(np.tanh(mean_z)), np.nan, np.nan, n
    tcrit = _safe_float(stats.t.ppf(0.975, df=n - 1))
    ci_low_z = mean_z - tcrit * se
    ci_high_z = mean_z + tcrit * se
    return _safe_float(np.tanh(mean_z)), _safe_float(np.tanh(ci_low_z)), _safe_float(np.tanh(ci_high_z)), n


def partial_corr_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[float, float, int]:
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    if len(df) < 5 or df["y"].nunique() <= 1:
        return np.nan, np.nan, 0
    
    if method == "spearman":
        xr, yr = stats.rankdata(df["x"].to_numpy()), stats.rankdata(df["y"].to_numpy())
        Zr = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
        Xd = np.column_stack([np.ones(len(df)), Zr])
        bx, by = np.linalg.lstsq(Xd, xr, rcond=None)[0], np.linalg.lstsq(Xd, yr, rcond=None)[0]
        x_res, y_res = xr - Xd.dot(bx), yr - Xd.dot(by)
    else:
        Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
        bx, by = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0], np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
        x_res, y_res = df["x"].to_numpy() - Xd.dot(bx), df["y"].to_numpy() - Xd.dot(by)
    
    r_p, p_p = stats.pearsonr(x_res, y_res)
    return _safe_float(r_p), _safe_float(p_p), int(len(df))


def partial_residuals_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[pd.Series, pd.Series, int]:
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    if len(df) < 5 or df["y"].nunique() <= 1:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    if method == "spearman":
        xr, yr = stats.rankdata(df["x"].to_numpy()), stats.rankdata(df["y"].to_numpy())
        Zr = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
        Xd = np.column_stack([np.ones(len(df)), Zr])
        bx, by = np.linalg.lstsq(Xd, xr, rcond=None)[0], np.linalg.lstsq(Xd, yr, rcond=None)[0]
        x_res, y_res = xr - Xd.dot(bx), yr - Xd.dot(by)
    else:
        Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
        bx, by = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0], np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
        x_res, y_res = df["x"].to_numpy() - Xd.dot(bx), df["y"].to_numpy() - Xd.dot(by)
    
    return pd.Series(x_res, index=df.index), pd.Series(y_res, index=df.index), int(len(df))


def compute_partial_corr(x: pd.Series, y: pd.Series, Z: Optional[pd.DataFrame], method: str, *, logger=None, context: str = "") -> Tuple[float, float, int]:
    if Z is None or Z.empty:
        return np.nan, np.nan, 0
    if len(Z.columns) == 0:
        if logger:
            logger.warning(f"{context}: Z has no columns; skipping partial correlation" if context else "Z has no columns; skipping partial correlation")
        return np.nan, np.nan, 0
    
    _ensure_aligned_lengths_for_partial(x, y, Z, context=context, strict=True, logger=logger)
    data = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
    df = data.dropna()
    
    if logger and len(data) > len(df):
        dropped = len(data) - len(df)
        prefix = f"{context}: " if context else ""
        logger.warning(f"{prefix}partial correlation dropped {dropped} rows due to missing data (kept {len(df)}/{len(data)})")
    
    if len(df) < 5 or df["y"].nunique() <= 1:
        return np.nan, np.nan, 0
    
    return partial_corr_xy_given_Z(df["x"], df["y"], df[Z.columns], method)


def compute_partial_residuals(x: pd.Series, y: pd.Series, Z: Optional[pd.DataFrame], method: str, *, logger=None, context: str = "") -> Tuple[pd.Series, pd.Series, int]:
    if Z is None or Z.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    data = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
    df = data.dropna()
    
    if logger and len(data) > len(df):
        dropped = len(data) - len(df)
        prefix = f"{context}: " if context else ""
        logger.warning(f"{prefix}partial residuals dropped {dropped} rows due to missing data (kept {len(df)}/{len(data)})")
    
    if len(df) < 5 or df["y"].nunique() <= 1:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    return partial_residuals_xy_given_Z(df["x"], df["y"], df[Z.columns], method)


def fisher_ci(r: float, n: int) -> Tuple[float, float]:
    if not np.isfinite(r) or n < 4:
        return np.nan, np.nan
    r = _safe_float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    return _safe_float(np.tanh(z_lo)), _safe_float(np.tanh(z_hi))


def perm_pval_simple(x: pd.Series, y: pd.Series, method: str, n_perm: int, rng: np.random.Generator) -> float:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    
    obs, _ = stats.spearmanr(df["x"], df["y"], nan_policy="omit")
    ge = 1
    y_vals = df["y"].to_numpy()
    
    for _ in range(n_perm):
        y_pi = y_vals[rng.permutation(len(y_vals))]
        rp, _ = stats.spearmanr(df["x"], y_pi, nan_policy="omit")
        if np.abs(rp) >= np.abs(obs) - 1e-12:
            ge += 1
    return ge / (n_perm + 1)


def perm_pval_partial_freedman_lane(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str, n_perm: int, rng: np.random.Generator) -> float:
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    
    if method == "spearman":
        xr, yr = stats.rankdata(df["x"].to_numpy()), stats.rankdata(df["y"].to_numpy())
        Zr = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
        Xd = np.column_stack([np.ones(len(df)), Zr])
        bx, by = np.linalg.lstsq(Xd, xr, rcond=None)[0], np.linalg.lstsq(Xd, yr, rcond=None)[0]
        rx, ry = xr - Xd.dot(bx), yr - Xd.dot(by)
    else:
        Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
        bx, by = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0], np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
        rx, ry = df["x"].to_numpy() - Xd.dot(bx), df["y"].to_numpy() - Xd.dot(by)
    
    obs, _ = stats.pearsonr(rx, ry)
    ge = 1
    
    for _ in range(n_perm):
        ry_pi = ry[rng.permutation(len(ry))]
        rp, _ = stats.pearsonr(rx, ry_pi)
        if np.abs(rp) >= np.abs(obs) - 1e-12:
            ge += 1
    return ge / (n_perm + 1)


def bootstrap_corr_ci(
    x: pd.Series,
    y: pd.Series,
    method: str,
    n_boot: int = 1000,
    rng: Optional[np.random.Generator] = None,
    *,
    min_samples: int = 5,
    ci_percentiles: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, float]:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if n_boot <= 0 or len(df) < max(3, int(min_samples)):
        return np.nan, np.nan
    rng = rng or np.random.default_rng(42)
    x_vals = df["x"].to_numpy()
    y_vals = df["y"].to_numpy()
    boots: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(df), size=len(df))
        if method == "pearson":
            r, _ = stats.pearsonr(x_vals[idx], y_vals[idx])
        else:
            r, _ = stats.spearmanr(x_vals[idx], y_vals[idx], nan_policy="omit")
        if np.isfinite(r):
            boots.append(float(r))
    if not boots:
        return np.nan, np.nan
    lo, hi = np.percentile(boots, list(ci_percentiles))
    return float(lo), float(hi)


def compute_group_corr_stats(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    method: str,
    *,
    strategy: str,
    n_cluster_boot: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, int, int, Tuple[float, float], float]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for xi, yi in zip(x_lists, y_lists):
        xi = np.asarray(xi)
        yi = np.asarray(yi)
        if len(xi) != len(yi):
            raise ValueError(
                f"Group correlation requested with mismatched trial counts "
                f"(len(x)={len(xi)}, len(y)={len(yi)})."
            )
        mask = np.isfinite(xi) & np.isfinite(yi)
        if int(mask.sum()) >= 5:
            pairs.append((xi[mask], yi[mask]))
    if not pairs:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan

    rng = rng or np.random.default_rng(42)

    def _corr(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[float, float]:
        if method.lower() == "pearson":
            r, p = stats.pearsonr(x_arr, y_arr)
        else:
            r, p = stats.spearmanr(x_arr, y_arr, nan_policy="omit")
        return _safe_float(r), _safe_float(p)

    if strategy in {"pooled_trials", "within_subject_centered", "within_subject_zscored"}:
        valid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        subj_r: List[float] = []
        for xi, yi in pairs:
            x_use = xi.copy()
            y_use = yi.copy()
            if strategy == "within_subject_centered":
                x_use -= np.nanmean(x_use)
                y_use -= np.nanmean(y_use)
            elif strategy == "within_subject_zscored":
                sx = np.nanstd(x_use, ddof=1)
                sy = np.nanstd(y_use, ddof=1)
                if sx <= 0 or sy <= 0:
                    continue
                x_use = (x_use - np.nanmean(x_use)) / sx
                y_use = (y_use - np.nanmean(y_use)) / sy
            valid_pairs.append((x_use, y_use))
            r_i, _ = _corr(x_use, y_use)
            if np.isfinite(r_i):
                subj_r.append(_safe_float(np.clip(r_i, -0.999999, 0.999999)))

        if not valid_pairs:
            return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan

        X = np.concatenate([xp for xp, _ in valid_pairs])
        Y = np.concatenate([yp for _xp, yp in valid_pairs])
        r_obs, p_obs = _corr(X, Y)
        n_trials = len(X)
        n_subjects = len(valid_pairs)

        p_group = np.nan
        if len(subj_r) >= 2:
            z_vals = np.arctanh(np.array(subj_r))
            res = stats.ttest_1samp(z_vals, popmean=0.0, nan_policy="omit")
            p_group = _get_ttest_pvalue(res)

        ci = (np.nan, np.nan)
        if n_cluster_boot and n_subjects >= 2:
            boots: List[float] = []
            idx = np.arange(n_subjects)
            for _ in range(int(n_cluster_boot)):
                pick = rng.choice(idx, size=n_subjects, replace=True)
                bx: List[np.ndarray] = []
                by: List[np.ndarray] = []
                for j in pick:
                    xj, yj = valid_pairs[j]
                    if strategy == "within_subject_centered":
                        xj = xj - np.nanmean(xj)
                        yj = yj - np.nanmean(yj)
                    elif strategy == "within_subject_zscored":
                        sx = np.nanstd(xj, ddof=1)
                        sy = np.nanstd(yj, ddof=1)
                        if sx <= 0 or sy <= 0:
                            continue
                        xj = (xj - np.nanmean(xj)) / sx
                        yj = (yj - np.nanmean(yj)) / sy
                    bx.append(xj)
                    by.append(yj)
                if not bx or not by:
                    continue
                Xb = np.concatenate(bx)
                Yb = np.concatenate(by)
                rb, _ = _corr(Xb, Yb)
                if np.isfinite(rb):
                    boots.append(rb)
            if boots:
                ci = (
                    _safe_float(np.percentile(boots, 2.5)),
                    _safe_float(np.percentile(boots, 97.5)),
                )
        return _safe_float(r_obs), _safe_float(p_group), n_trials, n_subjects, ci, _safe_float(p_obs)

    # fisher_by_subject strategy
    r_subj: List[float] = []
    for xi, yi in pairs:
        r_i, _ = _corr(xi, yi)
        if np.isfinite(r_i):
            r_subj.append(_safe_float(np.clip(r_i, -0.999999, 0.999999)))
    if not r_subj:
        return np.nan, np.nan, 0, 0, (np.nan, np.nan), np.nan

    z = np.arctanh(np.array(r_subj))
    r_group = _safe_float(np.tanh(np.nanmean(z)))
    if len(z) >= 2:
        res = stats.ttest_1samp(z, popmean=0.0, nan_policy="omit")
        p_group = _get_ttest_pvalue(res)
    else:
        p_group = np.nan

    n_trials = int(sum(len(xi) for xi, _ in pairs))
    ci = (np.nan, np.nan)
    if n_cluster_boot and len(r_subj) >= 2:
        boots = []
        idx = np.arange(len(r_subj))
        for _ in range(int(n_cluster_boot)):
            pick = rng.choice(idx, size=len(r_subj), replace=True)
            zb = np.mean(z[pick])
            boots.append(_safe_float(np.tanh(zb)))
        if boots:
            ci = (
                _safe_float(np.percentile(boots, 2.5)),
                _safe_float(np.percentile(boots, 97.5)),
            )
    return r_group, _safe_float(p_group), n_trials, len(r_subj), ci, np.nan


###################################################################
# Internal helpers
###################################################################

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _get_ttest_pvalue(ttest_result) -> float:
    if ttest_result is None or not hasattr(ttest_result, "pvalue"):
        return np.nan
    return _safe_float(ttest_result.pvalue)


def _ensure_aligned_lengths_for_partial(x, y, Z, *, context: str = "", strict: bool = True, logger=None) -> None:
    if Z is None:
        return
    non_null = [obj for obj in (x, y, Z) if obj is not None]
    if len(non_null) < 2:
        return
    lengths = {len(obj) for obj in non_null}
    if len(lengths) > 1:
        msg = f"{context}: Length mismatch detected: {[len(obj) for obj in non_null]}"
        if strict:
            raise ValueError(msg)
        if logger:
            logger.warning(msg)


__all__ = [
    # FDR
    "fdr_bh",
    "fdr_bh_reject",
    "fdr_bh_mask",
    "fdr_bh_values",
    # EEG cluster utilities
    "get_eeg_adjacency",
    "build_full_mask_from_eeg",
    "cluster_mask_from_clusters",
    "cluster_test_two_sample_arrays",
    "cluster_test_epochs",
    # Correlation utilities
    "bh_adjust",
    "fisher_aggregate",
    "partial_corr_xy_given_Z",
    "partial_residuals_xy_given_Z",
    "compute_partial_corr",
    "compute_partial_residuals",
    "fisher_ci",
    "perm_pval_simple",
    "perm_pval_partial_freedman_lane",
    "bootstrap_corr_ci",
    "compute_group_corr_stats",
    "_safe_float",
    "_get_ttest_pvalue",
]









