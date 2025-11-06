from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.masking import apply_mask
from nilearn.image import resample_to_img
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore", message="The following unexpected columns in events data will be ignored", category=UserWarning, module="nilearn")
warnings.filterwarnings("ignore", message="Matrix is singular at working precision, regularizing", category=UserWarning, module="nilearn")

import sys
from pathlib import Path as PathType
sys.path.insert(0, str(PathType(__file__).parent.parent))
from utils.io_utils import load_confounds, get_bold_info


###################################################################
# Confounds
###################################################################

def extract_confounds(confounds_path: Path, motion_24_cols: List[str], motion_outlier_prefix: str, n_volumes: int) -> Tuple[pd.DataFrame, Dict]:
    confounds = pd.read_csv(confounds_path, sep='\t')
    if len(confounds) != n_volumes:
        raise ValueError(f"Confounds rows ({len(confounds)}) != BOLD volumes ({n_volumes})")
    if missing := [col for col in motion_24_cols if col not in confounds.columns]:
        raise ValueError(f"Missing motion columns: {missing}")
    outlier_cols = sorted([col for col in confounds.columns if col.startswith(motion_outlier_prefix)])
    confounds_clean = confounds[motion_24_cols + outlier_cols].copy()
    has_nans = confounds_clean.isna().any().any()
    if has_nans:
        confounds_clean.fillna(0.0, inplace=True)
    return confounds_clean, {
        'motion_outliers_found': len(outlier_cols),
        'filled_nans': has_nans,
        'output_rows': n_volumes,
        'output_columns': 24 + len(outlier_cols)
    }


###################################################################
# Design Matrix
###################################################################

def create_design_matrix(events_df: pd.DataFrame, n_volumes: int, tr: float, temp_labels: List[str], nuisance_events: List[str],
                        hrf_model: str, high_pass_hz: float, confounds_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    events_filtered = events_df[events_df['trial_type'].isin(set(temp_labels) | set(nuisance_events))].copy()
    if events_filtered.empty:
        raise ValueError('No relevant events found')
    return make_first_level_design_matrix(
        frame_times=np.arange(n_volumes) * tr,
        events=events_filtered[['onset', 'duration', 'trial_type']],
        hrf_model=hrf_model,
        drift_model='cosine',
        high_pass=high_pass_hz,
        add_regs=confounds_df.values if confounds_df is not None else None,
        add_reg_names=list(confounds_df.columns) if confounds_df is not None else None,
        min_onset=0
    )


def compute_design_correlations(design_matrix: pd.DataFrame, temp_labels: List[str], confound_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corr_matrix = design_matrix.corr()
    task_cols = [col for col in design_matrix.columns if any(temp in col for temp in temp_labels)]
    if not task_cols:
        return pd.DataFrame(), corr_matrix, pd.DataFrame()

    task_corr = corr_matrix.loc[task_cols, task_cols]
    corr_summary = [
        {
            'regressor_1': col1,
            'regressor_2': col2,
            'correlation': float(task_corr.loc[col1, col2]),
            'abs_correlation': abs(float(task_corr.loc[col1, col2])),
            'flag_high': abs(float(task_corr.loc[col1, col2])) > 0.2
        }
        for i, col1 in enumerate(task_cols) for col2 in task_cols[i + 1:]
    ]

    task_confound_rows = []
    if confound_columns:
        confound_cols = [col for col in confound_columns if col in design_matrix.columns]
        for task_col in task_cols:
            series = corr_matrix.loc[task_col, confound_cols].dropna()
            if series.empty:
                continue
            top_confound = series.abs().idxmax()
            r_val = float(series[top_confound])
            task_confound_rows.append({
                'task_regressor': task_col,
                'confound_regressor': top_confound,
                'correlation': r_val,
                'abs_correlation': abs(r_val),
                'flag_high': abs(r_val) > 0.2
            })

    return pd.DataFrame(corr_summary), corr_matrix, pd.DataFrame(task_confound_rows)


def _compute_standardized_condition_number(design_matrix: pd.DataFrame) -> Optional[float]:
    try:
        values = design_matrix.values
        stds = values.std(axis=0, ddof=1)
        nonzero_std_cols = stds > 0
        if not np.any(nonzero_std_cols):
            return None
        standardized = (values[:, nonzero_std_cols] - values[:, nonzero_std_cols].mean(axis=0)) / stds[nonzero_std_cols]
        singular_values = np.linalg.svd(standardized, compute_uv=False)
        if singular_values.size == 0 or singular_values[-1] == 0:
            return None
        return float(singular_values[0] / singular_values[-1])
    except np.linalg.LinAlgError:
        return None


def validate_design(design_matrix: pd.DataFrame, temp_labels: List[str], nuisance_events: List[str], events_df: pd.DataFrame = None) -> Dict:
    validation = {'valid': True, 'warnings': [], 'temp_regressors_found': [], 'nuisance_regressors_found': []}
    temp_in_events = set(events_df['trial_type'].unique()) & set(temp_labels) if events_df is not None else set(temp_labels)

    for temp in temp_labels:
        if any(temp in col for col in design_matrix.columns):
            validation['temp_regressors_found'].append(temp)
        elif temp in temp_in_events:
            validation['warnings'].append(f"Missing temperature: {temp}")
            validation['valid'] = False

    for nuisance in nuisance_events:
        if any(nuisance in col for col in design_matrix.columns):
            validation['nuisance_regressors_found'].append(nuisance)
        else:
            validation['warnings'].append(f"Missing nuisance: {nuisance}")
            validation['valid'] = False

    return validation


def select_task_columns(design_matrix: pd.DataFrame, temp_labels: Iterable[str]) -> List[str]:
    temp_set = set(temp_labels)
    return [col for col in design_matrix.columns if col in temp_set]


def compute_condition_number(design_matrix: pd.DataFrame) -> Optional[float]:
    values = design_matrix.values
    nonzero_std_cols = values.std(axis=0) > 0
    if not np.any(nonzero_std_cols):
        return None
    try:
        singular_values = np.linalg.svd(values[:, nonzero_std_cols], compute_uv=False)
        if singular_values.size == 0 or singular_values[-1] == 0:
            return None
        return float(singular_values[0] / singular_values[-1])
    except np.linalg.LinAlgError:
        return None


###################################################################
# Events Preparation
###################################################################

def prepare_events(events_path: Path, temp_labels: Iterable[str], nuisance_events: Iterable[str],
                  onset_offset_sec: float = 0.0, plateau_duration_sec: Optional[float] = None) -> pd.DataFrame:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    events = pd.read_csv(events_path, sep="\t")
    if missing := {"onset", "duration", "trial_type"} - set(events.columns):
        raise ValueError(f"Missing columns: {missing}")
    filtered = events[events["trial_type"].isin(set(temp_labels) | set(nuisance_events))].copy()
    if filtered.empty:
        raise ValueError("No events found")
    filtered["duration"] = pd.to_numeric(filtered["duration"], errors="coerce")
    if filtered["duration"].isna().any():
        raise ValueError("Missing durations")

    temp_mask = filtered["trial_type"].isin(temp_labels)
    if onset_offset_sec != 0.0 and temp_mask.any():
        filtered.loc[temp_mask, "onset"] += onset_offset_sec

    if plateau_duration_sec is not None and temp_mask.any():
        filtered.loc[temp_mask, "duration"] = plateau_duration_sec

    return filtered


###################################################################
# BOLD Conversion
###################################################################

def convert_bold_to_percent_signal(bold_img: nib.Nifti1Image, tr: float, baseline_window_sec: float = 120.0,
                                   min_baseline: float = 1e-3) -> nib.Nifti1Image:
    data = bold_img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD image, got shape {data.shape}")
    n_volumes = data.shape[-1]
    if n_volumes < 2:
        raise ValueError("Need at least 2 volumes to compute percent signal change")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if baseline_window_sec and baseline_window_sec > 0:
        window = max(int(round(baseline_window_sec / tr)), 1)
        window = min(window, n_volumes)
        if window > 1:
            baseline = uniform_filter1d(data, size=window, axis=-1, mode="nearest")
        else:
            baseline = data.mean(axis=-1, keepdims=True)
    else:
        baseline = data.mean(axis=-1, keepdims=True)

    baseline = np.nan_to_num(baseline, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = np.where(np.abs(baseline) < min_baseline, min_baseline, baseline)

    percent_signal = 100.0 * (data - baseline) / baseline
    percent_signal = np.nan_to_num(percent_signal, nan=0.0, posinf=0.0, neginf=0.0)

    header = bold_img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(percent_signal.astype(np.float32), bold_img.affine, header)


###################################################################
# GLM Fitting
###################################################################

def fit_glm(bold_path: Path, mask_path: Path, events: pd.DataFrame, confounds: pd.DataFrame, tr: float,
           hrf_model: str, high_pass_hz: float, convert_to_percent_signal: bool = False,
           percent_signal_window_sec: float = 120.0, percent_signal_min_baseline: float = 1e-3) -> Tuple[FirstLevelModel, Dict]:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    bold_img, mask_img = nib.load(str(bold_path)), nib.load(str(mask_path))
    if convert_to_percent_signal:
        bold_img = convert_bold_to_percent_signal(bold_img, tr, baseline_window_sec=percent_signal_window_sec, min_baseline=percent_signal_min_baseline)
    n_volumes = bold_img.shape[3] if bold_img.ndim == 4 else 1
    if len(confounds) != n_volumes:
        raise ValueError(f"Confounds mismatch: {len(confounds)} != {n_volumes}")
    glm = FirstLevelModel(t_r=tr, hrf_model=hrf_model, drift_model="cosine", high_pass=high_pass_hz,
                         mask_img=mask_img, smoothing_fwhm=None, minimize_memory=False, n_jobs=1, verbose=0)
    glm.fit(bold_img, events=events, confounds=confounds)
    design = glm.design_matrices_[0]
    design_rank = int(np.linalg.matrix_rank(design.values))
    dof = n_volumes - design_rank
    return glm, {
        "n_volumes": n_volumes,
        "n_regressors": design.shape[1],
        "n_events": len(events),
        "n_confounds": confounds.shape[1],
        "degrees_of_freedom": dof,
        "design_matrix_rank": design_rank,
        "design_columns": list(design.columns),
        "percent_signal_enabled": bool(convert_to_percent_signal),
        "percent_signal_window_sec": float(percent_signal_window_sec),
        "percent_signal_min_baseline": float(percent_signal_min_baseline),
    }


def compute_regressor_snr(glm: FirstLevelModel, bold_path: Path, mask_path: Path,
                         ignore_columns: Iterable[str] = ()) -> pd.DataFrame:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    try:
        bold_img = nib.load(str(bold_path))
        masker = getattr(glm, "masker_", None)
        if masker:
            bold_data = masker.transform(bold_img)
        else:
            mask_img = nib.load(str(mask_path))
            bold_data = apply_mask(bold_img, mask_img)
        bold_data = np.asarray(bold_data, dtype=np.float64)

        if bold_data.ndim != 2 or bold_data.shape[0] < 2:
            raise ValueError("Invalid BOLD data shape")

        n_timepoints, n_voxels = bold_data.shape
        bold_mean = bold_data.mean(axis=0)
        bold_std = bold_data.std(axis=0, ddof=1)
        valid_voxels = bold_std > 0
        if not np.any(valid_voxels):
            raise ValueError("All voxels zero variance")

        bold_z = (bold_data[:, valid_voxels] - bold_mean[valid_voxels]) / bold_std[valid_voxels]

        design = glm.design_matrices_[0]
        ignore_set = set(ignore_columns)
        results = []

        for column in design.columns:
            if column.startswith("drift") or column == "constant" or column in ignore_set:
                continue

            reg = design[column].values.astype(np.float64)
            reg_std = reg.std(ddof=1)
            if reg_std <= 0:
                continue

            reg_z = (reg - reg.mean()) / reg_std
            corrs = np.clip(reg_z @ bold_z / (n_timepoints - 1), -1.0, 1.0)
            abs_corr = np.abs(corrs)

            results.append({
                "regressor": column,
                "median_abs_correlation": float(np.median(abs_corr)),
                "mean_abs_correlation": float(abs_corr.mean()),
                "p90_abs_correlation": float(np.percentile(abs_corr, 90)),
                "max_abs_correlation": float(abs_corr.max()),
                "n_voxels_used": int(abs_corr.size)
            })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values("median_abs_correlation", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


###################################################################
# GLM Summarization
###################################################################

def summarize_glm_design(glm: FirstLevelModel, temp_labels: List[str], subject: str, run_num: int,
                        subject_dir: Path, qc_dir: Path, confound_columns: Optional[List[str]]) -> Dict[str, Optional[float]]:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    design = glm.design_matrices_[0].copy()

    design_path = subject_dir / f"run-{run_num:02d}_design_matrix_glm.tsv"
    design_path.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(design_path, sep="\t", index=False, float_format="%.6f")

    corr_matrix = design.corr()
    corr_matrix_path = subject_dir / f"run-{run_num:02d}_design_corr_matrix_glm.tsv"
    corr_matrix.to_csv(corr_matrix_path, sep="\t", float_format="%.6f")

    task_cols = select_task_columns(design, temp_labels)
    confound_cols = [col for col in (confound_columns or []) if col in design.columns]

    task_corr_summary_path = None
    task_confound_summary_path = None
    max_task_abs_corr = None
    max_task_confound_abs_corr = None

    if len(task_cols) >= 2:
        task_corr = corr_matrix.loc[task_cols, task_cols]
        rows = []
        for i, col1 in enumerate(task_cols):
            for col2 in task_cols[i + 1:]:
                r_val = float(task_corr.loc[col1, col2])
                rows.append({
                    "regressor_1": col1,
                    "regressor_2": col2,
                    "correlation": r_val,
                    "abs_correlation": abs(r_val),
                    "flag_high": abs(r_val) > 0.2,
                })
        task_summary = pd.DataFrame(rows)
        if not task_summary.empty:
            task_corr_summary_path = qc_dir / f"{subject}_run-{run_num:02d}_design_corr_glm.tsv"
            task_corr_summary_path.parent.mkdir(parents=True, exist_ok=True)
            task_summary.to_csv(task_corr_summary_path, sep="\t", index=False, float_format="%.6f")
            max_task_abs_corr = float(task_summary["abs_correlation"].max())

    task_confound_summary = pd.DataFrame()
    if confound_cols:
        cross_rows = []
        for task_col in task_cols:
            series = corr_matrix.loc[task_col, confound_cols].dropna()
            if series.empty:
                continue
            abs_series = series.abs()
            top_confound = abs_series.idxmax()
            r_val = float(series[top_confound])
            cross_rows.append({
                "task_regressor": task_col,
                "confound_regressor": top_confound,
                "correlation": r_val,
                "abs_correlation": abs(r_val),
                "flag_high": abs(r_val) > 0.2,
            })
        task_confound_summary = pd.DataFrame(cross_rows)
        if not task_confound_summary.empty:
            task_confound_summary_path = qc_dir / f"{subject}_run-{run_num:02d}_design_task_confound_corr_glm.tsv"
            task_confound_summary_path.parent.mkdir(parents=True, exist_ok=True)
            task_confound_summary.to_csv(task_confound_summary_path, sep="\t", index=False, float_format="%.6f")
            max_task_confound_abs_corr = float(task_confound_summary["abs_correlation"].max())

    condition_number = compute_condition_number(design)
    cond_std = _compute_standardized_condition_number(design)

    return {
        "design_matrix_path": str(design_path),
        "task_correlation_summary_path": str(task_corr_summary_path) if task_corr_summary_path else None,
        "task_confound_summary_path": str(task_confound_summary_path) if task_confound_summary_path else None,
        "correlation_matrix_path": str(corr_matrix_path),
        "condition_number": condition_number,
        "condition_number_standardized": cond_std,
        "max_task_abs_correlation": max_task_abs_corr,
        "max_task_confound_abs_correlation": max_task_confound_abs_corr,
        "n_task_regressors": len(task_cols),
        "n_confound_regressors": len(confound_cols),
    }


###################################################################
# Beta Maps
###################################################################

def extract_beta_maps(glm: FirstLevelModel, temp_labels: Iterable[str], output_dir: Path, run_num: int) -> Dict[str, Dict[str, Path]]:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    outputs = {}
    design = glm.design_matrices_[0]
    design_cols = list(design.columns)

    for temp in temp_labels:
        matching_cols = [col for col in design_cols if col == temp]
        if not matching_cols:
            continue
        if len(matching_cols) > 1:
            raise ValueError(f"Multiple columns match {temp}: {matching_cols}")
        col_name = matching_cols[0]
        col_idx = design_cols.index(col_name)
        contrast = np.zeros(len(design_cols), dtype=float)
        contrast[col_idx] = 1.0

        beta_img = glm.compute_contrast(contrast, output_type="effect_size")
        variance_img = glm.compute_contrast(contrast, output_type="effect_variance")

        beta_path = output_dir / f"run-{run_num:02d}_beta_{temp}.nii.gz"
        variance_path = output_dir / f"run-{run_num:02d}_beta_{temp}_variance.nii.gz"

        nib.save(beta_img, str(beta_path))
        nib.save(variance_img, str(variance_path))

        outputs[temp] = {"beta": beta_path, "variance": variance_path, "design_column": col_name}

    return outputs


###################################################################
# Combine Betas
###################################################################

def combine_betas_fixed_effects(beta_paths: List[Path], variance_paths: Optional[List[Path]] = None,
                                variance_method: str = 'uniform', analysis_mask_img: Optional[nib.Nifti1Image] = None) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image, np.ndarray]:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    if len(beta_paths) == 0:
        raise ValueError("No beta maps to combine")

    beta_imgs = [nib.load(str(path)) for path in beta_paths]
    ref_img = beta_imgs[0]
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()

    beta_arrays = []
    ref_shape = None
    for idx, img in enumerate(beta_imgs):
        if not np.allclose(img.affine, ref_affine):
            raise ValueError(f"Beta map {idx + 1} does not align with reference affine")
        data = img.get_fdata()
        if data.ndim == 4 and data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        if data.ndim != 3:
            raise ValueError(f"Beta map {beta_paths[idx].name} has unexpected shape {data.shape}; expected 3D or 4D with singleton time dimension")
        if idx == 0:
            ref_shape = data.shape
        elif data.shape != ref_shape:
            raise ValueError(f"Beta map {beta_paths[idx].name} shape {data.shape} does not match reference shape {ref_shape}")
        beta_arrays.append(data)

    ref_header.set_data_shape(ref_shape)
    beta_data = np.stack(beta_arrays, axis=-1).astype(np.float64)
    beta_data = np.where(np.isfinite(beta_data), beta_data, np.nan)

    if analysis_mask_img is not None:
        mask_img = analysis_mask_img
        if mask_img.shape != ref_shape or not np.allclose(mask_img.affine, ref_affine):
            mask_img = resample_to_img(mask_img, ref_img, interpolation='nearest')
        mask_data = mask_img.get_fdata()
        if mask_data.ndim == 4 and mask_data.shape[-1] == 1:
            mask_data = np.squeeze(mask_data, axis=-1)
        mask = mask_data > 0
    else:
        mask = np.any(np.isfinite(beta_data), axis=-1)

    mask = mask.astype(bool)
    if not np.any(mask):
        raise ValueError("Analysis mask is empty")

    beta_data = np.where(mask[..., None], beta_data, np.nan)

    variance_data = None
    if variance_method in {"glm", "uniform"}:
        if not variance_paths or len(variance_paths) != len(beta_paths):
            raise RuntimeError(f"{variance_method.upper()} variance weighting requires run-level variance maps for each beta.")
        variance_arrays = []
        for path_var in variance_paths:
            img = nib.load(str(path_var))
            if img.shape != ref_img.shape or not np.allclose(img.affine, ref_affine):
                img = resample_to_img(img, ref_img, interpolation='linear')
            data = img.get_fdata()
            if data.ndim == 4 and data.shape[-1] == 1:
                data = np.squeeze(data, axis=-1)
            if data.shape != ref_shape:
                raise ValueError(f"Variance map {path_var.name} shape {data.shape} does not match reference shape {ref_shape}")
            variance_arrays.append(data.astype(np.float64))
        variance_data = np.stack(variance_arrays, axis=-1)
        variance_data = np.where(np.isfinite(variance_data), variance_data, np.nan)
        variance_data = np.where(mask[..., None], variance_data, np.nan)
        invalid_mask = ~np.isfinite(variance_data) | (variance_data <= 0)
        if np.any(invalid_mask):
            frac_invalid = float(np.sum(invalid_mask) / invalid_mask.size)
        variance_data = np.where(invalid_mask, np.nan, variance_data)

    if variance_method == 'glm':
        if variance_data is None:
            raise RuntimeError("Variance data not available for GLM weighting.")

        valid_var = variance_data[np.isfinite(variance_data) & (variance_data > 0)]
        if valid_var.size > 0:
            var_floor = float(np.percentile(valid_var, 1.0))
            variance_data_floored = np.where((variance_data > 0) & (variance_data < var_floor), var_floor, variance_data)
        else:
            variance_data_floored = variance_data

        precision = np.where(np.isfinite(variance_data_floored) & (variance_data_floored > 0), 1.0 / variance_data_floored, np.nan)
        weighted_sum = np.nansum(beta_data * precision, axis=-1)
        precision_sum = np.nansum(precision, axis=-1)
        combined_data = np.zeros(ref_shape, dtype=np.float64)
        
        frac_invalid = np.sum(~np.isfinite(precision_sum)) / precision_sum.size
        combined_var = np.full(ref_shape, np.nan, dtype=np.float64)
        valid = precision_sum > 0
        combined_data[valid] = weighted_sum[valid] / precision_sum[valid]
        combined_var[valid] = 1.0 / precision_sum[valid]
        n_runs_contrib = np.sum(np.isfinite(precision) & (precision > 0), axis=-1, dtype=np.float64)
    elif variance_method == 'uniform':
        if variance_data is None:
            raise RuntimeError("Variance data not available for uniform weighting.")
        valid = np.isfinite(beta_data) & np.isfinite(variance_data)
        beta_valid = np.where(valid, beta_data, np.nan)
        variance_valid = np.where(valid, variance_data, np.nan)
        with np.errstate(invalid='ignore'):
            combined_data = np.nanmean(beta_valid, axis=-1)
        n_runs_contrib = np.sum(valid, axis=-1).astype(np.float64)
        with np.errstate(invalid='ignore'):
            sum_of_variances = np.nansum(variance_valid, axis=-1)
            combined_var = np.full(ref_shape, np.nan, dtype=np.float64)
            positive = n_runs_contrib > 0
            combined_var[positive] = sum_of_variances[positive] / (n_runs_contrib[positive]**2)
    else:
        raise ValueError(f"Unknown variance method: {variance_method}")

    combined_data = np.where(mask, np.nan_to_num(combined_data, nan=0.0), 0.0)
    combined_var = np.where(mask, np.nan_to_num(combined_var, nan=0.0), 0.0)
    n_runs_map = np.where(mask, n_runs_contrib, 0.0).astype(np.float32)

    combined_beta = nib.Nifti1Image(combined_data.astype(np.float32), ref_affine, ref_header.copy())
    combined_variance = nib.Nifti1Image(combined_var.astype(np.float32), ref_affine, ref_header.copy())
    n_runs_img = nib.Nifti1Image(n_runs_map.astype(np.float32), ref_affine, ref_header.copy())

    return combined_beta, combined_variance, n_runs_img, mask


def validate_output(beta_img: nib.Nifti1Image, analysis_mask: Optional[np.ndarray] = None) -> Dict:
    data, valid = beta_img.get_fdata(), np.isfinite(beta_img.get_fdata())
    if analysis_mask is not None:
        valid_in_mask = valid & analysis_mask.astype(bool)
        pct_finite = 100.0 * np.sum(valid_in_mask) / np.sum(analysis_mask)
    else:
        pct_finite = 100.0 * np.sum(valid) / data.size
    return {"pct_finite": float(pct_finite), "mean_beta": float(data[valid].mean()) if np.any(valid) else np.nan}

