from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img


###################################################################
# LSS Events
###################################################################

def create_lss_events_for_trial(events_path: Path, target_trial_idx: int, trial_info: pd.DataFrame,
                               nuisance_events: List[str]) -> pd.DataFrame:
    full_events = pd.read_csv(events_path, sep='\t')
    target_trial = trial_info.iloc[target_trial_idx]
    lss_events = []

    lss_events.append({
        'onset': target_trial['onset'],
        'duration': target_trial['duration'],
        'trial_type': 'target_trial'
    })

    other_trials = trial_info.drop(index=trial_info.index[target_trial_idx])
    for _, other_trial in other_trials.iterrows():
        lss_events.append({
            'onset': other_trial['onset'],
            'duration': other_trial['duration'],
            'trial_type': 'other_trials'
        })

    for nuisance_label in nuisance_events:
        nuisance_rows = full_events[full_events['trial_type'] == nuisance_label]
        for _, row in nuisance_rows.iterrows():
            lss_events.append({
                'onset': row['onset'],
                'duration': row['duration'],
                'trial_type': nuisance_label
            })

    return pd.DataFrame(lss_events).sort_values('onset').reset_index(drop=True)


def extract_trial_info(events_path: Path, run_num: int, global_trial_offset: int, onset_offset_sec: float = 0.0,
                      plateau_duration_sec: Optional[float] = None) -> pd.DataFrame:
    events = pd.read_csv(events_path, sep='\t')
    temp_trials = events[events['trial_type'].str.startswith('temp')].copy()

    if onset_offset_sec != 0.0:
        temp_trials['onset'] = temp_trials['onset'] + onset_offset_sec

    if plateau_duration_sec is not None:
        temp_trials['duration'] = plateau_duration_sec

    temp_trials['trial_idx_global'] = range(global_trial_offset, global_trial_offset + len(temp_trials))
    temp_trials['trial_idx_run'] = range(len(temp_trials))
    temp_trials['trial_regressor'] = temp_trials['trial_idx_global'].apply(lambda x: f'trial_{x:03d}')

    return temp_trials


###################################################################
# LSS GLM Fitting
###################################################################

def fit_lss_glm_for_trial(bold_img: nib.Nifti1Image, mask_img: nib.Nifti1Image, lss_events: pd.DataFrame,
                          confounds: pd.DataFrame, tr: float, hrf_model: str, high_pass_hz: float) -> Tuple[FirstLevelModel, float, float]:
    if bold_img.ndim == 4:
        n_volumes = bold_img.shape[3]
    else:
        n_volumes = 1

    if len(confounds) != n_volumes:
        raise ValueError(f"Confounds rows ({len(confounds)}) != BOLD volumes ({n_volumes})")

    nyquist_freq = 0.5 / tr
    if high_pass_hz <= 0 or high_pass_hz >= nyquist_freq:
        raise ValueError(f'high_pass_hz must be between 0 and Nyquist frequency ({nyquist_freq:.4f} Hz)')

    glm = FirstLevelModel(
        t_r=tr,
        hrf_model=hrf_model,
        drift_model='cosine',
        high_pass=high_pass_hz,
        mask_img=mask_img,
        smoothing_fwhm=None,
        minimize_memory=False,
        n_jobs=1,
        verbose=0
    )

    glm.fit(bold_img, events=lss_events, confounds=confounds)

    design_matrix = glm.design_matrices_[0]
    design_values = design_matrix.to_numpy(dtype=np.float64, copy=True)

    try:
        condition_number_raw = float(np.linalg.cond(design_values))
    except np.linalg.LinAlgError:
        condition_number_raw = float('inf')

    col_std = design_values.std(axis=0, ddof=1)
    keep = col_std > 0
    if np.any(keep):
        Xz = (design_values[:, keep] - design_values[:, keep].mean(axis=0)) / col_std[keep]
        try:
            condition_number_z = float(np.linalg.cond(Xz))
        except np.linalg.LinAlgError:
            condition_number_z = float('inf')
    else:
        condition_number_z = float('inf')

    return glm, condition_number_raw, condition_number_z


###################################################################
# Extract Target Trial Beta
###################################################################

def extract_target_trial_beta(glm: FirstLevelModel, signature_weights: nib.Nifti1Image) -> np.ndarray:
    design_cols = glm.design_matrices_[0].columns.tolist()
    sig_data = np.squeeze(signature_weights.get_fdata())
    sig_mask = np.isfinite(sig_data) & (sig_data != 0)

    target_col = next((col for col in design_cols if col == 'target_trial' or col.startswith('target_trial_')), None)
    if target_col is None:
        raise ValueError(f"'target_trial' regressor not found in design matrix. Columns: {design_cols}")

    col_idx = design_cols.index(target_col)
    contrast_vec = np.zeros(len(design_cols), dtype=float)
    contrast_vec[col_idx] = 1.0

    beta_map = glm.compute_contrast(contrast_vec, output_type='effect_size')
    beta_resampled = resample_to_img(beta_map, signature_weights, interpolation='linear', copy=True, force_resample=True)
    beta_data = np.squeeze(beta_resampled.get_fdata())
    beta_masked = beta_data[sig_mask]

    if np.any(~np.isfinite(beta_masked)):
        beta_masked = np.nan_to_num(beta_masked, nan=0.0, posinf=0.0, neginf=0.0)

    return beta_masked

