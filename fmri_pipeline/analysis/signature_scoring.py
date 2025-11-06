from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import nibabel as nib

import sys
from pathlib import Path as PathType
sys.path.insert(0, str(PathType(__file__).parent.parent))
from utils.image_utils import voxel_volume


###################################################################
# Scale Factor
###################################################################

def determine_scale_factor(beta_img: nib.Nifti1Image, weights_img: nib.Nifti1Image, original_beta_volume: Optional[float] = None) -> float:
    weights_vol = voxel_volume(weights_img)
    if weights_vol is None or weights_vol <= 0:
        return 1.0
    if original_beta_volume is not None:
        beta_vol = original_beta_volume
    else:
        beta_vol = voxel_volume(beta_img)
        if beta_vol is None:
            return 1.0
    if beta_vol <= 0:
        return 1.0
    return float(weights_vol / beta_vol)


###################################################################
# Signature Response
###################################################################

def compute_signature_response(beta_img: nib.Nifti1Image, weights_img: nib.Nifti1Image, mask: np.ndarray,
                               apply_scale_factor: bool = True, original_beta_volume_mm3: Optional[float] = None) -> Dict:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    INVALID_THRESHOLD = 0.05
    beta_data, weights_data = np.squeeze(beta_img.get_fdata()), np.squeeze(weights_img.get_fdata())
    if beta_data.shape != weights_data.shape:
        raise ValueError(f"Shape mismatch: {beta_data.shape} != {weights_data.shape}")
    beta_masked, weights_masked = beta_data[mask], weights_data[mask]
    pct_invalid = np.sum(~np.isfinite(beta_masked)) / len(beta_masked) if len(beta_masked) > 0 else 0.0
    if pct_invalid > INVALID_THRESHOLD:
        return None
    sanitized_beta = np.nan_to_num(beta_masked, nan=0.0, posinf=0.0, neginf=0.0)
    beta_mean = float(sanitized_beta.mean()) if sanitized_beta.size else 0.0
    beta_std = float(sanitized_beta.std(ddof=1)) if sanitized_beta.size > 1 else 0.0
    br_score_raw = float(np.dot(weights_masked, sanitized_beta))
    scale_factor_value = determine_scale_factor(beta_img, weights_img, original_beta_volume_mm3) if apply_scale_factor else 1.0
    br_score = br_score_raw * scale_factor_value
    scale_factor_applied = bool(apply_scale_factor and not bool(np.isclose(scale_factor_value, 1.0)))
    return {
        'br_score': br_score,
        'br_score_raw': br_score_raw,
        'scale_factor_applied': scale_factor_applied,
        'scale_factor_value': float(scale_factor_value),
        'n_voxels': int(np.sum(mask)),
        'n_beta_finite': int(np.sum(np.isfinite(beta_masked))),
        'n_beta_nonzero': int(np.sum(np.abs(sanitized_beta) > 0)),
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_min': float(sanitized_beta.min()) if sanitized_beta.size else 0.0,
        'beta_max': float(sanitized_beta.max()) if sanitized_beta.size else 0.0,
        'n_beta_nan': int(np.sum(np.isnan(beta_masked))),
        'n_beta_inf': int(np.sum(np.isinf(beta_masked))),
        'pct_voxels_invalid': float(pct_invalid * 100.0),
        'br_is_finite': bool(np.isfinite(br_score)),
        'quality_check_passed': True,
    }


def score_single_trial_beta(beta_masked: np.ndarray, signature_weights: nib.Nifti1Image, signature_voxel_volume: float,
                           apply_scale_factor: bool = True, original_beta_volume_mm3: Optional[float] = None) -> float:
    sig_data = np.squeeze(signature_weights.get_fdata())
    sig_mask = np.isfinite(sig_data) & (sig_data != 0)
    weights_masked = sig_data[sig_mask]

    score_raw = np.dot(weights_masked, beta_masked)

    if apply_scale_factor:
        if original_beta_volume_mm3 is None or original_beta_volume_mm3 <= 0:
            scale_factor = 1.0
        elif signature_voxel_volume <= 0:
            scale_factor = 1.0
        else:
            scale_factor = signature_voxel_volume / original_beta_volume_mm3
        score = score_raw * scale_factor
    else:
        score = score_raw

    return float(score)


###################################################################
# Validation Metrics
###################################################################

def compute_validation_metrics(results_df: pd.DataFrame, score_col: str = 'br_score') -> Dict:
    required_cols = ['temp_celsius', score_col]
    if not all(col in results_df.columns for col in required_cols):
        return {'forced_choice_accuracy': None, 'n_correct': 0, 'n_comparisons': 0, 'monotonic_violations': None}
    
    valid_df = results_df.dropna(subset=required_cols)
    if len(valid_df) < 2:
        return {'forced_choice_accuracy': None, 'n_correct': 0, 'n_comparisons': 0, 'monotonic_violations': None}
    
    temps = valid_df['temp_celsius'].values
    scores = valid_df[score_col].values
    n = len(temps)
    
    n_correct = 0
    violations = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            temp_diff = temps[j] - temps[i]
            score_diff = scores[j] - scores[i]
            
            if abs(temp_diff) < 1e-6:
                if abs(score_diff) < 0.01:
                    n_correct += 1
            elif temp_diff > 0:
                if score_diff > 0:
                    n_correct += 1
                else:
                    violations += 1
            else:
                if score_diff < 0:
                    n_correct += 1
                else:
                    violations += 1
    
    n_total = n * (n - 1) // 2
    return {
        'forced_choice_accuracy': (n_correct / n_total * 100.0) if n_total > 0 else None,
        'n_correct': n_correct,
        'n_comparisons': n_total,
        'monotonic_violations': violations
    }

