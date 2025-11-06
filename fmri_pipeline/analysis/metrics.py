from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path as PathType
sys.path.insert(0, str(PathType(__file__).parent.parent))
from utils.stats_utils import bootstrap_auc
from utils.io_utils import sanitize_for_json


###################################################################
# Dose Response Metrics
###################################################################

def compute_dose_response_metrics(level_df: pd.DataFrame, score_col: str = 'br_score') -> Dict:
    metrics = {
        'n_levels': len(level_df),
        'slope_BR_temp': np.nan,
        'r_BR_temp': np.nan,
        'p_BR_temp': np.nan,
        'r_BR_VAS': np.nan,
        'p_BR_VAS': np.nan,
        'warnings': []
    }
    
    valid_temp = level_df.dropna(subset=['temp_celsius', score_col])
    if len(valid_temp) < 3:
        metrics['warnings'].append(f"Insufficient temp data (n={len(valid_temp)})")
        return metrics

    try:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            valid_temp['temp_celsius'], valid_temp[score_col]
        )
        metrics.update({
            'slope_BR_temp': float(slope),
            'intercept_BR_temp': float(intercept),
            'r_BR_temp': float(r_value),
            'p_BR_temp': float(p_value)
        })
    except Exception as e:
        metrics['warnings'].append(f"BR~temp failed: {e}")

    valid_vas = level_df.dropna(subset=['mean_vas', score_col])
    if len(valid_vas) >= 3:
        try:
            r_vas, p_vas = stats.pearsonr(valid_vas['mean_vas'], valid_vas[score_col])
            metrics.update({'r_BR_VAS': float(r_vas), 'p_BR_VAS': float(p_vas)})
        except Exception as e:
            metrics['warnings'].append(f"BR~VAS failed: {e}")
    else:
        metrics['warnings'].append(f"Insufficient VAS (n={len(valid_vas)})")

    return metrics


###################################################################
# AUC Metrics
###################################################################

def compute_auc_warm_vs_pain(trial_df: pd.DataFrame, warm_threshold: float = 46.0, pain_threshold: float = 47.0, score_col: str = 'br_score') -> Tuple[float, int]:
    warm_trials = trial_df[trial_df['temp_celsius'] < warm_threshold].dropna(subset=[score_col])
    pain_trials = trial_df[trial_df['temp_celsius'] >= pain_threshold].dropna(subset=[score_col])
    if len(warm_trials) == 0 or len(pain_trials) == 0:
        return np.nan, 0
    scores = np.concatenate([warm_trials[score_col].values, pain_trials[score_col].values])
    labels = np.concatenate([np.zeros(len(warm_trials)), np.ones(len(pain_trials))])
    if len(np.unique(labels)) < 2:
        return np.nan, len(warm_trials) * len(pain_trials)
    return float(roc_auc_score(labels, scores)), int(len(warm_trials) * len(pain_trials))


###################################################################
# Discrimination Metrics
###################################################################

def compute_discrimination_metrics(trial_df: pd.DataFrame, pain_threshold: float = 100.0, score_col: str = 'br_score') -> Dict:
    metrics = {
        'n_trials': len(trial_df),
        'auc_pain': np.nan,
        'auc_ci_lower': np.nan,
        'auc_ci_upper': np.nan,
        'auc_warm_vs_pain': np.nan,
        'auc_warm_vs_pain_n_pairs': 0,
        'warnings': []
    }

    valid_df = trial_df.dropna(subset=[score_col])
    if len(valid_df) < 10:
        metrics['warnings'].append(f"Insufficient trial data (n={len(valid_df)})")
        return metrics

    pain_labels = _extract_pain_labels(valid_df, pain_threshold)
    if pain_labels is None:
        metrics['warnings'].append("No pain labels available (pain_binary or vas_rating)")
        return metrics

        if len(np.unique(pain_labels)) == 2:
            try:
                sig_scores = valid_df[score_col].values
                auc, ci_lower, ci_upper = bootstrap_auc(pain_labels, sig_scores)
                metrics['auc_pain'] = float(auc)
                metrics['auc_ci_lower'] = float(ci_lower)
                metrics['auc_ci_upper'] = float(ci_upper)
            except Exception as e:
                metrics['warnings'].append(f"Failed to compute AUC: {e}")
        else:
            metrics['warnings'].append("Only one class present in pain labels")

    if 'temp_celsius' in valid_df.columns:
        try:
            auc_wvp, n_pairs = compute_auc_warm_vs_pain(valid_df, score_col=score_col)
            metrics['auc_warm_vs_pain'] = float(auc_wvp) if not np.isnan(auc_wvp) else np.nan
            metrics['auc_warm_vs_pain_n_pairs'] = int(n_pairs)
            metrics['classification_metric'] = 'auc'
        except Exception as e:
            metrics['warnings'].append(f"Failed to compute warm-vs-pain AUC: {e}")
    else:
        metrics['warnings'].append("No temperature column for warm-vs-pain classification")

    return metrics


def _extract_pain_labels(df: pd.DataFrame, pain_threshold: float) -> Optional[np.ndarray]:
    if 'pain_binary' in df.columns:
        return df['pain_binary'].values
    elif 'vas_rating' in df.columns:
        return (df['vas_rating'] > pain_threshold).astype(int).values
    return None


###################################################################
# Subject Metrics Processing
###################################################################

def process_subject_metrics(subject: str, scores_dir: Path, signature_name: str) -> Optional[Dict]:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    subject_dir = scores_dir / subject
    score_col = f"{signature_name}_score"
    results = {
        'subject': subject,
        'has_level_data': False,
        'has_trial_data': False
    }

    level_path = subject_dir / f"level_{signature_name}.tsv"
    if not level_path.exists():
        return None

    try:
        level_df = pd.read_csv(level_path, sep='\t')
        results['has_level_data'] = True
    except Exception as e:
        log(f"{subject}: {e}", "ERROR")
        return None

    _load_metadata(subject_dir, results, log)
    dose_metrics = _process_dose_response(level_df, score_col, results, log)
    discrim_metrics = _process_trial_data(subject_dir, signature_name, score_col, results, log)
    _collect_warnings(dose_metrics, discrim_metrics, results, log)
    _save_metrics(subject_dir, results, log)

    return results


def _load_metadata(subject_dir: Path, results: Dict, log) -> None:
    metadata_path = subject_dir / "scoring_metadata.json"
    if not metadata_path.exists():
        return
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if 'validation' in metadata and 'forced_choice_accuracy' in metadata['validation']:
            results['forced_choice_accuracy'] = metadata['validation']['forced_choice_accuracy']
    except Exception:
        pass


def _process_dose_response(level_df: pd.DataFrame, score_col: str, results: Dict, log) -> Dict:
    dose_metrics = compute_dose_response_metrics(level_df, score_col=score_col)
    results.update(dose_metrics)
    return dose_metrics


def _process_trial_data(subject_dir: Path, signature_name: str, score_col: str, results: Dict, log) -> Optional[Dict]:
    trial_path = subject_dir / f"trial_{signature_name}.tsv"
    if not trial_path.exists():
        return None

    try:
        trial_df = pd.read_csv(trial_path, sep='\t')
        results['has_trial_data'] = True
        discrim_metrics = compute_discrimination_metrics(trial_df, score_col=score_col)
        results.update(discrim_metrics)
        return discrim_metrics
    except Exception:
        results['has_trial_data'] = False
        return None


def _collect_warnings(dose_metrics: Dict, discrim_metrics: Optional[Dict], results: Dict, log) -> None:
    all_warnings = []
    if 'warnings' in dose_metrics and dose_metrics['warnings']:
        all_warnings.extend(dose_metrics['warnings'])
    if results.get('has_trial_data') and discrim_metrics and 'warnings' in discrim_metrics and discrim_metrics['warnings']:
        all_warnings.extend(discrim_metrics['warnings'])

    if all_warnings:
        results['notes'] = '; '.join(all_warnings)
    else:
        results['notes'] = ''




def _save_metrics(subject_dir: Path, results: Dict, log) -> None:
    metrics_row = {
        'subject': results['subject'],
        'slope_BR_temp': results.get('slope_BR_temp', np.nan),
        'r_BR_temp': results.get('r_BR_temp', np.nan),
        'p_BR_temp': results.get('p_BR_temp', np.nan),
        'r_BR_VAS': results.get('r_BR_VAS', np.nan),
        'p_BR_VAS': results.get('p_BR_VAS', np.nan),
        'auc_pain': results.get('auc_pain', np.nan),
        'auc_ci_lower': results.get('auc_ci_lower', np.nan),
        'auc_ci_upper': results.get('auc_ci_upper', np.nan),
        'auc_warm_vs_pain': results.get('auc_warm_vs_pain', np.nan),
        'auc_warm_vs_pain_n_pairs': results.get('auc_warm_vs_pain_n_pairs', 0),
        'forced_choice_accuracy': results.get('forced_choice_accuracy', np.nan),
        'n_levels': results.get('n_levels', 0),
        'n_trials': results.get('n_trials', 0),
        'notes': results.get('notes', '')
    }

    metrics_tsv_path = subject_dir / "subject_metrics.tsv"
    pd.DataFrame([metrics_row]).to_csv(metrics_tsv_path, sep='\t', index=False, float_format='%.6f')

    metrics_json_path = subject_dir / "subject_metrics.json"
    sanitized_results = sanitize_for_json(results)
    metrics_json_path.write_text(json.dumps(sanitized_results, indent=2))

