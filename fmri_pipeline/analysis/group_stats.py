from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import sys
from pathlib import Path as PathType
sys.path.insert(0, str(PathType(__file__).parent.parent))
from utils.stats_utils import bias_corrected_bootstrap_ci, fisher_z_transform, alternative_symbol, one_sample_t_test
from utils.io_utils import load_inventory


###################################################################
# Load Subject Metrics
###################################################################

def load_subject_metrics(scores_dir: Path, subjects: List[str], signature_name: str) -> pd.DataFrame:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    metrics_list = []
    for subject in subjects:
        metrics_path = scores_dir / subject / "subject_metrics.tsv"
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path, sep='\t')
                if 'signature' not in df.columns:
                    df['signature'] = signature_name
                metrics_list.append(df)
            except Exception:
                pass
    if not metrics_list:
        raise ValueError("No metrics loaded")
    return pd.concat(metrics_list, ignore_index=True)


###################################################################
# Group Statistics
###################################################################

def _compute_t_test_metric(data: np.ndarray, test_description: str,
                          alternative: str, log) -> Dict:
    if len(data) < 3:
        return {'n': int(len(data)), 'error': 'insufficient_data'}
    
    mean_val, ci_lower, ci_upper = bias_corrected_bootstrap_ci(data)
    t_stat, p_val, df_val = one_sample_t_test(data, popmean=0, alternative=alternative)
    symbol = alternative_symbol(alternative)
    
    return {
        'n': int(len(data)),
        'mean': float(mean_val),
        'std': float(np.std(data, ddof=1)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'df': float(df_val),
        'test': test_description.replace('{symbol}', symbol),
        'alternative': alternative
    }


def _compute_fisher_z_metric(data: np.ndarray, test_description: str,
                            alternative: str, log) -> Dict:
    if len(data) < 3:
        return {'n': int(len(data)), 'error': 'insufficient_data'}
    
    mean_r, ci_lower, ci_upper = bias_corrected_bootstrap_ci(data)
    z_transformed = fisher_z_transform(data)
    t_stat, p_val, df_val = one_sample_t_test(z_transformed, popmean=0, alternative=alternative)
    symbol = alternative_symbol(alternative)
    
    log(f"n={len(data)}, mean r={mean_r:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
    log(f"t({df_val})={t_stat:.3f}, p={p_val:.4f} (Fisher z)")
    
    return {
        'n': int(len(data)),
        'mean': float(mean_r),
        'std': float(np.std(data, ddof=1)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'df': float(df_val),
        'test': test_description.replace('{symbol}', symbol),
        'alternative': alternative
    }


def _compute_wilcoxon_metric(data: np.ndarray, test_description: str,
                             null_value: float, alternative: str, log) -> Dict:
    if len(data) < 3:
        log(f"Insufficient data (n={len(data)})", "WARNING")
        return {'n': int(len(data)), 'error': 'insufficient_data'}
    
    mean_val, ci_lower, ci_upper = bias_corrected_bootstrap_ci(data)
    centered = data - null_value
    symbol = alternative_symbol(alternative)
    
    if np.allclose(centered, 0):
        w_stat = 0.0
        p_val = 1.0
        n_zero = len(data)
    else:
        try:
            w_stat, p_val = stats.wilcoxon(centered, zero_method='wilcox', alternative=alternative,
                                          correction=False, mode='auto')
        except ValueError:
            w_stat, p_val = stats.wilcoxon(centered, zero_method='pratt', alternative=alternative,
                                          correction=False)
        n_zero = int(np.sum(np.isclose(centered, 0)))
    
    log(f"n={len(data)}, mean={mean_val:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
    log(f"Wilcoxon W={w_stat:.3f}, p={p_val:.4f} (vs {null_value})")
    
    return {
        'n': int(len(data)),
        'mean': float(mean_val),
        'std': float(np.std(data, ddof=1)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'wilcoxon_statistic': float(w_stat),
        'p_value': float(p_val),
        'n_zero_differences': int(n_zero),
        'test': test_description.replace('{symbol}', symbol),
        'alternative': alternative
    }


def compute_group_statistics(df: pd.DataFrame, alternative: str) -> Dict:
    from utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    PRIMARY_ENDPOINTS = ['slope_BR_temp', 'r_BR_temp', 'r_BR_VAS', 'auc_pain', 'forced_choice_accuracy']
    
    results = {
        'n_subjects_total': len(df),
        'alternative': alternative,
        'metrics': {}
    }

    slope_data = df['slope_BR_temp'].dropna().values
    results['metrics']['slope_BR_temp'] = _compute_t_test_metric(
        slope_data, 'one-sample t-test (H1: slope {symbol} 0)', alternative, log
    )

    r_temp_data = df['r_BR_temp'].dropna().values
    results['metrics']['r_BR_temp'] = _compute_fisher_z_metric(
        r_temp_data, 'one-sample t-test on Fisher-z(r) (H1: z {symbol} 0)', alternative, log
    )

    r_vas_data = df['r_BR_VAS'].dropna().values
    results['metrics']['r_BR_VAS'] = _compute_fisher_z_metric(
        r_vas_data, 'one-sample t-test on Fisher-z(r) (H1: z {symbol} 0)', alternative, log
    )

    auc_data = df['auc_pain'].dropna().values
    results['metrics']['auc_pain'] = _compute_wilcoxon_metric(
        auc_data, 'Wilcoxon signed-rank (H1: AUC {symbol} 0.5)', 0.5, alternative, log
    )

    fc_data = df['forced_choice_accuracy'].dropna().values
    results['metrics']['forced_choice_acc'] = _compute_wilcoxon_metric(
        fc_data, 'Wilcoxon signed-rank (H1: AUC {symbol} 0.5)', 0.5, alternative, log
    )

    metric_name_map = {
        'forced_choice_accuracy': 'forced_choice_acc'
    }
    
    p_values = []
    metric_names = []
    for metric in PRIMARY_ENDPOINTS:
        result_key = metric_name_map.get(metric, metric)
        if result_key in results['metrics'] and 'p_value' in results['metrics'][result_key]:
            p_values.append(results['metrics'][result_key]['p_value'])
            metric_names.append(result_key)

    if len(p_values) > 0:
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        for i, metric in enumerate(metric_names):
            results['metrics'][metric]['p_value_fdr'] = float(p_adjusted[i])
            results['metrics'][metric]['fdr_significant'] = bool(reject[i])
            log(f"{metric}: p={p_values[i]:.4f}, p_FDR={p_adjusted[i]:.4f}, sig={'Yes' if reject[i] else 'No'}")
    else:
        log("No valid p-values for FDR correction", "WARNING")

    return results


###################################################################
# Summary Table
###################################################################

def create_summary_table(results: Dict) -> pd.DataFrame:
    rows = []
    for name, data in results['metrics'].items():
        if 'error' not in data:
            rows.append({
                'metric': name,
                'n': data['n'],
                'mean': data['mean'],
                'std': data['std'],
                'ci_lower': data['ci_lower'],
                'ci_upper': data['ci_upper'],
                't_statistic': data.get('t_statistic', np.nan),
                'df': data.get('df', np.nan),
                'p_value': data['p_value'],
                'p_value_fdr': data.get('p_value_fdr', np.nan),
                'fdr_significant': data.get('fdr_significant', False)
            })
    return pd.DataFrame(rows)

