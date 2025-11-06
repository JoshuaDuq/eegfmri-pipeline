from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


###################################################################
# Transformations
###################################################################

def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    return 0.5 * np.log((1 + np.clip(r, -0.9999, 0.9999)) / (1 - np.clip(r, -0.9999, 0.9999)))


###################################################################
# Bootstrap
###################################################################

def bias_corrected_bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, random_state: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(random_state)
    n = len(data)
    theta_hat = np.mean(data)
    bootstrap_means = np.array([np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)])
    z0 = stats.norm.ppf(np.mean(bootstrap_means < theta_hat))
    
    jackknife_means = np.array([np.mean(np.delete(data, i)) for i in range(n)])
    jackknife_mean = np.mean(jackknife_means)
    jackknife_diff = jackknife_mean - jackknife_means
    numerator = np.sum(jackknife_diff ** 3)
    denominator = 6 * (np.sum(jackknife_diff ** 2) ** 1.5)
    acceleration = numerator / denominator if denominator != 0 else 0.0
    
    alpha = 1 - ci
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)
    p_lower = np.clip(stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha))), 0.001, 0.999)
    p_upper = np.clip(stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - acceleration * (z0 + z_1_alpha))), 0.001, 0.999)
    return theta_hat, np.percentile(bootstrap_means, p_lower * 100), np.percentile(bootstrap_means, p_upper * 100)


def bootstrap_auc(y_true: np.ndarray, y_scores: np.ndarray, n_bootstraps: int = 1000, 
                  ci: float = 0.95, random_state: int = 42) -> Tuple[float, float, float]:
    auc = roc_auc_score(y_true, y_scores)
    rng = np.random.RandomState(random_state)
    bootstrapped_aucs = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            bootstrapped_auc = roc_auc_score(y_true[indices], y_scores[indices])
            bootstrapped_aucs.append(bootstrapped_auc)
        except ValueError:
            continue
    
    if not bootstrapped_aucs:
        return auc, np.nan, np.nan
    
    alpha_half = (1 - ci) / 2
    ci_lower = np.percentile(bootstrapped_aucs, alpha_half * 100)
    ci_upper = np.percentile(bootstrapped_aucs, (1 - alpha_half) * 100)
    return auc, ci_lower, ci_upper


###################################################################
# Correlations
###################################################################

def safe_spearman(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    valid = x.notna() & y.notna()
    x_valid = x[valid]
    y_valid = y[valid]
    
    if len(x_valid) < 3 or x_valid.nunique() < 2 or y_valid.nunique() < 2:
        return {'rho': float('nan'), 'p': float('nan'), 'n': int(len(x_valid))}
    
    rho, pval = spearmanr(x_valid, y_valid)
    return {'rho': float(rho), 'p': float(pval), 'n': int(len(x_valid))}


###################################################################
# Effect Sizes
###################################################################

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    
    if len(group1_clean) == 0 or len(group2_clean) == 0:
        return float('nan')
    
    mean1 = float(np.mean(group1_clean))
    mean2 = float(np.mean(group2_clean))
    
    n1, n2 = len(group1_clean), len(group2_clean)
    var1 = float(np.var(group1_clean, ddof=1)) if n1 > 1 else 0.0
    var2 = float(np.var(group2_clean, ddof=1)) if n2 > 1 else 0.0
    
    df = n1 + n2 - 2
    if df <= 0:
        return float('nan')
    
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    pooled_sd = np.sqrt(pooled_var)
    
    if pooled_sd <= 0:
        return float('nan')
    
    return (mean1 - mean2) / pooled_sd


###################################################################
# Hypothesis Testing
###################################################################

def alternative_symbol(alternative: str) -> str:
    return {'greater': '>', 'less': '<', 'two-sided': '!='}.get(alternative, '!=')


def one_sample_t_test(data: np.ndarray, popmean: float = 0, alternative: str = 'greater') -> Tuple[float, float, float]:
    result = stats.ttest_1samp(data, popmean, alternative=alternative)
    return result.statistic, result.pvalue, len(data) - 1

