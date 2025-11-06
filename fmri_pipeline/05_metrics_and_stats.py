#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from utils import load_config, get_log_function, PipelinePaths
from analysis.metrics import (
    compute_dose_response_metrics,
    compute_auc_warm_vs_pain,
    compute_discrimination_metrics,
    process_subject_metrics,
)
from analysis.group_stats import (
    load_subject_metrics,
    compute_group_statistics,
    create_summary_table,
)
from utils.stats_utils import (
    fisher_z_transform,
    bias_corrected_bootstrap_ci,
    bootstrap_auc,
    one_sample_t_test,
    alternative_symbol,
)


SCRIPT_NAME = Path(__file__).stem
log, _ = get_log_function(SCRIPT_NAME)

PRIMARY_ENDPOINTS = ['slope_BR_temp', 'r_BR_temp', 'r_BR_VAS', 'auc_pain', 'forced_choice_accuracy']


###################################################################
# Main
###################################################################

def main():
    parser = argparse.ArgumentParser(description='Subject metrics and group statistics')
    parser.add_argument('--config', default='utils/config.yaml', help='Configuration file')
    parser.add_argument('--subject', default=None, help='Process specific subject (for metrics only)')
    parser.add_argument('--output-dir', default='outputs', help='Output directory (default: outputs)')
    parser.add_argument('--qc-dir', default='qc', help='QC directory (default: qc)')
    parser.add_argument('--signatures', nargs='+', default=None, help='Signatures to process (e.g., nps siips1)')
    parser.add_argument('--skip-metrics', action='store_true', help='Skip subject-level metrics computation')
    parser.add_argument('--skip-stats', action='store_true', help='Skip group-level statistics')
    parser.add_argument('--alternative', choices=['greater', 'less', 'two-sided'], default='greater', help='Alternative hypothesis for group tests (default: greater)')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        log(f"Config: {e}", "ERROR")
        return 1

    if args.signatures:
        signatures = args.signatures
    else:
        signatures = config.get('enabled_signatures', ['nps'])

    paths = PipelinePaths.from_config(config, output_dir=args.output_dir, qc_dir=args.qc_dir)
    paths.ensure_core_roots()

    qc_dir = PipelinePaths.ensure_dir(paths.qc_stage_dir("05_metrics_and_stats"))

    all_success = True

    if not args.skip_metrics:
        log("Metrics")
        if args.subject:
            subjects = [args.subject]
        else:
            subjects = config['subjects']

        for sig_name in signatures:
            scores_dir = Path(paths.signature_scores_dir(sig_name))

            if not scores_dir.exists():
                all_success = False
                continue

            all_results = []

            for subject in subjects:
                try:
                    results = process_subject_metrics(subject, scores_dir, sig_name)

                    if results is None:
                        all_success = False
                        continue

                    results['signature'] = sig_name
                    all_results.append(results)

                except Exception as e:
                    log(f"{subject}: {e}", "ERROR")
                    all_success = False
                    continue

            if len(all_results) > 0:
                summary_rows = []
                for result in all_results:
                    summary_rows.append({
                        'subject': result['subject'],
                        'signature': result['signature'],
                        'slope_BR_temp': result.get('slope_BR_temp', np.nan),
                        'r_BR_temp': result.get('r_BR_temp', np.nan),
                        'p_BR_temp': result.get('p_BR_temp', np.nan),
                        'r_BR_VAS': result.get('r_BR_VAS', np.nan),
                        'p_BR_VAS': result.get('p_BR_VAS', np.nan),
                        'auc_pain': result.get('auc_pain', np.nan),
                        'auc_warm_vs_pain': result.get('auc_warm_vs_pain', np.nan),
                        'n_levels': result.get('n_levels', 0),
                        'n_trials': result.get('n_trials', 0),
                        'has_trial_data': result.get('has_trial_data', False)
                    })

                summary_df = pd.DataFrame(summary_rows)

                summary_path = qc_dir / f"{sig_name}_subject_metrics_summary.tsv"
                summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.6f')

    if not args.skip_stats:
        log("Stats")
        subjects = config['subjects']
        group_root = paths.group_root
        group_root.mkdir(parents=True, exist_ok=True)

        for sig_name in signatures:
            scores_dir = Path(paths.signature_scores_dir(sig_name))

            if not scores_dir.exists():
                all_success = False
                continue

            try:
                df = load_subject_metrics(scores_dir, subjects, sig_name)
            except Exception as e:
                log(f"{sig_name}: {e}", "ERROR")
                all_success = False
                continue

            missing_subjects = set(subjects) - set(df['subject'].values)

            try:
                results = compute_group_statistics(df, args.alternative)
            except Exception as e:
                log(f"{sig_name}: {e}", "ERROR")
                all_success = False
                continue

            summary_df = create_summary_table(results)

            sig_output_dir = group_root / sig_name
            sig_output_dir.mkdir(parents=True, exist_ok=True)

            summary_path = sig_output_dir / "group_stats.tsv"
            summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.6f')

            diagnostics = {
                'signature': sig_name,
                'n_subjects_total': len(subjects),
                'n_subjects_analyzed': len(df),
                'missing_subjects': list(missing_subjects),
                'subjects_analyzed': df['subject'].tolist(),
                'alternative': args.alternative
            }

            diagnostics_path = sig_output_dir / "group_diagnostics.json"
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

            verbose_path = sig_output_dir / "group_stats_verbose.json"
            verbose_path.write_text(json.dumps(results, indent=2))

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

