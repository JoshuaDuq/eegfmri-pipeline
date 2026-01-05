"""Features extraction CLI command."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
    create_progress_reporter,
    resolve_task,
)
from eeg_pipeline.pipelines.constants import (
    FEATURE_CATEGORIES,
    FREQUENCY_BANDS,
)
from eeg_pipeline.domain.features.constants import SPATIAL_MODES
from eeg_pipeline.cli.commands.base import FEATURE_VISUALIZE_CATEGORIES

FEATURE_CATEGORY_CHOICES = FEATURE_CATEGORIES + [
    category for category in FEATURE_VISUALIZE_CATEGORIES if category not in FEATURE_CATEGORIES
]

_COMPONENT_RANGE_RE = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*-\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$"
)


def _split_list_tokens(tokens: List[str]) -> List[str]:
    parts: List[str] = []
    for token in tokens:
        for chunk in re.split(r"[;,]", str(token)):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
    return parts


def _parse_pair_tokens(tokens: List[str], *, label: str) -> List[List[str]]:
    pairs: List[List[str]] = []
    for token in _split_list_tokens(tokens):
        sep = None
        for candidate in (":", "-", "/", "|"):
            if candidate in token:
                sep = candidate
                break
        if sep is None:
            raise ValueError(f"Invalid {label} pair token {token!r}; expected e.g. A:B or A-B")
        left, right = token.split(sep, 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Invalid {label} pair token {token!r}; expected e.g. A:B")
        pairs.append([left, right])
    return pairs


def _parse_erp_components(tokens: List[str]) -> List[dict]:
    components: List[dict] = []
    for token in _split_list_tokens(tokens):
        name = ""
        rest = ""
        if "=" in token:
            name, rest = token.split("=", 1)
        elif ":" in token:
            name, rest = token.split(":", 1)
        else:
            raise ValueError(
                f"Invalid ERP component token {token!r}; expected e.g. n2=0.20-0.35"
            )
        name = name.strip().lower()
        rest = rest.strip()
        if not name:
            raise ValueError(f"Invalid ERP component token {token!r}; missing name")
        m = _COMPONENT_RANGE_RE.match(rest)
        if not m:
            raise ValueError(
                f"Invalid ERP component range {rest!r}; expected start-end (seconds), e.g. 0.20-0.35"
            )
        start = float(m.group(1))
        end = float(m.group(2))
        if not (start < end):
            raise ValueError(f"Invalid ERP component range {rest!r}; expected start < end")
        components.append({"name": name, "start": start, "end": end})
    return components


def setup_features(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the features command parser."""
    parser = subparsers.add_parser(
        "features",
        help="Features analysis: extract or visualize",
        description="Features pipeline: extract features or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=FEATURE_CATEGORY_CHOICES,
        default=None,
        metavar="CATEGORY",
        help="Feature categories to process (some are compute-only or visualize-only)",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=FREQUENCY_BANDS,
        default=None,
        help="Frequency bands to compute (default: all)",
    )
    parser.add_argument(
        "--spatial",
        nargs="+",
        choices=SPATIAL_MODES,
        default=None,
        metavar="MODE",
        help="Spatial aggregation modes: roi, channels, global (default: roi, global)",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Start time in seconds for feature extraction window",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="End time in seconds for feature extraction window",
    )
    parser.add_argument(
        "--time-range",
        nargs=3,
        action="append",
        metavar=("NAME", "TMIN", "TMAX"),
        help="Define a named time range (e.g. baseline 0 1). Can be specified multiple times.",
    )
    parser.add_argument(
        "--aggregation-method",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method for spatial modes (default: mean)",
    )
    
    
    # Connectivity options
    parser.add_argument(
        "--connectivity-measures",
        nargs="+",
        choices=["wpli", "aec", "plv", "pli"],
        default=None,
        help="Connectivity measures to compute",
    )
    
    # PAC/CFC options
    parser.add_argument(
        "--pac-phase-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Phase frequency range for PAC/CFC (Hz)",
    )
    parser.add_argument(
        "--pac-amp-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Amplitude frequency range for PAC/CFC (Hz)",
    )
    
    # Aperiodic options
    parser.add_argument(
        "--aperiodic-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Frequency range for aperiodic fit (Hz)",
    )
    
    # Complexity options
    parser.add_argument(
        "--pe-order",
        type=int,
        default=None,
        help="Permutation entropy order (3-7, default: from config)",
    )

    # ERP options
    parser.add_argument("--erp-baseline", action="store_true", default=None, help="Enable baseline correction for ERP")
    parser.add_argument("--no-erp-baseline", action="store_false", dest="erp_baseline", help="Disable baseline correction for ERP")
    parser.add_argument("--erp-allow-no-baseline", action="store_true", default=None, help="Allow ERP extraction when baseline window is missing")
    parser.add_argument("--no-erp-allow-no-baseline", action="store_false", dest="erp_allow_no_baseline", help="Require baseline window when ERP baseline correction is enabled")
    parser.add_argument(
        "--erp-components",
        nargs="+",
        default=None,
        metavar="COMP",
        help="ERP component windows, e.g. n1=0.10-0.20 n2=0.20-0.35 p2=0.35-0.50",
    )

    # Burst options
    parser.add_argument("--burst-threshold", type=float, default=None, help="Z-score threshold for burst detection")
    parser.add_argument(
        "--burst-bands",
        nargs="+",
        default=None,
        metavar="BAND",
        help="Burst bands to compute, e.g. beta gamma",
    )

    # Power options
    parser.add_argument("--power-baseline-mode", choices=["logratio", "mean", "ratio", "zscore", "zlogratio"], default=None, help="Baseline normalization mode for power")
    parser.add_argument("--power-require-baseline", action="store_true", default=None, help="Require baseline for power normalization")
    parser.add_argument("--no-power-require-baseline", action="store_false", dest="power_require_baseline", help="Allow raw log power without baseline")

    # Spectral options
    parser.add_argument("--spectral-edge-percentile", type=float, default=None, help="Percentile for spectral edge frequency (0-1)")
    parser.add_argument(
        "--ratio-pairs",
        nargs="+",
        default=None,
        metavar="PAIR",
        help="Band power ratio pairs, e.g. theta:beta theta:alpha alpha:beta",
    )
    parser.add_argument(
        "--asymmetry-channel-pairs",
        nargs="+",
        default=None,
        metavar="PAIR",
        help="Channel pairs for asymmetry, e.g. F3:F4 C3:C4",
    )

    # Connectivity options (extend)
    parser.add_argument("--conn-output-level", choices=["full", "global_only"], default=None, help="Connectivity output level")
    parser.add_argument("--conn-graph-metrics", action="store_true", default=None, help="Enable graph metrics for connectivity")
    parser.add_argument("--no-conn-graph-metrics", action="store_false", dest="conn_graph_metrics", help="Disable graph metrics for connectivity")
    parser.add_argument("--conn-aec-mode", choices=["orth", "sym", "none"], default=None, help="AEC orthogonalization mode")

    # TFR options
    parser.add_argument("--tfr-freq-min", type=float, default=None, help="Minimum frequency for TFR (Hz)")
    parser.add_argument("--tfr-freq-max", type=float, default=None, help="Maximum frequency for TFR (Hz)")
    parser.add_argument("--tfr-n-freqs", type=int, default=None, help="Number of frequencies for TFR")
    parser.add_argument("--tfr-min-cycles", type=float, default=None, help="Minimum number of cycles for Morlet wavelets")
    parser.add_argument("--tfr-n-cycles-factor", type=float, default=None, help="Cycles factor (freq/factor) for Morlet wavelets")
    parser.add_argument("--tfr-decim", type=int, default=None, help="Decimation factor for TFR")
    parser.add_argument("--tfr-workers", type=int, default=None, help="Number of parallel workers for TFR computation")
    
    # New Advanced Options
    
    parser.add_argument("--aperiodic-peak-z", type=float, default=None, help="Peak rejection Z-threshold for aperiodic fit")
    parser.add_argument("--aperiodic-min-r2", type=float, default=None, help="Minimum R2 for aperiodic fit")
    parser.add_argument("--aperiodic-min-points", type=int, default=None, help="Minimum fit points for aperiodic")
    
    parser.add_argument("--conn-graph-prop", type=float, default=None, help="Proportion of top edges to keep for graph metrics")
    parser.add_argument("--conn-window-len", type=float, default=None, help="Sliding window length (s) for connectivity")
    parser.add_argument("--conn-window-step", type=float, default=None, help="Sliding window step (s) for connectivity")
    
    parser.add_argument("--pac-method", choices=["mvl", "kl", "tort", "ozkurt"], default=None, help="PAC estimation method")
    parser.add_argument("--pac-min-epochs", type=int, default=None, help="Minimum epochs for PAC computation")
    parser.add_argument(
        "--pac-pairs",
        nargs="+",
        default=None,
        metavar="PAIR",
        help="PAC band pairs, e.g. theta:gamma alpha:gamma (uses time_frequency_analysis.bands)",
    )
    
    parser.add_argument("--pe-delay", type=int, default=None, help="Permutation entropy delay")
    parser.add_argument("--burst-min-duration", type=int, default=None, help="Minimum burst duration (ms)")
    
    parser.add_argument("--min-epochs", type=int, default=None, help="Minimum epochs required for features")

    parser.add_argument("--fail-on-missing-windows", action="store_true", default=None, help="Fail if baseline/active windows are missing")
    parser.add_argument("--no-fail-on-missing-windows", action="store_false", dest="fail_on_missing_windows", help="Do not fail if baseline/active windows are missing")
    parser.add_argument("--fail-on-missing-named-window", action="store_true", default=None, help="Fail if a named time window is missing")
    parser.add_argument("--no-fail-on-missing-named-window", action="store_false", dest="fail_on_missing_named_window", help="Do not fail if a named time window is missing")
    
    add_path_args(parser)
    
    return parser


def run_features(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the features command."""
    from eeg_pipeline.pipelines.features import FeaturePipeline
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    
    if args.mode == "compute":
        # Apply feature-specific overrides to config
        if getattr(args, "connectivity_measures", None) is not None:
            config["feature_engineering.connectivity.measures"] = args.connectivity_measures
        
        if getattr(args, "pac_phase_range", None) is not None:
            config["feature_engineering.pac.phase_range"] = list(args.pac_phase_range)
        
        if getattr(args, "pac_amp_range", None) is not None:
            config["feature_engineering.pac.amp_range"] = list(args.pac_amp_range)
        
        if getattr(args, "aperiodic_range", None) is not None:
            config["feature_engineering.aperiodic.fmin"] = args.aperiodic_range[0]
            config["feature_engineering.aperiodic.fmax"] = args.aperiodic_range[1]
        
        if getattr(args, "pe_order", None) is not None:
            config["feature_engineering.complexity.pe_order"] = args.pe_order

        if getattr(args, "erp_baseline", None) is not None:
            config["feature_engineering.erp.baseline_correction"] = args.erp_baseline
        if getattr(args, "erp_allow_no_baseline", None) is not None:
            config["feature_engineering.erp.allow_no_baseline"] = args.erp_allow_no_baseline
        if getattr(args, "erp_components", None) is not None:
            config["feature_engineering.erp.components"] = _parse_erp_components(args.erp_components)
        
        if getattr(args, "burst_threshold", None) is not None:
            config["feature_engineering.bursts.threshold_z"] = args.burst_threshold
        if getattr(args, "burst_bands", None) is not None:
            config["feature_engineering.bursts.bands"] = list(_split_list_tokens(args.burst_bands))

        if getattr(args, "power_baseline_mode", None) is not None:
            config["time_frequency_analysis.baseline_mode"] = args.power_baseline_mode
        if getattr(args, "power_require_baseline", None) is not None:
            config["feature_engineering.power.require_baseline"] = args.power_require_baseline
            
        if getattr(args, "spectral_edge_percentile", None) is not None:
            config["feature_engineering.spectral.edge_percentile"] = args.spectral_edge_percentile
        if getattr(args, "ratio_pairs", None) is not None:
            config["feature_engineering.spectral.ratio_pairs"] = _parse_pair_tokens(args.ratio_pairs, label="ratio")
        if getattr(args, "asymmetry_channel_pairs", None) is not None:
            config["feature_engineering.asymmetry.channel_pairs"] = _parse_pair_tokens(args.asymmetry_channel_pairs, label="asymmetry")

        if getattr(args, "conn_output_level", None) is not None:
            config["feature_engineering.connectivity.output_level"] = args.conn_output_level
        if args.conn_graph_metrics is not None:
            config["feature_engineering.connectivity.enable_graph_metrics"] = args.conn_graph_metrics
        if getattr(args, "conn_aec_mode", None) is not None:
            config["feature_engineering.connectivity.aec_mode"] = args.conn_aec_mode
            
        # New Overrides
            
        if getattr(args, "aperiodic_peak_z", None) is not None:
            config["feature_engineering.aperiodic.peak_rejection_z"] = args.aperiodic_peak_z
        if getattr(args, "aperiodic_min_r2", None) is not None:
            config["feature_engineering.aperiodic.min_r2"] = args.aperiodic_min_r2
        if getattr(args, "aperiodic_min_points", None) is not None:
            config["feature_engineering.aperiodic.min_fit_points"] = args.aperiodic_min_points
            
        if getattr(args, "conn_graph_prop", None) is not None:
            config["feature_engineering.connectivity.graph_top_prop"] = args.conn_graph_prop
        if getattr(args, "conn_window_len", None) is not None:
            config["feature_engineering.connectivity.sliding_window_len"] = args.conn_window_len
        if getattr(args, "conn_window_step", None) is not None:
            config["feature_engineering.connectivity.sliding_window_step"] = args.conn_window_step
            
        if getattr(args, "pac_method", None) is not None:
            config["feature_engineering.pac.method"] = args.pac_method
        if getattr(args, "pac_min_epochs", None) is not None:
            config["feature_engineering.pac.min_epochs"] = args.pac_min_epochs
        if getattr(args, "pac_pairs", None) is not None:
            config["feature_engineering.pac.pairs"] = _parse_pair_tokens(args.pac_pairs, label="PAC")
            
        if getattr(args, "pe_delay", None) is not None:
            config["feature_engineering.complexity.pe_delay"] = args.pe_delay
        if getattr(args, "burst_min_duration", None) is not None:
            config["feature_engineering.bursts.min_duration_ms"] = args.burst_min_duration
            
        # TFR Config Overrides
        tfr_config = config.setdefault("time_frequency_analysis", {}).setdefault("tfr", {})
        if getattr(args, "tfr_freq_min", None) is not None:
            tfr_config["freq_min"] = args.tfr_freq_min
        if getattr(args, "tfr_freq_max", None) is not None:
            tfr_config["freq_max"] = args.tfr_freq_max
        if getattr(args, "tfr_n_freqs", None) is not None:
            tfr_config["n_freqs"] = args.tfr_n_freqs
        if getattr(args, "tfr_min_cycles", None) is not None:
            tfr_config["min_cycles"] = args.tfr_min_cycles
        if getattr(args, "tfr_n_cycles_factor", None) is not None:
            tfr_config["n_cycles_factor"] = args.tfr_n_cycles_factor
        if getattr(args, "tfr_decim", None) is not None:
            tfr_config["decim"] = args.tfr_decim
        if getattr(args, "tfr_workers", None) is not None:
            tfr_config["workers"] = args.tfr_workers
            
        if getattr(args, "min_epochs", None) is not None:
            config["feature_engineering.constants.min_epochs_for_features"] = args.min_epochs
            

        if getattr(args, "fail_on_missing_windows", None) is not None:
            config["feature_engineering.validation.fail_on_missing_windows"] = args.fail_on_missing_windows
        if getattr(args, "fail_on_missing_named_window", None) is not None:
            config["feature_engineering.validation.fail_on_missing_named_window"] = args.fail_on_missing_named_window
        
        # Prepare time ranges
        time_ranges = []
        if getattr(args, "time_range", None):
            for name, tmin, tmax in args.time_range:
                time_ranges.append({
                    "name": name,
                    "tmin": float(tmin) if tmin.lower() != "none" and tmin != "" else None,
                    "tmax": float(tmax) if tmax.lower() != "none" and tmax != "" else None,
                })
        
        pipeline = FeaturePipeline(config=config)
        pipeline.run_batch(
            subjects=subjects,
            task=task,
            feature_categories=categories,
            bands=getattr(args, "bands", None),
            spatial_modes=getattr(args, "spatial", None),
            tmin=getattr(args, "tmin", None),
            tmax=getattr(args, "tmax", None),
            time_ranges=time_ranges or None,
            aggregation_method=getattr(args, "aggregation_method", "mean"),
            progress=progress,
        )
    elif args.mode == "visualize":
        visualize_features_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=categories,
        )
