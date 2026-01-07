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
    
    # Storage options
    parser.add_argument("--save-subject-level-features", action="store_true", default=None, help="Save subject-level features for constant values")
    parser.add_argument("--no-save-subject-level-features", action="store_false", dest="save_subject_level_features", help="Do not save subject-level features")
    
    # Scientific Validity Options (new)
    parser.add_argument(
        "--itpc-method",
        choices=["global", "fold_global", "loo"],
        default=None,
        help="ITPC computation method: global (all trials), fold_global (training only, CV-safe), loo (leave-one-out)",
    )
    parser.add_argument(
        "--aperiodic-min-segment-sec",
        type=float,
        default=None,
        help="Minimum segment duration (seconds) for stable aperiodic fits (default: 2.0)",
    )
    parser.add_argument(
        "--aec-output",
        nargs="+",
        choices=["r", "z"],
        default=None,
        help="AEC output format: r (raw), z (Fisher-z transform), or both",
    )
    parser.add_argument(
        "--ratio-source",
        choices=["raw", "powcorr"],
        default=None,
        help="Power source for band ratios: raw (absolute) or powcorr (aperiodic-adjusted)",
    )
    parser.add_argument(
        "--conn-force-within-epoch-for-decoding",
        action="store_true",
        default=None,
        help="Force within_epoch phase estimator when train_mask detected (CV-safe)",
    )
    parser.add_argument(
        "--no-conn-force-within-epoch-for-decoding",
        action="store_false",
        dest="conn_force_within_epoch_for_decoding",
        help="Allow across_epochs phase estimator even in CV/decoding mode",
    )
    
    # ITPC additional options
    parser.add_argument("--itpc-allow-unsafe-loo", action="store_true", default=None, help="Allow unsafe LOO ITPC computation")
    parser.add_argument("--no-itpc-allow-unsafe-loo", action="store_false", dest="itpc_allow_unsafe_loo")
    parser.add_argument("--itpc-baseline-correction", choices=["none", "subtract"], default=None, help="ITPC baseline correction mode")
    
    # Spectral advanced options
    parser.add_argument("--spectral-include-log-ratios", action="store_true", default=None, help="Include log ratios in spectral features")
    parser.add_argument("--no-spectral-include-log-ratios", action="store_false", dest="spectral_include_log_ratios")
    parser.add_argument("--spectral-psd-method", choices=["multitaper", "welch"], default=None, help="PSD method for spectral features")
    parser.add_argument("--spectral-fmin", type=float, default=None, help="Min frequency for spectral features")
    parser.add_argument("--spectral-fmax", type=float, default=None, help="Max frequency for spectral features")
    parser.add_argument("--spectral-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from spectral computation")
    parser.add_argument("--no-spectral-exclude-line-noise", action="store_false", dest="spectral_exclude_line_noise")
    parser.add_argument("--spectral-line-noise-freq", type=float, default=None, help="Line noise frequency (50 or 60 Hz)")
    parser.add_argument("--spectral-segments", nargs="+", default=None, help="Segments for spectral features (e.g., baseline active)")
    parser.add_argument("--spectral-min-segment-sec", type=float, default=None, help="Minimum segment duration for spectral")
    parser.add_argument("--spectral-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at lowest frequency")
    
    # Band envelope options
    parser.add_argument("--band-envelope-pad-sec", type=float, default=None, help="Padding in seconds for band envelope")
    parser.add_argument("--band-envelope-pad-cycles", type=float, default=None, help="Padding in cycles for band envelope")
    
    # IAF options
    parser.add_argument("--iaf-enabled", action="store_true", default=None, help="Enable individualized alpha frequency")
    parser.add_argument("--no-iaf-enabled", action="store_false", dest="iaf_enabled")
    parser.add_argument("--iaf-alpha-width-hz", type=float, default=None, help="IAF alpha band width in Hz")
    parser.add_argument("--iaf-search-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="IAF search range in Hz")
    parser.add_argument("--iaf-min-prominence", type=float, default=None, help="IAF minimum peak prominence")
    parser.add_argument("--iaf-rois", nargs="+", default=None, help="ROIs for IAF detection")
    
    # Aperiodic advanced options
    parser.add_argument("--aperiodic-model", choices=["fixed", "knee"], default=None, help="Aperiodic model type")
    parser.add_argument("--aperiodic-psd-method", choices=["multitaper", "welch"], default=None, help="PSD method for aperiodic")
    parser.add_argument("--aperiodic-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from aperiodic fit")
    parser.add_argument("--no-aperiodic-exclude-line-noise", action="store_false", dest="aperiodic_exclude_line_noise")
    parser.add_argument("--aperiodic-line-noise-freq", type=float, default=None, help="Line noise frequency for aperiodic")
    
    # Connectivity advanced options
    parser.add_argument("--conn-granularity", choices=["trial", "condition", "subject"], default=None, help="Connectivity granularity")
    parser.add_argument("--conn-min-epochs-per-group", type=int, default=None, help="Min epochs per group for connectivity")
    parser.add_argument("--conn-min-cycles-per-band", type=float, default=None, help="Min cycles per band for connectivity")
    parser.add_argument("--conn-warn-no-spatial-transform", action="store_true", default=None, help="Warn if no spatial transform for phase connectivity")
    parser.add_argument("--no-conn-warn-no-spatial-transform", action="store_false", dest="conn_warn_no_spatial_transform")
    parser.add_argument("--conn-phase-estimator", choices=["within_epoch", "across_epochs"], default=None, help="Phase estimator mode")
    parser.add_argument("--conn-min-segment-sec", type=float, default=None, help="Min segment duration for connectivity")
    
    # PAC advanced options
    parser.add_argument("--pac-source", choices=["precomputed", "tfr"], default=None, help="PAC source: precomputed (Hilbert) or tfr (wavelet)")
    parser.add_argument("--pac-normalize", action="store_true", default=None, help="Normalize PAC values")
    parser.add_argument("--no-pac-normalize", action="store_false", dest="pac_normalize")
    parser.add_argument("--pac-n-surrogates", type=int, default=None, help="Number of surrogates for PAC (0=none)")
    parser.add_argument("--pac-allow-harmonic-overlap", action="store_true", default=None, help="Allow harmonic overlap in PAC")
    parser.add_argument("--no-pac-allow-harmonic-overlap", action="store_false", dest="pac_allow_harmonic_overlap")
    parser.add_argument("--pac-max-harmonic", type=int, default=None, help="Max harmonic to check for overlap")
    parser.add_argument("--pac-harmonic-tolerance-hz", type=float, default=None, help="Harmonic tolerance in Hz")
    parser.add_argument("--pac-compute-waveform-qc", action="store_true", default=None, help="Compute waveform QC for PAC")
    parser.add_argument("--no-pac-compute-waveform-qc", action="store_false", dest="pac_compute_waveform_qc")
    parser.add_argument("--pac-waveform-offset-ms", type=float, default=None, help="Waveform offset in ms for PAC QC")
    
    # Complexity advanced options
    parser.add_argument("--complexity-target-hz", type=float, default=None, help="Target sampling rate for complexity")
    parser.add_argument("--complexity-target-n-samples", type=int, default=None, help="Target number of samples for complexity")
    parser.add_argument("--complexity-zscore", action="store_true", default=None, help="Apply z-score normalization for complexity")
    parser.add_argument("--no-complexity-zscore", action="store_false", dest="complexity_zscore")
    
    # Quality options
    parser.add_argument("--quality-psd-method", choices=["welch", "multitaper"], default=None, help="PSD method for quality metrics")
    parser.add_argument("--quality-fmin", type=float, default=None, help="Min frequency for quality metrics")
    parser.add_argument("--quality-fmax", type=float, default=None, help="Max frequency for quality metrics")
    parser.add_argument("--quality-n-fft", type=int, default=None, help="FFT size for quality metrics")
    parser.add_argument("--quality-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from quality metrics")
    parser.add_argument("--no-quality-exclude-line-noise", action="store_false", dest="quality_exclude_line_noise")
    parser.add_argument("--quality-snr-signal-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Signal band for SNR computation")
    parser.add_argument("--quality-snr-noise-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Noise band for SNR computation")
    parser.add_argument("--quality-muscle-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Muscle band for artifact detection")
    
    # ERDS options
    parser.add_argument("--erds-use-log-ratio", action="store_true", default=None, help="Use dB (log ratio) instead of percent for ERDS")
    parser.add_argument("--no-erds-use-log-ratio", action="store_false", dest="erds_use_log_ratio")
    parser.add_argument("--erds-min-baseline-power", type=float, default=None, help="Min baseline power for ERDS")
    parser.add_argument("--erds-min-active-power", type=float, default=None, help="Min active power for ERDS")
    parser.add_argument("--erds-min-segment-sec", type=float, default=None, help="Min segment duration for ERDS")
    parser.add_argument("--erds-bands", nargs="+", default=None, help="Bands for ERDS computation (e.g., alpha beta)")
    
    # TFR advanced options
    parser.add_argument("--tfr-max-cycles", type=float, default=None, help="Maximum cycles for Morlet wavelets")
    parser.add_argument("--tfr-decim-power", type=int, default=None, help="Decimation factor for power TFR")
    parser.add_argument("--tfr-decim-phase", type=int, default=None, help="Decimation factor for phase TFR")
    
    # Spatial transform options (for volume conduction reduction)
    parser.add_argument(
        "--spatial-transform",
        choices=["none", "csd", "laplacian"],
        default=None,
        help="Spatial transform for volume conduction reduction: none, csd (current source density), or laplacian",
    )
    parser.add_argument("--spatial-transform-lambda2", type=float, default=None, help="Lambda2 regularization for CSD/Laplacian (default: 1e-5)")
    parser.add_argument("--spatial-transform-stiffness", type=float, default=None, help="Stiffness for CSD/Laplacian (default: 4.0)")
    
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
        
        # Storage options
        if getattr(args, "save_subject_level_features", None) is not None:
            config["feature_engineering.output.save_subject_level_features"] = args.save_subject_level_features
        
        # Scientific Validity Overrides (new)
        if getattr(args, "itpc_method", None) is not None:
            config["feature_engineering.itpc.method"] = args.itpc_method
        if getattr(args, "aperiodic_min_segment_sec", None) is not None:
            config["feature_engineering.aperiodic.min_segment_sec"] = args.aperiodic_min_segment_sec
        if getattr(args, "aec_output", None) is not None:
            config["feature_engineering.connectivity.aec_output"] = args.aec_output
        if getattr(args, "ratio_source", None) is not None:
            config["feature_engineering.spectral.ratio_source"] = args.ratio_source
        if getattr(args, "conn_force_within_epoch_for_decoding", None) is not None:
            config["feature_engineering.connectivity.force_within_epoch_for_decoding"] = args.conn_force_within_epoch_for_decoding
        
        # ITPC additional options
        if getattr(args, "itpc_allow_unsafe_loo", None) is not None:
            config["feature_engineering.itpc.allow_unsafe_loo"] = args.itpc_allow_unsafe_loo
        if getattr(args, "itpc_baseline_correction", None) is not None:
            config["feature_engineering.itpc.baseline_correction"] = args.itpc_baseline_correction
        
        # Spectral advanced options
        spectral_cfg = config.setdefault("feature_engineering", {}).setdefault("spectral", {})
        if getattr(args, "spectral_include_log_ratios", None) is not None:
            spectral_cfg["include_log_ratios"] = args.spectral_include_log_ratios
        if getattr(args, "spectral_psd_method", None) is not None:
            spectral_cfg["psd_method"] = args.spectral_psd_method
        if getattr(args, "spectral_fmin", None) is not None:
            spectral_cfg["fmin"] = args.spectral_fmin
        if getattr(args, "spectral_fmax", None) is not None:
            spectral_cfg["fmax"] = args.spectral_fmax
        if getattr(args, "spectral_exclude_line_noise", None) is not None:
            spectral_cfg["exclude_line_noise"] = args.spectral_exclude_line_noise
        if getattr(args, "spectral_line_noise_freq", None) is not None:
            spectral_cfg["line_noise_freqs"] = [args.spectral_line_noise_freq]
        if getattr(args, "spectral_segments", None) is not None:
            spectral_cfg["segments"] = args.spectral_segments
        if getattr(args, "spectral_min_segment_sec", None) is not None:
            spectral_cfg["min_segment_sec"] = args.spectral_min_segment_sec
        if getattr(args, "spectral_min_cycles_at_fmin", None) is not None:
            spectral_cfg["min_cycles_at_fmin"] = args.spectral_min_cycles_at_fmin
        
        # Band envelope options
        band_env_cfg = config.setdefault("feature_engineering", {}).setdefault("band_envelope", {})
        if getattr(args, "band_envelope_pad_sec", None) is not None:
            band_env_cfg["pad_sec"] = args.band_envelope_pad_sec
        if getattr(args, "band_envelope_pad_cycles", None) is not None:
            band_env_cfg["pad_cycles"] = args.band_envelope_pad_cycles
        
        # IAF options
        bands_cfg = config.setdefault("feature_engineering", {}).setdefault("bands", {})
        if getattr(args, "iaf_enabled", None) is not None:
            bands_cfg["use_iaf"] = args.iaf_enabled
        if getattr(args, "iaf_alpha_width_hz", None) is not None:
            bands_cfg["alpha_width_hz"] = args.iaf_alpha_width_hz
        if getattr(args, "iaf_search_range", None) is not None:
            bands_cfg["iaf_search_range_hz"] = list(args.iaf_search_range)
        if getattr(args, "iaf_min_prominence", None) is not None:
            bands_cfg["iaf_min_prominence"] = args.iaf_min_prominence
        if getattr(args, "iaf_rois", None) is not None:
            bands_cfg["iaf_rois"] = args.iaf_rois
        
        # Aperiodic advanced options
        aperiodic_cfg = config.setdefault("feature_engineering", {}).setdefault("aperiodic", {})
        if getattr(args, "aperiodic_model", None) is not None:
            aperiodic_cfg["model"] = args.aperiodic_model
        if getattr(args, "aperiodic_psd_method", None) is not None:
            aperiodic_cfg["psd_method"] = args.aperiodic_psd_method
        if getattr(args, "aperiodic_exclude_line_noise", None) is not None:
            aperiodic_cfg["exclude_line_noise"] = args.aperiodic_exclude_line_noise
        if getattr(args, "aperiodic_line_noise_freq", None) is not None:
            aperiodic_cfg["line_noise_freqs"] = [args.aperiodic_line_noise_freq]
        
        # Connectivity advanced options
        conn_cfg = config.setdefault("feature_engineering", {}).setdefault("connectivity", {})
        if getattr(args, "conn_granularity", None) is not None:
            conn_cfg["granularity"] = args.conn_granularity
        if getattr(args, "conn_min_epochs_per_group", None) is not None:
            conn_cfg["min_epochs_per_group"] = args.conn_min_epochs_per_group
        if getattr(args, "conn_min_cycles_per_band", None) is not None:
            conn_cfg["min_cycles_per_band"] = args.conn_min_cycles_per_band
        if getattr(args, "conn_warn_no_spatial_transform", None) is not None:
            conn_cfg["warn_if_no_spatial_transform"] = args.conn_warn_no_spatial_transform
        if getattr(args, "conn_phase_estimator", None) is not None:
            conn_cfg["phase_estimator"] = args.conn_phase_estimator
        if getattr(args, "conn_min_segment_sec", None) is not None:
            conn_cfg["min_segment_sec"] = args.conn_min_segment_sec
        
        # PAC advanced options
        pac_cfg = config.setdefault("feature_engineering", {}).setdefault("pac", {})
        if getattr(args, "pac_source", None) is not None:
            pac_cfg["source"] = args.pac_source
        if getattr(args, "pac_normalize", None) is not None:
            pac_cfg["normalize"] = args.pac_normalize
        if getattr(args, "pac_n_surrogates", None) is not None:
            pac_cfg["n_surrogates"] = args.pac_n_surrogates
        if getattr(args, "pac_allow_harmonic_overlap", None) is not None:
            pac_cfg["allow_harmonic_overlap"] = args.pac_allow_harmonic_overlap
        if getattr(args, "pac_max_harmonic", None) is not None:
            pac_cfg["max_harmonic"] = args.pac_max_harmonic
        if getattr(args, "pac_harmonic_tolerance_hz", None) is not None:
            pac_cfg["harmonic_tolerance_hz"] = args.pac_harmonic_tolerance_hz
        if getattr(args, "pac_compute_waveform_qc", None) is not None:
            pac_cfg["compute_waveform_qc"] = args.pac_compute_waveform_qc
        if getattr(args, "pac_waveform_offset_ms", None) is not None:
            pac_cfg["waveform_offset_ms"] = args.pac_waveform_offset_ms
        
        # Complexity advanced options
        complexity_cfg = config.setdefault("feature_engineering", {}).setdefault("complexity", {})
        if getattr(args, "complexity_target_hz", None) is not None:
            complexity_cfg["target_hz"] = args.complexity_target_hz
        if getattr(args, "complexity_target_n_samples", None) is not None:
            complexity_cfg["target_n_samples"] = args.complexity_target_n_samples
        if getattr(args, "complexity_zscore", None) is not None:
            complexity_cfg["zscore"] = args.complexity_zscore
        
        # Quality options
        quality_cfg = config.setdefault("feature_engineering", {}).setdefault("quality", {})
        if getattr(args, "quality_psd_method", None) is not None:
            quality_cfg["psd_method"] = args.quality_psd_method
        if getattr(args, "quality_fmin", None) is not None:
            quality_cfg["fmin"] = args.quality_fmin
        if getattr(args, "quality_fmax", None) is not None:
            quality_cfg["fmax"] = args.quality_fmax
        if getattr(args, "quality_n_fft", None) is not None:
            quality_cfg["n_fft"] = args.quality_n_fft
        if getattr(args, "quality_exclude_line_noise", None) is not None:
            quality_cfg["exclude_line_noise"] = args.quality_exclude_line_noise
        if getattr(args, "quality_snr_signal_band", None) is not None:
            quality_cfg["snr_signal_band"] = list(args.quality_snr_signal_band)
        if getattr(args, "quality_snr_noise_band", None) is not None:
            quality_cfg["snr_noise_band"] = list(args.quality_snr_noise_band)
        if getattr(args, "quality_muscle_band", None) is not None:
            quality_cfg["muscle_band"] = list(args.quality_muscle_band)
        
        # ERDS options
        erds_cfg = config.setdefault("feature_engineering", {}).setdefault("erds", {})
        if getattr(args, "erds_use_log_ratio", None) is not None:
            erds_cfg["use_log_ratio"] = args.erds_use_log_ratio
        if getattr(args, "erds_min_baseline_power", None) is not None:
            erds_cfg["min_baseline_power"] = args.erds_min_baseline_power
        if getattr(args, "erds_min_active_power", None) is not None:
            erds_cfg["min_active_power"] = args.erds_min_active_power
        if getattr(args, "erds_min_segment_sec", None) is not None:
            erds_cfg["min_segment_sec"] = args.erds_min_segment_sec
        if getattr(args, "erds_bands", None) is not None:
            erds_cfg["bands"] = args.erds_bands
        
        # TFR advanced options
        if getattr(args, "tfr_max_cycles", None) is not None:
            tfr_config["max_cycles"] = args.tfr_max_cycles
        if getattr(args, "tfr_decim_power", None) is not None:
            tfr_config["decim_power"] = args.tfr_decim_power
        if getattr(args, "tfr_decim_phase", None) is not None:
            tfr_config["decim_phase"] = args.tfr_decim_phase
        
        # Spatial transform options
        if getattr(args, "spatial_transform", None) is not None:
            config["feature_engineering.spatial_transform"] = args.spatial_transform
        if getattr(args, "spatial_transform_lambda2", None) is not None:
            config.setdefault("feature_engineering", {}).setdefault("spatial_transform_params", {})["lambda2"] = args.spatial_transform_lambda2
        if getattr(args, "spatial_transform_stiffness", None) is not None:
            config.setdefault("feature_engineering", {}).setdefault("spatial_transform_params", {})["stiffness"] = args.spatial_transform_stiffness
        
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
