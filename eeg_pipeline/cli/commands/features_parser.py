"""Parser construction for features extraction CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
)
from eeg_pipeline.domain.features.constants import SPATIAL_MODES
from eeg_pipeline.cli.commands.features_helpers import FEATURE_CATEGORY_CHOICES

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

    # Core arguments
    parser.add_argument("--categories", nargs="+", choices=FEATURE_CATEGORY_CHOICES, default=None, metavar="CATEGORY", help="Feature categories to process (some are compute-only or visualize-only)")
    parser.add_argument("--bands", nargs="+", default=None, help="Frequency bands to compute (default: all)")
    parser.add_argument("--frequency-bands", nargs="+", default=None, metavar="BAND_DEF", help="Custom frequency band definitions in format 'name:low:high' (e.g., delta:1.0:3.9 theta:4.0:7.9)")
    parser.add_argument("--rois", nargs="+", default=None, metavar="ROI_DEF", help="Custom ROI definitions in format 'name:ch1,ch2,...' (e.g., 'Frontal:Fp1,Fp2,F3,F4')")
    parser.add_argument("--spatial", nargs="+", choices=SPATIAL_MODES, default=None, metavar="MODE", help="Spatial aggregation modes: roi, channels, global (default: roi, global)")
    parser.add_argument("--spatial-transform", choices=["none", "csd", "laplacian"], default=None, help="Spatial transform to reduce volume conduction: none, csd, laplacian")
    parser.add_argument("--spatial-transform-lambda2", type=float, default=None, help="Lambda2 regularization for CSD/Laplacian (default: 1e-5)")
    parser.add_argument("--spatial-transform-stiffness", type=float, default=None, help="Stiffness for CSD/Laplacian (default: 4.0)")
    st_choices = ["inherit", "none", "csd", "laplacian"]
    parser.add_argument("--spatial-transform-connectivity", choices=st_choices, default=None, help="Per-family spatial transform override for connectivity")
    parser.add_argument("--spatial-transform-itpc", choices=st_choices, default=None, help="Per-family spatial transform override for ITPC")
    parser.add_argument("--spatial-transform-pac", choices=st_choices, default=None, help="Per-family spatial transform override for PAC")
    parser.add_argument("--spatial-transform-power", choices=st_choices, default=None, help="Per-family spatial transform override for power")
    parser.add_argument("--spatial-transform-aperiodic", choices=st_choices, default=None, help="Per-family spatial transform override for aperiodic")
    parser.add_argument("--spatial-transform-bursts", choices=st_choices, default=None, help="Per-family spatial transform override for bursts")
    parser.add_argument("--spatial-transform-erds", choices=st_choices, default=None, help="Per-family spatial transform override for ERDS")
    parser.add_argument("--spatial-transform-complexity", choices=st_choices, default=None, help="Per-family spatial transform override for complexity")
    parser.add_argument("--spatial-transform-ratios", choices=st_choices, default=None, help="Per-family spatial transform override for ratios")
    parser.add_argument("--spatial-transform-asymmetry", choices=st_choices, default=None, help="Per-family spatial transform override for asymmetry")
    parser.add_argument("--spatial-transform-spectral", choices=st_choices, default=None, help="Per-family spatial transform override for spectral")
    parser.add_argument("--spatial-transform-erp", choices=st_choices, default=None, help="Per-family spatial transform override for ERP")
    parser.add_argument("--spatial-transform-quality", choices=st_choices, default=None, help="Per-family spatial transform override for quality")
    parser.add_argument("--spatial-transform-microstates", choices=st_choices, default=None, help="Per-family spatial transform override for microstates")
    parser.add_argument("--tmin", type=float, default=None, help="Start time in seconds for feature extraction window")
    parser.add_argument("--tmax", type=float, default=None, help="End time in seconds for feature extraction window")
    parser.add_argument("--time-range", nargs=3, action="append", metavar=("NAME", "TMIN", "TMAX"), help="Define a named time range (e.g. baseline 0 1). Can be specified multiple times.")
    parser.add_argument("--aggregation-method", choices=["mean", "median"], default="mean", help="Aggregation method for spatial modes (default: mean)")
    parser.add_argument("--analysis-mode", choices=["group_stats", "trial_ml_safe"], default=None, help="Feature analysis mode: group_stats (default) or trial_ml_safe (ML/CV leakage-safe)")

    # Connectivity
    parser.add_argument("--connectivity-measures", nargs="+", choices=["wpli", "imcoh", "aec", "plv", "pli"], default=None, help="Connectivity measures to compute")
    parser.add_argument("--conn-output-level", choices=["full", "global_only"], default=None, help="Connectivity output level")
    parser.add_argument("--conn-graph-metrics", action="store_true", default=None, help="Enable graph metrics for connectivity")
    parser.add_argument("--no-conn-graph-metrics", action="store_false", dest="conn_graph_metrics", help="Disable graph metrics for connectivity")
    parser.add_argument("--conn-aec-mode", choices=["orth", "sym", "none"], default=None, help="AEC orthogonalization mode")
    parser.add_argument("--conn-graph-prop", type=float, default=None, help="Proportion of top edges to keep for graph metrics")
    parser.add_argument("--aec-output", nargs="+", choices=["r", "z"], default=None, help="AEC output format: r (raw), z (Fisher-z transform), or both")
    parser.add_argument("--conn-force-within-epoch-for-ml", action="store_true", default=None, help="Force within_epoch phase estimator when train_mask detected (CV-safe)")
    parser.add_argument("--no-conn-force-within-epoch-for-ml", action="store_false", dest="conn_force_within_epoch_for_ml", help="Allow across_epochs phase estimator even in CV/machine learning mode")
    parser.add_argument("--conn-window-len", type=float, default=None, help="Sliding window length for connectivity (seconds)")
    parser.add_argument("--conn-window-step", type=float, default=None, help="Sliding window step for connectivity (seconds)")
    parser.add_argument("--conn-granularity", choices=["trial", "condition", "subject"], default=None, help="Connectivity granularity")
    parser.add_argument("--conn-condition-column", default=None, help="Event column used to define connectivity condition groups when granularity='condition' (e.g., 'trial_type', 'binary_outcome').")
    parser.add_argument("--conn-condition-values", nargs="+", default=None, help="Condition values to include for connectivity grouping when granularity='condition' (space-separated). Other values are excluded (set to NaN).")
    parser.add_argument("--conn-min-epochs-per-group", type=int, default=None, help="Min epochs per group for connectivity")
    parser.add_argument("--conn-min-cycles-per-band", type=float, default=None, help="Min cycles per band for connectivity")
    parser.add_argument("--conn-warn-no-spatial-transform", action="store_true", default=None, help="Warn if no spatial transform for phase connectivity")
    parser.add_argument("--no-conn-warn-no-spatial-transform", action="store_false", dest="conn_warn_no_spatial_transform")
    parser.add_argument("--conn-phase-estimator", choices=["within_epoch", "across_epochs"], default=None, help="Phase estimator mode")
    parser.add_argument("--conn-min-segment-sec", type=float, default=None, help="Min segment duration for connectivity")
    parser.add_argument("--conn-mode", choices=["cwt_morlet", "multitaper", "fourier"], default=None, help="Connectivity time-frequency mode for phase measures (default: cwt_morlet)")
    parser.add_argument("--conn-aec-absolute", action="store_true", default=None, help="Use absolute envelope correlation (AEC) values")
    parser.add_argument("--no-conn-aec-absolute", action="store_false", dest="conn_aec_absolute")
    parser.add_argument("--conn-n-freqs-per-band", type=int, default=None, help="Number of frequencies sampled per band for phase connectivity")
    parser.add_argument("--conn-n-cycles", type=float, default=None, help="Fixed n_cycles for connectivity wavelets (overrides automatic)")
    parser.add_argument("--conn-decim", type=int, default=None, help="Decimation factor for connectivity computation")
    parser.add_argument("--conn-min-segment-samples", type=int, default=None, help="Minimum segment samples for connectivity computation")
    parser.add_argument("--conn-small-world-n-rand", type=int, default=None, help="Number of random graphs for small-world sigma estimation")
    parser.add_argument("--conn-enable-aec", action="store_true", default=None, help="Enable AEC computation when 'aec' is selected")
    parser.add_argument("--no-conn-enable-aec", action="store_false", dest="conn_enable_aec")
    parser.add_argument("--conn-dynamic", action="store_true", default=None, dest="conn_dynamic_enabled", help="Enable sliding-window dynamic connectivity features")
    parser.add_argument("--no-conn-dynamic", action="store_false", dest="conn_dynamic_enabled", help="Disable sliding-window dynamic connectivity features")
    parser.add_argument("--conn-dynamic-measures", nargs="+", choices=["wpli", "aec"], default=None, help="Dynamic connectivity measures (wpli and/or aec)")
    parser.add_argument("--conn-dynamic-autocorr-lag", type=int, default=None, help="Lag for dynamic connectivity autocorrelation features")
    parser.add_argument("--conn-dynamic-min-windows", type=int, default=None, help="Minimum sliding windows required for dynamic connectivity features")
    parser.add_argument("--conn-dynamic-roi-pairs", action="store_true", default=None, dest="conn_dynamic_include_roi_pairs", help="Include ROI-pair dynamic connectivity summaries")
    parser.add_argument("--no-conn-dynamic-roi-pairs", action="store_false", dest="conn_dynamic_include_roi_pairs", help="Disable ROI-pair dynamic connectivity summaries")
    parser.add_argument("--conn-dynamic-states", action="store_true", default=None, dest="conn_dynamic_state_enabled", help="Enable dynamic connectivity state-transition metrics (k-means)")
    parser.add_argument("--no-conn-dynamic-states", action="store_false", dest="conn_dynamic_state_enabled", help="Disable dynamic connectivity state-transition metrics")
    parser.add_argument("--conn-dynamic-n-states", type=int, default=None, dest="conn_dynamic_state_n_states", help="Number of k-means connectivity states for dynamic metrics")
    parser.add_argument("--conn-dynamic-state-min-windows", type=int, default=None, dest="conn_dynamic_state_min_windows", help="Minimum windows required for dynamic state metrics")
    parser.add_argument("--conn-dynamic-state-random-state", type=int, default=None, dest="conn_dynamic_state_random_state", help="Random seed for dynamic connectivity state clustering")

    # Directed connectivity
    parser.add_argument("--directed-connectivity-measures", nargs="+", choices=["psi", "dtf", "pdc"], default=None, help="Directed connectivity measures: psi (Phase Slope Index), dtf (Directed Transfer Function), pdc (Partial Directed Coherence)")
    parser.add_argument("--directed-conn-output-level", choices=["full", "global_only"], default=None, help="Directed connectivity output level: full (all channel pairs) or global_only (mean only)")
    parser.add_argument("--directed-conn-mvar-order", type=int, default=None, help="MVAR model order for DTF/PDC computation (default: 10)")
    parser.add_argument("--directed-conn-n-freqs", type=int, default=None, help="Number of frequency bins for directed connectivity (default: 16)")
    parser.add_argument("--directed-conn-min-segment-samples", type=int, default=None, help="Minimum segment samples for directed connectivity (default: 100)")
    parser.add_argument("--directed-conn-min-samples-per-mvar-param", type=int, default=None, help="Minimum samples per MVAR parameter for stable directed connectivity")

    # Source localization
    parser.add_argument("--source-method", choices=["lcmv", "eloreta"], default=None, help="Source localization method: lcmv (beamformer) or eloreta (inverse)")
    parser.add_argument("--source-spacing", choices=["oct5", "oct6", "ico4", "ico5"], default=None, help="Source space spacing (default: oct6)")
    parser.add_argument("--source-reg", type=float, default=None, help="LCMV regularization parameter (default: 0.05)")
    parser.add_argument("--source-snr", type=float, default=None, help="eLORETA assumed SNR for regularization (default: 3.0)")
    parser.add_argument("--source-loose", type=float, default=None, help="eLORETA loose orientation constraint 0-1 (default: 0.2)")
    parser.add_argument("--source-depth", type=float, default=None, help="eLORETA depth weighting 0-1 (default: 0.8)")
    parser.add_argument("--source-parc", choices=["aparc", "aparc.a2009s", "HCPMMP1"], default=None, help="Brain parcellation for ROI extraction (default: aparc)")
    parser.add_argument("--source-connectivity-method", choices=["aec", "wpli", "plv"], default=None, help="Connectivity method for source-space analysis (default: aec)")
    parser.add_argument("--source-subject", default=None, help="FreeSurfer subject name to use for source localization (e.g., sub-0001). If unset, defaults to sub-{subject}.")
    parser.add_argument("--source-subjects-dir", default=None, help="FreeSurfer SUBJECTS_DIR path for subject-specific source localization.")
    parser.add_argument("--source-trans", default=None, help="EEG↔MRI coregistration transform .fif (required for subject-specific/fMRI-constrained source localization).")
    parser.add_argument("--source-bem", default=None, help="BEM solution .fif (e.g., *-bem-sol.fif) (required for subject-specific/fMRI-constrained source localization).")
    parser.add_argument("--source-mindist-mm", type=float, default=None, help="Minimum distance from sources to inner skull (mm) (default: 5.0).")
    parser.add_argument("--source-create-trans", action="store_true", default=None, dest="source_create_trans", help="Auto-create coregistration transform via Docker (requires Docker; FS license from global config).")
    parser.add_argument("--source-create-bem-model", action="store_true", default=None, dest="source_create_bem_model", help="Auto-create BEM model via Docker (requires Docker; FS license from global config).")
    parser.add_argument("--source-create-bem-solution", action="store_true", default=None, dest="source_create_bem_solution", help="Auto-create BEM solution via Docker (requires Docker; FS license from global config).")
    parser.add_argument("--source-allow-identity-trans", action="store_true", default=None, dest="source_allow_identity_trans", help="Allow creating identity transform (DEBUG ONLY - scientifically invalid for production; use only when proper coregistration is unavailable).")

    # fMRI-informed source localization
    parser.add_argument("--source-fmri", action="store_true", default=None, dest="source_fmri_enabled", help="Enable fMRI-informed source localization (requires --source-subjects-dir/--source-trans/--source-bem and a stats map).")
    parser.add_argument("--no-source-fmri", action="store_false", dest="source_fmri_enabled", help="Disable fMRI-informed source localization (overrides config).")
    parser.add_argument("--source-fmri-stats-map", default=None, help="Path to an fMRI statistical map NIfTI in the same MRI space as the FreeSurfer subject (typically resampled to orig.mgz space).")
    parser.add_argument("--source-fmri-provenance", choices=["independent", "same_dataset"], default=None, help="Provenance of the fMRI constraint relative to EEG labels: independent (recommended) or same_dataset (circularity risk).")
    parser.add_argument("--source-fmri-require-provenance", action="store_true", default=None, dest="source_fmri_require_provenance", help="Require explicit fMRI provenance when using fMRI constraints.")
    parser.add_argument("--no-source-fmri-require-provenance", action="store_false", dest="source_fmri_require_provenance", help="Allow unknown fMRI provenance (not recommended).")
    parser.add_argument("--source-fmri-threshold", type=float, default=None, help="Threshold applied to fMRI stats map (default: 3.1).")
    parser.add_argument("--source-fmri-tail", choices=["pos", "abs"], default=None, help="Threshold tail: pos (positive only) or abs (absolute value) (default: pos).")
    parser.add_argument("--source-fmri-cluster-min-voxels", type=int, default=None, help="Minimum cluster size in voxels after thresholding (default: 50).")
    parser.add_argument("--source-fmri-cluster-min-mm3", type=float, default=None, dest="source_fmri_cluster_min_mm3", help="Minimum cluster volume in mm^3 after thresholding (preferred; overrides --source-fmri-cluster-min-voxels when set).")
    parser.add_argument("--source-fmri-max-clusters", type=int, default=None, help="Maximum number of clusters kept from fMRI map (default: 20).")
    parser.add_argument("--source-fmri-max-voxels-per-cluster", type=int, default=None, help="Maximum voxels sampled per cluster (default: 2000; set 0 for no limit).")
    parser.add_argument("--source-fmri-max-total-voxels", type=int, default=None, help="Maximum total voxels across all clusters (default: 20000; set 0 for no limit).")
    parser.add_argument("--source-fmri-random-seed", type=int, default=None, help="Random seed for voxel subsampling (default: 0 -> nondeterministic).")
    parser.add_argument("--source-fmri-window-a-name", default=None, help="Name for window A (e.g., 'window_a').")
    parser.add_argument("--source-fmri-window-a-tmin", type=float, default=None, help="Start time for window A in seconds.")
    parser.add_argument("--source-fmri-window-a-tmax", type=float, default=None, help="End time for window A in seconds.")
    parser.add_argument("--source-fmri-window-b-name", default=None, help="Name for window B (e.g., 'baseline').")
    parser.add_argument("--source-fmri-window-b-tmin", type=float, default=None, help="Start time for window B in seconds.")
    parser.add_argument("--source-fmri-window-b-tmax", type=float, default=None, help="End time for window B in seconds.")
    parser.add_argument("--source-fmri-contrast-enabled", action="store_true", default=None, dest="source_fmri_contrast_enabled", help="Enable building fMRI contrast from BOLD data (vs. loading pre-computed stats map).")
    parser.add_argument("--source-fmri-cond-a-column", default=None, help="Column for condition A in events.tsv (e.g., 'condition', 'binary_outcome').")
    parser.add_argument("--source-fmri-cond-a-value", default=None, help="Value for condition A (e.g., 'temp49p3', '1').")
    parser.add_argument("--source-fmri-cond-b-column", default=None, help="Column for condition B in events.tsv.")
    parser.add_argument("--source-fmri-cond-b-value", default=None, help="Value for condition B.")
    parser.add_argument("--source-fmri-contrast-type", choices=["t-test", "paired-t-test", "f-test", "custom"], default=None, help="Type of statistical contrast to compute.")
    parser.add_argument("--source-fmri-contrast-formula", default=None, help="Custom contrast formula (e.g., 'cond_a - cond_b').")
    parser.add_argument("--source-fmri-contrast-name", default=None, help="Name for the contrast output (default: 'contrast').")
    parser.add_argument("--source-fmri-runs", default=None, help="Comma-separated run numbers to include (e.g., '1,2,3').")
    parser.add_argument("--source-fmri-hrf-model", choices=["spm", "flobs", "fir"], default=None, help="HRF model for GLM (default: spm).")
    parser.add_argument("--source-fmri-drift-model", choices=["none", "cosine", "polynomial"], default=None, help="Drift model for GLM (default: cosine).")
    parser.add_argument("--source-fmri-high-pass", type=float, default=None, help="High-pass filter cutoff in Hz (default: 0.008).")
    parser.add_argument("--source-fmri-low-pass", type=float, default=None, help="Optional low-pass cutoff in Hz (default: disabled; avoid unless you know you need it).")
    parser.add_argument(
        "--source-fmri-stim-phases-to-model",
        type=str,
        default=None,
        help=(
            "Optional comma-separated allow-list of phase values to include when events.tsv has the configured phase column. "
            "If unset, no phase scoping is applied. "
            "Use 'all' to disable phase scoping."
        ),
    )
    parser.add_argument(
        "--source-fmri-phase-column",
        type=str,
        default=None,
        help="Events column used for --source-fmri-stim-phases-to-model (default: stim_phase or phase).",
    )
    parser.add_argument(
        "--source-fmri-phase-scope-column",
        type=str,
        default=None,
        help="Events column used to scope phase filtering to specific rows (default: event_columns.condition candidate).",
    )
    parser.add_argument(
        "--source-fmri-phase-scope-value",
        type=str,
        default=None,
        help="Optional value in --source-fmri-phase-scope-column to limit phase filtering; empty applies to all rows.",
    )
    parser.add_argument(
        "--source-fmri-condition-scope-trial-types",
        nargs="+",
        default=None,
        metavar="TT",
        help=(
            "Optional: restrict which events.tsv rows are eligible for condition A/B selection "
            "in source-fMRI contrast building (matched against --source-fmri-condition-scope-column). "
            "Use 'all' to disable scoping."
        ),
    )
    parser.add_argument(
        "--source-fmri-condition-scope-column",
        type=str,
        default=None,
        help="Events column used for --source-fmri-condition-scope-trial-types (default: event_columns.condition candidate).",
    )
    parser.add_argument("--source-fmri-cluster-correction", action="store_true", default=None, dest="source_fmri_cluster_correction", help="Enable cluster-extent filtering heuristic (NOT cluster-level FWE correction).")
    parser.add_argument("--source-fmri-cluster-p-threshold", type=float, default=None, help="Cluster-forming p-threshold (default: 0.001).")
    parser.add_argument("--source-fmri-output-type", choices=["z-score", "t-stat", "cope", "beta"], default=None, help="Output statistical map type (default: z-score).")
    parser.add_argument("--source-fmri-resample-to-fs", action="store_true", default=None, dest="source_fmri_resample_to_fs", help="Auto-resample stats map to FreeSurfer subject space.")
    parser.add_argument("--no-source-fmri-resample-to-fs", action="store_false", dest="source_fmri_resample_to_fs", help="Do not auto-resample stats map to FreeSurfer subject space.")
    parser.add_argument("--source-fmri-input-source", choices=["fmriprep", "bids_raw"], default=None, help="Input source for contrast builder: 'fmriprep' (default) or 'bids_raw' (uses files in func/).")
    parser.add_argument("--source-fmri-require-fmriprep", action="store_true", default=None, dest="source_fmri_require_fmriprep", help="Require fMRIPrep outputs for contrast building (default: true).")
    parser.add_argument("--no-source-fmri-require-fmriprep", action="store_false", dest="source_fmri_require_fmriprep", help="Allow using raw BIDS files if fMRIPrep outputs are missing.")

    # PAC
    parser.add_argument("--pac-phase-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Phase frequency range for PAC/CFC (Hz)")
    parser.add_argument("--pac-amp-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Amplitude frequency range for PAC/CFC (Hz)")
    parser.add_argument("--pac-method", choices=["mvl", "kl", "tort", "ozkurt"], default=None, help="PAC estimation method")
    parser.add_argument("--pac-min-epochs", type=int, default=None, help="Minimum epochs for PAC computation")
    parser.add_argument("--pac-pairs", nargs="+", default=None, metavar="PAIR", help="PAC band pairs, e.g. theta:gamma alpha:gamma (uses time_frequency_analysis.bands)")
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
    parser.add_argument("--pac-random-seed", type=int, default=None, help="Random seed for PAC surrogate testing")
    parser.add_argument("--pac-min-segment-sec", type=float, default=None, help="Minimum segment duration for PAC (sec)")
    parser.add_argument("--pac-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at PAC low-frequency bound")
    parser.add_argument("--pac-surrogate-method", choices=["trial_shuffle", "circular_shift", "swap_phase_amp", "time_shift"], default=None, help="PAC surrogate generation method")

    # Aperiodic
    parser.add_argument("--aperiodic-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Frequency range for aperiodic fit (Hz)")
    parser.add_argument("--aperiodic-peak-z", type=float, default=None, help="Peak rejection Z-threshold for aperiodic fit")
    parser.add_argument("--aperiodic-min-r2", type=float, default=None, help="Minimum R2 for aperiodic fit")
    parser.add_argument("--aperiodic-min-points", type=int, default=None, help="Minimum fit points for aperiodic")
    parser.add_argument("--aperiodic-subtract-evoked", action="store_true", default=None, help="Subtract evoked response for induced spectra (recommended for event-related paradigms)")
    parser.add_argument("--aperiodic-min-segment-sec", type=float, default=None, help="Minimum segment duration (seconds) for stable aperiodic fits (default: 2.0)")
    parser.add_argument("--aperiodic-psd-bandwidth", type=float, default=None, help="PSD bandwidth for multitaper aperiodic estimation (Hz)")
    parser.add_argument("--aperiodic-max-rms", type=float, default=None, help="Maximum RMS residual for acceptable aperiodic fits")
    parser.add_argument("--aperiodic-model", choices=["fixed", "knee"], default=None, help="Aperiodic model type")
    parser.add_argument("--aperiodic-psd-method", choices=["multitaper", "welch"], default=None, help="PSD method for aperiodic")
    parser.add_argument("--aperiodic-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from aperiodic fit")
    parser.add_argument("--no-aperiodic-exclude-line-noise", action="store_false", dest="aperiodic_exclude_line_noise")
    parser.add_argument("--aperiodic-line-noise-freq", type=float, default=None, help="Line noise frequency for aperiodic")
    parser.add_argument("--aperiodic-line-noise-width-hz", type=float, default=None, help="Line noise frequency band width to exclude from aperiodic fit")
    parser.add_argument("--aperiodic-line-noise-harmonics", type=int, default=None, help="Number of line noise harmonics to exclude from aperiodic fit")
    parser.add_argument("--aperiodic-max-freq-resolution-hz", type=float, default=None, help="Maximum allowed frequency-bin width for stable aperiodic fits")
    parser.add_argument("--aperiodic-multitaper-adaptive", action="store_true", default=None, help="Enable adaptive multitaper in aperiodic PSD estimation")

    # ERP
    parser.add_argument("--erp-baseline", action="store_true", default=None, help="Enable baseline correction for ERP")
    parser.add_argument("--no-erp-baseline", action="store_false", dest="erp_baseline", help="Disable baseline correction for ERP")
    parser.add_argument("--erp-allow-no-baseline", action="store_true", default=None, help="Allow ERP extraction when baseline window is missing")
    parser.add_argument("--no-erp-allow-no-baseline", action="store_false", dest="erp_allow_no_baseline", help="Require baseline window when ERP baseline correction is enabled")
    parser.add_argument("--erp-components", nargs="+", default=None, metavar="COMP", help="ERP component windows, e.g. n1=0.10-0.20 n2=0.20-0.35 p2=0.35-0.50")
    parser.add_argument("--erp-lowpass-hz", type=float, default=None, help="Low-pass filter frequency (Hz) for ERP peak detection (default: 30.0)")
    parser.add_argument("--erp-smooth-ms", type=float, default=None, help="Smoothing window length (ms) for ERP (0 = no smoothing)")
    parser.add_argument("--erp-peak-prominence-uv", type=float, default=None, help="Peak prominence threshold (µV) for ERP peak detection")

    # Burst
    parser.add_argument("--burst-threshold", type=float, default=None, help="Z-score threshold for burst detection (used with zscore/mad methods)")
    parser.add_argument("--burst-threshold-method", choices=["percentile", "zscore", "mad"], default=None, help="Burst threshold method: percentile, zscore, or mad (default: percentile)")
    parser.add_argument("--burst-threshold-percentile", type=float, default=None, help="Percentile threshold for burst detection (0-100, default: 95.0)")
    parser.add_argument("--burst-threshold-reference", choices=["trial", "subject", "condition"], default=None, help="Burst threshold reference: trial (trialwise-valid), subject (cross-trial), or condition (cross-trial within condition)")
    parser.add_argument("--burst-min-trials-per-condition", type=int, default=None, help="Minimum trials per condition when threshold_reference='condition' (default: 10)")
    parser.add_argument("--burst-min-segment-sec", type=float, default=None, help="Minimum segment duration (sec) before attempting bursts (default: 2.0)")
    parser.add_argument("--burst-skip-invalid-segments", action="store_true", default=None, help="Skip invalid segments for bursts")
    parser.add_argument("--no-burst-skip-invalid-segments", action="store_false", dest="burst_skip_invalid_segments")
    parser.add_argument("--burst-bands", nargs="+", default=None, metavar="BAND", help="Burst bands to compute, e.g. beta gamma")
    parser.add_argument("--burst-min-duration", type=int, default=None, help="Minimum burst duration (ms)")
    parser.add_argument("--burst-min-cycles", type=float, default=None, help="Minimum oscillatory cycles for burst detection")

    # Power
    parser.add_argument("--power-baseline-mode", choices=["logratio", "mean", "ratio", "zscore", "zlogratio"], default=None, help="Baseline normalization mode for power")
    parser.add_argument("--power-require-baseline", action="store_true", default=None, help="Require baseline for power normalization")
    parser.add_argument("--no-power-require-baseline", action="store_false", dest="power_require_baseline", help="Allow raw log power without baseline")
    parser.add_argument("--power-subtract-evoked", action="store_true", default=None, help="Subtract evoked response to isolate induced power (use with care in CV)")
    parser.add_argument("--no-power-subtract-evoked", action="store_false", dest="power_subtract_evoked", help="Do not subtract evoked response for power")
    parser.add_argument("--power-min-trials-per-condition", type=int, default=None, help="Minimum trials per condition for power computation (default: 2)")
    parser.add_argument("--power-exclude-line-noise", action="store_true", default=None, help="Exclude line noise frequencies from power computation")
    parser.add_argument("--no-power-exclude-line-noise", action="store_false", dest="power_exclude_line_noise")
    parser.add_argument("--power-line-noise-freq", type=float, default=None, help="Line noise frequency for power (50 or 60 Hz)")
    parser.add_argument("--power-line-noise-width-hz", type=float, default=None, help="Line noise frequency band width to exclude for power")
    parser.add_argument("--power-line-noise-harmonics", type=int, default=None, help="Number of line noise harmonics to exclude for power")
    parser.add_argument("--power-emit-db", action="store_true", default=None, help="Emit dB-scaled versions of log10-ratio power (10*log10)")
    parser.add_argument("--no-power-emit-db", action="store_false", dest="power_emit_db")

    # Spectral
    parser.add_argument("--spectral-edge-percentile", type=float, default=None, help="Percentile for spectral edge frequency (0-1)")
    parser.add_argument("--ratio-pairs", nargs="+", default=None, metavar="PAIR", help="Band power ratio pairs, e.g. theta:beta theta:alpha alpha:beta")
    parser.add_argument("--ratio-source", choices=["raw", "powcorr"], default=None, help="Power source for band ratios: raw (absolute) or powcorr (aperiodic-adjusted)")
    parser.add_argument("--spectral-include-log-ratios", action="store_true", default=None, help="Include log ratios in spectral features")
    parser.add_argument("--no-spectral-include-log-ratios", action="store_false", dest="spectral_include_log_ratios")
    parser.add_argument("--spectral-psd-method", choices=["multitaper", "welch"], default=None, help="PSD method for spectral features")
    parser.add_argument("--spectral-psd-adaptive", action="store_true", default=None, help="Enable adaptive PSD settings for spectral features")
    parser.add_argument("--no-spectral-psd-adaptive", action="store_false", dest="spectral_psd_adaptive")
    parser.add_argument("--spectral-multitaper-adaptive", action="store_true", default=None, help="Enable adaptive multitaper for spectral PSD")
    parser.add_argument("--no-spectral-multitaper-adaptive", action="store_false", dest="spectral_multitaper_adaptive")
    parser.add_argument("--spectral-fmin", type=float, default=None, help="Min frequency for spectral features")
    parser.add_argument("--spectral-fmax", type=float, default=None, help="Max frequency for spectral features")
    parser.add_argument("--spectral-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from spectral computation")
    parser.add_argument("--no-spectral-exclude-line-noise", action="store_false", dest="spectral_exclude_line_noise")
    parser.add_argument("--spectral-line-noise-freq", type=float, default=None, help="Line noise frequency (50 or 60 Hz)")
    parser.add_argument("--spectral-line-noise-width-hz", type=float, default=None, help="Line noise frequency band width to exclude")
    parser.add_argument("--spectral-line-noise-harmonics", type=int, default=None, help="Number of line noise harmonics to exclude")
    parser.add_argument("--spectral-segments", nargs="+", default=None, help="Segments for spectral features (e.g., baseline active)")
    parser.add_argument("--spectral-min-segment-sec", type=float, default=None, help="Minimum segment duration for spectral")
    parser.add_argument("--spectral-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at lowest frequency")

    # Asymmetry
    parser.add_argument("--asymmetry-channel-pairs", nargs="+", default=None, metavar="PAIR", help="Channel pairs for asymmetry, e.g. F3:F4 C3:C4")
    parser.add_argument("--asymmetry-activation-bands", nargs="+", default=None, metavar="BAND", help="Bands for activation-style asymmetry outputs (default: alpha)")
    parser.add_argument("--asymmetry-emit-activation-convention", action="store_true", default=None, help="Emit activation-style asymmetry (R-L)/(R+L) for activation bands")
    parser.add_argument("--no-asymmetry-emit-activation-convention", action="store_false", dest="asymmetry_emit_activation_convention")
    parser.add_argument("--asymmetry-min-segment-sec", type=float, default=None, help="Minimum segment duration for asymmetry (sec)")
    parser.add_argument("--asymmetry-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at lowest frequency for asymmetry")
    parser.add_argument("--asymmetry-skip-invalid-segments", action="store_true", default=None, help="Skip invalid segments for asymmetry")
    parser.add_argument("--no-asymmetry-skip-invalid-segments", action="store_false", dest="asymmetry_skip_invalid_segments")

    # Ratios (validity)
    parser.add_argument("--ratios-min-segment-sec", type=float, default=None, help="Minimum segment duration for band ratios (sec)")
    parser.add_argument("--ratios-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at lowest frequency for band ratios")
    parser.add_argument("--ratios-skip-invalid-segments", action="store_true", default=None, help="Skip invalid segments for band ratios")
    parser.add_argument("--no-ratios-skip-invalid-segments", action="store_false", dest="ratios_skip_invalid_segments")

    # TFR
    parser.add_argument("--tfr-freq-min", type=float, default=None, help="Minimum frequency for TFR (Hz)")
    parser.add_argument("--tfr-freq-max", type=float, default=None, help="Maximum frequency for TFR (Hz)")
    parser.add_argument("--tfr-n-freqs", type=int, default=None, help="Number of frequencies for TFR")
    parser.add_argument("--tfr-min-cycles", type=float, default=None, help="Minimum number of cycles for Morlet wavelets")
    parser.add_argument("--tfr-n-cycles-factor", type=float, default=None, help="Cycles factor (freq/factor) for Morlet wavelets")
    parser.add_argument("--tfr-decim", type=int, default=None, help="Decimation factor for TFR")
    parser.add_argument("--tfr-workers", type=int, default=None, help="Number of parallel workers for TFR computation")
    parser.add_argument("--tfr-max-cycles", type=float, default=None, help="Maximum cycles for Morlet wavelets")
    parser.add_argument("--tfr-decim-power", type=int, default=None, help="Decimation factor for power TFR")
    parser.add_argument("--tfr-decim-phase", type=int, default=None, help="Decimation factor for phase TFR")

    # ITPC
    parser.add_argument("--itpc-method", choices=["global", "fold_global", "loo", "condition"], default=None, help="ITPC computation method: global (all trials), fold_global (training only, CV-safe), loo (leave-one-out), condition (per condition group, avoids pseudo-replication)")
    parser.add_argument("--itpc-allow-unsafe-loo", action="store_true", default=None, help="Allow unsafe LOO ITPC computation")
    parser.add_argument("--no-itpc-allow-unsafe-loo", action="store_false", dest="itpc_allow_unsafe_loo")
    parser.add_argument("--itpc-baseline-correction", choices=["none", "subtract"], default=None, help="ITPC baseline correction mode")
    parser.add_argument("--itpc-condition-column", default=None, help="Column for condition-based ITPC (avoids pseudo-replication)")
    parser.add_argument("--itpc-condition-values", nargs="+", default=None, help="Condition values to include for ITPC when method='condition' (space-separated). Other values are excluded (set to NaN).")
    parser.add_argument("--itpc-min-trials-per-condition", type=int, default=None, help="Minimum trials per condition for reliable ITPC (default: 10)")
    parser.add_argument("--itpc-n-jobs", type=int, default=None, help="Number of parallel jobs for ITPC computation (-1 = all CPUs, default: -1)")
    parser.add_argument("--itpc-min-segment-sec", type=float, default=None, help="Minimum segment duration for ITPC (sec)")
    parser.add_argument("--itpc-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at ITPC low-frequency bound")

    # Band envelope
    parser.add_argument("--band-envelope-pad-sec", type=float, default=None, help="Padding in seconds for band envelope")
    parser.add_argument("--band-envelope-pad-cycles", type=float, default=None, help="Padding in cycles for band envelope")

    # IAF
    parser.add_argument("--iaf-enabled", action="store_true", default=None, help="Enable individualized alpha frequency")
    parser.add_argument("--no-iaf-enabled", action="store_false", dest="iaf_enabled")
    parser.add_argument("--iaf-alpha-width-hz", type=float, default=None, help="IAF alpha band width in Hz")
    parser.add_argument("--iaf-search-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="IAF search range in Hz")
    parser.add_argument("--iaf-min-prominence", type=float, default=None, help="IAF minimum peak prominence")
    parser.add_argument("--iaf-rois", nargs="+", default=None, help="ROIs for IAF detection")
    parser.add_argument("--iaf-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at IAF search fmin for stable peak detection")
    parser.add_argument("--iaf-min-baseline-sec", type=float, default=None, help="Additional absolute minimum baseline duration (sec) for IAF (0 disables)")
    parser.add_argument("--iaf-allow-full-fallback", action="store_true", default=None, help="If baseline is missing, allow using full segment for IAF (not recommended)")
    parser.add_argument("--no-iaf-allow-full-fallback", action="store_false", dest="iaf_allow_full_fallback")
    parser.add_argument("--iaf-allow-all-channels-fallback", action="store_true", default=None, help="If IAF ROIs are missing, allow using all channels (not recommended)")
    parser.add_argument("--no-iaf-allow-all-channels-fallback", action="store_false", dest="iaf_allow_all_channels_fallback")

    # Complexity
    parser.add_argument("--pe-order", type=int, default=None, help="Permutation entropy order (3-7, default: from config)")
    parser.add_argument("--pe-delay", type=int, default=None, help="Permutation entropy delay")
    parser.add_argument("--complexity-sampen-order", type=int, default=None, help="Sample entropy embedding dimension (default: from config)")
    parser.add_argument("--complexity-sampen-r", type=float, default=None, help="Sample entropy tolerance as fraction of SD")
    parser.add_argument("--complexity-mse-scale-min", type=int, default=None, help="Minimum MSE coarse-graining scale")
    parser.add_argument("--complexity-mse-scale-max", type=int, default=None, help="Maximum MSE coarse-graining scale")
    parser.add_argument("--complexity-signal-basis", choices=["filtered", "envelope"], default=None, help="Complexity signal basis")
    parser.add_argument("--complexity-min-segment-sec", type=float, default=None, help="Minimum segment duration for complexity (sec)")
    parser.add_argument("--complexity-min-samples", type=int, default=None, help="Minimum samples for complexity")
    parser.add_argument("--complexity-zscore", action="store_true", default=None, help="Apply z-score normalization for complexity")
    parser.add_argument("--no-complexity-zscore", action="store_false", dest="complexity_zscore")

    # Quality
    parser.add_argument("--quality-psd-method", choices=["welch", "multitaper"], default=None, help="PSD method for quality metrics")
    parser.add_argument("--quality-fmin", type=float, default=None, help="Min frequency for quality metrics")
    parser.add_argument("--quality-fmax", type=float, default=None, help="Max frequency for quality metrics")
    parser.add_argument("--quality-n-fft", type=int, default=None, help="FFT size for quality metrics")
    parser.add_argument("--quality-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from quality metrics")
    parser.add_argument("--no-quality-exclude-line-noise", action="store_false", dest="quality_exclude_line_noise")
    parser.add_argument("--quality-line-noise-freq", type=float, default=None, help="Line noise frequency for quality metrics")
    parser.add_argument("--quality-line-noise-width-hz", type=float, default=None, help="Line noise width for quality metrics")
    parser.add_argument("--quality-line-noise-harmonics", type=int, default=None, help="Line noise harmonics for quality metrics")
    parser.add_argument("--quality-snr-signal-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Signal band for SNR computation")
    parser.add_argument("--quality-snr-noise-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Noise band for SNR computation")
    parser.add_argument("--quality-muscle-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Muscle band for artifact detection")

    # Microstates
    parser.add_argument("--microstates-n-states", type=int, default=None, help="Number of microstate classes (default: 4)")
    parser.add_argument("--microstates-min-peak-distance-ms", type=float, default=None, help="Minimum GFP peak distance in ms (default: 10)")
    parser.add_argument("--microstates-max-gfp-peaks-per-epoch", type=int, default=None, help="Maximum GFP peaks sampled per epoch (default: 400)")
    parser.add_argument("--microstates-min-duration-ms", type=float, default=None, help="Minimum state duration in ms for temporal smoothing")
    parser.add_argument("--microstates-gfp-peak-prominence", type=float, default=None, help="Optional GFP peak prominence threshold")
    parser.add_argument("--microstates-random-state", type=int, default=None, help="Random seed for microstate template fitting")
    parser.add_argument("--microstates-assign-from-gfp-peaks", action="store_true", default=None, dest="microstates_assign_from_gfp_peaks", help="Assign states at GFP peaks and backfit labels")
    parser.add_argument("--no-microstates-assign-from-gfp-peaks", action="store_false", dest="microstates_assign_from_gfp_peaks")
    parser.add_argument(
        "--fixed-templates-path",
        type=str,
        default=None,
        help="Optional .npz file with fixed microstate templates ('templates' and optional 'ch_names'/'labels')",
    )

    # ERDS
    parser.add_argument("--erds-use-log-ratio", action="store_true", default=None, help="Use dB (log ratio) instead of percent for ERDS")
    parser.add_argument("--no-erds-use-log-ratio", action="store_false", dest="erds_use_log_ratio")
    parser.add_argument("--erds-min-baseline-power", type=float, default=None, help="Min baseline power for ERDS")
    parser.add_argument("--erds-min-active-power", type=float, default=None, help="Min active power for ERDS")
    parser.add_argument("--erds-min-segment-sec", type=float, default=None, help="Min segment duration for ERDS")
    parser.add_argument("--erds-bands", nargs="+", default=None, help="Bands for ERDS computation (e.g., alpha beta)")
    parser.add_argument("--erds-onset-threshold-sigma", type=float, default=None, help="Onset threshold in baseline SD units for trial-level alpha ERD latency")
    parser.add_argument("--erds-onset-min-duration-ms", type=float, default=None, help="Minimum sustained duration for ERD onset threshold crossing")
    parser.add_argument("--erds-rebound-min-latency-ms", type=float, default=None, help="Minimum latency after ERD peak before searching alpha rebound")
    parser.add_argument("--erds-infer-contralateral", action="store_true", default=None, help="Infer contralateral hemisphere when trial laterality metadata is missing")
    parser.add_argument("--no-erds-infer-contralateral", action="store_false", dest="erds_infer_contralateral")
    parser.add_argument("--erds-condition-marker-bands", nargs="+", default=None, help="Bands for ERDS condition-marker extraction")
    parser.add_argument("--erds-laterality-columns", nargs="+", default=None, help="Candidate events.tsv columns for stimulation laterality")
    parser.add_argument("--erds-somatosensory-left-channels", nargs="+", default=None, help="Left somatosensory channels for condition markers")
    parser.add_argument("--erds-somatosensory-right-channels", nargs="+", default=None, help="Right somatosensory channels for condition markers")
    parser.add_argument("--erds-onset-min-threshold-percent", type=float, default=None, help="Minimum percent threshold for ERD onset")
    parser.add_argument("--erds-rebound-threshold-sigma", type=float, default=None, help="Sigma threshold for ERD rebound detection")
    parser.add_argument("--erds-rebound-min-threshold-percent", type=float, default=None, help="Minimum percent threshold for ERD rebound")

    # Validation and output
    parser.add_argument("--min-epochs", type=int, default=None, help="Minimum epochs required for features")
    parser.add_argument("--compute-change-scores", action="store_true", default=None, dest="compute_change_scores", help="Compute within-subject change score columns")
    parser.add_argument("--no-compute-change-scores", action="store_false", dest="compute_change_scores", help="Disable change score columns")
    parser.add_argument("--change-scores-transform", choices=["difference", "percent", "log_ratio", "ratio"], default=None, dest="change_scores_transform", help="Change score transform: difference, percent, log_ratio (ratio = percent)")
    parser.add_argument("--change-scores-window-pairs", nargs="+", default=None, metavar="REF:TARGET", dest="change_scores_window_pairs", help="Window pairs for change scores (e.g. baseline:plateau baseline:active)")
    parser.add_argument("--save-tfr-with-sidecar", action="store_true", default=None, dest="save_tfr_with_sidecar", help="Save TFR arrays alongside feature tables")
    parser.add_argument("--no-save-tfr-with-sidecar", action="store_false", dest="save_tfr_with_sidecar", help="Do not save TFR arrays sidecar")
    parser.add_argument("--n-jobs-bands", type=int, default=None, help="Parallel jobs for band-wise precompute (-1 = all)")
    parser.add_argument("--n-jobs-connectivity", type=int, default=None, help="Parallel jobs for connectivity (-1 = all)")
    parser.add_argument("--n-jobs-aperiodic", type=int, default=None, help="Parallel jobs for aperiodic (-1 = all)")
    parser.add_argument("--n-jobs-complexity", type=int, default=None, help="Parallel jobs for complexity (-1 = all)")
    parser.add_argument("--save-subject-level-features", action="store_true", default=None, help="Save subject-level features for constant values")
    parser.add_argument("--no-save-subject-level-features", action="store_false", dest="save_subject_level_features", help="Do not save subject-level features")
    
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--also-save-csv",
        action="store_true",
        default=None,
        dest="also_save_csv",
        help="Also save feature tables as CSV files (in addition to parquet)",
    )
    output_group.add_argument(
        "--no-also-save-csv",
        action="store_false",
        dest="also_save_csv",
    )

    add_path_args(parser)

    return parser
