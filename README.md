# EEG Pipeline

<<<<<<< HEAD
This pipeline is still under development.

A EEG analysis pipeline for thermal pain studies.
=======
This pipeline is still under development

A comprehensive EEG analysis pipeline for thermal pain studies, designed for reproducibility and scientific rigor.
>>>>>>> cad687c (Sync local EEG pipeline updates with Thermal Pain repo)

## Overview

This pipeline processes EEG data through a complete analysis workflow:

```
Raw EEG Data → BIDS Conversion → Feature Extraction → Behavioral Correlations → Decoding
```

## Architecture

```
eeg_pipeline/
├── __init__.py                  # Package entry, exports types
├── types.py                     # Type definitions and protocols
│
├── pipelines/                   # High-level pipeline orchestration
│   ├── behavior.py              # Brain-behavior correlations
│   ├── features.py              # Feature extraction
│   ├── decoding.py              # ML decoding
│   └── erp.py                   # ERP statistics
│
├── analysis/                    # Core analysis modules
│   ├── behavior/                # Brain-behavior correlation analysis
│   │   ├── core.py              # BehaviorContext, CorrelationRecord
│   │   ├── power_roi.py         # Power ROI correlations
│   │   ├── connectivity.py      # Connectivity correlations
│   │   ├── temporal.py          # Time-frequency correlations
│   │   ├── precomputed_correlations.py
│   │   ├── condition_correlations.py
│   │   ├── specialized_features.py  # Specialized feature correlations
│   │   ├── correlations.py      # Generic correlation utilities
│   │   ├── cluster_tests.py     # Cluster permutation tests
│   │   ├── topomaps.py          # Topographic map correlations
│   │   ├── fdr_correction.py    # Global FDR correction
│   │   └── exports.py           # Export significant predictors
│   ├── decoding/                # ML-based prediction
│   │   ├── cv.py                # Cross-validation utilities
│   │   ├── pipelines.py         # ML pipeline factories
│   │   ├── data.py              # Data loading for decoding
│   │   ├── cross_validation.py  # LOSO, within-subject CV
│   │   ├── permutation.py       # Permutation testing
│   │   └── time_generalization.py
│   ├── features/                # Feature extraction
│   │   ├── core.py              # PrecomputedData, shared utilities
│   │   ├── pipeline.py          # Feature orchestration
│   │   ├── power.py             # Band power
│   │   ├── connectivity.py      # wPLI, AEC, graph metrics
│   │   ├── microstates.py       # Microstate dynamics
│   │   ├── aperiodic.py         # 1/f spectral features
│   │   ├── phase.py             # ITPC, PAC
│   │   ├── erds.py              # ERD/ERS
│   │   ├── spectral.py          # IAF, entropy, ratios
│   │   ├── temporal.py          # Statistical moments
│   │   ├── complexity.py        # PE, Hjorth, LZC
│   │   ├── global_features.py   # GFP, global synchrony
│   │   ├── roi_features.py      # ROI-averaged features
│   │   └── plateau.py           # Plateau-averaged features
│   └── group/                   # Group-level statistics
│       ├── behavior.py          # Group behavior analysis
│       ├── features.py          # Group feature aggregation
│       └── statistics.py        # Group statistical tests
│
├── plotting/                    # Visualization (lazy imports)
│   ├── config.py                # Plot configuration (PlotConfig dataclass)
│   ├── behavioral/              # Brain-behavior correlation plots
│   │   ├── builders.py          # Low-level scatter builders
│   │   ├── scatter.py           # Power-behavior scatter plots
│   │   ├── temporal.py          # Temporal correlation topomaps
│   │   ├── group.py             # Group-level aggregation plots
│   │   ├── effect_sizes.py      # Forest plots, effect size heatmaps
│   │   ├── mediation.py         # Mediation path diagrams
│   │   ├── mixed_effects.py     # ICC, variance decomposition
│   │   ├── robust.py            # Sensitivity analysis, outliers
│   │   ├── distributions.py     # Feature distribution plots (NEW)
│   │   ├── summary.py           # Dashboard visualizations (NEW)
│   │   ├── registry.py          # Plot registration system (NEW)
│   │   └── viz.py               # Orchestration
│   ├── core/                    # Core plotting utilities
│   ├── decoding/                # Performance plots
│   ├── erp/                     # ERP plots
│   ├── features/                # Feature distributions
│   │   ├── power.py, power_group.py
│   │   ├── connectivity.py
│   │   ├── microstates.py
│   │   ├── aperiodic.py
│   │   ├── phase.py
│   │   └── viz.py               # General feature viz
│   └── tfr/                     # Time-frequency plots
│
├── preprocessing/               # Data preprocessing
│   ├── raw_to_bids.py           # BrainVision to BIDS
│   └── merge_behavior_to_events.py
│
├── scripts/
│   └── run_pipeline.py          # Unified CLI
│
└── utils/                       # Shared utilities only
    ├── config/                  # Configuration loading
    │   ├── loader.py            # Config loader
    │   └── eeg_config.yaml      # Main configuration
    ├── data/                    # Data loading
    │   ├── loading.py           # Data loaders
    │   └── features.py          # Feature data utilities
    ├── io/                      # File I/O
    │   ├── general.py           # General I/O utilities
    │   └── decoding.py          # Decoding I/O
    ├── analysis/                # Stats, TFR helpers
    │   ├── stats.py             # Statistical utilities
    │   ├── tfr.py               # Time-frequency utilities
    │   ├── reliability.py       # Reliability analysis
    │   └── windowing.py         # Time windowing
    ├── progress.py              # Progress tracking utilities
    └── validation.py            # Data validation utilities
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                           │
└─────────────────────────────────────────────────────────────────────┘

1. PREPROCESSING
   source_data/sub-*/eeg/*.vhdr  ──────►  bids_output/sub-*/eeg/
                                           │
   source_data/sub-*/PsychoPy_Data/ ───────┘ (behavioral merge)

2. FEATURE EXTRACTION
   bids_output/  ──► preprocessed/ ──► derivatives/sub-*/eeg/features/
                       │                    │
                       ▼                    ▼
                  Cleaned epochs      Power, Connectivity,
                  (ICA, bad channels)  Microstates, Aperiodic,
                                      ITPC, PAC features

3. ANALYSIS
   derivatives/sub-*/eeg/features/ ──► derivatives/sub-*/eeg/stats/
                                   │
                                   ├── Behavioral correlations
                                   ├── FDR-corrected results
                                   └── Significant predictors

4. DECODING
   derivatives/sub-*/eeg/features/ ──► derivatives/decoding/
                                   │
                                   ├── LOSO predictions
                                   ├── Time-generalization
                                   └── Feature importance

5. GROUP ANALYSIS
   All subjects ──► derivatives/group/eeg/stats/
                          │
                          └── Aggregated statistics
```

---

## Pipeline Execution Trees

### Feature Extraction Pipeline

```
run_pipeline.py features compute --subject 0001
│
└── extract_all_features()                    [pipelines/features.py]
    │
    ├── LOAD DATA
    │   ├── Load cleaned epochs               → derivatives/preprocessed/sub-*/eeg/*-epo.fif
    │   ├── Load events with behavior         → Aligned VAS ratings, temperature
    │   └── Compute TFR (Morlet wavelets)     → Complex time-frequency representation
    │
    ├── POWER FEATURES                        [analysis/features/power.py]
    │   ├── extract_band_power_features()
    │   │   ├── Delta (1-4 Hz) power per channel
    │   │   ├── Theta (4-8 Hz) power per channel
    │   │   ├── Alpha (8-13 Hz) power per channel
    │   │   ├── Beta (13-30 Hz) power per channel
    │   │   └── Gamma (30-80 Hz) power per channel
    │   └── OUTPUT: features_eeg_direct.tsv
    │       └── Columns: pow_{band}_{channel} (e.g., pow_alpha_Cz)
    │
    ├── CONNECTIVITY FEATURES                 [analysis/features/connectivity.py]
    │   ├── extract_connectivity_features()
    │   │   ├── wPLI (weighted Phase Lag Index)
    │   │   │   └── Per band: wpli_{band}_{ch1}_{ch2}
    │   │   ├── AEC (Amplitude Envelope Correlation)
    │   │   │   └── Per band: aec_{band}_{ch1}_{ch2}
    │   │   ├── imCoh (Imaginary Coherence) [optional]
    │   │   │   └── Per band: imcoh_{band}_{ch1}_{ch2}
    │   │   ├── PLI (Phase Lag Index) [optional]
    │   │   │   └── Per band: pli_{band}_{ch1}_{ch2}
    │   │   └── Graph metrics: clustering, path_length, small_world
    │   └── OUTPUT: features_connectivity.tsv
    │
    ├── MICROSTATE FEATURES                   [analysis/features/microstates.py]
    │   ├── extract_microstate_features()
    │   │   ├── State coverage (% time in each state)
    │   │   │   └── ms_coverage_{A,B,C,D,...}
    │   │   ├── State duration (mean duration)
    │   │   │   └── ms_duration_{A,B,C,D,...}
    │   │   ├── State occurrence (frequency)
    │   │   │   └── ms_occurrence_{A,B,C,D,...}
    │   │   ├── Transition probabilities
    │   │   │   └── ms_transition_{from}_{to}
    │   │   └── Global explained variance (GEV)
    │   │       └── ms_gev_{A,B,C,D,...}
    │   └── OUTPUT: features_microstates.tsv
    │
    ├── APERIODIC FEATURES                    [analysis/features/aperiodic.py]
    │   ├── extract_aperiodic_features()
    │   │   ├── 1/f slope (spectral exponent)
    │   │   │   └── aper_slope_{channel}
    │   │   ├── 1/f offset (broadband power)
    │   │   │   └── aper_offset_{channel}
    │   │   └── Corrected band power
    │   │       └── powcorr_{band}_{channel}
    │   └── OUTPUT: Appended to features_eeg_direct.tsv
    │
    ├── PHASE FEATURES                        [analysis/features/phase.py]
    │   ├── extract_itpc_features()
    │   │   ├── Inter-Trial Phase Coherence per band/channel
    │   │   │   └── itpc_{band}_{channel}
    │   │   └── Per time window if configured
    │   │       └── itpc_{band}_{channel}_t{window}
    │   ├── compute_pac_comodulograms()
    │   │   ├── Phase-Amplitude Coupling strength
    │   │   │   └── pac_{phase_band}_{amp_band}_{channel}
    │   │   └── PAC statistics (MI, z-scores)
    │   └── OUTPUT: features_pac_trials.tsv, pac_comodulograms.npz
    │
    └── PRECOMPUTED FEATURES                  [analysis/features/pipeline.py]
        ├── extract_precomputed_features()
        │   │
        │   ├── ERD/ERS Features              [analysis/features/erds.py]
        │   │   ├── Event-Related Desynchronization
        │   │   │   └── erds_{band}_{channel}
        │   │   ├── Temporal ERD/ERS windows
        │   │   │   └── erds_{band}_{channel}_t{window}
        │   │   └── ERD/ERS slopes
        │   │       └── erds_slope_{band}_{channel}
        │   │
        │   ├── Spectral Features             [analysis/features/spectral.py]
        │   │   ├── Individual Alpha Frequency (IAF)
        │   │   │   └── iaf_{channel}, iaf_power_{channel}
        │   │   ├── Relative band power
        │   │   │   └── relative_{band}_{channel}
        │   │   ├── Band power ratios
        │   │   │   └── ratio_{band1}_{band2}_{channel}
        │   │   ├── Spectral entropy
        │   │   │   └── spectral_entropy_{channel}
        │   │   └── Peak frequencies per band
        │   │       └── peak_freq_{band}_{channel}
        │   │
        │   ├── Temporal Features             [analysis/features/temporal.py]
        │   │   ├── Statistical moments
        │   │   │   └── mean_{ch}, var_{ch}, skew_{ch}, kurt_{ch}
        │   │   ├── Amplitude features
        │   │   │   └── rms_{ch}, p2p_{ch}, line_length_{ch}
        │   │   ├── Waveform features
        │   │   │   └── zero_cross_{ch}, nle_{ch}
        │   │   ├── Percentile features
        │   │   │   └── p{5,25,75,95}_{channel}
        │   │   └── Derivative features
        │   │       └── mean_deriv_{ch}, var_deriv_{ch}
        │   │
        │   ├── Complexity Features           [analysis/features/complexity.py]
        │   │   ├── Permutation Entropy
        │   │   │   └── pe_{channel}
        │   │   ├── Sample Entropy [optional]
        │   │   │   └── sampen_{channel}
        │   │   ├── Hjorth Parameters
        │   │   │   └── hjorth_{activity,mobility,complexity}_{ch}
        │   │   └── Lempel-Ziv Complexity
        │   │       └── lzc_{channel}
        │   │
        │   ├── Global Features               [analysis/features/global_features.py]
        │   │   ├── GFP statistics
        │   │   │   └── gfp_{mean,std,peak_rate,peak_amp,...}
        │   │   ├── GFP per band
        │   │   │   └── gfp_{band}_{mean,std,...}
        │   │   ├── Global synchrony (PLV)
        │   │   │   └── global_plv_{band}
        │   │   └── Variance explained (PCA)
        │   │       └── var_explained_{pc1,pc3,total}
        │   │
        │   └── ROI Features                  [analysis/features/roi_features.py]
        │       ├── ROI-averaged power
        │       │   └── roi_pow_{band}_{roi}
        │       ├── Hemispheric asymmetry
        │       │   └── asymmetry_{band}_{pair}
        │       ├── Pain-relevant ROIs
        │       │   └── pain_roi_{band}_{region}
        │       └── ROI ERD/ERS
        │           └── roi_erds_{band}_{roi}
        │
        └── OUTPUT: features_precomputed.tsv
```

### Output Files (Feature Extraction)

```
derivatives/sub-{subject}/eeg/features/
├── features_eeg_direct.tsv       # Power + Aperiodic features
├── features_eeg_direct_columns.tsv  # Column metadata
├── features_eeg_plateau.tsv     # Plateau-averaged power features
├── features_eeg_plateau_columns.tsv  # Plateau column metadata
├── features_connectivity.tsv     # Connectivity matrices & metrics
├── features_microstates.tsv      # Microstate temporal dynamics
├── features_precomputed.tsv      # ERD/ERS, spectral, temporal, complexity, global, ROI
├── features_precomputed_columns.tsv  # Precomputed column metadata
├── features_condition.tsv        # Pain vs non-pain specific features
├── features_condition_columns.tsv   # Condition column metadata
├── features_itpc.tsv             # Inter-trial phase coherence
├── features_pac.tsv              # PAC comodulograms
├── features_pac_trials.tsv       # PAC per trial
├── features_pac_time.tsv          # PAC time-resolved values
├── features_all.tsv              # Combined features (power + connectivity)
├── target_vas_ratings.tsv        # Behavioral target vector
├── trial_alignment.tsv           # Trial alignment manifest
├── dropped_trials.tsv            # Dropped trials log
├── wpli_matrices.npz             # Full wPLI matrices per band (if saved)
├── aec_matrices.npz              # Full AEC matrices per band (if saved)
├── pac_comodulograms.npz         # PAC comodulogram arrays (if saved)
└── tfr_complex.h5                # Complex TFR (if saved)
```

---

### Behavior Computation Pipeline

```
run_pipeline.py behavior compute --subject 0001
│
└── process_subject()                         [pipelines/behavior.py]
    │
    ├── LOAD DATA ONCE (BehaviorContext)      [analysis/behavior/core.py]
    │   ├── Load epochs + epochs_info
    │   ├── Load aligned_events
    │   ├── Load power_df, connectivity_df
    │   ├── Load microstates_df, precomputed_df
    │   ├── Load targets (VAS ratings)
    │   ├── Load temperature series
    │   └── Build covariates matrices
    │
    ├── POWER ROI CORRELATIONS                [analysis/behavior/power_roi.py]
    │   ├── compute_power_roi_stats_from_context()
    │   │   ├── ROI-level correlations
    │   │   │   ├── For each band (delta, theta, alpha, beta, gamma):
    │   │   │   │   └── For each ROI (frontal, central, parietal, etc.):
    │   │   │   │       ├── Spearman/Pearson correlation with VAS
    │   │   │   │       ├── Bootstrap confidence intervals
    │   │   │   │       ├── Partial correlation (controlling covariates)
    │   │   │   │       ├── Permutation p-values
    │   │   │   │       └── Temperature correlation
    │   │   │   └── OUTPUT: corr_stats_pow_roi_vs_rating.tsv
    │   │   │              corr_stats_pow_roi_vs_temp.tsv
    │   │   │
    │   │   ├── Channel-level correlations
    │   │   │   └── For each band × channel:
    │   │   │       └── Correlation with VAS + statistics
    │   │   │   └── OUTPUT: corr_stats_pow_channel_vs_rating.tsv
    │   │   │
    │   │   ├── ITPC correlations
    │   │   │   └── For each band × channel:
    │   │   │       └── ITPC correlation with VAS
    │   │   │   └── OUTPUT: corr_stats_itpc_*.tsv
    │   │   │
    │   │   ├── Aperiodic correlations
    │   │   │   ├── Slope correlations per channel
    │   │   │   ├── Offset correlations per channel
    │   │   │   └── ROI-averaged aperiodic
    │   │   │   └── OUTPUT: corr_stats_aper_*.tsv
    │   │   │
    │   │   └── PAC correlations
    │   │       └── PAC strength vs VAS per band pair
    │   │       └── OUTPUT: corr_stats_pac_*.tsv
    │   │
    │   └── Mixed effects models (optional)
    │       └── OUTPUT: mixed_effects_results.tsv
    │
    ├── CONNECTIVITY ROI CORRELATIONS         [analysis/behavior/connectivity.py]
    │   ├── correlate_connectivity_roi_from_context()
    │   │   ├── For each measure (wPLI, AEC, imCoh, PLI):
    │   │   │   └── For each ROI pair (frontal-parietal, etc.):
    │   │   │       ├── Fisher-z transformed mean of edges
    │   │   │       ├── Correlation with VAS
    │   │   │       ├── Bootstrap CI
    │   │   │       ├── Partial correlation
    │   │   │       └── Permutation p-value
    │   │   └── OUTPUT: corr_stats_conn_roi_summary_{measure}_vs_rating.tsv
    │   │              corr_stats_conn_roi_summary_{measure}_vs_temp.tsv
    │   │
    │   ├── correlate_connectivity_heatmaps()
    │   │   ├── Edge-level correlations
    │   │   │   └── For each edge (ch1-ch2):
    │   │   │       └── Correlation with VAS
    │   │   └── OUTPUT: corr_stats_edges_{measure}_vs_rating.tsv
    │   │              conn_heatmap_{measure}.png
    │   │
    │   └── Sliding-window connectivity
    │       ├── Time-resolved connectivity states
    │       └── OUTPUT: sliding_connectivity_*.tsv
    │
    ├── TIME-FREQUENCY CORRELATIONS           [analysis/behavior/temporal.py]
    │   ├── compute_time_frequency_correlations()
    │   │   ├── TFR correlation with VAS at each time-freq point
    │   │   ├── Cluster-based correction
    │   │   └── OUTPUT: corr_stats_tf_clusters_*.tsv
    │   │              tfr_correlation_heatmap.png
    │   │
    │   └── compute_temporal_correlations_by_condition()
    │       ├── Pain vs non-pain temporal dynamics
    │       └── OUTPUT: corr_stats_temporal_*.tsv
    │
    ├── CLUSTER TESTS                         [analysis/behavior/cluster_tests.py]
    │   ├── run_cluster_permutation_tests()
    │   │   ├── Spatio-temporal clusters
    │   │   ├── Pain vs non-pain contrasts
    │   │   └── Permutation-corrected p-values
    │   └── OUTPUT: pain_nonpain_time_clusters_*.tsv
    │
    ├── PRECOMPUTED CORRELATIONS (NEW)        [analysis/behavior/precomputed_correlations.py]
    │   ├── compute_precomputed_correlations()
    │   │   │
    │   │   ├── ERD/ERS correlations
    │   │   │   └── erds_{band}_{channel} vs VAS
    │   │   │   └── OUTPUT: corr_stats_erds_*.tsv
    │   │   │
    │   │   ├── Spectral correlations
    │   │   │   ├── IAF vs VAS
    │   │   │   ├── Relative power vs VAS
    │   │   │   ├── Band ratios vs VAS
    │   │   │   └── Spectral entropy vs VAS
    │   │   │   └── OUTPUT: corr_stats_spectral_*.tsv
    │   │   │
    │   │   ├── Temporal/statistical correlations
    │   │   │   ├── Mean, variance, skewness, kurtosis vs VAS
    │   │   │   ├── RMS, peak-to-peak vs VAS
    │   │   │   └── Zero crossings, line length vs VAS
    │   │   │   └── OUTPUT: corr_stats_temporal_stat_*.tsv
    │   │   │
    │   │   ├── Complexity correlations
    │   │   │   ├── Permutation entropy vs VAS
    │   │   │   ├── Hjorth parameters vs VAS
    │   │   │   └── Lempel-Ziv complexity vs VAS
    │   │   │   └── OUTPUT: corr_stats_complexity_*.tsv
    │   │   │
    │   │   ├── Global feature correlations
    │   │   │   ├── GFP statistics vs VAS
    │   │   │   └── Global PLV vs VAS
    │   │   │   └── OUTPUT: corr_stats_gfp_*.tsv
    │   │   │
    │   │   └── ROI feature correlations
    │   │       ├── ROI-averaged power vs VAS
    │   │       └── Hemispheric asymmetry vs VAS
    │   │       └── OUTPUT: corr_stats_roi_*.tsv
    │   │
    │   └── correlate_microstate_features()
    │       ├── Coverage, duration, occurrence vs VAS
    │       ├── Transition probabilities vs VAS
    │       └── GEV vs VAS
    │       └── OUTPUT: corr_stats_microstates_*.tsv
    │
    ├── CONDITION-SPECIFIC CORRELATIONS       [analysis/behavior/condition_correlations.py]
    │   ├── compute_condition_correlations()
    │   │   │
    │   │   ├── Split trials by pain/non-pain condition
    │   │   │
    │   │   ├── Power correlations by condition
    │   │   │   ├── Power vs VAS (pain trials only)
    │   │   │   ├── Power vs VAS (non-pain trials only)
    │   │   │   └── OUTPUT: corr_stats_power_pain_*.tsv
    │   │   │              corr_stats_power_nonpain_*.tsv
    │   │   │
    │   │   ├── Connectivity correlations by condition
    │   │   │   └── OUTPUT: corr_stats_connectivity_pain_*.tsv
    │   │   │              corr_stats_connectivity_nonpain_*.tsv
    │   │   │
    │   │   ├── Precomputed correlations by condition
    │   │   │   └── OUTPUT: corr_stats_precomputed_pain_*.tsv
    │   │   │              corr_stats_precomputed_nonpain_*.tsv
    │   │   │
    │   │   ├── Microstate correlations by condition
    │   │   │   └── OUTPUT: corr_stats_microstates_pain_*.tsv
    │   │   │              corr_stats_microstates_nonpain_*.tsv
    │   │   │
    │   │   └── Condition comparison
    │   │       ├── r_diff = r_pain - r_nonpain
    │   │       ├── Fisher z-test for difference
    │   │       ├── Condition-specificity labels
    │   │       └── OUTPUT: corr_stats_*_condition_comparison_*.tsv
    │   │
    │   └── Combined summary
    │       └── OUTPUT: corr_stats_all_pain_*.tsv
    │                  corr_stats_all_nonpain_*.tsv
    │                  corr_stats_condition_comparison_all_*.tsv
    │
    ├── GLOBAL FDR CORRECTION                 [analysis/behavior/fdr_correction.py]
    │   ├── apply_global_fdr()
    │   │   ├── Collect ALL p-values across analyses
    │   │   ├── Apply Benjamini-Hochberg FDR
    │   │   └── Add q-values to all TSV files
    │   └── OUTPUT: global_fdr_results.tsv
    │              *_fdr.tsv (updated files)
    │
    └── EXPORT SIGNIFICANT PREDICTORS         [analysis/behavior/exports.py]
        ├── export_all_significant_predictors()
        │   ├── Filter q < 0.05
        │   ├── Combine across analyses
        │   └── Sort by effect size
        └── OUTPUT: significant_predictors.tsv
                   significant_predictors_summary.tsv
```

### Output Files (Behavior Computation)

```
derivatives/sub-{subject}/eeg/stats/
│
├── Power Correlations
│   ├── corr_stats_pow_roi_vs_rating.tsv
│   ├── corr_stats_pow_roi_vs_temp.tsv
│   ├── corr_stats_pow_channel_vs_rating.tsv
│   └── mixed_effects_results.tsv
│
├── Connectivity Correlations
│   ├── corr_stats_conn_roi_summary_wpli_vs_rating.tsv
│   ├── corr_stats_conn_roi_summary_aec_vs_rating.tsv
│   ├── corr_stats_edges_wpli_vs_rating.tsv
│   └── sliding_connectivity_*.tsv
│
├── Phase Correlations
│   ├── corr_stats_itpc_*.tsv
│   └── corr_stats_pac_*.tsv
│
├── Aperiodic Correlations
│   ├── corr_stats_aper_slope_*.tsv
│   └── corr_stats_aper_offset_*.tsv
│
├── Time-Frequency Correlations
│   ├── corr_stats_tf_clusters_*.tsv
│   ├── corr_stats_temporal_*.tsv
│   └── pain_nonpain_time_clusters_*.tsv
│
├── Precomputed Feature Correlations (Rating)
│   ├── corr_stats_precomputed_vs_rating_spearman.tsv    # All combined
│   ├── corr_stats_microstates_vs_rating_spearman.tsv
│   ├── corr_stats_erds_vs_rating_spearman.tsv
│   ├── corr_stats_spectral_vs_rating_spearman.tsv
│   ├── corr_stats_complexity_vs_rating_spearman.tsv
│   ├── corr_stats_gfp_vs_rating_spearman.tsv
│   └── corr_stats_roi_power_vs_rating_spearman.tsv
│
├── Precomputed Feature Correlations (Temperature)
│   ├── corr_stats_precomputed_vs_temp_spearman.tsv
│   ├── corr_stats_microstates_vs_temp_spearman.tsv
│   ├── corr_stats_erds_vs_temp_spearman.tsv
│   ├── corr_stats_spectral_vs_temp_spearman.tsv
│   ├── corr_stats_complexity_vs_temp_spearman.tsv
│   ├── corr_stats_gfp_vs_temp_spearman.tsv
│   └── corr_stats_roi_power_vs_temp_spearman.tsv
│
├── Condition-Specific Correlations (Pain vs Non-Pain vs Rating)
│   ├── corr_stats_power_pain_spearman.tsv
│   ├── corr_stats_power_nonpain_spearman.tsv
│   ├── corr_stats_connectivity_pain_spearman.tsv
│   ├── corr_stats_connectivity_nonpain_spearman.tsv
│   ├── corr_stats_precomputed_pain_spearman.tsv
│   ├── corr_stats_precomputed_nonpain_spearman.tsv
│   ├── corr_stats_microstates_pain_spearman.tsv
│   ├── corr_stats_microstates_nonpain_spearman.tsv
│   ├── corr_stats_all_pain_spearman.tsv           # Combined pain
│   ├── corr_stats_all_nonpain_spearman.tsv        # Combined non-pain
│   └── corr_stats_condition_comparison_all_spearman.tsv  # Comparison
│
├── Condition-Specific Correlations (Pain vs Non-Pain vs Temperature)
│   ├── corr_stats_power_vs_temp_pain_spearman.tsv
│   ├── corr_stats_power_vs_temp_nonpain_spearman.tsv
│   ├── corr_stats_precomputed_vs_temp_pain_spearman.tsv
│   ├── corr_stats_precomputed_vs_temp_nonpain_spearman.tsv
│   ├── corr_stats_all_vs_temp_pain_spearman.tsv
│   └── corr_stats_all_vs_temp_nonpain_spearman.tsv
│
├── FDR-Corrected Results
│   └── global_fdr_results.tsv
│
└── Summary
    ├── significant_predictors.tsv
    └── significant_predictors_summary.tsv

derivatives/sub-{subject}/eeg/plots/behavior/
│
├── Scatter Plots
│   ├── psychometrics/                 # Rating vs temperature
│   ├── overall/                       # Overall power vs behavior
│   └── roi_scatters/{roi}/            # ROI-specific scatters
│
├── Topomaps
│   ├── sub-*_temporal_correlations_by_temperature_*.png
│   ├── sub-*_temporal_correlations_by_pain_*.png
│   └── sub-*_pain_nonpain_cluster_*.png
│
├── Heatmaps
│   ├── sub-*_pac_comod_*.png          # PAC comodulograms
│   └── sub-*_feature_correlation_matrix.png  # (NEW)
│
├── Forest Plots
│   ├── effect_size_forest.png
│   ├── mixed_effects_forest.png
│   └── bootstrap_ci_comparison.png
│
├── Distribution Plots (NEW)
│   ├── sub-*_feature_distributions.png    # Violin plots
│   ├── sub-*_behavioral_summary.png       # Rating/temp summary
│   └── sub-*_features_by_condition.png    # Condition comparison
│
├── Dashboards (NEW)
│   ├── sub-*_analysis_dashboard.png       # 4-panel summary
│   └── sub-*_quality_overview.png         # Quality metrics
│
└── Diagnostic
    ├── residual_qc_*.png              # Regression diagnostics
    └── residual_diagnostics_*.png
```

---

### TSV Column Structure

All correlation output files share a consistent structure:

| Column | Description | Example |
|--------|-------------|---------|
| `{identifier}` | Feature identifier (channel, ROI, edge) | `Cz`, `frontal`, `Fz-Pz` |
| `band` | Frequency band | `alpha`, `theta`, `N/A` |
| `r` | Correlation coefficient | `0.342` |
| `p` | Raw p-value | `0.0012` |
| `q` | FDR-corrected p-value | `0.0089` |
| `n` | Number of valid samples | `156` |
| `method` | Correlation method | `spearman`, `pearson` |
| `ci_low` | 95% CI lower bound | `0.182` |
| `ci_high` | 95% CI upper bound | `0.489` |
| `r_partial` | Partial correlation (controlling covariates) | `0.298` |
| `p_partial` | Partial correlation p-value | `0.0034` |
| `p_perm` | Permutation p-value | `0.002` |
| `analysis` | Analysis type | `erds`, `complexity`, `gfp` |

---

## Installation

### Python Environment (recommended)

```bash
cd /path/to/EEG_fMRI_Pipeline

# Create virtual environment with Python 3.11
/opt/homebrew/bin/python3.11 -m venv .venv311

# Activate environment
source .venv311/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- MNE-Python for EEG analysis
- scikit-learn for machine learning
- See `requirements.txt` for full list

## Configuration

All analysis parameters are centralized in `eeg_pipeline/utils/config/eeg_config.yaml`.

### Key Configuration Sections

| Section | Description |
|---------|-------------|
| `project` | Root paths, task name, subject list |
| `frequency_bands` | EEG frequency band definitions |
| `rois` | Region of interest channel definitions |
| `time_frequency_analysis` | TFR and baseline parameters |
| `feature_engineering` | Feature extraction settings |
| `behavior_analysis` | Correlation and statistics settings |
| `decoding` | ML model hyperparameters |
| `plotting` | Visualization settings |

### Example: Modifying Frequency Bands

```yaml
frequency_bands:
  delta: [1.0, 3.9]
  theta: [4.0, 7.9]
  alpha: [8.0, 12.9]
  beta: [13.0, 30.0]
  gamma: [30.1, 80.0]
```

## Usage

### Unified CLI

All analyses are run through `run_pipeline.py`:

```bash
python eeg_pipeline/scripts/run_pipeline.py <command> <mode> [options]
```

### Preprocessing

**Convert BrainVision to BIDS:**
```bash
python eeg_pipeline/preprocessing/raw_to_bids.py \
    --source_root data/source_data \
    --bids_root data/bids_output \
    --task thermalactive \
    --montage easycap-M1
```

**Merge behavioral data:**
```bash
python eeg_pipeline/preprocessing/merge_behavior_to_events.py \
    --bids_root data/bids_output \
    --source_root data/source_data \
    --task thermalactive
```

### Feature Extraction

```bash
# Extract features for a single subject
python eeg_pipeline/scripts/run_pipeline.py features compute --subject 0001

# Extract specific feature categories
python eeg_pipeline/scripts/run_pipeline.py features compute \
    --subject 0001 \
    --feature-categories power connectivity microstates

# Extract for all subjects
python eeg_pipeline/scripts/run_pipeline.py features compute --all-subjects
```

**Available feature categories:**

| Category | Features Extracted | Output File |
|----------|-------------------|-------------|
| `power` | Band power (δ, θ, α, β, γ) per channel | `features_eeg_direct.tsv` |
| `connectivity` | wPLI, AEC, imCoh, PLI, graph metrics | `features_connectivity.tsv` |
| `microstates` | Coverage, duration, transitions, GEV | `features_microstates.tsv` |
| `aperiodic` | 1/f slope, offset, corrected power | `features_eeg_direct.tsv` |
| `itpc` | Inter-trial phase coherence | `features_eeg_direct.tsv` |
| `pac` | Phase-amplitude coupling | `features_pac_trials.tsv` |
| `precomputed` | ERD/ERS, spectral, temporal, complexity, global, ROI | `features_precomputed.tsv` |
| `condition` | **Pain vs non-pain specific features** | `features_condition.tsv` |

**Condition-specific features (`condition` category):**

Extracts features separately for pain and non-pain trials, creating columns like:
- `iaf_Cz_pain`, `iaf_Cz_nonpain` (Individual Alpha Frequency)
- `pe_Cz_pain`, `pe_Cz_nonpain` (Permutation Entropy)
- `erds_alpha_Cz_pain`, `erds_alpha_Cz_nonpain` (ERD/ERS)
- etc.

This allows ML models to learn from condition-specific neural patterns.

### Behavioral Correlations

```bash
# Compute all correlations
python eeg_pipeline/scripts/run_pipeline.py behavior compute \
    --subject 0001 \
    --correlation-method spearman \
    --n-perm 1000

# Compute only specific analyses
python eeg_pipeline/scripts/run_pipeline.py behavior compute \
    --subject 0001 \
    --computations power_roi connectivity_roi condition_correlations

# Available computations:
#   power_roi, connectivity_roi, connectivity_heatmaps, sliding_connectivity,
#   time_frequency, temporal_correlations, cluster_test, 
#   precomputed_correlations, condition_correlations, exports

# Visualize results (all plots)
python eeg_pipeline/scripts/run_pipeline.py behavior visualize --subject 0001

# Visualize specific plots only
python eeg_pipeline/scripts/run_pipeline.py behavior visualize --subject 0001 \
    --plots psychometrics dashboard feature_distributions

# Quick scatter plots only
python eeg_pipeline/scripts/run_pipeline.py behavior visualize --subject 0001 --scatter-only

# Aggregate across subjects
python eeg_pipeline/scripts/run_pipeline.py behavior aggregate --all-subjects
```

**Available visualization plots:**

| Plot | Category | Description |
|------|----------|-------------|
| `psychometrics` | scatter | Temperature vs rating psychometric curve |
| `power_roi_scatter` | scatter | Power-behavior scatter by ROI/band |
| `temporal_topomaps_temp` | topomap | Temporal correlations by temperature |
| `temporal_topomaps_pain` | topomap | Temporal correlations by pain condition |
| `pac_behavior` | heatmap | Phase-amplitude coupling correlations |
| `pain_clusters` | topomap | Pain vs non-pain cluster visualization |
| `effect_size_forest` | forest | Effect sizes with 95% CI |
| `mediation` | diagram | Mediation analysis path diagrams |
| `mixed_effects` | forest | Mixed-effects model results |
| `bootstrap_ci` | forest | Bootstrap CI comparison |
| `feature_distributions` | distribution | Feature violin plots (NEW) |
| `feature_correlations` | heatmap | Feature correlation matrix (NEW) |
| `condition_comparison` | comparison | Features by condition (NEW) |
| `dashboard` | summary | Comprehensive 4-panel dashboard (NEW) |
| `quality_overview` | summary | Feature quality metrics (NEW) |

**Condition-specific correlations (`condition_correlations`):**

Computes brain-behavior correlations separately for pain and non-pain trials:
- Identifies condition-specific neural predictors
- Compares correlation strength between conditions (Fisher z-test)
- Labels features as "pain_specific", "nonpain_specific", "both", or "neither"

### ERP Analysis

```bash
# Compute ERP statistics
python eeg_pipeline/scripts/run_pipeline.py erp compute --subject 0001

# Visualize ERPs
python eeg_pipeline/scripts/run_pipeline.py erp visualize \
    --subject 0001 \
    --crop-tmin -0.2 \
    --crop-tmax 1.0
```

### Time-Frequency Analysis

```bash
# Visualize TFR
python eeg_pipeline/scripts/run_pipeline.py tfr visualize --subject 0001

# Group-level TFR
python eeg_pipeline/scripts/run_pipeline.py tfr visualize --all-subjects --do-group
```

### Decoding Analysis

```bash
# Run LOSO decoding with permutation testing
python eeg_pipeline/scripts/run_pipeline.py decoding \
    --subject 0001 --subject 0002 \
    --n-perm 100 \
    --inner-splits 5

# Skip time-generalization (faster)
python eeg_pipeline/scripts/run_pipeline.py decoding \
    --all-subjects \
    --skip-time-gen
```

## Output Structure

```
data/derivatives/
├── preprocessed/           # Cleaned epochs
│   └── sub-*/eeg/
├── sub-*/eeg/              # Subject-level results
│   ├── features/           # Extracted features
│   │   ├── features_eeg_direct.tsv
│   │   ├── features_eeg_plateau.tsv
│   │   ├── features_connectivity.tsv
│   │   ├── features_microstates.tsv
│   │   ├── features_precomputed.tsv
│   │   ├── features_condition.tsv
│   │   ├── features_itpc.tsv
│   │   ├── features_pac*.tsv
│   │   ├── features_all.tsv
│   │   └── target_vas_ratings.tsv
│   ├── plots/              # Visualizations
│   └── stats/              # Statistical outputs
├── group/eeg/stats/        # Group-level results
└── decoding/               # Decoding results
    ├── regression/
    └── time_generalization/
```

## Statistical Methods

### FDR Correction
All p-values are corrected using Benjamini-Hochberg FDR at α=0.05.

### Permutation Testing
Null distributions are generated via label shuffling (within-subject for behavioral correlations, trial permutation for PAC).

### Cross-Validation
Decoding uses nested leave-one-subject-out (LOSO) cross-validation with inner GridSearchCV for hyperparameter tuning.

## Extending the Pipeline

### Adding New Features

1. Create extraction function in `analysis/features/`
2. Register in `pipelines/features.py`
3. Add to `FEATURE_CATEGORIES` in `scripts/run_pipeline.py`

### Adding New Analyses

1. Create analysis module in `analysis/`
2. Create pipeline wrapper in `pipelines/`
3. Add subcommand in `scripts/run_pipeline.py`

<<<<<<< HEAD
=======
### Adding New Visualizations

The behavioral plotting module uses a registry pattern for extensibility:

```python
from eeg_pipeline.plotting.behavioral import (
    visualize_subject_behavior,
    AVAILABLE_PLOTS,
    PlotRegistry,
    plot_registry,
)

# List all available plot types
print(list(AVAILABLE_PLOTS.keys()))

# Run specific plots programmatically
visualize_subject_behavior(
    subject="0001",
    task="thermalactive",
    config=config,
    logger=logger,
    plots=["dashboard", "feature_distributions", "effect_size_forest"],
)

# Create custom visualizations using low-level builders
from eeg_pipeline.plotting.behavioral import (
    plot_feature_distributions,
    plot_analysis_dashboard,
    plot_correlation_forest,
)
```

**Key visualization modules:**

| Module | Functions |
|--------|-----------|
| `distributions` | `plot_feature_distributions`, `plot_raincloud`, `plot_behavioral_summary`, `plot_feature_by_condition`, `plot_feature_correlation_matrix` |
| `summary` | `plot_analysis_dashboard`, `plot_group_summary_dashboard`, `plot_quality_overview` |
| `effect_sizes` | `plot_correlation_forest`, `plot_effect_size_comparison`, `plot_effect_size_heatmap` |
| `mediation` | `plot_mediation_diagram`, `plot_mediation_summary`, `plot_mediation_paths_grid` |
| `mixed_effects` | `plot_icc_bar_chart`, `plot_variance_decomposition`, `plot_mixed_effects_forest` |
| `robust` | `plot_outlier_influence`, `plot_bootstrap_ci_comparison`, `plot_sensitivity_analysis` |

## Troubleshooting
>>>>>>> cad687c (Sync local EEG pipeline updates with Thermal Pain repo)

