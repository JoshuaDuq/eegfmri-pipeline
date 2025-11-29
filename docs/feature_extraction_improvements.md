# EEG Feature Extraction Pipeline: Expert Review & Improvements

## Overview

This document provides a comprehensive review of the EEG feature extraction pipeline for thermal pain research, with specific recommendations for improvements. Each recommendation includes theoretical justification based on pain neuroscience literature.

---

## Table of Contents

1. [Current Pipeline Assessment](#current-pipeline-assessment)
2. [Theoretical Framework: EEG Signatures of Pain](#theoretical-framework-eeg-signatures-of-pain)
3. [Feature Improvements by Domain](#feature-improvements-by-domain)
4. [Output Organization](#output-organization)
5. [Implementation Priority](#implementation-priority)
6. [Naming Convention](#naming-convention)
7. [Quality Control](#quality-control)

---

## Current Pipeline Assessment

### Strengths
- Comprehensive coverage of major EEG feature domains
- Efficient precomputation architecture (compute once, reuse)
- Good baseline normalization practices
- Multiple connectivity measures (wPLI, PLV, AEC)
- Microstate analysis included
- Aperiodic (1/f) component extraction

### Gaps Identified
- Insufficient temporal resolution for HRF modeling
- Missing pain-specific latency features
- Incomplete baseline-relative normalization
- No feature manifest/schema documentation
- Inconsistent naming conventions
- Missing cross-frequency coupling beyond PAC
- No time-resolved aperiodic features
- Incomplete ROI × time × band feature matrix

---

## Theoretical Framework: EEG Signatures of Pain

### Pain Processing Stages in Thermal Stimulation

| Time Window | Process | EEG Signature | Key Bands |
|-------------|---------|---------------|-----------|
| **0-500ms** | Nociceptive input | Vertex potential (Cz), N1/P1 | Broadband |
| **500-1500ms** | Pain perception | Alpha ERD, Gamma burst | α, γ |
| **1500-3000ms** | Affective processing | Theta increase, Beta ERD | θ, β |
| **3000-10500ms** | Sustained pain (plateau) | Alpha suppression, Beta modulation | α, β |
| **Post-stimulus** | Recovery | Beta rebound, Alpha recovery | α, β |

### Key Neural Oscillatory Markers of Pain

#### Delta (1-4 Hz)
- **Role**: Salience detection, arousal modulation
- **Pain signature**: Increased delta in frontal regions during intense pain
- **Theoretical basis**: Delta reflects thalamocortical gating and may index pain-related arousal changes
- **References**: Nir et al., 2012; Ploner et al., 2017

#### Theta (4-8 Hz)
- **Role**: Pain memory, affective processing, attention
- **Pain signature**: Frontal theta increase correlates with pain unpleasantness
- **Theoretical basis**: Theta oscillations in ACC/mPFC reflect affective-motivational aspects of pain
- **References**: Schulz et al., 2015; Misra et al., 2017

#### Alpha (8-13 Hz)
- **Role**: Cortical inhibition, sensory gating
- **Pain signature**: Contralateral alpha ERD in sensorimotor cortex
- **Theoretical basis**: Alpha suppression reflects disinhibition of pain-processing regions
- **References**: Ploner et al., 2006; May et al., 2012; Tu et al., 2016

#### Beta (13-30 Hz)
- **Role**: Sensorimotor processing, top-down control
- **Pain signature**: 
  - Beta ERD during pain
  - Beta rebound post-stimulus (motor inhibition)
- **Theoretical basis**: Beta reflects sensorimotor engagement and subsequent inhibition
- **References**: Hauck et al., 2015; Misra et al., 2017

#### Gamma (30-100 Hz)
- **Role**: Local cortical processing, pain intensity encoding
- **Pain signature**: Gamma increase in S1/S2 correlates with pain intensity
- **Theoretical basis**: Gamma reflects local neuronal synchronization in pain matrix
- **References**: Gross et al., 2007; Zhang et al., 2012; Schulz et al., 2015

### Pain-Relevant Brain Regions (ROIs)

| ROI | Function in Pain | Expected EEG Pattern |
|-----|------------------|---------------------|
| **Sensorimotor (S1/M1)** | Sensory-discriminative | Contralateral α/β ERD, γ increase |
| **Frontal (ACC/mPFC)** | Affective-motivational | θ increase, α ERD |
| **Parietal (S2/PPC)** | Spatial attention, integration | α ERD, γ increase |
| **Temporal (Insula proxy)** | Interoception, salience | θ/α modulation |
| **Occipital** | Visual attention (control) | Minimal pain-related change |

---

## Feature Improvements by Domain

### 1. Power Features

#### Current Implementation
- Band power per channel per temporal bin (early/mid/late)
- Baseline power extraction
- Log-ratio normalization

#### Missing Features (Theoretically Justified)

##### 1.1 Time-Resolved Power (Sliding Window)
```
power_{band}_{channel}_t{1-7}_mean
power_{band}_{channel}_t{1-7}_std
```
**Justification**: Pain processing evolves over seconds. The hemodynamic response function (HRF) peaks at ~5-6s, requiring sub-second EEG features to model BOLD dynamics. Sliding windows (1s width, 0.5s step) capture this temporal evolution.

**Pain relevance**: 
- Alpha ERD onset and recovery timing differ between pain and non-pain
- Beta rebound timing correlates with pain relief
- Gamma bursts are transient and require fine temporal resolution

##### 1.2 Temporal Dynamics Features
```
power_{band}_{channel}_slope           # Linear trend over plateau
power_{band}_{channel}_early_late_diff # Late - Early difference  
power_{band}_{channel}_peak_time       # Time of maximum power
power_{band}_{channel}_trough_time     # Time of minimum power
power_{band}_{channel}_auc             # Area under curve (total power)
```
**Justification**: Pain adaptation and sensitization manifest as temporal trends. Slope captures whether pain response is building (sensitization) or declining (habituation).

**Pain relevance**:
- Positive alpha slope = recovery from pain
- Negative beta slope = sustained motor inhibition
- Peak time differences distinguish acute vs. tonic pain responses

##### 1.3 Baseline-Relative Normalization Variants
```
power_{band}_{channel}_{time}_zscore      # (power - baseline_mean) / baseline_std
power_{band}_{channel}_{time}_percent     # (power - baseline) / baseline * 100
power_{band}_{channel}_{time}_logratio    # log10(power / baseline)
power_{band}_{channel}_{time}_db          # 10 * log10(power / baseline)
```
**Justification**: Different normalization methods have different statistical properties. Z-score is robust to baseline variance; log-ratio is standard for TFR; dB is intuitive for power.

**Pain relevance**: Individual differences in baseline power are substantial; normalization enables cross-subject comparison.

---

### 2. ERD/ERS Features

#### Current Implementation
- Mean ERD/ERS per channel/band/time
- Temporal statistics (std, min, max, range)
- Percentiles, slope, peak/onset latency

#### Missing Features (Theoretically Justified)

##### 2.1 ERD vs ERS Separation
```
erds_{band}_{channel}_{time}_erd_magnitude   # Mean of negative values only
erds_{band}_{channel}_{time}_ers_magnitude   # Mean of positive values only
erds_{band}_{channel}_{time}_erd_duration    # Time spent in ERD
erds_{band}_{channel}_{time}_ers_duration    # Time spent in ERS
erds_{band}_{channel}_{time}_erd_onset       # First significant ERD
erds_{band}_{channel}_{time}_ers_onset       # First significant ERS (rebound)
```
**Justification**: ERD and ERS are distinct neural processes. ERD reflects cortical activation; ERS (especially beta rebound) reflects active inhibition. Separating them provides cleaner neural markers.

**Pain relevance**:
- Alpha ERD magnitude correlates with pain intensity
- Beta ERS (rebound) magnitude correlates with pain relief/motor inhibition
- ERD onset latency may distinguish nociceptive from non-nociceptive stimuli

##### 2.2 Contralateral-Ipsilateral Contrast
```
erds_{band}_contra_ipsi_diff_{time}    # Contralateral - Ipsilateral ERD
erds_{band}_contra_ipsi_ratio_{time}   # Contralateral / Ipsilateral ERD
erds_{band}_lateralization_index_{time} # (C - I) / (C + I)
```
**Justification**: Thermal pain on one hand produces contralateral sensorimotor activation. The lateralization index is a robust marker of somatotopic processing.

**Pain relevance**: Lateralization is disrupted in chronic pain conditions; preserved lateralization indicates intact somatosensory processing.

---

### 3. Connectivity Features

#### Current Implementation
- wPLI, PLV, AEC (orthogonalized), imCoh, PLI
- Graph metrics (efficiency, clustering, small-world)
- Sliding window connectivity

#### Missing Features (Theoretically Justified)

##### 3.1 Time-Resolved Connectivity
```
conn_{measure}_{band}_early_mean
conn_{measure}_{band}_mid_mean
conn_{measure}_{band}_late_mean
conn_{measure}_{band}_slope           # Connectivity trend over time
conn_{measure}_{band}_early_late_diff
```
**Justification**: Pain processing involves dynamic network reconfiguration. Early connectivity may reflect bottom-up nociceptive processing; late connectivity may reflect top-down modulation.

**Pain relevance**:
- Theta connectivity increases during sustained pain (affective network)
- Alpha connectivity decreases (sensorimotor disinhibition)
- Beta connectivity changes reflect motor network engagement

##### 3.2 ROI-to-ROI Connectivity
```
conn_{measure}_{band}_{roi1}_to_{roi2}_{time}
```
Key pairs for pain:
- Sensorimotor_L ↔ Sensorimotor_R (interhemispheric)
- Frontal ↔ Sensorimotor (top-down modulation)
- Frontal ↔ Parietal (attention network)
- Sensorimotor ↔ Parietal (sensory integration)

**Justification**: Whole-brain connectivity is high-dimensional; ROI-to-ROI reduces dimensionality while preserving interpretability.

**Pain relevance**: 
- Frontal-sensorimotor theta connectivity correlates with pain modulation
- Reduced interhemispheric connectivity in chronic pain

##### 3.3 Directed Connectivity (Information Flow)
```
conn_psi_{band}_{source}_to_{target}_{time}  # Phase Slope Index
conn_gc_{band}_{source}_to_{target}_{time}   # Granger Causality (spectral)
```
**Justification**: Undirected connectivity doesn't capture information flow direction. PSI and spectral GC reveal whether frontal regions are driving sensorimotor activity (top-down) or vice versa (bottom-up).

**Pain relevance**:
- Bottom-up flow dominates in acute pain
- Top-down flow increases with pain modulation/placebo

##### 3.4 Network Dynamics
```
conn_graph_{band}_efficiency_{time}
conn_graph_{band}_clustering_{time}
conn_graph_{band}_modularity_{time}
conn_graph_{band}_hub_disruption_{time}  # Change in hub structure from baseline
```
**Justification**: Pain alters global network topology. Efficiency decreases may reflect network "overload"; modularity changes may reflect integration/segregation balance.

**Pain relevance**: Chronic pain is associated with altered network topology; acute pain may show transient topology changes.

---

### 4. Phase Features

#### Current Implementation
- ITPC per band/channel/time bin
- Trial-wise ITPC (leave-one-out)
- PAC comodulograms (theta-gamma)

#### Missing Features (Theoretically Justified)

##### 4.1 ITPC Temporal Dynamics
```
phase_itpc_{band}_{channel}_peak_time      # When is phase-locking maximal?
phase_itpc_{band}_{channel}_onset_latency  # First significant ITPC
phase_itpc_{band}_{channel}_duration       # Duration of significant ITPC
phase_itpc_{band}_{channel}_slope          # ITPC trend over time
phase_itpc_{band}_{channel}_auc            # Total phase-locking (integral)
```
**Justification**: ITPC reflects stimulus-locked neural processing. Peak latency indicates when sensory processing is most consistent; duration indicates processing stability.

**Pain relevance**:
- Gamma ITPC peak latency correlates with pain intensity ratings
- Alpha ITPC duration may reflect attention to pain
- Theta ITPC in frontal regions reflects affective processing timing

##### 4.2 Phase Consistency (Single-Trial)
```
phase_consistency_{band}_{channel}_{time}  # Circular variance
phase_concentration_{band}_{channel}_{time} # von Mises kappa
phase_mean_angle_{band}_{channel}_{time}   # Mean phase angle
```
**Justification**: Beyond ITPC (cross-trial), within-trial phase consistency reflects local neural synchronization quality.

**Pain relevance**: Phase consistency in gamma band correlates with subjective pain clarity.

##### 4.3 Cross-Frequency Phase Coupling
```
phase_ppc_theta_alpha_{channel}_{time}   # Theta-alpha phase-phase coupling
phase_ppc_alpha_beta_{channel}_{time}    # Alpha-beta phase-phase coupling
```
**Justification**: Phase-phase coupling reflects coordination between oscillatory processes at different frequencies.

**Pain relevance**: Theta-alpha coupling in frontal regions may coordinate affective and sensory processing.

##### 4.4 PAC Temporal Features
```
pac_{roi}_theta_gamma_{time}           # PAC per time window
pac_{roi}_theta_gamma_peak_time        # When is PAC maximal?
pac_{roi}_theta_gamma_modulation_index # Strength of coupling
pac_{roi}_alpha_gamma_{time}           # Alpha-gamma PAC (sensorimotor)
```
**Justification**: PAC reflects local computation (gamma nested in theta/alpha). Time-resolved PAC captures when this computation is most active.

**Pain relevance**:
- Theta-gamma PAC in frontal regions correlates with pain unpleasantness
- Alpha-gamma PAC in sensorimotor cortex correlates with pain intensity

---

### 5. Aperiodic Features

#### Current Implementation
- 1/f slope and offset from baseline PSD
- Per-channel and global

#### Missing Features (Theoretically Justified)

##### 5.1 Time-Resolved Aperiodic
```
aper_slope_{channel}_{time}           # Slope per time window
aper_offset_{channel}_{time}          # Offset per time window
aper_slope_change_{channel}_{time}    # Slope change from baseline
aper_slope_early_late_diff_{channel}  # Temporal dynamics
```
**Justification**: The aperiodic slope reflects excitation/inhibition (E/I) balance. Pain may shift E/I balance dynamically.

**Pain relevance**:
- Steeper slope (more negative) = more inhibition
- Pain may flatten slope (increased excitation)
- Slope changes correlate with BOLD signal changes

##### 5.2 Aperiodic-Corrected Band Power
```
power_corrected_{band}_{channel}_{time}  # Band power after removing 1/f
```
**Justification**: Raw band power conflates periodic (oscillatory) and aperiodic (1/f) components. Separating them provides cleaner oscillatory measures.

**Pain relevance**: True alpha oscillation (not just 1/f) may be more specifically related to pain processing.

##### 5.3 Aperiodic Knee Frequency
```
aper_knee_{channel}_{time}  # Knee frequency (if using knee model)
```
**Justification**: The knee frequency indicates where the spectrum transitions from flat to 1/f. It may reflect neural time constants.

**Pain relevance**: Knee frequency changes may reflect altered neural dynamics in pain states.

---

### 6. Spectral Shape Features

#### Current Implementation
- Peak frequency per band
- Spectral entropy
- Spectral edge frequencies (50%, 75%, 90%, 95%)
- Spectral slope within band

#### Missing Features (Theoretically Justified)

##### 6.1 Spectral Moments
```
spec_centroid_{band}_{channel}_{time}    # Center of mass frequency
spec_bandwidth_{band}_{channel}_{time}   # Spread around centroid
spec_skewness_{band}_{channel}_{time}    # Asymmetry of spectrum
spec_kurtosis_{band}_{channel}_{time}    # Peakedness of spectrum
```
**Justification**: Spectral shape provides information beyond power. A narrow, peaked spectrum indicates strong oscillation; a broad spectrum indicates noise-like activity.

**Pain relevance**:
- Alpha peak sharpness correlates with attention
- Broader gamma spectrum may indicate more distributed processing

##### 6.2 Individual Alpha Frequency (IAF) Dynamics
```
spec_iaf_{channel}_{time}           # IAF per time window
spec_iaf_shift_{channel}            # IAF change from baseline
spec_iaf_global_{time}              # Global IAF
```
**Justification**: IAF is a stable individual trait but may shift with cognitive state. Pain-related IAF shifts could indicate altered cortical excitability.

**Pain relevance**: IAF slowing has been reported in chronic pain; acute pain effects on IAF are less studied but theoretically relevant.

##### 6.3 Spectral Flux
```
spec_flux_{band}_{channel}  # Change in spectrum over time (temporal derivative)
```
**Justification**: Spectral flux captures how much the spectrum changes over time, indicating dynamic neural processing.

**Pain relevance**: High spectral flux may indicate unstable pain processing or attention fluctuations.

---

### 7. Microstate Features

#### Current Implementation
- Coverage, duration, occurrence, GEV per state
- Transition probabilities
- Valid fraction QC

#### Missing Features (Theoretically Justified)

##### 7.1 Time-Resolved Microstates
```
ms_coverage_{state}_{time}      # Coverage per time window
ms_duration_{state}_{time}      # Mean duration per time window
ms_occurrence_{state}_{time}    # Occurrence rate per time window
ms_dominant_state_{time}        # Most frequent state per window
```
**Justification**: Microstate dynamics evolve over time. Pain may alter which states dominate at different processing stages.

**Pain relevance**:
- Early: Sensory-related microstates may dominate
- Late: Affective/evaluative microstates may increase

##### 7.2 Microstate Sequence Complexity
```
ms_transition_entropy           # Entropy of transition matrix
ms_sequence_entropy             # Entropy of state sequence
ms_hurst_exponent               # Long-range temporal correlations
ms_complexity_lzc               # Lempel-Ziv complexity of sequence
```
**Justification**: Microstate sequences are not random; they have temporal structure. Complexity measures capture this structure.

**Pain relevance**: Altered microstate complexity in pain may reflect disrupted neural dynamics.

##### 7.3 Microstate-Pain Correlations
```
ms_coverage_{state}_pain_correlation
ms_duration_{state}_pain_correlation
```
**Justification**: Pre-computed correlations with pain ratings provide immediate interpretability.

**Pain relevance**: Identifies which microstates are most pain-relevant.

---

### 8. Complexity Features

#### Current Implementation
- Permutation entropy (PE)
- Hjorth parameters (activity, mobility, complexity)
- Lempel-Ziv complexity (LZC)

#### Missing Features (Theoretically Justified)

##### 8.1 Time-Resolved Complexity
```
comp_pe_{band}_{channel}_{time}
comp_hjorth_activity_{band}_{channel}_{time}
comp_hjorth_mobility_{band}_{channel}_{time}
comp_hjorth_complexity_{band}_{channel}_{time}
comp_lzc_{band}_{channel}_{time}
```
**Justification**: Complexity may change over the pain time course, reflecting different processing stages.

**Pain relevance**: Increased complexity during pain may reflect more distributed processing.

##### 8.2 Additional Entropy Measures
```
comp_sample_entropy_{band}_{channel}_{time}
comp_approximate_entropy_{band}_{channel}_{time}
comp_multiscale_entropy_{band}_{channel}_{time}
```
**Justification**: Different entropy measures capture different aspects of signal complexity. Sample entropy is more robust than approximate entropy for short signals.

**Pain relevance**: Entropy changes in pain reflect altered neural dynamics.

##### 8.3 Fractal Dimension
```
comp_hurst_{band}_{channel}           # Hurst exponent (long-range correlations)
comp_dfa_{band}_{channel}             # Detrended fluctuation analysis
comp_higuchi_fd_{band}_{channel}      # Higuchi fractal dimension
```
**Justification**: Fractal measures capture self-similarity across time scales, reflecting neural criticality.

**Pain relevance**: Pain may push the brain away from criticality, altering fractal properties.

---

### 9. Temporal/Waveform Features

#### Current Implementation
- Variance, std, skewness, kurtosis
- RMS, peak-to-peak, MAD
- Zero-crossings, line length, nonlinear energy

#### Missing Features (Theoretically Justified)

##### 9.1 Time-Resolved Waveform Features
```
temp_var_{band}_{channel}_{time}
temp_rms_{band}_{channel}_{time}
temp_skew_{band}_{channel}_{time}
temp_kurt_{band}_{channel}_{time}
```
**Justification**: Waveform statistics may change over the pain time course.

**Pain relevance**: Increased kurtosis may indicate more "peaky" activity during pain.

##### 9.2 Amplitude Dynamics
```
temp_amplitude_range_{band}_{channel}_{time}
temp_amplitude_cv_{band}_{channel}_{time}      # Coefficient of variation
temp_amplitude_stability_{band}_{channel}      # 1 / CV
```
**Justification**: Amplitude stability reflects consistency of neural activity.

**Pain relevance**: Unstable amplitude may indicate fluctuating attention to pain.

---

### 10. Asymmetry/Lateralization Features

#### Current Implementation
- Hemispheric asymmetry for predefined ROI pairs

#### Missing Features (Theoretically Justified)

##### 10.1 Complete Lateralization Matrix
```
asym_{band}_{roi_pair}_{time}           # (R - L) / (R + L)
asym_{band}_frontal_{time}
asym_{band}_central_{time}
asym_{band}_parietal_{time}
asym_{band}_temporal_{time}
asym_{band}_occipital_{time}
```
**Justification**: Lateralization should be computed for all major regions, not just predefined pairs.

**Pain relevance**: Frontal asymmetry relates to approach/avoidance; central asymmetry relates to sensorimotor processing.

##### 10.2 Contralateral vs Ipsilateral (Stimulus-Relative)
```
asym_{band}_contra_ipsi_power_{time}    # Contralateral / Ipsilateral power
asym_{band}_contra_ipsi_erds_{time}     # Contralateral - Ipsilateral ERD
asym_{band}_contra_ipsi_conn_{time}     # Connectivity asymmetry
```
**Justification**: For unilateral stimulation, contralateral vs ipsilateral is more meaningful than left vs right.

**Pain relevance**: Contralateral dominance is a hallmark of intact somatosensory processing.

---

### 11. Global Field Power (GFP) Features

#### Current Implementation
- GFP statistics (mean, std, min, max, range, CV)
- Baseline-normalized GFP
- Peak detection
- Per-window GFP
- Band-specific GFP

#### Missing Features (Theoretically Justified)

##### 11.1 GFP Temporal Dynamics
```
gfp_slope_{time}                    # GFP trend
gfp_peak_count_{time}               # Number of GFP peaks per window
gfp_peak_amplitude_mean_{time}      # Mean peak amplitude
gfp_interpeak_interval_{time}       # Mean time between peaks
```
**Justification**: GFP peaks indicate moments of maximal neural synchronization. Their timing and frequency are informative.

**Pain relevance**: More frequent GFP peaks may indicate more active processing.

##### 11.2 GFP-Microstate Relationship
```
gfp_at_microstate_{state}_mean      # Mean GFP when in each microstate
gfp_at_microstate_{state}_std
```
**Justification**: GFP and microstates are related; GFP peaks are used for microstate clustering. Their relationship during the task is informative.

**Pain relevance**: Higher GFP during pain-related microstates indicates stronger activation.

---

### 12. ROI-Aggregated Features

#### Current Implementation
- ROI power and ERD/ERS

#### Missing Features (Theoretically Justified)

##### 12.1 Complete ROI Feature Set
For each ROI, extract all features that are computed at channel level:
```
roi_power_{band}_{roi}_{time}_mean
roi_power_{band}_{roi}_{time}_std
roi_erds_{band}_{roi}_{time}_mean
roi_erds_{band}_{roi}_{time}_slope
roi_itpc_{band}_{roi}_{time}_mean
roi_conn_within_{band}_{roi}_{time}   # Within-ROI connectivity
roi_complexity_{measure}_{band}_{roi}_{time}
```
**Justification**: ROI aggregation reduces dimensionality while preserving spatial interpretability.

**Pain relevance**: ROIs map to functional brain regions in the pain matrix.

##### 12.2 ROI Definitions for Pain Research
```yaml
rois:
  Sensorimotor_Contra:  # Contralateral to stimulation
    - C3, C1, CP3, CP1 (for right-hand stim)
    - C4, C2, CP4, CP2 (for left-hand stim)
  Sensorimotor_Ipsi:    # Ipsilateral to stimulation
    - C4, C2, CP4, CP2 (for right-hand stim)
    - C3, C1, CP3, CP1 (for left-hand stim)
  Frontal_Midline:      # ACC/mPFC proxy
    - Fz, FCz, Cz
  Frontal_Left:
    - F3, F1, FC3, FC1
  Frontal_Right:
    - F4, F2, FC4, FC2
  Parietal_Midline:
    - Pz, CPz, POz
  Parietal_Left:        # S2/PPC
    - P3, P1, CP3
  Parietal_Right:
    - P4, P2, CP4
  Temporal_Left:        # Insula proxy
    - T7, TP7, FT7
  Temporal_Right:
    - T8, TP8, FT8
  Occipital:            # Control region
    - O1, Oz, O2
```

---

## Output Organization

### Directory Structure
```
derivatives/sub-{id}/eeg/features/
├── features_all.tsv                    # Complete feature matrix (trials × features)
├── features_manifest.json              # Feature schema and descriptions
├── features_metadata.json              # Extraction parameters, software versions
│
├── by_domain/                          # Features organized by domain
│   ├── power.tsv
│   ├── erds.tsv
│   ├── connectivity.tsv
│   ├── phase.tsv
│   ├── aperiodic.tsv
│   ├── spectral.tsv
│   ├── microstates.tsv
│   ├── complexity.tsv
│   ├── temporal.tsv
│   ├── asymmetry.tsv
│   └── gfp.tsv
│
├── by_band/                            # Features organized by frequency band
│   ├── delta.tsv
│   ├── theta.tsv
│   ├── alpha.tsv
│   ├── beta.tsv
│   └── gamma.tsv
│
├── by_time/                            # Features organized by time window
│   ├── baseline.tsv
│   ├── early.tsv
│   ├── mid.tsv
│   └── late.tsv
│
├── by_roi/                             # Features organized by ROI
│   ├── sensorimotor_contra.tsv
│   ├── sensorimotor_ipsi.tsv
│   ├── frontal_midline.tsv
│   └── ...
│
├── connectivity_matrices/              # Full connectivity matrices
│   ├── wpli_{band}_{time}.npy
│   ├── plv_{band}_{time}.npy
│   └── ...
│
├── qc/                                 # Quality control
│   ├── feature_completeness.json
│   ├── nan_report.tsv
│   ├── outlier_report.tsv
│   └── extraction_log.txt
│
└── fmri_regressors/                    # fMRI-ready regressors
    ├── sub-{id}_task-{task}_regressor_{feature}.tsv
    └── ...
```

### Feature Manifest Schema
```json
{
  "version": "2.0.0",
  "extraction_date": "2024-01-15T10:30:00Z",
  "software_version": "eeg_pipeline 1.0.0",
  "mne_version": "1.6.0",
  "features": [
    {
      "name": "power_alpha_Cz_early_mean",
      "domain": "power",
      "band": "alpha",
      "band_range": [8.0, 13.0],
      "location": "Cz",
      "location_type": "channel",
      "time_window": "early",
      "time_range": [3.0, 5.0],
      "statistic": "mean",
      "unit": "µV²",
      "normalization": "log-ratio to baseline",
      "description": "Mean alpha power at Cz during early plateau window, log-ratio normalized to baseline",
      "theoretical_relevance": "Alpha ERD in central regions reflects sensorimotor processing of pain",
      "expected_direction": "negative (ERD) for pain vs non-pain"
    }
  ]
}
```

---

## Implementation Priority

### Phase 1: Critical (High Impact, Moderate Effort)
| Feature | Justification | Files |
|---------|---------------|-------|
| Time-resolved power (7 bins) | Essential for HRF modeling | `power.py` |
| Temporal dynamics (slope, diff) | Captures pain adaptation | `pipeline.py` |
| ERD/ERS separation | Cleaner neural markers | `pipeline.py` |
| Time-resolved connectivity | Network dynamics | `connectivity.py` |
| Feature manifest generator | Documentation | New `manifest.py` |

### Phase 2: Important (High Impact, Higher Effort)
| Feature | Justification | Files |
|---------|---------------|-------|
| ITPC temporal dynamics | Phase-locking timing | `phase.py` |
| Time-resolved aperiodic | E/I balance dynamics | `aperiodic.py` |
| ROI-to-ROI connectivity | Interpretable networks | `connectivity.py` |
| Complete ROI features | Dimensionality reduction | `pipeline.py` |
| Microstate temporal bins | State dynamics | `microstates.py` |

### Phase 3: Valuable (Moderate Impact)
| Feature | Justification | Files |
|---------|---------------|-------|
| PAC temporal features | Cross-frequency dynamics | `phase.py` |
| Spectral shape features | Oscillation quality | `spectral.py` |
| Directed connectivity | Information flow | `connectivity.py` |
| Additional complexity | Entropy variants | `complexity.py` |
| Lateralization complete | All ROI pairs | `pipeline.py` |

### Phase 4: Nice-to-Have (Lower Priority)
| Feature | Justification | Files |
|---------|---------------|-------|
| Phase-phase coupling | Advanced cross-frequency | `phase.py` |
| Fractal dimension | Neural criticality | `complexity.py` |
| Aperiodic knee | Neural time constants | `aperiodic.py` |
| GFP-microstate relationship | Integration | `microstates.py` |

---

## Naming Convention

### Standard Format
```
{domain}_{measure}_{band}_{location}_{time}_{statistic}
```

### Domain Prefixes
| Prefix | Domain |
|--------|--------|
| `power` | Band power |
| `erds` | Event-related desynchronization/synchronization |
| `conn` | Connectivity |
| `phase` | Phase features (ITPC, consistency) |
| `pac` | Phase-amplitude coupling |
| `aper` | Aperiodic (1/f) |
| `spec` | Spectral shape |
| `ms` | Microstates |
| `comp` | Complexity |
| `temp` | Temporal/waveform |
| `asym` | Asymmetry/lateralization |
| `gfp` | Global field power |
| `roi` | ROI-aggregated |

### Measure Names
| Measure | Description |
|---------|-------------|
| `mean`, `std`, `max`, `min` | Basic statistics |
| `cv` | Coefficient of variation |
| `zscore` | Z-score normalized |
| `percent` | Percent change |
| `logratio` | Log ratio |
| `slope` | Temporal trend |
| `diff` | Difference (e.g., early-late) |
| `onset` | Onset latency |
| `peak` | Peak latency or value |
| `duration` | Duration |
| `auc` | Area under curve |
| `wpli`, `plv`, `aec` | Connectivity measures |
| `pe`, `lzc` | Complexity measures |

### Location Names
| Type | Examples |
|------|----------|
| Channel | `Cz`, `Fz`, `C3` |
| ROI | `Sensorimotor_Contra`, `Frontal_Midline` |
| Global | `global` |
| Pair | `Fz_Cz`, `Frontal_to_Parietal` |

### Time Windows
| Label | Range (s) | Description |
|-------|-----------|-------------|
| `baseline` | [-5.0, -0.01] | Pre-stimulus |
| `early` | [3.0, 5.0] | Early plateau |
| `mid` | [5.0, 7.5] | Middle plateau |
| `late` | [7.5, 10.5] | Late plateau |
| `t1`-`t7` | 1s bins | Fine temporal resolution |
| `full` | [3.0, 10.5] | Full plateau |

---

## Quality Control

### Feature Completeness Validation
```python
def validate_feature_completeness(
    df: pd.DataFrame,
    expected_features: List[str],
    logger: Any,
) -> Dict[str, Any]:
    """
    Validate that all expected features are present and have valid values.
    
    Returns:
        {
            "complete": bool,
            "missing_features": List[str],
            "nan_features": List[str],
            "nan_fraction": float,
            "constant_features": List[str],
        }
    """
```

### Outlier Detection
```python
def detect_feature_outliers(
    df: pd.DataFrame,
    method: str = "iqr",  # or "zscore", "mad"
    threshold: float = 3.0,
    logger: Any,
) -> pd.DataFrame:
    """
    Detect outliers in feature values.
    
    Returns DataFrame with outlier flags per feature.
    """
```

### QC Report Generation
```python
def generate_qc_report(
    features_df: pd.DataFrame,
    manifest: Dict,
    output_dir: Path,
    logger: Any,
) -> None:
    """
    Generate comprehensive QC report including:
    - Feature completeness
    - NaN analysis
    - Outlier detection
    - Distribution summaries
    - Correlation with expected patterns
    """
```

---

## References

1. Ploner, M., Sorg, C., & Gross, J. (2017). Brain rhythms of pain. Trends in Cognitive Sciences, 21(2), 100-110.
2. Schulz, E., et al. (2015). Prefrontal gamma oscillations encode tonic pain in humans. Cerebral Cortex, 25(11), 4407-4414.
3. May, E. S., et al. (2012). Pre-stimulus alpha power predicts subjective pain intensity. Journal of Neuroscience, 32(6), 2241-2247.
4. Tu, Y., et al. (2016). Alpha and gamma oscillation amplitudes synergistically predict the perception of forthcoming nociceptive stimuli. Human Brain Mapping, 37(2), 501-514.
5. Hauck, M., et al. (2015). Neuroimage, 108, 144-150.
6. Gross, J., et al. (2007). Gamma oscillations in human primary somatosensory cortex reflect pain perception. PLoS Biology, 5(5), e133.
7. Zhang, Z. G., et al. (2012). Gamma-band oscillations in the primary somatosensory cortex. Journal of Neuroscience, 32(22), 7429-7438.
8. Misra, G., et al. (2017). Cortical oscillations during pain. Neuroscience, 338, 1-14.
9. Nir, R. R., et al. (2012). Pain assessment by continuous EEG. Pain, 153(9), 1863-1870.

---

## Appendix: Complete Feature List

### Power Features (per band × channel × time)
- `power_{band}_{ch}_{time}_mean`
- `power_{band}_{ch}_{time}_std`
- `power_{band}_{ch}_{time}_zscore`
- `power_{band}_{ch}_{time}_logratio`
- `power_{band}_{ch}_slope`
- `power_{band}_{ch}_early_late_diff`
- `power_{band}_{ch}_peak_time`
- `power_{band}_{ch}_auc`
- `power_{band}_global_{time}_mean`
- `power_{band}_global_{time}_std`

### ERD/ERS Features (per band × channel × time)
- `erds_{band}_{ch}_{time}_percent`
- `erds_{band}_{ch}_{time}_zscore`
- `erds_{band}_{ch}_{time}_std`
- `erds_{band}_{ch}_erd_magnitude`
- `erds_{band}_{ch}_ers_magnitude`
- `erds_{band}_{ch}_erd_onset`
- `erds_{band}_{ch}_ers_onset`
- `erds_{band}_{ch}_erd_duration`
- `erds_{band}_{ch}_slope`
- `erds_{band}_{ch}_peak_latency`
- `erds_{band}_contra_ipsi_diff_{time}`
- `erds_{band}_lateralization_{time}`
- `erds_{band}_global_{time}_mean`

### Connectivity Features (per measure × band × time)
- `conn_{measure}_{band}_{time}_mean`
- `conn_{measure}_{band}_{time}_std`
- `conn_{measure}_{band}_{time}_max`
- `conn_{measure}_{band}_slope`
- `conn_{measure}_{band}_{roi1}_to_{roi2}_{time}`
- `conn_graph_{band}_efficiency_{time}`
- `conn_graph_{band}_clustering_{time}`
- `conn_graph_{band}_modularity_{time}`

### Phase Features (per band × channel × time)
- `phase_itpc_{band}_{ch}_{time}_mean`
- `phase_itpc_{band}_{ch}_peak_time`
- `phase_itpc_{band}_{ch}_onset`
- `phase_itpc_{band}_{ch}_duration`
- `phase_itpc_{band}_{ch}_slope`
- `phase_consistency_{band}_{ch}_{time}`
- `pac_{roi}_{phase_band}_{amp_band}_{time}`
- `pac_{roi}_{phase_band}_{amp_band}_peak_time`

### Aperiodic Features (per channel × time)
- `aper_slope_{ch}_{time}`
- `aper_offset_{ch}_{time}`
- `aper_slope_change_{ch}_{time}`
- `aper_slope_global_{time}`
- `aper_r2_{ch}_{time}`

### Spectral Features (per band × channel)
- `spec_peak_freq_{band}_{ch}`
- `spec_peak_power_{band}_{ch}`
- `spec_entropy_{band}_{ch}`
- `spec_centroid_{band}_{ch}`
- `spec_bandwidth_{band}_{ch}`
- `spec_iaf_{ch}_{time}`
- `spec_iaf_shift_{ch}`

### Microstate Features (per state × time)
- `ms_coverage_{state}_{time}`
- `ms_duration_{state}_{time}`
- `ms_occurrence_{state}_{time}`
- `ms_gev_{state}_{time}`
- `ms_trans_{from}_to_{to}_{time}`
- `ms_transition_entropy_{time}`
- `ms_dominant_state_{time}`

### Complexity Features (per band × channel × time)
- `comp_pe_{band}_{ch}_{time}`
- `comp_hjorth_activity_{band}_{ch}_{time}`
- `comp_hjorth_mobility_{band}_{ch}_{time}`
- `comp_hjorth_complexity_{band}_{ch}_{time}`
- `comp_lzc_{band}_{ch}_{time}`
- `comp_sample_entropy_{band}_{ch}_{time}`

### Temporal Features (per band × channel × time)
- `temp_var_{band}_{ch}_{time}`
- `temp_rms_{band}_{ch}_{time}`
- `temp_skew_{band}_{ch}_{time}`
- `temp_kurt_{band}_{ch}_{time}`
- `temp_zerocross_{band}_{ch}_{time}`
- `temp_linelen_{band}_{ch}_{time}`

### Asymmetry Features (per band × roi_pair × time)
- `asym_{band}_{roi_pair}_{time}`
- `asym_{band}_contra_ipsi_{time}`
- `asym_{band}_lateralization_{time}`

### GFP Features (per band × time)
- `gfp_{time}_mean`
- `gfp_{time}_std`
- `gfp_{time}_max`
- `gfp_{time}_peak_count`
- `gfp_{band}_{time}_mean`
- `gfp_baseline_ratio_{time}`
- `gfp_slope`

### ROI Features (per domain × band × roi × time)
- `roi_power_{band}_{roi}_{time}_mean`
- `roi_erds_{band}_{roi}_{time}_percent`
- `roi_itpc_{band}_{roi}_{time}_mean`
- `roi_conn_within_{band}_{roi}_{time}`
- `roi_complexity_{band}_{roi}_{time}`

---

*Document generated: Feature Extraction Pipeline Review*
*Version: 2.0*
*Last updated: 2024*

---

## Appendix B: Pain-Specific Feature Validation

### Expected Feature Patterns for Thermal Pain

Based on the pain neuroscience literature, the following patterns should be observed in a well-functioning feature extraction pipeline:

#### Power Features
| Feature | Pain vs Non-Pain | Effect Size (Cohen's d) | Key Channels |
|---------|------------------|------------------------|--------------|
| Alpha power (8-13 Hz) | Decreased (ERD) | 0.4-0.8 | C3/C4, CP3/CP4 |
| Beta power (13-30 Hz) | Decreased (ERD) | 0.3-0.6 | C3/C4, Cz |
| Gamma power (30-100 Hz) | Increased | 0.3-0.5 | C3/C4, Cz |
| Theta power (4-8 Hz) | Increased | 0.2-0.5 | Fz, FCz |
| Delta power (1-4 Hz) | Variable | 0.1-0.3 | Frontal |

#### Temporal Dynamics
| Feature | Expected Pattern | Interpretation |
|---------|------------------|----------------|
| Alpha ERD onset | 500-1500ms post-stimulus | Sensory processing initiation |
| Alpha ERD peak | 2000-4000ms | Maximum sensory engagement |
| Beta rebound onset | Post-stimulus offset | Motor inhibition |
| Gamma burst | 200-500ms, transient | Nociceptive encoding |
| Theta increase | Sustained during pain | Affective processing |

#### Connectivity
| Feature | Pain vs Non-Pain | Interpretation |
|---------|------------------|----------------|
| Alpha connectivity | Decreased | Sensorimotor network disinhibition |
| Theta connectivity | Increased (frontal) | Affective network engagement |
| Gamma connectivity | Increased (local) | Enhanced local processing |
| Fronto-parietal | Increased | Top-down attention to pain |

#### Lateralization
| Feature | Expected Pattern | Interpretation |
|---------|------------------|----------------|
| Contralateral alpha ERD | Stronger than ipsilateral | Somatotopic processing |
| Lateralization index | Positive (contra > ipsi) | Intact sensory processing |
| Frontal asymmetry | Variable | Approach/avoidance motivation |

### Validation Checks

```python
def validate_pain_patterns(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    config: Any,
    logger: Any,
) -> Dict[str, Any]:
    """
    Validate that extracted features show expected pain-related patterns.
    
    Returns:
        {
            "alpha_erd_present": bool,
            "alpha_erd_effect_size": float,
            "lateralization_present": bool,
            "theta_increase_present": bool,
            "gamma_increase_present": bool,
            "warnings": List[str],
        }
    """
```

---

## Appendix C: HRF-Aligned Feature Extraction

### Hemodynamic Response Function Considerations

For EEG-fMRI integration, features must be aligned with the hemodynamic response:

#### HRF Timing
- **Onset**: ~2s after neural activity
- **Peak**: ~5-6s after neural activity
- **Return to baseline**: ~15-20s

#### Implications for Feature Extraction

1. **Temporal Resolution**: 
   - EEG features should have ~1s resolution to capture HRF dynamics
   - Current 3-bin approach (early/mid/late) may be too coarse

2. **Convolution Approach**:
   - EEG features can be convolved with canonical HRF
   - Requires continuous (not binned) feature time courses

3. **Recommended Temporal Bins for HRF Modeling**:
```yaml
temporal_bins_hrf:
  - {start: 3.0, end: 4.0, label: "t1"}   # Pre-HRF peak
  - {start: 4.0, end: 5.0, label: "t2"}   # Rising phase
  - {start: 5.0, end: 6.0, label: "t3"}   # Near peak
  - {start: 6.0, end: 7.0, label: "t4"}   # Peak
  - {start: 7.0, end: 8.0, label: "t5"}   # Falling phase
  - {start: 8.0, end: 9.0, label: "t6"}   # Late falling
  - {start: 9.0, end: 10.5, label: "t7"}  # Return to baseline
```

4. **Feature-BOLD Correlation Windows**:
   - EEG feature at time T correlates with BOLD at time T+5s (approximately)
   - Sliding correlation analysis recommended

---

## Appendix D: Machine Learning Considerations

### Feature Selection for Pain Classification

#### High-Priority Features (Based on Literature)
1. **Alpha ERD magnitude** (sensorimotor channels)
2. **Gamma power** (central channels)
3. **Theta power** (frontal channels)
4. **Lateralization index** (alpha band)
5. **Frontal theta connectivity**
6. **Aperiodic slope**

#### Feature Redundancy Analysis

Many features are highly correlated. Recommended approach:
1. Compute feature correlation matrix
2. Identify clusters of correlated features (r > 0.8)
3. Select representative feature from each cluster
4. Or use dimensionality reduction (PCA, UMAP)

#### Feature Importance Validation

```python
def compute_feature_importance(
    features_df: pd.DataFrame,
    target: pd.Series,
    method: str = "mutual_information",  # or "random_forest", "permutation"
    logger: Any,
) -> pd.DataFrame:
    """
    Compute feature importance for pain prediction.
    
    Returns DataFrame with:
    - feature_name
    - importance_score
    - p_value (if applicable)
    - theoretical_category (power, connectivity, etc.)
    """
```

---

## Appendix E: Computational Efficiency

### Current Bottlenecks

1. **TFR Computation**: Most expensive operation
2. **Connectivity Matrices**: O(n_channels²) per band per epoch
3. **Microstate Clustering**: K-means on all GFP peaks

### Optimization Strategies

1. **Parallel Processing**:
   - Band filtering can be parallelized
   - Epoch-wise feature extraction can be parallelized

2. **Caching**:
   - Cache TFR results
   - Cache filtered band data

3. **Selective Extraction**:
   - Allow user to specify which feature domains to extract
   - Skip expensive computations if not needed

4. **Memory Management**:
   - Process epochs in batches for large datasets
   - Use memory-mapped arrays for connectivity matrices

---

## Appendix F: Configuration Schema

### Recommended Configuration Updates

```yaml
feature_engineering:
  # Temporal resolution
  temporal_bins:
    coarse:  # For quick analysis
      - {start: 3.0, end: 5.0, label: "early"}
      - {start: 5.0, end: 7.5, label: "mid"}
      - {start: 7.5, end: 10.5, label: "late"}
    fine:    # For HRF modeling
      - {start: 3.0, end: 4.0, label: "t1"}
      - {start: 4.0, end: 5.0, label: "t2"}
      - {start: 5.0, end: 6.0, label: "t3"}
      - {start: 6.0, end: 7.0, label: "t4"}
      - {start: 7.0, end: 8.0, label: "t5"}
      - {start: 8.0, end: 9.0, label: "t6"}
      - {start: 9.0, end: 10.5, label: "t7"}
  
  # Feature domains to extract
  domains:
    power: true
    erds: true
    connectivity: true
    phase: true
    aperiodic: true
    spectral: true
    microstates: true
    complexity: true
    temporal: true
    asymmetry: true
    gfp: true
  
  # Normalization options
  normalization:
    power: "logratio"      # logratio, zscore, percent, db
    erds: "percent"        # percent, zscore
    connectivity: "none"   # none, zscore
  
  # Output options
  output:
    save_by_domain: true
    save_by_band: true
    save_by_time: true
    save_by_roi: true
    save_connectivity_matrices: true
    generate_manifest: true
    generate_qc_report: true
  
  # ROI definitions for pain research
  rois:
    Sensorimotor_Contra:
      channels: ["C3", "C1", "CP3", "CP1"]
      description: "Contralateral sensorimotor cortex"
    Sensorimotor_Ipsi:
      channels: ["C4", "C2", "CP4", "CP2"]
      description: "Ipsilateral sensorimotor cortex"
    Frontal_Midline:
      channels: ["Fz", "FCz", "Cz"]
      description: "Midline frontal (ACC/mPFC proxy)"
    Parietal:
      channels: ["Pz", "P3", "P4", "CPz"]
      description: "Parietal cortex (S2/PPC)"
    Temporal_Left:
      channels: ["T7", "TP7"]
      description: "Left temporal (insula proxy)"
    Temporal_Right:
      channels: ["T8", "TP8"]
      description: "Right temporal (insula proxy)"
  
  # Asymmetry pairs
  asymmetry_pairs:
    - {left: "Sensorimotor_Ipsi", right: "Sensorimotor_Contra", name: "sensorimotor"}
    - {left: "Temporal_Left", right: "Temporal_Right", name: "temporal"}
    - {left: "F3", right: "F4", name: "frontal"}
    - {left: "P3", right: "P4", name: "parietal"}
  
  # Connectivity options
  connectivity:
    measures: ["wpli", "plv", "aec_orth"]
    compute_roi_to_roi: true
    compute_graph_metrics: true
    graph_threshold: 0.3
  
  # Phase options
  phase:
    compute_itpc_dynamics: true
    compute_pac: true
    pac_phase_bands: ["theta"]
    pac_amp_bands: ["gamma"]
  
  # Aperiodic options
  aperiodic:
    compute_time_resolved: true
    fit_range: [2.0, 40.0]
    model: "fixed"  # fixed or knee
  
  # Complexity options
  complexity:
    measures: ["pe", "hjorth", "lzc", "sample_entropy"]
    pe_order: 3
    pe_delay: 1
  
  # Quality control
  qc:
    min_valid_fraction: 0.5
    outlier_method: "iqr"
    outlier_threshold: 3.0
    flag_constant_features: true
    flag_high_nan_features: true
    nan_threshold: 0.2
```

---

## Appendix G: Testing Strategy

### Unit Tests for Feature Extraction

```python
# Test that features have expected shape
def test_feature_shape():
    """All feature DataFrames should have n_epochs rows."""
    
# Test that features are finite
def test_feature_finite():
    """Features should not contain inf values."""
    
# Test naming convention
def test_feature_naming():
    """All feature names should follow naming convention."""
    
# Test completeness
def test_feature_completeness():
    """All expected features should be present."""
```

### Integration Tests

```python
# Test full pipeline on synthetic data
def test_full_pipeline_synthetic():
    """Run full extraction on synthetic epochs."""
    
# Test pain pattern validation
def test_pain_patterns():
    """Verify expected pain patterns in real data."""
    
# Test output file generation
def test_output_files():
    """Verify all output files are created correctly."""
```

### Regression Tests

```python
# Test that feature values are stable across versions
def test_feature_stability():
    """Compare features to reference values."""
```

---

## Appendix H: Migration Guide

### Updating Existing Code

If you have code that uses the current feature extraction pipeline, here are the changes needed:

#### Column Name Changes
```python
# Old naming
"pow_alpha_Cz_early" -> "power_alpha_Cz_early_mean"
"erds_alpha_Cz" -> "erds_alpha_Cz_full_percent"
"wpli_theta_mean" -> "conn_wpli_theta_full_mean"

# Migration helper
def migrate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert old column names to new convention."""
    rename_map = {
        # Add mappings here
    }
    return df.rename(columns=rename_map)
```

#### New Feature Access
```python
# Old way
power_features = results["pow_df"]
conn_features = results["conn_df"]

# New way
all_features = results.get_combined_df()
power_features = results.get_feature_group_df("power")
conn_features = results.get_feature_group_df("connectivity")
```

#### Configuration Changes
```python
# Old config
config.get("feature_engineering.features.temporal_bins")

# New config (supports coarse and fine)
config.get("feature_engineering.temporal_bins.coarse")
config.get("feature_engineering.temporal_bins.fine")
```

---

## Summary

This document outlines a comprehensive improvement plan for the EEG feature extraction pipeline, with:

1. **Theoretical justification** for each feature based on pain neuroscience literature
2. **Complete feature matrix** covering all domains × bands × channels × time windows
3. **Standardized naming convention** for clean, organized outputs
4. **Output organization** with multiple views (by domain, band, time, ROI)
5. **Quality control** procedures for validation
6. **Implementation priority** based on impact and effort
7. **Configuration schema** for flexible feature extraction
8. **Testing strategy** for reliability

The improvements prioritize:
- **Temporal resolution** for HRF modeling
- **Pain-specific features** with theoretical backing
- **Clean, organized outputs** with full documentation
- **Completeness** across all feature domains

---

## Appendix I: Additional Pain-Relevant Features Not Yet Covered

### 1. Evoked Response Features (ERP-like from TFR)

While this pipeline focuses on induced (non-phase-locked) activity, some evoked features are pain-relevant:

```
evoked_n1_amplitude_{channel}          # First negative peak (nociceptive)
evoked_n1_latency_{channel}            # N1 timing
evoked_p2_amplitude_{channel}          # P2 component (pain perception)
evoked_p2_latency_{channel}            # P2 timing
evoked_n2p2_amplitude_{channel}        # N2-P2 complex (vertex potential)
```

**Theoretical basis**: The vertex potential (N2-P2) at Cz is the most reliable EEG marker of nociceptive input. It reflects activity in the cingulate and opercular cortices.

**Pain relevance**: N2-P2 amplitude correlates with stimulus intensity; latency reflects conduction velocity.

### 2. Habituation/Sensitization Features

Pain responses change over repeated stimulation:

```
power_{band}_{channel}_habituation_slope    # Slope across trials
power_{band}_{channel}_first_last_ratio     # First trial / Last trial
erds_{band}_{channel}_habituation_slope     # ERD habituation
conn_{measure}_{band}_habituation_slope     # Connectivity habituation
```

**Theoretical basis**: Habituation (decreasing response) is normal; sensitization (increasing response) may indicate central sensitization.

**Pain relevance**: Chronic pain patients often show reduced habituation or sensitization.

### 3. Anticipation Features (Pre-Stimulus)

If there's a cue before thermal stimulation:

```
power_{band}_{channel}_anticipation         # Power during cue-stimulus interval
erds_{band}_{channel}_anticipation          # ERD during anticipation
conn_{measure}_{band}_anticipation          # Connectivity during anticipation
```

**Theoretical basis**: Anticipation of pain activates similar networks as pain itself. Alpha ERD during anticipation predicts subsequent pain perception.

**Pain relevance**: Anticipatory anxiety amplifies pain; anticipatory features may predict pain ratings.

### 4. Recovery Features (Post-Stimulus)

After stimulus offset:

```
power_{band}_{channel}_recovery_slope       # Rate of return to baseline
power_{band}_{channel}_recovery_time        # Time to reach 50% of baseline
erds_{band}_{channel}_rebound_magnitude     # Beta rebound strength
erds_{band}_{channel}_rebound_latency       # Beta rebound timing
```

**Theoretical basis**: Beta rebound reflects active motor inhibition and return to resting state. Delayed recovery may indicate prolonged pain processing.

**Pain relevance**: Slower recovery in chronic pain; stronger beta rebound with pain relief.

### 5. Trial-to-Trial Variability Features

Variability itself is informative:

```
power_{band}_{channel}_trial_cv             # Coefficient of variation across trials
power_{band}_{channel}_trial_std            # Standard deviation across trials
erds_{band}_{channel}_trial_consistency     # 1 - CV
conn_{measure}_{band}_trial_stability       # Connectivity stability
```

**Theoretical basis**: High variability may indicate unstable neural processing or fluctuating attention.

**Pain relevance**: Increased variability in chronic pain; decreased variability with focused attention.

### 6. Pain Rating Prediction Features

Pre-computed correlations with pain ratings (for interpretability):

```
power_{band}_{channel}_{time}_pain_r        # Correlation with pain rating
power_{band}_{channel}_{time}_pain_p        # P-value
erds_{band}_{channel}_{time}_pain_r         # ERD-pain correlation
```

**Note**: These are computed post-hoc and should be used for interpretation, not prediction (to avoid circularity).

---

## Appendix J: Stimulus-Specific Considerations

### Thermal Pain Paradigm Specifics

Your paradigm uses thermal stimulation with a plateau period. Key considerations:

#### 1. Ramp Period (0-3s)
- Temperature is rising
- Mixed sensory (warmth) and nociceptive (pain) processing
- **Recommendation**: Extract features from ramp period separately if scientifically relevant

```
power_{band}_{channel}_ramp_mean
power_{band}_{channel}_ramp_slope           # Power change during ramp
```

#### 2. Plateau Period (3-10.5s)
- Constant temperature
- Sustained pain processing
- **Current focus**: This is where most features are extracted
- **Key insight**: Early plateau may differ from late plateau (adaptation)

#### 3. Offset Period (>10.5s)
- Temperature returning to baseline
- Pain relief processing
- Beta rebound expected
- **Recommendation**: Extract offset features if epoch extends beyond plateau

```
power_{band}_{channel}_offset_mean
erds_{band}_{channel}_offset_rebound
```

#### 4. Pain vs Non-Pain Trials
- Non-pain trials: Warm but not painful temperature
- Key contrast for identifying pain-specific features
- **Recommendation**: Ensure condition labels are properly aligned with features

### Temperature-Dependent Features

If temperature varies across trials:

```
power_{band}_{channel}_{time}_temp_slope    # Slope with temperature
power_{band}_{channel}_{time}_temp_r        # Correlation with temperature
```

**Theoretical basis**: EEG responses scale with stimulus intensity. Gamma power shows strongest intensity coding.

---

## Appendix K: Inter-Individual Difference Features

### Subject-Level Normalization

For group analysis, consider:

```
power_{band}_{channel}_{time}_subject_zscore  # Z-scored within subject
power_{band}_{channel}_{time}_rank            # Rank within subject
```

### Trait-Like Features

Some features may reflect stable individual differences:

```
iaf_global_mean                              # Individual alpha frequency
aperiodic_slope_baseline_global              # Baseline 1/f slope
power_alpha_baseline_global                  # Baseline alpha power
```

**Theoretical basis**: IAF and baseline aperiodic slope are relatively stable traits that may moderate pain responses.

---

## Appendix L: Feature Interaction Terms

### Cross-Domain Interactions

Some feature combinations may be more predictive than individual features:

```
# Power × Connectivity
power_alpha_erds × conn_wpli_alpha           # ERD with connectivity change

# Band × Band
power_theta × power_gamma                    # Theta-gamma interaction

# Time × Time  
power_alpha_early × power_alpha_late         # Temporal interaction

# Channel × Channel
power_alpha_contra × power_alpha_ipsi        # Lateralization interaction
```

**Recommendation**: Compute these interaction terms if using linear models; tree-based models will capture them automatically.

---

## Appendix M: Negative Controls

### Features That Should NOT Correlate with Pain

To validate specificity, check that these features don't show pain effects:

1. **Occipital alpha** (unless visual attention is involved)
2. **Line noise (50/60 Hz)** - should be removed
3. **Eye movement artifacts** - should be removed
4. **Muscle artifacts (EMG)** - should be removed

### Sanity Checks

```python
def validate_negative_controls(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: Any,
) -> Dict[str, Any]:
    """
    Verify that control features don't show pain effects.
    
    Returns warnings if:
    - Occipital alpha correlates with pain (unexpected)
    - High-frequency noise correlates with pain (artifact)
    """
```

---

## Appendix N: Feature Documentation Template

For each feature, document:

```yaml
feature_name: "power_alpha_Cz_early_mean"
domain: "power"
band: "alpha"
band_range_hz: [8.0, 13.0]
channel: "Cz"
roi: null
time_window: "early"
time_range_s: [3.0, 5.0]
statistic: "mean"
unit: "log(µV²/µV²)"
normalization: "log-ratio to baseline [-5, -0.01]s"
baseline_window_s: [-5.0, -0.01]

theoretical_relevance: |
  Alpha oscillations in central regions reflect sensorimotor cortex 
  activity. Alpha ERD (power decrease) indicates cortical activation
  during sensory processing. During pain, contralateral sensorimotor
  cortex shows alpha ERD reflecting nociceptive input processing.

expected_pain_effect: "decrease (ERD)"
expected_effect_size: "Cohen's d = 0.4-0.8"
key_references:
  - "Ploner et al., 2006"
  - "May et al., 2012"
  - "Tu et al., 2016"

quality_notes: |
  - Requires clean data (artifacts inflate alpha)
  - Sensitive to eye closure (increases alpha)
  - May be affected by drowsiness

related_features:
  - "erds_alpha_Cz_early_percent"
  - "power_alpha_Sensorimotor_Contra_early_mean"
```

---

## Appendix O: Complete Implementation Checklist

### Phase 1: Foundation (Week 1-2)
- [ ] Implement standardized naming convention
- [ ] Create feature manifest generator
- [ ] Add fine temporal bins (t1-t7)
- [ ] Restructure output directory
- [ ] Add QC validation framework

### Phase 2: Power/ERD Features (Week 2-3)
- [ ] Add time-resolved power (7 bins)
- [ ] Add temporal dynamics (slope, diff, peak_time)
- [ ] Add ERD/ERS separation
- [ ] Add baseline-relative variants (zscore, percent, logratio)
- [ ] Add global summaries for all channel features

### Phase 3: Connectivity Features (Week 3-4)
- [ ] Add time-resolved connectivity (3 bins)
- [ ] Add ROI-to-ROI connectivity
- [ ] Add temporal dynamics (slope, diff)
- [ ] Complete graph metrics per time window

### Phase 4: Phase Features (Week 4-5)
- [ ] Add ITPC temporal dynamics (peak_time, onset, duration)
- [ ] Add phase consistency features
- [ ] Add PAC temporal features
- [ ] Add time-resolved PAC

### Phase 5: Other Domains (Week 5-6)
- [ ] Add time-resolved aperiodic
- [ ] Add spectral shape features
- [ ] Add microstate temporal features
- [ ] Add complexity per time window
- [ ] Complete asymmetry features

### Phase 6: Integration (Week 6-7)
- [ ] Generate complete feature manifest
- [ ] Implement output organization (by domain, band, time, ROI)
- [ ] Add QC report generation
- [ ] Add pain pattern validation
- [ ] Update documentation

### Phase 7: Testing (Week 7-8)
- [ ] Unit tests for all extractors
- [ ] Integration tests
- [ ] Regression tests
- [ ] Performance benchmarks
- [ ] Validation on real data

---

## Appendix P: Estimated Feature Counts

### Current Pipeline
| Domain | Approximate Features |
|--------|---------------------|
| Power | ~300 (5 bands × 64 ch × ~1 time) |
| Connectivity | ~500 (pairwise + graph) |
| Microstates | ~50 |
| Aperiodic | ~130 |
| ITPC | ~100 |
| PAC | ~50 |
| Precomputed | ~2000 |
| **Total** | **~3000** |

### After Improvements
| Domain | Approximate Features |
|--------|---------------------|
| Power | ~2240 (5 bands × 64 ch × 7 times) |
| ERD/ERS | ~2240 (5 bands × 64 ch × 7 times) |
| Connectivity | ~1500 (measures × bands × times) |
| Phase | ~1000 (ITPC + PAC + dynamics) |
| Aperiodic | ~500 (time-resolved) |
| Spectral | ~500 (shape features) |
| Microstates | ~200 (time-resolved) |
| Complexity | ~1000 (time-resolved) |
| Temporal | ~1000 (waveform features) |
| Asymmetry | ~200 |
| GFP | ~200 |
| ROI | ~500 |
| **Total** | **~11,000** |

### Dimensionality Reduction Recommendations
With ~11,000 features, dimensionality reduction is essential:
1. **ROI aggregation**: Reduces channel dimension from 64 to ~6-10 ROIs
2. **Feature selection**: Keep top 100-500 by importance
3. **PCA/UMAP**: Reduce to 50-100 components
4. **Domain-specific models**: Train separate models per domain, then ensemble

---

## Appendix Q: ML-Ready Output Specifications

### Design Principle
The feature extraction pipeline should produce **clean, standardized outputs** that can be directly consumed by any downstream ML pipeline without additional preprocessing.

### Output Requirements for ML Compatibility

#### 1. Data Format Standards
```
- File format: TSV (tab-separated values) for portability
- Encoding: UTF-8
- Missing values: Represented as empty string or "NaN" (consistent)
- Numeric precision: 8 decimal places (sufficient for float32)
- No trailing whitespace
- Unix line endings (\n)
```

#### 2. Column Naming Requirements
```
- No spaces (use underscores)
- No special characters except underscores
- Lowercase only
- Consistent prefix for feature domain
- Parseable structure: {domain}_{measure}_{band}_{location}_{time}_{stat}
```

#### 3. Row Alignment
```
- One row per trial/epoch
- Rows aligned with events DataFrame
- Trial index column for joining
- Condition column for stratification
- Subject ID column for grouping
```

#### 4. Metadata Columns (Non-Feature)
These columns should be present but clearly marked as non-features:
```
meta_trial_idx          # Trial index within session
meta_subject_id         # Subject identifier
meta_session_id         # Session identifier (if applicable)
meta_condition          # pain / nonpain
meta_temperature        # Stimulus temperature (if available)
meta_rating             # Pain rating (if available)
meta_timestamp          # Epoch timestamp
meta_is_valid           # QC flag (True/False)
```

#### 5. Feature Column Requirements
```
- All feature columns are numeric (float64)
- No categorical features (encode before saving if needed)
- No infinite values (replace with NaN)
- Consistent units within feature type
- Documented in manifest
```

### Output File Specifications

#### Primary Output: `features_all.tsv`
```
Columns:
- meta_* columns (non-features, for joining/filtering)
- All feature columns in consistent order

Rows:
- One per trial
- Sorted by subject, then trial index
- No duplicate rows
```

#### Feature Manifest: `features_manifest.json`
```json
{
  "version": "2.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "pipeline_version": "1.0.0",
  "n_features": 11000,
  "n_trials": 500,
  "n_subjects": 10,
  
  "meta_columns": [
    "meta_trial_idx",
    "meta_subject_id", 
    "meta_condition",
    ...
  ],
  
  "feature_columns": [
    "power_alpha_Cz_early_mean",
    "power_alpha_Cz_mid_mean",
    ...
  ],
  
  "feature_groups": {
    "power": ["power_alpha_Cz_early_mean", ...],
    "erds": ["erds_alpha_Cz_early_percent", ...],
    ...
  },
  
  "feature_metadata": {
    "power_alpha_Cz_early_mean": {
      "domain": "power",
      "band": "alpha",
      "channel": "Cz",
      "time_window": "early",
      "statistic": "mean",
      "unit": "log(µV²/µV²)",
      "description": "..."
    },
    ...
  }
}
```

#### QC Report: `qc/feature_quality.json`
```json
{
  "total_features": 11000,
  "total_trials": 500,
  
  "completeness": {
    "features_with_no_nan": 10500,
    "features_with_some_nan": 400,
    "features_with_all_nan": 100,
    "trials_with_no_nan": 450,
    "trials_with_some_nan": 50
  },
  
  "nan_by_feature": {
    "power_alpha_Cz_early_mean": 0.02,
    ...
  },
  
  "constant_features": [
    "feature_name_1",
    ...
  ],
  
  "high_nan_features": [
    {"name": "feature_x", "nan_fraction": 0.45},
    ...
  ],
  
  "recommended_exclusions": {
    "features": ["list of features to exclude"],
    "trials": ["list of trial indices to exclude"]
  }
}
```

---

## Appendix R: Feature Grouping for Organized Access

### Hierarchical Feature Organization

Features should be accessible by multiple grouping schemes:

#### By Domain
```python
feature_groups_by_domain = {
    "power": [...],      # All power features
    "erds": [...],       # All ERD/ERS features
    "connectivity": [...],
    "phase": [...],
    "aperiodic": [...],
    "spectral": [...],
    "microstates": [...],
    "complexity": [...],
    "temporal": [...],
    "asymmetry": [...],
    "gfp": [...],
    "roi": [...],
}
```

#### By Frequency Band
```python
feature_groups_by_band = {
    "delta": [...],      # All delta-band features
    "theta": [...],
    "alpha": [...],
    "beta": [...],
    "gamma": [...],
    "broadband": [...],  # Band-agnostic features
}
```

#### By Time Window
```python
feature_groups_by_time = {
    "baseline": [...],   # Baseline features
    "early": [...],      # Early plateau (3-5s)
    "mid": [...],        # Mid plateau (5-7.5s)
    "late": [...],       # Late plateau (7.5-10.5s)
    "t1": [...],         # Fine bin 1 (3-4s)
    "t2": [...],         # Fine bin 2 (4-5s)
    ...
    "full": [...],       # Full plateau average
    "dynamics": [...],   # Temporal dynamics (slope, diff)
}
```

#### By Spatial Scale
```python
feature_groups_by_spatial = {
    "channel": [...],    # Per-channel features
    "roi": [...],        # ROI-aggregated features
    "global": [...],     # Global/whole-brain features
    "pair": [...],       # Channel-pair features (connectivity)
    "roi_pair": [...],   # ROI-pair features
}
```

#### By Feature Type
```python
feature_groups_by_type = {
    "magnitude": [...],  # Power, amplitude, etc.
    "latency": [...],    # Timing features
    "duration": [...],   # Duration features
    "slope": [...],      # Temporal trends
    "ratio": [...],      # Ratio features
    "index": [...],      # Normalized indices (lateralization, etc.)
    "count": [...],      # Count features (peaks, transitions)
    "entropy": [...],    # Entropy/complexity features
}
```

### Feature Subset Presets

For common use cases, provide preset feature subsets:

```python
feature_presets = {
    # Minimal set for quick analysis
    "minimal": {
        "description": "Core pain-relevant features only",
        "n_features": ~100,
        "includes": [
            "erds_alpha_Sensorimotor_*",
            "erds_beta_Sensorimotor_*",
            "power_gamma_Cz_*",
            "power_theta_Fz_*",
            "asym_alpha_sensorimotor_*",
        ]
    },
    
    # ROI-only (reduced dimensionality)
    "roi_only": {
        "description": "ROI-aggregated features, no channel-level",
        "n_features": ~500,
        "includes": ["roi_*", "*_global_*"]
    },
    
    # Time-resolved (for HRF modeling)
    "time_resolved": {
        "description": "Features with fine temporal resolution",
        "n_features": ~3000,
        "includes": ["*_t1_*", "*_t2_*", ..., "*_t7_*"]
    },
    
    # Pain-optimized (based on literature)
    "pain_optimized": {
        "description": "Features with strongest pain relevance",
        "n_features": ~500,
        "includes": [
            "erds_alpha_*",
            "erds_beta_*", 
            "power_gamma_*",
            "power_theta_Frontal_*",
            "asym_*_sensorimotor_*",
            "aper_slope_*",
            "conn_wpli_theta_Frontal_*",
        ]
    },
    
    # Comprehensive (all features)
    "comprehensive": {
        "description": "All extracted features",
        "n_features": ~11000,
        "includes": ["*"]
    }
}
```

---

## Appendix S: Data Integrity Checks

### Pre-Save Validation

Before saving feature files, perform these checks:

```python
def validate_features_before_save(
    df: pd.DataFrame,
    manifest: Dict,
    logger: Any,
) -> Tuple[bool, List[str]]:
    """
    Validate feature DataFrame before saving.
    
    Checks:
    1. No duplicate column names
    2. No duplicate rows
    3. All feature columns are numeric
    4. No infinite values
    5. Column names follow naming convention
    6. Row count matches expected trial count
    7. Meta columns are present
    8. Feature columns match manifest
    
    Returns:
        (is_valid, list_of_issues)
    """
```

### Post-Load Validation

When loading features for downstream use:

```python
def validate_features_after_load(
    df: pd.DataFrame,
    manifest: Dict,
    logger: Any,
) -> Tuple[bool, List[str]]:
    """
    Validate loaded feature DataFrame.
    
    Checks:
    1. All expected columns present
    2. Data types are correct
    3. No unexpected NaN patterns
    4. Values are within expected ranges
    5. Manifest matches data
    
    Returns:
        (is_valid, list_of_issues)
    """
```

### Checksums and Versioning

```python
def compute_feature_checksum(df: pd.DataFrame) -> str:
    """Compute MD5 checksum of feature data for integrity verification."""
    
def save_with_versioning(
    df: pd.DataFrame,
    output_path: Path,
    manifest: Dict,
    logger: Any,
) -> None:
    """
    Save features with versioning metadata.
    
    Creates:
    - features_all.tsv (data)
    - features_manifest.json (schema)
    - features_checksum.txt (integrity)
    - features_version.json (versioning)
    """
```

---

## Appendix T: Handling Edge Cases

### Missing Data Strategies

#### Trial-Level Missing Data
```
Scenario: Entire trial has artifacts, no valid features
Strategy: 
- Set all features to NaN for that trial
- Set meta_is_valid = False
- Document in QC report
- Do NOT drop the row (maintain alignment with events)
```

#### Feature-Level Missing Data
```
Scenario: Specific feature cannot be computed (e.g., not enough data for ITPC)
Strategy:
- Set that feature to NaN
- Document reason in QC report
- Other features for that trial remain valid
```

#### Channel-Level Missing Data
```
Scenario: Bad channel excluded from analysis
Strategy:
- Set all features for that channel to NaN
- ROI features computed from remaining channels
- Document which channels were excluded
```

### Constant Features

```
Scenario: Feature has same value for all trials
Strategy:
- Keep the feature (don't drop)
- Flag in QC report as "constant"
- Downstream ML pipeline can decide to exclude
```

### Outlier Features

```
Scenario: Feature has extreme outliers
Strategy:
- Do NOT modify values (preserve raw data)
- Flag outliers in QC report
- Provide outlier indices for downstream filtering
- Downstream ML pipeline handles outlier treatment
```

### Zero-Variance Epochs

```
Scenario: Epoch has zero variance (flat line)
Strategy:
- Set power features to NaN (log of zero undefined)
- Set connectivity features to NaN
- Set meta_is_valid = False
- Document in QC report
```

---

## Appendix U: Reproducibility Requirements

### Deterministic Outputs

Feature extraction should be fully reproducible:

```python
# Required for reproducibility
np.random.seed(config.get("random.seed", 42))

# Document all parameters
extraction_params = {
    "mne_version": mne.__version__,
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "pipeline_version": __version__,
    "config_hash": compute_config_hash(config),
    "random_seed": config.get("random.seed", 42),
    "extraction_timestamp": datetime.now().isoformat(),
}
```

### Version Tracking

```json
// features_version.json
{
  "data_version": "1.0.0",
  "schema_version": "2.0.0",
  "pipeline_version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "created_by": "extract_features.py",
  "input_files": {
    "epochs": "sub-0001_task-thermalactive_epo.fif",
    "events": "sub-0001_task-thermalactive_events.tsv"
  },
  "config_hash": "abc123...",
  "dependencies": {
    "mne": "1.6.0",
    "numpy": "1.24.0",
    "scipy": "1.11.0"
  }
}
```

---

## Appendix V: Feature Naming Deep Dive

### Complete Naming Grammar

```
feature_name := domain "_" measure ["_" band] ["_" location] ["_" time] ["_" statistic]

domain := "power" | "erds" | "conn" | "phase" | "pac" | "aper" | "spec" | 
          "ms" | "comp" | "temp" | "asym" | "gfp" | "roi"

measure := <domain-specific, see below>

band := "delta" | "theta" | "alpha" | "beta" | "gamma" | "broadband"

location := channel_name | roi_name | "global" | pair_name

time := "baseline" | "early" | "mid" | "late" | "t1" | ... | "t7" | "full" | 
        "ramp" | "offset"

statistic := "mean" | "std" | "max" | "min" | "cv" | "zscore" | "percent" | 
             "logratio" | "db" | "slope" | "diff" | "onset" | "peak" | 
             "duration" | "auc" | "count"
```

### Domain-Specific Measures

#### Power Domain
```
power_mean          # Mean power
power_std           # Power standard deviation
power_max           # Maximum power
power_min           # Minimum power
power_auc           # Area under curve
```

#### ERD/ERS Domain
```
erds_percent        # Percent change from baseline
erds_zscore         # Z-scored change
erds_erd            # ERD magnitude (negative values only)
erds_ers            # ERS magnitude (positive values only)
erds_onset          # Onset latency
erds_peak           # Peak latency
erds_duration       # Duration of significant change
erds_slope          # Temporal slope
```

#### Connectivity Domain
```
conn_wpli           # Weighted Phase Lag Index
conn_plv            # Phase Locking Value
conn_aec            # Amplitude Envelope Correlation
conn_imcoh          # Imaginary Coherence
conn_pli            # Phase Lag Index
conn_psi            # Phase Slope Index (directed)
conn_efficiency     # Graph efficiency
conn_clustering     # Graph clustering coefficient
conn_modularity     # Graph modularity
```

#### Phase Domain
```
phase_itpc          # Inter-Trial Phase Coherence
phase_consistency   # Within-trial phase consistency
phase_concentration # von Mises concentration (kappa)
phase_angle         # Mean phase angle
```

#### PAC Domain
```
pac_mi              # Modulation Index
pac_zscore          # Z-scored PAC (vs surrogates)
```

#### Aperiodic Domain
```
aper_slope          # 1/f slope (exponent)
aper_offset         # 1/f offset (intercept)
aper_knee           # Knee frequency (if knee model)
aper_r2             # Fit quality
aper_change         # Change from baseline
```

#### Spectral Domain
```
spec_peak           # Peak frequency
spec_power          # Peak power
spec_entropy        # Spectral entropy
spec_centroid       # Spectral centroid
spec_bandwidth      # Spectral bandwidth
spec_skewness       # Spectral skewness
spec_kurtosis       # Spectral kurtosis
spec_edge50         # 50% spectral edge
spec_edge95         # 95% spectral edge
spec_iaf            # Individual alpha frequency
```

#### Microstate Domain
```
ms_coverage         # Time coverage
ms_duration         # Mean duration
ms_occurrence       # Occurrence rate
ms_gev              # Global explained variance
ms_trans            # Transition probability
ms_entropy          # Sequence entropy
```

#### Complexity Domain
```
comp_pe             # Permutation entropy
comp_se             # Sample entropy
comp_ae             # Approximate entropy
comp_lzc            # Lempel-Ziv complexity
comp_hurst          # Hurst exponent
comp_dfa            # Detrended fluctuation analysis
comp_hjorth_act     # Hjorth activity
comp_hjorth_mob     # Hjorth mobility
comp_hjorth_comp    # Hjorth complexity
```

#### Temporal Domain
```
temp_var            # Variance
temp_std            # Standard deviation
temp_rms            # Root mean square
temp_ptp            # Peak-to-peak amplitude
temp_mad            # Mean absolute deviation
temp_skew           # Skewness
temp_kurt           # Kurtosis
temp_zerocross      # Zero-crossing rate
temp_linelen        # Line length
temp_nle            # Nonlinear energy
```

#### Asymmetry Domain
```
asym_index          # Lateralization index: (R-L)/(R+L)
asym_diff           # Simple difference: R-L
asym_ratio          # Ratio: R/L
asym_contra_ipsi    # Contralateral/Ipsilateral
```

#### GFP Domain
```
gfp_mean            # Mean GFP
gfp_std             # GFP standard deviation
gfp_max             # Maximum GFP
gfp_peak_count      # Number of GFP peaks
gfp_peak_amp        # Mean peak amplitude
gfp_baseline_ratio  # Ratio to baseline
```

---

## Appendix W: Channel and ROI Naming

### Standard Channel Names
Use 10-20 system names exactly as provided by MNE:
```
Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8,
CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10, ...
```

### ROI Names (Standardized)
```
Sensorimotor_Contra     # Contralateral to stimulus
Sensorimotor_Ipsi       # Ipsilateral to stimulus
Sensorimotor_L          # Left sensorimotor (if no lateralization info)
Sensorimotor_R          # Right sensorimotor
Frontal_Midline         # Fz, FCz, Cz
Frontal_L               # Left frontal
Frontal_R               # Right frontal
Parietal_Midline        # Pz, CPz, POz
Parietal_L              # Left parietal
Parietal_R              # Right parietal
Temporal_L              # Left temporal
Temporal_R              # Right temporal
Occipital               # Occipital (control)
Central                 # Central strip
```

### Pair Names (for Connectivity)
```
# Channel pairs
Fz_Cz                   # Fz to Cz
C3_C4                   # Interhemispheric central

# ROI pairs
Frontal_to_Parietal     # Frontal-parietal connectivity
Sensorimotor_L_to_R     # Interhemispheric sensorimotor
Frontal_to_Sensorimotor # Top-down modulation
```

---

## Appendix X: Time Window Definitions

### Standard Time Windows

```yaml
time_windows:
  # Pre-stimulus
  baseline:
    start: -5.0
    end: -0.01
    description: "Pre-stimulus baseline for normalization"
  
  # Stimulus onset
  onset:
    start: 0.0
    end: 0.5
    description: "Immediate stimulus response"
  
  # Ramp period (temperature rising)
  ramp:
    start: 0.0
    end: 3.0
    description: "Temperature ramp period"
  
  # Plateau period (constant temperature)
  plateau:
    start: 3.0
    end: 10.5
    description: "Full plateau period"
  
  # Coarse plateau bins
  early:
    start: 3.0
    end: 5.0
    description: "Early plateau, initial pain response"
  mid:
    start: 5.0
    end: 7.5
    description: "Middle plateau, sustained processing"
  late:
    start: 7.5
    end: 10.5
    description: "Late plateau, adaptation/habituation"
  
  # Fine plateau bins (for HRF modeling)
  t1:
    start: 3.0
    end: 4.0
    description: "Fine bin 1"
  t2:
    start: 4.0
    end: 5.0
    description: "Fine bin 2"
  t3:
    start: 5.0
    end: 6.0
    description: "Fine bin 3"
  t4:
    start: 6.0
    end: 7.0
    description: "Fine bin 4"
  t5:
    start: 7.0
    end: 8.0
    description: "Fine bin 5"
  t6:
    start: 8.0
    end: 9.0
    description: "Fine bin 6"
  t7:
    start: 9.0
    end: 10.5
    description: "Fine bin 7"
  
  # Post-stimulus (if epoch extends)
  offset:
    start: 10.5
    end: 12.0
    description: "Post-plateau offset period"
  
  # Full active period
  full:
    start: 3.0
    end: 10.5
    description: "Full plateau average"
```

### Temporal Dynamics Features

These features capture change over time (not tied to specific window):

```
*_slope             # Linear trend across plateau
*_early_late_diff   # Late - Early difference
*_peak_time         # Time of maximum value
*_trough_time       # Time of minimum value
*_onset_latency     # Time of first significant change
*_duration          # Duration of significant change
```

---

## Appendix Y: Unit Specifications

### Standard Units by Feature Type

| Feature Type | Unit | Notes |
|--------------|------|-------|
| Raw power | µV² | Absolute power |
| Log power | log₁₀(µV²) | Log-transformed |
| Normalized power | log₁₀(µV²/µV²) | Log-ratio to baseline |
| ERD/ERS | % | Percent change from baseline |
| Connectivity (wPLI, PLV) | [0, 1] | Unitless, bounded |
| Connectivity (AEC) | [-1, 1] | Correlation coefficient |
| Graph metrics | varies | See specific metric |
| ITPC | [0, 1] | Unitless, bounded |
| PAC (MI) | bits | Modulation index |
| Aperiodic slope | [unitless] | Exponent of 1/f |
| Aperiodic offset | log₁₀(µV²) | Intercept |
| Frequency | Hz | Peak frequency, IAF |
| Latency | s | Time in seconds |
| Duration | s | Duration in seconds |
| Entropy | bits or [0, 1] | Normalized or raw |
| Amplitude | µV | Voltage |
| Variance | µV² | Squared voltage |
| Count | [integer] | Number of events |
| Rate | Hz or /s | Events per second |
| Index | [-1, 1] or [0, 1] | Normalized index |

---

## Appendix Z: Final Checklist Before Implementation

### Architecture Decisions
- [ ] Confirm naming convention with team
- [ ] Decide on coarse vs fine temporal bins (or both)
- [ ] Finalize ROI definitions
- [ ] Confirm which feature domains to include
- [ ] Decide on output file structure

### Technical Decisions
- [ ] Choose NaN representation (empty string vs "NaN")
- [ ] Decide on numeric precision (8 decimal places?)
- [ ] Confirm file format (TSV vs CSV vs Parquet)
- [ ] Decide on compression (gzip for large files?)

### Validation Decisions
- [ ] Define acceptable NaN thresholds
- [ ] Define outlier detection method
- [ ] Define constant feature handling
- [ ] Define QC report format

### Documentation Decisions
- [ ] Confirm manifest schema
- [ ] Decide on feature description detail level
- [ ] Confirm versioning strategy

---

## Document Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial draft |
| 2.0 | 2024-01 | Added theoretical framework, expanded appendices |
| 2.1 | 2024-01 | Added ML-ready output specs, edge cases, reproducibility |
| 2.2 | 2024-01 | Added naming deep dive, unit specifications |

---

*End of Document*

---

## Appendix AA: Cross-Feature Consistency Requirements

### Alignment Guarantees

All feature outputs must maintain strict alignment:

```
Guarantee 1: Row Alignment
- Row i in features_all.tsv corresponds to epoch i
- Row i in events.tsv corresponds to epoch i
- All feature domain files have same row order

Guarantee 2: Column Consistency
- Same feature computed in different contexts has same name
- Feature names are unique across all domains
- No column name collisions

Guarantee 3: Value Consistency
- Same underlying computation produces same value
- roi_power_alpha_Sensorimotor_Contra_early_mean uses same data as 
  power_alpha_C3_early_mean (just aggregated)
- No rounding differences between files
```

### Cross-Domain Feature Relationships

Document expected relationships between features:

```python
# These should be mathematically related
assert roi_power == mean(channel_powers_in_roi)
assert erds_percent == (power - baseline_power) / baseline_power * 100
assert gfp_band == std(power_across_channels)
assert asym_index == (right - left) / (right + left)

# These should be correlated (sanity check)
# power_alpha and erds_alpha should be negatively correlated
# conn_wpli and conn_plv should be positively correlated
# gfp and power_global should be positively correlated
```

---

## Appendix AB: Feature Extraction Order

### Recommended Extraction Sequence

For efficiency and dependency management:

```
Phase 1: Precomputation (compute once, reuse)
├── Band filtering (all bands)
├── Hilbert transform (analytic signal)
├── Power computation
├── PSD computation
├── GFP computation
└── Time window masks

Phase 2: Primary Features (no dependencies between domains)
├── Power features
├── ERD/ERS features
├── Spectral features
├── Aperiodic features
├── Temporal features
└── Complexity features

Phase 3: Derived Features (depend on Phase 2)
├── ROI aggregation (from power, ERD/ERS)
├── Asymmetry (from power, ERD/ERS)
├── Global summaries (from all channel features)
└── Temporal dynamics (from time-binned features)

Phase 4: Connectivity Features (expensive, separate)
├── Pairwise connectivity
├── ROI-to-ROI connectivity
├── Graph metrics
└── Connectivity dynamics

Phase 5: Phase Features (require complex TFR)
├── ITPC
├── ITPC dynamics
├── PAC
└── PAC dynamics

Phase 6: Microstates (separate pipeline)
├── Template extraction/matching
├── Microstate metrics
└── Microstate dynamics

Phase 7: Aggregation
├── Combine all features
├── Generate manifest
├── Run QC checks
└── Save outputs
```

---

## Appendix AC: Memory and Performance Considerations

### Memory Footprint Estimates

```
Per epoch, per channel:
- Raw data: 64 ch × 15s × 500Hz × 8 bytes = 3.84 MB
- TFR (complex): 64 ch × 100 freqs × 7500 times × 16 bytes = 768 MB
- Band-filtered (5 bands): 64 ch × 7500 times × 5 bands × 8 bytes = 19.2 MB
- Connectivity matrix: 64 × 64 × 5 bands × 8 bytes = 0.16 MB

For 50 epochs:
- TFR: 38.4 GB (too large to hold in memory)
- Band-filtered: 960 MB (manageable)
- Features: ~11000 features × 50 epochs × 8 bytes = 4.4 MB (small)
```

### Optimization Strategies

```
1. Process TFR in chunks (don't hold all epochs)
2. Compute connectivity per band (not all at once)
3. Use float32 for intermediate computations
4. Clear intermediate arrays after use
5. Use memory-mapped files for large arrays
```

---

## Appendix AD: Feature Sanity Checks

### Expected Value Ranges

```python
feature_ranges = {
    # Power features (log-ratio normalized)
    "power_*_*_*_mean": (-5.0, 5.0),  # Log-ratio, typically -2 to 2
    
    # ERD/ERS (percent)
    "erds_*_*_*_percent": (-100.0, 500.0),  # Can exceed 100% for ERS
    
    # Connectivity
    "conn_wpli_*_*_mean": (0.0, 1.0),
    "conn_plv_*_*_mean": (0.0, 1.0),
    "conn_aec_*_*_mean": (-1.0, 1.0),
    
    # Phase
    "phase_itpc_*_*_mean": (0.0, 1.0),
    
    # Aperiodic
    "aper_slope_*": (-4.0, 0.0),  # Typically -2 to -1
    
    # Spectral
    "spec_peak_*_*": (1.0, 100.0),  # Hz
    "spec_entropy_*_*": (0.0, 1.0),  # Normalized
    
    # Microstates
    "ms_coverage_*": (0.0, 1.0),
    "ms_duration_*": (0.01, 0.5),  # Seconds
    "ms_occurrence_*": (0.0, 20.0),  # Per second
    
    # Complexity
    "comp_pe_*_*": (0.0, 1.0),  # Normalized
    "comp_lzc_*_*": (0.0, 1.0),  # Normalized
    
    # Asymmetry
    "asym_*_*": (-1.0, 1.0),
    
    # Latency
    "*_onset_*": (0.0, 10.0),  # Seconds
    "*_peak_time_*": (0.0, 10.0),  # Seconds
}
```

### Automated Sanity Checks

```python
def run_sanity_checks(
    df: pd.DataFrame,
    feature_ranges: Dict[str, Tuple[float, float]],
    logger: Any,
) -> Dict[str, List[str]]:
    """
    Check that feature values are within expected ranges.
    
    Returns:
        {
            "out_of_range": [list of features with values outside range],
            "unexpected_nan": [list of features with unexpected NaN],
            "unexpected_constant": [list of constant features],
            "warnings": [list of warning messages]
        }
    """
```

---

## Appendix AE: Feature Correlation Structure

### Expected Correlation Patterns

For QC, verify these expected correlations:

```
High positive correlation (r > 0.7):
- power_alpha_Cz_early and power_alpha_Cz_mid (temporal consistency)
- power_alpha_C3 and power_alpha_C4 (spatial consistency)
- conn_wpli_alpha and conn_plv_alpha (measure consistency)
- erds_alpha_Cz and power_alpha_Cz (definitional)

Moderate negative correlation (r ~ -0.3 to -0.5):
- erds_alpha (ERD) and power_gamma (different processes)
- aper_slope and power_alpha (1/f vs oscillatory)

Near-zero correlation (expected independence):
- power_alpha_Cz and power_alpha_O1 (different regions)
- ms_coverage_A and ms_coverage_D (different states)
- Different subjects' features (after normalization)
```

### Correlation QC Report

```python
def generate_correlation_qc(
    df: pd.DataFrame,
    expected_correlations: Dict[Tuple[str, str], Tuple[float, float]],
    logger: Any,
) -> Dict[str, Any]:
    """
    Check that feature correlations match expectations.
    
    Returns:
        {
            "unexpected_high_correlation": [...],
            "unexpected_low_correlation": [...],
            "correlation_matrix_summary": {...}
        }
    """
```

---

## Appendix AF: Subject-Level vs Trial-Level Features

### Feature Scope Clarification

```
Trial-Level Features (one value per trial):
- All power, ERD/ERS, connectivity features
- All phase features
- All complexity features
- All temporal features
- Most microstate features

Subject-Level Features (one value per subject):
- Individual Alpha Frequency (IAF) - stable trait
- Baseline aperiodic slope - stable trait
- Microstate templates - derived from all trials
- Mean feature values across trials

Session-Level Features (one value per session):
- Session-specific IAF
- Session-specific baseline
- QC metrics for the session
```

### Aggregation for Subject-Level Analysis

```python
def aggregate_to_subject_level(
    trial_features: pd.DataFrame,
    aggregation_method: str = "mean",  # or "median", "robust_mean"
    logger: Any,
) -> pd.DataFrame:
    """
    Aggregate trial-level features to subject-level.
    
    For each subject:
    - Compute mean/median across trials
    - Compute std across trials (variability)
    - Compute CV across trials (normalized variability)
    
    Returns DataFrame with one row per subject.
    """
```

---

## Appendix AG: Handling Multi-Session Data

### Session Alignment

If subjects have multiple sessions:

```
features/
├── sub-0001/
│   ├── ses-01/
│   │   ├── features_all.tsv
│   │   └── features_manifest.json
│   ├── ses-02/
│   │   ├── features_all.tsv
│   │   └── features_manifest.json
│   └── combined/
│       ├── features_all_sessions.tsv  # All sessions concatenated
│       └── features_subject_level.tsv  # Aggregated across sessions
```

### Session Normalization

```
Option 1: Within-session normalization
- Each session normalized to its own baseline
- Sessions are comparable after normalization

Option 2: Cross-session normalization
- All sessions normalized to first session baseline
- Captures session-to-session changes

Option 3: No normalization
- Raw values preserved
- Downstream pipeline handles normalization
```

---

## Appendix AH: Batch Processing Considerations

### Parallel Processing Strategy

```python
def extract_features_batch(
    subjects: List[str],
    config: Any,
    n_jobs: int = -1,
    logger: Any,
) -> None:
    """
    Extract features for multiple subjects in parallel.
    
    Strategy:
    1. Parallelize across subjects (not within subject)
    2. Each subject processed independently
    3. Results saved to disk immediately
    4. Combine at the end if needed
    """
```

### Incremental Processing

```python
def extract_features_incremental(
    subject: str,
    config: Any,
    force_recompute: bool = False,
    logger: Any,
) -> None:
    """
    Extract features with caching.
    
    Strategy:
    1. Check if features already exist
    2. Check if source data is newer than features
    3. Only recompute if necessary
    4. Support partial recomputation (specific domains)
    """
```

---

## Appendix AI: Integration with Existing Pipeline

### Current Pipeline Integration Points

```
preprocessing/
└── raw_to_bids.py → epochs.fif

pipelines/
└── features.py → extract_all_features()
    ├── Uses: analysis/features/*.py
    ├── Outputs: features/*.tsv
    └── Calls: save_features(), generate_manifest()

analysis/features/
├── core.py → PrecomputedData, precompute_data()
├── pipeline.py → extract_precomputed_features()
├── power.py → extract_band_power_features()
├── connectivity.py → extract_connectivity_features()
├── phase.py → extract_itpc_features(), compute_pac_comodulograms()
├── aperiodic.py → extract_aperiodic_features()
├── spectral.py → extract_spectral_features()
├── microstates.py → extract_microstate_features()
└── complexity.py → complexity measures
```

### New Modules to Add

```
analysis/features/
├── manifest.py → generate_feature_manifest()
├── validation.py → validate_features(), run_qc_checks()
├── output.py → save_organized_features(), FeatureWriter
└── naming.py → standardize_column_names(), parse_feature_name()
```

---

## Appendix AJ: Backward Compatibility

### Migration from Current Output Format

```python
# Current column names → New column names
migration_map = {
    "pow_alpha_Cz_early": "power_alpha_Cz_early_mean",
    "erds_alpha_Cz": "erds_alpha_Cz_full_percent",
    "wpli_theta_mean": "conn_wpli_theta_full_mean",
    "itpc_alpha_Cz_early": "phase_itpc_alpha_Cz_early_mean",
    "ms_coverage_0": "ms_coverage_A_full",
    "aperiodic_slope_Cz": "aper_slope_Cz_baseline",
    # ... etc
}

def migrate_feature_names(
    df: pd.DataFrame,
    migration_map: Dict[str, str],
    logger: Any,
) -> pd.DataFrame:
    """Convert old column names to new naming convention."""
```

### Deprecation Strategy

```
Version 2.0:
- New naming convention is default
- Old names still work but emit deprecation warning
- Migration helper provided

Version 2.1:
- Old names no longer work
- Migration must be complete
```

---

## Summary of All Appendices

| Appendix | Topic | Key Content |
|----------|-------|-------------|
| A | Complete Feature List | All features by domain |
| B | Pain-Specific Validation | Expected patterns, effect sizes |
| C | HRF-Aligned Extraction | Temporal bins for fMRI |
| D | ML Considerations | Feature selection, importance |
| E | Computational Efficiency | Bottlenecks, optimization |
| F | Configuration Schema | Full YAML config |
| G | Testing Strategy | Unit, integration, regression |
| H | Migration Guide | Updating existing code |
| I | Additional Pain Features | Evoked, habituation, anticipation |
| J | Stimulus-Specific | Ramp, plateau, offset |
| K | Individual Differences | Subject normalization, traits |
| L | Feature Interactions | Cross-domain combinations |
| M | Negative Controls | Sanity checks |
| N | Documentation Template | Per-feature documentation |
| O | Implementation Checklist | 7-phase plan |
| P | Feature Counts | Current vs improved |
| Q | ML-Ready Outputs | Format standards |
| R | Feature Grouping | Hierarchical organization |
| S | Data Integrity | Validation checks |
| T | Edge Cases | Missing data handling |
| U | Reproducibility | Determinism, versioning |
| V | Naming Deep Dive | Complete grammar |
| W | Channel/ROI Naming | Standardized names |
| X | Time Window Definitions | All temporal bins |
| Y | Unit Specifications | Units by feature type |
| Z | Final Checklist | Pre-implementation decisions |
| AA | Cross-Feature Consistency | Alignment guarantees |
| AB | Extraction Order | Dependency sequence |
| AC | Memory/Performance | Optimization strategies |
| AD | Sanity Checks | Value range validation |
| AE | Correlation Structure | Expected correlations |
| AF | Subject vs Trial Level | Feature scope |
| AG | Multi-Session Data | Session handling |
| AH | Batch Processing | Parallel strategies |
| AI | Pipeline Integration | Integration points |
| AJ | Backward Compatibility | Migration strategy |

---

*Document Complete*
*Total Appendices: 36*
*Estimated Implementation Time: 6-8 weeks*

---

## Appendix AK: Feature Stability Across Preprocessing Variations

### Robustness Considerations

Features should be stable across minor preprocessing variations:

```
High Stability (robust features):
- ROI-aggregated features (averaging reduces noise)
- Global features (spatial averaging)
- Relative power (ratio cancels scaling)
- Connectivity measures (phase-based, amplitude-independent)
- Normalized indices (lateralization, ERD/ERS percent)

Moderate Stability:
- Channel-level power (sensitive to reference, filtering)
- Spectral features (sensitive to window length)
- Complexity measures (sensitive to data length)

Lower Stability (use with caution):
- Absolute power (sensitive to impedance, reference)
- Peak latencies (sensitive to noise, smoothing)
- Microstate transitions (sensitive to template matching)
```

### Reference Sensitivity

```
Average Reference (recommended for this pipeline):
- Power: Redistributed across channels
- Connectivity: Minimal effect on phase-based measures
- Microstates: Required for proper GFP computation

Linked Mastoids:
- Power: Different spatial distribution
- Connectivity: May introduce spurious correlations
- Microstates: Not recommended

REST (Reference Electrode Standardization Technique):
- Power: Approximates ideal reference
- Connectivity: Best for source-level analysis
- Microstates: Compatible
```

---

## Appendix AL: Feature Interpretability Guidelines

### Interpretation Framework

For each feature domain, provide interpretation guidance:

#### Power Features
```
Interpretation:
- Positive log-ratio = power increase relative to baseline
- Negative log-ratio = power decrease (ERD)
- Magnitude indicates strength of change
- Spatial pattern indicates involved regions

Caveats:
- Absolute power depends on impedance, skull thickness
- Always use normalized measures for cross-subject comparison
- Log-ratio assumes multiplicative baseline relationship
```

#### ERD/ERS Features
```
Interpretation:
- Negative percent = Event-Related Desynchronization (activation)
- Positive percent = Event-Related Synchronization (inhibition/rebound)
- Alpha/beta ERD = cortical activation
- Beta ERS (rebound) = active inhibition

Caveats:
- ERD/ERS is relative to baseline; baseline quality matters
- Very low baseline power can inflate ERD/ERS percent
- Consider both magnitude and spatial extent
```

#### Connectivity Features
```
Interpretation:
- Higher wPLI/PLV = stronger phase synchronization
- Higher AEC = stronger amplitude coupling
- Graph efficiency = network integration
- Clustering = local processing

Caveats:
- Volume conduction can inflate connectivity (use wPLI, not PLV for EEG)
- Short epochs reduce reliability
- Connectivity is sensitive to SNR
```

#### Aperiodic Features
```
Interpretation:
- Steeper slope (more negative) = more inhibition-dominated
- Flatter slope = more excitation-dominated
- Slope change during task = E/I balance shift

Caveats:
- Slope estimation requires sufficient frequency range
- Periodic peaks can bias slope if not properly modeled
- Individual differences in baseline slope are large
```

---

## Appendix AM: Pain-Specific Feature Interpretation

### Expected Pain Effects by Feature

| Feature | Pain Effect | Interpretation | Confidence |
|---------|-------------|----------------|------------|
| Alpha ERD (sensorimotor) | Stronger | Increased sensory processing | High |
| Beta ERD (sensorimotor) | Stronger | Motor system engagement | High |
| Gamma power (central) | Increased | Local nociceptive processing | High |
| Theta power (frontal) | Increased | Affective/attentional processing | Moderate |
| Theta connectivity (frontal) | Increased | Affective network engagement | Moderate |
| Alpha connectivity | Decreased | Sensorimotor network disinhibition | Moderate |
| Aperiodic slope | Flatter | Increased E/I ratio | Moderate |
| Lateralization (alpha) | Contralateral | Somatotopic processing | High |
| ITPC (gamma) | Increased | Stimulus-locked processing | Moderate |
| Microstate duration | Altered | Changed cognitive state dynamics | Low-Moderate |

### Non-Pain Confounds to Consider

```
Attention/Arousal:
- Alpha power decreases with attention (not pain-specific)
- Theta increases with cognitive load
- Gamma increases with attention

Motor Preparation:
- Beta ERD occurs with motor preparation
- Distinguish from pain-related beta ERD

Anticipation:
- Pre-stimulus alpha ERD predicts pain perception
- May reflect expectation, not pain itself

Habituation:
- Responses decrease over repeated stimulation
- Control for trial order effects
```

---

## Appendix AN: Feature Quality Tiers

### Tier Classification

Classify features by reliability and interpretability:

```
Tier 1: Gold Standard (high reliability, well-validated)
- Alpha ERD in sensorimotor cortex
- Gamma power in central regions
- Theta power in frontal regions
- Lateralization index
- Global field power

Tier 2: Reliable (good reliability, established interpretation)
- Beta ERD/ERS
- wPLI connectivity
- Aperiodic slope
- Spectral entropy
- ITPC

Tier 3: Exploratory (moderate reliability, emerging evidence)
- PAC (theta-gamma)
- Microstate metrics
- Complexity measures
- Graph metrics
- Directed connectivity

Tier 4: Experimental (lower reliability, limited validation)
- Fine-grained latency features
- Cross-frequency phase coupling
- Microstate sequence complexity
- High-gamma (>60 Hz) features
```

### Recommended Feature Sets by Use Case

```
Publication-Ready Analysis:
- Use Tier 1 and Tier 2 features
- Report effect sizes and confidence intervals
- Validate with cross-validation

Exploratory Analysis:
- Include Tier 3 features
- Use appropriate multiple comparison correction
- Replicate findings before publication

Hypothesis Generation:
- Include all tiers
- Use for pattern discovery
- Require independent validation
```

---

## Appendix AO: Temporal Precision Considerations

### Timing Accuracy

```
Feature Type          | Temporal Precision | Notes
---------------------|-------------------|-------
Power (time-binned)  | ~1s               | Limited by bin width
ERD/ERS onset        | ~100ms            | Depends on smoothing
ITPC peak            | ~50-100ms         | Depends on frequency
Microstate duration  | ~10-50ms          | Depends on sampling rate
Connectivity         | ~1-2s             | Requires stable estimate
PAC                  | ~500ms-1s         | Requires multiple cycles
```

### Jitter and Variability

```
Sources of Temporal Variability:
1. Stimulus onset jitter (hardware-dependent)
2. Neural response variability (individual differences)
3. Estimation uncertainty (statistical)
4. Epoch alignment errors (preprocessing)

Mitigation Strategies:
1. Use precise stimulus markers
2. Report confidence intervals on latencies
3. Use robust estimation methods
4. Verify epoch alignment
```

---

## Appendix AP: Cross-Subject Normalization Strategies

### Normalization Options

```
Option 1: Within-Subject Z-Score
- For each subject, z-score features across trials
- Removes between-subject baseline differences
- Preserves within-subject trial-to-trial variability
- Best for: Within-subject analyses, repeated measures

Option 2: Within-Subject Baseline Normalization
- Normalize to pre-stimulus baseline
- Already done for ERD/ERS, log-ratio power
- Preserves absolute magnitude of change
- Best for: Comparing effect sizes across subjects

Option 3: Group Z-Score
- Z-score across all subjects and trials
- Assumes similar distributions across subjects
- May be biased by outlier subjects
- Best for: Group-level pattern discovery

Option 4: Rank Transform
- Convert to percentile ranks
- Robust to outliers and non-normality
- Loses magnitude information
- Best for: Non-parametric analyses

Option 5: No Normalization
- Preserve raw values
- Let downstream ML handle normalization
- Best for: When ML pipeline has its own normalization
```

### Recommended Approach

```
For this pipeline (feature extraction only):
1. Apply baseline normalization (ERD/ERS, log-ratio power)
2. Do NOT apply cross-subject normalization
3. Document normalization applied
4. Let downstream ML pipeline decide on additional normalization

Rationale:
- Baseline normalization is theoretically motivated
- Cross-subject normalization is analysis-dependent
- Preserving raw (baseline-normalized) values is most flexible
```

---

## Appendix AQ: Feature Redundancy Analysis

### Expected Redundancy Patterns

```
High Redundancy (r > 0.9):
- power_alpha_Cz and power_alpha_C1 (adjacent channels)
- erds_alpha_Cz_early and power_alpha_Cz_early (definitional)
- conn_wpli_alpha and conn_plv_alpha (similar measures)

Moderate Redundancy (r = 0.5-0.9):
- power_alpha and power_beta (correlated bands)
- power_alpha_early and power_alpha_mid (temporal correlation)
- roi_power and channel_power (aggregation)

Low Redundancy (r < 0.5):
- power_alpha and power_gamma (different processes)
- power and connectivity (different domains)
- Different ROIs (spatial independence)
```

### Redundancy Reduction Strategies

```
Strategy 1: ROI Aggregation
- Reduce 64 channels to 6-10 ROIs
- Preserves spatial interpretability
- Reduces redundancy by ~10x

Strategy 2: Band Selection
- Focus on pain-relevant bands (alpha, beta, gamma)
- Exclude highly correlated bands
- Reduces redundancy by ~2x

Strategy 3: Time Window Selection
- Use coarse bins (early, mid, late) instead of fine (t1-t7)
- Or use temporal dynamics features (slope, diff)
- Reduces redundancy by ~2-3x

Strategy 4: Domain Selection
- Focus on primary domains (power, ERD/ERS, connectivity)
- Exclude redundant domains (temporal overlaps with power)
- Reduces redundancy by ~2x

Combined Effect:
- From ~11,000 features to ~500-1000 features
- Maintains interpretability
- Reduces multicollinearity for ML
```

---

## Appendix AR: Documentation Standards

### Code Documentation

Each feature extraction function should document:

```python
def extract_feature_x(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract feature X from epochs.
    
    Theoretical Background
    ----------------------
    [Brief description of what this feature measures and why it's
    relevant for pain research. Include key references.]
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs, should be baseline-corrected and artifact-free.
    bands : List[str]
        Frequency bands to analyze (e.g., ["alpha", "beta"]).
    config : Any
        Configuration object with extraction parameters.
    logger : Any
        Logger for status messages.
    
    Returns
    -------
    df : pd.DataFrame
        Feature values with shape (n_epochs, n_features).
        Columns follow naming convention: {domain}_{band}_{channel}_{time}_{stat}
    columns : List[str]
        List of column names in order.
    
    Output Features
    ---------------
    - feature_x_{band}_{channel}_{time}_mean : Mean value
    - feature_x_{band}_{channel}_{time}_std : Standard deviation
    - feature_x_{band}_global_{time}_mean : Global average
    
    Notes
    -----
    - Requires at least 10 epochs for reliable estimation
    - NaN values indicate insufficient data
    - See Appendix X in feature_extraction_improvements.md for details
    
    References
    ----------
    [1] Author et al. (Year). Title. Journal.
    [2] Author et al. (Year). Title. Journal.
    
    Examples
    --------
    >>> df, cols = extract_feature_x(epochs, ["alpha", "beta"], config, logger)
    >>> print(df.shape)
    (50, 640)  # 50 epochs, 64 channels × 5 bands × 2 stats
    """
```

### Manifest Documentation

Each feature in the manifest should include:

```json
{
  "name": "power_alpha_Cz_early_mean",
  "domain": "power",
  "band": "alpha",
  "band_range_hz": [8.0, 13.0],
  "channel": "Cz",
  "roi": null,
  "time_window": "early",
  "time_range_s": [3.0, 5.0],
  "statistic": "mean",
  "unit": "log(µV²/µV²)",
  "normalization": "log-ratio to baseline",
  "baseline_window_s": [-5.0, -0.01],
  
  "description": "Mean alpha-band power at electrode Cz during the early plateau period (3-5s), expressed as log-ratio relative to pre-stimulus baseline.",
  
  "theoretical_relevance": "Alpha oscillations in central regions reflect sensorimotor cortex activity. Alpha ERD (power decrease) indicates cortical activation during sensory processing. During pain, contralateral sensorimotor cortex shows alpha ERD reflecting nociceptive input processing.",
  
  "expected_pain_effect": "decrease",
  "expected_effect_size": "Cohen's d = 0.4-0.8",
  "reliability": "high",
  "tier": 1,
  
  "computation_details": {
    "method": "Morlet wavelet TFR",
    "n_cycles": "frequency-dependent (freq/2)",
    "baseline_method": "log-ratio",
    "aggregation": "mean over time window"
  },
  
  "quality_notes": [
    "Requires clean data (artifacts inflate alpha)",
    "Sensitive to eye closure (increases alpha)",
    "May be affected by drowsiness"
  ],
  
  "references": [
    "Ploner et al., 2006",
    "May et al., 2012",
    "Tu et al., 2016"
  ],
  
  "related_features": [
    "erds_alpha_Cz_early_percent",
    "power_alpha_Sensorimotor_Contra_early_mean",
    "roi_power_alpha_Central_early_mean"
  ]
}
```

---

## Appendix AS: Version Control for Features

### Feature Schema Versioning

```
Schema Version: Major.Minor.Patch

Major: Breaking changes
- Column names changed
- Feature computation changed
- Output format changed

Minor: Backward-compatible additions
- New features added
- New metadata fields
- New QC checks

Patch: Bug fixes
- Computation bug fixed
- Documentation updated
- No output changes
```

### Changelog Format

```markdown
# Feature Schema Changelog

## [2.1.0] - 2024-02-01
### Added
- Fine temporal bins (t1-t7) for all power features
- ITPC temporal dynamics features
- ROI-to-ROI connectivity features

### Changed
- Renamed `pow_*` to `power_*` for consistency

### Fixed
- Corrected ERD/ERS baseline window alignment

## [2.0.0] - 2024-01-15
### Added
- Standardized naming convention
- Feature manifest generation
- QC validation framework

### Changed
- Complete restructure of output format
- New column naming scheme

### Removed
- Deprecated `extract_legacy_features()` function
```

---

## Final Summary

This document provides a comprehensive blueprint for improving the EEG feature extraction pipeline with:

### Core Improvements
1. **Standardized naming convention** for all ~11,000 features
2. **Fine temporal resolution** (7 bins) for HRF modeling
3. **Complete feature matrix** across domains × bands × channels × times
4. **Theoretical justification** for each feature based on pain literature

### Output Organization
5. **Hierarchical output structure** (by domain, band, time, ROI)
6. **Feature manifest** with full documentation
7. **QC reports** for data quality validation
8. **ML-ready format** for downstream pipelines

### Quality Assurance
9. **Data integrity checks** (pre-save, post-load)
10. **Sanity checks** (value ranges, correlations)
11. **Reproducibility** (versioning, checksums)
12. **Edge case handling** (missing data, outliers)

### Documentation
13. **Per-feature documentation** in manifest
14. **Code documentation standards**
15. **Interpretation guidelines**
16. **Migration guide** for existing code

---

*Document Version: 2.3*
*Total Sections: 49 (13 main + 36 appendices)*
*Total Lines: ~3200*
*Ready for Implementation Review*


