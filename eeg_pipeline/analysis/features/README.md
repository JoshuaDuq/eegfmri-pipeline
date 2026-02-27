# Feature Extraction Pipeline

**Module:** `eeg_pipeline.analysis.features`

This document is the methods reference for the trial-level EEG feature extraction pipeline.
Each trial (epoch) produces one row in a feature matrix.
The pipeline targets event-related paradigms with well-defined time windows (e.g. pain research),
but is applicable to any EEG study with the same structure.

---

## Table of Contents

1. [Notation](#1-notation)
2. [Module Structure](#2-module-structure)
3. [Configuration, Categories, and Naming](#3-configuration-categories-and-naming)
4. [Time Windows](#4-time-windows)
5. [Frequency Bands and Individual Alpha Frequency](#5-frequency-bands-and-individual-alpha-frequency)
6. [Precomputed Intermediates and Spatial Transforms](#6-precomputed-intermediates-and-spatial-transforms)
7. [Extraction Pipelines](#7-extraction-pipelines)
8. [Feature Definitions](#8-feature-definitions)
9. [Change Scores](#9-change-scores)
10. [Normalization](#10-normalization)
11. [Cross-Validation Hygiene](#11-cross-validation-hygiene)
12. [Result Containers](#12-result-containers)
13. [Dependencies](#13-dependencies)

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $e \in \{1,\dots,N_\text{trials}\}$ | Trial (epoch) index |
| $c \in \{1,\dots,N_\text{ch}\}$ | Channel index |
| $r$ | Region-of-interest (ROI) index |
| $f$ | Frequency (Hz) |
| $t$ | Time (s) |
| $x_{e,c}(t)$ | Time-domain EEG signal |
| $P_{e,c}(f,t)$ | Time–frequency power (Morlet TFR) |
| $\mathrm{PSD}_{e,c}(f)$ | Power spectral density (µV²/Hz) |
| $\mathcal{H}(\cdot)$ | Hilbert transform |
| $B = [f_\text{min}^B, f_\text{max}^B]$ | Frequency band with bounds |
| $T_\text{seg}$ | Time-window segment (baseline, active, …) |
| $M_\text{seg}(t) \in \{0,1\}$ | Boolean mask for segment $T_\text{seg}$ |
| $\Delta f$ | Frequency bin width |
| $\varepsilon$ | Small positive constant to prevent division by zero |

---

## 2. Module Structure

```
features/
├── api.py                  # Entry points: extract_all_features, extract_precomputed_features
├── selection.py            # Feature category resolution and spatial-mode filtering
├── preparation.py          # Epoch validation, TFR setup, precompute_data (band filtering, PSD, GFP, evoked subtraction), baseline metrics
├── cv_hygiene.py           # Cross-validation guards: fold-specific IAF, train-mask enforcement
├── normalization.py        # Train/test-separated normalization schemes
├── results.py              # FeatureExtractionResult, ExtractionResult dataclasses
├── spectral.py             # Power (TFR-based), spectral descriptors
├── aperiodic.py            # 1/f aperiodic component decomposition
├── erp.py                  # ERP components (peak, mean, AUC, latency)
├── connectivity.py         # Undirected and directed connectivity
├── phase.py                # ITPC, PLV, PAC, phase-amplitude coupling
├── complexity.py           # LZC, permutation entropy, sample entropy, MSE
├── bursts.py               # Transient oscillation detection
├── microstates.py          # EEG microstate segmentation and statistics
├── quality.py              # Trial- and channel-level QC metrics
├── source_localization.py  # LCMV / eLORETA source-space features
└── precomputed/
    ├── __init__.py         # Package init
    ├── erds.py             # ERDS extraction from precomputed band envelopes
    └── extras.py           # Band-power ratios, hemispheric asymmetry
```

---

## 3. Configuration, Categories, and Naming

### 3.1 Feature Categories

Feature categories are resolved by `resolve_feature_categories` in `selection.py`, in priority order:

1. Explicitly requested categories (CLI or API call).
2. `feature_engineering.feature_categories` from configuration.
3. All entries in `FEATURE_CATEGORIES` if neither of the above is set.

Available categories (order matches `FEATURE_CATEGORIES` in `eeg_pipeline.domain.features.constants`):

```
power, spectral, aperiodic, erp, erds, ratios, asymmetry, microstates,
connectivity, directedconnectivity, itpc, pac, sourcelocalization,
complexity, bursts, quality
```

### 3.2 Column Naming Schema

Most feature columns follow the pattern:

```
{domain}_{segment}_{band}_{scope}_{statistic}
```

| Component | Values |
|-----------|--------|
| `domain` | `power`, `erp`, `itpc`, `erds`, `conn`, `comp`, … |
| `segment` | `baseline`, `active`, task-specific window names |
| `band` | `alpha`, `theta`, `beta`, `broadband`, … |
| `scope` | `ch` (channel), `roi`, `global`, `chpair` |
| `statistic` | `logratio`, `mean`, `peak_freq`, `db`, … |

**Exceptions:**
- Source-space features use `src_*` prefix for fMRI pipeline compatibility.
- Connectivity and microstate metrics encode method or state names in the suffix.

### 3.3 Spatial Aggregation Modes

Controlled by `feature_engineering.spatial_modes` and enforced by
`filter_features_by_spatial_modes` in `api.py`:

| Mode | Retained columns |
|------|-----------------|
| `channels` | `*_ch_*`, `*_chpair_*` |
| `roi` | `*_roi_*`, names matching ROI labels |
| `global` | `*_global_*`, columns ending with `_global` |

Non-spatial features (global scalar summaries) are always retained.
Default when not set in config: `["roi", "global"]` (from `FeatureContext`); config key
`feature_engineering.spatial_modes` can override. Omit `--spatial` to use config, or pass
e.g. `--spatial roi global` to restrict to a subset.

---

## 4. Time Windows

Time windows are represented by `TimeWindows` (fields: `baseline_range`, `active_range`,
`masks`, `ranges`, `times`) and constructed from `TimeWindowSpec` objects via
`time_windows_from_spec`.

For a segment with mask $M_\text{seg}(t)$, the masked mean is:

```math
\bar{x}_{e,c}^{(\text{seg})} =
\frac{\sum_t M_\text{seg}(t)\, x_{e,c}(t)}{\sum_t M_\text{seg}(t)}.
```

When epochs are cropped to $[t_\text{min}, t_\text{max}]$ in the TFR pipeline,
masks are rebuilt on the cropped time axis to avoid shape mismatches.
Baseline and active ranges are kept in physical time so baseline-dependent
computations remain unaffected.

Segments failing per-feature duration or cycle-count requirements are skipped
for that feature.

---

## 5. Frequency Bands and Individual Alpha Frequency

### 5.1 Base Frequency Bands

Read from configuration via `get_frequency_bands(config)`. Standard bands:

```
delta, theta, alpha, beta, gamma
```

Each band $B$ has bounds $[f_\text{min}^B, f_\text{max}^B]$.

### 5.2 IAF Estimation

When `feature_engineering.bands.use_iaf = true`, the Individual Alpha Frequency (IAF)
is estimated at the subject level by `precompute_data`:

1. Extract baseline data using the baseline mask (required; no fallback to full segment).
2. Compute PSD in a broad range (default ≈ 1–40 Hz) using `mne.time_frequency.psd_array_multitaper`.
3. Remove the aperiodic 1/f trend by fitting a log–log linear regression:

```math
\log_{10} P_\text{resid}(f) =
\log_{10} P(f) - \bigl(\beta_0 + \beta_1 \log_{10} f\bigr).
```

4. Estimate $\hat{f}_\alpha$ as either:
   - the most prominent residual peak in the search range (default 7–13 Hz), or
   - a residual-power-weighted frequency centroid if no peak passes the prominence threshold.

5. Adjust bands:

```math
\begin{aligned}
\text{alpha} &= [\hat{f}_\alpha - w_\alpha,\; \hat{f}_\alpha + w_\alpha],\\
\text{theta} &= [\max(3,\; \hat{f}_\alpha - 6),\; \max(4,\; f_\text{min}^\alpha)],\\
\text{beta}  &= [\max(13,\; f_\text{max}^\alpha),\; f_\text{max}^\beta].
\end{aligned}
```

All IAF-dependent steps include explicit duration and cycle-count checks.

### 5.3 Fold-Specific IAF in Cross-Validation

`cv_hygiene.compute_iaf_for_fold` restricts IAF estimation to training trials:

- Input: boolean `train_mask` over epochs.
- Output: $\text{IAF}_\text{fold}$ (Hz) and IAF-adjusted band dictionary $\mathcal{B}_\text{fold}$.
- If the training set is too small, IAF estimation is skipped and `(None, None)` is returned with a warning logged.

This prevents spectral band definitions from leaking test-trial information.

---

## 6. Precomputed Intermediates and Spatial Transforms

### 6.1 Spatial Transform Selection

Applied by `precompute_data` before band filtering. Configured globally via
`feature_engineering.spatial_transform ∈ {none, csd, laplacian}`, with optional
per-family overrides in `feature_engineering.spatial_transform_per_family`.

The standard transform is Current Source Density (CSD):

```math
x^\text{CSD}(t) =
\text{compute\_current\_source\_density}\!\bigl(x(t);\; \lambda^2,\; \text{stiffness}\bigr),
```

where $\lambda^2$ and stiffness are configured parameters.
A failed transform raises a `RuntimeError`; it is never silently skipped.

### 6.2 Precomputed Data Container

`PrecomputedData` stores shared intermediates used by multiple feature families:

- Band-filtered analytic signals (envelope, phase, band-specific GFP)
- PSD estimates (full or baseline-restricted)
- `TimeWindows` object
- Broadband GFP
- Frequency band definitions (IAF-adjusted if applicable)
- Quality-control metadata (finite fractions, medians, frequency ranges, window counts)

The main function is `precompute_data(epochs, bands, config, logger, ...)`, which accepts
optional keyword-only arguments `compute_bands` and `compute_psd_data` (both default `True`)
to control whether band-filtered data and PSD are computed; the precomputed-only pipeline
may set them to `False` for unused feature groups to avoid unnecessary computation. It:

1. Selects EEG channels and applies the configured spatial transform.
2. Rebuilds `TimeWindows` on the actual time axis if needed.
3. Computes band-filtered data $x_{e,c}^B(t)$, per-band GFP $\mathrm{GFP}^B_e(t)$,
   PSD data, and global GFP.
4. Stores QC measures.

Precomputed objects are keyed per feature family and shared across families with
compatible spatial transform requirements.

### 6.3 Evoked-Subtracted Data (Induced Power)

When `feature_engineering.power.subtract_evoked = true`, the evoked (phase-locked)
response is removed per condition before band filtering:

```math
x^\text{induced}_{e,c}(t) = x_{e,c}(t) - \bar{x}_{k(e),c}(t),
```

where $\bar{x}_{k(e),c}(t)$ is the condition-$k$ trial average.

In `trial_ml_safe` mode, a valid `train_mask` is required for evoked subtraction.
Without it, a `ValueError` is raised, since condition averages computed over all
trials would leak test-trial information.

---

## 7. Extraction Pipelines

### 7.1 TFR-Based Pipeline (`extract_all_features`)

Entry point in `api.py`. Performs the following steps in order:

1. Optionally crop epochs to $[t_\text{min}, t_\text{max}]$ and rebuild `TimeWindows`.
2. Precompute band data when any of the following categories are requested:
   `connectivity`, `directedconnectivity`, `erds`, `ratios`, `asymmetry`, `complexity`, `bursts`, `spectral`, `aperiodic`.
3. Validate epochs (montage, sampling rate, duration).
4. Compute TFR and baseline metrics once for categories `power`, `itpc`, `pac`.
5. Call per-category extractors via `_extract_feature_with_error_handling`, which enforces:
   - Strict trial-count matching between output and input.
   - Timing and logging per category.
   - Immediate surfacing of shape mismatches or unexpected return types.
6. Apply spatial-mode filtering to the following feature DataFrames: power, ERP, aperiodic, connectivity, complexity, bursts, spectral. (Directed connectivity, phase, ITPC, PAC, ERDS, ratios, asymmetry, microstates, quality, and source are not passed through this filter.)
7. Append change scores (see §9).

Output: `FeatureExtractionResult` dataclass with one DataFrame per domain.

### 7.2 Precomputed-Only Pipeline (`extract_precomputed_features`)

A lighter-weight entry point for cases where TFR features are not needed or
precomputed intermediates already exist:

- Accepts `epochs`, `bands`, `config`, `logger`, and optional `events_df`, `precomputed`,
  and `feature_groups` (list of group names to extract; default `["erds", "spectral"]` when `None`).
- Calls `precompute_data` if `precomputed` is `None`.
- Extracts the requested feature groups. Note: the directed-connectivity group key is
  `directed_connectivity` (with underscore), not `directedconnectivity` (the TFR pipeline category).
- Returns an `ExtractionResult` with a `features` dict and optional condition labels.

This is the preferred path for large-scale batch processing and cross-modality sharing
of features (e.g. PAC and connectivity features shared across models).

---

## 8. Feature Definitions

### 8.1 Power (Oscillatory Power)

**Module:** `spectral.py` → `extract_power_features`

Starting from TFR power $P_{e,c}(f,t)$:

**Step 1 — Segment average per frequency:**

```math
\bar{P}_{e,c}^{B,\text{seg}}(f) =
\frac{\sum_t M_\text{seg}(t)\, P_{e,c}(f,t)}{\sum_t M_\text{seg}(t)}.
```

**Step 2 — Band-integrated power:**

```math
P_{e,c}^{B,\text{seg}} =
\frac{\sum_{f \in B} \bar{P}_{e,c}^{B,\text{seg}}(f)\,\Delta f}{\sum_{f \in B} \Delta f}.
```

**Step 3 — Baseline-normalized log-ratio** (when baseline is available and TFR is not pre-baselined):

```math
\text{logratio}_{e,c}^B =
\log_{10}\!\left(
  \frac{\max(P_{e,c}^{B,\text{active}},\, \varepsilon)}
       {\max(P_{e,c}^{B,\text{baseline}},\, \varepsilon)}
\right), \quad \varepsilon = 10^{-20}.
```

**Step 4 — dB scaling:**

```math
\mathrm{dB}_{e,c}^B = 10 \cdot \text{logratio}_{e,c}^B.
```

If the TFR was pre-baselined (e.g. `logratio` or `percent` mode), the baselined
values are used directly.
Line-noise harmonics within a configurable exclusion window around mains-frequency
multiples are removed before band integration.

**Outputs:** `power_{segment}_{band}_{scope}_logratio`;
optionally `..._db` (`emit_db = true`) and `..._log10raw` (no baseline, `require_baseline = false`).

---

### 8.2 Spectral Descriptors

**Module:** `spectral.py` → `extract_spectral_features`

PSD estimated per trial and channel (multitaper or Welch). Within band $B$:

**Center frequency (spectral CoG):**

```math
f_\text{cog} =
\frac{\sum_{f \in B} f\,\mathrm{PSD}(f)\,\Delta f}
     {\sum_{f \in B} \mathrm{PSD}(f)\,\Delta f}.
```

**Bandwidth (power-weighted standard deviation):**

```math
\sigma_B =
\sqrt{
  \frac{\sum_{f \in B} (f - f_\text{cog})^2\,\mathrm{PSD}(f)\,\Delta f}
       {\sum_{f \in B} \mathrm{PSD}(f)\,\Delta f}
}.
```

**Normalized spectral entropy** (with $p(f) = \mathrm{PSD}(f)\,\Delta f \,/\, \sum_{f \in B} \mathrm{PSD}(f)\,\Delta f$):

```math
H_B = -\frac{\sum_{f \in B} p(f)\ln p(f)}{\ln N_B},
```

where $N_B$ is the number of frequency bins in $B$.

**Peak features** (via shared aperiodic residuals):

- Residual: $r(f) = \log_{10}\mathrm{PSD}(f) - \widehat{\log_{10}\mathrm{PSD}}_\text{aperiodic}(f)$.
- Peak frequency: $f^* = \arg\max_{f \in B} r(f)$; CoG is used when peak prominence is insufficient.
- Peak power, ratio, and residual derived at $f^*$.

**Broadband spectral edge:**

$f_\text{edge,95}$ is the smallest $f$ satisfying:

```math
\frac{\sum_{f' \le f} \mathrm{PSD}(f')\,\Delta f'}
     {\sum_{f'} \mathrm{PSD}(f')\,\Delta f'} \ge 0.95.
```

Segments shorter than `min_segment_sec` or with fewer than `min_cycles_at_fmin` cycles are skipped.

---

### 8.3 Aperiodic (1/f) Components

**Module:** `aperiodic.py` → `extract_aperiodic_features`

**Step 1 — Log–log transform:**

```math
x(f) = \log_{10} f, \qquad y(f) = \log_{10}\mathrm{PSD}(f).
```

**Step 2 — Iterative aperiodic fit** in $[f_\text{min}, f_\text{max}]$ (e.g. 2–40 Hz):

- Fit model; compute residuals $r(f) = y(f) - \hat{y}(f)$.
- Remove bins with large positive residuals (putative oscillatory peaks, threshold in MAD units).
- Repeat for a fixed maximum number of iterations.

**Step 3 — Model options:**

*Fixed-slope (linear):*
```math
y(f) = \text{offset} + \text{slope} \cdot x(f).
```

*Knee model:*
```math
y(f) = \text{offset} - \log_{10}\!\bigl(\text{knee} + f^\text{exponent}\bigr).
```

**Outputs per segment:** `slope`, `offset`, `exponent`, `knee`, `r2`, `rms`;
aperiodic-corrected band powers:

```math
\text{powcorr}_B = \sum_{f \in B} 10^{r(f)}\,\Delta f;
```

aperiodic-corrected theta/beta ratio:

```math
\text{tbr} =
\frac{\mathrm{mean}_{f \in \theta} 10^{r(f)}}
     {\mathrm{mean}_{f \in \beta}  10^{r(f)}}, \qquad
\text{tbr\_raw} =
\frac{\mathrm{mean}_{f \in \theta} \mathrm{PSD}(f)}
     {\mathrm{mean}_{f \in \beta}  \mathrm{PSD}(f)}.
```

Residual spectral peaks are summarized by center frequency, bandwidth, and height.

---

### 8.4 ERP (Evoked Potentials)

**Module:** `erp.py` → `extract_erp_features`

For each ERP component window $T_\text{comp}$ (e.g. N2, P2) and channel or ROI:

**Baseline correction:**

```math
\tilde{x}_{e,c}(t) = x_{e,c}(t) - \bar{x}_{e,c}^{(\text{baseline})}.
```

**Component mean:**

```math
\text{mean}_{e,c}^\text{comp} =
\frac{1}{|T_\text{comp}|} \sum_{t \in T_\text{comp}} \tilde{x}_{e,c}(t).
```

**Peak amplitude and latency** (polarity $s \in \{\text{identity}, -1, |\cdot|\}$):

```math
t^* = \arg\max_{t \in T_\text{comp}} s\bigl(\tilde{x}_{e,c}(t)\bigr), \qquad
\text{peak}_{e,c}^\text{comp} = \tilde{x}_{e,c}(t^*), \qquad
\text{latency}_{e,c}^\text{comp} = t^*.
```

**Area under the curve:**

```math
\text{auc}_{e,c}^\text{comp} =
\int_{t \in T_\text{comp}} \tilde{x}_{e,c}(t)\, dt,
```

computed as a trapezoidal sum over contiguous valid intervals.

Paired components (e.g. N2–P2) yield peak-to-peak amplitude and latency differences.

---

### 8.5 ERDS (Event-Related Desynchronization / Synchronization)

**Module:** `precomputed/erds.py` → `extract_erds_from_precomputed`

Using precomputed band envelopes $|\mathcal{H}(x_{e,c}^B(t))|$:

**Baseline and active power:**

```math
P^{B,\text{baseline}}_{e,c} =
\mathrm{mean}_{t \in T_\text{baseline}} |\mathcal{H}(x_{e,c}^B(t))|^2, \qquad
P^{B,\text{active}}_{e,c} =
\mathrm{mean}_{t \in T_\text{active}} |\mathcal{H}(x_{e,c}^B(t))|^2.
```

**ERDS percentage:**

```math
\text{ERDS\%}_{e,c}^B =
100 \cdot
\frac{P^{B,\text{active}}_{e,c} - P^{B,\text{baseline}}_{e,c}}
     {P^{B,\text{baseline}}_{e,c}}.
```

**ERDS in dB:**

```math
\text{ERDS}_\text{dB} =
10 \log_{10}\!\left(\frac{P^{B,\text{active}}_{e,c}}{P^{B,\text{baseline}}_{e,c}}\right).
```

Very low baseline power is treated as invalid; no clamping is applied.

Additional outputs include per-channel ERDS, ROI/global aggregates, slopes, onset and peak
latencies, and pain-specific markers (contralateral somatosensory ERD, rebound magnitude).

---

### 8.6 Ratios (Band Power Ratios)

**Module:** `precomputed/extras.py` → `extract_band_ratios_from_precomputed`

PSD-integrated band power:

```math
P^B_{e,c} = \sum_{f \in B} \mathrm{PSD}_{e,c}(f)\,\Delta f.
```

For a numerator band $B_\text{num}$ and denominator band $B_\text{den}$:

```math
\text{power\_ratio}_e =
\frac{P^{B_\text{num}}_e}{P^{B_\text{den}}_e}, \qquad
\text{log\_ratio}_e =
\ln\!\bigl(P^{B_\text{num}}_e + \varepsilon\bigr) -
\ln\!\bigl(P^{B_\text{den}}_e + \varepsilon\bigr).
```

ROI and global ratios are computed from PSDs averaged across channels, not from averages
of per-channel ratios.

---

### 8.7 Asymmetry (Hemispheric Indices)

**Module:** `precomputed/extras.py` → `extract_asymmetry_from_precomputed`

For a left–right electrode pair $(L, R)$ and band $B$:

```math
\text{index} =
\frac{P^B_R - P^B_L}{P^B_R + P^B_L}, \qquad
\text{logdiff} = \ln P^B_R - \ln P^B_L.
```

Optional activation-convention alpha asymmetry (higher value → greater right-hemisphere activation):

```math
\text{logdiff\_activation} = -\text{logdiff}.
```

---

### 8.8 Connectivity (Undirected)

**Module:** `connectivity.py` → `extract_connectivity_features`

Let $X_i(f,t)$ and $X_j(f,t)$ be complex Fourier coefficients or analytic signals for
channels $i, j$, with cross-spectrum $S_{ij}(f)$ and auto-spectra $S_{ii}(f)$, $S_{jj}(f)$.

**Weighted PLI (wPLI):**

```math
\text{wPLI}_{ij} =
\frac{\bigl|\mathbb{E}[\mathrm{Im}(X_i X_j^*)]\bigr|}
     {\mathbb{E}[|\mathrm{Im}(X_i X_j^*)|]}.
```

**Phase Lag Index (PLI):**

```math
\text{PLI}_{ij} =
\bigl|\mathbb{E}[\mathrm{sign}(\mathrm{Im}(X_i X_j^*))]\bigr|.
```

**Imaginary Coherence:**

```math
\text{imCoh}_{ij} =
\mathrm{Im}\!\left(\frac{S_{ij}}{\sqrt{S_{ii}\, S_{jj}}}\right).
```

**Phase Locking Value (PLV):**

```math
\text{PLV}_{ij} =
\bigl|\mathbb{E}[e^{i\Delta\varphi_{ij}}]\bigr|.
```

**Amplitude Envelope Correlation (AEC):**

```math
r_{ij} = \mathrm{corr}(A_i, A_j), \qquad
z_{ij} = \mathrm{atanh}\!\bigl(\mathrm{clip}(r_{ij}, -0.9999, 0.9999)\bigr).
```

Connectivity can be estimated trial-wise, condition-wise, or subject-wise.
Optional graph metrics (clustering coefficient, global efficiency, small-world index) are
derived from thresholded or weighted connectivity matrices.
Phase-based measures computed without a spatial transform (CSD/Laplacian) emit a
volume-conduction warning.

---

### 8.9 Directed Connectivity

**Module:** `connectivity.py` → `extract_directed_connectivity_features`

**Phase Slope Index (PSI):**

```math
C_{ij}(f) = \frac{S_{ij}(f)}{\sqrt{S_{ii}(f)\,S_{jj}(f)}}, \qquad
\text{PSI}_{ij} =
\mathrm{Im}\!\left(\sum_f C_{ij}^*(f)\, C_{ij}(f + \Delta f)\right).
```

**Directed Transfer Function (DTF)** from MVAR transfer matrix $H(f)$:

```math
\text{DTF}_{i \leftarrow j}(f) =
\frac{|H_{ij}(f)|}{\sqrt{\sum_k |H_{ik}(f)|^2}}.
```

**Partial Directed Coherence (PDC)** from MVAR coefficient matrix $A(f)$:

```math
\text{PDC}_{i \leftarrow j}(f) =
\frac{|A_{ij}(f)|}{\sqrt{\sum_k |A_{kj}(f)|^2}}.
```

Outputs include forward/backward directed influence and asymmetry summaries at trial and global
levels. MVAR model order is automatically reduced when data do not support the requested order.

---

### 8.10 ITPC and Phase Metrics

**Module:** `phase.py` → `extract_phase_features`, `extract_itpc_from_precomputed`

Unit phasor from complex time–frequency data $Z_e(f,t)$:

```math
u_e(f,t) = \frac{Z_e(f,t)}{|Z_e(f,t)| + \varepsilon}.
```

**ITPC** over trial set $\mathcal{T}$:

```math
\text{ITPC}(f,t) =
\left|\frac{1}{|\mathcal{T}|}\sum_{e \in \mathcal{T}} u_e(f,t)\right|.
```

Supported averaging modes:

| Mode | Description |
|------|-------------|
| `global` | All trials |
| `fold_global` | Training trials only (default; CV-safe) |
| `loo` | Leave-one-out |
| `condition` | Per condition |

ITPC is band- and segment-averaged to yield scalar features in $[0, 1]$.

---

### 8.11 PAC (Phase–Amplitude Coupling)

**Module:** `phase.py` → `extract_pac_from_precomputed`, `compute_pac_comodulograms`

For phase band $B_\phi$ and amplitude band $B_A$:

**Mean vector length (MVL):**

```math
u(t) = \mathbb{E}_{f_\phi \in B_\phi}\!\left[e^{i\phi(f_\phi,t)}\right], \qquad
A(t) = \mathbb{E}_{f_A \in B_A}\!\left[\mathrm{amp}(f_A,t)\right],
```

```math
\text{MVL} = \frac{\left|\sum_t A(t)\,u(t)\right|}{\sum_t A(t)}.
```

**Surrogate-based $z$-score** (optional):

```math
z = \frac{\text{MVL}_\text{obs} - \mu_\text{surr}}{\sigma_\text{surr}},
```

where $\mu_\text{surr}$ and $\sigma_\text{surr}$ are estimated from surrogate PAC values
(trial-shuffled and/or circularly time-shifted).

**Output formats:**
- Trial-wise scalar features per band pair and segment.
- Full comodulograms with phase and amplitude frequency axes.
- Time-resolved PAC traces (optional).

Harmonic overlap guards reject scientifically invalid band combinations.

---

### 8.12 Source Localization

**Module:** `source_localization.py` → `extract_source_localization_features`

Using a forward model and LCMV or eLORETA inverse solution, source signals $x_v(t)$ are
projected to ROI time courses:

```math
x_\text{ROI}(t) = \frac{1}{|V_\text{ROI}|}\sum_{v \in V_\text{ROI}} x_v(t).
```

**Source-space band power:**

```math
\text{src\_power}^{B,\text{seg}}_\text{ROI} =
\frac{\sum_{f \in B} \mathrm{PSD}_\text{ROI}(f)\,\Delta f}{\sum_{f \in B} \Delta f}.
```

Additional outputs include source-space Hilbert envelopes averaged over segments
and global averages across ROIs.
When fMRI constraints are enabled, the source space is restricted to suprathreshold
fMRI activation clusters.

---

### 8.13 Complexity

**Module:** `complexity.py` → `extract_complexity_from_precomputed`

Computed on band-filtered time series or envelopes.

**Lempel–Ziv Complexity (LZC):**

Binarize: $b_t = \mathbf{1}[x_t > \mathrm{median}(x)]$.
Let $c$ be the LZ76 parsing complexity count; normalize by the random-sequence expectation:

```math
\text{LZC} = \frac{c}{n / \log_2 n}, \quad n = \text{sequence length}.
```

**Permutation Entropy (PE):**

For embedding dimension $m$ and delay $\tau$, compute ordinal pattern probabilities $p(\pi)$:

```math
\text{PE} = -\frac{\sum_\pi p(\pi)\log_2 p(\pi)}{\log_2(m!)}.
```

**Sample Entropy (SampEn):**

With pattern length $m$ and tolerance $r\sigma_x$, let $B$ be the number of template matches
at length $m$ and $A$ at length $m+1$:

```math
\text{SampEn}(m, r) = -\log\frac{A}{B}.
```

**Multiscale Entropy (MSE):**

For scale $s$, coarse-grain $x(t)$ by averaging non-overlapping blocks of length $s$,
then compute $\text{SampEn}$ of the coarse-grained series to yield $\text{MSE}(s)$.

---

### 8.14 Bursts (Transient Oscillations)

**Module:** `bursts.py` → `extract_burst_features`

From band envelopes $E_{e,c}^B(t) = |\mathcal{H}(x_{e,c}^B(t))|$:

**Threshold estimation** (configured method):

| Method | Formula |
|--------|---------|
| Percentile | $\theta = \mathrm{percentile}(E_\text{baseline},\, q)$ |
| z-score | $\theta = \mu + z\sigma$ |
| MAD | $\theta = \mathrm{median}(E_\text{baseline}) + z \cdot 1.4826 \cdot \mathrm{MAD}(E_\text{baseline})$ |

**Burst detection:**

Identify contiguous intervals where $E_{e,c}^B(t) > \theta$ with minimum duration:

```math
T_\text{burst} \ge \max\!\left(T_\text{min},\; \frac{\text{min\_cycles}}{f_\text{center}}\right).
```

**Outputs per segment and band:** burst count, rate (Hz), mean duration (ms),
mean amplitude, and occupancy fraction.

---

### 8.15 Microstates

**Module:** `microstates.py` → `extract_microstate_features`

**Global Field Power:**

```math
\text{GFP}_e(t) =
\sqrt{\frac{1}{N_\text{ch}} \sum_c \bigl(x_{e,c}(t) - \bar{x}_e(t)\bigr)^2}.
```

GFP peaks are detected; scalp maps at these peaks are normalized (zero-mean, unit-norm,
sign-standardized) and clustered into $K$ microstate templates via K-means or provided fixed
templates. Labels are backfitted to the full time series with minimum-duration smoothing.

**Per-state statistics** for state $k$:

```math
\text{coverage}_k = \frac{N_k}{N_t}, \qquad
\text{mean duration (ms)}, \qquad
\text{occurrence rate (Hz)}.
```

where $N_k$ is the number of time points with $s(t) = k$ and $N_t$ is the total number of time points.

Transition probabilities between states are also computed.
In `trial_ml_safe` mode, template clustering is restricted to training trials.

---

### 8.16 Quality Metrics

**Module:** `quality.py` → `extract_quality_features`, `compute_trial_quality_metrics`

Per segment and channel:

| Metric | Formula |
|--------|---------|
| Variance | $\mathrm{Var}(x) = \frac{1}{N_t-1}\sum_t (x(t)-\bar{x})^2$ |
| Peak-to-peak | $\mathrm{ptp} = \max_t x(t) - \min_t x(t)$ |
| Finite fraction | $\mathrm{finite} = N_\text{finite} / N_t$ |
| SNR (dB) | $10\log_{10}(d_\text{signal} / d_\text{noise})$, where $d$ is band-limited power density |
| Muscle artifact index | Fraction of PSD power in high-frequency band (e.g. 30–80 Hz) |

These metrics serve both as QC features and as gatekeepers for downstream processing.

---

## 9. Change Scores

After primary feature extraction, `_add_change_scores_to_results` appends change-score
columns when `feature_engineering.compute_change_scores = true`.

For a feature column $X$ with baseline and active variants, the transform is configured
via `feature_engineering.change_scores.transform` (or `--change-scores-transform`):
`difference` (default), `percent`, `log_ratio`, or `ratio` (CLI alias for `percent`). Example (difference):

```math
X_\Delta = X^\text{active} - X^\text{baseline}.
```

`compute_change_features` inspects column names to identify paired segments and produces
consistent columns with segment name `change` (or `pct_change`/`log_ratio`) without recomputing
the underlying features.
Change scores are stored alongside original features to avoid redundant derivation downstream.

Supported families: power, connectivity, directedconnectivity, aperiodic, itpc (in `phase_df`), pac, erds,
spectral, ratios, asymmetry, complexity, microstates, sourcelocalization.

---

## 10. Normalization

**Module:** `normalization.py`

Normalization is intentionally separated from feature extraction.
All schemes estimate parameters from a reference set $x^\text{ref}$, which must be
the training subset in cross-validation contexts.

### 10.1 Normalization Methods

**Z-score:**

```math
z_i =
\frac{x_i - \mu^\text{ref}}{\max(\sigma^\text{ref},\, \varepsilon)}, \qquad
\mu^\text{ref} = \mathrm{mean}(x^\text{ref}), \quad
\sigma^\text{ref} = \mathrm{sd}(x^\text{ref}).
```

**Robust (median / MAD):**

```math
z_i^\text{robust} =
\frac{x_i - m^\text{ref}}
     {\max\!\bigl(\mathrm{MAD}(x^\text{ref})_\text{normal},\, \varepsilon\bigr)}, \qquad
m^\text{ref} = \mathrm{median}(x^\text{ref}),
```

where $\mathrm{MAD}(\cdot)_\text{normal}$ is the normal-consistent MAD (scaled by 1.4826).

**Min–max** to $[a, b]$:

```math
x'_i = a + (b-a)
\frac{x_i - x_\text{min}^\text{ref}}
     {\max\!\bigl(x_\text{max}^\text{ref} - x_\text{min}^\text{ref},\, \varepsilon\bigr)}.
```

**Rank-based** (0–1 normalized ranks on finite values):

```math
r_i =
\frac{\mathrm{rank}(x_i) - 1}{n_\text{finite} - 1},
```

where ties are handled by a configurable ranking method (default: average) and
$n_\text{finite}$ is the count of finite entries.

**Log:**

```math
\log_b(x_i) = \frac{\ln(x_i + \varepsilon)}{\ln b}, \quad b \in \{e, 10\}.
```

Columns with fewer than two finite values are left as `NaN`.

### 10.2 Reference Modes

`normalize_features` computes normalization parameters within groups:

| Mode | Reference set |
|------|--------------|
| `all` | All rows (default) |
| `condition` | Within each condition |
| `run` | Within each run (useful for scanner drift correction) |

For a grouping variable $g_i \in \mathcal{G}$:

```math
z_i^{(g)} =
\frac{x_i - \mu_{g_i}}{\max(\sigma_{g_i},\, \varepsilon)}, \qquad
\mu_g = \mathrm{mean}\{x_j : g_j = g\}, \quad
\sigma_g = \mathrm{sd}\{x_j : g_j = g\}.
```

If the requested grouping column is absent from the DataFrame, normalization falls back to
using all rows (no grouping). Callers should ensure the grouping column exists when using
`condition` or `run` reference if group-wise normalization is intended.

### 10.3 Train/Test-Separated Normalization

- `normalize_train_test(train_df, test_df, method)` — normalizes both DataFrames using
  parameters estimated on `train_df` only.
- `FeatureNormalizer(method)` — a stateful normalizer: `.fit(train_df)` stores per-column
  parameters; `.transform(df)` applies them to any DataFrame.

Both utilities exclude non-feature metadata columns (`condition`, `epoch`, `subject`, `trial`,
`run`, `run_id`, …) from transformation.

---

## 11. Cross-Validation Hygiene

**Module:** `cv_hygiene.py`

Several pipeline components can introduce data leakage if not properly constrained to
training trials. The following table summarizes the safeguards:

| Component | Leakage risk | Safeguard |
|-----------|-------------|-----------|
| IAF estimation | Test-trial spectra influence band definitions | `compute_iaf_for_fold` uses `train_mask` only |
| Evoked subtraction | Condition averages include test trials | `ValueError` raised unless `train_mask` is provided |
| ITPC | Trial-average phase involves test trials | `fold_global` mode restricts to training trials |
| PAC surrogates | Surrogate distribution estimated on all trials | Surrogates computed from training trials only |
| Microstate templates | Clustering sees test-trial scalp maps | K-means restricted to training trials |
| Connectivity (condition-wise) | Condition average includes test trials | Training-only aggregation enforced |

**Analysis modes:**

- `trial_ml_safe` — all cross-trial computations that could leak test information require a
  valid `train_mask`. Any violation raises an error.
- `group_stats` — aggregations over all available trials; appropriate for group-level
  descriptive statistics only.

Any configuration that would silently mix training and test information is rejected.

---

## 12. Result Containers

### 12.1 `FeatureExtractionResult` (TFR-Based)

Flat dataclass in `results.py`. One DataFrame per feature domain:

```
pow_df, aper_df, erp_df, conn_df, dconn_df, phase_df,
pac_trials_df, pac_time_df, erds_df, spectral_df,
ratios_df, asymmetry_df, comp_df, bursts_df,
microstates_df, quality_df, source_df
```

ITPC and phase features are stored in `phase_df` / `phase_cols`. The dataclass also defines
`itpc_trial_df` / `itpc_trial_cols`, but these are not populated by `extract_all_features`.

Also stores associated column lists and baseline/TFR metadata.

### 12.2 `ExtractionResult` (Precomputed-Based)

Stores feature groups as:

```python
features[group_name] = FeatureSet(df, cols, name)
```

Key methods:

| Method | Returns |
|--------|---------|
| `get_combined_df(include_condition)` | Single design matrix from all groups |
| `get_feature_group_df(group, include_condition=True)` | DataFrame for one group |
| `get_qc_summary()` | Aggregated QC across groups |

### 12.3 Feature Manifests

`generate_manifest` and `save_features_organized` (in `eeg_pipeline.domain.features.naming`)
produce:

- A machine-readable manifest: feature names, domains, bands, scopes, QC flags.
- A standardized directory structure per subject and task.
- Optional condition labels alongside features.

This ensures downstream analyses (fMRI integration, ML models) can reconstruct exactly
which features were extracted and under what preprocessing choices.

---

## 13. Dependencies

| Library | Role |
|---------|------|
| **MNE-Python** | TFR, PSD, forward/inverse solutions, CSD |
| **mne-connectivity** | Phase- and amplitude-based connectivity |
| **NumPy / SciPy** | Numerical computing, signal processing, optimization |
| **scikit-learn** | Clustering (microstates, dynamic connectivity states) |
| **NetworkX** | Graph-theoretic connectivity metrics |
| **NiBabel** | NIfTI reading for fMRI-constrained source localization |
| **joblib** | Parallelization for band precomputation, aperiodic fitting, ITPC, asymmetry |
