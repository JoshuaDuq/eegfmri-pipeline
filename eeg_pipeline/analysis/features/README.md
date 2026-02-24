## Feature Extraction Pipeline

This document describes the **trial-level EEG feature extraction pipeline** implemented in `eeg_pipeline.analysis.features`.  
Each trial (epoch) produces one row in the feature matrix. The design targets pain research paradigms but is applicable to any event‑related EEG study with well‑defined time windows.

The README is organized like a methods section:

- **Overview and notation**
- **Configuration, time windows, frequency bands (including IAF)**
- **Precomputed intermediates and spatial transforms**
- **Per‑category feature definitions with formulas**
- **Change scores and normalization**
- **Cross‑validation hygiene and analysis modes**
- **Result containers and naming**

All formulas are given in LaTeX; random‑effects modeling and inferential statistics are intentionally left to downstream analysis.

---

### Global Notation

Let:

- $e \in \{1,\dots, N_{\text{trials}}\}$ index epochs (trials),
- $c \in \{1,\dots, N_{\text{ch}}\}$ index channels,
- $r$ index regions of interest (ROIs),
- $f$ denote frequency in Hz,  
- $t$ denote time in seconds,
- $x_{e,c}(t)$ denote the time‑domain EEG signal,
- $P_{e,c}(f,t)$ denote time–frequency power (Morlet TFR),
- $\mathrm{PSD}_{e,c}(f)$ denote power spectral density (µV$^2$/Hz),
- $\mathcal{H}(\cdot)$ denote the Hilbert transform.

We use:

- $B$ for a frequency band with range $[f_{\text{min}}, f_{\text{max}}]$,
- $T_{\text{seg}}$ for a time window segment (baseline, active, etc.),
- $\varepsilon$ as a small positive constant to avoid division by zero.

---

## 1. Configuration, Categories, and Naming

### 1.1 Feature Categories

Feature categories are resolved by `resolve_feature_categories` in `selection.py`:

- User‑requested categories (CLI or API) take precedence.
- Otherwise `feature_engineering.feature_categories` from the config is used.
- If neither is given, all entries in `FEATURE_CATEGORIES` are enabled.

Categories currently include:
$$
\{\text{power}, \text{spectral}, \text{aperiodic}, \text{erp}, \text{erds},
\text{ratios}, \text{asymmetry}, \text{connectivity}, \text{directedconnectivity},
\text{itpc}/\text{phase}, \text{pac}, \text{complexity}, \text{bursts},
\text{microstates}, \text{quality}, \text{sourcelocalization}\}.
$$
### 1.2 Column Naming Schema

Most features follow:

```text
{domain}_{segment}_{band}_{scope}_{statistic}
```

- **`domain`** – feature family (e.g. `power`, `erp`, `itpc`, `erds`, `conn`, `comp`).
- **`segment`** – time window label (e.g. `baseline`, `active`, task‑specific names).
- **`band`** – frequency band name (e.g. `alpha`, `theta`, `broadband`).
- **`scope`** – spatial aggregation:  
  - `ch`  = channel,  
  - `roi` = region of interest,  
  - `global` = all channels,  
  - `chpair` = channel pair.
- **`statistic`** – scalar statistic (e.g. `logratio`, `mean`, `peak_freq`).

**Exceptions**

- Source‑space features use explicit `src_*` names for compatibility with fMRI pipelines.
- Some connectivity and microstate metrics encode method/state names in the suffix.

### 1.3 Spatial Aggregation Modes

Spatial modes are controlled by `feature_engineering.spatial_modes` and enforced by
`filter_features_by_spatial_modes` in `api.py`:

- `channels` – keep per‑channel and channel‑pair features (`*_ch_*`, `*_chpair_*`),
- `roi` – keep ROI‑aggregated features (`*_roi_*` or names matching ROI labels),
- `global` – keep global features (`*_global_*` or columns ending with `_global`).

Non‑spatial features (e.g. global scalar summary metrics) are always retained.

---

## 2. Time Windows and Segments

Time windows are represented by `TimeWindows` (`baseline_range`, `active_range`, `masks`, `ranges`, `times`).

- Windows are built from configuration via `TimeWindowSpec` and `time_windows_from_spec`.
- In the TFR‑based pipeline, when epochs are cropped to $[t_{\text{min}}, t_{\text{max}}]$,
  new masks are constructed on the cropped time axis to avoid shape mismatches.
- Baseline and active ranges are kept in physical time, so baseline‑dependent computations
  can still reference the original windows.

For any window with mask $M_{\text{seg}}(t) \in \{0,1\}$, we define:
$$
\bar{x}_{e,c}^{(\text{seg})} = \frac{\sum_t M_{\text{seg}}(t)x_{e,c}(t)}{\sum_t M_{\text{seg}}(t)}.
$$
Segments with fewer than the required number of samples or cycles (per‑feature criteria)
are skipped for that feature.

---

## 3. Frequency Bands and Individual Alpha Frequency (IAF)

### 3.1 Base Frequency Bands

Base frequency bands are read from the config via `get_frequency_bands(config)` and typically include:
$$
\text{delta},\ \text{theta},\ \text{alpha},\ \text{beta},\ \text{gamma}.
$$
Each band $B$ has bounds $[f_{\text{min}}^B, f_{\text{max}}^B]$.

### 3.2 IAF‑Adjusted Bands (Global Precomputation)

The precomputation module (`precomputed/__init__.py`) can adapt bands to an **Individual Alpha Frequency** (IAF) when `feature_engineering.bands.use_iaf = true`.

Workflow:

1. Extract baseline data (if available) using the baseline mask; otherwise optionally fall back to the full segment if explicitly allowed.
2. Compute PSD in a broad range (default $\approx 1$–$40$ Hz) using `mne.time_frequency.psd_array_multitaper`.
3. Remove a 1/f trend:
$$
\log_{10} P_{\text{resid}}(f) = \log_{10} P(f) - \left(\beta_0 + \beta_1 \log_{10} f\right),
$$
estimated by robust or ordinary linear regression in log–log space.

4. Estimate IAF, $\hat{f}_\alpha$, either as the most prominent residual peak in a search range (e.g. 7–13 Hz) or, if no peak passes prominence criteria, as a residual‑power‑weighted average in that range.
5. Adjust bands (schematic):
$$
\begin{aligned}
\text{alpha} &= [\hat{f}_\alpha - w_\alpha,\ \hat{f}_\alpha + w_\alpha],\\
\text{theta} &= [\max(3,\ \hat{f}_\alpha - 6),\ \max(4,\ f^\text{alpha}_{\text{min}})],\\
\text{beta}  &= [\max(13,\ f^\text{alpha}_{\text{max}}),\ f^\text{beta}_{\text{max}}].
\end{aligned}
$$
All IAF‑dependent logic includes explicit **duration and cycle‑count checks** to avoid
ill‑posed estimates.

### 3.3 Fold‑Specific IAF in Cross‑Validation

`cv_hygiene.py` implements **fold‑specific IAF** for cross‑validation:

- Given a boolean `train_mask` on epochs, only training trials are used to estimate IAF.
- The same residual‑based method as above is used, but restricted to training data.
- This yields:

  - $\text{IAF}_{\text{fold}}$ in Hz,
  - an IAF‑adjusted frequency‑band dictionary $\mathcal{B}_{\text{fold}}$.

This guards against **data leakage** from test trials into band definitions.

---

## 4. Precomputed Intermediates and Spatial Transforms

### 4.1 Spatial Transform Selection

Many feature families (connectivity, PAC, ERDS, complexity, etc.) can be computed after a spatial transform such as surface Laplacian / CSD. `precompute_data` applies:

- A global setting `feature_engineering.spatial_transform` in $\{\text{"none"}, \text{"csd"}, \text{"laplacian"}\}$, or
- A per‑family override `feature_engineering.spatial_transform_per_family[family]`.

Spatial transforms are **never silently skipped**. If a requested transform fails
(e.g. due to montage problems), a `RuntimeError` is raised.

The standard transform is CSD via:
$$
x^\text{CSD}(t) = \text{compute\_current\_source\_density}(x(t);\ \lambda^2,\ \text{stiffness}),
$$
where $\lambda^2$ and stiffness are configured parameters with sensible defaults.

### 4.2 Precomputed Data Container

`PrecomputedData` (in `eeg_pipeline.types`) stores:

- Band‑filtered analytic signals (envelope, phase, band‑specific GFP),
- PSD estimates,
- Time windows (`TimeWindows`),
- GFP of broadband data,
- Frequency band definitions (possibly IAF‑adjusted),
- Quality‑control metadata.

The main precomputation function is:

```python
precompute_data(epochs, bands, config, logger, ...)
```

which:

1. Picks EEG channels and applies the configured spatial transform.
2. Builds windows on the actual time axis (rebuilding if necessary to match the data length).
3. Computes:

   - **Band data** $x_{e,c}^B(t)$ for requested bands $B$,
   - **Band GFP** per band: $\mathrm{GFP}^B_e(t)$,
   - **PSD data** (possibly baseline‑restricted),
   - Global GFP.

4. Stores QC measures (finite fractions, medians, frequency ranges, window sample counts).

Precomputed objects are keyed per **feature family** and can be shared between
families as long as their spatial transform requirements are compatible.

### 4.3 Evoked‑Subtracted Data (Induced Power)

For pain paradigms, induced power can be computed by subtracting condition‑wise evoked responses:

1. Let $x_{e,c}(t)$ be the trial waveform and $\bar{x}_{k,c}(t)$ the average over trials in condition $k$.
2. The **induced** signal is:
$$
x^{\text{induced}}_{e,c}(t) = x_{e,c}(t) - \bar{x}_{k(e),c}(t).
$$
3. This is implemented by `subtract_evoked` and is used:

   - In precomputation (`precompute_data`) when `feature_engineering.precomputed.subtract_evoked` or `feature_engineering.power.subtract_evoked` is enabled.
   - In TFR computation for power when `feature_engineering.power.subtract_evoked = true`.

In `trial_ml_safe` mode, a **train_mask is mandatory** for evoked subtraction; otherwise the code raises a `ValueError`, since cross‑trial averages would otherwise leak test information.

---

## 5. TFR‑Based Extraction vs Precomputed‑Based Extraction

### 5.1 TFR‑Based Pipeline (`extract_all_features`)

`extract_all_features(ctx)` (in `api.py`) is the main TFR‑based entry point:

1. Prepare working epochs, possibly cropped to $[t_{\text{min}}, t_{\text{max}}]$.
2. Rebuild `TimeWindows` on the cropped time axis.
3. Optionally precompute band data (`precompute_data`) when requested feature categories depend on it (e.g. ERDS, aperiodic, spectral, complexity, bursts, ratios, asymmetry, connectivity).
4. Validate epochs (montage, sampling rate, duration, etc.).
5. Compute TFR and baseline metrics once if any of the TFR‑dependent categories (`power`, `itpc`, `pac`) are requested.
6. Sequentially call the per‑category extractors using `_extract_feature_with_error_handling`, which enforces:

   - Strict trial‑count matching,
   - Timing and logging for each category,
   - Immediate surfacing of any mismatch or unexpected return type.

7. Apply spatial‑mode filtering to each feature DataFrame.
8. Compute **change scores** (Section 7) and append them as additional columns.

Outputs are stored in a `FeatureExtractionResult` dataclass with one DataFrame per domain (power, ERP, connectivity, etc.).

### 5.2 Precomputed‑Only Pipeline (`extract_precomputed_features`)

`extract_precomputed_features` provides a lighter‑weight API when precomputed intermediates are already available or when TFR‑based features are not needed:

- Takes `epochs`, `bands`, `config`, `logger`, and optional `events_df` and `precomputed`.
- If `precomputed` is `None`, it calls `precompute_data`.
- Extracts a subset of feature **groups** (e.g. `["erds", "spectral", "aperiodic", "connectivity", ...]`).
- Stores results in an `ExtractionResult` object with a `features` dict and optional `condition` labels.

This is the preferred path for large‑scale precomputation and for use in other modalities (e.g. shared PAC / connectivity features across models).

---

## 6. Per‑Category Feature Definitions

Below, we summarize each feature family with the principal formulas. Details such as QC thresholds and configuration keys are documented inline in the code and reflected in this description.

### 6.1 Power (Oscillatory Power)

**Module:** `spectral.py` → `extract_power_features`

Starting from TFR power $P_{e,c}(f,t)$:

1. **Segment average per frequency**:
$$
\bar{P}_{e,c}^{B,\text{seg}}(f) =
\frac{\sum_t M_{\text{seg}}(t)P_{e,c}(f,t)}{\sum_t M_{\text{seg}}(t)}.
$$
2. **Frequency‑weighted band power** over band $B$ with bin widths $\Delta f$:
$$
P_{e,c}^{B,\text{seg}} =
\frac{\sum_{f \in B} \bar{P}_{e,c}^{B,\text{seg}}(f)\Delta f}
     {\sum_{f \in B} \Delta f}.
$$
3. **Baseline‑normalized log‑ratio** (when baseline is available and TFR is not already baselined):
$$
\text{logratio}_{e,c}^{B} =
\log_{10}\left(
  \frac{\max(P_{e,c}^{B,\text{active}}, \varepsilon)}
       {\max(P_{e,c}^{B,\text{baseline}}, \varepsilon)}
\right),
\quad \varepsilon = 10^{-20}.
$$
4. **dB scaling**:
$$
\text{dB}_{e,c}^{B} = 10 \cdot \text{logratio}_{e,c}^{B}.
$$
If the TFR was already baselined (e.g. `logratio` or `percent` mode during TFR computation), the baselined power values are used directly, and baseline self‑normalization is avoided.

**Outputs (per band × segment × scope)**

- `power_{segment}_{band}_{scope}_logratio`,
- optional `..._db` if `emit_db = true`,
- optional raw log‑power `log10raw` when no baseline is available and `require_baseline = false`.

Line‑noise harmonics inside a configurable exclusion window around multiples of the mains frequency are removed prior to averaging.

---

### 6.2 Spectral Descriptors

**Module:** `spectral.py` → `extract_spectral_features`

For each trial and channel, a PSD is computed either by multitaper or Welch. Within a band $B$:

1. **Center frequency (spectral CoG)**:
$$
f_{\text{cog}} =
\frac{\sum_{f \in B} f\mathrm{PSD}(f)\Delta f}
     {\sum_{f \in B} \mathrm{PSD}(f)\Delta f}.
$$
2. **Bandwidth (power‑weighted standard deviation)**:
$$
\sigma_B =
\sqrt{
  \frac{\sum_{f \in B} (f - f_{\text{cog}})^2\mathrm{PSD}(f)\Delta f}
       {\sum_{f \in B} \mathrm{PSD}(f)\Delta f}
}.
$$
3. **Normalized spectral entropy**:

Let
$$
p(f) = \frac{\mathrm{PSD}(f)\Delta f}
            {\sum_{f \in B} \mathrm{PSD}(f)\Delta f}.
$$
Then
$$
H_B = -\frac{\sum_{f \in B} p(f)\ln p(f)}{\ln N_B},
$$
where $N_B$ is the number of frequency bins in $B$.

4. **Peak features** use a robust aperiodic fit (shared with the aperiodic module):

- Residuals: $r(f) = \log_{10}\mathrm{PSD}(f) - \widehat{\log_{10} \mathrm{PSD}}_{\text{aperiodic}}(f)$.
- Peak frequency: $f^* = \arg\max_{f \in B} r(f)$ (with CoG fallback for low‑prominence peaks).
- Peak power, peak ratio, and peak residual are derived at $f^*$.

5. **Broadband spectral edge**:

`edge_freq_95` is the smallest $f$ such that:
$$
\frac{\sum_{f' \le f} \mathrm{PSD}(f')\Delta f'}
     {\sum_{f'} \mathrm{PSD}(f')\Delta f'} \ge 0.95.
$$
Segments shorter than a configured `min_segment_sec` or with fewer than `min_cycles_at_fmin` cycles are skipped.

---

### 6.3 Aperiodic (1/f) Components

**Module:** `aperiodic.py` → `extract_aperiodic_features`

The aperiodic component is modeled in log–log space:

1. Compute PSD and transform:
$$
x(f) = \log_{10} f,\quad
y(f) = \log_{10} \mathrm{PSD}(f).
$$
2. Iteratively fit the aperiodic trend:

- Fit an initial model in $[f_{\text{min}}, f_{\text{max}}]$ (e.g. 2–40 Hz).
- Compute residuals $r(f) = y(f) - \hat{y}(f)$.
- Remove bins with large positive residuals (putative oscillatory peaks) exceeding a threshold in MAD units.
- Refit up to a fixed number of iterations.

3. Models:

- **Fixed‑slope (linear)**
$$
y(f) = \text{offset} + \text{slope} \cdot x(f).
$$
- **Knee model**
$$
y(f) = \text{offset} - \log_{10}\bigl(\text{knee} + f^{\text{exponent}}\bigr).
$$
4. Features per segment:

- `slope`, `offset`, `exponent`, `knee`,
- Goodness‑of‑fit metrics (`r2`, `rms`),
- Aperiodic‑corrected band powers:
$$
\text{powcorr}_B = \sum_{f \in B} 10^{r(f)}\Delta f,
$$
- Aperiodic‑corrected theta/beta ratio:
$$
\text{tbr} =
\frac{\mathrm{mean}_{f \in \theta} 10^{r(f)}}
     {\mathrm{mean}_{f \in \beta}  10^{r(f)}},
\quad
\text{tbr\_raw} =
\frac{\mathrm{mean}_{f \in \theta} \mathrm{PSD}(f)}
     {\mathrm{mean}_{f \in \beta}  \mathrm{PSD}(f)}.
$$
Residual peaks are further summarized by center frequency, bandwidth, and height (FOOOF‑like).

---

### 6.4 ERP (Evoked Potentials)

**Module:** `erp.py` → `extract_erp_features`

For each ERP component window $T_{\text{comp}}$ (e.g. N2, P2) and each channel or ROI:

1. Baseline‑corrected signal:
$$
\tilde{x}_{e,c}(t) = x_{e,c}(t) - \bar{x}_{e,c}^{(\text{baseline})}.
$$
2. Component mean:
$$
\text{mean}_{e,c}^{\text{comp}} =
\frac{1}{|T_{\text{comp}}|} \sum_{t \in T_{\text{comp}}} \tilde{x}_{e,c}(t).
$$
3. Peak amplitude and latency:

- Depending on polarity (`neg`, `pos`, or `abs`), find:
$$
t^* = \arg\max_{t \in T_{\text{comp}}} s(\tilde{x}_{e,c}(t)),
$$
where $s$ is identity, minus, or absolute value. Then:
$$
\text{peak}_{e,c}^{\text{comp}} = \tilde{x}_{e,c}(t^*),\quad
\text{latency}_{e,c}^{\text{comp}} = t^*.
$$
4. Area under the curve (AUC):
$$
\text{auc}_{e,c}^{\text{comp}} = \int_{t \in T_{\text{comp}}} \tilde{x}_{e,c}(t)dt,
$$
computed as a sum of trapezoids over contiguous valid segments.

Paired components (e.g. N2–P2) yield peak‑to‑peak amplitude and latency differences.

---

### 6.5 ERDS (Event‑Related Desynchronization/Synchronization)

**Module:** `precomputed/erds.py` → `extract_erds_from_precomputed`

Using precomputed band envelopes $|\mathcal{H}(x_{e,c}^B(t))|$:

1. Baseline and active power:
$$
P^{B,\text{baseline}}_{e,c} = \mathrm{mean}_{t \in T_{\text{baseline}}}
   |\mathcal{H}(x_{e,c}^B(t))|^2,\quad
P^{B,\text{active}}_{e,c}   = \mathrm{mean}_{t \in T_{\text{active}}}
   |\mathcal{H}(x_{e,c}^B(t))|^2.
$$
2. ERDS percentage:
$$
\text{ERDS\%}_{e,c}^B =
100 \cdot
\frac{P^{B,\text{active}}_{e,c} - P^{B,\text{baseline}}_{e,c}}
     {P^{B,\text{baseline}}_{e,c}}.
$$
3. dB form:
$$
\text{ERDS}_{\text{dB}} =
10 \log_{10}
\left(
  \frac{P^{B,\text{active}}_{e,c}}
       {P^{B,\text{baseline}}_{e,c}}
\right).
$$
Very low baseline power is treated as invalid rather than artificially clamped.

Outputs include per‑channel ERDS, ROI/global aggregates, slopes, onset and peak latencies, and pain‑specific markers (contralateral somatosensory ERD, rebound magnitude, etc.).

---

### 6.6 Ratios (Band Power Ratios)

**Module:** `precomputed/extras.py` → `extract_band_ratios_from_precomputed`

From PSD‑integrated band powers $P^B_{e,c}$:
$$
P^B_{e,c} = \sum_{f \in B} \mathrm{PSD}_{e,c}(f)\Delta f.
$$
For a ratio pair $(B_{\text{num}}, B_{\text{den}})$:
$$
\text{power\_ratio}_e =
\frac{P^{B_{\text{num}}}_e}{P^{B_{\text{den}}}_e},\quad
\text{log\_ratio}_e =
\ln\bigl(P^{B_{\text{num}}}_e + \varepsilon\bigr)
 - \ln\bigl(P^{B_{\text{den}}}_e + \varepsilon\bigr).
$$
ROI/global ratios use **averaged PSDs** across channels rather than averages of per‑channel ratios.

---

### 6.7 Asymmetry (Hemispheric Indices)

**Module:** `precomputed/extras.py` → `extract_asymmetry_from_precomputed`

For a left–right pair $(L,R)$ and band $B$, with integrated powers $P^B_L, P^B_R$:
$$
\text{index} = \frac{P^B_R - P^B_L}{P^B_R + P^B_L},\quad
\text{logdiff} = \ln P^B_R - \ln P^B_L.
$$
Optionally, an **activation convention** alpha asymmetry:
$$
\text{logdiff\_activation} = -\text{logdiff},
$$
so that higher values correspond to greater cortical activation on the right side under the usual alpha‑suppression interpretation.

---

### 6.8 Connectivity (Undirected)

**Module:** `connectivity.py` → `extract_connectivity_features`

Connectivity is computed in the frequency domain (wPLI, PLI, imaginary coherence, PLV) and in the amplitude domain (AEC). Let $X_i(f,t)$ and $X_j(f,t)$ be complex Fourier coefficients or analytic signals for channels $i,j$.

1. Cross‑spectrum $S_{ij}(f)$ and auto‑spectra $S_{ii}(f), S_{jj}(f)$.
2. wPLI:
$$
\text{wPLI}_{ij} =
\frac{\left|\mathbb{E}\bigl[\mathrm{Im}(X_i X_j^\ast)\bigr]\right|}
     {\mathbb{E}\bigl[\left|\mathrm{Im}(X_i X_j^\ast)\right|\bigr]}.
$$
3. PLI:
$$
\text{PLI}_{ij} =
\left|\mathbb{E}\bigl[\mathrm{sign}(\mathrm{Im}(X_i X_j^\ast))\bigr]\right|.
$$
4. Imaginary coherence:
$$
\text{imCoh}_{ij} =
\mathrm{Im}\left(
  \frac{S_{ij}}{\sqrt{S_{ii} S_{jj}}}
\right).
$$
5. PLV:
$$
\text{PLV}_{ij} =
\left|\mathbb{E}\bigl[e^{\mathrm{i}\Delta\varphi_{ij}}\bigr]\right|.
$$
6. AEC (envelope correlation):
$$
r_{ij} = \mathrm{corr}(A_i, A_j),
\quad
z_{ij} = \mathrm{atanh}(\mathrm{clip}(r_{ij}, -0.9999, 0.9999)).
$$
Connectivity can be:

- Trial‑wise,
- Condition‑wise (across trials of a condition),
- Subject‑wise (all trials).

Optional graph metrics (clustering coefficient, global efficiency, small‑world index) are computed from thresholded or weighted connectivity matrices.

Volume‑conduction guards emit warnings when phase‑based measures are computed without a CSD/Laplacian transform.

---

### 6.9 Directed Connectivity

**Module:** `connectivity.py` → `extract_directed_connectivity_features`

Directed connectivity is based on either spectral phase‑slope (PSI) or MVAR models (DTF, PDC):

1. **PSI**:

Let
$$
C_{ij}(f) = \frac{S_{ij}(f)}{\sqrt{S_{ii}(f) S_{jj}(f)}}.
$$
Then
$$
\text{PSI}_{ij} =
\mathrm{Im}\left(
  \sum_f C_{ij}^\ast(f)C_{ij}(f + \Delta f)
\right).
$$
2. **DTF**:

From an MVAR model with frequency‑domain transfer matrix $H(f)$,
$$
\text{DTF}_{i \leftarrow j}(f) =
\frac{|H_{ij}(f)|}
     {\sqrt{\sum_k |H_{ik}(f)|^2}}.
$$
3. **PDC**:

From the MVAR coefficient matrix $A(f)$,
$$
\text{PDC}_{i \leftarrow j}(f) =
\frac{|A_{ij}(f)|}
     {\sqrt{\sum_k |A_{kj}(f)|^2}}.
$$
Outputs include forward/backward directed influence and asymmetry summaries at the trial and global level. MVAR order is automatically reduced when the data do not support the requested model order.

---

### 6.10 ITPC and Phase Metrics

**Module:** `phase.py` → `extract_phase_features`, `extract_itpc_from_precomputed`

Given complex time–frequency data $Z_e(f,t)$, define unit vectors
$$
u_e(f,t) = \frac{Z_e(f,t)}{|Z_e(f,t)| + \varepsilon}.
$$
ITPC over a set of trials $\mathcal{T}$ is:
$$
\text{ITPC}(f,t) = \left|\frac{1}{|\mathcal{T}|}\sum_{e \in \mathcal{T}} u_e(f,t)\right|.
$$
The pipeline supports:

- Global (all trials),
- Fold‑global (training trials only; default, CV‑safe),
- Leave‑one‑out (LOO),
- Condition‑wise ITPC.

ITPC is then band‑ and segment‑averaged to yield scalar features in $[0,1]$.

---

### 6.11 PAC (Phase–Amplitude Coupling)

**Module:** `phase.py` → `extract_pac_from_precomputed`, `compute_pac_comodulograms`

For a phase band $B_\phi$ and amplitude band $B_A$:

1. Extract phase $\phi(t)$ and amplitude $A(t)$ from either:

- Hilbert‑based precomputed analytic signals, or
- Complex TFR data (TFR‑based PAC).

2. Mean vector length (MVL):
$$
u(t) = \mathbb{E}_{f_\phi \in B_\phi}\left[e^{i\phi(f_\phi,t)}\right],\quad
A(t) = \mathbb{E}_{f_A \in B_A}\left[\text{amp}(f_A,t)\right],
$$
$$
\text{MVL} = \frac{\left|\sum_t A(t)u(t)\right|}{\sum_t A(t)}.
$$
3. Optional surrogate‑based $z$‑scoring:
$$
z = \frac{\text{MVL}_{\text{obs}} - \mu_{\text{surr}}}{\sigma_{\text{surr}}},
$$
where $\mu_{\text{surr}}$ and $\sigma_{\text{surr}}$ are estimated from PAC values computed on surrogate data (trial‑shuffled and/or circularly shifted).

PAC can be exported as:

- Trial‑wise scalar features (per band pair × segment),
- Full comodulograms with phase and amplitude frequency axes,
- Optional time‑resolved PAC traces.

Harmonic overlap guards skip scientifically dubious band combinations.

---

### 6.12 Source Localization

**Module:** `source_localization.py` → `extract_source_localization_features`

Using a forward model and inverse solution (LCMV or eLORETA), source‑space signals
$x_v(t)$ are mapped to ROI time courses:
$$
x_{\text{ROI}}(t) = \frac{1}{|V_{\text{ROI}}|}\sum_{v \in V_{\text{ROI}}} x_v(t).
$$
Features include:

- Source‑space band power:
$$
\text{src\_power}^{B,\text{seg}}_{\text{ROI}} =
\frac{\sum_{f \in B} \mathrm{PSD}_{\text{ROI}}(f)\Delta f}
     {\sum_{f \in B} \Delta f},
$$
- Source‑space Hilbert envelopes averaged over segments,  
- Global averages across ROIs.

Optional fMRI constraints restrict the source space to suprathreshold fMRI clusters.

---

### 6.13 Complexity

**Module:** `complexity.py` → `extract_complexity_from_precomputed`

Complexity metrics (LZC, permutation entropy, sample entropy, multiscale entropy) are computed on either band‑filtered time series or envelopes.

Key definitions:

- **Lempel–Ziv Complexity (LZC)**:

  - Binary sequence $b_t = \mathbf{1}[x_t > \mathrm{median}(x)]$,
  - Complexity count $c$ from the LZ76 parsing algorithm,
  - Normalized:
$$
  \text{LZC} = \frac{c}{n / \log_2 n},
$$
  where $n$ is sequence length.

- **Permutation Entropy (PE)**:

  - For embedding dimension $m$ and delay $\tau$, compute ordinal patterns $\pi$ and their probabilities $p(\pi)$,
  - Then
$$
  \text{PE} =
  -\frac{\sum_{\pi} p(\pi)\log_2 p(\pi)}{\log_2(m!)}.
$$
- **Sample Entropy (SampEn)**:

  - With pattern length $m$, tolerance $r\sigma_x$,
  - Let $B$ be the count of template matches at length $m$,
  - Let $A$ be the count at length $m+1$,
  - Then
$$
  \text{SampEn}(m,r) = -\log \frac{A}{B}.
$$
- **Multiscale Entropy (MSE)**:

  - For each scale $s$, coarse‑grain $x(t)$ by averaging non‑overlapping blocks of length $s$,
  - Compute $\text{SampEn}$ of the coarse‑grained series to obtain $\text{MSE}(s)$.

Per‑band, per‑segment metrics are exported for each spatial scope.

---

### 6.14 Bursts (Transient Oscillations)

**Module:** `bursts.py` → `extract_burst_features`

From band envelopes $E_{e,c}^B(t) = |\mathcal{H}(x_{e,c}^B(t))|$:

1. Estimate a baseline envelope distribution (per trial, subject, or condition).
2. Define a threshold $\theta$ via one of:

- Percentile:
$$
  \theta = \mathrm{percentile}(E_{\text{baseline}}, q),
$$
- z‑score:
$$
  \theta = \mu + z\sigma,
$$
- MAD:
$$
  \theta = \mathrm{median}(E_{\text{baseline}})
          + z \cdot 1.4826 \cdot \mathrm{MAD}(E_{\text{baseline}}).
$$
3. Identify contiguous intervals where $E_{e,c}^B(t) > \theta$.
4. Enforce a minimum duration:
$$
T_{\text{burst}} \ge \max\left(T_{\text{min}}, \frac{\text{min\_cycles}}{f_{\text{center}}}\right).
$$
Per segment and band, the pipeline outputs burst counts, rates, mean durations, mean amplitudes, and occupancy fractions.

---

### 6.15 Microstates

**Module:** `microstates.py` → `extract_microstate_features`

Global Field Power (GFP) is:
$$
\text{GFP}_e(t) =
\sqrt{\frac{1}{N_{\text{ch}}}\sum_c \left(x_{e,c}(t) - \bar{x}_e(t)\right)^2}.
$$
GFP peaks are detected; scalp maps at these peaks are normalized (zero‑mean, unit‑norm, sign‑standardized) and clustered into microstates (fixed templates or K‑means). Labels are backfitted to the full time series with minimum‑duration smoothing.

Per state $k$, features include:

- Coverage:
$$
\text{coverage}_k =
\frac{\#\{t: s(t) = k\}}{N_t},
$$
- Mean duration (ms),
- Occurrence rate (Hz),
- Transition probabilities between states.

In `trial_ml_safe` mode, clustering is restricted to training trials.

---

### 6.16 Quality Metrics

**Module:** `quality.py` → `extract_quality_features`, `compute_trial_quality_metrics`

Per segment and channel:

- Variance:
$$
\mathrm{Var}_t(x) = \frac{1}{N_t - 1}\sum_t (x(t) - \bar{x})^2.
$$
- Peak‑to‑peak:
$$
\text{ptp} = \max_t x(t) - \min_t x(t).
$$
- Fraction of finite samples:
$$
\text{finite} = \frac{\#\{t: x(t)\ \text{finite}\}}{N_t}.
$$
- SNR (using band‑limited PSDs):
$$
\text{SNR}_{\text{dB}} = 10\log_{10}
\left(
  \frac{d_{\text{signal}}}{d_{\text{noise}}}
\right),
$$
where $d$ is a band‑limited integrated power density.

- Muscle artifact index: fraction of PSD power in a high‑frequency band (e.g. 30–80 Hz).

These metrics serve both as QC features and as filters for downstream processing.

---

## 7. Change Scores (Active – Baseline)

After primary features are computed, `_add_change_scores_to_results` adds **change‑score features** for multiple families (power, connectivity, aperiodic, phase, PAC, ERDS, spectral, ratios, asymmetry, microstates, etc.), when `feature_engineering.compute_change_scores = true`.

Conceptually, for a feature column $X$ with baseline and active variants:
$$
X_\Delta = X^\text{active} - X^\text{baseline}.
$$
More complicated window structures (e.g. multiple active segments) are handled by `compute_change_features`, which inspects column names and produces consistent `_delta`‑style columns without recomputing the original feature.

Change scores are stored alongside original features to avoid repeated derivation downstream.

---

## 8. Normalization

Normalization is intentionally separated from feature extraction and implemented in `normalization.py`.
All schemes avoid data leakage by allowing **train/test‑separated** estimation of parameters.

### 8.1 Per‑column normalization methods

Given a feature column $x \in \mathbb{R}^N$ and a reference vector $x^\text{ref}$ (typically the
training subset), the following methods are available:

- **Z‑score**:
$$
z_i = \frac{x_i - \mu^\text{ref}}{\max(\sigma^\text{ref}, \varepsilon)},\quad
\mu^\text{ref} = \mathrm{mean}(x^\text{ref}),\quad
\sigma^\text{ref} = \mathrm{sd}(x^\text{ref}).
$$
- **Robust (median/MAD)**:
$$
z^{\text{robust}}_i =
\frac{x_i - m^\text{ref}}
     {\max\bigl(\mathrm{MAD}(x^\text{ref})_{\text{normal}}, \varepsilon\bigr)},
\quad
m^\text{ref} = \mathrm{median}(x^\text{ref}),
$$
where $\mathrm{MAD}(\cdot)_{\text{normal}}$ is the normal‑consistent MAD (scaled by 1.4826).

- **Min–max** to range $[a,b]$:
$$
x'_i = a + (b-a)
\frac{x_i - x_{\text{min}}^\text{ref}}
     {\max\bigl(x_{\text{max}}^\text{ref} - x_{\text{min}}^\text{ref}, \varepsilon\bigr)},
$$
with $x_{\text{min}}^\text{ref} = \min(x^\text{ref})$ and
$x_{\text{max}}^\text{ref} = \max(x^\text{ref})$.

- **Rank‑based** (0–1 ranks on finite values):
$$
r_i =
\frac{\mathrm{rank}(x_i) - 1}{n_{\text{finite}} - 1},
$$
where ties are handled by a configurable ranking method (default: average rank) and
$n_{\text{finite}}$ is the number of finite entries.

- **Log**:
$$
\log_b(x_i) = \frac{\ln(x_i + \varepsilon)}{\ln b},
$$
with $b \in \{e, 10\}$ or any positive base and $\varepsilon > 0$ drawn from configuration
(default $10^{-12}$).

Columns with fewer than two finite values are left as `NaN` rather than artificially
normalized.

### 8.2 Condition‑wise and run‑wise references

`normalize_features` supports **reference modes** that compute normalization parameters
within groups:

- `reference="all"` – use all rows (default),
- `reference="condition"` – normalize **within each condition**,
- `reference="run"` – normalize **within each run** (useful for scanner or session drift).

For a grouping variable $g_i \in \mathcal{G}$ (e.g. condition or run), z‑score
normalization within groups is:
$$
z_i^{(g)} =
\frac{x_i - \mu_{g_i}}{\max(\sigma_{g_i}, \varepsilon)},\quad
\mu_{g} = \mathrm{mean}\{x_j : g_j = g\},\quad
\sigma_{g} = \mathrm{sd}\{x_j : g_j = g\}.
$$
Analogous formulas apply for robust and min–max normalization with group‑specific
medians/MADs or minima/maxima.

When a grouping column is requested but missing, the implementation falls back to
`reference="all"` rather than guessing group structure.

### 8.3 Train/test‑separated and fitted normalizers

To avoid cross‑validation leakage:

- `normalize_train_test(train_df, test_df, method)` normalizes both DataFrames
  using parameters estimated on `train_df` only. For each numeric feature column
$x$, `train_df` plays the role of $x^\text{ref}$.
- `FeatureNormalizer(method)` can be:

  - **fit** on a training DataFrame to store per‑column parameters
    (mean/std, median/MAD, or min/max), and
  - **applied** to new DataFrames via `.transform(df)` using those stored
    parameters.

Both utilities respect an exclusion list (by default including `condition`, `subject`,
`trial`, `run`, `run_id`, etc.), so that non‑feature metadata columns are never
transformed.

---

## 9. Cross‑Validation Hygiene and Analysis Modes

Several modules collaborate to prevent data leakage:

- **Analysis modes** (see also the earlier Cross‑Validation section):

  - `trial_ml_safe`: All cross‑trial computations that could leak test information require `train_mask` and are restricted to training trials (e.g. IAF estimation, PAC surrogates, microstate templates, certain connectivity modes).
  - `group_stats`: Aggregations may use all available trials (appropriate for group‑level descriptive statistics).

- **IAF in CV**: `cv_hygiene.compute_iaf_for_fold` computes fold‑specific IAF and band definitions from training trials only; if the `train_mask` is too small, IAF is skipped rather than guessed.

- **Evoked subtraction in CV**: Both precomputation and power TFR require a valid `train_mask` when `subtract_evoked = true` and `analysis_mode = "trial_ml_safe"`. Otherwise, a `ValueError` is raised.

- **ITPC / PAC / Connectivity**: Methods that aggregate across trials default to **fold‑safe options** when possible (e.g. `fold_global` for ITPC, training‑only surrogates for PAC).

Any configuration that would silently mix train and test information is rejected rather than “helpfully” altered.

---

## 10. Result Containers and Manifests

### 10.1 `FeatureExtractionResult` (TFR‑Based)

`FeatureExtractionResult` (in `results.py`) is a flat dataclass used by the TFR‑based pipeline. It stores:

- `pow_df`, `aper_df`, `erp_df`, `conn_df`, `dconn_df`, `phase_df`, `pac_trials_df`, `pac_time_df`, `erds_df`, `spectral_df`, `ratios_df`, `asymmetry_df`, `comp_df`, `bursts_df`, `microstates_df`, `quality_df`, `source_df`,
- plus associated column lists and baseline/TFR metadata.

### 10.2 `ExtractionResult` (Precomputed‑Based)

`ExtractionResult` collects feature groups in a dict:

- `features[group_name] = FeatureSet(df, cols, name)`,
- optional `condition` vector if events are provided,
- QC metadata per group.

It provides:

- `get_combined_df(include_condition=True)` – concatenates all groups into a single design matrix,
- `get_feature_group_df(group)` – returns a single group’s DataFrame,
- `get_qc_summary()` – summarized QC across groups.

### 10.3 Manifests and Organized Saving

Using `generate_manifest` and `save_features_organized` from `eeg_pipeline.domain.features.naming`, features can be saved with:

- A machine‑readable feature manifest (names, domains, bands, scopes, QC flags),
- A standardized directory structure per subject and task,
- Optional condition labels.

This ensures that downstream analyses (e.g. fMRI integration, ML models) can reconstruct exactly which features were used and under what preprocessing choices.

---

## 11. Dependencies

- **MNE‑Python** – TFR, PSD, forward/inverse solutions, CSD.
- **mne‑connectivity** – Phase‑based and amplitude‑based connectivity.
- **NumPy / SciPy** – Numerical computing, signal processing, optimization.
- **scikit‑learn** – Clustering (microstates, dynamic connectivity states).
- **NetworkX** – Graph‑theoretic connectivity metrics.
- **NiBabel** – NIfTI reading for fMRI‑constrained source localization.
- **joblib** – Parallelization for band precomputation, aperiodic fitting, ITPC, and asymmetry.

All computations are designed to be **transparent, scientifically interpretable, and reproducible**, with explicit guards against silent fallbacks and cross‑validation leakage.
