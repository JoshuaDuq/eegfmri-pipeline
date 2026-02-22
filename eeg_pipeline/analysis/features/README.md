# Feature Computation Pipeline

Efficient, trial-level feature extraction for EEG analysis, optimized for pain research paradigms. Each trial (epoch) produces one row of features across all categories.

## Naming Convention

All features follow a structured schema:

```
{domain}_{segment}_{band}_{scope}_{statistic}
```

| Field       | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `domain`    | Feature category (e.g., `power`, `erp`, `itpc`)                   |
| `segment`   | Time window name (e.g., `baseline`, `active`, `pain`)             |
| `band`      | Frequency band (e.g., `alpha`, `theta`) or `broadband`            |
| `scope`     | Spatial aggregation: `ch`, `roi`, `global`, or `chpair`           |
| `statistic` | Computed metric (e.g., `logratio`, `peak_freq`, `mean`)           |

## Spatial Aggregation Modes

Features are computed at three spatial granularities (configurable via `feature_engineering.spatial_modes`):

- **`channels`** — Per-channel values (no aggregation)
- **`roi`** — Mean across channels within each region of interest
- **`global`** — Mean across all EEG channels

---

## Feature Categories

### 1. Power

**Module:** `spectral.py` → `extract_power_features`

Computes band power from a Morlet wavelet time-frequency representation (TFR).

**Method:**

1. Compute TFR using Morlet wavelets (via MNE `tfr_morlet`), yielding power at each (epoch, channel, frequency, time) point.
2. For each frequency band, select frequency bins within `[fmin, fmax]` and compute a **frequency-weighted mean** across those bins (weights = `np.gradient(freqs)` to correct for non-uniform frequency spacing).
3. Average the weighted power across the time points within the target segment.
4. **Baseline normalization** — log-ratio relative to baseline segment power:
   ```
   feature = log10(power_active / power_baseline)
   ```
   Both numerator and denominator are floored to `ε = 1e-20` (symmetric epsilon strategy) to prevent division by zero and numerical instability.
5. If the TFR was already baseline-corrected (e.g., `logratio` or `percent` mode applied during TFR computation), the raw baselined values are emitted directly.
6. For the baseline segment itself, raw mean power is emitted (not self-normalized).

**Outputs per band × segment × scope:**
- `logratio` (or `log10raw` if no baseline is available and `require_baseline=false`)
- `db` — decibel-scaled version (`logratio × 10`); emitted alongside `logratio` by default (`emit_db=true`)

**Line noise handling:** Frequency bins within `±width` Hz of line noise fundamentals and harmonics are excluded before band averaging.

---

### 2. Spectral

**Module:** `spectral.py` → `extract_spectral_features`

Computes spectral descriptor features from the PSD of each trial.

**Method:**

1. Compute PSD per trial using either **multitaper** (`psd_array_multitaper`, preferred for short segments) or **Welch** (`psd_array_welch` with 50% overlap).
2. Optionally exclude line noise frequencies and harmonics from the PSD.
3. For each frequency band, extract:

| Feature            | Computation                                                                                                  |
|--------------------|--------------------------------------------------------------------------------------------------------------|
| `peak_freq`        | Frequency of maximum power after subtracting a linear aperiodic fit in log-log space (aperiodic-adjusted argmax). |
| `peak_power`       | PSD value at the detected peak frequency.                                                                    |
| `peak_ratio`       | Ratio of peak power to the aperiodic background at the peak frequency.                                       |
| `peak_residual`    | Residual (observed − aperiodic fit) at the peak frequency in log-log space.                                  |
| `center_freq`      | Spectral center of gravity: `Σ(f × PSD(f) × Δf) / Σ(PSD(f) × Δf)` within the band.                        |
| `bandwidth`        | Spectral bandwidth: `sqrt(Σ((f − center)² × PSD(f) × Δf) / Σ(PSD(f) × Δf))`.                              |
| `entropy`          | Normalized spectral entropy: `−Σ(p × ln(p)) / ln(N)`, where `p = PSD(f)×Δf / Σ(PSD×Δf)`. Values near 1 indicate uniform (flat) spectra; near 0 indicate peaked spectra. |

4. Broadband features (across all frequencies):

| Feature            | Computation                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `edge_freq_95`     | Frequency below which 95% of total spectral power is contained.            |

**Segment duration validation:** Segments shorter than `min_segment_sec` (default 2.0 s) or with fewer than `min_cycles_at_fmin` cycles at the band's lowest frequency are skipped.

---

### 3. Aperiodic

**Module:** `aperiodic.py` → `extract_aperiodic_features`

Extracts the aperiodic (1/f) component of the power spectrum, separating oscillatory peaks from the broadband background. Reflects excitation/inhibition (E/I) balance.

**Method:**

1. Compute PSD per trial (multitaper or Welch).
2. Transform to log-log space: `log10(frequency)` vs `log10(PSD)`.
3. Fit an aperiodic model using **iterative residual-based peak rejection**:
   - Fit initial linear model on all valid frequency bins.
   - Compute residuals `(log_PSD − fit)`.
   - Reject bins with positive residuals exceeding `peak_rejection_z × MAD` (oscillatory peaks).
   - Refit on remaining bins. Repeat up to 3 iterations until convergence.
4. Two model types:
   - **Fixed:** `log10(P) = offset + slope × log10(f)` — linear fit via `np.polyfit`.
   - **Knee:** `log10(P) = offset − log10(knee + f^exponent)` — nonlinear fit via `scipy.optimize.curve_fit`.
5. Segment-level validity gates:
   - Minimum segment duration (`min_segment_sec`)
   - Minimum fit points (`min_fit_points`)
   - Maximum allowed frequency-bin spacing (`max_freq_resolution_hz`)

**Outputs per segment × scope:**

| Feature              | Description                                                                                          |
|----------------------|------------------------------------------------------------------------------------------------------|
| `slope`              | Log-log PSD slope (typically negative; steeper = more 1/f dominance).                               |
| `offset`             | Broadband power level (y-intercept in log-log space).                                               |
| `exponent`           | 1/f exponent (`−slope`); emitted for both fixed and knee models.                                    |
| `knee`               | Knee frequency (knee model only; frequency where spectrum transitions).                             |
| `r2`                 | Goodness-of-fit R² of the aperiodic model.                                                          |
| `rms`                | Root-mean-square residual of the aperiodic fit.                                                     |
| `peakfreq`           | Alpha peak frequency from the aperiodic-adjusted residual spectrum (center-of-gravity).             |
| `tbr`                | Theta/beta ratio computed on the aperiodic-corrected residual power.                                |
| `tbr_raw`            | Conventional theta/beta ratio computed on raw PSD power.                                            |
| `{band}_powcorr`     | Per-band aperiodic-corrected power: integrated residual power above the aperiodic fit.              |
| `{band}_center_freq` | Center frequency (Hz) of the strongest oscillatory peak in-band from the residual spectrum.         |
| `{band}_bandwidth`   | Full-width at half-maximum (Hz) of the strongest in-band oscillatory residual peak.                 |
| `{band}_peak_height` | Peak amplitude above the aperiodic fit (log10 power residual) for the strongest in-band peak.      |

**Peak characterization:** Rejected residual peaks are summarized with center frequency, FWHM bandwidth, and height (FOOOF-like parameterization).

---

### 4. ERP (Evoked Response Potential)

**Module:** `erp.py` → `extract_erp_features`

Time-domain features for pain-related evoked potentials (ERP/LEP), computed per trial.

**Method:**

1. Optionally low-pass filter epochs (default 30 Hz) for cleaner peak detection.
2. Apply baseline correction by subtracting the mean of the baseline window.
3. For each user-defined ERP component window (e.g., N2: 150–250 ms, P2: 250–500 ms):
   - **Peak polarity** is inferred from the component name (`N` → negative, `P` → positive) or configured explicitly.
   - Optionally apply Savitzky-Golay smoothing before peak detection.

**Outputs per component × scope:**

| Feature        | Computation                                                                    |
|----------------|--------------------------------------------------------------------------------|
| `mean`         | Mean amplitude within the component window.                                    |
| `peak_{mode}`  | Peak amplitude (negative, positive, or absolute depending on component).       |
| `latency_{mode}` | Latency (in seconds) of the detected peak. Uses `scipy.signal.find_peaks` with optional prominence threshold. |
| `auc`          | Area under the curve (`np.trapz`) within the component window.                 |

4. **Peak-to-peak features** for matched N/P component pairs (e.g., N2-P2):

| Feature        | Computation                                        |
|----------------|-----------------------------------------------------|
| `ptp`          | Peak-to-peak amplitude: `|P_peak − N_peak|`.       |
| `latency_diff` | Latency difference between the P and N peaks.       |

---

### 5. ERDS (Event-Related Desynchronization/Synchronization)

**Module:** `precomputed/erds.py` → `extract_erds_from_precomputed`

Quantifies event-related power changes relative to baseline using band-limited amplitude envelopes from precomputed (bandpass + Hilbert) data.

**Method:**

1. For each band and channel, compute baseline reference power as the mean of the squared Hilbert envelope during the baseline window.
2. Compute active-segment mean power.
3. ERDS percentage:
   ```
   ERDS% = ((active_power − baseline_power) / baseline_power) × 100
   ```
   Optionally use log-ratio: `log10(active / baseline)`.
4. Baseline power is floored to `ε` to prevent division by zero. Channels with baseline power below threshold are flagged.

**Pain-specific markers** (when enabled):
- **Contralateral somatosensory ERD:** Computed over somatosensory channels contralateral to the stimulated side (auto-detected from metadata or configured).
- **ERD onset latency:** First sustained crossing below `onset_threshold_sigma × baseline_noise` (with minimum duration gate).
- **ERS rebound magnitude/latency:** Peak positive rebound after the ERD trough, with configurable minimum latency and threshold.

**Outputs per band × segment × scope:**
- `percent_mean` — Mean ERDS percentage across channels (or `log_ratio_mean` when `use_log_ratio=true`)
- Per-channel: `peak_latency`, `onset_latency`, `erd_magnitude`, `erd_duration`, `ers_magnitude`, `ers_duration`
- Pain markers (ROI scope, contralateral somatosensory): `peak_latency`, `erd_magnitude`, `onset_latency`, `rebound_magnitude`, `rebound_latency`, `ers_magnitude`

---

### 6. Ratios

**Module:** `precomputed/extras.py` → `extract_band_ratios_from_precomputed`

Computes band power ratios from PSD-integrated power (not Hilbert envelope power), which is the scientifically valid approach for cross-band comparisons.

**Method:**

1. Compute PSD per trial per segment (multitaper or Welch, bandwidth-normalized: µV²/Hz).
2. Integrate PSD within each band's frequency range.
3. For each configured ratio pair `(numerator_band, denominator_band)`:
   ```
   power_ratio = PSD_integrated(num) / PSD_integrated(den)
   log_ratio   = ln(PSD_integrated(num) + ε) − ln(PSD_integrated(den) + ε)
   ```
4. ROI and global ratios are computed from the mean PSD-integrated power across the relevant channels (not from per-channel ratios).

**Segment validation:** Segments must contain at least `min_cycles_at_fmin` cycles at the lowest frequency of either band in the pair.

**Outputs per pair × segment × scope:**
- `power_ratio` — Linear ratio
- `log_ratio` — Log-transformed ratio (optional, enabled by default)

---

### 7. Asymmetry

**Module:** `precomputed/extras.py` → `extract_asymmetry_from_precomputed`

Hemispheric power asymmetry indices for configured left-right channel pairs.

**Method:**

1. Compute PSD-integrated band power per channel per segment (same PSD pipeline as ratios).
2. For each configured channel pair `(Left, Right)` and each frequency band:
   ```
   index   = (P_right − P_left) / (P_right + P_left)
   logdiff = ln(P_right) − ln(P_left)
   ```
3. **Activation convention** (optional): For alpha-band asymmetry, emit `logdiff_activation = −logdiff`, following the convention that lower alpha power indicates greater cortical activation (Davidson, 1992).

**Default channel pairs:** `(F3, F4)`, `(F7, F8)`, `(C3, C4)`, `(P3, P4)`, `(O1, O2)`.

**Outputs per pair × band × segment (scope: `chpair`):**
- `index` — Normalized asymmetry index `∈ [−1, 1]`
- `logdiff` — Log-difference (primary metric for frontal alpha asymmetry)
- `logdiff_activation` — Activation-convention sign flip; disabled by default (`emit_activation_convention=false`)

---

### 8. Connectivity

**Module:** `connectivity.py` → `extract_connectivity_features`

Functional connectivity between channel pairs using phase-based and amplitude-based measures.

**Measures:**

| Measure   | Method                                                                                         |
|-----------|------------------------------------------------------------------------------------------------|
| **wPLI**  | Weighted Phase Lag Index — `mne_connectivity.spectral_connectivity_time` with `method='wpli'`. Robust to volume conduction (zero-lag signals contribute zero weight). |
| **PLI**   | Phase Lag Index — Fraction of time points with consistent phase lead/lag.                      |
| **imCoh** | Imaginary part of Coherency — Sensitive only to non-zero-lag interactions.                     |
| **PLV**   | Phase Locking Value — Magnitude of mean phase difference across time.                          |
| **AEC**   | Amplitude Envelope Correlation — orthogonalized amplitude envelope correlation to remove zero-lag leakage. Outputs `aec` (raw r, default) and/or `aec_z` (Fisher-z transformed), controlled by `aec_output` config (`["r"]` by default). |

**Computation:**
1. For each segment and band, extract data and compute connectivity matrices using `spectral_connectivity_time` (CWT Morlet mode) or `envelope_correlation`.
2. Granularity modes: `trial` (per-epoch), `condition` (per-condition group), or `subject` (all epochs).
3. For `condition`/`subject` granularity with phase measures, the pipeline uses across-epochs estimation (scientifically valid; avoids averaging per-epoch phase estimates). In CV mode (`train_mask` present), leakage guards keep phase estimation within-epoch unless explicitly overridden.

**Graph metrics** (optional, when `enable_graph_metrics=true`):

| Metric              | Computation                                                         |
|---------------------|---------------------------------------------------------------------|
| `{method}_clust`    | Weighted clustering coefficient (NetworkX).                         |
| `{method}_geff`     | Global efficiency on thresholded adjacency matrix.                  |
| `{method}_smallworld` | Small-world index σ = (C/C_rand) / (L/L_rand) from random graphs. |

**Dynamic connectivity** (optional, when `dynamic_enabled=true`):
- Sliding-window connectivity with configurable window length and step.
- Per-window summary statistics (stat suffix `{method}sw*`):
  - `{method}swmean`, `{method}swstd` — mean and std of edge weights across windows
  - `{method}swac{lag}` — autocorrelation at specified lag
  - `{method}swtopostab` — adjacent-window topographic stability (global)
- K-means state clustering (when `dynamic_state_enabled=true`): `{method}swswitch` (state switch rate), `{method}swdwellsec` (mean dwell time), `{method}swstateent` (state entropy)

**Volume conduction warning:** A warning is emitted if phase-based measures are used without CSD/Laplacian spatial transform.

---

### 9. Directed Connectivity

**Module:** `connectivity.py` → `extract_directed_connectivity_features`

Directed (causal) connectivity between channel pairs.

**Measures:**

| Measure | Method                                                                                  |
|---------|-----------------------------------------------------------------------------------------|
| **PSI** | Phase Slope Index — Frequency-resolved directional coupling via `mne_connectivity`.     |
| **DTF** | Directed Transfer Function — Multivariate autoregressive model-based directed coupling. |
| **PDC** | Partial Directed Coherence — Normalized DTF accounting for indirect pathways.           |

For DTF/PDC, MVAR order is automatically reduced when segment length/channel count is too small for stable estimation (`min_samples_per_mvar_parameter` safeguard).

**Defaults:** PSI is enabled by default; DTF and PDC are disabled by default (`enable_psi=true`, `enable_dtf=false`, `enable_pdc=false`).

**Outputs per method × band × segment (domain prefix: `dconn`):**
- Per channel pair: `{method}_fwd`, `{method}_bwd`
- Global: `{method}_fwd_mean`, `{method}_bwd_mean`, `{method}_asymmetry` (mean forward − backward)

---

### 10. ITPC (Inter-Trial Phase Coherence)

**Module:** `phase.py` → `extract_phase_features`

Measures the consistency of oscillatory phase across trials at each time-frequency point.

**Method:**

1. Obtain complex-valued TFR (Morlet wavelets, preserving phase).
2. Normalize each complex value to a unit vector: `z / (|z| + ε)`.
3. Compute ITPC using one of four methods:

| Method         | Formula / Description                                                                                     |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `fold_global`  | `|mean(unit_vectors[train_mask])|` — ITPC from training trials only. **Default; CV-safe.**                |
| `global`       | `|mean(unit_vectors)|` across all trials. **Not CV-safe** (leaks test information).                       |
| `loo`          | Leave-one-out: for each training trial, ITPC is computed from all other training trials. Requires `train_mask` and explicit opt-in. |
| `condition`    | Per-condition ITPC: computed within each condition group using training trials only.                       |

4. Average ITPC across frequencies within each band and across time within each segment.
5. Optional baseline correction: subtract baseline-segment ITPC from active-segment ITPC.
6. For `global`/`fold_global` methods, the single ITPC map is broadcast to all trials.

**Outputs per band × segment × scope:**
- `val` — ITPC value `∈ [0, 1]` (1 = perfect phase consistency)

---

### 11. PAC (Phase-Amplitude Coupling)

**Module:** `phase.py` → `extract_pac_from_precomputed`

Cross-frequency coupling between a low-frequency phase signal and a high-frequency amplitude signal.

**Method:**

1. From the complex TFR, extract:
   - **Phase signal:** Unit vectors at the phase-band frequencies (e.g., theta 4–8 Hz).
   - **Amplitude signal:** Absolute values at the amplitude-band frequencies (e.g., gamma 30–80 Hz).
2. Compute **Mean Vector Length (MVL):**
   ```
   MVL = |Σ(amplitude × exp(i × phase))| / Σ(amplitude)
   ```
   (Normalized by total amplitude to prevent bias from amplitude differences.)
3. **Surrogate-based z-scoring (optional):** If `n_surrogates > 0`, generate null PAC samples using `surrogate_method`:
   - `trial_shuffle` (default): cross-epoch amplitude shuffling with within-epoch circular shift
   - `circular_shift`: within-epoch circular shift only
   - In `analysis_mode="trial_ml_safe"`, `trial_shuffle` is restricted to training trials (or falls back to `circular_shift` if no valid training pool is available).
   Compute z-score:
   ```
   z = (MVL_observed − mean(MVL_surrogates)) / std(MVL_surrogates)
   ```
4. **Harmonic overlap rejection:** Band pairs where the amplitude band overlaps with harmonics of the phase band are skipped by default (configurable).
5. **Segment validity gates:** PAC is skipped when segment duration is below `max(min_segment_sec, min_cycles_at_fmin / fmin_phase)` for each phase band.

**Outputs per phase-amplitude band pair × segment × scope:**
- `val` — Mean Vector Length (raw PAC strength)
- `z` — Surrogate-corrected z-score (only when `n_surrogates > 0`)

---

### 12. Source Localization

**Module:** `source_localization.py` → `extract_source_localization_features`

Source-space ROI features using inverse solutions applied to sensor-space EEG.

**Method:**

1. **Forward model:** Construct using `mne.make_forward_solution` with fsaverage template (or subject-specific MRI). Supports surface (`oct6`) and volume source spaces.
2. **Inverse methods:**
   - **LCMV beamformer** — Adaptive spatial filter maximizing source power while suppressing interference.
   - **eLORETA** — Exact Low Resolution Electromagnetic Tomography; minimum-norm inverse with standardization.
3. **ROI extraction:** Source time courses are averaged within anatomical ROIs (Desikan-Killiany atlas or custom parcellation).
4. **fMRI-constrained source localization** (optional):
   - Threshold an fMRI statistical map (z-map or FDR-corrected) to define active clusters.
   - Convert suprathreshold voxels to volume source-space coordinates.
   - Construct a volume forward model restricted to fMRI-defined ROIs.
   - Apply inverse solution only to fMRI-constrained source space.

**Outputs per ROI × band × segment:**
- `src_{segment}_{method}_{band}_{roi}_power` — Band power within each ROI.
- `src_{segment}_{method}_{band}_{roi}_envelope` — Mean amplitude envelope within each ROI.
- `src_{segment}_{method}_{band}_global_power` — Global mean band power across all ROIs.

---

### 13. Complexity

**Module:** `complexity.py` → `extract_complexity_from_precomputed`

Nonlinear signal complexity metrics computed per trial, per channel, per band.

**Signal basis** (configurable): `filtered` (bandpass time series, default) or `envelope` (Hilbert amplitude envelope).

**Metrics:**

| Metric   | Computation                                                                                                  |
|----------|--------------------------------------------------------------------------------------------------------------|
| **LZC**  | Lempel-Ziv Complexity — Binarize signal at median, count distinct patterns. Higher = more complex.           |
| **PE**   | Permutation Entropy — Frequency of ordinal patterns of length `order` (default 3) with delay `delay` (default 1). Normalized to `[0, 1]`. |
| **SampEn** | Sample Entropy — Negative log of conditional probability that sequences of length `m+1` match within tolerance `r × std` given they match at length `m`. Default: `m=2`, `r=0.2`. |
| **MSE**  | Multiscale Entropy — Sample entropy computed at multiple coarse-graining scales (default 1–20). Each scale `s` averages consecutive non-overlapping windows of `s` samples before computing SampEn. |

**Preprocessing:** Optional z-score standardization per channel per trial (default enabled). Minimum segment duration and sample count gates prevent unstable estimates.

**Outputs per band × segment × scope (domain prefix: `comp`):**
- `lzc`, `pe`, `sampen` — Single-scale metrics
- `mse01` … `mse20` — Multiscale entropy at each scale (zero-padded two-digit index, e.g. `mse01`, `mse10`, `mse20`)

---

### 14. Bursts

**Module:** `bursts.py` → `extract_burst_features`

Detects oscillatory bursts using band-limited amplitude envelopes from precomputed data.

**Method:**

1. Compute amplitude envelope for each band (Hilbert transform of bandpass-filtered signal).
2. Determine burst threshold using one of three methods:

| Method       | Threshold                                                          |
|--------------|--------------------------------------------------------------------|
| `percentile` | `np.percentile(baseline_envelope, q)` (default `q=95`)            |
| `zscore`     | `mean + z × std` of baseline envelope (default `z=2.0`)           |
| `mad`        | `median + z × 1.4826 × MAD` of baseline envelope                 |

3. Threshold reference: `trial` (per-epoch), `subject` (across all epochs), or `condition` (per-condition group).
4. Identify contiguous intervals where envelope exceeds threshold.
5. Apply minimum duration gate: `max(min_duration_ms, min_cycles / band_center_freq)`.

**Outputs per band × segment × scope:**

| Feature         | Description                                                    |
|-----------------|----------------------------------------------------------------|
| `count`         | Number of detected bursts.                                     |
| `rate`          | Burst rate (bursts per second).                                |
| `duration_mean` | Mean burst duration (seconds).                                 |
| `amp_mean`      | Mean peak amplitude within bursts.                             |
| `fraction`      | Fraction of time above threshold.                              |

---

### 15. Microstates

**Module:** `microstates.py` → `extract_microstate_features`

EEG microstate dynamics using GFP-peak topographic clustering.

**Method:**

1. Compute **Global Field Power (GFP):** `std(channels)` at each time point (after demeaning).
2. Detect GFP peaks using `scipy.signal.find_peaks` with configurable minimum distance and prominence.
3. Extract topographic maps at GFP peaks; normalize each map (zero-mean, unit-norm, sign-standardized by largest component).
4. **Template fitting:**
   - **Fixed templates** (recommended): Use pre-defined canonical templates (A/B/C/D style labels when provided).
   - **K-means clustering** (fallback): Fit `n_states` (default 4) cluster centers on GFP-peak topographies using `sklearn.KMeans` with 20 initializations. Output classes use neutral labels (`state1..stateN`) because fitted cluster identities are not guaranteed to be comparable across subjects.
5. **State assignment (default):** Assign labels at GFP peaks, then backfit sample-wise labels by midpoint segmentation between neighboring labeled peaks (`assign_from_gfp_peaks=true`).
   - Optional legacy mode: direct sample-wise max-similarity assignment (`assign_from_gfp_peaks=false`).
6. **Minimum-duration smoothing:** Runs shorter than `min_duration_ms` are reassigned using neighboring-run context (longer-neighbor preference; tie split) to avoid directional bias.
7. Compute per-trial metrics:

| Feature                  | Computation                                                     |
|--------------------------|-----------------------------------------------------------------|
| `coverage_{label}`       | Fraction of time points assigned to each microstate class.      |
| `duration_ms_{label}`    | Mean duration (ms) of contiguous runs of each class.            |
| `occurrence_hz_{label}`  | Occurrence rate (Hz) of each class.                             |
| `trans_{i}_to_{j}_prob`  | Adjacent-sample transition probability from class `i` to class `j` (diagonal entries encode persistence; `NaN` when class `i` has no outgoing transitions). |

**CV/leakage behavior:** In `trial_ml_safe` mode with `train_mask`, template fitting is restricted to training trials.
**Statistical note:** Subject-fitted templates induce cross-trial dependence (non-i.i.d. trial rows); fixed templates do not.

---

### 16. Quality

**Module:** `quality.py` → `extract_quality_features`

Trial-level signal quality metrics for QC and artifact flagging.

**Metrics per segment (baseline and active):**

| Metric     | Computation                                                                                    |
|------------|------------------------------------------------------------------------------------------------|
| `variance` | `np.var(data, axis=time)` — Per-channel temporal variance.                                     |
| `ptp`      | `np.ptp(data, axis=time)` — Peak-to-peak amplitude.                                           |
| `finite`   | `mean(isfinite(data))` — Fraction of non-NaN/non-Inf samples (missing data indicator).         |
| `snr`      | `10 × log10(signal_density / noise_density)` — Bandwidth-normalized PSD power density ratio: signal band (default 1–30 Hz) vs noise band (default 40–80 Hz). |
| `muscle`   | `Σ PSD(muscle_band) / Σ PSD(total)` — High-frequency power fraction (default muscle band: 30–80 Hz). Elevated values indicate muscle artifact contamination. |

**PSD computation:** Welch (default) or multitaper, with line noise exclusion (fundamentals + harmonics).

---

## Precomputed Data Pipeline

Before feature extraction, a shared intermediate representation is computed once and reused across feature categories:

1. **Bandpass filtering** — FIR filter for each frequency band.
2. **Hilbert transform** — Analytic signal yielding instantaneous amplitude envelope, phase, and power.
3. **PSD computation** — Multitaper or Welch PSD for spectral features.
4. **TFR computation** — Morlet wavelet TFR (real-valued for power; complex-valued for ITPC/PAC).
5. **Time windows** — Baseline and active segment masks derived from configuration.

This avoids redundant computation across feature categories that share the same intermediate data.

## Cross-Validation Safety

Features that aggregate across trials (ITPC, connectivity, bursts, microstates) can create data leakage in trial-level ML pipelines. Two analysis modes control this:

| Mode              | Behavior                                                                              |
|-------------------|---------------------------------------------------------------------------------------|
| `trial_ml_safe`   | Cross-trial features require `train_mask`; test trials are excluded from aggregation.  |
| `group_stats`     | Cross-trial features use all trials (appropriate for group-level statistical analysis).|

Additional safeguards:
- Connectivity `condition`/`subject` granularity avoids per-epoch phase averaging by using across-epochs phase estimation outside CV mode.
- Subject-fitted microstates are flagged as non-i.i.d. in feature provenance metadata.
- Source-space `wpli`/`plv` connectivity outputs are marked as broadcast/non-i.i.d.; in `trial_ml_safe`, cross-epoch estimates use training trials when `train_mask` is available.

## Dependencies

- **MNE-Python** — TFR, PSD, forward models, inverse solutions
- **mne-connectivity** — wPLI, PLI, imCoh, PLV, AEC (`spectral_connectivity_time`, `envelope_correlation`); PSI, DTF, PDC use custom CSD/MVAR implementations
- **NumPy / SciPy** — Numerical computation, signal processing, optimization
- **scikit-learn** — K-means clustering (microstates, dynamic connectivity states)
- **NetworkX** — Graph metrics (clustering, efficiency, small-world)
- **NiBabel** — NIfTI loading for fMRI-constrained source localization
- **joblib** — Parallel computation for aperiodic fitting, ITPC, asymmetry
