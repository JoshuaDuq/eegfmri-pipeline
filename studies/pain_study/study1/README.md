# Study 1 - EEG Prediction of Trial-Wise fMRI Pain-Signature Expression

For operational reruns, use the Study 1 run guide: [RUN_GUIDE.md](RUN_GUIDE.md).
The run guide contains the copy-paste Kingston command block, smoke-test command, required
outputs, and full-picture report tables. This README documents the scientific rationale,
estimands, preprocessing assumptions, and interpretation framework.

## 1. Problem Statement

Pain responses vary among individuals exposed to the same nociceptive stimulus (Coghill et al.,
2003; Kim et al., 2004). EEG can measure fast oscillatory dynamics during painful stimulation
(Kim and Davis, 2021; Mari et al., 2022), but scalp alpha, beta, and gamma power are spatially
ambiguous and can reflect nociception, aversion, movement, facial muscle activity, cardiac
contamination, scanner artifact, or expectancy (Allen et al., 1998, 2000;
Muthukumaraswamy, 2013).

Simultaneous fMRI provides spatially defined targets for EEG validation (Davis et al., 2020,
van der Miesen et al., 2019). The Neurologic Pain Signature (NPS, Wager et al., 2013) indexes
evoked nociceptive signature expression, whereas the Stimulus Intensity Independent Pain
Signature-1 (SIIPS1, Woo et al., 2017) targets pain-related variance less dependent on stimulus
intensity.

## 2. Objectives

The primary objective is to test whether plateau-window EEG spectral power across alpha, beta,
and gamma bands predicts trial-wise NPS expression during simultaneous EEG-fMRI
thermal stimulation beyond measured stimulus, acquisition, and physiological nuisance structure.
The primary estimand is subject-held-out incremental prediction. Secondary objectives apply the
same staged incremental framework to SIIPS1, with NPS included in the SIIPS1 nuisance branch.
Interpretation of any EEG result is conditional on a preregistered fMRI target-validity gate that
is evaluated before EEG model interpretation.

The prespecified primary gate is NPS prediction with ElasticNet using alpha+beta+gamma
individual-channel spectral power. The required frequency audit additionally evaluates alpha,
beta, gamma, and alpha+beta models as secondary spectral tests after the primary inference. Delta,
theta, delta+theta, and all-band models are retained as exploratory report outputs rather than
confirmatory frequency presets.

## 3. Hypotheses and Confirmatory Estimands

The primary hypothesis is that EEG features add out-of-sample predictive value for trial-wise fMRI
pain-signature expression beyond a nuisance-only model.

$$
\Delta R^2_{\text{LOSO}} = R^2_{\text{nuisance+EEG}} - R^2_{\text{nuisance-only}} > 0.
$$

The primary confirmatory estimand is the subject-weighted mean incremental raw-target prediction:

$$
\Delta R^2_{\text{LOSO}} = \frac{1}{S}\sum_{s=1}^{S} \Delta R_s^2.
$$

The primary gate supports the hypothesis when $\Delta R^2_{\text{LOSO}} > 0$ and the one-sided
upper-tail permutation p-value is ≤ 0.05. Effect magnitude and bootstrap confidence intervals are
reported with the primary p-value.

The secondary spectral hypothesis is that prediction of pain-signature expression may be
distributed across multiple physiologically plausible oscillatory regimes, including gamma-band
power. Gamma effects are interpreted cautiously because scalp gamma can reflect nociceptive
processing, salience or aversion, facial or cranial muscle activity, scanner residuals, or other
high-frequency artifacts. A gamma-containing model therefore supports a neural pain-signature
interpretation only when the result survives the prespecified nuisance design, Fp1/Fp2 exclusion,
artifact diagnostics, and sensitivity checks.

Secondary confirmatory analyses evaluate the 2 targets (NPS, SIIPS1) × 2 linear models
(ElasticNet, Ridge) × 5 frequency presets (alpha, beta, gamma, alpha+beta,
alpha+beta+gamma) grid on the individual-channel spectral-power matrix. Delta, theta,
delta+theta, and all-band models are reported as exploratory low-frequency and broad-band audits.
The single primary gate is not corrected across this secondary grid. Holm correction (Holm, 1979)
is applied to the secondary raw-target prediction family. ROI-level and global-average feature
matrices, subjective-rating residualization, Random Forest, deep regression, exploratory feature
families, and alternative designs are exploratory.

## 4. Study Design and Data Scope

### 4.1 Participants

The planned sample comprises 60 healthy adults, balanced by sex (30 women and 30 men). Eligible
participants are 18-50 years old, right-handed according to the adapted Edinburgh Handedness
Inventory (Oldfield, 1971; laterality quotient > +40), and have normal or corrected-to-normal
vision. Exclusion criteria include chronic or persistent pain, diagnosed neurological or
psychiatric conditions, pregnancy or breastfeeding, medication use that alters attention or
vigilance, MRI contraindications, and hair or hairstyle characteristics incompatible with
high-quality EEG recording.

Participants complete one approximately 180 min session at the CERVO neuroimaging unit, including
EEG preparation, task familiarization, structural MRI, resting-state fMRI, and the simultaneous
EEG-fMRI thermal pain task.

Self-report instruments characterize demographic, psychological, sleep, handedness, and pain-related
individual differences: sociodemographic and hormonal status, PHQ-9 (Kroenke et al., 2001),
GAD-7 (Spitzer et al., 2006), RU-SATED (Buysse, 2014), adapted Edinburgh Handedness Inventory
(Oldfield, 1971), Gender Role Expectation of Pain questionnaire (Robinson et al., 2001), and
Pain Catastrophizing Scale (Sullivan et al., 1995). Exploratory moderators are restricted to sex
assigned at recruitment, hormonal-status variables, PCS total score, GREP self-ratings, PHQ-9,
GAD-7, and RU-SATED total score. Moderator interactions with the out-of-sample EEG residual
prediction are tested in separate mixed models and Holm-corrected within outcome.

The primary confirmatory analysis is planned for at least 30 analyzable subjects after EEG, fMRI,
synchronization, and artifact-quality exclusions. The final report will include the full sample
flow and retained-trial counts.

### 4.2 Thermal Pain Protocol

Thermal stimulation is delivered to the inner forearm contralateral to the response hand using an
MRI-compatible QST.Lab T11 thermode with five independently controlled contact surfaces over 9 cm².
Temperatures are not individually calibrated to subjective pain intensity. Temperature, ratings,
and nuisance structure are modeled separately.

Before scanning, two practice trials verify task comprehension and maximum-temperature tolerability.
Participants unable to tolerate the maximum temperature do not proceed.

The MRI task comprises six runs of 11 trials, for a total of 66 thermal trials. Six temperatures
from 44.3 to 49.3 °C are presented 11 times each. Trial order is generated by constrained
randomization so that consecutive stimuli are not delivered on the same thermode surface,
auto-transitions between identical temperatures are excluded, and ordered transitions between
distinct temperatures are balanced across the session. The first trial of the first run delivers
49.3 °C as a protocol-fixed exposure to the highest planned intensity. If acquisition files use a
separate BIDS `run` label (Gorgolewski et al., 2016), that label is retained as acquisition
metadata; confirmatory trial-order and permutation rules use the explicit task-run identifier.
The target table records the original event-level trial order from
`trial_number` when available and from `trial_index` otherwise. One of these columns is required.
Censored trials retain their original labels and are not renumbered.

All quality-controlled thermal trials are retained for the primary fMRI-signature prediction
analysis. Temperature enters the Level 2 nuisance design, and binary pain reports and continuous
ratings are criterion and sensitivity variables. Because SIIPS1 was developed after removing
non-painful trials, SIIPS1 analyses include a painful-trials-only scope sensitivity whenever the
primary target table contains both painful and non-painful thermal trials.

The deterministic first high-temperature trial is modeled by trial onset, within-run trial
number, and task-run index. Sensitivity analyses repeat the primary incremental model after
excluding the first trial and, separately, after excluding the first run; direction and magnitude
changes are reported.

Trial-history sensitivity analyses augment the Level 2 nuisance design with previous-trial
temperature, signed temperature change, cumulative exposure count, and previous-trial within-scale
rating when available.

Each trial begins with a variable 15-20 s fixation interval while the thermode is held at 35.0 °C.
Thermal stimulation lasts 12.5 s, including a 3.0 s ramp-up, a 7.5 s plateau, and a 2.0 s return to
baseline. After a 4.5-8.5 s post-stimulus fixation interval, participants report whether the
stimulus was painful and then rate intensity on a visual analogue scale.
Non-painful trials use a heat-intensity scale from 0 to 99, whereas painful trials use a
pain-intensity scale from 100 to 200. Behavioral data are stored as the binary pain report, the raw
displayed rating, and a within-scale thermal/pain intensity score. The within-scale score retains
the 0 to 99 heat rating for non-painful trials and subtracts 100 from painful-trial ratings,
yielding a 0 to 100 pain-intensity score. Responses are collected with an MRI-compatible
five-button Pyka response device, and visual stimuli are projected to an MRI-compatible display
viewed through a head-coil mirror.

Behaviorally implausible responses are excluded before EEG or fMRI outcome inspection. A trial is
implausible when the raw displayed rating is outside the logged response scale or equals 0
("no sensation") at a stimulus temperature ≥ 47.3 °C. Participant-level exclusion requires more
than 10% behaviorally implausible trials among otherwise synchronized thermal trials.

### 4.3 Acquisition Summary

EEG is acquired with a 64-channel BrainCap MR system and BrainAmp MR Plus amplifier at 5,000 Hz
before scanner-artifact correction and downsampling. Passive Ag/AgCl electrodes are positioned
according to the extended 10-20 system (Jasper, 1958). Hardware recording filters are set to
0.1-100 Hz, and electrode impedances are maintained below 20 kΩ, with a target below 10 kΩ for
most channels. An ECG channel records cardiac activity for artifact correction and physiological
nuisance quantification.
EEG and fMRI timing are synchronized through BrainVision volume markers.

Functional MRI is acquired on a Siemens MAGNETOM Prisma 3 T system using a multiband T2*-weighted
echo-planar sequence (TR = 900 ms, TE = 20 ms, multiband factor 3, 3 × 3 × 3 mm voxels,
54 axial slices). A T1-weighted MP-RAGE anatomical image (1 × 1 × 1 mm voxels), a 10 min
resting-state fMRI acquisition, and paired field maps support anatomical registration,
resting-state characterization, and distortion correction.

### 4.4 Modeling Scope

EEG epochs span −7.0 to 15.0 s relative to stimulus onset. The active EEG prediction window is the
thermal plateau interval from 3.0 to 10.5 s post-onset, and fMRI trial targets are LSS plateau
estimates aligned to the same protocol-defined plateau phase.

The multimodal dataset links fMRI targets to clean EEG trials using unique subject, acquisition-run,
task-run, and trial-index identifiers. EEG trigger onsets and fMRI plateau onsets are compared
after applying the protocol offset from trigger to plateau start. Retained trials require residual
absolute mismatch ≤ 0.010 s.

## 5. EEG Preprocessing and Artifact Controls

### 5.1 Simultaneous EEG-fMRI Preprocessing

MRI-induced EEG artifacts are corrected in BrainVision Analyzer 2.3 (Brain Products GmbH).
Scanner-gradient artifacts are removed using continuous-mode sliding-window template subtraction
(Allen et al., 2000) aligned to `V 1` volume markers (TR = 900 ms, 0 ms offset). Templates are
estimated over 21 scanner-artifact intervals using the full interval for baseline correction,
with channel-specific bad-interval handling. This correction is applied to all 64 channels.
The data are then downsampled from 5,000 Hz to 1,000 Hz and low-pass filtered (100 Hz IIR,
24 dB/oct).

Cardioballistic artifacts are subsequently removed in a two-step procedure (Allen et al., 1998).
R-peaks are automatically detected from the ECG channel (45–80 bpm permitted range, 0.6 coherence
threshold). Pulse-artifact templates are computed over 21 cardiac intervals and applied before
export to MNE-Python. ICA is fit on preliminary 1.0 Hz high-pass-filtered epochs. ICA spatial
weights are then applied to the 0.1-100 Hz continuous analysis data. ICLabel
(Pion-Tonachini et al., 2019) is used for component classification.
Components are rejected when their predicted probability exceeds 0.8 for any non-brain category
other than "other."

Retained components undergo a prespecified artifact audit blind to NPS, SIIPS1, ratings, and model
outcomes. Components are rejected for volume-repetition spectral peak robust $z > 3$, cardiac
phase-locking above the 95th percentile of the within-subject circular-shift null, or a
high-frequency topographic artifact rule. The topographic rule requires 70-95 Hz component power
robust $z > 3$ plus either at least 50% absolute IC topographic mass on Fp1, Fp2, FT9, FT10,
TP9, and TP10, or within-subject Pearson $r \ge 0.50$ between the component 70-95 Hz plateau
envelope and the Fp1/Fp2 artifact proxy.

The preprocessing sequence is fixed in this order.

1. Create preliminary epochs on a 1.0 Hz high-pass-filtered copy.
2. Fit ICA on these filtered epochs.
3. Apply ICA spatial weights to the 0.1-100 Hz continuous analysis data.
4. Extract final analysis epochs from −7.0 to 15.0 s relative to stimulus onset.
5. Apply autoreject (Jas et al., 2017) in local mode with candidate interpolation counts
   {4, 8, 16} for trial rejection.

Subject-specific electrode positions are digitized with EasyCap M1 channel labels and co-registered
to individual MRI. These digitizations support source-level analyses. Subject-level unsupervised
EEG preprocessing is performed independently within each subject before cross-validation.

### 5.2 Epochs, Baselines, and Frequency Bands

A −0.2 to 0.0 s pre-stimulus voltage baseline removes DC offset before ERP and amplitude-based
analyses. For time-frequency decompositions, the primary log-ratio baseline is −5.0 to −0.01 s,
chosen to estimate plateau power relative to the oscillatory state preceding the trial rather than
relative to a neutral-brain interval. Because pre-stimulus oscillations may contribute to
subsequent pain perception, reference-window sensitivity analyses repeat feature extraction with
shorter pre-stimulus baselines of −2.0 to −0.01 s and −0.2 to −0.01 s. A separate
sensitivity uses unnormalized active-window log power with −5.0 to −0.01 s reference power
retained as a covariate.

Neural oscillations are operationalized as delta (1.0–3.9 Hz), theta (4.0–7.9 Hz), alpha
(8.0–12.9 Hz), beta (13.0–30.0 Hz), and scanner-clean gamma. Gamma is not treated as one
contiguous 30.1–80.0 Hz interval. The retained gamma intervals are 30.1–38.0 Hz
(`gamma_low_clean`), 43.0–56.0 Hz (`gamma_mid_clean`), and 67.0–77.0 Hz
(`gamma_high_clean`). The excluded intervals are 38.0–43.0 Hz, 56.0–67.0 Hz, and
77.0–85.0 Hz.

This definition is based on the scanner-harmonic benchmark run after BrainVision Analyzer
gradient- and pulse-artifact correction. In the production thermal-pain recordings with valid
headers, residual narrow-band peaks were subject-consistent in the 18–23, 38–43, 56–67, and
77–85 Hz windows. The final-clean QC sample comprises 77 runs from 13 numbered participants,
excluding the prespecified pilot `sub-0006`. Event-locked QC showed that the early stimulation
window did not materially increase
these peaks relative to pre-stimulus baseline, whereas late stimulation increased both broad gamma
and scanner-harmonic prominence. A broad 30.1–80.0 Hz gamma feature would therefore combine
physiological high-frequency activity with residual scanner-locked energy. The confirmatory gamma
estimand keeps gamma in the study while removing the empirically contaminated windows.

The standalone `scanner_harmonic_spectrum.svg` QC figure estimates 15–90 Hz Welch spectra with
`n_fft = n_per_seg = 8192` and 50% overlap. Within each run it takes the median linear PSD across
EEG channels before conversion to dB; it then takes the median across runs within participant and
subtracts each participant's across-frequency median as a constant display offset. The cohort
curve is the equally weighted participant median with a 95% interval from 10,000 paired
participant-bootstrap resamples. A second panel compares participant-median peak centres with the
18th, 37th, 55th, and 74th harmonics of the 0.9 s volume repetition time, using one Welch bin as
the frequency-agreement reference. This panel distinguishes scanner-locked peaks from arbitrary
narrow spectral features.

In the final 13-participant audit, the cohort medians of the participant-median peak centres are
20.020, 41.138, 61.096, and 82.214 Hz. Their offsets from the specified TR harmonics are +0.020,
+0.027, −0.015, and −0.008 Hz, respectively; every offset is smaller than the 0.061 Hz Welch-bin
width. Every run contains a qualifying peak in all four windows, and the minimum observed
run-level prominence across those windows is 7.54 dB.

The residual approximately 20 Hz peak lies inside the conventional 13–30 Hz beta band. Beta is
retained as a prespecified feature family, but beta-containing results are not described as
scanner-clean and require the scanner-spectrum and artifact-control evidence for interpretation.
Exploratory `beta_scanner_clean` and `alpha_beta_scanner_clean_gamma` specifications omit
18.0–23.0 Hz using 13.0–17.9 Hz and 23.1–30.0 Hz beta sub-bands. These quantify sensitivity to the
residual scanner peak without replacing or gating the prespecified conventional-beta analysis.

The primary gate still uses the alpha+beta+gamma preset for interpretability, but this preset is
implemented as alpha, beta, and the three scanner-clean gamma sub-bands. The required secondary
frequency audit includes alpha, beta, scanner-clean gamma, and alpha+beta presets. Delta, theta,
delta+theta, and all-band presets are exploratory report outputs; all-band likewise uses the
scanner-clean gamma sub-bands rather than broad gamma.

### 5.3 Fp1/Fp2 Frontal High-Frequency Artifact Proxy

Artifact control relies on a prespecified Fp1/Fp2 high-frequency proxy and the dedicated ECG
channel. Anterior spatial patterns are interpreted accordingly.

The Fp1/Fp2 proxy is computed from the gradient- and BCG-corrected continuous signal after
downsampling, band-pass filtering, and notch filtering, but before PyPREP interpolation, ICA,
autoreject, or epoch-level rejection. For each plateau trial, Fp1 and Fp2 are refiltered to
70–95 Hz with the existing 60 Hz notch excluded. Hilbert power is averaged over 3.0–10.5 s and
across Fp1 and Fp2. This trial-level artifact-power covariate is scaled within folds during nuisance
modeling.

The HRF-weighted Fp1/Fp2 nuisance regressor is the canonical-HRF convolution of this trial series
sampled at each trial's fMRI plateau regressor peak. The unweighted proxy supports categorical
artifact censoring and continuous artifact-effect reporting. Fp1/Fp2 high-frequency power and
channels are excluded from confirmatory predictive features.

Required artifact covariates must be computable for at least 90% of otherwise retained plateau
trials. A subject is excluded from nuisance-controlled primary analysis when Fp1, Fp2, ECG, or event
synchronization is absent or invalid. Missing physiological artifact metrics are not imputed.

Artifact censoring thresholds use framewise displacement and DVARS metrics (Power et al., 2012):
framewise displacement > 0.5 mm, Fp1/Fp2 high-frequency power robust $z > 3$, DVARS robust $z > 3$,
cardiac phase-locking above the 95th percentile of the within-subject circular-shift null, and
scanner-frequency residual peaks robust $z > 3$. Censoring analyses report direction, magnitude,
and inferential changes, with special attention to gamma-band effects.

## 6. fMRI Signature Target Construction

Trial-wise fMRI effects are estimated with Least-Squares Separate (LSS) models (Mumford et al.,
2012) restricted to thermal plateau trials. For each eligible target trial, one GLM includes a
target-trial plateau regressor with onset at plateau start and duration equal to the plateau hold.
Other eligible plateau trials in the same acquisition run are modeled with the prespecified pooled
`other_trials` nuisance regressor (`lss_other_regressors: all`). Ramp-up, ramp-down, fixation, and
response epochs are modeled as non-plateau nuisance events when timing is available.

Primary LSS models use a canonical SPM hemodynamic response function (Friston et al., 1998), cosine
drift model, and 0.008 Hz high-pass filter. BOLD is spatially smoothed with a 6 mm FWHM Gaussian
kernel before single-trial estimation, matching the spatial scale at which the NPS and SIIPS1
weights were developed (Wager et al., 2013; Woo et al., 2017). First-level nuisance regression
uses only the 24-parameter rigid-body motion expansion (Friston et al., 1996), consistent with
signature-development pipelines and avoiding WM/CSF/CompCor regression that can attenuate
subcortical signature support (PAG, thalamus, nucleus accumbens). fMRIPrep motion-outlier spike
regressors are included when present (Esteban et al., 2019). Trials with framewise displacement
> 0.5 mm or standardized DVARS robust $z > 3$ are ineligible as target trials, but their thermal-event
timing remains in the nuisance-event design when valid.
A target GLM is ineligible if the target regressor is absent, duplicate-labeled, nonestimable, or
pushes the design condition number above 3000.

HRF and timing sensitivity are assessed by repeating target construction with HRF temporal and
dispersion derivatives, with a finite-impulse-response model, and after shifting the EEG active
window by ± 2.0 s. Direction, magnitude, and inferential changes are reported relative to the
primary model.

NPS and SIIPS1 maps are registered to MNI152NLin2009cAsym space (Fonov et al., 2009, 2011).
Published weights were trained in SPM MNI152 space; resampling to fMRIPrep's asymmetric template
introduces minor expected misregistration (typically 1–2 mm) that is standard for CANlab signature
application on fMRIPrep data.
Signature assets are `NPS/weights_NSF_grouppred_cvpcr.nii.gz` and
`SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz`, resolved relative to the signature-map root.
SIIPS1 provenance is anchored to the CANlab Neuroimaging_Pattern_Masks repository. NPS provenance is
anchored to Wager et al. (2013) and the authorized CANlab NPS distribution or access record.
CanlabCore pattern-mask conventions define scoring (CANlab, n.d.). Signatures with incomplete
provenance or modified files are ineligible.

When image grids differ, LSS beta maps are resampled to the signature-weight grid using continuous
interpolation. Non-finite beta-map values outside the explicit analysis mask are treated as
background during this resampling step; non-finite values inside the mask remain invalid. Published
weight sign and scale are preserved. Weights are not normalized, re-estimated, rescaled,
thresholded, or sign-flipped using study data. For each signature, the scoring mask $V^{(k)}$ is
fixed a priori by intersecting the finite signature-weight grid with a standard
MNI152NLin2009cAsym brain mask (`study1.targets.signature_scoring_mask_path`) after
nearest-neighbor resampling to the scoring grid. This mask is configured up front and is therefore
required: the scored extent is defined independently of the analyzed sample, so it cannot drift as
subjects are added and a single truncated field of view cannot shrink the scored extent for the whole
cohort. Sample-derived scoring masks are not supported. The reference mask is materialized with
`studies/pain_study/scripts/build_apriori_scoring_mask.py`, which fetches the TemplateFlow brain mask
and verifies signature coverage (NPS 0.998, SIIPS1 0.957 of nonzero support retained). Voxel count
and scoring-mask extent are identical across retained subjects, runs, and trials for the same
signature.

A signature target is valid only when the scoring mask retains at least 90% of original nonzero
signature support, retains at least 90% of positive- and negative-weight support, and changes
positive or negative total absolute weight mass by no more than 10% after resampling.
Sensitivity analyses use the canonical signature grid.

LSS beta maps are not z-scored or trial-normalized before scoring. Signature expression is a
within-subject relative index: absolute dot-product values are not compared to published NPS/SIIPS1
classification thresholds or across signatures (different native voxel grids). Robustness repeats
target construction with unsmoothed BOLD and with 8 mm FWHM smoothing. Signature expression is:

$$
y_{s,i}^{(k)} = \sum_{v \in V^{(k)}} \beta_{s,i}(v) \, M_k(v),
$$

where $\beta_{s,i}(v)$ is the LSS-derived BOLD estimate for subject $s$, trial $i$, voxel $v$, and
$M_k(v)$ is the a priori weight map for target $k \in \{\text{NPS}, \text{SIIPS1}\}$.

For each target LSS GLM with design matrix $X$ and unit contrast $c_{\text{target}}$ selecting the
target regressor, design efficiency is
$1/(c_{\text{target}}^\top (X^\top X)^+ c_{\text{target}})$ on the fitted Nilearn design matrix
(Abraham et al., 2014). Subjects are excluded from confirmatory analyses if they
retain fewer than 25 plateau trials, have LSS design condition number above 3000, or have LSS design
efficiency below 0.1. Subjects with 15-24 retained plateau trials are summarized in feasibility or
exploratory analyses.
Reliability-informed sensitivity analyses use split-half reliability $r \ge 0.4$ and $\ge 30$
plateau trials.

Target reliability is computed separately for NPS and SIIPS1 from retained raw LSS signature
expression after acquisition and design-estimability exclusions and before EEG feature inspection.
A fixed random seed produces 1,000 split-half partitions stratified within subject and stimulus
temperature. Each subject-by-temperature cell contributes the mean expression from half A and
half B, and the reliability statistic is the median Spearman-Brown-corrected (Brown, 1910;
Spearman, 1910) Pearson correlation across valid split-half vectors. A split is valid when both
halves contain finite values for every included subject-by-temperature cell. Values below $r = 0.4$,
insufficient retained trials, or inability to generate valid stratified splits qualify biological
interpretation of the corresponding signature target.

### 6.1 Target QC Metrics

The report writes retained Study 1 target metrics before EEG feature inspection. For both
targets, the table includes retained-trial counts, subject counts, stimulus-temperature
correlations, pain-rating correlations, and split-half reliability values.

The report stage writes `reports/full_picture/target_qc_metrics.tsv`.

The report writes five standalone supplementary validity figures to
`reports/figures/supplementary/validity/`. The behavioral, NPS, and SIIPS1 dose-response SVGs show
retained participant-level temperature trajectories behind the equally weighted cohort mean and
its 95% paired participant-bootstrap confidence interval. The behavioral figure additionally
marks the protocol pain threshold at 100. Missing participant-by-temperature cells remain missing
and are not imputed.

The additional `nps_behavioral_validity.svg` and `siips1_behavioral_validity.svg` coefficient
plots test whether signature expression tracks reported pain beyond delivered temperature. Each
participant model jointly estimates standardized partial coefficients for the binary pain report
and the protocol-defined within-scale intensity score while adjusting categorical temperature.
The SIIPS1 model additionally adjusts NPS. Participants are equally weighted in the cohort mean,
and uncertainty is a 95% interval from 10,000 paired participant-bootstrap resamples. The
continuous construct estimand uses the within-scale 0–100 score, never the discontinuous 0–200
display code. Participant coefficients, explicit non-estimability status, and cohort summaries
are written to `reports/full_picture/behavior_signature_validity_by_subject.tsv` and
`reports/full_picture/behavior_signature_validity_summary.tsv`.

The standalone outcome-blind `cohort_power_spectral_density.svg` summarizes final-clean continuous
EEG from 1 to 90 Hz. Run spectra are combined within each participant by taking the median in
linear power before conversion to dB; the cohort curve is the median of participant spectra. The
shaded interval is a pointwise 95% percentile interval from paired participant-bootstrap
resampling. Participant trajectories remain visible, exact scanner-harmonic exclusion windows are
overlaid on the linear frequency axis, and a separate strip marks conventional and scanner-clean
frequency bands. The header reports participants, total runs, and the participant run-count
distribution.

The preprocessing-checkpoint PSD command writes separate `raw`, `processed`, and `mne` versions
of that figure and their run-, participant-, and cohort-level audits. Each header names the exact
stored checkpoint, source sampling frequency, and the common 16.384 s Welch duration with 50%
overlap. The three artifacts are descriptive QC views rather than a combined inferential contrast:
cohort or run availability may differ between checkpoints, so apparent stage differences must not
be attributed to preprocessing without a matched-run analysis.

The outcome-blind scanner-harmonic spectrum is generated separately because it reads all
continuous final-clean EEG runs and should not be recomputed whenever the model report is rebuilt.
It writes `scanner_harmonic_spectrum.svg`, `scanner_harmonic_spectrum_by_run.tsv`, and
`scanner_harmonic_spectrum_by_subject.tsv` under the same supplementary validity directory;
matching parquet audits preserve the same schemas.

The standalone `power_construct_validity.svg` tests whether the prespecified global EEG power
summary tracks delivered temperature and subjective intensity. For each band and trial, global
power is reconstructed as
`10 × log10(mean(active linear channel power) / mean(baseline linear channel power))`; linear
power is averaged before the logarithm. Panel a shows participant-centered temperature
trajectories for alpha, beta, and the three scanner-clean gamma intervals in five vertically
aligned rows with a shared temperature and dB scale. The equally weighted cohort trajectory has a
simultaneous 95% max-studentized participant-bootstrap band across all 30 prespecified
band-temperature cells. Panel b is an aligned forest plot of participant-level partial
correlations between power and the within-scale 0–100 intensity score after adjusting temperature,
run, thermode surface, within-run trial order, residual ECG coupling, and the Fp1/Fp2 high-frequency
artifact proxy. Diamonds show equal-weight means and bars show pointwise 95% percentile
participant-bootstrap intervals. The header reports cohort and retained-trial counts. Fp1/Fp2
inclusion is configured explicitly and defaults to true; the complementary channel sensitivity is
always retained in audit tables when enabled. Trials are never treated as independent inferential
units.

The standalone `band_power_epoch_evolution.svg` shows the time-resolved global power construct
underlying those windowed summaries. Five vertically aligned panels contain alpha, beta, and the
three scanner-clean gamma intervals from −5 to 14.5 s around stimulus onset. The final 0.5 s of the
epoch is omitted to limit right-edge Morlet convolution effects. For every retained trial,
Morlet power is frequency-weighted within band and averaged in linear units across EEG channels
after excluding Fp1/Fp2; it is then converted to dB relative to that trial's complete −5.0 to
−0.01 s baseline. Thin curves show participant retained-trial means, and the colored curve shows
the equally weighted participant mean with a pointwise 95% percentile interval obtained by
resampling participants as complete trajectories. The shared symmetric dB scale supports direct
comparison across bands, and the figure reports cohort size, retained-trial median and range, and
the common channel count. Display downsampling is applied only after baseline normalization. A
direct protocol bar marks baseline, ramp-up, plateau, ramp-down, and post-stimulus intervals; the
figure adds no timepoint-wise hypothesis tests.

The `band_time_frequency/` family provides a complementary frequency-resolved view. It writes one
SVG per participant and band plus one equally weighted cohort SVG per band, using each participant's
newest final-clean epoch file. Power is estimated with the FieldTrip-style `mtmconvol` convention:
one symmetric Hanning taper, frequency-dependent 7-cycle windows, 0.05 s steps, trial-wide DC
removal, and 1 Hz resolution. Each trial, channel, and frequency is converted to dB using only
complete Hanning windows inside the −5.0 to −0.01 s baseline. All EEG channels are retained through
normalization and then averaged. Participant maps show the delivered-temperature OLS slope in dB/°C,
adjusted for run, thermode surface, and within-run trial order; cohort maps give participant slopes
equal weight. Trials missing required model metadata are excluded explicitly and reconciled against
the clean epoch count in the source audit. Participant maps share a robust per-band family scale,
whereas each cohort map uses its own robust scale. TSV/parquet audits retain exact slopes, matched
event and epoch paths, source timestamps, modelled and excluded trial counts, design diagnostics,
sampling frequencies, and the full participant-specific EEG channel lists.

Two standalone sensor-space figures resolve the same power construct at individual electrodes.
Both use the five prespecified bands: alpha, beta, low gamma (`gamma_low_clean`), mid gamma
(`gamma_mid_clean`), and high gamma (`gamma_high_clean`). `sensor_power_topographies.svg` shows
participant-level temperature slopes in dB per degree Celsius in its first row and partial
correlations with within-scale subjective intensity in its second row. The temperature slopes are
fit to each participant's six temperature-cell means. The intensity association adjusts
categorical temperature, task run, and thermode surface plus within-run trial order, residual ECG
coupling, and the Fp1/Fp2 high-frequency artifact proxy. Its primary channel scope follows the
power-construct configuration; the complementary Fp1/Fp2 scope is retained as a descriptive
sensitivity audit.

`signature_power_topographies.svg` shows participant-level partial correlations between each
electrode's power and NPS or SIIPS1 expression. Power and the target are residualized using the
target-specific nuisance design before their correlation is calculated; the SIIPS1 design
additionally includes NPS. The signature maps use the feature-benchmark channel scope and
therefore exclude the configured Fp1/Fp2 channels. These maps are univariate nuisance-adjusted
associations, not predictive-model importance measures.

Each figure is an independent 10-map inferential family. Cohort maps display the unthresholded
participant mean (with Fisher-z averaging and back-transformation for correlations). Dark sensor
rings mark clusters surviving a two-sided cluster-forming threshold of `p = 0.01` and joint
max-cluster-mass family-wise correction at `alpha = 0.05` across both rows and all five bands.
Adjacency is defined by Delaunay triangulation of the exact top-view montage positions, and cluster
mass is the sum of absolute one-sample t statistics. Participant sign flips are synchronized
across all 10 maps: the procedure enumerates the sign-symmetric exact space when it contains at
most 10,000 patterns and otherwise draws 10,000 unique nonobserved patterns with the configured
deterministic seed. Participants, never trials or sensors, are the independent inferential units.
No 30-participant article-readiness gate or preliminary label is applied; non-estimable data and
inferential designs instead fail explicitly. The figures remain sensor-space results and do not
imply cortical source localization.

The standalone `fmri_construct_validity.svg` supplies the whole-brain spatial manipulation check
that cannot be represented adequately by the signature summaries. Panel a shows the unthresholded
participant-mean BOLD effect per 1 °C increase in delivered temperature. Panel b shows the
participant-mean BOLD effect per 10-point increase in within-scale subjective intensity after
categorically adjusting the six delivered temperatures. Both panels use separate multi-run
first-level GLMs with the configured HRF, drift, smoothing, motion24 confounds, explicit censoring,
and within-run trial-order nuisance terms. Dark outlines indicate two-sided voxelwise max-T FWE
`p < 0.05` from 10,000 deterministic participant-level sign-flipping permutations. Surface and
axial coordinates are fixed before inspecting the results; peak coordinates remain in the paired
audit table, and no post-hoc ROI analysis is drawn.

The report-driven `temporal_specificity.svg` is also generated separately. It shows held-out
participant $\Delta R^2$ values and cohort 95% bootstrap intervals for NPS and SIIPS1 across the
two pre-stimulus, ramp-up, and three plateau windows, all using the same raw-log-power temporal
control transform. The command requires the exact six-window protocol in the current
configuration and rejects legacy report rows. It writes subject-level and cohort-level TSV and
parquet audits beside the SVG. The primary full-plateau estimator is intentionally absent because
its log-ratio feature transform is not commensurate with the raw-log-power temporal controls.

The main-results `primary_prediction_estimation.svg` uses only the prespecified ElasticNet
alpha+beta+scanner-clean-gamma cell. For each target it pairs every participant's nuisance-only
and nuisance+EEG held-out $R^2$, then shows the participant $\Delta R^2$ distribution beside the
equally weighted mean and its 95% participant-bootstrap interval. Negative held-out $R^2$ values
are valid and remain visible. Trials and runs are never displayed as independent observations.
The matching subject-level and cohort-level TSV/parquet audits preserve every plotted value and
the source fold-table paths.

The complementary `spectral_specificity.svg` restricts the frequency comparison to the five
prespecified confirmatory families: alpha, beta, scanner-clean gamma, alpha+beta, and
alpha+beta+scanner-clean-gamma. It shows held-out participant $\Delta R^2$ values and the cohort
95% bootstrap interval separately for NPS and SIIPS1. Ridge and exploratory delta, theta,
delta+theta, and all-band models remain in report tables. The plot therefore describes spectral
specificity of incremental prediction rather than ranking every fitted model.

### 6.2 Reported QC Metrics and Sensitivity Analyses

The report records the following metrics without deriving automatic interpretation columns:

- **Primary prediction metrics.** $\Delta R^2_{\text{LOSO}}$ and the one-sided permutation
  p-value for the single prespecified primary cell. The report includes a Holm-adjusted
  primary-gate field for schema consistency, but this field equals the raw primary p-value because
  the primary family contains one test.
- **Target QC metrics.** NPS relations with stimulus temperature and reported pain, the SIIPS1
  residual relation, and split-half reliability from 1,000 stratified within-cell splits with
  Spearman-Brown correction.
- **Residual-target attainability and noise ceiling.** The fraction of target variance remaining
  after the primary nuisance design (`residual_target_variance_fraction`, one minus the in-sample
  nuisance $R^2$) is reported per target. Together with the condition-level split-half reliability
  above, it bounds the staged-residual prediction gain a priori: a low residual fraction or low
  reliability caps the achievable $\Delta R^2$, so a null EEG result under either condition is
  uninformative rather than evidence of absence. Single-trial signature expression has no repeated
  measurement, so its reliability is estimated at the reproducible condition level rather than per
  trial.
- **Temporal negative controls.** Reported per target and model from the pre-stimulus and
  wrong-lag control cells. Wrong-lag controls are constrained to the pre-plateau ramp-up interval
  so they do not include the held-temperature plateau or ramp-down.

Reported as sensitivity analyses:
artifact-censoring robustness, HRF and timing, baseline-window, FWHM smoothing, first-exposure,
within-subject centered $\Delta R^2$, and staged nuisance-adjusted incremental magnitude. The primary artifact
control is the prespecified nuisance design (Fp1/Fp2 exclusion plus HRF-weighted framewise
displacement, standardized DVARS, Fp1/Fp2 high-frequency artifact power, and residual ECG
coupling); censoring-based robustness is a confirmatory sensitivity run on the final cohort.

## 7. EEG Feature Construction

Spectral power features are extracted with Morlet wavelets (Cohen, 2014) using frequency-adaptive
cycle counts ($\text{cycles}=f/2.0$, bounded between 3.0 and 15.0) and decimation factor 4. The
primary precomputed benchmark uses baseline-normalized total power without evoked-response
subtraction: a condition-agnostic ERP template estimated across all trials would leak held-out
subjects, while a fold-level template would require CV-owned signal extraction and fold-specific
feature matrices. ERP-subtracted power is eligible only as a separate sensitivity analysis when
extraction is fold-owned and provenance records the training-fold template.

Spectral power is log-ratio baseline-corrected using the primary pre-stimulus reference window and
averaged within 3.0–10.5 s. Each retained trial contributes one power value per channel and band.
The primary feature family uses active-window individual-channel log-ratio columns only, evaluated
across the required frequency presets. Presets labelled gamma use the three scanner-clean gamma
sub-bands defined above, so the model can retain gamma information without admitting the residual
scanner-harmonic windows. ROI-level and global-average matrices are spatial-resolution
sensitivities. Reference-window sensitivity runs repeat the same extraction with −2.0 to −0.01 s
and −0.2 to −0.01 s baselines. A non-normalized sensitivity uses active-window raw log power and
includes the corresponding −5.0 to −0.01 s reference power as a covariate. All confirmatory
feature matrices exclude Fp1 and Fp2. Matrices retaining Fp1/Fp2 are exploratory
artifact-sensitivity analyses.

The ROI feature matrix uses fixed, non-overlapping scalp groups resolved from normalized extended
10-20 channel names before model fitting. Fp1 and Fp2 are excluded. ROI inclusion requires at
least two listed channels after preprocessing. ROI features are arithmetic means of
channel-level log-ratio power within each ROI and frequency preset.

Exploratory feature families include spectral peaks, aperiodic slope and offset (specparam fixed
model; Donoghue et al., 2020), event-related desynchronization/synchronization, band-power ratios,
hemispheric alpha asymmetry, nonlinear complexity, and oscillatory burst statistics. Each family is
reported and Holm-corrected across tested target-model-frequency cells.

## 8. Nuisance Structure and Residualization

The primary benchmark predicts raw NPS/SIIPS1 dot-product expression with staged nuisance control.
The nuisance branch residualizes targets against an intercept, stimulus temperature, task-run
index, trial onset time, within-run trial number, selected thermode surface, HRF-weighted
framewise displacement, HRF-weighted standardized DVARS,
HRF-weighted Fp1/Fp2 high-frequency artifact power, and residual ECG coupling from the dedicated
ECG channel. Stimulus temperature and selected thermode surface enter as categorical regressors. A
continuous nonlinear temperature basis is retained only as a sensitivity analysis. Binary pain
condition is excluded. For SIIPS1, NPS expression is additionally included in the nuisance branch so
the incremental EEG term estimates SIIPS1 prediction beyond NPS.

This staged estimator tests whether EEG predicts fMRI signature expression beyond prespecified
stimulus and acquisition structure.

Rating-residualized sensitivity analyses augment this nuisance design with the binary pain report
and within-scale thermal/pain intensity score, not the raw discontinuous 0 to 200 displayed rating.
These analyses are construct-attenuation sensitivities rather than part of the primary benchmark.

The primary nuisance design is fixed across LOSO folds. Before SVD fitting, rank is checked from the
centered and scaled training-fold nuisance matrix. The design is full rank only when every
non-intercept singular value satisfies $\sigma_j / \sigma_{\max} \ge 10^{-10}$. Cells with a
rank-deficient nuisance design in any outer training fold are ineligible for confirmatory
interpretation.

Within the staged estimator, nuisance coefficients are estimated exclusively on training subjects
using SVD-based least squares.

## 9. Predictive Modeling

### 9.1 Primary Incremental Model

The benchmark predicts raw NPS and SIIPS1 expression with a nuisance-only model and a combined
nuisance-plus-EEG model. The primary test is the prespecified NPS ElasticNet alpha+beta+gamma
individual-channel cell. The nuisance-only model uses unpenalized ordinary least squares with the
same rank-stable nuisance design in every fold. If the nuisance-only design becomes rank deficient,
the affected cell is ineligible for confirmatory interpretation.

The full required benchmark repeats the same nuisance-only versus nuisance-plus-EEG comparison for
alpha, beta, gamma, alpha+beta, and alpha+beta+gamma presets. Exploratory report outputs retain
delta, theta, delta+theta, and all-band presets to document low-frequency and broad-band
sensitivity without assigning them to the confirmatory primary family.

The nuisance-plus-EEG estimator is staged residual learning rather than a joint penalized
regression. The nuisance component is unpenalized ordinary least squares fit on the raw target
scale. Within each fold, the nuisance model, residual-target transformation, feature
residualization, standardization, imputation, and EEG model are learned from training subjects
only. The EEG model is fit to the Yeo-Johnson-transformed training residuals from the raw-scale
nuisance model. Held-out EEG residual predictions are inverse-transformed back to the raw
residual scale and then added to the held-out raw-scale nuisance prediction before scoring.

### 9.2 Feature-Based Models

A nested LOSO framework is used. The primary gate model is ElasticNet regression
(Zou and Hastie, 2005) on individual-channel alpha+beta+gamma spectral power. ElasticNet and Ridge
(Hoerl and Kennard, 1970) are repeated across the required frequency audit; Ridge supports
Haufe-style forward-pattern sensitivity analyses (Haufe et al., 2014). Random Forest is an
exploratory nonlinear model (Breiman, 2001).

Feature preprocessing is fold-contained: feature statistics, imputation medians, variance
thresholds, standardization means, and standard deviations are estimated from training subjects
only. Imputation is limited to isolated nonfatal missing feature values within otherwise valid
trials and features. Non-imputable missing data result in exclusions.

For imputation-eligible feature values, the missingness rate must be below 5% for each feature
within the outer training fold and below 10% for each retained subject after feature filtering.
Features exceeding the feature-level limit are removed within the training fold. Values that remain
imputation-eligible are imputed with the corresponding training-cohort median for that feature.
Exceeding the subject-level limit, or removing all features, makes the affected cell ineligible for
confirmatory interpretation. Retained features are standardized to zero mean and unit variance, with
constant features removed.

The Yeo-Johnson transformation (Yeo and Johnson, 2000) is applied in every confirmatory ElasticNet
and Ridge cell, only to the target component learned by the penalized EEG model. In the primary
incremental analysis this is $r_{\mathrm{train}}$. The nuisance-only prediction remains on the raw
target scale. In secondary residualized-target models, targets are residualized before
fold-contained transformation.
Predictions are inverse-transformed before primary metrics are reported. Analyses without target
transformation are sensitivities.

Hyperparameters are tuned with 5-fold inner GroupKFold cross-validation restricted to training
subjects. In the staged primary incremental model, inner selection uses subject-weighted $R^2$ in
the transformed residual-target space learned inside the outer training fold; final LOSO inference
uses raw-scale subject-weighted $\Delta R^2_{\text{LOSO}}$ after inverse transformation and
nuisance prediction add-back. ElasticNet uses $\rho \in \{0.2, 0.5, 0.8\}$ and the configured
$\alpha \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$ grid. ElasticNet uses 10,000 maximum iterations.
Ridge uses $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$. Random Forest uses 500 estimators with
max depths $\in \{5, 10, 20, \text{None}\}$, min samples split $\in \{2, 5, 10\}$, and min
samples leaf $\in \{1, 2, 4\}$.

Confirmatory inference is planned for at least 30 analyzable subjects. Retained-sample precision is
summarized with a preregistered simulation using the observed subject count, retained-trial counts,
task-run structure, temperature sequence, nuisance matrix, and target split-half reliability. The
report includes the expected confidence-interval width for the primary
$\Delta R^2_{\text{LOSO}}$ estimate.

### 9.3 Exploratory Deep Regression Model

The BandTemporalRegressor is exploratory. It uses band-limited Hilbert-power tensors cropped to
3.0–10.5 s, training-cohort standardization, band-specific spatial filtering, temporal
integration, dropout, AdamW optimization (Loshchilov and Hutter, 2019), and subject-held-out
validation.

## 10. Statistical Inference

Primary metrics are computed on the original target scale after inverse-transforming predictions
and, for the primary incremental analysis, adding back the held-out nuisance prediction. The
coefficient of determination uses the training-fold target mean as the zero-skill baseline.

$$
R_f^2 =
1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}
{\sum_{i \in f}(y_i - \bar{y}_{\mathrm{train},f})^2}.
$$

Subject-wise metrics use configured percentile bootstrap confidence intervals. Group-level
intervals resample subjects; within-subject intervals use task-run-level resampling or circular
run bootstrap.

Study 1 estimates out-of-sample prediction of fMRI pain-signature expression in held-out subjects
during the simultaneous EEG-fMRI thermal protocol. Within-person trial tracking is assessed with
the within-subject-centered $\Delta R^2_{\text{LOSO}}$ diagnostic.

Primary inference uses nonparametric permutation testing (Winkler et al., 2014) with 5,000 valid
permutations and a one-sided upper-tail p-value for positive $\Delta R^2_{\text{LOSO}}$. The
primary null repeats the full observed-analysis training procedure, including fold-level
preprocessing statistics, imputation, constant-feature filtering, target transformation, and inner
GroupKFold hyperparameter selection. Frozen-hyperparameter permutations are computational
sensitivities.

For nuisance-controlled cells, the primary null is a fold-contained reduced-model residual
permutation. Within each outer fold, the nuisance-only model is fit on the original training
subjects, nuisance predictions are held fixed for the training and held-out rows in that fold, and
the fold's nuisance residuals are circular-shifted within subject and task run. Permuted targets
are reconstructed as nuisance prediction plus shifted residual, and the full nuisance-plus-EEG
pipeline is then refit. This tests whether EEG residual information adds prediction beyond the
prespecified nuisance structure while preserving the fold-specific nuisance-target relationship.

Circular shifts use the six 11-trial task runs as exchangeability units, retaining censored runs
when they still support a valid circular shift. Within each run, trials are ordered by the
original trial-order label after censoring; censored trials are not imputed. A permutation-valid
run must retain at least 8 plateau trials and allow at least four distinct nonzero circular
shifts after excluding shifts shorter than 5 original trial positions. Runs not meeting these
rules are excluded before confirmatory model fitting. A subject is excluded from confirmatory
prediction analysis if fewer than three permutation-valid task runs or fewer than 25 retained
plateau trials remain. Each permutation refits the nuisance-only and nuisance-plus-EEG models,
including inner-fold hyperparameter selection.

The $R^2$ denominator uses the permuted training-target mean for that fold, matching the
observed-analysis zero-skill baseline. Sensitivity nulls include run-label shuffling, removal of
subject and task-run means from both EEG features and targets before permutation, and within-run
random shuffling.

A permutation draw is invalid if any outer fold cannot be scored under the prespecified pipeline:
rank-deficient nuisance design, target-transformation error, zero target or prediction variance
needed for the metric, no retained EEG features after fold-contained filtering, invalid inner
GroupKFold split, model non-convergence after the prespecified maximum iterations, or a retained
trial structure that violates the permutation-valid run rules. Invalid draws are resampled before
outcome inspection. Confirmatory inference uses 5,000 valid draws, and the number of attempted
draws is reported. The planned null size is fixed before outcome inspection.

## 11. Validity and Sensitivity Analyses

### 11.1 Behavioral Target Diagnostics

Behavioral reports are used to document target validity rather than to define additional EEG
prediction endpoints. The report summarizes the association of NPS and SIIPS1 with the binary pain
report and the within-scale thermal or pain intensity score. These summaries are diagnostic
outputs and are not model endpoints for Study 1.

### 11.2 Temporal Specificity and Anticipatory Controls

Temporal control analyses use the same nuisance-only versus nuisance-plus-EEG
$\Delta R^2_{\text{LOSO}}$ framework. Models trained on individual-channel alpha+beta+gamma EEG
features from −5.0 to −0.01 s and −0.2 to −0.01 s predict post-stimulus target expression.
These controls use raw log-power summaries with no TFR baseline correction
(`feature_baseline_window: null`).

For NPS, pre-stimulus prediction is interpreted as a negative-control result for evoked nociceptive
signature expression. For SIIPS1, pre-stimulus prediction may reflect expectancy or other
anticipatory top-down pain processes, so it is interpreted as temporal-specificity and anticipatory
evidence rather than a pure failed negative control. For each target and window, the report includes
$\Delta R^2_{\text{LOSO}}$, bootstrap confidence intervals, and Holm-corrected p-values.

The wrong-lag window is ramp-up ($0.0$-$3.0$ s), before the held-temperature plateau begins. Plateau
sensitivity windows split the 7.5 s hold into early ($3.0$-$5.5$ s), mid ($5.5$-$8.0$ s), and late
($8.0$-$10.5$ s) intervals. These plateau windows are response-period sensitivity analyses, not
negative controls, and they are constrained to end at the plateau boundary so they do not include
ramp-down. The temporal-control analysis repeats full inner GroupKFold hyperparameter selection for
every configured pre-stimulus, wrong-lag, and plateau-sensitivity window.

### 11.3 Reference-Power Sensitivity

The primary baseline is not interpreted as neural neutrality. It estimates plateau power relative
to the immediately preceding oscillatory context. The reference-window sensitivity family therefore
reruns the Study 1 feature extraction and benchmark with −2.0 to −0.01 s and −0.2 to −0.01 s
baselines, preserving the same active plateau window, feature families, target construction,
folding, nuisance residualization, and permutation scheme.

The raw-active-power sensitivity disables baseline normalization for the active 3.0–10.5 s plateau
power features and includes −5.0 to −0.01 s reference power as a covariate. This analysis asks
whether predictive value depends on the ratio transform itself or remains when active power and
pre-stimulus oscillatory state are modeled separately.

## References

Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J.,
Gramfort, A., Thirion, B., & Varoquaux, G. (2014). Machine learning for neuroimaging
with scikit-learn. Frontiers in Neuroinformatics, 8, 14. doi.org/10.3389/fninf.2014.00014

Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging artifact from
continuous EEG recorded during functional MRI. NeuroImage, 12(2), 230-239.
doi.org/10.1006/nimg.2000.0599

Allen, P. J., Polizzi, G., Krakow, K., Fish, D. R., & Lemieux, L. (1998). Identification of EEG
events in the MR scanner. The problem of pulse artifact and a method for its subtraction.
NeuroImage, 8(3), 229-239. doi.org/10.1006/nimg.1998.0361

Appelhoff, S., Sanderson, M., Brooks, T. L., van Vliet, M., Quentin, R., Holdgraf, C.,
Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C.,
Rockhill, A. P., Larson, E., Gramfort, A., & Jas, M. (2019). MNE-BIDS. Organizing
electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open
Source Software, 4(44), 1896. doi.org/10.21105/joss.01896

Behzadi, Y., Restom, K., Liau, J., & Liu, T. T. (2007). A component based noise correction method
(CompCor) for BOLD and perfusion based fMRI. NeuroImage, 37(1), 90-101.
doi.org/10.1016/j.neuroimage.2007.04.042

Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline.
Standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16.
doi.org/10.3389/fninf.2015.00016

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
doi.org/10.1023/A:1010933404324

Brown, W. (1910). Some experimental results in the correlation of mental abilities. British Journal
of Psychology, 3(3), 296-322. doi.org/10.1111/j.2044-8295.1910.tb00207.x

Buysse, D. J. (2014). Sleep health. Can we define it? Does it matter? Sleep, 37(1), 9-17.
doi.org/10.5665/sleep.3298

CANlab. (n.d.). CanlabCore documentation. Retrieved May 17, 2026, from
https://canlabcore.readthedocs.io/

Coghill, R. C., McHaffie, J. G., & Yen, Y.-F. (2003). Neural correlates of interindividual
differences in the subjective experience of pain. Proceedings of the National Academy of Sciences
of the United States of America, 100(14), 8538-8542. doi.org/10.1073/pnas.1430684100

Cohen, M. X. (2014). Analyzing neural time series data. Theory and practice. MIT Press.

Davis, K. D., Aghaeepour, N., Ahn, A. H., Angst, M. S., Borsook, D., Brenton, A., Burczynski,
M. E., Crean, C., Edwards, R., Gaudilliere, B., Hergenroeder, G. W., Iadarola, M. J., Iyengar, S.,
Jiang, Y., Kong, J.-T., Mackey, S., Saab, C. Y., Sang, C. N., Scholz, J., ... Pelleymounter, M. A.
(2020). Discovery and validation of biomarkers to aid the development of safe and effective pain
therapeutics. Challenges and opportunities. Nature Reviews Neurology, 16(7), 381-400.
doi.org/10.1038/s41582-020-0362-2

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T.,
Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020). Parameterizing
neural power spectra into periodic and aperiodic components. Nature Neuroscience, 23(12),
1655-1665. doi.org/10.1038/s41593-020-00744-x

Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A.,
Kent, J. D., Goncalves, M., DuPre, E., Snyder, M., Oya, H., Ghosh, S. S., Wright, J.,
Durnez, J., Poldrack, R. A., & Gorgolewski, K. J. (2019). fMRIPrep. A robust preprocessing
pipeline for functional MRI. Nature Methods, 16(1), 111-116. doi.org/10.1038/s41592-018-0235-4

Fonov, V. S., Evans, A. C., McKinstry, R. C., Almli, C. R., & Collins, D. L. (2009). Unbiased
nonlinear average age-appropriate brain templates from birth to adulthood. NeuroImage,
47(Suppl. 1), S102. doi.org/10.1016/S1053-8119(09)70884-5

Fonov, V. S., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C., & Collins, D. L. (2011).
Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, 54(1), 313-327.
doi.org/10.1016/j.neuroimage.2010.07.033

Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D., & Turner, R. (1998).
Event-related fMRI. Characterizing differential responses. NeuroImage, 7(1), 30-40.
doi.org/10.1006/nimg.1997.0306

Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P.,
Flandin, G., Ghosh, S. S., Glatard, T., Halchenko, Y. O., Handwerker, D. A., Hanke, M.,
Keator, D., Li, X., Michael, Z., Maumet, C., Nichols, B. N., Nichols, T. E., Pellman, J.,
... Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and
describing outputs of neuroimaging experiments. Scientific Data, 3, 160044.
doi.org/10.1038/sdata.2016.44

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.
(2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging.
NeuroImage, 87, 96-110. doi.org/10.1016/j.neuroimage.2013.10.067

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression. Biased estimation for nonorthogonal
problems. Technometrics, 12(1), 55-67. doi.org/10.1080/00401706.1970.10488634

Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of
Statistics, 6(2), 65-70.

Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject. Automated
artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
doi.org/10.1016/j.neuroimage.2017.06.030

Jas, M., Larson, E., Engemann, D. A., Leppäkangas, J., Taulu, S., Hämäläinen, M., & Gramfort, A.
(2018). A reproducible MEG/EEG group study with the MNE software. Recommendations, quality
assessments, and good practices. Frontiers in Neuroscience, 12, 530.
doi.org/10.3389/fnins.2018.00530

Jasper, H. H. (1958). The ten-twenty electrode system of the International Federation.
Electroencephalography and Clinical Neurophysiology, 10, 371-375.

Kim, H., Neubert, J. K., San Miguel, A., Xu, K., Krishnaraju, R. K., Iadarola, M. J., Goldman, D.,
& Dionne, R. A. (2004). Genetic influence on variability in human acute experimental pain
sensitivity associated with gender, ethnicity and psychological temperament. Pain, 109(3), 488-496.
doi.org/10.1016/j.pain.2004.02.027

Kim, J. A., & Davis, K. D. (2021). Neural oscillations. Understanding a neural code of pain. The
Neuroscientist, 27(5), 544-570. doi.org/10.1177/1073858420958629

Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9. Validity of a brief
depression severity measure. Journal of General Internal Medicine, 16(9), 606-613.
doi.org/10.1046/j.1525-1497.2001.016009606.x

Lee, T.-W., Girolami, M., & Sejnowski, T. J. (1999). Independent component analysis using an
extended infomax algorithm for mixed sub-Gaussian and super-Gaussian sources. Neural Computation,
11(2), 417-441. doi.org/10.1162/089976699300016719

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International
Conference on Learning Representations. https://openreview.net/forum?id=Bkg6RiCqY7

Mari, T., Henderson, J., Maden, M., Nevitt, S. J., Duarte, R., & Fallon, N. (2022). Systematic
review of the effectiveness of machine learning algorithms for classifying pain intensity,
phenotype or treatment outcomes using electroencephalogram data. The Journal of Pain, 23(3),
349-369. doi.org/10.1016/j.jpain.2021.07.011

Mumford, J. A., Turner, B. O., Ashby, F. G., & Poldrack, R. A. (2012). Deconvolving BOLD activation
in event-related designs for multivoxel pattern classification analyses. NeuroImage, 59(3),
2636-2643. doi.org/10.1016/j.neuroimage.2011.08.076

Muthukumaraswamy, S. D. (2013). High-frequency brain activity and muscle artifacts in MEG/EEG.
A review and recommendations. Frontiers in Human Neuroscience, 7, 138.
doi.org/10.3389/fnhum.2013.00138

Oldfield, R. C. (1971). The assessment and analysis of handedness. The Edinburgh inventory.
Neuropsychologia, 9(1), 97-113. doi.org/10.1016/0028-3932(71)90067-4

Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel. An automated
electroencephalographic independent component classifier, dataset, and website. NeuroImage, 198,
181-197. doi.org/10.1016/j.neuroimage.2019.05.026

Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012). Spurious
but systematic correlations in functional connectivity MRI networks arise from subject motion.
NeuroImage, 59(3), 2142-2154. doi.org/10.1016/j.neuroimage.2011.10.018

Robinson, M. E., Riley, J. L. III, Myers, C. D., Papas, R. K., Wise, E. A., Waxenberg, L. B.,
& Fillingim, R. B. (2001). Gender role expectations of pain. Relationship to sex differences in
pain. The Journal of Pain, 2(5), 251-257. doi.org/10.1054/jpai.2001.24551

Spearman, C. (1910). Correlation calculated from faulty data. British Journal of Psychology, 3(3),
271-295. doi.org/10.1111/j.2044-8295.1910.tb00206.x

Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Löwe, B. (2006). A brief measure for assessing
generalized anxiety disorder. The GAD-7. Archives of Internal Medicine, 166(10), 1092-1097.
doi.org/10.1001/archinte.166.10.1092

Sullivan, M. J. L., Bishop, S. R., & Pivik, J. (1995). The Pain Catastrophizing Scale.
Development and validation. Psychological Assessment, 7(4), 524-532.
doi.org/10.1037/1040-3590.7.4.524

van der Miesen, M. M., Lindquist, M. A., & Wager, T. D. (2019). Neuroimaging-based biomarkers for
pain. State of the field and current directions. Pain Reports, 4(4), e751.
doi.org/10.1097/PR9.0000000000000751

Wager, T. D., Atlas, L. Y., Lindquist, M. A., Roy, M., Woo, C.-W., & Kross, E. (2013). An
fMRI-based neurologic signature of physical pain. New England Journal of Medicine, 368(15),
1388-1397. doi.org/10.1056/NEJMoa1204471

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation
inference for the general linear model. NeuroImage, 92, 381-397.
doi.org/10.1016/j.neuroimage.2014.01.060

Woo, C.-W., Schmidt, L., Krishnan, A., Jepma, M., Roy, M., Lindquist, M. A., Atlas, L. Y., &
Wager, T. D. (2017). Quantifying cerebral contributions to pain beyond nociception. Nature
Communications, 8, 14211. doi.org/10.1038/ncomms14211

Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or
symmetry. Biometrika, 87(4), 954-959. doi.org/10.1093/biomet/87.4.954

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of
the Royal Statistical Society: Series B (Statistical Methodology), 67(2), 301-320.
doi.org/10.1111/j.1467-9868.2005.00503.x
