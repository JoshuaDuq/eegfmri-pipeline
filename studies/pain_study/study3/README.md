# Study 3 - EEG-Only MRI Simulator Validation of Thermal/Pain-Related EEG Features

## 1. Problem Statement

Study 1 tests whether plateau-window EEG spectral power predicts trial-wise fMRI
pain-signature expression during simultaneous EEG-fMRI thermal stimulation. Simultaneous
acquisition provides spatially defined targets, but EEG recorded in the MRI environment can retain
scanner-gradient, cardioballistic, acoustic, vibration, posture, movement, and high-frequency muscle
artifact structure after preprocessing (Allen et al., 1998, 2000; Muthukumaraswamy, 2013).

Study 3 adds an EEG-only MRI-simulator session to test whether the Study 1 EEG feature expression
replicates under matched posture, task timing, scanner acoustics, thermal stimulation, and rating
demands without MRI acquisition. The simulator session contributes EEG, behavioral ratings, and
artifact covariates.

## 2. Objectives

The primary objective is to test whether the frozen Study 1 NPS ElasticNet alpha+beta
individual-channel spectral-power expression is associated with within-scale thermal/pain intensity
during EEG-only MRI-simulator acquisition after accounting for stimulus temperature.

Secondary objectives are to:

1. Compare EEG-fMRI and simulator-session EEG feature expression within the same participants.
2. Test alpha and beta contribution scores separately.
3. Estimate repetition, habituation, and sensitization effects.
4. Compare behavioral thermal/pain responses across sessions.
5. Characterize resting-state EEG and artifact structure across acquisition environments.

Study 3 uses the Study 1 primary cell, frequency bands, preprocessing recipe, active window,
baseline window, feature mask, and model weights without refitting.

## 3. Hypotheses and Estimands

### 3.1 Primary Hypothesis

The primary hypothesis is that frozen Study 1 EEG feature expression is positively associated with
temperature-adjusted within-scale thermal/pain intensity during the simulator session.

For participant $s$ and trial $t$, the simulator-session feature-expression score is:

$$
\eta_{s,t}^{\mathrm{sim}} = X_{s,t}^{\mathrm{sim}} W_{\mathrm{Study1}, -s}.
$$

$X_{s,t}^{\mathrm{sim}}$ is the simulator alpha+beta spectral-power feature vector processed with
the Study 1 feature recipe. $W_{\mathrm{Study1}, -s}$ is the corresponding Study 1 outer-fold
model trained without participant $s$.

The primary estimand is the subject-weighted mean temperature-adjusted thermal/pain association:

$$
\bar{\beta}_{\mathrm{sim}} =
\frac{1}{S}\sum_{s=1}^{S}\beta_{s,\mathrm{sim}}.
$$

$\beta_{s,\mathrm{sim}}$ is the within-participant slope relating
$\eta_{s,t}^{\mathrm{sim}}$ to within-scale thermal/pain intensity after adjustment for ordered
temperature and the primary nuisance design. The predicted direction follows the frozen
transformed-space Study 1 linear predictor: larger $\eta$ means stronger NPS-predictive EEG
expression.

Group inference uses a one-sided subject-level sign-flip test. The standardized mean slope, 95%
bootstrap confidence interval, and p-value are reported. The planned sample is $S = 16$.

### 3.2 Behavioral Criteria

The primary behavioral criterion is the Study 1 within-scale thermal/pain intensity score:
non-painful heat ratings remain on the 0 to 99 scale, and painful-trial ratings subtract 100 from
the displayed 100 to 200 pain scale.

Secondary behavioral criteria are:

1. Ordered stimulus temperature.
2. Binary pain report.
3. Painful-trials-only intensity.

Temperature-only effects are interpreted as nociceptive-dose sensitivity.

### 3.3 Cross-Session Concordance

The same frozen feature-expression score is computed for the EEG-fMRI and simulator sessions.
Paired analyses estimate:

$$
\Delta\beta_s = \beta_{s,\mathrm{sim}} - \beta_{s,\mathrm{fmri}},
$$

and

$$
\rho_{\mathrm{paired}} =
\mathrm{corr}(\beta_{s,\mathrm{sim}}, \beta_{s,\mathrm{fmri}}).
$$

Cross-session concordance describes whether the EEG feature has similar thermal/pain associations
with and without MRI acquisition.

## 4. Study Design

### 4.1 Participants

Study 3 includes 16 volunteers from the 60 Study 1 participants, balanced by sex: 8 men and
8 women. Participants are invited to return to the CERVO research center for a second
experimental session of approximately 120 min within two weeks of the EEG-fMRI session. They may
accept or decline this optional session without consequence for Study 1 participation or
compensation. A separate compensation of 60$ is provided for the second visit.

Eligibility, questionnaires, handedness criteria, thermode tolerability checks, and EEG
compatibility criteria match Study 1. Sex-stratified estimates are reported.

### 4.2 Session Timing

The acquisition order is fixed: participants complete Study 1 EEG-fMRI first and the simulator EEG
session second. Calendar interval, time of day and thermode site are recorded.

Paired models include elapsed days, cumulative thermal exposure, previous-session mean rating, and
previous-session maximum-temperature tolerability.

### 4.3 MRI Simulator Setup

The simulator session is conducted in an MRI simulator located in a room adjacent to the main MRI
scanner. The simulator reproduces the physical MRI environment without magnetic field exposure or
image acquisition. The setup includes the bore, scanner table, head-coil mirror, visual display,
thermode placement, and supine posture used in Study 1.

The response interface is replaced by a programmable VAYDEER keyboard with two functional keys.
Scanner-acoustic recordings from the Study 1 EEG-fMRI acquisitions are replayed through two
speakers positioned on the left and right sides of the bore. Playback follows the corresponding
acquisition phase. Sound level is calibrated at the head position and logged. Hearing protection
matches the EEG-fMRI session.

The simulator session includes resting-state EEG, thermal stimulation, subjective ratings, and task
EEG. MRI-specific procedures and outputs are absent: anatomical MRI, resting-state fMRI, field maps,
EPI sequences, volume triggers, scanner-gradient correction, fMRI motion, DVARS, NPS, and SIIPS1.

## 5. Thermal Pain Protocol

The EEG-only task reuses each participant's EEG-fMRI thermal sequence: same temperature levels,
trial order, thermode surfaces, and block structure. Sequence repetition is intentional for paired
comparability; trial-history and elapsed-time terms quantify habituation, sensitization, and
expectancy effects.

Thermal stimulation follows the Study 1 structure. Each trial begins with a variable fixation
interval while the thermode is held at baseline temperature. Thermal stimulation lasts 12.5 s,
including ramp-up, plateau, and return-to-baseline phases. Participants then report whether the
stimulus was painful and provide a visual analogue rating.

The active EEG analysis window is the thermal plateau from 3.0 to 10.5 s after stimulus onset.
Trial identity is defined by participant ID, session, block, trial index, temperature, thermode
surface, and original event order.

Implausible behavioral trials are excluded before EEG-feature analyses: ratings outside the logged
response scale or ratings of 0 ("no sensation") at temperatures >= 47.3 °C.

## 6. EEG Acquisition and Preprocessing

The simulator session uses the same EEG montage, acquisition hardware family, recording filters,
sampling-rate target, impedance targets, ECG recording, thermode, and rating interface as the
EEG-fMRI session. Passive Ag/AgCl electrodes follow the extended 10-20 system (Jasper, 1958). ECG
supports cardiac artifact quantification and ICA review. Behavioral responses are collected with
the VAYDEER two-key device.

Simulator preprocessing mirrors the Study 1 EEG steps for task EEG. MRI-gradient correction and
MRI-volume-locked cardioballistic template subtraction are omitted.

The preprocessing sequence is:

1. Validate channel names, sampling rate, event markers, impedance log, ECG channel, and task logs.
2. Apply the prespecified filtering and line-noise handling.
3. Detect and document bad channels before ICA.
4. Fit ICA on a 1.0 Hz high-pass-filtered copy of the task epochs.
5. Apply ICA spatial weights to the continuous analysis data.
6. Reject ICA components using ICLabel and ocular, cardiac, muscle, and high-frequency rules.
7. Extract epochs from -7.0 to 15.0 s relative to stimulus onset.
8. Apply autoreject in local mode using the Study 1 candidate interpolation grid.

Preprocessing is blind to ratings, temperature effects, fMRI targets, and feature-expression
outcomes.

## 7. Feature Construction

Feature construction uses the Study 1 individual-channel spectral-power recipe. For each retained
trial, power is summarized over the plateau window, baseline-corrected as a log-ratio, and assembled
into the frozen feature matrix.

Fixed feature definitions:

1. Alpha: 8.0-12.9 Hz.
2. Beta: 13.0-30.0 Hz.
3. Alpha+beta: the frozen Study 1 primary feature preset.

Gamma is exploratory. Fp1 and Fp2 high-frequency power are retained as artifact covariates and
excluded from the predictive feature matrix.

Simulator features are compared with the Study 1 training-fold distribution using feature-mask
agreement, fold-owned scaling parameters, robust feature $z$-scores, and PCA-space distance with
Ledoit-Wolf shrinkage covariance. These diagnostics describe whether the frozen Study 1 model is
being applied to comparable EEG feature distributions.

## 8. Statistical Analysis

### 8.1 Primary Model

All continuous non-intercept regressors are centered and scaled within participant. The primary
trial-level simulator model is:

$$
\eta_{s,t}^{\mathrm{sim}} =
\alpha_s + \beta_{s,\mathrm{rating}} R_{s,t} + \beta_{s,\mathrm{temp}} T_{s,t}
+ N_{s,t}^{\mathrm{primary}}\gamma_s + \epsilon_{s,t}.
$$

$R_{s,t}$ is within-scale thermal/pain intensity. $T_{s,t}$ is ordered temperature.
$N_{s,t}^{\mathrm{primary}}$ contains task-block intercepts, thermode surface, trial index within
block, cumulative exposure count, Fp1/Fp2 high-frequency artifact power, ECG-derived cardiac
metrics, ocular artifact metrics, muscle artifact metrics, and retained bad-channel or
interpolation counts.

Subject-level slopes are estimated first. Group inference is performed on subject-level slopes with
a one-sided sign-flip test. The primary report includes the p-value, standardized mean slope, and
95% bootstrap confidence interval.

### 8.2 Secondary Models

Secondary analyses estimate:

1. Ordered-temperature association.
2. Binary pain-report association.
3. Painful-trials-only intensity association.
4. Alpha and beta contribution-score associations.
5. Paired EEG-fMRI versus simulator slope differences.
6. Channel-pattern concordance across sessions.
7. Resting-state alpha, beta, gamma, aperiodic slope, line-noise, ECG, and frontal high-frequency
   metrics.

Secondary families are Holm-corrected within family. Exploratory models are reported separately.

### 8.3 Robustness Analyses

Robustness analyses repeat the primary model after:

1. Excluding the first deterministic high-temperature trial.
2. Excluding the first task block.
3. Adding previous-trial temperature and signed temperature change.
4. Adding elapsed-time terms.
5. Censoring high-artifact trials.
6. Using immediate and early pre-cue baselines.
7. Equalizing retained trial counts across sessions.

Temporal negative controls repeat the primary model in pre-stimulus, immediate pre-stimulus,
ramp-up, and return-to-baseline windows.

## 9. Artifact Checks

Simulator artifact covariates include Fp1/Fp2 70-95 Hz power, ECG-derived cardiac metrics, ocular
ICA component expression, muscle ICA component expression, channel interpolation count, epoch
peak-to-peak amplitude, line-noise burden, and response movement indicators.

EEG-fMRI artifact covariates include scanner-frequency residual power, volume-locked residual peaks,
framewise displacement, and DVARS.

Artifact checks summarize sign, magnitude, covariation with artifact metrics, EEG-fMRI-specific
scanner-residual interactions, and simulator/EEG-fMRI channel-map concordance.

## References

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

Cohen, M. X. (2014). Analyzing neural time series data. Theory and practice. MIT Press.

Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P.,
Flandin, G., Ghosh, S. S., Glatard, T., Halchenko, Y. O., Handwerker, D. A., Hanke, M.,
Keator, D., Li, X., Michael, Z., Maumet, C., Nichols, B. N., Nichols, T. E., Pellman, J.,
... Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and
describing outputs of neuroimaging experiments. Scientific Data, 3, 160044.
doi.org/10.1038/sdata.2016.44

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.
(2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging.
NeuroImage, 87, 96-110. doi.org/10.1016/j.neuroimage.2013.10.067

Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of
Statistics, 6(2), 65-70.

Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject. Automated
artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
doi.org/10.1016/j.neuroimage.2017.06.030

Jasper, H. H. (1958). The ten-twenty electrode system of the International Federation.
Electroencephalography and Clinical Neurophysiology, 10, 371-375.

Kim, J. A., & Davis, K. D. (2021). Neural oscillations. Understanding a neural code of pain. The
Neuroscientist, 27(5), 544-570. doi.org/10.1177/1073858420958629

Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data.
Journal of Neuroscience Methods, 164(1), 177-190. doi.org/10.1016/j.jneumeth.2007.03.024

Muthukumaraswamy, S. D. (2013). High-frequency brain activity and muscle artifacts in MEG/EEG.
A review and recommendations. Frontiers in Human Neuroscience, 7, 138.
doi.org/10.3389/fnhum.2013.00138

Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel. An automated
electroencephalographic independent component classifier, dataset, and website. NeuroImage, 198,
181-197. doi.org/10.1016/j.neuroimage.2019.05.026

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014).
Permutation inference for the general linear model. NeuroImage, 92, 381-397.
doi.org/10.1016/j.neuroimage.2014.01.060
