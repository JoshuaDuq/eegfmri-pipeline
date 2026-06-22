# Study 3 - Paired EEG Comparison of MRI and Simulator Thermal-Pain Sessions

## 1. Problem Statement

Study 1 tests whether plateau-window EEG spectral power predicts trial-wise fMRI
pain-signature expression during simultaneous EEG-fMRI thermal stimulation. Simultaneous
acquisition provides spatially defined targets, but EEG recorded in the MRI environment can retain
scanner-gradient, cardioballistic, acoustic, vibration, posture, movement, and high-frequency muscle
artifact structure after preprocessing (Allen et al., 1998, 2000; Muthukumaraswamy, 2013).

Study 3 adds an EEG-only MRI-simulator session to estimate how the acquisition environment changes
thermal/pain-related EEG activity. The same participants complete matched task sequences during
simultaneous EEG-fMRI and simulator EEG, allowing direct within-participant comparisons of spectral
power, behavioral ratings, and artifact structure without fitting or applying a machine-learning
model.

## 2. Objectives

The primary objective is to estimate paired differences in plateau-window, baseline-corrected
whole-scalp alpha, beta, and gamma power between simultaneous EEG-fMRI and simulator EEG during
matched thermal-stimulation trials.

Secondary objectives are to:

1. Compare channel-level spectral-power patterns across acquisition environments.
2. Compare behavioral thermal/pain responses across sessions.
3. Estimate repetition, habituation, and sensitization effects.
4. Test within-session associations of spectral power with temperature and ratings.
5. Characterize resting-state EEG and artifact structure across acquisition environments.

Study 3 reuses Study 1 task definitions, fixed frequency bands, preprocessing principles, active
window, and baseline window to support paired measurement. It does not use Study 1 model weights,
feature masks, prediction scores, or other machine-learning outputs. Study 2 outputs are outside the
Study 3 design. With 10 participants, all analyses are framed as pilot feasibility and effect-size
estimation rather than confirmatory validation.

## 3. Hypotheses and Estimands

### 3.1 Primary Estimands

The primary analysis estimates whether baseline-corrected spectral power differs between the two
acquisition environments. It does not specify a directional hypothesis.

For participant $s$, matched trial $t$, and frequency band $b$, the paired trial difference is:

$$
D_{s,t,b} = P_{s,t,b}^{\mathrm{sim}} - P_{s,t,b}^{\mathrm{fmri}}.
$$

$P_{s,t,b}$ is the mean log-ratio power across the participant's channels that are valid in both
sessions, excluding Fp1 and Fp2. The participant-level difference is the mean across matched,
retained trials:

$$
\bar{D}_{s,b} = \frac{1}{T_s}\sum_{t=1}^{T_s}D_{s,t,b}.
$$

The primary estimand for each of alpha, beta, and gamma is the subject-weighted mean paired
difference:

$$
\bar{D}_b = \frac{1}{S}\sum_{s=1}^{S}\bar{D}_{s,b}.
$$

The primary report includes the paired difference, standardized paired effect size, 95% bootstrap
confidence interval, and an exact two-sided subject-level sign-flip p-value. The three band-level
p-values are Holm-corrected and treated as supportive evidence. The planned sample is $S = 10$.

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

Secondary paired analyses estimate band-wise cross-session concordance across matched trials and
participants. Channel-level maps describe whether the spatial distribution of alpha, beta, and
gamma power is similar with and without MRI acquisition. These estimates are descriptive because
the pilot sample is too small for stable participant-level correlation inference.

## 4. Study Design

### 4.1 Participants

Study 3 includes 10 volunteers from the 60 Study 1 participants, balanced by sex: 5 men and
5 women. Participants are invited to return to the CERVO research center for a second experimental
session of approximately 120 min during the week following their EEG-fMRI session. They may accept
or decline this optional session without consequence for Study 1 participation or compensation. A
separate compensation of 60$ is provided for the second visit.

Eligibility, questionnaires, handedness criteria, thermode tolerability checks, and EEG
compatibility criteria match Study 1. Sex-stratified summaries are descriptive.

### 4.2 Session Timing

The acquisition order is fixed: participants complete Study 1 EEG-fMRI first and return during the
following week for the simulator EEG session. Elapsed days, time of day, and thermode site are
recorded. Cross-session differences therefore estimate MRI-environment plus repeat-exposure/order
effects, not a randomized pure environment contrast. Elapsed time and prior-session thermal
exposure are described but are not added as between-participant covariates to the primary analysis
because the pilot sample cannot support that adjustment reliably.

### 4.3 MRI Simulator Setup

The simulator session is conducted in an MRI simulator located in a room adjacent to the main MRI
scanner. The simulator reproduces the physical MRI environment without magnetic field exposure or
image acquisition. The setup includes the bore, scanner table, head-coil mirror, visual display,
thermode placement, and supine posture used in Study 1.

The physical response device is replaced by a programmable VAYDEER keyboard with two functional
keys, while the rating task, scale definitions, and response mapping match Study 1.
Scanner-acoustic recordings from the Study 1 EEG-fMRI acquisitions are replayed through two
speakers positioned on the left and right sides of the bore. Playback follows the corresponding
acquisition phase. Sound level is calibrated at the head position and logged. Hearing protection
matches the EEG-fMRI session.

The simulator session includes resting-state EEG, thermal stimulation, subjective ratings, and task
EEG. MRI-specific procedures and outputs are absent: anatomical MRI, resting-state fMRI, field maps,
EPI sequences, volume triggers, scanner-gradient correction, fMRI motion, DVARS, NPS, and SIIPS1.
The simulator reproduces posture and acoustics, but it does not reproduce magnetic-field exposure,
scanner-gradient electromagnetic artifacts, table vibration, or true scan-trigger timing.

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

Implausible behavioral trials are excluded before band-power analyses: ratings outside the logged
response scale or ratings of 0 ("no sensation") at temperatures >= 47.3 °C.

## 6. EEG Acquisition and Preprocessing

The simulator session uses the same EEG montage, acquisition hardware family, recording filters,
sampling-rate target, impedance targets, ECG recording, thermode, rating task, and response mapping
as the EEG-fMRI session. Passive Ag/AgCl electrodes follow the extended 10-20 system (Jasper, 1958).
ECG supports cardiac artifact quantification and ICA review. Behavioral responses are collected with
the VAYDEER two-key device.

Simulator preprocessing mirrors the Study 1 EEG steps for task EEG. MRI-gradient correction and
MRI-volume-locked cardioballistic template subtraction are omitted.
Because EEG-fMRI requires artifact-correction steps that are absent from simulator EEG, paired
differences reflect the acquisition environment and its required preprocessing rather than a pure
difference in neural activity.

The preprocessing sequence is:

1. Validate channel names, sampling rate, event markers, impedance log, ECG channel, and task logs.
2. Apply the prespecified filtering and line-noise handling.
3. Detect and document bad channels before ICA.
4. Fit ICA on a 1.0 Hz high-pass-filtered copy of the task epochs.
5. Apply ICA spatial weights to the continuous analysis data.
6. Reject ICA components using ICLabel and ocular, cardiac, muscle, and high-frequency rules.
7. Extract epochs from -7.0 to 15.0 s relative to stimulus onset.
8. Apply autoreject in local mode using the Study 1 candidate interpolation grid.

Preprocessing thresholds are fixed before cross-session comparisons and applied without reference
to ratings, temperature effects, or band-power differences.

## 7. Band-Power Construction

Band-power construction uses the same spectral definitions in both sessions. For each retained
trial and channel, power is summarized over the plateau window and baseline-corrected as a
log-ratio. The primary whole-scalp summary is the mean across channels that are valid in both
sessions for that participant, excluding Fp1 and Fp2.

Primary comparisons require a matched trial in both sessions using participant ID, block, trial
index, temperature, thermode surface, and original event order. If either member fails behavioral
or EEG quality control, the pair is excluded from the primary comparison and retained only in the
session-specific quality-control report.

Fixed band definitions:

1. Alpha: 8.0-12.9 Hz.
2. Beta: 13.0-30.0 Hz.
3. Gamma: 30.1-80.0 Hz with the Study 1 line-noise exclusion.

Alpha, beta, and gamma are analyzed as three separate prespecified measures; they are not combined
by fitted weights. Gamma interpretation remains contingent on high-frequency artifact diagnostics.
Fp1 and Fp2 high-frequency power are retained as artifact measures and excluded from the primary
whole-scalp summary. Channel-level band-power maps are secondary and use no data-driven feature
selection.

## 8. Statistical Analysis

### 8.1 Primary Paired Analysis

The primary analysis follows the estimands in Section 3.1. Trial-level simulator-minus-EEG-fMRI
differences are averaged within participant before group analysis, so participants rather than
trials define the inferential sample. Alpha, beta, and gamma are analyzed separately.

For each band, the report includes the participant-level differences, subject-weighted mean paired
difference, standardized paired effect size, and 95% participant-bootstrap confidence interval. An
exact two-sided subject-level sign-flip test provides a supportive p-value. Holm correction is
applied across the three band-level tests. Raw and corrected p-values are reported regardless of
threshold crossing.

### 8.2 Secondary Models

Secondary analyses estimate:

1. Paired differences in within-scale thermal/pain intensity, binary pain report, and
   painful-trials-only intensity.
2. Within-session associations of each band with ordered temperature and thermal/pain intensity.
3. Paired EEG-fMRI versus simulator differences in those within-session slopes.
4. Channel-level band-power differences and spatial concordance across sessions.
5. Repetition, habituation, sensitization, and trial-history effects.
6. Resting-state alpha, beta, gamma, aperiodic slope, line-noise, ECG, and frontal high-frequency
   metrics.

Secondary analyses emphasize effect sizes and uncertainty. Any secondary p-values are
Holm-corrected within their stated outcome family. Sex-specific summaries are descriptive; no sex
interaction test is planned with five participants per sex.

### 8.3 Robustness Analyses

Robustness analyses repeat the primary paired comparisons after:

1. Excluding the first deterministic high-temperature trial.
2. Excluding the first task block.
3. Matching on previous-trial temperature and signed temperature change.
4. Censoring trial pairs when either session has high artifact burden.
5. Using immediate and early pre-cue baselines.
6. Replacing the channel mean with the channel median.

Temporal negative controls repeat the primary paired comparison in pre-stimulus, immediate
pre-stimulus, ramp-up, and return-to-baseline windows.

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
