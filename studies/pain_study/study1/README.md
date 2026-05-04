# Materials and Methods

## Research Objectives
This study tested whether trial-wise expression of established fMRI pain signatures can be predicted above chance from concurrent EEG-derived features. The predictive targets were the Neurologic Pain Signature (NPS; Wager et al., 2013) and the Stimulus Intensity Independent Pain Signature-1 (SIIPS1; Woo et al., 2017), modeled via a subject-level, trial-resolved multimodal architecture. The primary confirmatory estimand was prospective, leave-one-subject-out (LOSO) prediction of stimulus- and acquisition-controlled signature expression: NPS and SIIPS1 dot-product scores residualized against prespecified stimulus, session, and acquisition nuisance variables, without using any distributional statistic from the held-out subject. Raw signature prediction was retained as a descriptive upper-bound analysis; subjective-rating residualization was treated as a construct-attenuation sensitivity analysis.

The NPS was developed as a brain-based marker of nociceptive processing (Wager et al., 2013), and its prediction may partly reflect stimulus intensity. The SIIPS1 captures stimulus-intensity-independent pain-related variance (Woo et al., 2017), making it conceptually closer to the residualized objective. SIIPS1 was nevertheless evaluated under the same residualization levels to verify independence in this dataset. All analyses were restricted to the plateau phase of thermal stimulation, including pain and non-pain thermal trials, utilizing fMRI inputs normalized to MNI152NLin2009cAsym space.

## Data Preconditions and Frequency Definitions
Analyses relied on task-evoked EEG epochs from $-7.0$ to $15.0$ s relative to stimulus onset. A $-0.2$ to $0.0$ s pre-stimulus baseline was applied to epoch-level voltage to remove DC offset prior to ERP and amplitude-based analyses. For time-frequency decompositions, the primary log-ratio baseline correction used the $-5.0$ to $-0.01$ s pre-stimulus window, selected to provide stable power estimates given that Morlet wavelet decomposition at alpha frequencies requires several cycles for adequate spectral resolution. Because this window may contain cue-locked expectancy activity, sensitivity analyses were prespecified using an immediate pre-stimulus baseline ($-0.2$ to $-0.01$ s) and an early pre-stimulus baseline ($-7.0$ to $-5.5$ s, interpreted only when event timing confirmed it preceded task cues). Baseline conclusions were considered stable only if the sign and inferential status of primary alpha and beta effects were unchanged across the primary and immediate-baseline analyses. All baseline windows were applied identically across conditions.

The active stimulation window for predictive modeling was $3.0$–$10.5$ s post-onset. Neural oscillations were operationalized into standard frequency bands: alpha ($8.0$–$12.9$ Hz), beta ($13.0$–$30.0$ Hz), and gamma ($30.1$–$80.0$ Hz). Composite alpha-to-gamma features were also constructed. Gamma-band EEG during simultaneous fMRI is vulnerable to facial muscle activity, jaw tension, scanner gradient residuals, and cardiac artifacts, all of which can be elicited by painful stimulation; gamma-band analyses were therefore exploratory and were not used to support confirmatory claims. Gamma robustness was evaluated using temporal/frontal high-frequency electromyographic power, framewise displacement, DVARS, cardiac phase-locking, and residual spectral peaks at the volume-repetition frequency. Prespecified artifact thresholds were: framewise displacement $> 0.5$ mm, temporal/frontal high-frequency power robust $z > 3$, DVARS robust $z > 3$, cardiac phase-locking above the 95th percentile of the within-subject circular-shift null, and scanner-frequency residual peaks robust $z > 3$. Gamma effects were reported as artifact-sensitive when removing threshold-exceeding trials changed the effect direction or eliminated significance. The gamma band excludes a $\pm 1.0$ Hz notch around 60 Hz due to line-noise removal.

## Simultaneous EEG-fMRI Preprocessing
Simultaneous EEG-fMRI acquisition introduces structured artifacts requiring correction prior to feature extraction (Bullock et al., 2021). Preprocessing proceeded in two stages.

In the first stage, MRI-induced artifacts were corrected using BrainVision Analyzer 2.0 (Brain Products GmbH). Gradient artifacts were removed via average artifact subtraction using a sliding-window template. Ballistocardiogram (BCG) artifacts were corrected via template subtraction aligned to the detected cardiac cycle.

In the second stage, the corrected data were imported into the MNE-BIDS-Pipeline. Continuous data were downsampled to 500 Hz, band-pass filtered (0.1–100 Hz), and notch-filtered at 60 Hz.

Bad channels were identified using PyPREP (Bigdely-Shamlo et al., 2015) with deviation-based and correlation-based rejection criteria. Detection was repeated three times with independent random seeds; a channel was marked bad only if flagged in a strict majority. Bad channels were interpolated via spherical spline interpolation, and data were re-referenced to the common average.

Independent component analysis was performed using the extended infomax algorithm (0.99 variance explained). ICA was fitted on 1.0 Hz high-pass filtered epochs to improve decomposition stability. Component classification used ICLabel (Pion-Tonachini et al., 2019); components were rejected if their predicted probability exceeded 0.8 for any non-brain category other than "other." Because ICLabel was trained on standard (non-fMRI) EEG, retained components were additionally inspected for spectral peaks at the volume-repetition frequency, cardiac phase-locking, and spatial topographies consistent with known artifact patterns.

Epochs were extracted from $-7.0$ to $15.0$ s relative to stimulus onset. Trial rejection used autoreject (Jas et al., 2017) in local mode with candidate interpolation counts of $\{4, 8, 16\}$. Electrode locations were digitized using the EasyCap M1 montage and co-registered to individual MRI.

## fMRI Signature Target Construction
Trial-by-trial fMRI effect estimates were extracted using the Least-Squares Separate (LSS) approach, restricted to thermal plateau trials. The LSS event regressor onset was defined at the beginning of the thermal plateau, with duration equal to the plateau hold time. Ramp-up, ramp-down, and non-stimulation trial types were modeled as separate nuisance regressors. This design aligns the fMRI event with the protocol-defined plateau phase used for EEG feature extraction, though the resulting LSS beta is an HRF-convolved event-amplitude estimate rather than a millisecond-scale neural measure.

The primary LSS models used a canonical SPM hemodynamic response function, a cosine drift model, and a $0.008$ Hz high-pass filter without spatial smoothing. The fMRI denoising design included the 24-parameter rigid-body motion expansion, white-matter and CSF signals, framewise displacement, CompCor regressors, and motion-outlier regressors from fMRIPrep. Trials exceeding framewise displacement $> 0.5$ mm or standardized DVARS robust $z > 3$ were excluded; runs losing more than 20% of plateau trials failed inclusion. HRF/timing robustness was evaluated by repeating target construction with HRF temporal and dispersion derivatives and with a finite-impulse-response model, and by shifting the EEG active window by $\pm 2.0$ s. A result was considered HRF/timing-stable only if the Level 2 ElasticNet estimate retained the same sign for both co-primary targets, performance changed by no more than 25%, and inferential status was unchanged after Holm correction.

The NPS weight map (Wager et al., 2013) and the SIIPS1 map (Woo et al., 2017) were registered to MNI152NLin2009cAsym space. When image grids differed, LSS beta maps were resampled to the signature-weight grid using continuous interpolation. For each signature, the scoring mask $V^{(k)}$ was fixed by the finite signature-weight voxels intersected with the fMRIPrep brain mask; the contributing voxel count was identical across all retained subjects, runs, and trials. LSS beta maps were not smoothed, z-scored, or trial-normalized before scoring; signature weights were not re-estimated from study data. The signature-expression metric was computed as:

$$y_{s,i}^{(k)} = \sum_{v \in V^{(k)}} \beta_{s,i}(v) \, M_k(v)$$

where $\beta_{s,i}(v)$ is the LSS-derived BOLD estimate for subject $s$, trial $i$, voxel $v$, and $M_k(v)$ is the a priori weight map for target $k \in \{\text{NPS}, \text{SIIPS1}\}$.

To construct the multimodal dataset, fMRI targets were matched to clean EEG trials using unique subject, run/block, and trial-index identifiers. EEG trigger onsets and fMRI plateau onsets were compared after applying the protocol offset from trigger to plateau start; residual absolute mismatch was required to be $\leq 0.010$ s for every retained trial.

## Hierarchical Nuisance Residualization
To characterize whether EEG prediction reflects pain-relevant fMRI variance or confound-correlated variance, fMRI signature expression was evaluated at three prespecified residualization levels representing distinct estimands.

**Level 1 — Raw expression.** No residualization. Models predict the full NPS/SIIPS1 dot-product expression, reflecting stimulus intensity, condition, temporal structure, subjective experience, and neural pain processing in an unknown mixture.

**Level 2 — Stimulus- and acquisition-controlled expression.** Targets were residualized against an intercept, dummy-coded stimulus temperature (lowest retained temperature as reference), dummy-coded thermode contact-surface index, block/run index, trial onset time, within-run trial number, HRF-weighted framewise displacement, HRF-weighted standardized DVARS, peripheral low-gamma power, and residual ECG coupling. Temperature and thermode contact surface were modeled categorically. Binary pain condition was excluded because it is a deterministic or near-deterministic recoding of the thermal manipulation and can render the estimand unidentified. Level 2 is the primary confirmatory target: it asks whether EEG predicts signature variance beyond measured stimulus level, thermode contact surface, session structure, trial order, and acquisition noise.

**Level 3 — Rating-residualized sensitivity.** The Level 2 design was augmented with subjective pain rating. Because ratings are a criterion measure of pain experience rather than a pure nuisance variable, this level may remove construct-relevant variance. Level 3 was interpreted only as a sensitivity analysis.

Within each level, nuisance coefficients were estimated exclusively on training subjects via SVD-based least squares:

$$\hat{\gamma} = \underset{\gamma}{\mathrm{argmin}} \; \| y_{train} - Z_{train}\gamma \|_2^2$$

Residualized targets for both training and test sets were computed by applying training-derived coefficients:

$$y_{train}^{resid} = y_{train} - Z_{train}\hat{\gamma}, \qquad y_{test}^{resid} = y_{test} - Z_{test}\hat{\gamma}$$

If prediction was significant at Level 1 but not Level 2, EEG features were interpreted as tracking stimulus or session structure. If prediction survived Level 2 but not Level 3, prediction was interpreted as overlapping with subjective pain-report variance.

## EEG Feature Extraction
Spectral power features were extracted using Morlet wavelet decomposition with frequency-adaptive cycle counts (cycles $= f / 2.0$, bounded between 3.0 and 15.0) and decimation factor 4. To isolate induced (non-phase-locked) oscillatory activity, the evoked response was subtracted from each trial prior to decomposition. The ERP was estimated as the fold-level average across all training trials; condition-specific ERP subtraction was evaluated as a sensitivity analysis. The ERP was computed exclusively from training-fold trials to prevent leakage.

Spectral power was log-ratio baseline-corrected ($-5.0$ to $-0.01$ s) and averaged within the active window ($3.0$–$10.5$ s) to yield one power value per trial, channel, and frequency band. Power features were computed at three spatial resolutions: individual channels, predefined regions of interest, and global average.

An exploratory feature set encompassed spectral peak frequency and bandwidth, aperiodic slope and offset (specparam; fixed model), event-related desynchronization/synchronization, band power ratios ($\theta/\beta$, $\theta/\alpha$, $\alpha/\beta$, $\delta/\alpha$, $\delta/\theta$), hemispheric alpha asymmetry, nonlinear complexity measures (permutation entropy, sample entropy, Lempel-Ziv complexity), and oscillatory burst statistics.

## Feature-Based Machine Learning Architecture
A nested LOSO cross-validation framework ensured true out-of-sample generalization. The primary confirmatory model was ElasticNet regression on spectral power features. Ridge regression was included as a secondary confirmatory model. Random Forest was exploratory due to its sensitivity to correlated predictors in high-dimensional, low-subject neuroimaging settings.

Preprocessing was applied independently within each outer training fold. Feature statistics (medians, means, standard deviations, variance thresholds) were estimated from training subjects only. Missing trials were imputed using training-cohort medians. Features were standardized to zero mean and unit variance; constant features were removed.

fMRI targets were normalized using the Yeo-Johnson power transformation fitted on training targets only. Predictions were inverse-transformed before reporting primary metrics. The ElasticNet objective was:

$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \alpha \rho \|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|_2^2$$

The Ridge objective was:

$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \frac{\alpha}{2}\|\beta\|_2^2$$

Hyperparameters were tuned via 5-fold inner GroupKFold cross-validation restricted to training subjects. Confirmatory inference required at least 20 analyzable subjects or a preregistered precision simulation showing a 95% CI half-width $\leq 0.10$ for primary out-of-sample $R^2$. For ElasticNet, $\alpha \in \{0.001, 0.01, 0.1, 1, 10\}$ and $\rho \in \{0.2, 0.5, 0.8\}$ with 10,000 maximum iterations. For Ridge, $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$. Random Forest used 500 estimators with max depths $\in \{5, 10, 20, \text{None}\}$, min samples split $\in \{2, 5, 10\}$, and min samples leaf $\in \{1, 2, 4\}$.

## Deep Regression Neural Architecture
An exploratory deep regression lane learned directly from continuous, band-limited EEG dynamics. The deep network is designated exploratory unless the subject count justifies confirmatory status.

Continuous EEG was band-pass filtered into target frequency ranges, and instantaneous power was extracted via the Hilbert transform. Resulting tensors were cropped to $3.0$–$10.5$ s, yielding input shape $N \times B \times C \times T$ (trials $\times$ bands $\times$ channels $\times$ time), standardized per channel and band using training-cohort statistics:

$$\tilde{X}_{n,b,c,t} = \frac{X_{n,b,c,t} - \mu_{b,c}}{\sigma_{b,c}^{*}}$$

A custom convolutional neural network enforced neurophysiological constraints by isolating spatial topographies prior to temporal integration. The depthwise spatial convolution used a kernel spanning all channels with independent filter groups per band:

$$H^{(1)}_{n,b,f,t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{c} W^{spat}_{b,f,c}\tilde{X}_{n,b,c,t}\right)\right)$$

A temporal convolution (kernel length 15, 8 filters) then integrated across bands and time:

$$H^{(2)}_{n,f',t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{b,f,u} W^{temp}_{f',b,f,u} H^{(1)}_{n,b,f,t+u}\right)\right)$$

The latent representation was downsampled via average pooling (factor 8) and passed through a dropout-regularized ($p = 0.25$) fully connected regression head:

$$z_n = \mathrm{Pool}_{8}\left(\mathrm{Dropout}(H^{(2)}_n)\right)$$

$$\hat{y}_n = W_2 \, \mathrm{Dropout}\left(\mathrm{ELU}\left(W_1 \, \mathrm{vec}(z_n) + b_1\right)\right) + b_2$$

Optimization used AdamW (learning rate $= 0.001$, weight decay $= 0.0001$) with mean squared error loss over 25 epochs (batch size 32). Twenty percent of training subjects were reserved for early stopping (patience $= 5$ epochs).

## Evaluation Metrics and Statistical Inference
Primary metrics were computed on the active target scale after inverse-transforming predictions. The coefficient of determination used the training-fold target mean as the zero-skill baseline:

$$R_f^2 = 1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}{\sum_{i \in f}(y_i - \bar{y}_{train,f})^2}$$

The primary confirmatory statistic was the subject-weighted mean LOSO $R^2$. A confirmatory cell was positive only when mean $R^2 > 0$ and its permutation $p$ value survived Holm correction. Subject-wise metrics are reported with 95% BCa bootstrap confidence intervals (10,000 resamples). Group-level intervals resampled subjects; within-subject intervals used run/block-level resampling or circular block bootstrap.

Statistical inference used nonparametric permutation testing (5,000 permutations). The permutation null operated on nuisance-residualized targets: within each fold, residual targets were circularly shifted relative to EEG features within each run (minimum shift distance of 5 trials), approximately preserving within-block temporal structure (Winkler et al., 2014). For each permutation, the full prediction pipeline was refit. Within-block random shuffling was evaluated as a sensitivity analysis.

The Holm-corrected confirmatory family comprised 2 targets (NPS, SIIPS1) $\times$ 2 models (ElasticNet, Ridge) $\times$ Level 2 $\times$ 3 frequency presets (alpha, beta, alpha+beta). Gamma, Level 1, Level 3, Random Forest, deep regression, and alternative designs were reported outside this family.

## Nuisance Prediction Reporting
Supplementary models were trained to predict each Level 2 nuisance variable and subjective pain ratings from the same EEG features. Strong prediction of nuisance variables indicated caution that the model may track task structure or acquisition noise. Strong prediction of ratings was interpreted as construct overlap with pain report. The Level 2 framework provides the primary evidence: EEG features must predict stimulus- and acquisition-controlled fMRI pain-signature expression above chance before any pain-relevant interpretation is made.

## Temporal Negative Controls
To assess temporal specificity, models were trained using EEG features from the pre-stimulus window ($-5.0$ to $0.0$ s) and the immediate pre-stimulus window ($-0.2$ to $0.0$ s) to predict post-stimulus NPS expression. If either pre-stimulus model predicted comparably, the model was interpreted as capturing expectancy, block structure, or subject-level state rather than nociceptive processing. Wrong-lag windows were prespecified as ramp-up ($0.0$–$3.0$ s), late ramp-down ($10.5$–$15.0$ s), early-shifted active ($1.0$–$8.5$ s), and late-shifted active ($5.0$–$12.5$ s), evaluated without reselecting model hyperparameters.
