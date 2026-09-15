# Cryocooler Comb Validation and Signal-Recovery Report

Date: 2026-08-24

## Final decision

The approximately 1.2 Hz comb in the MRI EEG is physically consistent with the MRI
cryocooler. The Siemens magnet documentation specifies a cold-head speed of 72 rpm on
60 Hz power: exactly 1.2 revolutions per second. Its motor, Scotch-yoke displacer,
rotary helium valve, and pressure cycle are non-sinusoidal, so integer harmonics are
expected. Mechanical electrode/lead motion in the static magnetic field provides a
plausible voltage-generation mechanism.

The original conclusion that only gamma was affected was false. Its 1-4 Hz local
baseline included neighboring 1.2 Hz teeth and hid lower harmonics. An inter-tooth
baseline and independent phase-stability test found MRI-specific evidence in alpha,
beta, and gamma. Delta and theta remain unresolved; the absence of selected low teeth
is not evidence that those bands are artifact-free.

The production decision is:

- preserve the pre-decomb BIDS EEG as the authoritative waveform for thermal ERPs and
  continuous morphology;
- write a separate `_desc-cryocoolerregressed` derivative;
- require the explicit 1.2 Hz hardware fundamental, with no `1/TR` fallback;
- regress only frozen leave-one-subject-out-supported harmonics using phase tracking
  and cycle cross-fitting;
- declare plus/minus 0.1 Hz around every modeled tooth unavailable for spectral and
  connectivity inference;
- keep gradient correction, mains handling, and cryocooler correction as distinct
  stages; and
- exclude broad automatic `spectrum_fit` subtraction and residual FIR notching from
  the production path.

This is targeted artifact suppression, not exact reconstruction of neural activity at
a modeled tooth. Connectivity remains a sensitivity analysis.

## Cohorts and marker semantics

The complete evaluation included:

- 90 MRI recordings: 15 participants times 6 runs;
- 24 artifact-free MRI-simulator recordings: 4 participants times 6 runs; and
- exactly 11 thermode events in every run.

`Trig_therm` is the thermode-stimulation marker. `Stimulus/S 1-3` must never be used as
a substitute. The erroneous generic labels came from bypassing BIDS event semantics in
an early discarded benchmark. Final benchmarks and production use the BIDS-aware reader
and require exactly 11 descriptions beginning with `Trig_therm`; stimulation events do
not enter artifact fitting.

MRI source:

`/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg`

Simulator source:

`/Volumes/KINGSTON/Données_Lorie_Ève/sourcedata/sub-*/eeg/original_untrimmed_5khz`

## Corrected harmonic evidence

The corrected screen analyzed all 79 possible 1.2 Hz harmonics below 100 Hz in all 90
MRI and 24 simulator recordings. Each tooth was measured against bins 0.25-0.55 Hz away,
which lie between adjacent comb teeth. A separate statistic measured stability of the
complex harmonic phase across 10-second windows. Inference used participant-level run
medians, exact one-sided Mann-Whitney tests, and Benjamini-Hochberg correction over all
79 harmonics.

Representative lower-frequency evidence included:

| Harmonic | Frequency | Evidence | MRI median | Simulator median | FDR q |
| ---: | ---: | --- | ---: | ---: | ---: |
| 8 | 9.6 Hz | inter-tooth prominence | 0.84 dB | 0.25 dB | 0.011 |
| 11 | 13.2 Hz | weighted phase stability | 0.306 | 0.218 | 0.010 |
| 18 | 21.6 Hz | inter-tooth prominence | 1.01 dB | 0.21 dB | 0.0006 |
| 22 | 26.4 Hz | inter-tooth prominence | 1.78 dB | 0.26 dB | 0.008 |
| 24 | 28.8 Hz | inter-tooth prominence | 4.21 dB | 0.24 dB | 0.002 |
| 33 | 39.6 Hz | inter-tooth prominence | 4.39 dB | 0.22 dB | 0.0006 |
| 41 | 49.2 Hz | inter-tooth prominence | 2.80 dB | 0.08 dB | 0.001 |

The frozen participant models contain 49-53 harmonics, from harmonic 8 (9.6 Hz) through
harmonic 79 (94.8 Hz), with participant-specific LOSO support. The evaluated participant
is removed from both MRI and simulator environments before selection. Harmonics 43, 45,
49, and 51 are the high-SNR phase anchors.

## Production method

The cleaner performs the following operations:

1. Build the nominal phase from the explicit 1.2 Hz hardware rate.
2. Estimate a shared drifting fundamental phase from the four anchor teeth in
   10-second windows, using the median across anchors and EEG channels.
3. Interpolate and edge-extrapolate the phase trajectory.
4. In 60-second regression windows, detrend each complete cryocooler cycle.
5. Estimate complex coefficients in seven cycle folds; a cycle is predicted only from
   coefficients learned from other folds.
6. Apply positive reliability shrinkage so unsupported coefficients collapse toward
   zero rather than fitting noise.
7. Subtract only manifest-declared harmonics from good EEG channels.

Bad EEG, EOG, ECG, miscellaneous channels, channel order, sample count, sampling rate,
and annotations are preserved. Every derivative is replayed from the source and must
match exactly after BrainVision float32 quantization.

Production files:

- implementation: `src/decomb/cryocooler.py`;
- BIDS boundary: `src/decomb/cryocooler_pipeline.py`;
- frozen models: `src/decomb/models/cryocooler_harmonics.tsv`;
- selection provenance: `src/decomb/models/cryocooler_harmonics.json`;
- derivative manifest: `cryocooler_regression_manifest.tsv`; and
- verification: `cryocooler_regression_verification.tsv`.

## Full artifact-free simulator comparison

All three transforms were replayed on all 24 simulator recordings using the same scoring
functions. Because the simulator has the same task and thermal sequence but no magnet or
cryocooler, its unmodified EEG is the known target.

| Metric | Old pipeline | Restricted high candidate | Phase-tracked method |
| --- | ---: | ---: | ---: |
| Continuous RRMSE | 13.059% | 1.125% | **0.568%** |
| Correlation with source | 0.991320 | 0.999932 | **0.999983** |
| Thermal-ERP RRMSE | 10.898% | 1.556% | **0.831%** |
| Induced-band change | 7.663 dB | 0.0111 dB | **0.0054 dB** |
| Band-coherence change | 0.00371 | 0.00113 | **0.00021** |

The phase-tracked method beat the restricted candidate in 22/24 runs for continuous
morphology and correlation, 21/24 for thermal ERP and induced power, and 23/24 for
coherence. It won all four participant medians for every tested outcome.

## Independent lower-plus-upper-comb injection

The injected artifact was estimated independently by global least-squares projection of
the matching MRI recording, not by the cleaner under test. Model selection excluded the
test participant in both MRI and simulator environments.

| Result | Median RRMSE from known clean simulator EEG |
| --- | ---: |
| Injected artifact, no cleaning | 1.796% |
| Old pipeline | 11.488% |
| Restricted high candidate | 1.802% |
| Phase-tracked method | **1.134%** |

The phase-tracked method beat no cleaning, the restricted candidate, and the old pipeline
in all 24 recordings and all four participant medians. This reverses the earlier result
from the restricted high-frequency model: once the independently injected lower and upper
teeth are represented, targeted regression improves recovery.

## Full MRI comparison

Change from the pre-decomb MRI source is not ground-truth error, but it quantifies the
tradeoff between suppression and waveform preservation.

| Metric | Old pipeline | Restricted candidate | Phase-tracked method |
| --- | ---: | ---: | ---: |
| Continuous change | 4.952% | **2.284%** | 2.577% |
| Correlation with source | 0.997943 | **0.999537** | 0.999353 |
| Thermal-ERP change | 9.617% | **4.626%** | 5.084% |
| Induced-band change | 1.927 dB | **0.132 dB** | 0.466 dB |
| Coherence change | 0.0252 | **0.0122** | 0.0204 |
| Reduction on new harmonic grid | 4.244 dB | 0.136 dB | **2.990 dB** |
| Reduction on lower harmonic grid | 0.0002 dB | 0.0008 dB | **0.446 dB** |

The phase-tracked method changed the MRI source slightly more than the restricted model
because it treats the newly validated lower teeth. It remained better than the old
pipeline in all 90 recordings for morphology, correlation, ERP, and induced power, and in
72/90 for coherence. The simulator and independent-injection tests provide the stronger
ground-truth evidence and favor the phase-tracked method.

## Retained bandwidth

Availability is geometric bandwidth outside the fixed plus/minus 0.1 Hz masks, not signal
power. It is derived from the final frozen model, not from the withdrawn old table. The
width benchmark tested plus/minus 0.05, 0.10, 0.15, and 0.20 Hz on all 4,548 modeled
teeth in all 90 MRI recordings and on all 24 independent simulator injections. The
largest tracked center displacement was 0.0756 Hz, so 0.10 Hz contained every tracked
center. The narrower mask produced only very small retained-bin errors in the injection
control (maximum 0.061 dB for band power and 0.0011 for coherence). Plus/minus 0.05 Hz
was rejected because it failed to contain the tracked center in 47/90 recordings.

| Recording availability | Delta | Theta | Alpha | Beta | Gamma |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% common | 100.0% | 100.0% | 87.8% | 95.3% | 84.8% |
| At least 95% | 100.0% | 100.0% | 87.8% | 95.3% | 84.8% |
| At least 90% | 100.0% | 100.0% | 87.8% | 95.3% | 85.6% |
| Mean per recording | 100.0% | 100.0% | 94.6% | 96.2% | 86.8% |

The 95% and 100% masks are identical because one participant contributes 6/90 recordings
(6.67%): any participant-specific exclusion lowers availability below 95%. Delta and
theta show 100% retained only because no low-frequency tooth met the frozen selection
criterion; those bands remain an unresolved sensitivity limit.

## Outcome recommendations

- Thermal ERPs: use the source as primary and the targeted derivative as sensitivity.
  Thermal onsets were not phase-concentrated at 1.2 Hz.
- Continuous morphology: keep the source authoritative; use the derivative for
  line-suppressed inspection or sensitivity.
- Induced power: analyze the source and/or derivative with the same declared tooth masks,
  normalize by retained bandwidth, and report sensitivity to mask width.
- Connectivity: require source-versus-derivative sensitivity and exclude all modeled
  intervals. Imaginary coherence or weighted phase-lag index may be secondary checks, not
  guarantees against mechanical coupling.
- Never use frequency interpolation as observed data for confirmatory inference. It is
  acceptable only for visualization.

If `observed(f) = neural(f) + cryocooler(f)`, EEG alone cannot uniquely identify both
terms when they occupy the same channel, time, phase, and frequency subspace. Exact neural
power, phase, or connectivity at a modeled tooth cannot be claimed as recovered.

## Evidence files

- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_intertooth_screen.tsv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_intertooth_summary.tsv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_phase_tracked_120s.tsv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_full_simulator_outcomes.tsv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_full_mri_outcomes.tsv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_phase_tracked_injection.tsv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_mask_width_benchmark.csv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_mask_width_by_window.csv`
- `/Users/joduq24/Desktop/decomb/outputs/cryocooler_mask_width_injection.csv`

## Equipment documents

- Siemens, *Functional Description - Magnet*, pages 47-48:
  `/Users/joduq24/Downloads/973558028-Functional-Description-Magnet.pdf`
- Sumitomo, *SRDK-305 Series Cryocooler Operation Manual*, pages 18-19 and 38:
  `/Users/joduq24/Downloads/597248323-ColdHead.pdf`

## References

- Rothlübbers S, et al. Characterisation and reduction of the EEG artefact caused by the
  helium cooling pump in the MR environment. *Brain Topography*. 2015;28:208-220.
  https://doi.org/10.1007/s10548-014-0408-0
- Krishnaswamy P, et al. Reference-free removal of EEG-fMRI ballistocardiogram artifacts
  with harmonic regression. *NeuroImage*. 2016;128:398-412.
  https://doi.org/10.1016/j.neuroimage.2015.06.088
- de Cheveigné A, Nelken I. Filters: when, why, and how (not) to use them. *Neuron*.
  2019;102:280-293. https://doi.org/10.1016/j.neuron.2019.02.039
- MNE-Python filtering documentation:
  https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html
