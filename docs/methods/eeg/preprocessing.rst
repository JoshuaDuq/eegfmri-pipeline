EEG Preprocessing
=================

.. raw:: html

   <p class="hero-lede">
     Automated, reproducible EEG preprocessing built on MNE-Python,
     MNE-BIDS-Pipeline, PyPREP, and MNE-ICAlabel. Operates on
     BIDS-formatted data and produces clean, epoched datasets ready for
     feature extraction. Supports task-based and resting-state paradigms.
   </p>

.. grid:: 2
   :gutter: 2
   :class-container: meta-cards

   .. grid-item-card:: Inputs

      BIDS EEG (``.vhdr`` / ``.edf``) · ``channels.tsv`` ·
      ``events.tsv`` (event-related mode)

   .. grid-item-card:: Outputs

      ``*_proc-clean_epo.fif`` · ``*_proc-clean_events.tsv`` (event-related mode) ·
      ICA files · preprocessing stats TSV

   .. grid-item-card:: CLI

      ``eeg-pipeline preprocessing [bad-channels | ica | epochs]``

   .. grid-item-card:: Config

      ``pyprep``, ``ica``, ``epochs`` sections of ``eeg_config.yaml``

.. seealso::

   :doc:`features`
      Feature extraction begins from the clean epochs produced here.

   :doc:`../../user_guide/data_layout`
      Input BIDS layout and events.tsv column requirements.

   :doc:`../../user_guide/configuration`
      Full ``preprocessing``, ``pyprep``, ``ica``, and ``epochs`` key reference.

   :doc:`../../user_guide/cli/preprocessing`
      CLI flags for all preprocessing modes.

.. note::

   Set ``preprocessing.task_is_rest: true`` to create fixed-length segments
   without requiring ``events.tsv`` condition labels.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - :math:`f`
     - Frequency (Hz)
   * - :math:`n_\text{cycles}(f)`
     - Number of Morlet wavelet cycles at frequency :math:`f`
   * - :math:`p_k`
     - ICLabel predicted probability for component class :math:`k`
   * - ``l_freq``, ``h_freq``
     - High-pass and low-pass filter cutoffs (Hz)
   * - :term:`ICA`
     - Independent Component Analysis
   * - :term:`ITPC`
     - Inter-Trial Phase Clustering / Coherence
   * - PTP
     - Peak-to-peak amplitude
   * - :term:`TFR`
     - Time-Frequency Representation

Pipeline Overview
-----------------

BIDS EEG (``.vhdr`` / ``.edf``) enters at step 1; each stage feeds the next until
clean epochs and derivatives are written.

.. list-table::
   :header-rows: 1
   :widths: 5 25 70

   * - Step
     - Transformer
     - Behavior
   * - 1
     - ``NoisyChannels`` (PyPREP)
     - Deviation + correlation; optional RANSAC. Repeated :math:`N` times independently; channels marked bad in a strict majority of runs are retained. Writes ``channels.tsv``
   * - 2
     - ``synchronize_bad_channels_across_runs`` *(opt-in)*
     - Only runs when ``pyprep.bad_channel_sync_policy: "subject_union"``. Default ``"per_run"`` keeps each run's bads independent. When enabled, takes the union of bads across runs per subject and writes it back to every run's ``channels.tsv`` before ICA
   * - 3
     - MNE-BIDS-Pipeline (ICA fit)
     - Subprocess: ``init`` → ``_01`` → ``_04`` → ``_05`` → ``_06a1`` → ``_06a2`` — bandpass, artifact regression, near-rank extended Infomax ICA, and native artifact classification
   * - 4
     - MNE-BIDS-Pipeline / ICLabel review
     - Native ICLabel probabilities and exclusion candidates are written to the component table for manual review before epoching
   * - 5
     - MNE-BIDS-Pipeline (epochs)
     - ``_07`` → ``_08a`` → ``_09`` — make epochs, apply ICA, PTP / autoreject
   * - 6
     - Clean events export
     - Event-related mode: epoch-aligned ``*_proc-clean_events.tsv`` with rejected epochs excluded; resting-state mode skips clean-events export
   * - 7
     - ``collect_preprocessing_stats``
     - Per-subject summary TSV: bad channels, ICA exclusions, epoch rejection counts, per-condition tallies
   * - 8
     - ``custom_tfr`` utility *(optional)*
     - Optional post-preprocessing helper for Morlet wavelets on clean epochs; not part of the default ``preprocessing`` CLI stage sequence

Input Data Requirements
-----------------------

The pipeline expects BIDS-formatted EEG data:

.. code-block:: text

   bids_root/
   ├── dataset_description.json
   └── sub-XXXX/
       └── eeg/
           ├── sub-XXXX_task-<task>_eeg.vhdr
           ├── sub-XXXX_task-<task>_eeg.vmrk
           ├── sub-XXXX_task-<task>_eeg.eeg
           ├── sub-XXXX_task-<task>_channels.tsv
           └── sub-XXXX_task-<task>_events.tsv

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - File
     - Requirement
   * - EEG recording
     - BrainVision (``.vhdr``) by default; configurable via ``pyprep.file_extension``
   * - ``channels.tsv``
     - Must exist alongside each EEG file. Used to identify channel types (``EEG``, ``EOG``, ``ECG``, ``EMG``, ``MISC``) and to read/write bad channel status
   * - ``events.tsv``
     - Required for event-related paradigms. Must contain a ``trial_type`` column. Multi-run designs use per-run files (e.g., ``run-01_events.tsv``) combined automatically
   * - Montage
     - Standard electrode montage (default: ``easycap-M1``) applied when the recording lacks digitized positions

Step 1 — Bad Channel Detection
--------------------------------

.. container:: module-ref

   Module: ``preprocessing/pipeline/preprocess.py`` → ``run_bads_detection()``

Automated detection of noisy channels using PyPREP's ``NoisyChannels`` class,
operating on continuous raw data before ICA or epoching.

Method
~~~~~~

1. Load raw data via ``mne_bids.read_raw_bids()``.
2. Apply standard montage (``easycap-M1`` by default) when digitized positions are absent.
3. Optional low-pass filter at ``h_freq`` Hz (default 100 Hz) on EEG channels only.
4. Optional notch filter at ``notch_freq`` Hz (default 60 Hz) on EEG channels.
5. Optional average re-reference before detection (disabled by default).
6. Repeated bad channel detection (``repeats`` independent runs, default 3). Each run starts from the same ``raw.info["bads"]`` baseline (any pre-existing bads when ``consider_previous_bads`` is true) and increments ``random_state`` by the run index for reproducibility:

   - ``find_bad_by_deviation()`` — channels whose robust z-scored amplitude deviates from the cross-channel median.
   - ``find_bad_by_correlation()`` — channels with low Pearson correlation to neighboring channels.
   - ``find_bad_by_ransac()`` — channels that cannot be predicted from neighbors via RANSAC interpolation (optional; enabled by default via ``pyprep.ransac: true``).
   - After all runs complete, channels flagged in a strict majority of runs (i.e. at least ``floor(repeats / 2) + 1``) are retained as the final PyPREP bad set.

7. Inject custom bad channels from ``custom_bad_dict`` (per-task, per-subject dict).
8. Write results to ``channels.tsv`` (sets ``status = "bad"`` for detected channels).
9. Write per-file CSV log with detected bads, parameters, and any errors.

Supports parallel execution across files via ``joblib.Parallel`` (``n_jobs``).

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``pyprep.ransac``
     - ``true``
     - Enable RANSAC-based detection
   * - ``pyprep.repeats``
     - ``3``
     - Number of independent PyPREP runs; final bads are channels flagged in a strict majority of runs
   * - ``pyprep.average_reref``
     - ``false``
     - Average re-reference before detection
   * - ``pyprep.file_extension``
     - ``".vhdr"``
     - Raw data file extension
   * - ``pyprep.consider_previous_bads``
     - ``true``
     - Retain previously marked bads from ``channels.tsv``
   * - ``pyprep.custom_bad_dict``
     - *(not set in YAML defaults)*
     - Optional manual bad-channel map: ``{task: {subject: [channels]}}``.
       Typically provided via runtime overrides (for example
       ``--set pyprep.custom_bad_dict=...``)
   * - ``preprocessing.h_freq``
     - ``100``
     - Low-pass cutoff applied before detection (Hz)
   * - ``preprocessing.notch_freq``
     - ``60``
     - Notch filter frequency (Hz)

Step 2 — Bad Channel Synchronization (opt-in)
-----------------------------------------------

.. container:: module-ref

   Module: ``preprocessing/pipeline/preprocess.py`` → ``synchronize_bad_channels_across_runs()``

For multi-run paradigms, bad channels detected in any run can optionally be
propagated to all runs of the same subject so that ICA fitting sees a
consistent channel set. This step is gated by ``pyprep.bad_channel_sync_policy``
and is **disabled by default**.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Policy
     - Behavior
   * - ``"per_run"`` *(default)*
     - No synchronization; each run's PyPREP-detected bads remain independent.
   * - ``"subject_union"``
     - Take the union of bads across runs per subject and write it back to every run's ``channels.tsv``.

When ``"subject_union"`` is selected, the runtime sequence is:

1. For each subject, glob all ``channels.tsv`` files matching the task.
2. Compute the union of all channels marked ``status == "bad"`` across runs.
3. Reset all channels to ``status = "good"`` and re-mark only the unified union as ``"bad"`` in every run's ``channels.tsv``.

MNE-BIDS-Pipeline reads ``channels.tsv`` to determine which channels to exclude
from ICA fitting and interpolation. Inconsistent bad sets across runs can
produce incompatible ICA decompositions, which is why ``"subject_union"`` is
recommended for multi-run task designs that share the same channel context.

Step 3 — ICA Fitting
---------------------

.. container:: module-ref

   Module: ``preprocessing/pipeline/preprocess.py`` + ``pipelines/preprocessing.py`` → ``_run_ica_fitting()``

ICA fitting is delegated to MNE-BIDS-Pipeline via subprocess. A temporary Python
config file is generated from the YAML configuration and passed to the pipeline runner.

MNE-BIDS-Pipeline Steps
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Step
     - Description
   * - ``init``
     - Initialize pipeline; validate BIDS dataset
   * - ``preprocessing/_01_data_quality``
     - Data quality assessment; Maxwell filtering (if applicable)
   * - ``preprocessing/_04_frequency_filter``
     - Bandpass filter: ``l_freq``–``h_freq`` (default 0.1–100 Hz)
   * - ``preprocessing/_05_regress_artifact``
     - Regress out artifact signals (EOG/ECG regression)
   * - ``preprocessing/_06a1_fit_ica``
     - Fit ICA decomposition
   * - ``preprocessing/_06a2_find_ica_artifacts``
     - MNE built-in EOG/ECG artifact detection (only when ``use_icalabel = false``)

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``ica.algorithm``
     - ``"extended_infomax"``
     - ICA algorithm. Options: ``"extended_infomax"``, ``"picard"``, ``"fastica"``
   * - ``ica.n_components``
     - ``null``
     - Near-rank decomposition; float = variance explained; int = exact count
   * - ``ica.l_freq``
     - ``1.0``
     - High-pass cutoff for ICA fitting epochs (Hz)
   * - ``preprocessing.l_freq``
     - ``0.1``
     - Main bandpass high-pass cutoff (Hz)
   * - ``preprocessing.h_freq``
     - ``100``
     - Main bandpass low-pass cutoff (Hz)
   * - ``preprocessing.resample_freq``
     - ``500``
     - Resampling frequency (Hz); ``null`` to skip
   * - ``eeg.reference``
     - ``"average"``
     - EEG reference

.. note::

   ``"extended_infomax"`` is required when using ICLabel (Step 4), because the ICLabel
   classifier was trained on extended infomax decompositions.

Step 4 — ICA Component Labeling
---------------------------------

MNE-BIDS-Pipeline's native MNE-ICLabel integration classifies components and
writes the standard ``*_proc-ica_components.tsv`` table. Epoch creation remains
blocked until that table has been reviewed and
``ica.manual_review_complete: true`` has been set.

Method
~~~~~~

1. Fit ICA on average-referenced, 1–100 Hz data.
2. Assign a probability to every ICLabel class for every component.
3. Flag a component when a non-retained class reaches its configured exclusion
   threshold (default 0.8) and neither retained class reaches MNE-BIDS-Pipeline's
   class threshold.
4. Add Analyzer-marker CTPS scores as diagnostic columns without changing
   component status.
5. Review and edit ``*_proc-ica_components.tsv``.
6. Set ``ica.manual_review_complete: true`` and run ``epochs`` to apply the
   reviewed exclusions.

Direct ECG and Cardiac ICA Review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``ica.cardiac_review.enabled`` is true, R peaks are detected directly from
the configured ECG channel with MNE's ECG detector. This diagnostic therefore
remains available when BrainVision Analyzer did not preserve or detect every R
marker. Each run shows a representative ECG segment with detected peaks,
beat-to-beat heart rate, and R-locked EEG global field power before versus after
the current provisional ICA exclusions. All runs remain visible and contribute
to the component display; the pipeline does not assign a run-quality grade.

The report then presents one carousel slide per standard ICA component. Each
slide combines the component topography, run-level R-locked source averages,
normalized ECG timing, and MNE run-resolved ECG-correlation and CTPS outputs. The
median curve and 16th–84th
percentile band summarize variation between runs rather than treating individual
heartbeats as independent subjects. CTPS uses MNE's sampling-aware automatic
threshold by default. MNE flags are displayed without a pipeline-specific score,
classification, ranking, or recommendation and never modify ``ICA.exclude`` or
the component TSV status. The slide also states the component's current ICA status
and exclusion reason. The evidence is placed before the frequency-specific
dossiers so it can inform manual review.

ICA Component Review and Exploratory Band Appendix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``ica.band_specific_report.enabled`` is true, the pipeline additionally
adds authoritative component-review dossiers before MNE's standard ICA component
section. Each frequency-band section is a carousel ordered by the standard ICA
component number. A slide keeps that component's topography, band-limited Welch
spectrum, grand-average DPSS multitaper TFR, and all configured condition TFRs
together. ICLabel probability and the current automatic exclusion status remain
visible in the slide title and caption.

The report also retains independent extended-infomax decompositions fitted to
1–8, 8–13, 13–30, 30–100, and 1–30 Hz copies of the ICA-fitting epochs as an
exploratory appendix. Component numbers in these independent decompositions do
not identify the same source across bands and never control artifact removal.

The TFR reproduces the FieldTrip ``mtmconvol`` convention with 1 Hz frequency
spacing, 100 ms time spacing, a −5 to −0.01 second dB baseline, and band-specific
windows and half-bandwidth smoothing: 3 seconds / ±1 Hz for 1–8 Hz, 2 seconds /
±1.5 Hz for alpha, 2 seconds / ±2.5 Hz for beta, and 1 second / ±5 Hz for gamma.
Figures use baseline-relative dB and symmetric color scaling. Condition A and B
share one scale so their amplitudes can be compared directly; the A-minus-B
difference uses its own zero-centred symmetric scale.

Configured ``ica.band_specific_report.comparisons`` are evaluated twice. During
``ica``, step ``_07_make_epochs`` creates pre-ICA task epochs solely for
provisional component-review TFRs; ICA is not applied and trials are not rejected.
During ``epochs``, original MNE event-selection identities map the retained clean
epochs back to rows of the saved pre-ICA task epochs, and the aligned
``proc-clean_events.tsv`` supplies the requested metadata column. Each comparison
displays group A, group B, and baseline-normalized group A minus group B. Missing
columns, values, or trials are hard errors. Provisional dossiers are replaced in
place by the finalized retained-epoch dossiers.

The independent appendix decompositions are diagnostic only. A spatial
topography is not itself frequency-filtered; it is the mixing pattern estimated
from band-filtered data. ICLabel was developed for conventional broadband
decompositions, so its narrow-band results are explicitly labeled exploratory
and never modify the authoritative 1–100 Hz component table or the components
applied during epoch creation.

ICLabel Classes
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Class
     - Description
     - Default action
   * - Brain
     - Neural activity
     - Retained
   * - Muscle
     - EMG artifact
     - Excluded if :math:`p > 0.8`
   * - Eye
     - Ocular artifact (blinks, saccades)
     - Excluded if :math:`p > 0.8`
   * - Heart
     - Cardiac artifact
     - Excluded if :math:`p > 0.8`
   * - Line Noise
     - Power line interference
     - Excluded if :math:`p > 0.8`
   * - Channel Noise
     - Electrode/hardware noise
     - Excluded if :math:`p > 0.8`
   * - Other
     - Unclassifiable / mixed
     - Retained

Step 5 — Epoch Creation and Artifact Rejection
-----------------------------------------------

.. container:: module-ref

   Module: ``preprocessing/pipeline/preprocess.py`` + ``pipelines/preprocessing.py`` → ``_run_epoch_creation()``

MNE-BIDS-Pipeline Steps
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Step
     - Description
   * - ``preprocessing/_07_make_epochs``
     - Segment continuous data into epochs around events
   * - ``preprocessing/_08a_apply_ica``
     - Subtract excluded ICA components from epoched data
   * - ``preprocessing/_09_ptp_reject``
     - Reject remaining bad epochs via PTP threshold or autoreject

Epoch Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``epochs.tmin``
     - ``-7.0``
     - Epoch start time relative to event onset (s)
   * - ``epochs.tmax``
     - ``15.0``
     - Epoch end time relative to event onset (s)
   * - ``epochs.baseline``
     - ``[-0.2, 0.0]``
     - Baseline correction window (s); ``null`` to skip
   * - ``epochs.reject``
     - ``"autoreject_local"``
     - Rejection method
   * - ``epochs.autoreject_n_interpolate``
     - ``[4, 8, 16]``
     - Channels to interpolate per trial (autoreject cross-validation)

Rejection Methods
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``"autoreject_local"``
     - Per-channel, per-trial adaptive thresholding. Interpolates a configurable number of bad channels per trial before rejecting epochs that remain too noisy. Recommended for long epochs.
   * - ``"autoreject_global"``
     - Global PTP threshold estimated by autoreject across all channels.
   * - ``{"eeg": <value>}``
     - Fixed PTP threshold in volts (e.g., ``{"eeg": 150e-6}``).
   * - ``null`` / ``"none"``
     - No epoch rejection.

Step 6 — Clean Events Export
------------------------------

In event-related mode, after epoch rejection, a clean events table is written
to derivatives containing only events for kept (non-rejected) epochs.
The canonical ``trial_id`` column is the only supported alignment contract for
downstream trialwise artifacts.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``preprocessing.write_clean_events``
     - ``true``
     - Enable clean events export
   * - ``preprocessing.clean_events_overwrite``
     - ``true``
     - Overwrite existing clean events files
   * - ``preprocessing.clean_events_strict``
     - ``true``
     - Raise error if clean events cannot be written

Step 7 — Preprocessing Statistics
-----------------------------------

.. container:: module-ref

   Module: ``preprocessing/pipeline/stats.py`` → ``collect_preprocessing_stats()``

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Metric
     - Description
   * - ``n_bad_channels``
     - Bad channels detected (from ``*_bads.tsv``)
   * - ``n_bad_ica``
     - Excluded ICA components (from ``*_components.tsv``)
   * - ``total_clean_epochs``
     - Epochs surviving rejection
   * - ``n_removed_epochs``
     - Epochs rejected
   * - ``boundary_n_removed_epochs``
     - Epochs rejected due to ``BAD boundary`` annotations
   * - ``<condition>_total_clean_epochs``
     - Per-condition epoch counts

Step 8 — Time-Frequency Representation Utility (Optional)
----------------------------------------------------------

.. container:: module-ref

   Module: ``preprocessing/pipeline/tfr.py`` → ``custom_tfr()``

Morlet wavelet :term:`TFR` decomposition on clean epochs.
This is a utility in ``preprocessing/pipeline/tfr.py`` and is not executed by
default in the ``preprocessing`` CLI modes.

Frequencies are configurable (default 1–99 Hz). Cycles adapt as
:math:`n_\text{cycles}(f) = f / 3` by default (higher frequency → better
frequency resolution).

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Key
     - Default
     - Description
   * - ``custom_tfr_freqs``
     - ``np.arange(1, 100, 1)``
     - Frequency vector (Hz)
   * - ``custom_tfr_n_cycles``
     - ``freqs / 3.0``
     - Cycles per frequency (adaptive)
   * - ``custom_tfr_decim``
     - ``1``
     - Decimation factor
   * - ``custom_tfr_return_itc``
     - ``true``
     - Compute inter-trial coherence
   * - ``custom_tfr_average``
     - ``true``
     - Average across epochs (vs. single-trial)

Execution Modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Steps executed
   * - ``bad-channels``
     - Bad channel detection only
   * - ``ica``
     - ICA fitting + ICA labeling only
   * - ``epochs``
     - Epoch creation + statistics (requires reviewed ICA and explicit acknowledgement)

Output Structure
----------------

.. code-block:: text

   derivatives/preprocessed/eeg/
   ├── sub-XXXX/
   │   └── eeg/
   │       ├── sub-XXXX_proc-icafit_ica.fif
   │       ├── sub-XXXX_proc-icafit_epo.fif
   │       ├── sub-XXXX_proc-ica_ica.fif
   │       ├── sub-XXXX_proc-ica_components.tsv
   │       ├── sub-XXXX_task-<task>_proc-clean_epo.fif
   │       ├── sub-XXXX_task-<task>_proc-clean_events.tsv     # event-related mode
   │       ├── sub-XXXX_task-<task>_bads.tsv
   │       ├── sub-XXXX_task-<task>_power_epo-tfr.h5            # (optional, single-trial; average=False)
   │       ├── sub-XXXX_task-<task>_itc_epo-tfr.h5              # (optional, single-trial; average=False)
   │       ├── sub-XXXX_task-<task>_power_<cond>_avg-tfr.h5     # (optional, per-condition average when average=False and return_average=True)
   │       ├── sub-XXXX_task-<task>_itc_<cond>_avg-tfr.h5       # (optional, per-condition ITC when average=False and return_average=True)
   │       ├── sub-XXXX_task-<task>_power+<cond>_avg-tfr.h5     # (optional, per-condition average when average=True)
   │       └── sub-XXXX_task-<task>_itc+<cond>_avg-tfr.h5       # (optional, per-condition ITC when average=True)
   ├── pyprep_task_<task>_log.csv
   ├── task_<task>_preprocessing_stats.tsv
   └── task_<task>_preprocessing_stats_desc.tsv
