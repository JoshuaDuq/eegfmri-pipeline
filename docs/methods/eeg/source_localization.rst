EEG Source Localization
=======================

Two source localization paths are available. Both use :term:`MNE-Python` as
the forward/inverse backend and write results to the :term:`Parquet` feature
table format.

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :stub-columns: 1

   * - Inputs
     - ``*_proc-clean_epo.fif``; optionally FreeSurfer subject + BEM + fMRI contrast map
   * - Outputs
     - Source-band power and envelope features per trial/window (Parquet)
   * - CLI
     - ``eeg-pipeline features compute --categories sourcelocalization``
   * - Config
     - ``feature_engineering.sourcelocalization`` section of ``eeg_config.yaml``

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Path
     - Requires
     - Use case
   * - **EEG-only**
     - Cleaned epochs, standard montage
     - Quick validation, template-level analysis
   * - **fMRI-constrained**
     - FreeSurfer subject, :term:`BEM`, trans, fMRI stats map
     - Research-grade, subject-specific, fMRI-guided

What Is Computed
----------------

- Per-trial/window: source-band power and source-band envelope features.
- Optional: subject-level source contrast tables (``sourcecontrast``) for
  condition A vs B — only computed when ``--source-fmri-contrast-enabled``
  is passed.
- fMRI-informed outputs: ``cluster`` space (subject-specific fMRI clusters),
  ``atlas`` space (subject-space ``aparc+aseg`` labels), or ``dual`` (both).

.. warning::

   ``feature_engineering.sourcelocalization.fmri.time_windows`` is
   explicitly unsupported and raises ``ValueError`` if set. Remove this
   key from your config before running fMRI-constrained localization.

Other constraints:

- fMRI-constrained eLORETA requires ``--source-loose 1.0``.
- Use atlas-harmonized source outputs for inferential cross-subject statistics.

Path 1: EEG-Only (Template-Based)
----------------------------------

Uses the ``fsaverage`` template head model. No subject MRI or fMRI required.
Template fallback is **opt-in** via ``feature_engineering.sourcelocalization.allow_template_fallback: true``.

.. code-block:: bash

   eeg-pipeline features compute \
     --subject 0001 \
     --categories sourcelocalization \
     --spatial roi global

   # Custom method and spacing
   eeg-pipeline features compute \
     --subject 0001 \
     --categories sourcelocalization \
     --source-method lcmv \
     --source-spacing oct6 \
     --source-reg 0.05

Path 2: fMRI-Constrained
--------------------------

Uses subject-specific MRI + fMRI statistical map to constrain the source space.
Requires FreeSurfer ``recon-all``, BEM model/solution, coregistration transform,
and an fMRI stats map in subject MRI space.

Automated TUI Workflow (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TUI wizard can auto-generate the BEM model, BEM solution, and coregistration
transform via Docker. Navigate to **Features → Source Localization**, set **Mode**
to ``fMRI-informed``, enable **Create Trans**, **Create BEM Model**, and
**Create BEM Solution**, then provide the FreeSurfer subject name and fMRI stats map.

CLI equivalent:

.. code-block:: bash

   eeg-pipeline features compute \
     --subject 0001 \
     --categories sourcelocalization \
     --source-fmri \
     --source-fmri-stats-map /path/to/sub-0001_pain_vs_baseline_zmap.nii.gz \
     --source-subject sub-0001 \
     --source-subjects-dir /path/to/freesurfer \
     --source-create-trans \
     --source-create-bem-model \
     --source-create-bem-solution \
     --source-fs-license /path/to/license_freesurfer.txt

Manual Docker Workflow
~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Build FreeSurfer + MNE Docker image**

.. code-block:: bash

   docker build --platform linux/amd64 -t freesurfer-mne:7.4.1 \
     -f eeg_pipeline/docker_setup/Dockerfile.freesurfer-mne .

**Step 2: Run FreeSurfer recon-all (1–3 hours)**

.. code-block:: bash

   docker run --rm \
     -v /path/to/project:/data \
     -v $SUBJECTS_DIR:/subjects \
     -v $FS_LICENSE:/usr/local/freesurfer/.license \
     --platform linux/amd64 freesurfer-mne:7.4.1 \
     recon-all \
       -subjid sub-0001 \
       -i /data/anat/sub-0001_T1w.nii.gz \
       -all -sd /subjects

**Step 3: Generate BEM model and solution**

.. code-block:: bash

   docker run --rm \
     -v $SUBJECTS_DIR:/subjects \
     -v $FS_LICENSE:/usr/local/freesurfer/.license \
     --platform linux/amd64 freesurfer-mne:7.4.1 \
     bash -lc "
       source \$FREESURFER_HOME/SetUpFreeSurfer.sh
       mne watershed_bem --subject sub-0001 --overwrite
       python -c 'import mne; bem_model = mne.make_bem_model(\"sub-0001\", ico=4, subjects_dir=\"/subjects\"); \
         bem_sol = mne.make_bem_solution(bem_model); \
         mne.write_bem_solution(\"/subjects/sub-0001/bem/sub-0001-bem-sol.fif\", bem_sol, overwrite=True)'
     "

**Step 4: Create coregistration transform**

.. code-block:: python

   import mne
   mne.gui.coregistration(
       subject="sub-0001",
       subjects_dir="/path/to/freesurfer",
       inst=raw,
   )

Save the transform as ``sub-0001-trans.fif``.

**Step 5: Generate fMRI statistical map**

*Option A — Automated contrast builder:*

.. code-block:: bash

   eeg-pipeline features compute \
     --subject 0001 --categories sourcelocalization \
     --source-fmri --source-fmri-contrast-enabled \
     --source-fmri-cond-a-column trial_type --source-fmri-cond-a-value stimulation \
     --source-fmri-cond-b-column trial_type --source-fmri-cond-b-value fixation_rest \
     --source-fmri-contrast-name stim_vs_rest \
     --source-fmri-resample-to-fs

*Option B — Pre-computed stats map:*

Provide a 3D NIfTI aligned to the FreeSurfer subject space via
``--source-fmri-stats-map /path/to/zmap.nii.gz``.

**Step 6: Run fMRI-constrained source localization**

.. code-block:: bash

   eeg-pipeline features compute \
     --subject 0001 --categories sourcelocalization \
     --source-fmri \
     --source-fmri-stats-map /path/to/sub-0001_zmap.nii.gz \
     --source-fmri-threshold 3.1 \
     --source-fmri-tail pos \
     --source-subject sub-0001 \
     --source-subjects-dir /path/to/freesurfer \
     --source-trans /path/to/sub-0001-trans.fif \
     --source-bem /path/to/sub-0001-bem-sol.fif \
     --source-method lcmv \
     --spatial global

CLI Flags Reference
--------------------

Source localization flags:

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Flag
     - Description
     - Default
   * - ``--source-method``
     - Inverse method: ``lcmv`` or ``eloreta``
     - ``lcmv``
   * - ``--source-spacing``
     - Source space spacing: ``oct5``, ``oct6``, ``ico4``, ``ico5``
     - ``oct6``
   * - ``--source-reg``
     - LCMV regularization parameter
     - ``0.05``
   * - ``--source-snr``
     - eLORETA assumed SNR
     - ``3.0``
   * - ``--source-loose``
     - eLORETA loose orientation constraint (0–1)
     - ``0.2``
   * - ``--source-depth``
     - eLORETA depth weighting (0–1)
     - ``0.8``
   * - ``--source-parc``
     - Parcellation: ``aparc``, ``aparc.a2009s``, ``HCPMMP1``
     - ``aparc``
   * - ``--source-subject``
     - FreeSurfer subject name
     - ``sub-{subject}``
   * - ``--source-subjects-dir``
     - FreeSurfer ``SUBJECTS_DIR`` path
     - (none)
   * - ``--source-trans``
     - EEG ↔ MRI coregistration transform ``.fif``
     - (none)
   * - ``--source-bem``
     - BEM solution ``.fif``
     - (none)
   * - ``--source-mindist-mm``
     - Minimum distance from sources to inner skull (mm)
     - ``5.0``

fMRI constraint flags:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Flag
     - Description
     - Default
   * - ``--source-fmri``
     - Enable fMRI-constrained source localization
     - ``False``
   * - ``--source-fmri-stats-map``
     - Path to fMRI statistical map NIfTI
     - (none)
   * - ``--source-fmri-threshold``
     - Threshold applied to fMRI stats map
     - ``3.1``
   * - ``--source-fmri-threshold-mode``
     - Thresholding mode: ``z`` (z-score) or ``fdr``
     - ``z``
   * - ``--source-fmri-fdr-q``
     - FDR q-value when threshold mode is ``fdr``
     - ``0.05``
   * - ``--source-fmri-tail``
     - Threshold tail: ``pos`` or ``abs``
     - ``pos``
   * - ``--source-fmri-cluster-min-voxels``
     - Minimum cluster size in voxels after thresholding
     - ``50``
   * - ``--source-fmri-cluster-min-mm3``
     - Minimum cluster volume (mm³); overrides ``--source-fmri-cluster-min-voxels`` when set
     - (none)
   * - ``--source-fmri-max-clusters``
     - Maximum number of clusters kept from fMRI map
     - ``20``
   * - ``--source-fmri-max-voxels-per-cluster``
     - Maximum voxels sampled per cluster
     - ``2000``
   * - ``--source-fmri-max-total-voxels``
     - Maximum total voxels across all clusters
     - ``20000``
   * - ``--source-fmri-output-space``
     - Feature output family: ``cluster``, ``atlas``, or ``dual``
     - ``dual``

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Error
     - Resolution
   * - ``No cleaned epochs found for sub-XXXX``
     - Verify epochs: ``find derivatives/preprocessed -name "*proc-clean_epo.fif"``
   * - ``fMRI constraint enabled but no stats map path provided``
     - Add ``--source-fmri-stats-map /path/to/stats.nii.gz``
   * - ``fMRI stats map threshold produced empty mask``
     - Lower ``--source-fmri-threshold`` (e.g. ``2.5``); check stats map contains positive values
   * - ``Forward model: 0 sources``
     - Check BEM solution validity; verify electrode digitization in ``epochs.info["dig"]``

References
----------

- `MNE-Python source localization <https://mne.tools/stable/auto_tutorials/source-modeling/30_source_localization.html>`_
- `FreeSurfer recon-all <https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all>`_

.. seealso::

   :doc:`preprocessing`
      Produces the clean epochs and electrode montage consumed here.

   :doc:`../../methods/fmri/pipeline`
      First-level GLM contrast maps used to constrain the fMRI-informed path.

   :doc:`../../user_guide/configuration`
      Full ``feature_engineering.sourcelocalization`` key reference.

   :doc:`../../install`
      Docker image build instructions for FreeSurfer + MNE.
