Plotting
========

.. note::

   The plotting pipeline is **still under active development**. The plot
   catalog, CLI surface, and default outputs may change between releases.

Render visualization suites driven by the pipeline's curated plot catalog.

.. code-block:: bash

   eeg-pipeline plotting [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Mode
     - Description
   * - ``visualize``
     - Render the configured plot suites for a subject or at group level.
   * - ``tfr``
     - Time-frequency representation plots for a single subject.

Plot Groups
-----------

Select groups with ``--groups``. Use ``--all-plots`` to render everything.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Group
     - Typical content
   * - ``power``
     - Band power summaries, topographies, time-courses
   * - ``connectivity``
     - Connectivity matrices, graph metric summaries
   * - ``aperiodic``
     - 1/f slope / offset distributions and fit QC
   * - ``phase``
     - Phase-based summaries (ITPC, PAC)
   * - ``erds``
     - ERDS time-course and percent-change figures
   * - ``complexity``
     - Complexity feature distributions (LZC, PE, MSE)
   * - ``spectral``
     - Spectral edge and peak frequency summaries
   * - ``ratios``
     - Band-ratio distributions and condition comparisons
   * - ``asymmetry``
     - Hemispheric asymmetry indices
   * - ``microstates``
     - Microstate statistics and transition matrices
   * - ``bursts``
     - Burst rate, duration, and amplitude summaries
   * - ``erp``
     - ERP component amplitude and latency summaries
   * - ``tfr``
     - Time-frequency power maps
   * - ``behavior``
     - Behavioral summary and model-fit plots

Key Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Option
     - Description
     - Default
   * - ``--all-plots``
     - Render every available plot group
     - disabled
   * - ``--groups``
     - Space-separated list of plot groups to render
     - all configured groups
   * - ``--formats``
     - Output file formats: ``png``, ``svg``, ``pdf`` (space-separated)
     - ``png``
   * - ``--analysis-scope``
     - ``subject`` (per-subject plots) or ``group`` (cross-subject aggregates)
     - ``subject``
   * - ``--dpi``
     - Figure resolution in dots per inch
     - config ``plotting.defaults.dpi``

Examples
--------

.. tab-set::

   .. tab-item:: Visualize

      .. code-block:: bash

         # All available plots for one subject
         eeg-pipeline plotting visualize --subject 0001 --all-plots

         # Specific plot groups only
         eeg-pipeline plotting visualize --subject 0001 \
           --groups power behavior connectivity

         # Export in multiple formats
         eeg-pipeline plotting visualize --subject 0001 \
           --all-plots --formats png svg pdf

         # Group-level aggregate plots across subjects
         eeg-pipeline plotting visualize \
           --subject 0001 --subject 0002 --subject 0003 \
           --analysis-scope group

         # All subjects, group scope
         eeg-pipeline plotting visualize --all-subjects \
           --analysis-scope group

   .. tab-item:: TFR

      .. code-block:: bash

         # Time-frequency representation for one subject
         eeg-pipeline plotting tfr --subject 0001

         # Custom output format
         eeg-pipeline plotting tfr --subject 0001 --formats svg

.. seealso::

   :doc:`../subject_selection`
      Shared ``--subject``, ``--all-subjects``, ``--task``, and ``--set`` flags.

   :doc:`../output_formats`
      Where plot files are written and the ``--formats`` option.
