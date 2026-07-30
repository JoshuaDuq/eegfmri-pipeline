# Study support: helpers for running study1 and study2

**Studies:** [`../../study1/`](../../study1/), [`../../study2/`](../../study2/) ·
**Run guide:** [`../../RUN_FULL_STUDIES_README.md`](../../RUN_FULL_STUDIES_README.md)

## Why this folder exists

Study 1 and Study 2 are not one command. They are large batches — a benchmark grid across
partitions, targets and feature specs — that get dispatched to a cluster, come back
partially complete, and need the missing pieces identified and resubmitted.

These scripts are the machinery around that loop. They are not analysis: the analysis lives
in `study1/` and `study2/`. They exist so that running the analysis at cohort scale is
repeatable and so a half-finished sweep can be resumed rather than restarted.

They are kept apart from the workflow folders (`line_comb/`, `cardiac_gaps/`, `conversion/`)
because those describe *stages of the data*, while these describe *how a study gets run*.

## The files

| File | What it contributes |
|---|---|
| `study1_benchmark_cell.py` | Runs one cell of the Study 1 benchmark grid — one partition, target and feature spec. The unit a cluster array job dispatches. |
| `study1_missing_benchmark_cells.py` | Reads what has completed and emits the manifest of cells that have not. This is what makes a partial sweep resumable. |
| `study1_timing_audit.py` | Audits event timing across runs, so a timing fault is found before it is interpreted as an effect. |
| `study2_prepare_source_stage_input.py` | Assembles the inputs the Study 2 source stage expects. |
| `study_subject_qc_summary.py` | One-row-per-subject QC summary across the cohort — the table used to decide who is included. |
| `build_apriori_scoring_mask.py` | Fetches the TemplateFlow brain mask and builds the a-priori scoring mask, so the scoring region is fixed in advance rather than chosen after seeing results. |
| `trillium_run_study1_all_in_one.sh` | The all-in-one Trillium submission wrapper. |

## Related

Cluster dispatch wrappers that call into these live in
[`local_workflows/alliance_canada/study1/`](../../../../local_workflows/alliance_canada/study1/).
If a script here is renamed, those shell scripts need updating with it — they are not
covered by the Python test suite.
