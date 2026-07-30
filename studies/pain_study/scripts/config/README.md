# Shared paradigm configuration

## Why this folder exists

Two different kinds of configuration end up here, and it is worth knowing which is which.

**Override templates** are *not* loaded automatically. They record the values this paradigm
needs in the core `eeg_pipeline/utils/config/eeg_config.yaml`, so that the core config can
stay paradigm-agnostic and still be usable here. You apply them yourself — through the TUI's
Global Setup, or by copying the values into your project config.

**Per-script configs** are loaded, by the one script named in them.

Configuration belonging to a *single workflow* does not live here. It lives beside the code
it configures — [`../line_comb/config.yaml`](../line_comb/config.yaml) and
[`../cardiac_gaps/config.yaml`](../cardiac_gaps/config.yaml) — because a number that only
one workflow has an opinion about is easier to trust when it sits next to the code that
reads it. See [`../workflow_config.py`](../workflow_config.py) for how those resolve.

## The files

| File | Kind | What it is for |
|---|---|---|
| `thermal_pain_eeg_overrides.yaml` | Override template | EEG-side values for this paradigm: task label, thermode trigger prefixes, event column names, ERDS and source-localization windows. |
| `thermal_pain_fmri_overrides.yaml` | Override template | fMRI-side values: onset reference, event granularity, and the rest. |
| `t1_electrode_localization.yaml` | Per-script config | Read by [`../t1/run_t1_electrode_localization.py`](../t1/run_t1_electrode_localization.py). |
| `t1_template_montage.yaml` | Per-script config | Read by [`../t1/run_t1_template_montage.py`](../t1/run_t1_template_montage.py). |

## Where paths come from

None of these files should restate `bids_root`, `deriv_root` or `source_data`. Those name
the same three directories for every stage of every pipeline and are answered once, in the
core `eeg_config.yaml`. Copying them per file means editing five places the day the drive
moves, and five chances for them to disagree.
