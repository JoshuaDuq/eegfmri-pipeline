# T1: electrode localization and template montage

**Configs:** [`../config/t1_electrode_localization.yaml`](../config/t1_electrode_localization.yaml),
[`../config/t1_template_montage.yaml`](../config/t1_template_montage.yaml)

## Why this folder exists

Source localization is only as good as the electrode positions it is given. A template
montage assumes every head is the average head; where a participant's own T1 exists, the
electrodes can be found in it instead, and the forward model built on where the electrodes
actually were.

This folder does both, because not every participant has a usable T1 and the study needs a
defined fallback rather than a silent one.

## The files

Two runnable entrypoints, and the modules they are built from:

| File | What it contributes |
|---|---|
| `run_t1_electrode_localization.py` | **Entrypoint.** Locates electrodes in a participant's own T1. |
| `run_t1_template_montage.py` | **Entrypoint.** The fallback: fits a template montage to the anatomy when per-participant localization is not available. |
| `t1_electrode_localization.py` | The localization itself — thresholding, candidate detection, and the `LocalizationParameters` / `ElectrodeLocalization` types the rest of the folder passes around. |
| `t1_electrode_localization_inputs.py` | Gathers and validates what a localization run needs before it starts. |
| `t1_electrode_localization_outputs.py` | Writes the result out in the layout downstream montage code reads. |
| `t1_anatomical_reference.py` | The shared anatomical reference both paths need — the common ground that keeps localized and templated montages comparable. |
| `t1_template_montage.py` | Template fitting, built on the same anatomical reference. |

The split into `_inputs` / `_outputs` modules is deliberate: it keeps the localization
itself testable without a filesystem, which is why
[`tests/scripts/test_t1_electrode_localization.py`](../../../../tests/scripts/test_t1_electrode_localization.py)
can exercise it directly.
