"""How the recordings were acquired, as one question asked in one place.

``preprocessing.eeg_fmri`` says the EEG was recorded inside an MR scanner. That single
fact decides whether a whole family of stages has anything to measure: the gradient and
ballistocardiogram artifacts, the scanner-harmonic comb, the pulse markers a correction
leaves behind, and the ECG channel every cardiac measurement reads.

It lives here rather than on the preprocessing pipeline because the stages that depend on
it are spread across modules, and each one answering the question for itself is how they
drifted apart: some keyed off the Analyzer switch, some off an ECG channel name, and one
off nothing at all.
"""

from __future__ import annotations

from typing import Any

from eeg_pipeline.utils.config.loader import get_config_value

_MISSING = object()


def is_eeg_fmri(config: Any) -> bool:
    """Whether these recordings were acquired inside an MR scanner.

    Explicit rather than inferred. Guessing from the presence of ``Volume`` markers would
    make behaviour depend on how a conversion step happened to name its annotations, and
    would silently drop every scanner stage for a dataset whose markers were renamed.

    A config written before this key existed falls back to
    ``preprocessing.brainvision_analyzer.enabled``, which is what used to decide both
    questions. Defaulting to ``False`` instead would strip the scanner QC from every such
    config without saying so. Configs that set the key get its answer, whatever the
    Analyzer switch says.
    """
    declared = get_config_value(config, "preprocessing.eeg_fmri", _MISSING)
    if declared is _MISSING or declared is None:
        return bool(get_config_value(config, "preprocessing.brainvision_analyzer.enabled", False))
    return bool(declared)


__all__ = ["is_eeg_fmri"]
