"""``eeg_pipeline/`` holds what is true of any EEG study. This checks it stayed that way.

The BCG gap-fill engine and the line-comb settings were both written inside
``eeg_pipeline/`` and used by exactly one study. They now live under ``studies/pain_study/``.
The distinction is not stylistic: ``line_comb`` encodes the measured frequencies of one
scanner room, and the BCG work exists to repair one vendor's failure mode on one set of
exports. Neither generalises, and the scripts README tells anyone adapting this repo to a
new paradigm not to touch core.

Nothing enforces that but this file.
"""

from __future__ import annotations

import os
import re

import pytest

from tests import REPO_ROOT

os.environ["MNE_DONTWRITE_HOME"] = "true"

#: Moved out of core. A file reappearing here means paradigm code went back into the pipeline.
RELOCATED = (
    "eeg_pipeline/preprocessing/bcg",
    "eeg_pipeline/preprocessing/residual_gradient.py",
    "eeg_pipeline/preprocessing/report/scanner.py",
    "eeg_pipeline/preprocessing/report/cohort/gradient.py",
    "eeg_pipeline/preprocessing/report/analyzer_qc.py",
    "eeg_pipeline/preprocessing/report/cohort/analyzer.py",
    "eeg_pipeline/preprocessing/report/cohort_qc.py",
    "eeg_pipeline/preprocessing/report/cohort/noise_floor.py",
    "eeg_pipeline/preprocessing/pulse_artifact_qc.py",
    "eeg_pipeline/preprocessing/brainvision_markers.py",
    "eeg_pipeline/utils/config/acquisition.py",
    "eeg_pipeline/utils/config/presets/eeg_only.yaml",
    "eeg_pipeline/analysis/qc",
    "eeg_pipeline/plotting/scanner_harmonic_comb.py",
    "eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py",
    "eeg_pipeline/cli/commands/harmonics.py",
    "eeg_pipeline/cli/commands/harmonics_parser.py",
    "eeg_pipeline/cli/commands/harmonics_orchestrator.py",
)

#: Where each now lives.
RELOCATED_TO = (
    "studies/pain_study/analysis/bcg",
    "studies/pain_study/analysis/line_comb",
    "studies/pain_study/scripts/line_comb",
    "studies/pain_study/scripts/cardiac_gaps",
    "studies/pain_study/analysis/gradient",
    "studies/pain_study/scripts/gradient",
    "studies/pain_study/scripts/bcg",
)

#: Directories whose Python must not name this paradigm.
CORE_TREES = ("eeg_pipeline", "fmri_pipeline")

#: Strings that couple core code to this study. A module naming ``pain_study`` is reaching
#: into the study tree, which is what the layout exists to prevent.
#:
#: The task label ``thermalactive`` is deliberately *not* checked. It appears in core
#: docstrings as an illustrative BIDS id -- ``eeg_pipeline/preprocessing/report/style.py``
#: uses ``sub-0015_task-thermalactive_run-1`` to explain what a run label strips -- and an
#: example in prose does not make the module paradigm-dependent.
PARADIGM_MARKERS = ("pain_study",)


def _core_python_files():
    for tree in CORE_TREES:
        root = REPO_ROOT / tree
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


@pytest.mark.parametrize("relative_path", RELOCATED)
def test_paradigm_code_is_gone_from_core(relative_path: str) -> None:
    assert not (
        REPO_ROOT / relative_path
    ).exists(), f"{relative_path} is paradigm-specific and belongs under studies/pain_study/"


@pytest.mark.parametrize("relative_path", RELOCATED_TO)
def test_the_study_owns_it_instead(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).exists()


def test_core_does_not_import_the_study() -> None:
    """A core module importing ``studies`` inverts the dependency the layout depends on."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _core_python_files()
        if "import studies" in path.read_text(encoding="utf-8")
    ]

    assert not offenders, f"core modules importing the study: {offenders}"


@pytest.mark.parametrize("marker", PARADIGM_MARKERS)
def test_core_python_does_not_name_this_paradigm(marker: str) -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _core_python_files()
        if marker in path.read_text(encoding="utf-8")
    ]

    assert not offenders, f"core modules naming {marker!r}: {offenders}"


def test_the_line_comb_settings_left_the_core_config() -> None:
    """They describe one room. They belong with the code that applies them."""
    core_config = (REPO_ROOT / "eeg_pipeline/utils/config/eeg_config.yaml").read_text(
        encoding="utf-8"
    )
    workflow_config = REPO_ROOT / "studies/pain_study/scripts/line_comb/config.yaml"

    assert "line_comb_removal:" not in core_config
    assert "line_comb_removal:" in workflow_config.read_text(encoding="utf-8")


def test_the_workflow_scripts_do_not_pin_a_drive() -> None:
    """Absolute drive paths in source are what the workflow config replaced."""
    offenders = []
    for folder in ("line_comb", "cardiac_gaps"):
        root = REPO_ROOT / "studies/pain_study/scripts" / folder
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "/Volumes/" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, f"drive paths hardcoded in {offenders}"


def test_the_harmonics_subsystem_left_core() -> None:
    for relative_path in (
        "eeg_pipeline/analysis/qc",
        "eeg_pipeline/plotting/scanner_harmonic_comb.py",
        "eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py",
        "eeg_pipeline/cli/commands/harmonics.py",
        "eeg_pipeline/cli/commands/harmonics_parser.py",
        "eeg_pipeline/cli/commands/harmonics_orchestrator.py",
    ):
        assert not (REPO_ROOT / relative_path).exists(), f"{relative_path} is scanner code"


def test_the_marker_sanitation_left_core() -> None:
    assert not (REPO_ROOT / "eeg_pipeline/preprocessing/brainvision_markers.py").exists()
    assert (
        REPO_ROOT / "studies/pain_study/scripts/conversion/brainvision_markers.py"
    ).exists()


# Compound terms, deliberately. Bare "gradient" would hit np.gradient,
# GradientBoostingRegressor and cnn_gradient_clip_norm, all legitimate and all under
# eeg_pipeline/analysis/. Bare "fmri" would hit resolve_fmri_bids_root and the
# eeg-pipeline fmri commands, which drive this repo's separate fMRI pipeline.
SCANNER_MARKERS = (
    "volume_locked",
    "repetition_time_s",
    "volume_marker",
    "Pulse Artifact/R",
    "scanner gradient",
    "brainvision_analyzer",
)

# Bare words, safe in this tree and nowhere else: eeg_pipeline/preprocessing/ has no
# np.gradient, no GradientBoostingRegressor, no gradient_clip, and no "scanner RAS" --
# those all live under eeg_pipeline/analysis/. "volume" is deliberately absent: ordinary
# English, and an MNE source-space term.
PREPROCESSING_FORBIDDEN_WORDS = (
    "scanner",
    "gradient",
    "bore",
    "analyzer",
    "ballistocardiogram",
    "bcg",
)


@pytest.mark.parametrize("marker", SCANNER_MARKERS)
def test_core_python_does_not_name_a_scanner_concept(marker: str) -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "eeg_pipeline").rglob("*.py")
        if "__pycache__" not in path.parts
        and marker.lower() in path.read_text(encoding="utf-8").lower()
    ]

    assert not offenders, f"core modules naming {marker!r}: {offenders}"


@pytest.mark.parametrize("word", PREPROCESSING_FORBIDDEN_WORDS)
def test_the_preprocessing_tree_is_free_of_scanner_prose(word: str) -> None:
    pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "eeg_pipeline/preprocessing").rglob("*.py")
        if "__pycache__" not in path.parts and pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, f"preprocessing modules naming {word!r}: {offenders}"


@pytest.mark.parametrize("marker", SCANNER_MARKERS)
def test_the_core_config_does_not_name_a_scanner_concept(marker: str) -> None:
    text = (REPO_ROOT / "eeg_pipeline/utils/config/eeg_config.yaml").read_text(encoding="utf-8")

    assert marker.lower() not in text.lower()
