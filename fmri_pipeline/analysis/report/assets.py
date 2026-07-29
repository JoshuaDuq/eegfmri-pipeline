"""Discovery of the background, mask, and tissue images a report figure needs.

One implementation on purpose. This job was previously done twice with different
answers: ``FmriAnalysisPipeline._discover_plot_assets`` searched a list of
derivative layouts, while ``reporting.py`` separately built a mean-BOLD background
from run metadata and fell back to nilearn's MNI template.

Discovery is best-effort throughout: the fMRIPrep root is configured per study and
may not be mounted. A missing asset yields ``None``, and figures degrade to a
plain background rather than failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MNI_SPACE_TOKENS = ("MNI152NLin2009cAsym", "MNI152NLin6Asym")
_TISSUE_CLASSES = ("GM", "WM", "CSF")


@dataclass(frozen=True)
class PlotAssets:
    """Paths a figure may use as context. Every field is optional."""

    background: Optional[Path] = None
    mask: Optional[Path] = None
    probseg: Dict[str, Path] = field(default_factory=dict)
    dseg: Optional[Path] = None


def _subject_dirs(deriv_root: Path, subject: str) -> List[Path]:
    """Return candidate subject directories across the layouts in use."""
    return [
        deriv_root / "preprocessed" / "fmri" / subject,
        deriv_root / "preprocessed" / "fmri" / "fmriprep" / subject,
        deriv_root / "fmriprep" / subject,
    ]


def _first_glob(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _space_tokens(space: str) -> List[str]:
    return list(_MNI_SPACE_TOKENS) if space == "mni" else ["T1w"]


def _find_background(
    subject_dir: Path, subject: str, task: str, space: str
) -> Optional[Path]:
    """Prefer an anatomical image; fall back to a boldref.

    An anatomical background makes an overlay anatomically interpretable in a way a
    single BOLD reference volume does not, so it wins whenever it exists.
    """
    anat = subject_dir / "anat"
    for token in _space_tokens(space):
        if space == "mni":
            candidate = anat / f"{subject}_space-{token}_desc-preproc_T1w.nii.gz"
        else:
            candidate = anat / f"{subject}_desc-preproc_T1w.nii.gz"
        if candidate.exists():
            return candidate

    func = subject_dir / "func"
    for token in _space_tokens(space):
        for pattern in (
            f"{subject}_task-{task}_*space-{token}_desc-preproc_boldref.nii.gz",
            f"{subject}_task-{task}_*space-{token}_boldref.nii.gz",
        ):
            found = _first_glob(func, pattern)
            if found is not None:
                return found
    return None


def _find_mask(subject_dir: Path, subject: str, task: str, space: str) -> Optional[Path]:
    func = subject_dir / "func"
    for token in _space_tokens(space):
        found = _first_glob(
            func, f"{subject}_task-{task}_*space-{token}_desc-brain_mask.nii.gz"
        )
        if found is not None:
            return found
    return None


def _find_tissue(
    subject_dir: Path, subject: str
) -> Tuple[Dict[str, Path], Optional[Path]]:
    """Return per-class probability maps and a discrete segmentation.

    Probability maps are preferred by the carpet, which assigns each voxel to its
    highest-probability class; the discrete segmentation is the fallback.
    """
    anat = subject_dir / "anat"
    probseg: Dict[str, Path] = {}
    for tissue in _TISSUE_CLASSES:
        found = _first_glob(anat, f"{subject}_*label-{tissue}_probseg.nii.gz")
        if found is not None:
            probseg[tissue] = found

    dseg = _first_glob(anat, f"{subject}_*desc-aseg_dseg.nii.gz")
    if dseg is None:
        dseg = _first_glob(anat, f"{subject}_*dseg.nii.gz")
    return probseg, dseg


def discover_plot_assets(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
    space: str,
) -> PlotAssets:
    """Locate plotting context for one subject, task, and space.

    ``space`` is ``"native"`` or ``"mni"``. Absent assets are reported as ``None``
    rather than raised, because a figure without a background is still a figure.
    """
    space = str(space or "").strip().lower()
    background: Optional[Path] = None
    mask: Optional[Path] = None
    probseg: Dict[str, Path] = {}
    dseg: Optional[Path] = None

    for subject_dir in _subject_dirs(Path(deriv_root), subject):
        if not subject_dir.exists():
            continue
        if background is None:
            background = _find_background(subject_dir, subject, task, space)
        if mask is None:
            mask = _find_mask(subject_dir, subject, task, space)
        if not probseg and dseg is None:
            probseg, dseg = _find_tissue(subject_dir, subject)

    if background is None:
        logger.debug(
            "No background image found for %s task-%s space-%s", subject, task, space
        )
    return PlotAssets(background=background, mask=mask, probseg=probseg, dseg=dseg)


__all__ = ["PlotAssets", "discover_plot_assets"]
