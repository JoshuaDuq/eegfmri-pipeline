"""Per-run estimates of a contrast, taken from the model that was already fitted.

A first-level contrast over six runs is a fixed-effects combination. nilearn combines
runs with equal weight -- ``compute_fixed_effect_contrast`` sums ``Contrast`` objects
and scales by 1/n, and ``Contrast.__mul__`` scales variance by the square of the
scalar, so the statistic reduces to ``sum(e) / sqrt(sum(v))``. A noisy run therefore
contributes its effect to the numerator at full strength while inflating the
denominator; the combination is *not* dominated by the most precise run.
``tests/fmri/test_pooling_rule.py`` pins that rule, because it is nilearn's rather than
ours and a release could change it.

Either way, an effect resting entirely on one run and an effect present in all six
produce the same map, the same z, and the same cluster table. Nothing else in the
report distinguishes them, and the distinction is usually the first thing anyone asks
about a single-subject result.

Nothing here fits a model. Nilearn's ``FirstLevelModel`` keeps ``labels_`` and
``results_`` as one entry per run, and ``compute_contrast`` combines across them at
call time -- so the per-run estimates already exist inside a fitted model and only need
reading out. That is why this is cheap enough to do unconditionally, and why it cannot
live in the report package: it needs the fitted object, which exists only during the
analysis run.

The maps are written as one 4D volume per quantity, with run on the fourth axis. Twelve
separate files per contrast is the same bytes and eleven more chances for a partial
write to leave a contrast half-described.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunLevelContrast:
    """Per-run effect and variance for one contrast, plus the runs they belong to."""

    effect: Any
    variance: Any
    run_labels: Tuple[str, ...]

    @property
    def n_runs(self) -> int:
        return len(self.run_labels)


def _contrast_vectors(flm: Any, contrast_def: Any) -> List[np.ndarray]:
    """Expand a contrast definition into one weight vector per run's design.

    Matching is by the run's own design columns rather than by position: runs need
    not share a column order, or even a column set once a run lacks a condition.
    """
    from nilearn.glm.contrasts import expression_to_contrast_vector

    designs = list(getattr(flm, "design_matrices_", []) or [])
    if not designs:
        raise ValueError("The fitted model carries no design matrices.")

    if isinstance(contrast_def, (list, tuple)) and len(contrast_def) == len(designs):
        per_run = list(contrast_def)
    else:
        per_run = [contrast_def] * len(designs)

    vectors: List[np.ndarray] = []
    for definition, design in zip(per_run, designs):
        if isinstance(definition, str):
            vectors.append(
                np.asarray(
                    expression_to_contrast_vector(
                        definition, design.columns.tolist()
                    ),
                    dtype=float,
                )
            )
        else:
            vectors.append(np.asarray(definition, dtype=float))
    return vectors


def compute_run_level_contrast(
    flm: Any,
    contrast_def: Any,
    *,
    run_labels: Sequence[str] = (),
) -> Optional[RunLevelContrast]:
    """Return per-run effect and variance maps for ``contrast_def``.

    ``None`` when the model holds a single run -- there is nothing to compare -- or
    when the per-run estimates cannot be read out. Best-effort throughout: this is a
    diagnostic, and the contrast it describes is already on disk by the time it runs.
    """
    from nilearn.glm.contrasts import compute_contrast

    labels = list(getattr(flm, "labels_", []) or [])
    results = list(getattr(flm, "results_", []) or [])
    masker = getattr(flm, "masker_", None)
    if masker is None or not labels or len(labels) != len(results):
        logger.info("The fitted model exposes no per-run results; skipping run level.")
        return None
    if len(labels) < 2:
        return None

    try:
        vectors = _contrast_vectors(flm, contrast_def)
    except Exception as exc:
        logger.warning("Could not expand the contrast per run (%s)", exc)
        return None

    effects: List[np.ndarray] = []
    variances: List[np.ndarray] = []
    for index, (run_labels_, run_results, vector) in enumerate(
        zip(labels, results, vectors)
    ):
        try:
            contrast = compute_contrast(run_labels_, run_results, vector, stat_type="t")
            effects.append(np.asarray(contrast.effect_size(), dtype=np.float32))
            variances.append(np.asarray(contrast.effect_variance(), dtype=np.float32))
        except Exception as exc:
            logger.warning(
                "Could not compute the run-level contrast for run %d (%s)", index, exc
            )
            return None

    try:
        effect_img = masker.inverse_transform(np.vstack(effects))
        variance_img = masker.inverse_transform(np.vstack(variances))
    except Exception as exc:
        logger.warning("Could not map the run-level contrast back to a volume (%s)", exc)
        return None

    labels_out = [
        str(run_labels[i]) if i < len(run_labels) else f"run-{i + 1:02d}"
        for i in range(len(effects))
    ]
    return RunLevelContrast(
        effect=effect_img, variance=variance_img, run_labels=tuple(labels_out)
    )


def write_run_level_maps(
    result: RunLevelContrast, *, out_dir: Path, stem: str, cfg_hash: str
) -> Tuple[Optional[Path], Optional[Path]]:
    """Write the per-run effect and variance volumes. Returns their paths."""
    import nibabel as nib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Optional[Path]] = []
    for image, quantity in ((result.effect, "effect_size"), (result.variance, "effect_variance")):
        path = out_dir / f"{stem}_stat-{quantity}_desc-perrun_{cfg_hash}.nii.gz"
        try:
            nib.save(image, str(path))
            written.append(path)
        except Exception as exc:
            logger.warning("Could not write %s (%s)", path.name, exc)
            written.append(None)
    return written[0], written[1]


__all__ = [
    "RunLevelContrast",
    "compute_run_level_contrast",
    "write_run_level_maps",
]
