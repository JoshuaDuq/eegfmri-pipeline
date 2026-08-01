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


@dataclass(frozen=True)
class SignFlipNull:
    """A familywise height from sign-flipping run-level contributions.

    Exchangeability is over runs, which is the unit the design actually replicates.
    The null is exact and enumerated rather than sampled: with ``n`` runs there are
    ``2**(n-1)`` distinct sign patterns for a two-sided maximum statistic, because a
    pattern and its global negation give identical ``|z|`` maps.

    Why this exists beside Bonferroni and FDR: both of those assume every voxel is
    drawn from N(0, 1). A single-subject map combined across runs is routinely
    over-dispersed relative to that -- on this study's own data the fitted null is
    N(-0.61, 1.51^2) -- so both corrections are computed against a distribution the
    map demonstrably does not follow. Sign-flipping runs makes no distributional
    assumption at all; it asks how large a maximum this same data produces when the
    only thing changed is which runs are labelled positive.
    """

    null_max: Tuple[float, ...]
    observed_max: float
    fwe_height: float
    fwe_survivors: int
    global_p: float
    p_floor: float
    n_runs: int
    n_patterns: int
    alpha: float


def _sign_patterns(n_runs: int) -> np.ndarray:
    """Return the distinct sign patterns, identity first.

    The first run's sign is pinned to +1: a pattern and its global negation produce
    the same ``|z|`` map, so enumerating both would double the null with copies and
    halve the apparent resolution of the p-value for nothing.
    """
    import itertools

    patterns = [
        (1.0,) + rest for rest in itertools.product((1.0, -1.0), repeat=n_runs - 1)
    ]
    # Identity first, so callers can read the observed map off row zero.
    patterns.sort(key=lambda pattern: [sign < 0 for sign in pattern])
    return np.asarray(patterns, dtype=float)


def compute_sign_flip_null(
    flm: Any, contrast_def: Any, *, alpha: float = 0.05
) -> Optional[SignFlipNull]:
    """Enumerate the run sign-flip null for ``contrast_def``.

    Each pattern is evaluated by passing the per-run vectors ``s_i * c_i`` through the
    same ``compute_contrast`` call the pipeline already uses, so the pooling rule is
    nilearn's own and the identity pattern reproduces the stored map exactly rather
    than approximating it.

    ``None`` when the model holds a single run, or when the contrast cannot be
    expanded. Best-effort throughout: this is a diagnostic, and the contrast it
    describes is already on disk by the time it runs.
    """
    designs = list(getattr(flm, "design_matrices_", []) or [])
    masker = getattr(flm, "masker_", None)
    if masker is None or len(designs) < 2:
        return None

    try:
        vectors = _contrast_vectors(flm, contrast_def)
    except Exception as exc:
        logger.warning("Could not expand the contrast for the sign-flip null (%s)", exc)
        return None

    patterns = _sign_patterns(len(vectors))

    maxima: List[float] = []
    observed: Optional[np.ndarray] = None
    for index, signs in enumerate(patterns):
        flipped = [sign * vector for sign, vector in zip(signs, vectors)]
        try:
            z_img = flm.compute_contrast(flipped, output_type="z_score")
        except Exception as exc:
            logger.warning("Sign-flip pattern %d failed (%s)", index, exc)
            return None
        z = np.asarray(masker.transform(z_img), dtype=float).ravel()
        z = z[np.isfinite(z)]
        if z.size == 0:
            return None
        maxima.append(float(np.abs(z).max()))
        if index == 0:
            observed = z

    if observed is None:
        return None

    null_max = np.asarray(maxima, dtype=float)
    observed_max = float(null_max[0])
    height = float(np.quantile(null_max, 1.0 - alpha))
    # The identity is a member of the null set and always ties the observed maximum,
    # so the +1 in the numerator is not a continuity correction -- it is that tie. It
    # is also why p can never fall below ``p_floor``; see that field.
    global_p = float((np.sum(null_max >= observed_max) + 1) / (null_max.size + 1))
    return SignFlipNull(
        null_max=tuple(float(value) for value in null_max),
        observed_max=observed_max,
        fwe_height=height,
        fwe_survivors=int(np.sum(np.abs(observed) >= height)),
        global_p=global_p,
        p_floor=float(2 / (null_max.size + 1)),
        n_runs=len(vectors),
        n_patterns=int(null_max.size),
        alpha=float(alpha),
    )


def write_sign_flip_null(
    null: SignFlipNull, *, out_dir: Path, stem: str, cfg_hash: str
) -> Optional[Path]:
    """Write the enumerated null, one row per sign pattern."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_desc-signflipnull_{cfg_hash}.tsv"

    patterns = _sign_patterns(null.n_runs)
    frame = pd.DataFrame(
        {
            "pattern": np.arange(1, null.n_patterns + 1),
            "signs": [
                "".join("+" if sign > 0 else "-" for sign in row) for row in patterns
            ],
            "max_abs_z": null.null_max,
        }
    )
    try:
        frame.to_csv(path, sep="\t", index=False)
    except Exception as exc:
        logger.warning("Could not write %s (%s)", path.name, exc)
        return None
    return path


@dataclass(frozen=True)
class RunInfluence:
    """What dropping one run does to the combined map."""

    dropped_run: str
    survivors: int
    delta: int
    max_abs_z: float
    correlation: float


def compute_run_influence(
    flm: Any,
    contrast_def: Any,
    *,
    run_labels: Sequence[str] = (),
    threshold: float = 2.3,
) -> Optional[Tuple[RunInfluence, ...]]:
    """Recombine the contrast with each run dropped in turn.

    A dropped run is expressed as an all-zero contrast vector, which nilearn's
    ``compute_fixed_effect_contrast`` skips outright while dividing by the count of
    *surviving* contrasts. That is what makes this exact rather than a second pooling
    rule: the all-runs case is the stored map, so the deltas reconcile with the cluster
    table instead of merely resembling it.

    The forest panel answers a related but different question -- how each run estimates
    the contrast at the chosen peaks. A run can carry the largest peak estimates while
    another run moves the map more, because peak estimates and map-wide survivor counts
    are not the same measurement.

    ``None`` for a single-run model, which has nothing to drop.
    """
    designs = list(getattr(flm, "design_matrices_", []) or [])
    masker = getattr(flm, "masker_", None)
    if masker is None or len(designs) < 2:
        return None

    try:
        vectors = _contrast_vectors(flm, contrast_def)
    except Exception as exc:
        logger.warning("Could not expand the contrast for run influence (%s)", exc)
        return None

    def _z(vecs: List[np.ndarray]) -> Optional[np.ndarray]:
        try:
            img = flm.compute_contrast(vecs, output_type="z_score")
        except Exception as exc:
            logger.warning("Could not recombine the contrast (%s)", exc)
            return None
        values = np.asarray(masker.transform(img), dtype=float).ravel()
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    combined = _z(vectors)
    if combined is None:
        return None
    baseline = int(np.sum(np.abs(combined) > threshold))

    labels = [
        str(run_labels[i]) if i < len(run_labels) else f"run-{i + 1:02d}"
        for i in range(len(vectors))
    ]

    rows: List[RunInfluence] = []
    for index, label in enumerate(labels):
        held_out = [
            np.zeros_like(vector) if i == index else vector
            for i, vector in enumerate(vectors)
        ]
        reduced = _z(held_out)
        if reduced is None:
            return None
        if np.std(reduced) == 0 or np.std(combined) == 0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(reduced, combined)[0, 1])
        survivors = int(np.sum(np.abs(reduced) > threshold))
        rows.append(
            RunInfluence(
                dropped_run=label,
                survivors=survivors,
                delta=survivors - baseline,
                max_abs_z=float(np.abs(reduced).max()),
                correlation=correlation,
            )
        )
    return tuple(rows)


def write_run_influence(
    rows: Sequence[RunInfluence], *, out_dir: Path, stem: str, cfg_hash: str
) -> Optional[Path]:
    """Write one row per dropped run."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_desc-runinfluence_{cfg_hash}.tsv"
    frame = pd.DataFrame(
        [
            {
                "dropped_run": row.dropped_run,
                "survivors": row.survivors,
                "delta": row.delta,
                "max_abs_z": row.max_abs_z,
                "correlation": row.correlation,
            }
            for row in rows
        ]
    )
    try:
        frame.to_csv(path, sep="\t", index=False)
    except Exception as exc:
        logger.warning("Could not write %s (%s)", path.name, exc)
        return None
    return path


__all__ = [
    "RunInfluence",
    "RunLevelContrast",
    "SignFlipNull",
    "compute_run_influence",
    "compute_run_level_contrast",
    "compute_sign_flip_null",
    "write_run_influence",
    "write_run_level_maps",
    "write_sign_flip_null",
]
