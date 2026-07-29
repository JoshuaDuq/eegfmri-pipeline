"""What one fitted contrast records for the report.

This is the seam that separates rendering from fitting. The analysis run writes a
manifest beside its stat maps; the report reads manifests and nothing else. Without
it the report can only be produced from inside the GLM path, which is why subject
QC was previously recomputed for every contrast and why iterating on a figure meant
re-running the model.

Everything here is plain JSON and plain paths. The report must be reproducible from
a derivatives tree alone, so a manifest may not carry live objects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "report_manifest.json"

#: Fields holding a single optional path.
_PATH_FIELDS = ("stat_map", "effect_map", "variance_map", "mask")
#: Fields holding a tuple of paths.
_PATH_TUPLE_FIELDS = ("design_matrices", "bold_paths", "confounds_paths")


@dataclass(frozen=True)
class ContrastManifest:
    """Everything the report needs about one fitted contrast."""

    subject: str
    task: str
    contrast_name: str
    space: str

    stat_map: Path
    effect_map: Optional[Path]
    variance_map: Optional[Path]
    mask: Optional[Path]

    threshold_mode: str
    z_threshold: float
    fdr_q: float
    cluster_min_voxels: int
    two_sided: bool
    radiological: bool

    design_matrices: Tuple[Path, ...]
    contrast_vector: Optional[Tuple[float, ...]]
    contrast_columns: Tuple[str, ...]

    included_runs: Tuple[str, ...]
    #: ``(run, reason)`` pairs. The reason is carried because a report that says a
    #: run was dropped without saying why gives a reader nothing to act on.
    excluded_runs: Tuple[Tuple[str, str], ...]

    bold_paths: Tuple[Path, ...]
    confounds_paths: Tuple[Path, ...]

    t_r: Optional[float]
    smoothing_fwhm: Optional[float]
    #: Whether the model applied signal scaling, which decides whether an effect
    #: size may be labelled "% signal change" or only "arbitrary BOLD units".
    signal_scaling: bool
    confound_strategy: str


def _encode(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def write_manifest(manifest: ContrastManifest, path: Path) -> Path:
    """Write ``manifest`` as indented JSON, and return the path written."""
    payload = {key: _encode(value) for key, value in asdict(manifest).items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(path: Path) -> ContrastManifest:
    """Load a manifest, restoring paths and tuples.

    Unknown keys are dropped rather than raising, so a manifest written by a newer
    version stays readable by an older one.
    """
    payload: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {field.name for field in fields(ContrastManifest)}
    data = {key: value for key, value in payload.items() if key in known}

    for name in _PATH_FIELDS:
        if data.get(name) is not None:
            data[name] = Path(data[name])
    for name in _PATH_TUPLE_FIELDS:
        data[name] = tuple(Path(p) for p in data.get(name) or ())

    data["contrast_vector"] = (
        tuple(float(v) for v in data["contrast_vector"])
        if data.get("contrast_vector") is not None
        else None
    )
    data["contrast_columns"] = tuple(data.get("contrast_columns") or ())
    data["included_runs"] = tuple(data.get("included_runs") or ())
    data["excluded_runs"] = tuple(
        (str(run), str(reason)) for run, reason in (data.get("excluded_runs") or ())
    )
    return ContrastManifest(**data)


def discover_manifests(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
) -> List[ContrastManifest]:
    """Return every contrast manifest for one subject and task.

    A manifest that cannot be read is skipped with a warning rather than aborting
    the report: one malformed contrast should cost its own section, not the whole
    document. Results are ordered by contrast name so a regenerated report keeps a
    stable section order.
    """
    root = Path(deriv_root) / subject / "fmri" / "first_level" / f"task-{task}"
    if not root.exists():
        return []

    found: List[ContrastManifest] = []
    for path in sorted(root.glob(f"contrast-*/{MANIFEST_FILENAME}")):
        try:
            manifest = read_manifest(path)
        except Exception as exc:
            logger.warning("Skipping unreadable manifest %s (%s)", path, exc)
            continue
        if manifest.task == task:
            found.append(manifest)
    return sorted(found, key=lambda m: m.contrast_name)


def sample_masks_from_confounds(paths: Sequence[Any]) -> List[np.ndarray]:
    """One boolean keep-mask per run, matching the censoring the GLM applied.

    Reuses ``_is_censor_column`` so the report censors exactly what the model
    censored -- motion outliers, explicit outliers, and non-steady-state volumes.
    A report that censored differently from the model would be describing a
    different analysis than the one that ran.

    Lives here rather than in ``reporting`` because the report path must not import
    the module that pulls in the contrast builder; that import is what the
    decoupling exists to prevent.
    """
    import pandas as pd

    from fmri_pipeline.utils.bold_discovery import _is_censor_column

    masks: List[np.ndarray] = []
    for path in paths:
        frame = pd.read_csv(str(path), sep="\t")
        censor = [column for column in frame.columns if _is_censor_column(column)]
        if not censor:
            masks.append(np.ones(len(frame), dtype=bool))
            continue
        flagged = frame[censor].to_numpy(dtype=float) > 0
        keep = ~flagged.any(axis=1)
        if not keep.any():
            # Censoring everything erases the run from exactly the panels that
            # exist to show what censoring did: the carpet cannot be standardised
            # and tSNR cannot be computed from nothing. Keep the frames and say so,
            # so the reader sees a fully censored run rather than a failed panel.
            logger.warning(
                "Every frame of %s is flagged for censoring; QC panels will show "
                "all frames uncensored.",
                path,
            )
            keep = np.ones(len(frame), dtype=bool)
        masks.append(keep)
    return masks


__all__ = [
    "MANIFEST_FILENAME",
    "ContrastManifest",
    "discover_manifests",
    "read_manifest",
    "sample_masks_from_confounds",
    "write_manifest",
]
