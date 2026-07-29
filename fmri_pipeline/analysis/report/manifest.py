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

    #: Whether ``mask`` is the mask the GLM was fitted inside, rather than one
    #: discovered from the preprocessing derivatives. The two differ: the fitted mask
    #: is the intersection across runs, a discovered one is a single run's. Only the
    #: first justifies the claim that voxels outside it were not tested, so the panel
    #: that makes that claim checks this rather than assuming it.
    #:
    #: Defaulted, and last, so a manifest written before this field existed still
    #: loads -- and loads as the conservative answer.
    mask_is_analysis_mask: bool = False

    #: Every model setting that shaped this contrast, as ordered label/value pairs.
    #:
    #: A report that shows a result without the settings that produced it cannot be
    #: reproduced from, and cannot be compared against another study whose HRF or
    #: high-pass differed. Pairs rather than typed fields because the set grows with
    #: the model options and the report only ever renders them in order.
    model_settings: Tuple[Tuple[str, str], ...] = ()

    #: The confound columns actually regressed out, by name.
    #:
    #: A strategy name is not the same fact: "auto" resolves to a different column set
    #: per run depending on what fMRIPrep wrote, and the difference decides what the
    #: residuals contain.
    confound_columns: Tuple[str, ...] = ()


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
    data["confound_columns"] = tuple(str(c) for c in (data.get("confound_columns") or ()))
    data["model_settings"] = tuple(
        (str(label), str(value)) for label, value in (data.get("model_settings") or ())
    )
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


def _run_label(path: Any, fallback_index: int) -> str:
    """Name a run from its BIDS filename, or by position when it carries no entity."""
    text = str(path)
    for part in Path(text).name.split("_"):
        if part.startswith("run-"):
            return part
    return f"run-{fallback_index:02d}"


def _report_space(analysis_space: Any) -> str:
    """Collapse an fMRIPrep space label to what the report reasons about.

    The report only asks one question of a space: whether the glass-brain
    projection is defined for it. Everything that is not MNI is native as far as
    that question goes.
    """
    text = str(analysis_space or "").strip().lower()
    return "mni" if text.startswith("mni") else "native"


#: Model settings worth recording, as ``(attribute, label, formatter)``.
#:
#: Only settings that change the numbers. A report that shows a result without these
#: cannot be reproduced from and cannot be compared against a study whose HRF basis or
#: high-pass cutoff differed -- and neither difference is visible in any map.
_MODEL_SETTINGS: Tuple[Tuple[str, str, Any], ...] = (
    ("hrf_model", "HRF model", str),
    ("drift_model", "Drift model", lambda v: str(v) if v else "none"),
    ("high_pass_hz", "High-pass cutoff", lambda v: f"{float(v):.4g} Hz ({1.0 / float(v):.0f} s)" if v else "none"),
    ("low_pass_hz", "Low-pass cutoff", lambda v: f"{float(v):.4g} Hz" if v else "none"),
    ("confounds_strategy", "Confound strategy", str),
    ("auto_compcor_n", "CompCor components", lambda v: str(int(v))),
    ("output_type", "Statistic requested", str),
    ("fmriprep_space", "Fit in space", str),
    ("resample_to_freesurfer", "Resampled to FreeSurfer", lambda v: "yes" if v else "no"),
    ("contrast_type", "Contrast type", str),
    ("formula", "Contrast formula", lambda v: str(v) if v else "not a formula contrast"),
)


def model_settings_from_config(contrast_cfg: Any) -> Tuple[Tuple[str, str], ...]:
    """Extract the settings that shaped a fit, as ordered label/value pairs.

    Reads whatever the config object actually carries and skips the rest, so a config
    that gains or loses an option does not break the manifest. An unreadable setting
    is omitted rather than guessed: a wrong value here would be worse than a missing
    one, because a reader has no way to check it against the maps.
    """
    if contrast_cfg is None:
        return ()
    missing = object()
    pairs: List[Tuple[str, str]] = []
    for attribute, label, formatter in _MODEL_SETTINGS:
        # The read is inside the guard, not before it. `hasattr` only swallows
        # AttributeError, so probing a property that raises anything else propagates
        # and costs the whole configuration rather than the one setting.
        try:
            value = getattr(contrast_cfg, attribute, missing)
            if value is missing:
                continue
            pairs.append((label, formatter(value)))
        except Exception:
            continue
    return tuple(pairs)


def write_report_manifest(
    *,
    contrast_dir: Path,
    subject: str,
    task: str,
    contrast_name: str,
    stat_map: Path,
    run_meta: Any,
    effect_map: Optional[Path] = None,
    variance_map: Optional[Path] = None,
    mask: Optional[Path] = None,
    design_matrices: Sequence[Path] = (),
    contrast_vector: Optional[Sequence[float]] = None,
    contrast_columns: Sequence[str] = (),
    threshold_mode: str = "z",
    z_threshold: float = 2.3,
    fdr_q: float = 0.05,
    cluster_min_voxels: int = 0,
    two_sided: bool = True,
    radiological: bool = False,
    smoothing_fwhm: Optional[float] = None,
    signal_scaling: bool = False,
    mask_is_analysis_mask: bool = False,
    contrast_cfg: Any = None,
) -> Optional[Path]:
    """Record what was fit, beside what was fit.

    This is the seam that lets a report be rendered from a derivatives tree without
    the model. The report reads manifests and nothing else, so anything it needs
    about a contrast has to be written here.

    Returns ``None`` and logs rather than raising: by the time this runs the GLM has
    already been fitted and its maps are on disk, and losing that to a problem with
    reporting metadata would be the most expensive failure available.
    """
    try:
        meta: Dict[str, Any] = run_meta if isinstance(run_meta, dict) else {}
        if not isinstance(run_meta, dict):
            raise TypeError(f"run_meta must be a mapping, got {type(run_meta).__name__}")

        bold_paths = [Path(str(p)) for p in meta.get("included_bold_paths", []) or []]
        confounds_paths = [
            Path(str(p))
            for p in (meta.get("included_confounds_paths", []) or [])
            if p is not None
        ]
        included_runs = tuple(
            _run_label(path, index) for index, path in enumerate(bold_paths, start=1)
        )
        excluded_runs = tuple(
            (
                f"run-{int(entry.get('run_index', 0)):02d}",
                str(entry.get("reason", "unspecified")),
            )
            for entry in (meta.get("skipped_runs", []) or [])
            if isinstance(entry, dict)
        )

        t_r = meta.get("tr")
        manifest = ContrastManifest(
            subject=subject,
            task=task,
            contrast_name=contrast_name,
            space=_report_space(meta.get("analysis_space")),
            stat_map=Path(stat_map),
            effect_map=Path(effect_map) if effect_map else None,
            variance_map=Path(variance_map) if variance_map else None,
            mask=Path(mask) if mask else None,
            threshold_mode=threshold_mode,
            z_threshold=float(z_threshold),
            fdr_q=float(fdr_q),
            cluster_min_voxels=int(cluster_min_voxels),
            two_sided=bool(two_sided),
            radiological=bool(radiological),
            design_matrices=tuple(Path(p) for p in design_matrices),
            contrast_vector=(
                tuple(float(v) for v in contrast_vector)
                if contrast_vector is not None
                else None
            ),
            contrast_columns=tuple(str(c) for c in contrast_columns),
            included_runs=included_runs,
            excluded_runs=excluded_runs,
            bold_paths=tuple(bold_paths),
            confounds_paths=tuple(confounds_paths),
            t_r=None if t_r is None else float(t_r),
            smoothing_fwhm=(
                None if smoothing_fwhm is None else float(smoothing_fwhm)
            ),
            signal_scaling=bool(signal_scaling),
            confound_strategy=str(meta.get("confounds_strategy", "unspecified")),
            # Only true when the caller passed the mask the model was fitted inside.
            # A mask discovered from the preprocessing derivatives is a single run's,
            # and the report's coverage panel makes a claim only the fitted one earns.
            mask_is_analysis_mask=bool(mask_is_analysis_mask and mask is not None),
            model_settings=model_settings_from_config(contrast_cfg),
            # The resolved column names, not just the strategy that chose them:
            # "auto" lands on a different set per run depending on what fMRIPrep
            # wrote, and that difference decides what the residuals contain.
            confound_columns=tuple(
                str(column) for column in (meta.get("confound_columns") or [])
            ),
        )
        return write_manifest(manifest, Path(contrast_dir) / MANIFEST_FILENAME)
    except Exception as exc:
        logger.warning(
            "Could not write the report manifest for %s/%s (%s). The contrast's "
            "maps are unaffected; the report will not cover this contrast.",
            subject,
            contrast_name,
            exc,
        )
        return None


__all__ = [
    "MANIFEST_FILENAME",
    "ContrastManifest",
    "discover_manifests",
    "model_settings_from_config",
    "read_manifest",
    "sample_masks_from_confounds",
    "write_manifest",
    "write_report_manifest",
]
