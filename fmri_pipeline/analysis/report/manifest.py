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
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MANIFEST_FILENAME = "report_manifest.json"
REPORT_MANIFEST_SCHEMA_VERSION = 2

#: Fields holding a single optional path.
_PATH_FIELDS = (
    "stat_map",
    "effect_map",
    "variance_map",
    "mask",
    "run_effect_map",
    "run_variance_map",
    "sign_flip_null_tsv",
    "run_influence_tsv",
)
#: Fields holding a tuple of paths.
_PATH_TUPLE_FIELDS = (
    "design_matrices",
    "bold_paths",
    "confounds_paths",
    "residual_paths",
    "predicted_paths",
)


@dataclass(frozen=True)
class ContrastManifest:
    """Everything the report needs about one fitted contrast."""

    schema_version: int
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

    t_r: float
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

    #: How the model scaled its signal, from
    #: :func:`~fmri_pipeline.utils.bold_discovery.fitted_signal_scaling_mode`.
    #:
    #: ``signal_scaling`` says whether any scaling happened; this says which, and the
    #: three modes do not produce the same quantity. Only ``voxel-mean`` divides each
    #: voxel by its own temporal mean, which is what makes an effect a percentage of
    #: that voxel's own baseline -- percent signal change. Recorded separately so the
    #: units line can name the right one rather than assuming the common case.
    signal_scaling_mode: Optional[str] = None

    #: Per-run estimates of this contrast, as 4D volumes with run on the fourth axis.
    #:
    #: A first-level contrast over several runs is a fixed-effects combination,
    #: weighted equally per run: an effect resting on one run and an effect present in
    #: all of them produce the same map, the same z, and the same cluster table. These
    #: are what let the report tell the two apart.
    #:
    run_effect_map: Optional[Path] = None
    run_variance_map: Optional[Path] = None

    #: The enumerated run sign-flip null, one row per sign pattern, and the
    #: leave-one-run-out table. Both are diagnostics of the same thing -- how much of
    #: this contrast is a property of which runs were labelled positive -- so they are
    #: written together or not at all.
    sign_flip_null_tsv: Optional[Path] = None
    run_influence_tsv: Optional[Path] = None

    #: The sign-flip null's summary, duplicated out of the TSV so a panel that only
    #: needs the height does not have to reopen and parse the enumeration.
    #:
    #: ``sign_flip_p_floor`` is stored rather than recomputed because it is the number
    #: that makes ``sign_flip_global_p`` readable: the unflipped pattern is always a
    #: member of the null and always ties the observed maximum, so the p can never fall
    #: below ``2 / (2**(n_runs-1) + 1)``. Six runs floor at 0.061, and a p of 0.061
    #: printed without its floor reads as a near-miss when it is the smallest value the
    #: test can return.
    sign_flip_fwe_height: Optional[float] = None
    sign_flip_fwe_survivors: Optional[int] = None
    sign_flip_global_p: Optional[float] = None
    sign_flip_p_floor: Optional[float] = None
    sign_flip_n_patterns: Optional[int] = None
    sign_flip_n_runs: Optional[int] = None
    sign_flip_observed_max: Optional[float] = None

    #: Retained-frame model residuals and predictions reconstructed as ``Y - X beta``
    #: and ``X beta`` from the fitted design and coefficients.
    #:
    #: One 4D image per included run. These are exact fitted-model quantities, not
    #: report summaries, and are kept separate so run boundaries remain explicit.
    residual_paths: Tuple[Path, ...] = ()
    predicted_paths: Tuple[Path, ...] = ()
    retained_frame_indices: Tuple[Tuple[int, ...], ...] = ()
    model_fit_series_space: str = "unwhitened-model-response"


def _encode(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def validate_manifest(manifest: ContrastManifest) -> None:
    """Reject internal contradictions in one report manifest."""
    if manifest.schema_version != REPORT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported report manifest schema "
            f"{manifest.schema_version!r}; expected {REPORT_MANIFEST_SCHEMA_VERSION}."
        )

    if manifest.stat_map is None:
        raise ValueError("Report manifest requires stat_map.")
    if not math.isfinite(float(manifest.t_r)) or manifest.t_r <= 0:
        raise ValueError(f"t_r must be finite and positive, got {manifest.t_r!r}.")
    if manifest.mask is None or not manifest.mask_is_analysis_mask:
        raise ValueError("Report manifest requires the fitted analysis mask.")

    run_count = len(manifest.included_runs)
    if run_count == 0:
        raise ValueError("Report manifest requires at least one included run.")
    if run_count != len(manifest.bold_paths):
        raise ValueError(
            "included_runs and bold_paths must have equal lengths, got "
            f"{run_count} and {len(manifest.bold_paths)}."
        )
    if any(not str(run).strip() for run in manifest.included_runs):
        raise ValueError("included_runs may not contain blank labels.")
    if len(set(manifest.included_runs)) != run_count:
        raise ValueError("included_runs must contain unique labels.")
    expected_run_labels = run_labels_from_bold_paths(manifest.bold_paths)
    if manifest.included_runs != expected_run_labels:
        raise ValueError("included_runs must match the BOLD run entities.")
    if any(
        not str(run).strip() or not str(reason).strip() for run, reason in manifest.excluded_runs
    ):
        raise ValueError("excluded_runs requires a nonblank run label and reason.")
    excluded_labels = tuple(run for run, _reason in manifest.excluded_runs)
    if len(set(excluded_labels)) != len(excluded_labels):
        raise ValueError("excluded_runs must contain unique run labels.")
    if any(not _is_bids_run_label(run) for run in excluded_labels):
        raise ValueError("excluded_runs requires explicit BIDS run labels.")
    if set(manifest.included_runs) & set(excluded_labels):
        raise ValueError("A run cannot be both included and excluded.")
    if manifest.confounds_paths and len(manifest.confounds_paths) != run_count:
        raise ValueError(
            "confounds_paths must be empty or align with included_runs, got "
            f"{len(manifest.confounds_paths)} and {run_count}."
        )
    if manifest.design_matrices and len(manifest.design_matrices) != run_count:
        raise ValueError(
            "design_matrices must be empty or align with included_runs, got "
            f"{len(manifest.design_matrices)} and {run_count}."
        )
    if not manifest.residual_paths or not manifest.predicted_paths:
        raise ValueError(
            "Report manifest requires residual_paths and predicted_paths for model-fit evidence."
        )
    if len(manifest.residual_paths) != run_count or len(manifest.predicted_paths) != run_count:
        raise ValueError(
            "Manifest model-fit paths must align with included_runs, got "
            f"{len(manifest.residual_paths)}, {len(manifest.predicted_paths)}, "
            f"and {run_count}."
        )
    if len(manifest.retained_frame_indices) != run_count:
        raise ValueError(
            "retained_frame_indices must align with included_runs, got "
            f"{len(manifest.retained_frame_indices)} and {run_count}."
        )
    for run_label, retained_indices in zip(
        manifest.included_runs,
        manifest.retained_frame_indices,
    ):
        if not retained_indices:
            raise ValueError(f"retained_frame_indices for {run_label} may not be empty.")
        if any(
            isinstance(index, bool) or not isinstance(index, (int, np.integer))
            for index in retained_indices
        ):
            raise TypeError(f"retained_frame_indices for {run_label} must contain integers.")
        if retained_indices[0] < 0 or any(
            current >= following
            for current, following in zip(retained_indices, retained_indices[1:])
        ):
            raise ValueError(
                f"retained_frame_indices for {run_label} must be strictly increasing "
                "nonnegative values."
            )
    if manifest.model_fit_series_space != "unwhitened-model-response":
        raise ValueError("model_fit_series_space must be 'unwhitened-model-response'.")

    has_vector = manifest.contrast_vector is not None
    has_columns = bool(manifest.contrast_columns)
    if has_vector != has_columns:
        raise ValueError(
            "contrast_vector and contrast_columns must either both be present or " "both be absent."
        )
    if has_vector and len(manifest.contrast_vector or ()) != len(manifest.contrast_columns):
        raise ValueError(
            "contrast_vector and contrast_columns must have equal lengths, got "
            f"{len(manifest.contrast_vector or ())} and "
            f"{len(manifest.contrast_columns)}."
        )

    if (manifest.run_effect_map is None) != (manifest.run_variance_map is None):
        raise ValueError(
            "run_effect_map and run_variance_map must either both be present or " "both be absent."
        )

    if (manifest.sign_flip_null_tsv is None) != (manifest.sign_flip_fwe_height is None):
        raise ValueError(
            "sign_flip_null_tsv and sign_flip_fwe_height must either both be present "
            "or both be absent: a height with no enumerated null cannot be checked, "
            "and a null with no height was never summarised."
        )


def validate_manifest_artifacts(manifest: ContrastManifest) -> None:
    """Require every recorded artifact to exist and share the fitted grid."""
    validate_manifest(manifest)
    single_paths = {
        "stat_map": manifest.stat_map,
        "effect_map": manifest.effect_map,
        "variance_map": manifest.variance_map,
        "mask": manifest.mask,
        "run_effect_map": manifest.run_effect_map,
        "run_variance_map": manifest.run_variance_map,
        "sign_flip_null_tsv": manifest.sign_flip_null_tsv,
        "run_influence_tsv": manifest.run_influence_tsv,
    }
    path_groups = {
        "design_matrices": manifest.design_matrices,
        "bold_paths": manifest.bold_paths,
        "confounds_paths": manifest.confounds_paths,
        "residual_paths": manifest.residual_paths,
        "predicted_paths": manifest.predicted_paths,
    }

    for field_name, path in single_paths.items():
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Manifest {field_name} does not exist: {path}")
    for field_name, paths in path_groups.items():
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Manifest {field_name} artifact does not exist: {path}")

    _validate_image_geometry(manifest)
    _validate_model_fit_geometry(manifest)


def _validate_image_geometry(manifest: ContrastManifest) -> None:
    """Require inferential maps to use the stat map's spatial geometry."""
    import nibabel as nib

    stat_image = nib.load(str(manifest.stat_map))
    if len(stat_image.shape) != 3:
        raise ValueError(f"Manifest stat_map must be 3D, got shape {stat_image.shape}.")

    spatial_maps = {
        "effect_map": manifest.effect_map,
        "variance_map": manifest.variance_map,
        "mask": manifest.mask,
    }
    for field_name, path in spatial_maps.items():
        if path is None:
            continue
        image = nib.load(str(path))
        if image.shape != stat_image.shape or not np.allclose(image.affine, stat_image.affine):
            raise ValueError(f"Manifest {field_name} geometry does not match stat_map.")

    run_maps = {
        "run_effect_map": manifest.run_effect_map,
        "run_variance_map": manifest.run_variance_map,
    }
    expected_run_shape = (*stat_image.shape, len(manifest.included_runs))
    for field_name, path in run_maps.items():
        if path is None:
            continue
        image = nib.load(str(path))
        if image.shape != expected_run_shape or not np.allclose(image.affine, stat_image.affine):
            raise ValueError(
                f"Manifest {field_name} geometry does not match stat_map and included_runs."
            )


def _validate_model_fit_geometry(manifest: ContrastManifest) -> None:
    """Require per-run fit series to align with fitted inputs and retained frames."""
    import nibabel as nib
    import pandas as pd

    design_paths: Sequence[Optional[Path]] = manifest.design_matrices or (None,) * len(
        manifest.included_runs
    )
    confounds_paths: Sequence[Optional[Path]] = manifest.confounds_paths or (None,) * len(
        manifest.included_runs
    )
    for (
        run_label,
        bold_path,
        design_path,
        confounds_path,
        residual_path,
        predicted_path,
        retained_indices,
    ) in zip(
        manifest.included_runs,
        manifest.bold_paths,
        design_paths,
        confounds_paths,
        manifest.residual_paths,
        manifest.predicted_paths,
        manifest.retained_frame_indices,
    ):
        bold_image = nib.load(str(bold_path))
        residual_image = nib.load(str(residual_path))
        predicted_image = nib.load(str(predicted_path))

        if len(bold_image.shape) != 4:
            raise ValueError(
                f"Manifest BOLD input for {run_label} must be 4D, got {bold_image.shape}."
            )
        if len(residual_image.shape) != 4 or residual_image.shape != predicted_image.shape:
            raise ValueError(f"Manifest model-fit series for {run_label} require matching shapes.")
        if not np.allclose(residual_image.affine, predicted_image.affine):
            raise ValueError(f"Manifest model-fit series for {run_label} require matching affines.")
        if residual_image.shape[:3] != bold_image.shape[:3] or not np.allclose(
            residual_image.affine, bold_image.affine
        ):
            raise ValueError(
                f"Manifest model-fit series for {run_label} do not match BOLD geometry."
            )
        bold_frames = int(bold_image.shape[3])
        if retained_indices[-1] >= bold_frames:
            raise ValueError(
                f"Manifest retained frame indices for {run_label} exceed the "
                f"{bold_frames} BOLD timepoints."
            )
        retained_frames = len(retained_indices)
        if residual_image.shape[3] != retained_frames:
            raise ValueError(
                f"Manifest model-fit series for {run_label} have "
                f"{residual_image.shape[3]} retained timepoints; retained frame indices "
                f"contain {retained_frames}."
            )
        if design_path is not None:
            design_rows = len(pd.read_csv(design_path, sep="\t"))
            if design_rows != retained_frames:
                raise ValueError(
                    f"Manifest design matrix for {run_label} has {design_rows} rows; "
                    f"retained frame indices contain {retained_frames}."
                )
        if confounds_path is not None:
            confounds_rows = len(pd.read_csv(confounds_path, sep="\t"))
            if confounds_rows != bold_frames:
                raise ValueError(
                    f"Manifest confounds for {run_label} have {confounds_rows} rows; "
                    f"BOLD timepoints contain {bold_frames}."
                )
            strategy = str(manifest.confound_strategy).strip().lower()
            if strategy not in {"none", "no", "off"}:
                censor_keep = sample_masks_from_confounds((confounds_path,))[0]
                if not censor_keep[np.asarray(retained_indices, dtype=int)].all():
                    raise ValueError(
                        f"Manifest retained frame indices for {run_label} include "
                        "confound-censored frames."
                    )


def validate_manifest_collection(manifests: Sequence[ContrastManifest]) -> None:
    """Require manifests combined into one report to describe one run set."""
    if not manifests:
        raise ValueError("Cannot validate an empty report manifest collection.")

    first = manifests[0]
    identity = (first.subject, first.task)
    run_inputs = (
        first.space,
        first.t_r,
        first.included_runs,
        first.bold_paths,
        first.confounds_paths,
        first.retained_frame_indices,
    )
    for manifest in manifests:
        validate_manifest(manifest)
        if (manifest.subject, manifest.task) != identity:
            raise ValueError("All report manifests must describe the same subject and task.")
        candidate_inputs = (
            manifest.space,
            manifest.t_r,
            manifest.included_runs,
            manifest.bold_paths,
            manifest.confounds_paths,
            manifest.retained_frame_indices,
        )
        if candidate_inputs != run_inputs:
            raise ValueError("All report manifests must describe the same run inputs.")


def write_manifest(manifest: ContrastManifest, path: Path) -> Path:
    """Write ``manifest`` as indented JSON, and return the path written."""
    validate_manifest(manifest)
    payload = {key: _encode(value) for key, value in asdict(manifest).items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(path: Path) -> ContrastManifest:
    """Load one manifest whose schema exactly matches this renderer."""
    payload: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Report manifest {path} must contain a JSON object.")
    if "schema_version" not in payload:
        raise ValueError(f"Report manifest {path} is missing schema_version.")
    if payload["schema_version"] != REPORT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported report manifest schema "
            f"{payload['schema_version']!r} in {path}; "
            f"expected {REPORT_MANIFEST_SCHEMA_VERSION}."
        )

    known = {field.name for field in fields(ContrastManifest)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"Unknown report manifest field(s) in {path}: {unknown}.")
    missing = sorted(known - set(payload))
    if missing:
        raise ValueError(f"Missing report manifest field(s) in {path}: {missing}.")
    data = dict(payload)

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
    data["retained_frame_indices"] = tuple(
        tuple(indices) for indices in (data.get("retained_frame_indices") or ())
    )
    data["confound_columns"] = tuple(str(c) for c in (data.get("confound_columns") or ()))
    data["model_settings"] = tuple(
        (str(label), str(value)) for label, value in (data.get("model_settings") or ())
    )
    data["excluded_runs"] = tuple(
        (str(run), str(reason)) for run, reason in (data.get("excluded_runs") or ())
    )
    manifest = ContrastManifest(**data)
    validate_manifest(manifest)
    return manifest


def discover_manifests(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
) -> List[ContrastManifest]:
    """Return every valid contrast manifest for one subject and task."""
    root = Path(deriv_root) / subject / "fmri" / "first_level" / f"task-{task}"
    if not root.exists():
        return []

    found: List[ContrastManifest] = []
    for path in sorted(root.glob(f"contrast-*/{MANIFEST_FILENAME}")):
        manifest = read_manifest(path)
        if manifest.subject != subject:
            raise ValueError(
                f"Report manifest {path} declares subject {manifest.subject!r}; "
                f"expected {subject!r}."
            )
        if manifest.task != task:
            raise ValueError(
                f"Report manifest {path} declares task {manifest.task!r}; " f"expected {task!r}."
            )
        found.append(manifest)
    ordered = sorted(found, key=lambda m: m.contrast_name)
    if ordered:
        validate_manifest_collection(ordered)
    return ordered


def sample_masks_from_confounds(paths: Sequence[Any]) -> List[np.ndarray]:
    """Return the frames not flagged by explicit fMRIPrep censor columns.

    This does not reconstruct the fitted sample mask: selected nuisance regressors can
    also exclude initial non-finite rows. ``retained_frame_indices`` is the authoritative
    record of what entered the model. This helper only cross-checks that no explicitly
    censored frame is recorded as retained.

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
            raise ValueError(
                f"Confounds file {path} flags every frame for censoring; "
                "the as-modelled QC inputs are empty."
            )
        masks.append(keep)
    return masks


def _is_bids_run_label(value: Any) -> bool:
    """Whether ``value`` is an explicit positive BIDS run entity."""
    label = str(value)
    suffix = label.removeprefix("run-")
    return label.startswith("run-") and suffix.isdigit() and int(suffix) > 0


def _run_entity(path: Any) -> Optional[str]:
    """Return the BIDS run entity carried by ``path``, when present."""
    for part in Path(str(path)).name.split("_"):
        if not part.startswith("run-"):
            continue
        if not _is_bids_run_label(part):
            raise ValueError(f"BOLD path contains an invalid BIDS run entity: {path}")
        return part
    return None


def run_labels_from_bold_paths(paths: Sequence[Any]) -> Tuple[str, ...]:
    """Return explicit run labels without inventing positional BIDS entities."""
    labels = tuple(_run_entity(path) for path in paths)
    if len(labels) == 1 and labels[0] is None:
        return ("runless",)
    if any(label is None for label in labels):
        raise ValueError("Each BOLD path in a multi-run analysis must contain a BIDS run entity.")

    resolved = tuple(label for label in labels if label is not None)
    if len(set(resolved)) != len(resolved):
        raise ValueError("BOLD paths must contain unique BIDS run entities.")
    return resolved


def _retained_frame_indices_from_metadata(value: Any) -> Tuple[Tuple[int, ...], ...]:
    """Encode the exact per-run source-frame indices supplied to the fitted model."""
    if value is None:
        raise ValueError("run_meta requires retained_frame_indices from the fitted model.")
    if not isinstance(value, list):
        raise TypeError("run_meta.retained_frame_indices must be a list of index lists.")

    retained: List[Tuple[int, ...]] = []
    for run_indices in value:
        if not isinstance(run_indices, list):
            raise TypeError("Each retained_frame_indices entry must be a list of integers.")
        retained.append(tuple(run_indices))
    return tuple(retained)


def _excluded_runs_from_metadata(value: Any) -> Tuple[Tuple[str, str], ...]:
    """Validate and encode run exclusions from fitted-model metadata."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("run_meta.skipped_runs must be a list of mappings.")

    excluded_runs: List[Tuple[str, str]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise TypeError("Each run_meta.skipped_runs entry must be a mapping.")
        if "run_label" not in entry:
            raise ValueError("Each skipped run requires run_label.")
        run_label = entry["run_label"]
        if not _is_bids_run_label(run_label):
            raise ValueError("Skipped run_label must be an explicit BIDS run label.")
        if "reason" not in entry:
            raise ValueError("Each skipped run requires a reason.")
        reason = entry["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Skipped run reason must be a nonblank string.")
        excluded_runs.append((str(run_label), reason.strip()))

    labels = [run for run, _reason in excluded_runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Skipped runs must contain unique run_label values.")
    return tuple(excluded_runs)


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
    (
        "high_pass_hz",
        "High-pass cutoff",
        lambda v: f"{float(v):.4g} Hz ({1.0 / float(v):.0f} s)" if v else "none",
    ),
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

    Reads every setting the config object carries. Missing attributes are absent from
    the result; errors raised while reading or formatting a present attribute surface.
    """
    if contrast_cfg is None:
        return ()
    missing = object()
    pairs: List[Tuple[str, str]] = []
    for attribute, label, formatter in _MODEL_SETTINGS:
        value = getattr(contrast_cfg, attribute, missing)
        if value is missing:
            continue
        pairs.append((label, formatter(value)))
    return tuple(pairs)


def write_report_manifest(
    *,
    contrast_dir: Path,
    subject: str,
    task: str,
    contrast_name: str,
    stat_map: Path,
    run_meta: Any,
    residual_paths: Sequence[Path],
    predicted_paths: Sequence[Path],
    effect_map: Optional[Path] = None,
    variance_map: Optional[Path] = None,
    mask: Optional[Path] = None,
    run_effect_map: Optional[Path] = None,
    run_variance_map: Optional[Path] = None,
    sign_flip_null_tsv: Optional[Path] = None,
    run_influence_tsv: Optional[Path] = None,
    sign_flip_fwe_height: Optional[float] = None,
    sign_flip_fwe_survivors: Optional[int] = None,
    sign_flip_global_p: Optional[float] = None,
    sign_flip_p_floor: Optional[float] = None,
    sign_flip_n_patterns: Optional[int] = None,
    sign_flip_n_runs: Optional[int] = None,
    sign_flip_observed_max: Optional[float] = None,
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
    signal_scaling_mode: Optional[str] = None,
    mask_is_analysis_mask: bool = False,
    contrast_cfg: Any = None,
) -> Path:
    """Record what was fit, beside what was fit.

    This is the seam that lets a report be rendered from a derivatives tree without
    the model. The report reads manifests and nothing else, so anything it needs
    about a contrast has to be written here.

    Invalid metadata raises immediately. A fitted map without a valid manifest cannot
    support a scientifically auditable report.
    """
    if not isinstance(run_meta, Mapping):
        raise TypeError(f"run_meta must be a mapping, got {type(run_meta).__name__}")
    if contrast_cfg is None:
        raise ValueError("Report manifest requires contrast_cfg provenance.")
    meta: Dict[str, Any] = run_meta

    bold_paths = [Path(str(p)) for p in meta.get("included_bold_paths", []) or []]
    confounds_paths = [
        Path(str(p)) for p in (meta.get("included_confounds_paths", []) or []) if p is not None
    ]
    included_runs = run_labels_from_bold_paths(bold_paths)
    excluded_runs = _excluded_runs_from_metadata(meta.get("skipped_runs", []))
    retained_frame_indices = _retained_frame_indices_from_metadata(
        meta.get("retained_frame_indices")
    )

    t_r = meta.get("tr")
    if t_r is None:
        raise ValueError("Report manifest requires the fitted model TR.")
    analysis_space = meta.get("analysis_space")
    if not str(analysis_space or "").strip():
        raise ValueError("Report manifest requires the fitted analysis_space.")

    manifest = ContrastManifest(
        schema_version=REPORT_MANIFEST_SCHEMA_VERSION,
        subject=subject,
        task=task,
        contrast_name=contrast_name,
        space=_report_space(analysis_space),
        stat_map=Path(stat_map),
        effect_map=Path(effect_map) if effect_map else None,
        variance_map=Path(variance_map) if variance_map else None,
        mask=Path(mask) if mask else None,
        run_effect_map=Path(run_effect_map) if run_effect_map else None,
        run_variance_map=Path(run_variance_map) if run_variance_map else None,
        sign_flip_null_tsv=Path(sign_flip_null_tsv) if sign_flip_null_tsv else None,
        run_influence_tsv=Path(run_influence_tsv) if run_influence_tsv else None,
        sign_flip_fwe_height=(
            None if sign_flip_fwe_height is None else float(sign_flip_fwe_height)
        ),
        sign_flip_fwe_survivors=(
            None if sign_flip_fwe_survivors is None else int(sign_flip_fwe_survivors)
        ),
        sign_flip_global_p=(
            None if sign_flip_global_p is None else float(sign_flip_global_p)
        ),
        sign_flip_p_floor=(None if sign_flip_p_floor is None else float(sign_flip_p_floor)),
        sign_flip_n_patterns=(
            None if sign_flip_n_patterns is None else int(sign_flip_n_patterns)
        ),
        sign_flip_n_runs=(None if sign_flip_n_runs is None else int(sign_flip_n_runs)),
        sign_flip_observed_max=(
            None if sign_flip_observed_max is None else float(sign_flip_observed_max)
        ),
        threshold_mode=threshold_mode,
        z_threshold=float(z_threshold),
        fdr_q=float(fdr_q),
        cluster_min_voxels=int(cluster_min_voxels),
        two_sided=bool(two_sided),
        radiological=bool(radiological),
        design_matrices=tuple(Path(p) for p in design_matrices),
        contrast_vector=(
            tuple(float(v) for v in contrast_vector) if contrast_vector is not None else None
        ),
        contrast_columns=tuple(str(c) for c in contrast_columns),
        included_runs=included_runs,
        excluded_runs=excluded_runs,
        bold_paths=tuple(bold_paths),
        confounds_paths=tuple(confounds_paths),
        t_r=float(t_r),
        smoothing_fwhm=(None if smoothing_fwhm is None else float(smoothing_fwhm)),
        # Derived from the mode rather than passed alongside it, so the two cannot
        # disagree about whether scaling happened.
        signal_scaling=signal_scaling_mode is not None,
        signal_scaling_mode=signal_scaling_mode,
        confound_strategy=str(meta.get("confounds_strategy", "unspecified")),
        # Only true when the caller passed the mask the model was fitted inside.
        # A mask discovered from the preprocessing derivatives is a single run's,
        # and the report's coverage panel makes a claim only the fitted one earns.
        mask_is_analysis_mask=bool(mask_is_analysis_mask and mask is not None),
        model_settings=model_settings_from_config(contrast_cfg),
        # The resolved column names, not just the strategy that chose them:
        # "auto" lands on a different set per run depending on what fMRIPrep
        # wrote, and that difference decides what the residuals contain.
        confound_columns=tuple(str(column) for column in (meta.get("confound_columns") or [])),
        residual_paths=tuple(Path(path) for path in residual_paths),
        predicted_paths=tuple(Path(path) for path in predicted_paths),
        retained_frame_indices=retained_frame_indices,
    )
    return write_manifest(manifest, Path(contrast_dir) / MANIFEST_FILENAME)


__all__ = [
    "MANIFEST_FILENAME",
    "REPORT_MANIFEST_SCHEMA_VERSION",
    "ContrastManifest",
    "discover_manifests",
    "model_settings_from_config",
    "read_manifest",
    "sample_masks_from_confounds",
    "validate_manifest",
    "validate_manifest_artifacts",
    "validate_manifest_collection",
    "write_manifest",
    "write_report_manifest",
]
