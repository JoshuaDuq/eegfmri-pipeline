from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from fmri_pipeline.utils.text import safe_slug

logger = logging.getLogger(__name__)

_SUPPORTED_SECOND_LEVEL_MODELS = frozenset(
    {"one-sample", "two-sample", "paired", "repeated-measures"}
)
_REQUIRED_GROUP_SPACE = "MNI152NLin2009cAsym"
_REQUIRED_FIRST_LEVEL_OUTPUT = "effect_size"


@dataclass(frozen=True)
class SecondLevelPermutationConfig:
    enabled: bool = False
    n_permutations: int = 5000
    two_sided: bool = True

    def normalized(self) -> "SecondLevelPermutationConfig":
        n_permutations = int(self.n_permutations)
        if n_permutations <= 0:
            raise ValueError(
                "Second-level permutation inference requires n_permutations > 0."
            )
        return SecondLevelPermutationConfig(
            enabled=bool(self.enabled),
            n_permutations=n_permutations,
            two_sided=bool(self.two_sided),
        )


@dataclass(frozen=True)
class SecondLevelConfig:
    model: str
    contrast_names: tuple[str, ...]
    input_root: Optional[str] = None
    condition_labels: Optional[tuple[str, ...]] = None
    formula: Optional[str] = None
    output_name: Optional[str] = None
    output_dir: Optional[str] = None
    covariates_file: Optional[str] = None
    subject_column: str = "subject"
    covariate_columns: Optional[tuple[str, ...]] = None
    group_column: Optional[str] = None
    group_a_value: Optional[str] = None
    group_b_value: Optional[str] = None
    write_design_matrix: bool = True
    permutation: SecondLevelPermutationConfig = field(
        default_factory=SecondLevelPermutationConfig
    )

    def normalized(self) -> "SecondLevelConfig":
        model = str(self.model or "one-sample").strip().lower()
        if model not in _SUPPORTED_SECOND_LEVEL_MODELS:
            supported = ", ".join(sorted(_SUPPORTED_SECOND_LEVEL_MODELS))
            raise ValueError(
                f"Unsupported fmri_group_level.model {self.model!r}. "
                f"Supported values: {supported}."
            )

        contrast_names = _normalize_string_tuple(
            self.contrast_names,
            field_name="fmri_group_level.contrast_names",
        )
        if not contrast_names:
            raise ValueError(
                "fmri_group_level.contrast_names must contain at least one "
                "first-level contrast name."
            )

        condition_labels = _normalize_optional_string_tuple(self.condition_labels)
        if condition_labels is not None and len(condition_labels) != len(
            contrast_names
        ):
            raise ValueError(
                "fmri_group_level.condition_labels must match "
                "fmri_group_level.contrast_names in length."
            )

        formula = _normalize_optional_string(self.formula)
        output_name = _normalize_optional_string(self.output_name)
        input_root = _normalize_optional_string(self.input_root)
        output_dir = _normalize_optional_string(self.output_dir)
        covariates_file = _normalize_optional_string(self.covariates_file)
        subject_column = str(self.subject_column or "subject").strip() or "subject"
        covariate_columns = _normalize_optional_string_tuple(self.covariate_columns)
        group_column = _normalize_optional_string(self.group_column)
        group_a_value = _normalize_optional_string(self.group_a_value)
        group_b_value = _normalize_optional_string(self.group_b_value)

        if model in {"one-sample", "two-sample"} and len(contrast_names) != 1:
            raise ValueError(
                f"{model} second-level analysis requires exactly one input contrast."
            )
        if model == "paired" and len(contrast_names) != 2:
            raise ValueError(
                "paired second-level analysis requires exactly two input "
                "contrasts ordered as A B."
            )
        if model == "repeated-measures" and len(contrast_names) < 2:
            raise ValueError(
                "repeated-measures second-level analysis requires at least two "
                "input contrasts."
            )

        if covariate_columns and covariates_file is None:
            raise ValueError(
                "fmri_group_level.covariates_file is required when "
                "fmri_group_level.covariate_columns is set."
            )

        if model == "two-sample":
            if covariates_file is None:
                raise ValueError(
                    "two-sample second-level analysis requires "
                    "fmri_group_level.covariates_file."
                )
            if group_column is None:
                raise ValueError(
                    "two-sample second-level analysis requires "
                    "fmri_group_level.group_column."
                )
            if group_a_value is None or group_b_value is None:
                raise ValueError(
                    "two-sample second-level analysis requires "
                    "fmri_group_level.group_a_value and group_b_value."
                )
            if group_a_value == group_b_value:
                raise ValueError(
                    "fmri_group_level.group_a_value and group_b_value must "
                    "be different."
                )
        else:
            if group_column is not None or group_a_value is not None or group_b_value is not None:
                raise ValueError(
                    f"{model} second-level analysis does not use group_column / "
                    "group_a_value / group_b_value."
                )

        if model == "repeated-measures" and covariate_columns:
            raise ValueError(
                "repeated-measures second-level analysis does not support "
                "subject-level covariates because they are aliased by the "
                "within-subject design."
            )

        permutation = (
            self.permutation.normalized()
            if bool(self.permutation.enabled)
            else SecondLevelPermutationConfig()
        )

        return SecondLevelConfig(
            model=model,
            contrast_names=contrast_names,
            input_root=input_root,
            condition_labels=condition_labels,
            formula=formula,
            output_name=output_name,
            output_dir=output_dir,
            covariates_file=covariates_file,
            subject_column=subject_column,
            covariate_columns=covariate_columns,
            group_column=group_column,
            group_a_value=group_a_value,
            group_b_value=group_b_value,
            write_design_matrix=bool(self.write_design_matrix),
            permutation=permutation,
        )


@dataclass(frozen=True)
class FirstLevelMapRecord:
    subject: str
    subject_label: str
    contrast_name: str
    map_path: Path
    sidecar_path: Path
    contrast_cfg: Dict[str, Any]


@dataclass(frozen=True)
class PreparedSecondLevelInput:
    image_paths: tuple[Path, ...]
    design_matrix: pd.DataFrame
    manifest: pd.DataFrame
    contrast_spec: Any
    stat_type: str
    output_name: str
    output_dir: Path
    metadata: Dict[str, Any]


def _normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_string_tuple(
    values: Iterable[Any],
    *,
    field_name: str,
) -> tuple[str, ...]:
    items = _normalize_optional_string_tuple(values)
    if items is None:
        raise ValueError(f"{field_name} must not be empty.")
    return items


def _normalize_optional_string_tuple(
    values: Optional[Iterable[Any]],
) -> Optional[tuple[str, ...]]:
    if values is None:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        text = _normalize_optional_string(raw_value)
        if text is None:
            continue
        lowered = text.lower()
        if lowered in seen:
            raise ValueError(f"Duplicate value {text!r} is not allowed.")
        seen.add(lowered)
        normalized.append(text)

    if not normalized:
        return None
    return tuple(normalized)


def _normalize_subject_id(value: Any) -> str:
    text = _normalize_optional_string(value)
    if text is None:
        raise ValueError("Encountered empty subject identifier.")
    return text[4:] if text.startswith("sub-") else text


def _subject_label(subject: str) -> str:
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def _safe_design_column(prefix: str, label: str) -> str:
    token = safe_slug(label, default="item").replace("-", "_")
    return f"{prefix}_{token}"


def _load_sidecar_json(sidecar_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON sidecar: {sidecar_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object sidecar, got: {sidecar_path}")
    return payload


def _candidate_first_level_dirs(
    *,
    input_root: Path,
    subject_label: str,
    task: str,
    contrast_name: str,
) -> list[Path]:
    candidate_dirs = [
        input_root
        / subject_label
        / "fmri"
        / "first_level"
        / f"task-{task}"
        / f"contrast-{safe_slug(contrast_name, default='contrast')}",
        input_root / f"contrast-{safe_slug(contrast_name, default='contrast')}",
    ]
    return [path for path in candidate_dirs if path.exists()]


def _discover_first_level_effect_size_map(
    *,
    input_root: Path,
    subject: str,
    task: str,
    contrast_name: str,
) -> FirstLevelMapRecord:
    subject_id = _normalize_subject_id(subject)
    subject_label = _subject_label(subject_id)
    candidate_dirs = _candidate_first_level_dirs(
        input_root=input_root,
        subject_label=subject_label,
        task=task,
        contrast_name=contrast_name,
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            "Could not find first-level outputs for "
            f"{subject_label}, task-{task}, contrast {contrast_name!r} under {input_root}. "
            "Second-level analysis requires first-level maps written under the "
            "standard derivatives layout or under a shared --group-input-root "
            "created via --output-dir."
        )

    matches: list[FirstLevelMapRecord] = []
    for contrast_dir in candidate_dirs:
        for sidecar_path in sorted(contrast_dir.glob("*.json")):
            payload = _load_sidecar_json(sidecar_path)
            if payload.get("subject") != subject_label:
                continue
            if payload.get("task") != task:
                continue
            if payload.get("contrast_name") != contrast_name:
                continue
            if payload.get("output_type_actual") != _REQUIRED_FIRST_LEVEL_OUTPUT:
                continue

            contrast_cfg = payload.get("contrast_cfg")
            if not isinstance(contrast_cfg, dict):
                raise ValueError(
                    f"Expected contrast_cfg object in first-level sidecar: {sidecar_path}"
                )
            space = str(contrast_cfg.get("fmriprep_space") or "").strip()
            if space != _REQUIRED_GROUP_SPACE:
                continue

            nifti_path = sidecar_path.with_suffix("").with_suffix(".nii.gz")
            if not nifti_path.exists():
                raise FileNotFoundError(
                    f"Missing first-level NIfTI for sidecar: {sidecar_path}"
                )
            matches.append(
                FirstLevelMapRecord(
                    subject=subject_id,
                    subject_label=subject_label,
                    contrast_name=contrast_name,
                    map_path=nifti_path,
                    sidecar_path=sidecar_path,
                    contrast_cfg=contrast_cfg,
                )
            )

    if not matches:
        raise FileNotFoundError(
            "Could not find a first-level effect-size map in "
            f"{_REQUIRED_GROUP_SPACE} for {subject_label}, task-{task}, contrast "
            f"{contrast_name!r}. Re-run first-level analysis with "
            "--fmriprep-space MNI152NLin2009cAsym and --output-type cope."
        )
    if len(matches) > 1:
        paths = ", ".join(str(record.map_path) for record in matches)
        raise ValueError(
            "Found multiple matching first-level maps for "
            f"{subject_label}, task-{task}, contrast {contrast_name!r}: {paths}"
        )
    return matches[0]


def _validate_consistent_first_level_configs(
    records: Sequence[FirstLevelMapRecord],
) -> None:
    by_contrast: dict[str, str] = {}
    by_paths: dict[str, Path] = {}
    for record in records:
        normalized_cfg = json.dumps(record.contrast_cfg, sort_keys=True)
        prior = by_contrast.get(record.contrast_name)
        if prior is None:
            by_contrast[record.contrast_name] = normalized_cfg
            by_paths[record.contrast_name] = record.sidecar_path
            continue
        if prior != normalized_cfg:
            raise ValueError(
                "Second-level analysis requires identical first-level contrast "
                f"settings across subjects for contrast {record.contrast_name!r}. "
                f"Conflict detected between {by_paths[record.contrast_name]} and "
                f"{record.sidecar_path}."
            )


def _load_subject_covariates(
    *,
    covariates_file: Path,
    subject_column: str,
) -> pd.DataFrame:
    if not covariates_file.exists():
        raise FileNotFoundError(
            f"Second-level covariates file does not exist: {covariates_file}"
        )

    suffix = covariates_file.suffix.lower()
    if suffix == ".tsv":
        df = pd.read_csv(covariates_file, sep="\t")
    elif suffix == ".csv":
        df = pd.read_csv(covariates_file)
    else:
        raise ValueError(
            "Second-level covariates file must be .tsv or .csv, got "
            f"{covariates_file.name!r}."
        )

    if subject_column not in df.columns:
        raise ValueError(
            f"Second-level covariates file is missing subject column {subject_column!r}."
        )

    df = df.copy()
    df[subject_column] = df[subject_column].map(_normalize_subject_id)
    if df[subject_column].duplicated().any():
        duplicates = sorted(
            df.loc[df[subject_column].duplicated(), subject_column].unique().tolist()
        )
        raise ValueError(
            "Second-level covariates file contains duplicate subjects: "
            f"{duplicates}"
        )
    return df.set_index(subject_column, drop=False)


def _select_covariate_columns(
    *,
    cov_df: pd.DataFrame,
    selected_subjects: Sequence[str],
    covariate_columns: Optional[Sequence[str]],
) -> pd.DataFrame:
    if not covariate_columns:
        return pd.DataFrame(index=[_normalize_subject_id(s) for s in selected_subjects])

    missing = [
        column for column in covariate_columns if column not in cov_df.columns
    ]
    if missing:
        raise ValueError(
            "Second-level covariates file is missing requested columns: "
            f"{missing}"
        )

    rows: list[pd.Series] = []
    for subject in selected_subjects:
        subject_id = _normalize_subject_id(subject)
        if subject_id not in cov_df.index:
            raise ValueError(
                "Second-level covariates file is missing selected subject "
                f"{_subject_label(subject_id)}."
            )
        rows.append(cov_df.loc[subject_id, list(covariate_columns)])

    selected = pd.DataFrame(rows, index=[_normalize_subject_id(s) for s in selected_subjects])
    centered = pd.DataFrame(index=selected.index)
    used_names: set[str] = set()
    for raw_column in selected.columns:
        numeric = pd.to_numeric(selected[raw_column], errors="coerce")
        if numeric.isna().any():
            bad_subjects = selected.index[numeric.isna()].tolist()
            raise ValueError(
                f"Second-level covariate {raw_column!r} contains non-numeric or "
                f"missing values for subjects: {bad_subjects}"
            )
        if np.isclose(float(numeric.std(ddof=0)), 0.0):
            raise ValueError(
                f"Second-level covariate {raw_column!r} is constant across the "
                "selected subjects."
            )
        design_name = _safe_design_column("cov", raw_column)
        if design_name in used_names:
            raise ValueError(
                "Sanitized second-level covariate names collide after cleaning. "
                f"Rename covariates before use: {raw_column!r}"
            )
        used_names.add(design_name)
        centered[design_name] = numeric - float(numeric.mean())
    return centered


def _resolve_two_sample_groups(
    *,
    cov_df: pd.DataFrame,
    selected_subjects: Sequence[str],
    group_column: str,
    group_a_value: str,
    group_b_value: str,
) -> pd.DataFrame:
    if group_column not in cov_df.columns:
        raise ValueError(
            f"Second-level covariates file is missing group column {group_column!r}."
        )

    rows: list[dict[str, Any]] = []
    for subject in selected_subjects:
        subject_id = _normalize_subject_id(subject)
        if subject_id not in cov_df.index:
            raise ValueError(
                "Second-level covariates file is missing selected subject "
                f"{_subject_label(subject_id)}."
            )
        group_value = _normalize_optional_string(cov_df.loc[subject_id, group_column])
        if group_value not in {group_a_value, group_b_value}:
            raise ValueError(
                f"Subject {_subject_label(subject_id)} has {group_column!r}="
                f"{group_value!r}; expected {group_a_value!r} or {group_b_value!r}."
            )
        rows.append(
            {
                "subject": subject_id,
                _safe_design_column("group", group_a_value): int(
                    group_value == group_a_value
                ),
                _safe_design_column("group", group_b_value): int(
                    group_value == group_b_value
                ),
                "group_label": group_value,
            }
        )

    group_df = pd.DataFrame(rows).set_index("subject", drop=False)
    counts = {
        group_a_value: int(
            group_df[_safe_design_column("group", group_a_value)].sum()
        ),
        group_b_value: int(
            group_df[_safe_design_column("group", group_b_value)].sum()
        ),
    }
    if min(counts.values()) == 0:
        raise ValueError(
            "Two-sample second-level analysis requires at least one subject in "
            f"each requested group. Counts: {counts}"
        )
    return group_df


def _load_image_signature(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    import nibabel as nib

    img = nib.load(str(path))
    return tuple(int(v) for v in img.shape), np.asarray(img.affine, dtype=float)


def _validate_same_grid(paths: Sequence[Path]) -> None:
    reference_path = paths[0]
    reference_shape, reference_affine = _load_image_signature(reference_path)
    for path in paths[1:]:
        shape, affine = _load_image_signature(path)
        if shape != reference_shape or not np.allclose(
            affine, reference_affine, atol=1e-6
        ):
            raise ValueError(
                "Second-level analysis requires all input maps to share the "
                "same voxel grid. Conflict detected between "
                f"{reference_path} and {path}."
            )


def _write_design_matrix_files(
    *,
    output_dir: Path,
    design_matrix: pd.DataFrame,
) -> Dict[str, str]:
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, str] = {}
    tsv_path = qc_dir / "second_level_design_matrix.tsv"
    design_matrix.to_csv(tsv_path, sep="\t", index=False)
    out["design_matrix_tsv"] = str(tsv_path)

    try:
        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_design_matrix

        ax = plot_design_matrix(design_matrix)
        figure = ax.figure
        png_path = qc_dir / "second_level_design_matrix.png"
        figure.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        out["design_matrix_png"] = str(png_path)
    except Exception as exc:
        logger.warning("Failed writing second-level design matrix figure: %s", exc)

    return out


def _write_manifest(output_dir: Path, manifest: pd.DataFrame) -> Path:
    manifest_path = output_dir / "input_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


def _save_nifti_map(
    *,
    image: Any,
    output_dir: Path,
    prefix: str,
    suffix: str,
) -> Path:
    import nibabel as nib

    path = output_dir / f"{prefix}_stat-{suffix}.nii.gz"
    nib.save(image, str(path))
    return path


def _write_metadata_sidecar(
    *,
    prepared: PreparedSecondLevelInput,
    config: SecondLevelConfig,
    saved_maps: Dict[str, str],
    manifest_path: Path,
    design_outputs: Dict[str, str],
) -> Path:
    payload = {
        "config": asdict(config),
        "metadata": prepared.metadata,
        "design_columns": list(prepared.design_matrix.columns),
        "contrast_spec": _serialize_contrast_spec(prepared.contrast_spec),
        "stat_type": prepared.stat_type,
        "manifest_path": str(manifest_path),
        "design_outputs": design_outputs,
        "saved_maps": saved_maps,
    }
    sidecar_path = prepared.output_dir / "second_level_metadata.json"
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return sidecar_path


def _serialize_contrast_spec(contrast_spec: Any) -> Any:
    if isinstance(contrast_spec, str):
        return contrast_spec
    if isinstance(contrast_spec, np.ndarray):
        return contrast_spec.tolist()
    return contrast_spec


def _derive_output_name(
    *,
    config: SecondLevelConfig,
    condition_labels: tuple[str, ...],
    default_label: str,
) -> str:
    if config.output_name:
        return safe_slug(config.output_name, default="second_level")
    if config.formula:
        return safe_slug(config.formula, default="second_level")
    if config.model == "one-sample":
        return safe_slug(f"{condition_labels[0]}_group_mean", default="second_level")
    if config.model == "two-sample":
        return safe_slug(default_label, default="second_level")
    if config.model == "paired":
        return safe_slug(
            f"{condition_labels[1]}_minus_{condition_labels[0]}",
            default="second_level",
        )
    if config.model == "repeated-measures":
        return safe_slug(default_label, default="second_level")
    return safe_slug(default_label, default="second_level")


def _resolve_output_dir(
    *,
    deriv_root: Path,
    task: str,
    config: SecondLevelConfig,
    output_name: str,
) -> Path:
    if config.output_dir:
        return Path(config.output_dir).expanduser().resolve()
    return (
        deriv_root
        / "group"
        / "fmri"
        / "second_level"
        / f"task-{task}"
        / f"model-{safe_slug(config.model, default='model')}"
        / f"contrast-{output_name}"
    )


def _difference_map(
    *,
    minuend_path: Path,
    subtrahend_path: Path,
    output_path: Path,
) -> Path:
    import nibabel as nib

    shape_a, affine_a = _load_image_signature(minuend_path)
    shape_b, affine_b = _load_image_signature(subtrahend_path)
    if shape_a != shape_b or not np.allclose(affine_a, affine_b, atol=1e-6):
        raise ValueError(
            "Paired second-level analysis requires each within-subject map pair "
            f"to share the same voxel grid. Conflict detected between "
            f"{minuend_path} and {subtrahend_path}."
        )

    img_a = nib.load(str(minuend_path))
    img_b = nib.load(str(subtrahend_path))
    data = np.asanyarray(img_a.dataobj, dtype=np.float32) - np.asanyarray(
        img_b.dataobj, dtype=np.float32
    )
    out_img = nib.Nifti1Image(data, affine=img_a.affine, header=img_a.header.copy())
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(output_path))
    return output_path


def _prepare_one_sample_input(
    *,
    config: SecondLevelConfig,
    subjects: Sequence[str],
    task: str,
    input_root: Path,
    output_dir: Path,
) -> PreparedSecondLevelInput:
    contrast_name = config.contrast_names[0]
    records = tuple(
        _discover_first_level_effect_size_map(
            input_root=input_root,
            subject=subject,
            task=task,
            contrast_name=contrast_name,
        )
        for subject in subjects
    )
    _validate_consistent_first_level_configs(records)

    image_paths = tuple(record.map_path for record in records)
    _validate_same_grid(image_paths)

    design_matrix = pd.DataFrame({"intercept": np.ones(len(records), dtype=float)})
    metadata: Dict[str, Any] = {
        "model": "one-sample",
        "input_contrast_names": list(config.contrast_names),
    }

    if config.covariates_file:
        cov_df = _load_subject_covariates(
            covariates_file=Path(config.covariates_file).expanduser().resolve(),
            subject_column=config.subject_column,
        )
        covariates = _select_covariate_columns(
            cov_df=cov_df,
            selected_subjects=subjects,
            covariate_columns=config.covariate_columns,
        )
        if not covariates.empty:
            design_matrix = pd.concat(
                [design_matrix.reset_index(drop=True), covariates.reset_index(drop=True)],
                axis=1,
            )
            metadata["covariates"] = list(covariates.columns)

    manifest = pd.DataFrame(
        {
            "subject": [record.subject_label for record in records],
            "contrast_name": [record.contrast_name for record in records],
            "condition_label": [contrast_name] * len(records),
            "map_path": [str(record.map_path) for record in records],
        }
    )
    output_name = _derive_output_name(
        config=config,
        condition_labels=(contrast_name,),
        default_label=f"{contrast_name}_group_mean",
    )
    return PreparedSecondLevelInput(
        image_paths=image_paths,
        design_matrix=design_matrix,
        manifest=manifest,
        contrast_spec=config.formula or "intercept",
        stat_type="t",
        output_name=output_name,
        output_dir=output_dir,
        metadata=metadata,
    )


def _prepare_two_sample_input(
    *,
    config: SecondLevelConfig,
    subjects: Sequence[str],
    task: str,
    input_root: Path,
    output_dir: Path,
) -> PreparedSecondLevelInput:
    contrast_name = config.contrast_names[0]
    records = tuple(
        _discover_first_level_effect_size_map(
            input_root=input_root,
            subject=subject,
            task=task,
            contrast_name=contrast_name,
        )
        for subject in subjects
    )
    _validate_consistent_first_level_configs(records)

    image_paths = tuple(record.map_path for record in records)
    _validate_same_grid(image_paths)

    cov_df = _load_subject_covariates(
        covariates_file=Path(config.covariates_file).expanduser().resolve(),
        subject_column=config.subject_column,
    )
    group_df = _resolve_two_sample_groups(
        cov_df=cov_df,
        selected_subjects=subjects,
        group_column=config.group_column or "",
        group_a_value=config.group_a_value or "",
        group_b_value=config.group_b_value or "",
    )

    design_matrix = group_df[
        [
            _safe_design_column("group", config.group_a_value or ""),
            _safe_design_column("group", config.group_b_value or ""),
        ]
    ].reset_index(drop=True)
    covariates = _select_covariate_columns(
        cov_df=cov_df,
        selected_subjects=subjects,
        covariate_columns=config.covariate_columns,
    )
    if not covariates.empty:
        design_matrix = pd.concat(
            [design_matrix.reset_index(drop=True), covariates.reset_index(drop=True)],
            axis=1,
        )

    group_a_col = _safe_design_column("group", config.group_a_value or "")
    group_b_col = _safe_design_column("group", config.group_b_value or "")
    contrast_formula = config.formula or f"{group_b_col} - {group_a_col}"

    manifest = pd.DataFrame(
        {
            "subject": [record.subject_label for record in records],
            "contrast_name": [record.contrast_name for record in records],
            "condition_label": [contrast_name] * len(records),
            "group_label": group_df["group_label"].to_list(),
            "map_path": [str(record.map_path) for record in records],
        }
    )
    output_name = _derive_output_name(
        config=config,
        condition_labels=(contrast_name,),
        default_label=f"{contrast_name}_{config.group_b_value}_minus_{config.group_a_value}",
    )
    metadata: Dict[str, Any] = {
        "model": "two-sample",
        "input_contrast_names": list(config.contrast_names),
        "group_column": config.group_column,
        "group_a_value": config.group_a_value,
        "group_b_value": config.group_b_value,
        "group_design_columns": {
            "group_a": group_a_col,
            "group_b": group_b_col,
        },
    }
    if not covariates.empty:
        metadata["covariates"] = list(covariates.columns)

    return PreparedSecondLevelInput(
        image_paths=image_paths,
        design_matrix=design_matrix,
        manifest=manifest,
        contrast_spec=contrast_formula,
        stat_type="t",
        output_name=output_name,
        output_dir=output_dir,
        metadata=metadata,
    )


def _prepare_paired_input(
    *,
    config: SecondLevelConfig,
    subjects: Sequence[str],
    task: str,
    input_root: Path,
    output_dir: Path,
) -> PreparedSecondLevelInput:
    contrast_a, contrast_b = config.contrast_names
    labels = config.condition_labels or config.contrast_names
    records_a = tuple(
        _discover_first_level_effect_size_map(
            input_root=input_root,
            subject=subject,
            task=task,
            contrast_name=contrast_a,
        )
        for subject in subjects
    )
    records_b = tuple(
        _discover_first_level_effect_size_map(
            input_root=input_root,
            subject=subject,
            task=task,
            contrast_name=contrast_b,
        )
        for subject in subjects
    )
    _validate_consistent_first_level_configs([*records_a, *records_b])

    difference_dir = output_dir / "intermediate" / "paired_differences"
    difference_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    for record_a, record_b in zip(records_a, records_b):
        difference_path = difference_dir / (
            f"{record_a.subject_label}_task-{task}_"
            f"{safe_slug(labels[1], default='b')}_minus_"
            f"{safe_slug(labels[0], default='a')}.nii.gz"
        )
        image_paths.append(
            _difference_map(
                minuend_path=record_b.map_path,
                subtrahend_path=record_a.map_path,
                output_path=difference_path,
            )
        )
        manifest_rows.append(
            {
                "subject": record_a.subject_label,
                "contrast_name_a": contrast_a,
                "contrast_name_b": contrast_b,
                "condition_label_a": labels[0],
                "condition_label_b": labels[1],
                "map_path_a": str(record_a.map_path),
                "map_path_b": str(record_b.map_path),
                "difference_map_path": str(difference_path),
            }
        )

    _validate_same_grid(image_paths)
    design_matrix = pd.DataFrame({"intercept": np.ones(len(image_paths), dtype=float)})
    metadata: Dict[str, Any] = {
        "model": "paired",
        "input_contrast_names": list(config.contrast_names),
        "condition_labels": list(labels),
        "difference_direction": f"{labels[1]} - {labels[0]}",
    }

    if config.covariates_file:
        cov_df = _load_subject_covariates(
            covariates_file=Path(config.covariates_file).expanduser().resolve(),
            subject_column=config.subject_column,
        )
        covariates = _select_covariate_columns(
            cov_df=cov_df,
            selected_subjects=subjects,
            covariate_columns=config.covariate_columns,
        )
        if not covariates.empty:
            design_matrix = pd.concat(
                [design_matrix.reset_index(drop=True), covariates.reset_index(drop=True)],
                axis=1,
            )
            metadata["covariates"] = list(covariates.columns)

    output_name = _derive_output_name(
        config=config,
        condition_labels=labels,
        default_label=f"{labels[1]}_minus_{labels[0]}",
    )
    return PreparedSecondLevelInput(
        image_paths=tuple(image_paths),
        design_matrix=design_matrix,
        manifest=pd.DataFrame(manifest_rows),
        contrast_spec=config.formula or "intercept",
        stat_type="t",
        output_name=output_name,
        output_dir=output_dir,
        metadata=metadata,
    )


def _prepare_repeated_measures_input(
    *,
    config: SecondLevelConfig,
    subjects: Sequence[str],
    task: str,
    input_root: Path,
    output_dir: Path,
) -> PreparedSecondLevelInput:
    labels = config.condition_labels or config.contrast_names
    contrast_records: dict[str, tuple[FirstLevelMapRecord, ...]] = {}
    all_records: list[FirstLevelMapRecord] = []
    for contrast_name in config.contrast_names:
        records = tuple(
            _discover_first_level_effect_size_map(
                input_root=input_root,
                subject=subject,
                task=task,
                contrast_name=contrast_name,
            )
            for subject in subjects
        )
        contrast_records[contrast_name] = records
        all_records.extend(records)

    _validate_consistent_first_level_configs(all_records)

    manifest_rows: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    design_rows: list[dict[str, float]] = []
    subject_ids = [_normalize_subject_id(subject) for subject in subjects]
    first_subject = subject_ids[0]
    condition_columns = {
        label: _safe_design_column("condition", label) for label in labels
    }

    for subject_index, subject in enumerate(subjects):
        subject_id = _normalize_subject_id(subject)
        for contrast_name, label in zip(config.contrast_names, labels):
            record = contrast_records[contrast_name][subject_index]
            image_paths.append(record.map_path)
            row = {
                column_name: float(column_name == condition_columns[label])
                for column_name in condition_columns.values()
            }
            for subject_token in subject_ids[1:]:
                row[_safe_design_column("subject", subject_token)] = float(
                    subject_id == subject_token
                )
            design_rows.append(row)
            manifest_rows.append(
                {
                    "subject": record.subject_label,
                    "contrast_name": contrast_name,
                    "condition_label": label,
                    "map_path": str(record.map_path),
                }
            )

    _validate_same_grid(image_paths)
    design_matrix = pd.DataFrame(design_rows)

    if config.formula:
        contrast_spec: Any = config.formula
        stat_type = "t"
        default_label = f"repeated_{safe_slug(config.formula, default='contrast')}"
    else:
        reference_label = labels[0]
        reference_column = condition_columns[reference_label]
        contrast_rows: list[np.ndarray] = []
        for label in labels[1:]:
            row = np.zeros(len(design_matrix.columns), dtype=float)
            row[design_matrix.columns.get_loc(condition_columns[label])] = 1.0
            row[design_matrix.columns.get_loc(reference_column)] = -1.0
            contrast_rows.append(row)
        contrast_spec = np.vstack(contrast_rows)
        stat_type = "F"
        default_label = "within_subject_omnibus"

    output_name = _derive_output_name(
        config=config,
        condition_labels=labels,
        default_label=default_label,
    )
    metadata = {
        "model": "repeated-measures",
        "input_contrast_names": list(config.contrast_names),
        "condition_labels": list(labels),
        "condition_design_columns": condition_columns,
        "subject_reference": _subject_label(first_subject),
    }

    return PreparedSecondLevelInput(
        image_paths=tuple(image_paths),
        design_matrix=design_matrix,
        manifest=pd.DataFrame(manifest_rows),
        contrast_spec=contrast_spec,
        stat_type=stat_type,
        output_name=output_name,
        output_dir=output_dir,
        metadata=metadata,
    )


def prepare_second_level_input(
    *,
    config: SecondLevelConfig,
    subjects: Sequence[str],
    task: str,
    deriv_root: Path,
) -> PreparedSecondLevelInput:
    config = config.normalized()
    if len(subjects) < 2:
        raise ValueError(
            "Second-level fMRI analysis requires at least two selected subjects."
        )

    input_root = (
        Path(config.input_root).expanduser().resolve()
        if config.input_root
        else deriv_root
    )
    if not input_root.exists():
        raise FileNotFoundError(
            f"Second-level input root does not exist: {input_root}"
        )

    tentative_output_name = _derive_output_name(
        config=config,
        condition_labels=config.condition_labels or config.contrast_names,
        default_label="second_level",
    )
    output_dir = _resolve_output_dir(
        deriv_root=deriv_root,
        task=task,
        config=config,
        output_name=tentative_output_name,
    )

    if config.model == "one-sample":
        prepared = _prepare_one_sample_input(
            config=config,
            subjects=subjects,
            task=task,
            input_root=input_root,
            output_dir=output_dir,
        )
    elif config.model == "two-sample":
        prepared = _prepare_two_sample_input(
            config=config,
            subjects=subjects,
            task=task,
            input_root=input_root,
            output_dir=output_dir,
        )
    elif config.model == "paired":
        prepared = _prepare_paired_input(
            config=config,
            subjects=subjects,
            task=task,
            input_root=input_root,
            output_dir=output_dir,
        )
    else:
        prepared = _prepare_repeated_measures_input(
            config=config,
            subjects=subjects,
            task=task,
            input_root=input_root,
            output_dir=output_dir,
        )

    if prepared.output_name != tentative_output_name:
        output_dir = _resolve_output_dir(
            deriv_root=deriv_root,
            task=task,
            config=config,
            output_name=prepared.output_name,
        )
        prepared = PreparedSecondLevelInput(
            image_paths=prepared.image_paths,
            design_matrix=prepared.design_matrix,
            manifest=prepared.manifest,
            contrast_spec=prepared.contrast_spec,
            stat_type=prepared.stat_type,
            output_name=prepared.output_name,
            output_dir=output_dir,
            metadata=prepared.metadata,
        )

    return prepared


def run_second_level_analysis(
    *,
    config: SecondLevelConfig,
    subjects: Sequence[str],
    task: str,
    deriv_root: Path,
    dry_run: bool = False,
    progress: Any = None,
) -> Dict[str, Any]:
    from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference

    prepared = prepare_second_level_input(
        config=config,
        subjects=subjects,
        task=task,
        deriv_root=deriv_root,
    )
    prepared.output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info(
            "Dry-run: would run second-level %s analysis into %s",
            config.model,
            prepared.output_dir,
        )
        return {
            "output_dir": str(prepared.output_dir),
            "output_name": prepared.output_name,
            "n_maps": len(prepared.image_paths),
        }

    if progress is not None and hasattr(progress, "step"):
        progress.step("Fit second-level GLM")

    model = SecondLevelModel()
    model = model.fit(
        second_level_input=[str(path) for path in prepared.image_paths],
        design_matrix=prepared.design_matrix,
    )
    parametric_outputs = model.compute_contrast(
        second_level_contrast=prepared.contrast_spec,
        second_level_stat_type=prepared.stat_type,
        output_type="all",
    )
    if not isinstance(parametric_outputs, dict):
        raise TypeError(
            "Expected nilearn SecondLevelModel.compute_contrast(..., output_type='all') "
            "to return a dictionary of images."
        )

    prefix = (
        f"group_task-{task}_model-{safe_slug(config.model, default='model')}"
        f"_contrast-{prepared.output_name}"
    )
    saved_maps: Dict[str, str] = {}
    for key, image in sorted(parametric_outputs.items()):
        saved_maps[key] = str(
            _save_nifti_map(
                image=image,
                output_dir=prepared.output_dir,
                prefix=prefix,
                suffix=safe_slug(key, default="map"),
            )
        )

    if bool(config.permutation.enabled):
        if prepared.stat_type != "t":
            raise ValueError(
                "Permutation inference currently supports t-contrasts only. "
                "Provide a custom repeated-measures contrast formula for a "
                "pairwise t-test instead of the default omnibus F-test."
            )
        if progress is not None and hasattr(progress, "step"):
            progress.step("Run second-level permutation inference")
        permutation_image = non_parametric_inference(
            second_level_input=[str(path) for path in prepared.image_paths],
            design_matrix=prepared.design_matrix,
            second_level_contrast=prepared.contrast_spec,
            model_intercept=False,
            n_perm=config.permutation.n_permutations,
            two_sided_test=config.permutation.two_sided,
        )
        saved_maps["permutation_logp_max_t"] = str(
            _save_nifti_map(
                image=permutation_image,
                output_dir=prepared.output_dir,
                prefix=prefix,
                suffix="logp_max_t",
            )
        )

    design_outputs: Dict[str, str] = {}
    if bool(config.write_design_matrix):
        design_outputs = _write_design_matrix_files(
            output_dir=prepared.output_dir,
            design_matrix=prepared.design_matrix,
        )

    manifest_path = _write_manifest(prepared.output_dir, prepared.manifest)
    metadata_path = _write_metadata_sidecar(
        prepared=prepared,
        config=config,
        saved_maps=saved_maps,
        manifest_path=manifest_path,
        design_outputs=design_outputs,
    )
    return {
        "output_dir": str(prepared.output_dir),
        "output_name": prepared.output_name,
        "saved_maps": saved_maps,
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "design_outputs": design_outputs,
    }


def load_second_level_config_section(config: Any) -> Dict[str, Any]:
    second_level_cfg = {}
    if hasattr(config, "get"):
        second_level_cfg = config.get("fmri_group_level", {}) or {}
    elif isinstance(config, dict):
        second_level_cfg = config.get("fmri_group_level", {}) or {}
    if not isinstance(second_level_cfg, dict):
        raise ValueError("fmri_group_level must be a mapping when provided.")
    return dict(second_level_cfg)


def load_second_level_config(config: Any) -> SecondLevelConfig:
    second_level_cfg = load_second_level_config_section(config)

    contrast_names_raw = second_level_cfg.get("contrast_names", [])
    if isinstance(contrast_names_raw, str):
        contrast_names = tuple(
            part.strip()
            for part in contrast_names_raw.replace(",", " ").split()
            if part.strip()
        )
    else:
        contrast_names = tuple(contrast_names_raw or ())

    condition_labels_raw = second_level_cfg.get("condition_labels")
    if isinstance(condition_labels_raw, str):
        condition_labels = tuple(
            part.strip()
            for part in condition_labels_raw.replace(",", " ").split()
            if part.strip()
        )
    elif condition_labels_raw is None:
        condition_labels = None
    else:
        condition_labels = tuple(condition_labels_raw or ())

    covariate_columns_raw = second_level_cfg.get("covariate_columns")
    if isinstance(covariate_columns_raw, str):
        covariate_columns = tuple(
            part.strip()
            for part in covariate_columns_raw.replace(",", " ").split()
            if part.strip()
        )
    elif covariate_columns_raw is None:
        covariate_columns = None
    else:
        covariate_columns = tuple(covariate_columns_raw or ())

    permutation_cfg = second_level_cfg.get("permutation", {}) or {}
    if not isinstance(permutation_cfg, dict):
        raise ValueError("fmri_group_level.permutation must be a mapping.")

    return SecondLevelConfig(
        model=second_level_cfg.get("model", "one-sample"),
        contrast_names=contrast_names,
        input_root=second_level_cfg.get("input_root"),
        condition_labels=condition_labels,
        formula=second_level_cfg.get("formula"),
        output_name=second_level_cfg.get("output_name"),
        output_dir=second_level_cfg.get("output_dir"),
        covariates_file=second_level_cfg.get("covariates_file"),
        subject_column=second_level_cfg.get("subject_column", "subject"),
        covariate_columns=covariate_columns,
        group_column=second_level_cfg.get("group_column"),
        group_a_value=second_level_cfg.get("group_a_value"),
        group_b_value=second_level_cfg.get("group_b_value"),
        write_design_matrix=bool(second_level_cfg.get("write_design_matrix", True)),
        permutation=SecondLevelPermutationConfig(
            enabled=bool(permutation_cfg.get("enabled", False)),
            n_permutations=int(permutation_cfg.get("n_permutations", 5000)),
            two_sided=bool(permutation_cfg.get("two_sided", True)),
        ),
    ).normalized()
