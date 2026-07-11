"""Write the standalone Study 1 whole-brain fMRI construct-validity figure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import pandas as pd

from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.fmri_construct_models import first_level_settings
from studies.pain_study.study1.figures.fmri_construct_validity import (
    ESTIMANDS,
    build_fmri_construct_validity_summary,
)
from studies.pain_study.study1.figures.fmri_construct_validity_plot import (
    build_fmri_construct_validity_figure,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_svg,
    validity_output_dir,
)

OUTPUT_FILENAME = "fmri_construct_validity.svg"


@dataclass(frozen=True)
class FmriConstructValidityPaths:
    """All article figure, map, audit, and provenance outputs."""

    svg: Path
    subject_maps_dir: Path
    temperature_mean_effect: Path
    temperature_neg_log10_fwe_p: Path
    rating_mean_effect: Path
    rating_neg_log10_fwe_p: Path
    subjects_tsv: Path
    subjects_parquet: Path
    design_audit_tsv: Path
    design_audit_parquet: Path
    peaks_tsv: Path
    peaks_parquet: Path
    provenance_json: Path


def write_fmri_construct_validity(
    *,
    task: str,
    config: Any,
    output_path: Path | None = None,
) -> FmriConstructValidityPaths:
    """Compute validated maps, write audits, and save the final editable SVG."""

    summary = build_fmri_construct_validity_summary(task=task, config=config)
    resolved_output = output_path or validity_output_dir(config) / OUTPUT_FILENAME
    if resolved_output.suffix != ".svg":
        raise ValueError(f"Study 1 fMRI validity output must be an SVG: {resolved_output}.")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    subject_maps_dir = resolved_output.with_name("fmri_construct_validity_subject_maps")
    subject_maps_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = summary.subjects["subject_id"].astype(str).tolist()
    _write_subject_maps(
        summary.subject_effects,
        subject_ids=subject_ids,
        output_dir=subject_maps_dir,
    )
    group_paths = _write_group_maps(summary.group_maps, output_dir=resolved_output.parent)
    table_paths = _write_audit_tables(summary, output_dir=resolved_output.parent)

    figure = build_fmri_construct_validity_figure(summary, config)
    save_publication_svg(
        figure,
        resolved_output,
        config,
        dimensions_mm=require_config_value(
            config,
            "study1.figures.fmri_construct_validity.dimensions_mm",
        ),
    )
    provenance_path = resolved_output.with_name("fmri_construct_validity_provenance.json")
    _write_provenance(
        provenance_path,
        task=task,
        config=config,
        summary=summary,
        output_dir=resolved_output.parent,
    )
    return FmriConstructValidityPaths(
        svg=resolved_output,
        subject_maps_dir=subject_maps_dir,
        temperature_mean_effect=group_paths["temperature"][0],
        temperature_neg_log10_fwe_p=group_paths["temperature"][1],
        rating_mean_effect=group_paths["rating"][0],
        rating_neg_log10_fwe_p=group_paths["rating"][1],
        subjects_tsv=table_paths["subjects"][0],
        subjects_parquet=table_paths["subjects"][1],
        design_audit_tsv=table_paths["design_audit"][0],
        design_audit_parquet=table_paths["design_audit"][1],
        peaks_tsv=table_paths["peaks"][0],
        peaks_parquet=table_paths["peaks"][1],
        provenance_json=provenance_path,
    )


def _write_subject_maps(
    subject_effects,
    *,
    subject_ids: list[str],
    output_dir: Path,
) -> None:
    if set(subject_effects) != set(ESTIMANDS):
        raise ValueError("Participant map output requires both fMRI estimands.")
    for estimand in ESTIMANDS:
        images = tuple(subject_effects[estimand])
        if len(images) != len(subject_ids):
            raise ValueError(f"The {estimand} participant-map count is inconsistent.")
        for subject_id, image in zip(subject_ids, images, strict=True):
            nib.save(
                image,
                output_dir / f"{subject_id}_{estimand}_effect.nii.gz",
            )


def _write_group_maps(group_maps, *, output_dir: Path) -> dict[str, tuple[Path, Path]]:
    if set(group_maps) != set(ESTIMANDS):
        raise ValueError("Group map output requires both fMRI estimands.")
    paths: dict[str, tuple[Path, Path]] = {}
    for estimand in ESTIMANDS:
        mean_path = output_dir / f"fmri_{estimand}_mean_effect.nii.gz"
        probability_path = output_dir / f"fmri_{estimand}_neg_log10_fwe_p.nii.gz"
        nib.save(group_maps[estimand].mean_effect, mean_path)
        nib.save(group_maps[estimand].neg_log10_fwe_p, probability_path)
        paths[estimand] = (mean_path, probability_path)
    return paths


def _write_audit_tables(summary, *, output_dir: Path) -> dict[str, tuple[Path, Path]]:
    peaks = pd.concat(
        [summary.group_maps[estimand].peaks for estimand in ESTIMANDS],
        ignore_index=True,
    )
    tables = {
        "subjects": summary.subjects,
        "design_audit": summary.design_audit,
        "peaks": peaks,
    }
    paths: dict[str, tuple[Path, Path]] = {}
    for name, frame in tables.items():
        if len(frame.columns) == 0:
            raise ValueError(f"The fMRI construct-validity {name} audit has no schema.")
        stem = output_dir / f"fmri_construct_validity_{name}"
        tsv_path = stem.with_suffix(".tsv")
        parquet_path = stem.with_suffix(".parquet")
        write_tsv(frame, tsv_path)
        write_parquet(frame, parquet_path)
        paths[name] = (tsv_path, parquet_path)
    return paths


def _write_provenance(
    path: Path,
    *,
    task: str,
    config: Any,
    summary,
    output_dir: Path,
) -> None:
    source_paths = _source_paths(summary.design_audit)
    inference = require_config_value(
        config,
        "study1.figures.fmri_construct_validity.inference",
    )
    payload = {
        "task": task,
        "n_subjects": summary.n_subjects,
        "article_ready": summary.article_ready,
        "estimands": list(ESTIMANDS),
        "first_level": asdict(first_level_settings(config)),
        "inference": dict(inference),
        "software": {
            "python": platform.python_version(),
            "nibabel": importlib.metadata.version("nibabel"),
            "nilearn": importlib.metadata.version("nilearn"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "source_sha256": {str(source): _sha256(source) for source in source_paths},
        "output_sha256": {
            output.relative_to(output_dir).as_posix(): _sha256(output)
            for output in sorted(output_dir.rglob("*"))
            if output.is_file() and output != path
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_paths(design_audit: pd.DataFrame) -> tuple[Path, ...]:
    path_columns = [column for column in design_audit.columns if column.endswith("_path")]
    paths = sorted(
        {
            Path(str(value)).expanduser().resolve()
            for column in path_columns
            for value in design_audit[column].dropna()
        }
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fMRI provenance source files are missing: {missing}.")
    return tuple(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> FmriConstructValidityPaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 whole-brain fMRI construct-validity figure and audits."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--deriv-root", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study1_config_defaults(config, arguments.study1_config)
    if arguments.deriv_root is not None:
        config["paths.deriv_root"] = str(arguments.deriv_root.expanduser().resolve())
    outputs = write_fmri_construct_validity(
        task=arguments.task,
        config=config,
        output_path=arguments.output,
    )
    print(outputs.svg)
    return outputs


if __name__ == "__main__":
    main()


__all__ = [
    "FmriConstructValidityPaths",
    "main",
    "write_fmri_construct_validity",
]
