"""Write the complete Study 2 primary cortical source figure family."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from eeg_pipeline.infra.tsv import write_tsv
from eeg_pipeline.utils.config.loader import load_config, require_config_value
from studies.pain_study.study2 import paths
from studies.pain_study.study2.config import apply_study2_config_defaults
from studies.pain_study.study2.figures.primary_source_associations import (
    CLUSTER_COLUMNS,
    SUMMARY_COLUMNS,
    VERTEX_COLUMNS,
    PrimarySourceAssociations,
    load_primary_source_associations,
)
from studies.pain_study.study2.figures.primary_source_associations_plot import (
    build_primary_source_associations_figure,
    load_common_source_surfaces,
)
from studies.pain_study.study2.figures.style import (
    save_publication_png,
    save_publication_svg,
)

FIGURE_CONFIG_KEY = "study2.figures.primary_source_associations"


@dataclass(frozen=True)
class PrimarySourceAssociationPaths:
    svg: Path
    png: Path
    vertices: Path
    clusters: Path
    summary: Path
    caption: Path
    manifest: Path

    @property
    def all_files(self) -> tuple[Path, ...]:
        return (
            self.svg,
            self.png,
            self.vertices,
            self.clusters,
            self.summary,
            self.caption,
            self.manifest,
        )


def write_primary_source_associations(
    *,
    config: Any,
    output_path: Path | None = None,
) -> PrimarySourceAssociationPaths:
    """Validate every input before writing the fixed article figure family."""

    resolved_output = output_path or paths.primary_source_figure_path(config)
    if resolved_output.suffix != ".svg":
        raise ValueError(f"Study 2 primary source output must be an SVG: {resolved_output}.")

    figure_config = _figure_config(config)
    summary = load_primary_source_associations(config)
    surfaces = load_common_source_surfaces(config, summary.vertices_manifest)
    figure = build_primary_source_associations_figure(summary, surfaces, config)
    outputs = _output_paths(resolved_output)
    outputs.svg.parent.mkdir(parents=True, exist_ok=True)

    save_publication_png(
        figure,
        outputs.png,
        dimensions_mm=figure_config["dimensions_mm"],
        font_family=str(figure_config["font_family"]),
        dpi=int(figure_config["png_dpi"]),
    )
    save_publication_svg(
        figure,
        outputs.svg,
        dimensions_mm=figure_config["dimensions_mm"],
        font_family=str(figure_config["font_family"]),
    )
    _write_table(summary.vertices, outputs.vertices)
    _write_table(summary.clusters, outputs.clusters)
    _write_table(summary.summary, outputs.summary)
    _write_text(outputs.caption, _caption(summary))
    _write_manifest(
        outputs.manifest,
        config=config,
        summary=summary,
        source_paths=(*summary.source_paths, *surfaces.source_paths),
        outputs=outputs,
    )
    return outputs


def _output_paths(svg: Path) -> PrimarySourceAssociationPaths:
    output_dir = svg.parent
    return PrimarySourceAssociationPaths(
        svg=svg,
        png=output_dir / "primary_source_associations.png",
        vertices=output_dir / "primary_source_associations_vertices.tsv",
        clusters=output_dir / "primary_source_associations_clusters.tsv",
        summary=output_dir / "primary_source_associations_summary.tsv",
        caption=output_dir / "primary_source_associations_caption.txt",
        manifest=output_dir / "primary_source_associations_manifest.json",
    )


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if len(frame.columns) == 0:
        raise ValueError(f"Study 2 primary source audit has no schema: {path.name}.")
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tsv",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        write_tsv(frame, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_text(path: Path, content: str) -> None:
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _caption(summary: PrimarySourceAssociations) -> str:
    alpha = summary.family_alpha
    cluster_p = summary.cluster_forming_p
    return (
        "Primary cortical source associations with the frozen, held-out Study 1 NPS "
        "prediction score. Maps show the equal-participant Fisher-z mean partial "
        "correlation back-transformed to r after the prespecified within-participant "
        "nuisance residualization; all cortical values are displayed without statistical "
        "thresholding. Dark contours identify clusters with target-retrained maximum-cluster "
        f"p <= {alpha:.2f} after a two-sided cluster-forming threshold of p = {cluster_p:g}, "
        "with the minimum cluster probability Holm corrected across alpha, beta, and "
        "scanner-clean gamma. These sLORETA surface estimates describe cortical spatial "
        "organization and do not identify unique or deep generators."
        "\n"
    )


def _write_manifest(
    path: Path,
    *,
    config: Any,
    summary: PrimarySourceAssociations,
    source_paths: tuple[Path, ...],
    outputs: PrimarySourceAssociationPaths,
) -> None:
    figure_config = _figure_config(config)
    missing = [source for source in source_paths if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Study 2 primary source provenance inputs are missing: {missing}.")
    output_files = tuple(output for output in outputs.all_files if output != path)
    payload = {
        "schema_version": 1,
        "bands": list(summary.bands),
        "n_subjects": summary.n_subjects,
        "n_vertices": summary.n_vertices,
        "common_subject": summary.vertices_manifest.common_subject,
        "common_source_space_spacing": summary.vertices_manifest.spacing,
        "cluster_forming_p": summary.cluster_forming_p,
        "family_alpha": summary.family_alpha,
        "display_limit": summary.display_limit,
        "figure_dimensions_mm": dict(figure_config["dimensions_mm"]),
        "software": {
            "python": platform.python_version(),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "mne": importlib.metadata.version("mne"),
            "nibabel": importlib.metadata.version("nibabel"),
            "nilearn": importlib.metadata.version("nilearn"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "source_sha256": {str(source.resolve()): _sha256(source) for source in source_paths},
        "output_sha256": {output.name: _sha256(output) for output in output_files},
    }
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _figure_config(config: Any) -> Mapping[str, object]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


def main(argv: Sequence[str] | None = None) -> PrimarySourceAssociationPaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 2 primary cortical source-association figure."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study2-config", type=Path)
    parser.add_argument("--deriv-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study2_config_defaults(config, arguments.study2_config)
    if arguments.deriv_root is not None:
        config["paths.deriv_root"] = str(arguments.deriv_root.expanduser().resolve())
    outputs = write_primary_source_associations(config=config, output_path=arguments.output)
    print(outputs.svg)
    return outputs


if __name__ == "__main__":
    main()


__all__ = [
    "CLUSTER_COLUMNS",
    "SUMMARY_COLUMNS",
    "VERTEX_COLUMNS",
    "PrimarySourceAssociationPaths",
    "main",
    "write_primary_source_associations",
]
