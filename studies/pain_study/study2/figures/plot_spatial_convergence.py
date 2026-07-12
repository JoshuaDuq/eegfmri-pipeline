"""Write the complete Study 2 spatial-convergence publication family."""

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
from studies.pain_study.study2.figures.primary_source_associations_plot import (
    load_common_source_surfaces,
)
from studies.pain_study.study2.figures.spatial_convergence import (
    AUDIT_COLUMNS,
    SpatialConvergenceSummary,
    load_spatial_convergence,
)
from studies.pain_study.study2.figures.spatial_convergence_plot import (
    build_spatial_convergence_figure,
)
from studies.pain_study.study2.figures.style import (
    save_publication_png,
    save_publication_svg,
)

FIGURE_CONFIG_KEY = "study2.figures.spatial_convergence"


@dataclass(frozen=True)
class SpatialConvergenceFigurePaths:
    svg: Path
    png: Path
    summary: Path
    caption: Path
    manifest: Path

    @property
    def all_files(self) -> tuple[Path, ...]:
        return self.svg, self.png, self.summary, self.caption, self.manifest


def write_spatial_convergence(
    *,
    config: Any,
    output_path: Path | None = None,
) -> SpatialConvergenceFigurePaths:
    """Validate every scientific input before writing the article figure family."""

    resolved_output = output_path or paths.spatial_convergence_figure_path(config)
    if resolved_output.suffix != ".svg":
        raise ValueError(f"Study 2 spatial-convergence output must be an SVG: {resolved_output}.")

    figure_config = _figure_config(config)
    summary = load_spatial_convergence(config)
    surfaces = load_common_source_surfaces(config, summary.vertices_manifest)
    source_paths = _source_paths((*summary.source_paths, *surfaces.source_paths))
    figure = build_spatial_convergence_figure(summary, surfaces, config)
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
    _write_table(summary.audit, outputs.summary)
    _write_text(outputs.caption, _caption())
    _write_manifest(
        outputs.manifest,
        config=config,
        summary=summary,
        source_paths=source_paths,
        outputs=outputs,
    )
    return outputs


def _output_paths(svg: Path) -> SpatialConvergenceFigurePaths:
    stem = svg.stem
    return SpatialConvergenceFigurePaths(
        svg=svg,
        png=svg.with_suffix(".png"),
        summary=svg.with_name(f"{stem}_summary.tsv"),
        caption=svg.with_name(f"{stem}_caption.txt"),
        manifest=svg.with_name(f"{stem}_manifest.json"),
    )


def _source_paths(source_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    resolved_paths = tuple(path.resolve() for path in source_paths)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("Study 2 spatial-convergence provenance contains duplicate paths.")
    missing = [path for path in resolved_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Study 2 spatial-convergence provenance inputs are missing: {missing}."
        )
    return resolved_paths


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if tuple(frame.columns) != AUDIT_COLUMNS:
        raise ValueError("Study 2 spatial-convergence audit has an invalid schema.")
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


def _caption() -> str:
    return (
        "Spatial convergence between the resolution-matched, nuisance-residualized "
        "NPS-L2 fMRI forward covariance map and alpha, beta, and scanner-clean gamma "
        "EEG group source partial-r maps. Panels display unthresholded masked maps. "
        "BrainSMASH surrogate correlations preserve spatial autocorrelation and define "
        "the map-level null distributions. Titles report observed Pearson r, two-sided "
        "plus-one p, and Holm adjustment across the three bands. Results quantify coarse "
        "spatial correspondence, not vertex independence or shared generators.\n"
    )


def _write_manifest(
    path: Path,
    *,
    config: Any,
    summary: SpatialConvergenceSummary,
    source_paths: tuple[Path, ...],
    outputs: SpatialConvergenceFigurePaths,
) -> None:
    figure_config = _figure_config(config)
    output_files = tuple(output for output in outputs.all_files if output != path)
    payload = {
        "schema_version": 1,
        "bands": list(summary.bands),
        "n_vertices": summary.n_vertices,
        "n_surrogates": summary.n_surrogates,
        "common_subject": summary.vertices_manifest.common_subject,
        "common_source_space_spacing": summary.vertices_manifest.spacing,
        "family_alpha": summary.family_alpha,
        "figure_dimensions_mm": dict(figure_config["dimensions_mm"]),
        "analysis_parameters": {
            "correlation": "Pearson r",
            "multiple_comparison_adjustment": "Holm across bands",
            "null_method": "BrainSMASH Base",
            "p_value": "two-sided plus-one",
        },
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
        "source_sha256": {str(source): _sha256(source) for source in source_paths},
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


def main(argv: Sequence[str] | None = None) -> SpatialConvergenceFigurePaths:
    parser = argparse.ArgumentParser(
        description="Write the Study 2 spatial-convergence publication figure."
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
    outputs = write_spatial_convergence(config=config, output_path=arguments.output)
    print(outputs.svg)
    return outputs


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_COLUMNS",
    "SpatialConvergenceFigurePaths",
    "main",
    "write_spatial_convergence",
]
