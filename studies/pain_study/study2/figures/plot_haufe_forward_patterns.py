"""Write the standalone Study 2 Haufe forward-pattern figure."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from eeg_pipeline.utils.config.loader import load_config, require_config_value
from studies.pain_study.study2 import paths
from studies.pain_study.study2.config import apply_study2_config_defaults
from studies.pain_study.study2.figures.haufe_forward_patterns import (
    load_haufe_forward_pattern_artifacts,
)
from studies.pain_study.study2.figures.haufe_forward_patterns_plot import (
    build_haufe_forward_patterns_figure,
)
from studies.pain_study.study2.figures.style import save_publication_svg


def write_haufe_forward_patterns(
    *,
    config: Any,
    output_path: Path | None = None,
) -> Path:
    """Read validated Study 2 artifacts and write one publication SVG."""

    summary = load_haufe_forward_pattern_artifacts(
        fold_path=paths.haufe_fold_patterns_path(config),
        aggregate_path=paths.haufe_aggregate_patterns_path(config),
        stability_path=paths.haufe_stability_path(config),
        config=config,
    )
    figure_config = require_config_value(config, "study2.figures.haufe_forward_patterns")
    if not isinstance(figure_config, Mapping):
        raise ValueError("study2.figures.haufe_forward_patterns must be a mapping.")
    figure = build_haufe_forward_patterns_figure(summary, config)
    return save_publication_svg(
        figure,
        output_path or paths.haufe_figure_path(config),
        dimensions_mm=figure_config["dimensions_mm"],
        font_family=str(figure_config["font_family"]),
    )


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Write the Study 2 NPS Haufe forward-pattern SVG.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study2-config", type=Path)
    parser.add_argument("--deriv-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    apply_study2_config_defaults(config, arguments.study2_config)
    if arguments.deriv_root is not None:
        config["paths.deriv_root"] = str(arguments.deriv_root.expanduser().resolve())
    output = write_haufe_forward_patterns(config=config, output_path=arguments.output)
    print(output)
    return output


if __name__ == "__main__":
    main()


__all__ = ["main", "write_haufe_forward_patterns"]
