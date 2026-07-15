"""Write standalone Study 1 signature-power sensor topographies."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eeg_pipeline.utils.config.loader import load_config
from studies.pain_study.study1.config.loader import apply_study1_config_defaults
from studies.pain_study.study1.figures.sensor_topography_estimands import (
    build_signature_effects,
)
from studies.pain_study.study1.figures.sensor_topography_outputs import (
    SensorTopographyPaths,
    write_sensor_topography_family,
)
from studies.pain_study.study1.figures.validity_style import validity_output_dir

OUTPUT_FILENAME = "signature_power_topographies.svg"


def write_signature_power_topographies(
    *,
    task: str,
    config: Any,
    output_path: Path | None = None,
) -> SensorTopographyPaths:
    """Publish NPS and SIIPS1 sensor-power association topographies."""

    resolved_output = output_path or validity_output_dir(config) / OUTPUT_FILENAME
    return write_sensor_topography_family(
        task=task,
        config=config,
        output_path=Path(resolved_output),
        family="signature",
        effect_builder=build_signature_effects,
    )


def main(argv: Sequence[str] | None = None) -> SensorTopographyPaths:
    parser = argparse.ArgumentParser(
        description="Write Study 1 NPS and SIIPS1 sensor-power topographies."
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
    outputs = write_signature_power_topographies(
        task=arguments.task,
        config=config,
        output_path=arguments.output,
    )
    print(outputs.svg)
    return outputs


if __name__ == "__main__":
    main()


__all__ = ["SensorTopographyPaths", "main", "write_signature_power_topographies"]
