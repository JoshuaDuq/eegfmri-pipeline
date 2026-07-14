"""Convert the pinned NeuXus QRS pickle to a non-executable NumPy archive."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import (
    NEUXUS_MODEL_HIDDEN_UNITS,
    NEUXUS_MODEL_WINDOW_SAMPLES,
    NEUXUS_WEIGHT_SHAPES,
)


def convert_weights(source_path: Path, output_path: Path) -> None:
    """Validate the trusted upstream pickle and write only numeric arrays."""
    with source_path.open("rb") as stream:
        upstream = pickle.load(stream)  # noqa: S301 - development-only pinned upstream asset
    expected_keys = set(NEUXUS_WEIGHT_SHAPES) | {"t", "u"}
    if set(upstream) != expected_keys:
        raise ValueError("Upstream NeuXus model keys do not match the pinned schema")
    if upstream["t"] != NEUXUS_MODEL_WINDOW_SAMPLES:
        raise ValueError("Unexpected NeuXus model window length")
    if upstream["u"] != NEUXUS_MODEL_HIDDEN_UNITS:
        raise ValueError("Unexpected NeuXus model hidden-unit count")

    weights = {name: np.asarray(upstream[name]) for name in NEUXUS_WEIGHT_SHAPES}
    for name, expected_shape in NEUXUS_WEIGHT_SHAPES.items():
        if weights[name].shape != expected_shape or weights[name].dtype != np.float32:
            raise ValueError(f"Invalid upstream NeuXus weight: {name}")
        if not np.all(np.isfinite(weights[name])):
            raise ValueError(f"Non-finite upstream NeuXus weight: {name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    convert_weights(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
