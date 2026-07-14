# SPDX-License-Identifier: GPL-3.0-only
"""Offline adaptation of the GPL-licensed NeuXus EEG-fMRI QRS detector.

LSTM inference is adapted from LaSEEB/NeuXus v0.0.4 and modified for deterministic
offline array processing. See ``THIRD_PARTY_NOTICES.md`` for full provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numba import njit

NEUXUS_MODEL_WINDOW_SAMPLES = 500
NEUXUS_MODEL_HIDDEN_UNITS = 64


def _weight_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for layer, input_units in ((1, 1), (2, 128)):
        for direction in ("f", "b"):
            for gate in ("i", "f", "l", "o"):
                shapes[f"wx{gate}{layer}{direction}"] = (
                    input_units,
                    NEUXUS_MODEL_HIDDEN_UNITS,
                )
                shapes[f"wh{gate}{layer}{direction}"] = (
                    NEUXUS_MODEL_HIDDEN_UNITS,
                    NEUXUS_MODEL_HIDDEN_UNITS,
                )
                shapes[f"b{gate}{layer}{direction}"] = (1, NEUXUS_MODEL_HIDDEN_UNITS)
    shapes["wd"] = (2 * NEUXUS_MODEL_HIDDEN_UNITS, 1)
    shapes["bd"] = (1, 1)
    return shapes


NEUXUS_WEIGHT_SHAPES = MappingProxyType(_weight_shapes())


@dataclass(frozen=True)
class NeuXusQrsModel:
    """Validated immutable NeuXus model parameters."""

    weights: Mapping[str, np.ndarray]
    sha256: str
    window_samples: int = NEUXUS_MODEL_WINDOW_SAMPLES
    hidden_units: int = NEUXUS_MODEL_HIDDEN_UNITS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_weights(weights: Mapping[str, np.ndarray]) -> None:
    expected_keys = set(NEUXUS_WEIGHT_SHAPES)
    observed_keys = set(weights)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(f"Invalid NeuXus model keys; missing={missing}, unexpected={unexpected}")
    for name, expected_shape in NEUXUS_WEIGHT_SHAPES.items():
        weight = weights[name]
        if weight.shape != expected_shape:
            raise ValueError(
                f"NeuXus model weight {name!r} has shape {weight.shape}, "
                f"expected {expected_shape}"
            )
        if weight.dtype != np.float32:
            raise ValueError(f"NeuXus model weight {name!r} must use float32, found {weight.dtype}")
        if not np.all(np.isfinite(weight)):
            raise ValueError(f"NeuXus model weight {name!r} contains non-finite values")


def load_neuxus_qrs_model(
    archive_path: str | Path,
    *,
    expected_sha256: str,
) -> NeuXusQrsModel:
    """Load and validate one non-executable NeuXus model archive."""
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(f"NeuXus model archive does not exist: {path}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"NeuXus model SHA-256 mismatch: expected {expected_sha256}, "
            f"found {observed_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        weights = {name: np.array(archive[name], copy=True) for name in archive.files}
    _validate_weights(weights)
    for weight in weights.values():
        weight.setflags(write=False)
    return NeuXusQrsModel(
        weights=MappingProxyType(weights),
        sha256=observed_sha256,
    )


def load_packaged_neuxus_qrs_model() -> NeuXusQrsModel:
    """Load the pinned NeuXus model distributed with this package."""
    assets = files(__package__).joinpath("assets")
    expected_sha256 = assets.joinpath("neuxus_qrs_weights.sha256").read_text().strip()
    with as_file(assets.joinpath("neuxus_qrs_weights.npz")) as archive_path:
        return load_neuxus_qrs_model(archive_path, expected_sha256=expected_sha256)


@njit(cache=True)
def _predict_lstm(
    inputs,
    hidden_template,
    cell_template,
    whf1f,
    wxf1f,
    bf1f,
    whi1f,
    wxi1f,
    bi1f,
    whl1f,
    wxl1f,
    bl1f,
    who1f,
    wxo1f,
    bo1f,
    whf1b,
    wxf1b,
    bf1b,
    whi1b,
    wxi1b,
    bi1b,
    whl1b,
    wxl1b,
    bl1b,
    who1b,
    wxo1b,
    bo1b,
    whf2f,
    wxf2f,
    bf2f,
    whi2f,
    wxi2f,
    bi2f,
    whl2f,
    wxl2f,
    bl2f,
    who2f,
    wxo2f,
    bo2f,
    whf2b,
    wxf2b,
    bf2b,
    whi2b,
    wxi2b,
    bi2b,
    whl2b,
    wxl2b,
    bl2b,
    who2b,
    wxo2b,
    bo2b,
    dense_weights,
    dense_bias,
):
    def sigmoid(values):
        one = np.float32(1.0)
        return one / (one + np.exp(-values))

    def cell(
        values,
        hidden,
        state,
        forget_hidden,
        forget_input,
        forget_bias,
        input_hidden,
        input_input,
        input_bias,
        candidate_hidden,
        candidate_input,
        candidate_bias,
        output_hidden,
        output_input,
        output_bias,
    ):
        forget_gate = sigmoid(hidden @ forget_hidden + values @ forget_input + forget_bias)
        input_gate = sigmoid(hidden @ input_hidden + values @ input_input + input_bias)
        candidate = np.tanh(hidden @ candidate_hidden + values @ candidate_input + candidate_bias)
        next_state = state * forget_gate + input_gate * candidate
        output_gate = sigmoid(hidden @ output_hidden + values @ output_input + output_bias)
        return next_state, np.tanh(next_state) * output_gate

    def forward_layer(
        values,
        hidden_values,
        state,
        forget_hidden,
        forget_input,
        forget_bias,
        input_hidden,
        input_input,
        input_bias,
        candidate_hidden,
        candidate_input,
        candidate_bias,
        output_hidden,
        output_input,
        output_bias,
    ):
        hidden = hidden_values[-1:]
        for index in range(values.shape[0]):
            state, hidden = cell(
                values[index : index + 1],
                hidden,
                state,
                forget_hidden,
                forget_input,
                forget_bias,
                input_hidden,
                input_input,
                input_bias,
                candidate_hidden,
                candidate_input,
                candidate_bias,
                output_hidden,
                output_input,
                output_bias,
            )
            hidden_values[index] = hidden
        return hidden_values

    def backward_layer(
        values,
        hidden_values,
        state,
        forget_hidden,
        forget_input,
        forget_bias,
        input_hidden,
        input_input,
        input_bias,
        candidate_hidden,
        candidate_input,
        candidate_bias,
        output_hidden,
        output_input,
        output_bias,
    ):
        hidden = hidden_values[:1]
        for index in range(values.shape[0] - 1, -1, -1):
            state, hidden = cell(
                values[index : index + 1],
                hidden,
                state,
                forget_hidden,
                forget_input,
                forget_bias,
                input_hidden,
                input_input,
                input_bias,
                candidate_hidden,
                candidate_input,
                candidate_bias,
                output_hidden,
                output_input,
                output_bias,
            )
            hidden_values[index] = hidden
        return hidden_values

    forward = forward_layer(
        inputs,
        hidden_template.copy(),
        cell_template.copy(),
        whf1f,
        wxf1f,
        bf1f,
        whi1f,
        wxi1f,
        bi1f,
        whl1f,
        wxl1f,
        bl1f,
        who1f,
        wxo1f,
        bo1f,
    )
    backward = backward_layer(
        inputs,
        hidden_template.copy(),
        cell_template.copy(),
        whf1b,
        wxf1b,
        bf1b,
        whi1b,
        wxi1b,
        bi1b,
        whl1b,
        wxl1b,
        bl1b,
        who1b,
        wxo1b,
        bo1b,
    )
    first_layer = np.concatenate((forward, backward), axis=1)
    forward = forward_layer(
        first_layer,
        hidden_template.copy(),
        cell_template.copy(),
        whf2f,
        wxf2f,
        bf2f,
        whi2f,
        wxi2f,
        bi2f,
        whl2f,
        wxl2f,
        bl2f,
        who2f,
        wxo2f,
        bo2f,
    )
    backward = backward_layer(
        first_layer,
        hidden_template.copy(),
        cell_template.copy(),
        whf2b,
        wxf2b,
        bf2b,
        whi2b,
        wxi2b,
        bi2b,
        whl2b,
        wxl2b,
        bl2b,
        who2b,
        wxo2b,
        bo2b,
    )
    second_layer = np.concatenate((forward, backward), axis=1)
    return sigmoid(second_layer @ dense_weights + dense_bias)[:, 0]


def predict_neuxus_qrs(window: np.ndarray, model: NeuXusQrsModel) -> np.ndarray:
    """Estimate an R-peak probability for every sample in one model window."""
    values = np.asarray(window)
    expected_shape = (model.window_samples,)
    if values.shape != expected_shape:
        raise ValueError(f"NeuXus input window has shape {values.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("NeuXus input window must contain only finite values")
    inputs = values.astype(np.float32, copy=False)[:, np.newaxis]
    hidden_template = np.zeros(
        (model.window_samples, model.hidden_units),
        dtype=np.float32,
    )
    cell_template = np.zeros((1, model.hidden_units), dtype=np.float32)
    weights = model.weights
    probabilities = _predict_lstm(
        inputs,
        hidden_template,
        cell_template,
        weights["whf1f"],
        weights["wxf1f"],
        weights["bf1f"],
        weights["whi1f"],
        weights["wxi1f"],
        weights["bi1f"],
        weights["whl1f"],
        weights["wxl1f"],
        weights["bl1f"],
        weights["who1f"],
        weights["wxo1f"],
        weights["bo1f"],
        weights["whf1b"],
        weights["wxf1b"],
        weights["bf1b"],
        weights["whi1b"],
        weights["wxi1b"],
        weights["bi1b"],
        weights["whl1b"],
        weights["wxl1b"],
        weights["bl1b"],
        weights["who1b"],
        weights["wxo1b"],
        weights["bo1b"],
        weights["whf2f"],
        weights["wxf2f"],
        weights["bf2f"],
        weights["whi2f"],
        weights["wxi2f"],
        weights["bi2f"],
        weights["whl2f"],
        weights["wxl2f"],
        weights["bl2f"],
        weights["who2f"],
        weights["wxo2f"],
        weights["bo2f"],
        weights["whf2b"],
        weights["wxf2b"],
        weights["bf2b"],
        weights["whi2b"],
        weights["wxi2b"],
        weights["bi2b"],
        weights["whl2b"],
        weights["wxl2b"],
        weights["bl2b"],
        weights["who2b"],
        weights["wxo2b"],
        weights["bo2b"],
        weights["wd"],
        weights["bd"],
    )
    return np.asarray(probabilities, dtype=np.float32)
