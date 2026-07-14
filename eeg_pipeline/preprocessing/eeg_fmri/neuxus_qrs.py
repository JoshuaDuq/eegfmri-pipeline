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
from typing import Mapping, Protocol

import numpy as np
from numba import njit
from scipy.signal import butter, resample_poly, sosfiltfilt

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


class QrsWindowPredictor(Protocol):
    """Probability-model interface consumed by the offline detector."""

    window_samples: int
    model_sha256: str

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Return one R-peak probability per input sample."""


@dataclass(frozen=True)
class NeuXusQrsDetectionParameters:
    """Fixed signal-conditioning and peak-consolidation parameters."""

    sampling_frequency_hz: float = 250.0
    low_frequency_hz: float = 0.5
    high_frequency_hz: float = 30.0
    window_stride_samples: int = 50
    probability_threshold: float = 0.05
    minimum_support_samples: int = 5
    refractory_period_seconds: float = 0.4
    edge_margin_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        if self.low_frequency_hz <= 0:
            raise ValueError("low_frequency_hz must be positive")
        if self.high_frequency_hz <= self.low_frequency_hz:
            raise ValueError("high_frequency_hz must exceed low_frequency_hz")
        if self.high_frequency_hz >= self.sampling_frequency_hz / 2.0:
            raise ValueError("high_frequency_hz must be below the detection Nyquist frequency")
        if self.window_stride_samples <= 0:
            raise ValueError("window_stride_samples must be positive")
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError("probability_threshold must be between zero and one")
        if self.minimum_support_samples < 1:
            raise ValueError("minimum_support_samples must be positive")
        if self.refractory_period_seconds <= 0:
            raise ValueError("refractory_period_seconds must be positive")
        if self.edge_margin_seconds < 0:
            raise ValueError("edge_margin_seconds must be non-negative")


@dataclass(frozen=True)
class NeuXusQrsDetection:
    """R-peak times and complete detection diagnostics."""

    times: np.ndarray
    peak_samples: np.ndarray
    filtered_ecg: np.ndarray
    probabilities: np.ndarray
    probability_support: np.ndarray
    sampling_frequency_hz: float
    model_sha256: str


@dataclass(frozen=True)
class NeuXusQrsPredictor:
    """Validated model exposed through the detector's narrow interface."""

    model: NeuXusQrsModel

    @property
    def window_samples(self) -> int:
        return self.model.window_samples

    @property
    def model_sha256(self) -> str:
        return self.model.sha256

    def predict(self, window: np.ndarray) -> np.ndarray:
        return predict_neuxus_qrs(window, self.model)


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


def _window_starts(n_samples: int, window_samples: int, stride_samples: int) -> np.ndarray:
    if n_samples < window_samples:
        raise ValueError(
            f"NeuXus detection requires at least {window_samples} samples, found {n_samples}"
        )
    starts = list(range(0, n_samples - window_samples + 1, stride_samples))
    final_start = n_samples - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return np.asarray(starts, dtype=int)


def _normalize_window(window: np.ndarray) -> np.ndarray:
    minimum = float(np.min(window))
    span = float(np.max(window) - minimum)
    if span <= np.finfo(np.float32).eps:
        raise ValueError("NeuXus ECG window is flat")
    return np.asarray(2.0 * (window - minimum) / span - 1.0, dtype=np.float32)


def _average_probabilities(
    ecg: np.ndarray,
    predictor: QrsWindowPredictor,
    parameters: NeuXusQrsDetectionParameters,
) -> tuple[np.ndarray, np.ndarray]:
    probability_sum = np.zeros(ecg.size, dtype=np.float64)
    support = np.zeros(ecg.size, dtype=np.int32)
    margin_samples = int(round(parameters.edge_margin_seconds * parameters.sampling_frequency_hz))
    valid_window_samples = predictor.window_samples - margin_samples
    if valid_window_samples <= 0:
        raise ValueError("edge_margin_seconds excludes the complete NeuXus window")
    starts = _window_starts(
        ecg.size,
        predictor.window_samples,
        parameters.window_stride_samples,
    )
    for start in starts:
        window = _normalize_window(ecg[start : start + predictor.window_samples])
        probabilities = np.asarray(predictor.predict(window))
        if probabilities.shape != (predictor.window_samples,):
            raise ValueError(
                "NeuXus predictor returned shape "
                f"{probabilities.shape}, expected {(predictor.window_samples,)}"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("NeuXus predictor returned non-finite probabilities")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("NeuXus predictor returned probabilities outside zero to one")
        stop = start + valid_window_samples
        probability_sum[start:stop] += probabilities[:valid_window_samples]
        support[start:stop] += 1
    averaged = np.zeros(ecg.size, dtype=np.float32)
    observed = support > 0
    averaged[observed] = (probability_sum[observed] / support[observed]).astype(np.float32)
    return averaged, support


def _supported_regions(mask: np.ndarray, minimum_samples: int) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum_samples
    ]


def _snap_to_local_maximum(ecg: np.ndarray, candidate: int, radius: int) -> int:
    start = max(0, candidate - radius)
    stop = min(ecg.size, candidate + radius + 1)
    return start + int(np.argmax(ecg[start:stop]))


def _enforce_refractory_period(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    ecg: np.ndarray,
    minimum_distance_samples: int,
) -> np.ndarray:
    retained: list[int] = []
    for candidate in candidates:
        if not retained or candidate - retained[-1] >= minimum_distance_samples:
            retained.append(int(candidate))
            continue
        previous = retained[-1]
        previous_score = (float(probabilities[previous]), float(ecg[previous]))
        candidate_score = (float(probabilities[candidate]), float(ecg[candidate]))
        if candidate_score > previous_score:
            retained[-1] = int(candidate)
    return np.asarray(retained, dtype=int)


def _consolidate_peaks(
    ecg: np.ndarray,
    probabilities: np.ndarray,
    parameters: NeuXusQrsDetectionParameters,
) -> np.ndarray:
    regions = _supported_regions(
        probabilities > parameters.probability_threshold,
        parameters.minimum_support_samples,
    )
    candidates = []
    for start, stop in regions:
        probability_peak = start + int(np.argmax(probabilities[start:stop]))
        candidates.append(_snap_to_local_maximum(ecg, probability_peak, radius=5))
    if not candidates:
        return np.empty(0, dtype=int)
    unique_candidates = np.unique(np.asarray(candidates, dtype=int))
    minimum_distance = int(
        round(parameters.refractory_period_seconds * parameters.sampling_frequency_hz)
    )
    return _enforce_refractory_period(
        unique_candidates,
        probabilities,
        ecg,
        minimum_distance,
    )


def _immutable(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NeuXusQrsDetector:
    """Condition ECG and detect R-peaks with overlapping NeuXus windows."""

    predictor: QrsWindowPredictor
    parameters: NeuXusQrsDetectionParameters

    def __post_init__(self) -> None:
        if self.parameters.window_stride_samples >= self.predictor.window_samples:
            raise ValueError("window_stride_samples must be smaller than the model window")

    def detect(
        self,
        ecg: np.ndarray,
        *,
        sampling_frequency_hz: float,
    ) -> NeuXusQrsDetection:
        values = np.asarray(ecg)
        if values.ndim != 1:
            raise ValueError("ECG must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError("ECG must contain only finite values")
        ratio = sampling_frequency_hz / self.parameters.sampling_frequency_hz
        downsampling = int(round(ratio))
        if downsampling < 1 or not np.isclose(ratio, downsampling, rtol=0.0, atol=1e-9):
            raise ValueError(
                "ECG sampling frequency must be an integer multiple of the NeuXus rate"
            )
        if np.ptp(values) <= np.finfo(float).eps:
            raise ValueError("ECG is flat")
        sos = butter(
            4,
            [
                self.parameters.low_frequency_hz,
                self.parameters.high_frequency_hz,
            ],
            btype="bandpass",
            fs=sampling_frequency_hz,
            output="sos",
        )
        filtered = sosfiltfilt(sos, values)
        detection_ecg = resample_poly(filtered, up=1, down=downsampling)
        probabilities, support = _average_probabilities(
            detection_ecg,
            self.predictor,
            self.parameters,
        )
        peak_samples = _consolidate_peaks(
            detection_ecg,
            probabilities,
            self.parameters,
        )
        times = peak_samples.astype(float) / self.parameters.sampling_frequency_hz
        return NeuXusQrsDetection(
            times=_immutable(times),
            peak_samples=_immutable(peak_samples),
            filtered_ecg=_immutable(detection_ecg),
            probabilities=_immutable(probabilities),
            probability_support=_immutable(support),
            sampling_frequency_hz=self.parameters.sampling_frequency_hz,
            model_sha256=self.predictor.model_sha256,
        )
