"""Signature-map helpers for pain EEG-BOLD coupling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import nibabel as nib
import numpy as np
from nilearn.surface import vol_to_surf

from eeg_pipeline.utils.config.loader import get_config_value
from fmri_pipeline.analysis.multivariate_signatures import discover_signature_files
from fmri_pipeline.utils.signature_paths import discover_signature_root_and_specs

_SURFACE_DEPTH_FRACTIONS = tuple(np.linspace(0.0, 1.0, 7))


def resolve_signature_paths(config: Any) -> Dict[str, Path]:
    deriv_root = get_config_value(config, "paths.deriv_root", None)
    signature_root, signature_specs = discover_signature_root_and_specs(
        config,
        deriv_root,
    )
    if signature_root is None or not signature_specs:
        return {}
    discovered = discover_signature_files(signature_root, signature_specs)
    return {
        str(name): Path(path).expanduser().resolve()
        for name, path in discovered.items()
    }


@dataclass(frozen=True)
class LocalSignatureExpressionConfig:
    enabled: bool
    signatures: Tuple[str, ...]
    abs_weight_threshold: float
    normalize_weights: str

    @classmethod
    def from_config(cls, config: Any) -> "LocalSignatureExpressionConfig":
        raw = get_config_value(
            config,
            "eeg_bold_coupling.secondary.local_signature_expression",
            {},
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                "eeg_bold_coupling.secondary.local_signature_expression must be a mapping."
            )
        normalize = str(raw.get("normalize_weights", "l1")).strip().lower()
        if normalize not in {"none", "l1", "l2"}:
            raise ValueError(
                "eeg_bold_coupling.secondary.local_signature_expression.normalize_weights "
                "must be 'none', 'l1', or 'l2'."
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            signatures=tuple(
                str(value).strip()
                for value in raw.get("signatures", [])
                if str(value).strip()
            ),
            abs_weight_threshold=float(raw.get("abs_weight_threshold", 0.0)),
            normalize_weights=normalize,
        )


def sample_signatures_to_subject_surface(
    *,
    signature_paths: Mapping[str, Path],
    subject_surface_paths: Mapping[str, Mapping[str, Path]],
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for signature_name, path in signature_paths.items():
        img = nib.load(str(path))
        out[signature_name] = {}
        for hemi in ("lh", "rh"):
            surf_paths = subject_surface_paths[hemi]
            values = vol_to_surf(
                img,
                surf_mesh=str(surf_paths["pial"]),
                inner_mesh=str(surf_paths["white"]),
                kind="depth",
                depth=_SURFACE_DEPTH_FRACTIONS,
            )
            out[signature_name][hemi] = np.asarray(values, dtype=float)
    return out


def normalize_signature_weights(
    *,
    weights: np.ndarray,
    mode: str,
) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if mode == "none":
        return arr
    if mode == "l1":
        scale = float(np.sum(np.abs(arr)))
    else:
        scale = float(np.sqrt(np.sum(arr ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Signature weights have zero norm after thresholding.")
    return arr / scale


__all__ = [
    "LocalSignatureExpressionConfig",
    "normalize_signature_weights",
    "resolve_signature_paths",
    "sample_signatures_to_subject_surface",
]
