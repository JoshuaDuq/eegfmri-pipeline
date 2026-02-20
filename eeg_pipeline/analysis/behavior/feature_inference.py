from __future__ import annotations

from typing import Any, Sequence


def infer_feature_type_impl(feature: str, config: Any, *, feature_column_prefixes: Sequence[str]) -> str:
    try:
        from eeg_pipeline.domain.features.registry import classify_feature, get_feature_registry

        registry = get_feature_registry(config)
        ftype, _, _ = classify_feature(feature, include_subtype=False, registry=registry)
        return ftype
    except Exception:
        name = str(feature or "").strip().lower()
        for prefix in feature_column_prefixes:
            if name.startswith(prefix):
                return prefix.rstrip("_")
        return "unknown"


def infer_feature_band_impl(feature: str, config: Any) -> str:
    """Extract frequency band from feature name using naming schema."""
    try:
        from eeg_pipeline.domain.features.registry import classify_feature, get_feature_registry

        registry = get_feature_registry(config)
        _, _, meta = classify_feature(feature, include_subtype=True, registry=registry)
        band = meta.get("band", "N/A")
        return str(band) if band and band != "N/A" else "broadband"
    except Exception:
        name = str(feature or "").strip().lower()
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            token = f"_{band}"
            if token in name:
                return band
        return "broadband"
