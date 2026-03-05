from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from eeg_pipeline.utils.config.loader import get_config_value


def filter_feature_cols_by_band_impl(
    feature_cols: List[str],
    ctx: Any,
    *,
    feature_column_prefixes: Sequence[str],
) -> List[str]:
    """Filter feature columns to only include user-selected bands."""
    if not ctx.selected_bands:
        return feature_cols

    from eeg_pipeline.domain.features.naming import NamingSchema

    prefixes = sorted(feature_column_prefixes, key=len, reverse=True)
    selected = set(b.lower() for b in ctx.selected_bands)
    filtered: List[str] = []

    for col in feature_cols:
        col_str = str(col)
        parsed = NamingSchema.parse(col_str)
        if not parsed.get("valid"):
            matched_prefix = next((p for p in prefixes if col_str.startswith(p)), None)
            if not matched_prefix:
                raise ValueError(
                    f"Band filter: cannot parse feature column {col_str!r} "
                    f"(selected_bands={sorted(selected)})"
                )

            candidate = col_str[len(matched_prefix) :]
            parsed2 = NamingSchema.parse(candidate)
            if not parsed2.get("valid"):
                raise ValueError(
                    f"Band filter: cannot parse feature column {col_str!r} "
                    f"after stripping prefix {matched_prefix!r} -> {candidate!r} "
                    f"(selected_bands={sorted(selected)})"
                )
            parsed = parsed2

        band = parsed.get("band")
        if not band:
            filtered.append(col)
        elif str(band).lower() in selected:
            filtered.append(col)

    if len(filtered) < len(feature_cols):
        ctx.logger.info(
            "Band filter: kept %d/%d features for bands: %s",
            len(filtered),
            len(feature_cols),
            ", ".join(sorted(selected)),
        )

    return filtered


def filter_feature_cols_for_computation_impl(
    feature_cols: List[str],
    computation_name: str,
    ctx: Any,
    *,
    category_prefix_map: Dict[str, str],
) -> List[str]:
    """Filter feature columns based on per-computation feature selection."""
    if not ctx.computation_features or computation_name not in ctx.computation_features:
        return feature_cols

    selected_features = ctx.computation_features[computation_name]
    if not selected_features:
        return feature_cols

    allowed_prefixes = tuple(category_prefix_map[cat] for cat in selected_features if cat in category_prefix_map)
    if not allowed_prefixes:
        ctx.logger.warning(
            "Computation '%s' has feature filter %s but no matching prefixes found. Using all features.",
            computation_name,
            selected_features,
        )
        return feature_cols

    filtered = [c for c in feature_cols if str(c).startswith(allowed_prefixes)]
    if len(filtered) < len(feature_cols):
        ctx.logger.info(
            "%s: filtered features to %s (%d/%d kept)",
            computation_name,
            selected_features,
            len(filtered),
            len(feature_cols),
        )

    return filtered


def primary_unit_for_computation_impl(ctx: Any, computation_name: Optional[str]) -> str:
    mapping = {
        "correlations": "behavior_analysis.correlations.primary_unit",
        "regression": "behavior_analysis.regression.primary_unit",
        "condition": "behavior_analysis.condition.primary_unit",
        "condition_window_comparison": "behavior_analysis.condition.window_comparison.primary_unit",
    }
    key = mapping.get(str(computation_name or "").strip().lower(), None)
    if key is None:
        return "trial"
    return str(get_config_value(ctx.config, key, "trial") or "trial").strip().lower()


def filter_feature_cols_by_provenance_impl(
    feature_cols: List[str],
    ctx: Any,
    computation_name: Optional[str] = None,
    *,
    feature_column_prefixes: Sequence[str],
) -> List[str]:
    """Exclude non-i.i.d./broadcast features when performing trial-wise analyses."""
    if not feature_cols:
        return feature_cols

    primary_unit = primary_unit_for_computation_impl(ctx, computation_name)
    is_trial_unit = primary_unit in {"trial", "trial_level", "trialwise"}
    if not is_trial_unit:
        return feature_cols

    enabled = bool(
        get_config_value(
            ctx.config,
            "behavior_analysis.features.exclude_non_trialwise_features",
            True,
        )
    )
    if not enabled:
        return feature_cols

    from eeg_pipeline.domain.features.naming import infer_feature_provenance

    manifests = getattr(ctx, "feature_manifests", {}) or {}
    dropped: List[str] = []
    kept: List[str] = []

    prefixes = sorted(feature_column_prefixes, key=len, reverse=True)

    for col in feature_cols:
        col_str = str(col)
        matched_prefix = next((p for p in prefixes if col_str.startswith(p)), None)
        if matched_prefix is None:
            kept.append(col_str)
            continue

        table_key = matched_prefix.rstrip("_")
        raw_name = col_str[len(matched_prefix) :]

        manifest = manifests.get(table_key) or {}
        prov_cols = (manifest.get("provenance") or {}).get("columns") or {}

        props = prov_cols.get(raw_name)
        if props is None:
            props = prov_cols.get(col_str)

        if props is None:
            inferred = infer_feature_provenance(
                feature_columns=[raw_name],
                config=ctx.config,
                df_attrs={},
            )
            props = (inferred.get("columns") or {}).get(raw_name, {})
            if not props:
                inferred = infer_feature_provenance(
                    feature_columns=[col_str],
                    config=ctx.config,
                    df_attrs={},
                )
                props = (inferred.get("columns") or {}).get(col_str, {})

        trialwise_valid = bool(props.get("trialwise_valid", True))
        broadcasted = bool(props.get("broadcasted", False))
        if (not trialwise_valid) or broadcasted:
            dropped.append(col_str)
        else:
            kept.append(col_str)

    if dropped and ctx.logger is not None:
        ctx.logger.warning(
            "%s: excluded %d/%d non-trialwise (broadcast/cross-trial) features because primary_unit='trial'. Examples=%s",
            computation_name or "analysis",
            len(dropped),
            len(feature_cols),
            ",".join(dropped[:5]),
        )

    return kept
