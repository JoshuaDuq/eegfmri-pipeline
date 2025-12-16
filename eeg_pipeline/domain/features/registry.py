"""
Feature Registry and Classification
===================================

Shared feature registry utilities used by behavioral correlations and decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import get_config_value, load_config


_CHANNEL_NAMES = {
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8",
    "T7", "C3", "CZ", "C4", "T8",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "P7", "P3", "PZ", "P4", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "AF3", "AF4", "AF7", "AF8",
    "F5", "F1", "F2", "F6",
    "FC5", "FC1", "FC2", "FC6",
    "C5", "C1", "C2", "C6",
    "CP5", "CP1", "CP2", "CP6",
    "P5", "P1", "P2", "P6",
    "PO5", "PO6", "PO9", "PO10",
    "CB1", "CB2",
    "T9", "T10", "TP9", "TP10",
    "O9", "O10",
    "FPZ", "AFZ",
    "FCZ", "C1", "C2", "CPZ",
    "PO1", "PO2",
    "F9", "F10",
    "FT9", "FT10",
    "TP7", "TP8", "TP9", "TP10",
    "P1", "P2", "P5", "P6", "P9", "P10",
    "PO3", "PO4", "PO7", "PO8", "POZ",
    "I1", "I2", "IZ",
}


@dataclass
class FeatureRule:
    """Pattern-based feature classification rule loaded from config."""

    label: str
    startswith: Tuple[str, ...] = field(default_factory=tuple)
    contains: Tuple[str, ...] = field(default_factory=tuple)
    regex: Optional[re.Pattern] = None
    channel_pair: bool = False
    channel_name: bool = False
    exclude_startswith: Tuple[str, ...] = field(default_factory=tuple)
    exclude_contains: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class FeatureRegistry:
    """Container for config-driven feature metadata."""

    files: Dict[str, str]
    source_to_type: Dict[str, str]
    type_hierarchy: Dict[str, Any]
    patterns: Dict[str, re.Pattern]
    classifiers: List[FeatureRule]


_FEATURE_REGISTRY_CACHE: Optional[FeatureRegistry] = None


def _load_feature_rules(rule_cfg: List[Dict[str, Any]]) -> List[FeatureRule]:
    rules: List[FeatureRule] = []
    for entry in rule_cfg:
        regex = re.compile(entry["regex"], re.IGNORECASE) if "regex" in entry else None
        rules.append(
            FeatureRule(
                label=entry["label"],
                startswith=tuple(entry.get("startswith", [])),
                contains=tuple(entry.get("contains", [])),
                regex=regex,
                channel_pair=bool(entry.get("channel_pair", False)),
                channel_name=bool(entry.get("channel_name", False)),
                exclude_startswith=tuple(entry.get("exclude_startswith", [])),
                exclude_contains=tuple(entry.get("exclude_contains", [])),
            )
        )
    return rules


def _load_feature_patterns(pattern_cfg: Dict[str, str]) -> Dict[str, re.Pattern]:
    if not pattern_cfg:
        raise ValueError("behavior_analysis.feature_registry.feature_patterns must be defined.")
    return {name: re.compile(pattern, re.IGNORECASE) for name, pattern in pattern_cfg.items()}


def load_feature_registry(config: Any) -> FeatureRegistry:
    """Load the feature registry from config with strict validation."""
    registry_cfg = get_config_value(config, "behavior_analysis.feature_registry", None)
    if not registry_cfg:
        raise ValueError("behavior_analysis.feature_registry is required in eeg_config.yaml")

    files = registry_cfg.get("files")
    if not files:
        raise ValueError("behavior_analysis.feature_registry.files is required and cannot be empty.")

    source_to_type = registry_cfg.get("source_to_feature_type", {})
    type_hierarchy = registry_cfg.get("feature_type_hierarchy", {})
    patterns = _load_feature_patterns(registry_cfg.get("feature_patterns", {}))
    classifiers = _load_feature_rules(registry_cfg.get("feature_classifiers", []))

    return FeatureRegistry(
        files=files,
        source_to_type=source_to_type,
        type_hierarchy=type_hierarchy,
        patterns=patterns,
        classifiers=classifiers,
    )


def get_feature_registry(config: Any = None) -> FeatureRegistry:
    """Return cached registry or load using provided config/default config."""
    global _FEATURE_REGISTRY_CACHE

    if config is not None:
        return load_feature_registry(config)

    if _FEATURE_REGISTRY_CACHE is None:
        _FEATURE_REGISTRY_CACHE = load_feature_registry(load_config())

    return _FEATURE_REGISTRY_CACHE


def _is_channel_pair(name: str) -> bool:
    parts = name.split("_")
    if len(parts) != 2:
        return False
    return parts[0].upper() in _CHANNEL_NAMES and parts[1].upper() in _CHANNEL_NAMES


def _rule_matches(column: str, rule: FeatureRule) -> bool:
    col_lower = column.lower()

    if rule.exclude_startswith and any(
        col_lower.startswith(p.lower()) for p in rule.exclude_startswith
    ):
        return False
    if rule.exclude_contains and any(p.lower() in col_lower for p in rule.exclude_contains):
        return False
    if rule.channel_pair and not _is_channel_pair(column):
        return False
    if rule.channel_name and column.upper() not in _CHANNEL_NAMES:
        return False
    if rule.startswith and not any(col_lower.startswith(p.lower()) for p in rule.startswith):
        return False
    if rule.contains and not any(p.lower() in col_lower for p in rule.contains):
        return False
    if rule.regex and not rule.regex.search(column):
        return False

    if not any([rule.startswith, rule.contains, rule.regex, rule.channel_pair, rule.channel_name]):
        return False
    return True


def _classify_subtype(
    column: str,
    feature_type: str,
    registry: FeatureRegistry,
    source_file_type: Optional[str] = None,
) -> str:
    col_lower = column.lower()

    if feature_type in registry.type_hierarchy:
        subtypes = registry.type_hierarchy[feature_type].get("subtypes", [])
        for subtype in subtypes:
            if subtype in col_lower:
                return subtype

    if source_file_type and source_file_type in registry.source_to_type:
        return registry.source_to_type[source_file_type]

    if feature_type == "connectivity":
        if "graph" in col_lower or col_lower.startswith("graph_"):
            return "graph"
        if "edge" in col_lower or "-" in column:
            return "edge"
        return "roi"

    if feature_type == "power":
        if "baseline" in col_lower:
            return "baseline"
        if "plateau" in col_lower:
            return "plateau"
        if "roi" in col_lower:
            return "roi"
        return "direct"

    if feature_type == "microstate":
        if "transition" in col_lower:
            return "transition"
        if col_lower.startswith("ms_"):
            return "state"
        return "summary"

    if feature_type == "pac":
        if "trial" in col_lower:
            return "trial"
        if "amp" in col_lower:
            return "amplitude"
        if "phase" in col_lower:
            return "phase"
        return "comodulogram"

    hierarchy_subtypes = registry.type_hierarchy.get(feature_type, {}).get("subtypes", [])
    if hierarchy_subtypes:
        for subtype in hierarchy_subtypes:
            if subtype in col_lower:
                return subtype

    return "unknown"


def _match_feature_patterns(
    column: str, registry: FeatureRegistry
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    patterns = registry.patterns

    for ftype, pattern in patterns.items():
        match = pattern.match(column)
        if not match:
            continue
        groups = match.groups()
        meta = {"identifier": column, "band": "N/A", "source": "inferred", "subtype": "unknown"}

        if ftype == "erds":
            meta.update({"band": groups[0], "identifier": groups[1], "channel": groups[1]})
            return "power", "erds", meta
        if ftype == "erds_windowed":
            meta.update(
                {
                    "band": groups[0],
                    "channel": groups[1],
                    "window": groups[2],
                    "identifier": f"{groups[1]}_{groups[2]}",
                }
            )
            return "power", "erds_windowed", meta
        if ftype == "relative_power":
            meta.update({"band": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "power", "relative", meta
        if ftype == "band_ratio":
            meta.update({"band": f"{groups[0]}/{groups[1]}", "channel": groups[2], "identifier": groups[2]})
            return "power", "ratio", meta
        if ftype in ("temporal_stat", "amplitude"):
            meta.update({"stat": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "temporal", groups[0], meta
        if ftype == "hjorth":
            meta.update({"param": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "complexity", "hjorth", meta
        if ftype == "roi_power":
            meta.update({"band": groups[0], "roi": groups[1], "identifier": groups[1]})
            return "power", "roi", meta
        if ftype in ("roi_asymmetry", "roi_laterality"):
            meta.update({"band": groups[0], "pair": groups[1], "identifier": groups[1]})
            return "roi", "asymmetry", meta
        if ftype == "ms_transition":
            meta.update(
                {
                    "from_state": groups[0],
                    "to_state": groups[1],
                    "identifier": f"{groups[0]}->{groups[1]}",
                }
            )
            return "microstate", "transition", meta
        if ftype.startswith("ms_"):
            meta.update({"state": groups[0], "identifier": groups[0]})
            return "microstate", ftype.replace("ms_", ""), meta
        if ftype == "itpc":
            meta.update({"band": groups[0], "channel": groups[1], "time_bin": groups[2], "identifier": groups[1]})
            return "itpc", "itpc", meta
        if ftype == "aperiodic":
            meta.update({"param": groups[0], "channel": groups[1], "identifier": groups[1], "band": "aperiodic"})
            return "aperiodic", groups[0], meta
        if ftype == "powcorr":
            meta.update({"band": groups[0], "channel": groups[1], "identifier": groups[1]})
            return "power", "correlation", meta
        if ftype == "conn_graph":
            meta.update({"measure": groups[0], "band": groups[1], "metric": groups[2], "identifier": f"{groups[0]}_{groups[2]}"})
            return "connectivity", "graph", meta
        if ftype == "gfp":
            meta.update({"metric": groups[0], "identifier": groups[0], "band": "global"})
            return "gfp", groups[0], meta
        if ftype == "power_segmented":
            segment, band, ident = groups[0], groups[1], groups[2]
            meta.update({"band": band, "identifier": ident, "segment": segment})
            return "power", segment, meta
        if ftype == "connectivity_segmented":
            measure, segment, band, ident = groups[0], groups[1], groups[2], groups[3]
            meta.update({"band": band, "identifier": ident, "measure": measure, "segment": segment})
            return "connectivity", segment, meta
        if ftype == "itpc_segmented":
            band, ch, segment = groups[0], groups[1], groups[2]
            meta.update({"band": band, "channel": ch, "segment": segment, "identifier": ch})
            return "itpc", segment, meta
        if ftype == "dynamics_burst_segmented":
            band, segment, metric = groups[0], groups[1], groups[2]
            meta.update({"band": band, "segment": segment, "metric": metric, "identifier": f"{band}_{segment}_{metric}"})
            return "dynamics", segment, meta

        meta.update({"identifier": groups[0] if groups else column})
        return ftype, "unknown", meta

    return None


def _parse_feature_metadata(column: str, ftype: str) -> Dict[str, Any]:
    parts = column.split("_")
    meta = {"identifier": column, "band": "N/A"}

    freq_bands = {
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "low_gamma",
        "high_gamma",
        "low_beta",
        "high_beta",
        "mu",
        "spindle",
    }

    band_candidates = [p.lower() for p in parts if p.lower() in freq_bands]
    if band_candidates:
        meta["band"] = band_candidates[0]

    if ftype == "power":
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])

    elif ftype == "connectivity":
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])

    elif ftype == "graph":
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])

    elif ftype == "aperiodic":
        meta["band"] = "aperiodic"
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column

    elif ftype in ("microstate", "itpc", "pac"):
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])

    elif ftype == "spectral":
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])

    elif ftype == "temporal":
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column

    elif ftype == "complexity":
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column

    elif ftype == "roi":
        if len(parts) >= 2:
            if parts[1].lower() in freq_bands:
                meta["band"] = parts[1].lower()
                meta["identifier"] = "_".join(parts[2:]) if len(parts) > 2 else "all"
            else:
                meta["identifier"] = "_".join(parts[1:])

    elif ftype == "precomputed":
        if len(parts) >= 2:
            for i, part in enumerate(parts):
                if part.lower() in freq_bands:
                    meta["band"] = part.lower()
                    remaining = parts[:i] + parts[i + 1 :]
                    meta["identifier"] = "_".join(remaining) if remaining else column
                    break
            else:
                meta["identifier"] = column

    elif ftype == "gfp":
        meta["band"] = "global"
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column

    return meta


def classify_feature(
    column: str,
    source_file_type: Optional[str] = None,
    include_subtype: bool = True,
    registry: Optional[FeatureRegistry] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify feature and extract metadata using config-driven registry."""
    if not column or not isinstance(column, str):
        meta = {
            "identifier": str(column) if column else "unknown",
            "band": "N/A",
            "source": source_file_type or "unknown",
            "subtype": "unknown",
        }
        return ("unknown", "unknown", meta) if include_subtype else ("unknown", "", meta)

    registry = registry or get_feature_registry()

    feature_type = registry.source_to_type.get(source_file_type, source_file_type)
    meta: Dict[str, Any] = {
        "identifier": column,
        "band": "N/A",
        "source": source_file_type or "inferred",
        "subtype": "unknown",
    }

    parsed = NamingSchema.parse(column)
    if parsed.get("valid"):
        schema_group = parsed["group"]
        if schema_group in {"conn", "conn_legacy"}:
            feature_type = "connectivity"
        elif schema_group == "microstates":
            feature_type = "microstate"
        elif schema_group == "asymmetry":
            feature_type = "roi"
        else:
            feature_type = schema_group
        subtype = parsed.get("segment", "unknown")

        meta.update(
            {
                "identifier": parsed.get("identifier") or parsed.get("stat") or column,
                "band": parsed.get("band", "N/A"),
                "stat": parsed.get("stat"),
                "scope": parsed.get("scope"),
                "segment": parsed.get("segment"),
                "source": source_file_type or "inferred",
            }
        )

        if feature_type == "power":
            if subtype == "baseline":
                meta["subtype"] = "baseline"
            elif subtype == "plateau":
                meta["subtype"] = "plateau"

        if feature_type == "roi" and schema_group == "asymmetry":
            meta["subtype"] = "asymmetry"
            subtype = "asymmetry"

        meta["subtype"] = subtype
        return (feature_type, subtype, meta) if include_subtype else (feature_type, "", meta)

    pattern_match = _match_feature_patterns(column, registry)
    if pattern_match:
        feature_type, subtype, meta = pattern_match
        meta["source"] = source_file_type or meta.get("source", "inferred")
        meta["subtype"] = subtype
        return (feature_type, subtype, meta) if include_subtype else (feature_type, "", meta)

    for rule in registry.classifiers:
        if _rule_matches(column, rule):
            feature_type = rule.label
            break

    feature_type = registry.source_to_type.get(feature_type, feature_type) or "unknown"
    subtype = _classify_subtype(column, feature_type, registry, source_file_type)
    meta.update(_parse_feature_metadata(column, feature_type))
    meta["subtype"] = subtype

    return (feature_type, subtype, meta) if include_subtype else (feature_type, "", meta)


__all__ = [
    "FeatureRule",
    "FeatureRegistry",
    "classify_feature",
    "get_feature_registry",
    "load_feature_registry",
    "_CHANNEL_NAMES",
]
