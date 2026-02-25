"""
Feature Registry and Classification
===================================

Shared feature registry utilities used by behavioral correlations and machine learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import get_config_value, load_config


_CHANNEL_NAMES = {
    "FP1", "FP2", "FPZ",
    "F7", "F3", "FZ", "F4", "F8", "F5", "F1", "F2", "F6", "F9", "F10",
    "FT7", "FC3", "FCZ", "FC4", "FT8", "FC5", "FC1", "FC2", "FC6", "FT9", "FT10",
    "T7", "C3", "CZ", "C4", "T8", "C5", "C1", "C2", "C6", "T9", "T10",
    "TP7", "CP3", "CPZ", "CP4", "TP8", "CP5", "CP1", "CP2", "CP6", "TP9", "TP10",
    "P7", "P3", "PZ", "P4", "P8", "P5", "P1", "P2", "P6", "P9", "P10",
    "PO7", "PO3", "POZ", "PO4", "PO8", "PO5", "PO6", "PO9", "PO10", "PO1", "PO2",
    "O1", "OZ", "O2", "O9", "O10",
    "AF3", "AF4", "AF7", "AF8", "AFZ",
    "CB1", "CB2",
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
    """Load feature classification rules from configuration."""
    if not rule_cfg:
        return []

    rules: List[FeatureRule] = []
    for entry in rule_cfg:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid rule entry: expected dict, got {type(entry)}")
        if "label" not in entry:
            raise ValueError("Rule entry must have 'label' field")

        regex_pattern = entry.get("regex")
        regex = re.compile(regex_pattern, re.IGNORECASE) if regex_pattern else None

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
    """Load and compile feature name patterns from configuration."""
    if not pattern_cfg:
        raise ValueError("behavior_analysis.feature_registry.feature_patterns must be defined.")

    patterns: Dict[str, re.Pattern] = {}
    for name, pattern_str in pattern_cfg.items():
        if not isinstance(pattern_str, str):
            raise ValueError(f"Pattern '{name}' must be a string, got {type(pattern_str)}")
        try:
            patterns[name] = re.compile(pattern_str, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{name}': {e}") from e

    return patterns


def load_feature_registry(config: Any) -> FeatureRegistry:
    """Load the feature registry from config with strict validation."""
    if config is None:
        raise ValueError("config cannot be None")

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
    """Check if name represents a channel pair (e.g., 'F3_F4')."""
    parts = name.split("_")
    if len(parts) != 2:
        return False
    first_channel = parts[0].upper()
    second_channel = parts[1].upper()
    return first_channel in _CHANNEL_NAMES and second_channel in _CHANNEL_NAMES


def _matches_exclusion_criteria(column_lower: str, rule: FeatureRule) -> bool:
    """Check if column matches any exclusion criteria."""
    if rule.exclude_startswith and any(column_lower.startswith(pattern.lower()) for pattern in rule.exclude_startswith):
        return True
    if rule.exclude_contains and any(pattern.lower() in column_lower for pattern in rule.exclude_contains):
        return True
    return False


def _matches_inclusion_criteria(column: str, column_lower: str, rule: FeatureRule) -> bool:
    """Check if column matches any inclusion criteria."""
    if rule.channel_pair and not _is_channel_pair(column):
        return False
    if rule.channel_name and column.upper() not in _CHANNEL_NAMES:
        return False
    if rule.startswith and not any(column_lower.startswith(pattern.lower()) for pattern in rule.startswith):
        return False
    if rule.contains and not any(pattern.lower() in column_lower for pattern in rule.contains):
        return False
    if rule.regex and not rule.regex.search(column):
        return False
    return True


def _rule_matches(column: str, rule: FeatureRule) -> bool:
    """Check if column matches the feature classification rule."""
    if not column:
        return False

    column_lower = column.lower()

    if _matches_exclusion_criteria(column_lower, rule):
        return False

    if not (rule.startswith or rule.contains or rule.regex or rule.channel_pair or rule.channel_name):
        return False

    return _matches_inclusion_criteria(column, column_lower, rule)


def _classify_connectivity_subtype(column: str, column_lower: str) -> str:
    """Classify connectivity feature subtype."""
    if "graph" in column_lower or column_lower.startswith("graph_"):
        return "graph"
    if "edge" in column_lower or "-" in column:
        return "edge"
    return "roi"


def _classify_power_subtype(column_lower: str) -> str:
    """Classify power feature subtype."""
    if "roi" in column_lower:
        return "roi"
    return "direct"


def _classify_pac_subtype(column_lower: str) -> str:
    """Classify PAC feature subtype."""
    if "trial" in column_lower:
        return "trial"
    if "amp" in column_lower:
        return "amplitude"
    if "phase" in column_lower:
        return "phase"
    return "comodulogram"


def _check_type_hierarchy_subtypes(
    column_lower: str, feature_type: str, registry: FeatureRegistry
) -> Optional[str]:
    """Check if column matches any subtype in type hierarchy."""
    if feature_type in registry.type_hierarchy:
        subtypes = registry.type_hierarchy[feature_type].get("subtypes", [])
        for subtype in subtypes:
            if subtype in column_lower:
                return subtype
    return None


def _classify_subtype(
    column: str,
    feature_type: str,
    registry: FeatureRegistry,
    source_file_type: Optional[str] = None,
) -> str:
    """Classify feature subtype based on column name and registry."""
    if not column:
        return "unknown"

    column_lower = column.lower()

    hierarchy_match = _check_type_hierarchy_subtypes(column_lower, feature_type, registry)
    if hierarchy_match:
        return hierarchy_match

    if source_file_type and source_file_type in registry.source_to_type:
        return registry.source_to_type[source_file_type]

    if feature_type == "connectivity":
        return _classify_connectivity_subtype(column, column_lower)
    if feature_type == "power":
        return _classify_power_subtype(column_lower)
    if feature_type == "pac":
        return _classify_pac_subtype(column_lower)

    return "unknown"


def _create_pattern_metadata(
    pattern_type: str, groups: Tuple[str, ...], column: str
) -> Tuple[str, str, Dict[str, Any]]:
    """Create feature metadata from matched pattern groups."""
    base_meta = {
        "identifier": column,
        "band": "N/A",
        "source": "inferred",
        "subtype": "unknown",
    }

    pattern_handlers = {
        "erds": lambda: (
            "power",
            "erds",
            {**base_meta, "band": groups[0], "identifier": groups[1], "channel": groups[1]},
        ),
        "erds_windowed": lambda: (
            "power",
            "erds_windowed",
            {
                **base_meta,
                "band": groups[0],
                "channel": groups[1],
                "window": groups[2],
                "identifier": f"{groups[1]}_{groups[2]}",
            },
        ),
        "relative_power": lambda: (
            "power",
            "relative",
            {**base_meta, "band": groups[0], "channel": groups[1], "identifier": groups[1]},
        ),
        "band_ratio": lambda: (
            "power",
            "ratio",
            {
                **base_meta,
                "band": f"{groups[0]}/{groups[1]}",
                "channel": groups[2],
                "identifier": groups[2],
            },
        ),
        "temporal_stat": lambda: (
            "temporal",
            groups[0],
            {**base_meta, "stat": groups[0], "channel": groups[1], "identifier": groups[1]},
        ),
        "amplitude": lambda: (
            "temporal",
            groups[0],
            {**base_meta, "stat": groups[0], "channel": groups[1], "identifier": groups[1]},
        ),
        "hjorth": lambda: (
            "complexity",
            "hjorth",
            {**base_meta, "param": groups[0], "channel": groups[1], "identifier": groups[1]},
        ),
        "roi_power": lambda: (
            "power",
            "roi",
            {**base_meta, "band": groups[0], "roi": groups[1], "identifier": groups[1]},
        ),
        "roi_asymmetry": lambda: (
            "roi",
            "asymmetry",
            {**base_meta, "band": groups[0], "pair": groups[1], "identifier": groups[1]},
        ),
        "roi_laterality": lambda: (
            "roi",
            "asymmetry",
            {**base_meta, "band": groups[0], "pair": groups[1], "identifier": groups[1]},
        ),
        "itpc": lambda: (
            "itpc",
            "itpc",
            {
                **base_meta,
                "band": groups[0],
                "channel": groups[1],
                "time_bin": groups[2],
                "identifier": groups[1],
            },
        ),
        "aperiodic": lambda: (
            "aperiodic",
            groups[0],
            {
                **base_meta,
                "param": groups[0],
                "channel": groups[1],
                "identifier": groups[1],
                "band": "aperiodic",
            },
        ),
        "powcorr": lambda: (
            "power",
            "correlation",
            {**base_meta, "band": groups[0], "channel": groups[1], "identifier": groups[1]},
        ),
        "conn_graph": lambda: (
            "connectivity",
            "graph",
            {
                **base_meta,
                "measure": groups[0],
                "band": groups[1],
                "metric": groups[2],
                "identifier": f"{groups[0]}_{groups[2]}",
            },
        ),
        "gfp": lambda: (
            "gfp",
            groups[0],
            {**base_meta, "metric": groups[0], "identifier": groups[0], "band": "global"},
        ),
        "power_segmented": lambda: (
            "power",
            groups[0],
            {
                **base_meta,
                "band": groups[1],
                "identifier": groups[2],
                "segment": groups[0],
            },
        ),
        "connectivity_segmented": lambda: (
            "connectivity",
            groups[1],
            {
                **base_meta,
                "band": groups[2],
                "identifier": groups[3],
                "measure": groups[0],
                "segment": groups[1],
            },
        ),
        "itpc_segmented": lambda: (
            "itpc",
            groups[2],
            {
                **base_meta,
                "band": groups[0],
                "channel": groups[1],
                "segment": groups[2],
                "identifier": groups[1],
            },
        ),
        "dynamics_burst_segmented": lambda: (
            "dynamics",
            groups[1],
            {
                **base_meta,
                "band": groups[0],
                "segment": groups[1],
                "metric": groups[2],
                "identifier": f"{groups[0]}_{groups[1]}_{groups[2]}",
            },
        ),
    }

    handler = pattern_handlers.get(pattern_type)
    if handler:
        return handler()

    identifier = groups[0] if groups else column
    return pattern_type, "unknown", {**base_meta, "identifier": identifier}


def _match_feature_patterns(
    column: str, registry: FeatureRegistry
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Match column name against feature patterns and extract metadata."""
    for pattern_type, pattern in registry.patterns.items():
        match = pattern.match(column)
        if match:
            groups = match.groups()
            return _create_pattern_metadata(pattern_type, groups, column)
    return None


_FREQUENCY_BANDS = {
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


def _extract_band_and_identifier(parts: List[str], freq_bands: Set[str]) -> Tuple[str, str]:
    """Extract frequency band and identifier from feature name parts."""
    if len(parts) < 2:
        return "N/A", "_".join(parts) if parts else "all"

    second_part_lower = parts[1].lower()
    if second_part_lower in freq_bands:
        band = second_part_lower
        identifier = "_".join(parts[2:]) if len(parts) > 2 else "all"
    else:
        band = "N/A"
        identifier = "_".join(parts[1:])
    return band, identifier


def _extract_band_from_any_position(parts: List[str], freq_bands: Set[str]) -> Tuple[str, str]:
    """Extract frequency band from any position in parts and build identifier."""
    for i, part in enumerate(parts):
        if part.lower() in freq_bands:
            band = part.lower()
            remaining = parts[:i] + parts[i + 1 :]
            identifier = "_".join(remaining) if remaining else "all"
            return band, identifier
    return "N/A", "_".join(parts) if parts else "all"


def _parse_feature_metadata(column: str, feature_type: str) -> Dict[str, Any]:
    """Parse feature metadata from column name based on feature type."""
    if not column:
        return {"identifier": "unknown", "band": "N/A"}

    parts = column.split("_")
    meta = {"identifier": column, "band": "N/A"}

    band_candidates = [part.lower() for part in parts if part.lower() in _FREQUENCY_BANDS]
    if band_candidates:
        meta["band"] = band_candidates[0]

    band_extractors = {
        "power", "connectivity", "graph", "itpc", "pac", "spectral", "roi"
    }
    if feature_type in band_extractors:
        band, identifier = _extract_band_and_identifier(parts, _FREQUENCY_BANDS)
        meta["band"] = band
        meta["identifier"] = identifier
    elif feature_type == "aperiodic":
        meta["band"] = "aperiodic"
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column
    elif feature_type in ("temporal", "complexity"):
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column
    elif feature_type == "precomputed":
        band, identifier = _extract_band_from_any_position(parts, _FREQUENCY_BANDS)
        meta["band"] = band
        meta["identifier"] = identifier
    elif feature_type == "gfp":
        meta["band"] = "global"
        meta["identifier"] = "_".join(parts[1:]) if len(parts) > 1 else column

    return meta


_SCHEMA_GROUP_TO_FEATURE_TYPE = {
    "conn": "connectivity",
    "asymmetry": "roi",
    "comp": "complexity",
    "qual": "quality",
    "ratio": "ratios",
    "aper": "aperiodic",
    "asym": "asymmetry",
}


def _create_unknown_feature_metadata(
    column: Any, source_file_type: Optional[str]
) -> Dict[str, Any]:
    """Create metadata for unknown/invalid feature columns."""
    return {
        "identifier": str(column) if column else "unknown",
        "band": "N/A",
        "source": source_file_type or "unknown",
        "subtype": "unknown",
    }


def _classify_from_naming_schema(
    column: str, parsed: Dict[str, Any], source_file_type: Optional[str]
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify feature using NamingSchema parsing results."""
    schema_group = parsed["group"]
    feature_type = _SCHEMA_GROUP_TO_FEATURE_TYPE.get(schema_group, schema_group)
    subtype = parsed.get("segment", "unknown")

    if feature_type == "roi" and schema_group == "asymmetry":
        subtype = "asymmetry"

    identifier = parsed.get("identifier") or parsed.get("stat") or column
    meta: Dict[str, Any] = {
        "identifier": identifier,
        "band": parsed.get("band", "N/A"),
        "stat": parsed.get("stat"),
        "scope": parsed.get("scope"),
        "segment": parsed.get("segment"),
        "source": source_file_type or "inferred",
        "subtype": subtype,
    }

    return feature_type, subtype, meta


def _classify_from_patterns(
    column: str, registry: FeatureRegistry, source_file_type: Optional[str]
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Classify feature using pattern matching."""
    pattern_match = _match_feature_patterns(column, registry)
    if pattern_match:
        feature_type, subtype, meta = pattern_match
        meta["source"] = source_file_type or meta.get("source", "inferred")
        meta["subtype"] = subtype
        return feature_type, subtype, meta
    return None


def _classify_from_rules(
    column: str, registry: FeatureRegistry, source_file_type: Optional[str]
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify feature using rule-based classifiers."""
    feature_type = "unknown"
    for rule in registry.classifiers:
        if _rule_matches(column, rule):
            feature_type = rule.label
            break

    feature_type = registry.source_to_type.get(feature_type, feature_type) or "unknown"
    subtype = _classify_subtype(column, feature_type, registry, source_file_type)

    meta = {
        "identifier": column,
        "band": "N/A",
        "source": source_file_type or "inferred",
        "subtype": "unknown",
    }
    meta.update(_parse_feature_metadata(column, feature_type))
    meta["subtype"] = subtype

    return feature_type, subtype, meta


def classify_feature(
    column: str,
    source_file_type: Optional[str] = None,
    include_subtype: bool = True,
    registry: Optional[FeatureRegistry] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Classify feature and extract metadata using config-driven registry."""
    if not column or not isinstance(column, str):
        meta = _create_unknown_feature_metadata(column, source_file_type)
        subtype = "unknown" if include_subtype else ""
        return "unknown", subtype, meta

    registry = registry or get_feature_registry()

    parsed = NamingSchema.parse(column)
    if parsed.get("valid"):
        feature_type, subtype, meta = _classify_from_naming_schema(column, parsed, source_file_type)
        return feature_type, subtype if include_subtype else "", meta

    pattern_result = _classify_from_patterns(column, registry, source_file_type)
    if pattern_result:
        feature_type, subtype, meta = pattern_result
        return feature_type, subtype if include_subtype else "", meta

    feature_type, subtype, meta = _classify_from_rules(column, registry, source_file_type)
    return feature_type, subtype if include_subtype else "", meta


__all__ = [
    "FeatureRule",
    "FeatureRegistry",
    "classify_feature",
    "get_feature_registry",
]
