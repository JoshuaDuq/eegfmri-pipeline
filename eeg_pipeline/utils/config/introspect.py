#!/usr/bin/env python3
"""Audit backend config key reachability from CLI/TUI.

Coverage model:
- Direct coverage: key is explicitly hydrated into the TUI model.
- Fallback coverage: key is reachable through generic `--set key=value` wiring
  from both CLI and TUI, with precedence applied after command-specific
  overrides.

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

KEY_RE_CONFIG_OVERRIDE_RULE = re.compile(
    r'ConfigOverrideRule\(\s*"[^"]+"\s*,\s*"([A-Za-z_][A-Za-z0-9_\.]*)"'
)
KEY_RE_CONFIG_SET = re.compile(r'config\.set\(\s*"([A-Za-z_][A-Za-z0-9_\.]*)"')
KEY_RE_APPLY_CONFIG_OVERRIDE = re.compile(
    r'_apply_config_override\(\s*config\s*,\s*"([A-Za-z_][A-Za-z0-9_\.]*)"'
)
KEY_RE_TUI_HYDRATION = re.compile(r'\{key:\s*"([A-Za-z_][A-Za-z0-9_\.]*)"')

KEY_RE_DIRECT_CONFIG_GET = re.compile(
    r'\b(?:self\.)?(?:[A-Za-z_][A-Za-z0-9_]*\.)*config\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_\.]*)["\']'
)
KEY_RE_DIRECT_GETTER = re.compile(
    r'\b(?:get_config_value|require_config_value|get_config_bool|_get_config_value)\(\s*[^,\n]+,\s*["\']([A-Za-z_][A-Za-z0-9_\.]*)["\']'
)
KEY_RE_ASSIGN_FROM_CONFIG_GET = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:self\.)?(?:[A-Za-z_][A-Za-z0-9_]*\.)*config\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_\.]*)["\']'
)
KEY_RE_ASSIGN_FROM_GETTER = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:get_config_value|_get_config_value)\(\s*[^,\n]+,\s*["\']([A-Za-z_][A-Za-z0-9_\.]*)["\']'
)
KEY_RE_ASSIGN_FROM_VAR_GET = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_\.]*)["\']'
)
KEY_RE_VAR_GET_ACCESS = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]*)\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_\.]*)["\']'
)

KEY_ALIAS_REPLACEMENTS: Sequence[Tuple[str, str]] = (
    (".condition_a.", ".cond_a."),
    (".condition_b.", ".cond_b."),
)

KNOWN_CONFIG_ROOTS: Set[str] = {
    "alignment",
    "analysis",
    "behavior_analysis",
    "channel_rois",
    "eeg",
    "epochs",
    "event_columns",
    "feature_categories",
    "feature_engineering",
    "fmri_analysis",
    "fmri_contrast",
    "fmri_preprocessing",
    "fmri_group_level",
    "frequency_bands",
    "ica",
    "icalabel",
    "machine_learning",
    "paths",
    "plotting",
    "preprocessing",
    "project",
    "pyprep",
    "rois",
    "statistics",
    "time_frequency_analysis",
    "validation",
    "visualization",
}

BACKEND_OVERRIDE_SOURCES: Sequence[str] = (
    "eeg_pipeline/cli/commands/preprocessing_overrides.py",
    "eeg_pipeline/cli/commands/features_helpers.py",
    "eeg_pipeline/cli/commands/behavior_config.py",
    "eeg_pipeline/cli/commands/machine_learning_overrides.py",
    "eeg_pipeline/cli/commands/plotting_config_overrides.py",
)

BACKEND_RUNTIME_SOURCES: Sequence[str] = (
    "eeg_pipeline/pipelines",
    "eeg_pipeline/analysis",
    "eeg_pipeline/context",
    "eeg_pipeline/plotting",
    "eeg_pipeline/utils/analysis",
    "eeg_pipeline/utils/data",
    "eeg_pipeline/utils/validation.py",
    "eeg_pipeline/infra",
    "fmri_pipeline/pipelines",
    "fmri_pipeline/analysis",
    "fmri_pipeline/utils",
)

TUI_HYDRATION_SOURCE = "eeg_pipeline/cli/tui/views/wizard/model_config_hydration.go"

CLI_SET_PARSER_SOURCES: Sequence[str] = (
    "eeg_pipeline/cli/commands/features_parser.py",
    "eeg_pipeline/cli/commands/behavior_parser.py",
    "eeg_pipeline/cli/commands/preprocessing_parser.py",
    "eeg_pipeline/cli/commands/machine_learning_parser.py",
    "eeg_pipeline/cli/commands/plotting_parser.py",
    "fmri_pipeline/cli/commands/fmri.py",
    "fmri_pipeline/cli/commands/fmri_analysis.py",
)

TUI_SET_OPTION_SOURCES: Sequence[str] = (
    "eeg_pipeline/cli/tui/views/wizard/model_options_features.go",
    "eeg_pipeline/cli/tui/views/wizard/model_options_behavior.go",
    "eeg_pipeline/cli/tui/views/wizard/model_options_ml.go",
    "eeg_pipeline/cli/tui/views/wizard/model_options_stage_preprocessing.go",
    "eeg_pipeline/cli/tui/views/wizard/model_options_stage_fmri.go",
    "eeg_pipeline/cli/tui/views/wizard/model_options_plotting.go",
)

SET_PRECEDENCE_RULES: Sequence[Tuple[str, str, str]] = (
    (
        "eeg_pipeline/cli/commands/features_orchestrator.py",
        "_apply_feature_config_overrides(args, config)",
        "apply_set_overrides(config, getattr(args, \"set_overrides\", None))",
    ),
    (
        "eeg_pipeline/cli/commands/behavior_orchestrator.py",
        "_configure_behavior_compute_mode(args, config)",
        "apply_set_overrides(config, getattr(args, \"set_overrides\", None))",
    ),
    (
        "eeg_pipeline/cli/commands/preprocessing_orchestrator.py",
        "_update_alignment_event_config(args, config)",
        "apply_set_overrides(config, getattr(args, \"set_overrides\", None))",
    ),
    (
        "eeg_pipeline/cli/commands/machine_learning_orchestrator.py",
        "config[\"feature_engineering.analysis_mode\"] = \"trial_ml_safe\"",
        "apply_set_overrides(config, getattr(args, \"set_overrides\", None))",
    ),
    (
        "eeg_pipeline/cli/commands/plotting_orchestrator.py",
        "apply_all_config_overrides(args, config)",
        "apply_set_overrides(config, getattr(args, \"set_overrides\", None))",
    ),
    (
        "fmri_pipeline/cli/commands/fmri.py",
        "_update_fmri_config_from_args(args, config)",
        "apply_set_overrides(config, getattr(args, \"set_overrides\", None))",
    ),
)


@dataclass(frozen=True)
class SetSupportStatus:
    enabled: bool
    cli_parser_support: bool
    tui_support: bool
    precedence_ok: bool
    failed_checks: List[str]


@dataclass(frozen=True)
class CoverageData:
    backend_override_key_count: int
    backend_runtime_key_count: int
    backend_union_key_count: int
    tui_hydration_key_count: int
    set_support: SetSupportStatus
    backend_keys_covered_directly: List[str]
    backend_keys_covered_via_set: List[str]
    backend_keys_missing_in_tui: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "counts": {
                "backend_override_key_count": self.backend_override_key_count,
                "backend_runtime_key_count": self.backend_runtime_key_count,
                "backend_union_key_count": self.backend_union_key_count,
                "tui_hydration_key_count": self.tui_hydration_key_count,
                "backend_keys_covered_directly": len(self.backend_keys_covered_directly),
                "backend_keys_covered_via_set": len(self.backend_keys_covered_via_set),
                "backend_keys_missing_in_tui": len(self.backend_keys_missing_in_tui),
            },
            "set_support": {
                "enabled": self.set_support.enabled,
                "cli_parser_support": self.set_support.cli_parser_support,
                "tui_support": self.set_support.tui_support,
                "precedence_ok": self.set_support.precedence_ok,
                "failed_checks": self.set_support.failed_checks,
            },
            "coverage": {
                "backend_keys_covered_directly": self.backend_keys_covered_directly,
                "backend_keys_covered_via_set": self.backend_keys_covered_via_set,
                "backend_keys_missing_in_tui": self.backend_keys_missing_in_tui,
            },
            "sources": {
                "backend_override_sources": list(BACKEND_OVERRIDE_SOURCES),
                "backend_runtime_sources": list(BACKEND_RUNTIME_SOURCES),
                "tui_hydration_source": TUI_HYDRATION_SOURCE,
                "cli_set_parser_sources": list(CLI_SET_PARSER_SOURCES),
                "tui_set_option_sources": list(TUI_SET_OPTION_SOURCES),
            },
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_paths(repo_root: Path, rel_paths: Sequence[str]) -> List[Path]:
    return [repo_root / rel for rel in rel_paths]


def _canonical_key(key: str) -> str:
    out = key.strip()
    for src, dst in KEY_ALIAS_REPLACEMENTS:
        out = out.replace(src, dst)
    return out


def _looks_like_config_key(key: str) -> bool:
    if not key:
        return False
    if "." in key:
        return True
    return key in KNOWN_CONFIG_ROOTS


def _join_prefix(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if "." in suffix:
        return suffix
    return f"{prefix}.{suffix}"


def _extract_backend_override_keys(paths: Iterable[Path]) -> Set[str]:
    keys: Set[str] = set()
    for path in paths:
        text = _read_text(path)
        for pattern in (
            KEY_RE_CONFIG_OVERRIDE_RULE,
            KEY_RE_CONFIG_SET,
            KEY_RE_APPLY_CONFIG_OVERRIDE,
        ):
            for key in pattern.findall(text):
                canonical = _canonical_key(key)
                if _looks_like_config_key(canonical):
                    keys.add(canonical)
    return keys


def _iter_runtime_python_files(repo_root: Path) -> Iterable[Path]:
    for rel in BACKEND_RUNTIME_SOURCES:
        candidate = repo_root / rel
        if candidate.is_file() and candidate.suffix == ".py":
            yield candidate
            continue
        if not candidate.exists() or not candidate.is_dir():
            continue
        for path in candidate.rglob("*.py"):
            rel_path = path.relative_to(repo_root).as_posix()
            if "/cli/" in rel_path or "/tests/" in rel_path:
                continue
            if "/preprocessing/scripts/" in rel_path:
                continue
            yield path


def _extract_runtime_keys_from_text(text: str) -> Set[str]:
    keys: Set[str] = set()
    aliases: Dict[str, str] = {}

    for key in KEY_RE_DIRECT_CONFIG_GET.findall(text):
        canonical = _canonical_key(key)
        if _looks_like_config_key(canonical):
            keys.add(canonical)
    for key in KEY_RE_DIRECT_GETTER.findall(text):
        canonical = _canonical_key(key)
        if _looks_like_config_key(canonical):
            keys.add(canonical)

    for line in text.splitlines():
        match = KEY_RE_ASSIGN_FROM_CONFIG_GET.search(line)
        if match:
            var, base = match.group(1), _canonical_key(match.group(2))
            if _looks_like_config_key(base):
                aliases[var] = base

        match = KEY_RE_ASSIGN_FROM_GETTER.search(line)
        if match:
            var, base = match.group(1), _canonical_key(match.group(2))
            if _looks_like_config_key(base):
                aliases[var] = base

        match = KEY_RE_ASSIGN_FROM_VAR_GET.search(line)
        if match:
            var, parent, leaf = match.group(1), match.group(2), _canonical_key(match.group(3))
            parent_prefix = aliases.get(parent)
            if parent_prefix:
                combined = _join_prefix(parent_prefix, leaf)
                if _looks_like_config_key(combined):
                    aliases[var] = combined

        for var, leaf in KEY_RE_VAR_GET_ACCESS.findall(line):
            parent_prefix = aliases.get(var)
            if not parent_prefix:
                continue
            combined = _canonical_key(_join_prefix(parent_prefix, leaf))
            if _looks_like_config_key(combined):
                keys.add(combined)

    return keys


def _extract_backend_runtime_keys(repo_root: Path) -> Set[str]:
    keys: Set[str] = set()
    for path in _iter_runtime_python_files(repo_root):
        text = _read_text(path)
        keys.update(_extract_runtime_keys_from_text(text))
    return keys


def _extract_tui_hydration_keys(path: Path) -> Set[str]:
    text = _read_text(path)
    return {_canonical_key(key) for key in KEY_RE_TUI_HYDRATION.findall(text)}


def _check_cli_set_parser_support(repo_root: Path) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    for rel in CLI_SET_PARSER_SOURCES:
        text = _read_text(repo_root / rel)
        if "add_output_format_args(" not in text and "--set" not in text:
            failures.append(f"CLI parser missing --set support: {rel}")
    return (len(failures) == 0, failures)


def _check_tui_set_support(repo_root: Path) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    commands = _read_text(repo_root / "eeg_pipeline/cli/tui/views/wizard/commands.go")
    if "parseConfigSetOverrides" not in commands:
        failures.append("TUI commands missing parseConfigSetOverrides in commands.go")
    if "--set" not in commands:
        failures.append("TUI commands missing --set emission in commands.go")

    option_types = _read_text(repo_root / "eeg_pipeline/cli/tui/views/wizard/model_option_types.go")
    if "optConfigSetOverrides" not in option_types:
        failures.append("TUI option types missing optConfigSetOverrides")

    for rel in TUI_SET_OPTION_SOURCES:
        text = _read_text(repo_root / rel)
        if "optConfigSetOverrides" not in text:
            failures.append(f"TUI stage missing Config Overrides option: {rel}")

    return (len(failures) == 0, failures)


def _check_set_precedence(repo_root: Path) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    for rel, before_token, after_token in SET_PRECEDENCE_RULES:
        text = _read_text(repo_root / rel)
        before_idx = text.find(before_token)
        if before_idx < 0:
            failures.append(f"Precedence check failed (missing before token) in {rel}: {before_token}")
            continue

        after_indices: List[int] = []
        start = 0
        while True:
            idx = text.find(after_token, start)
            if idx < 0:
                break
            after_indices.append(idx)
            start = idx + len(after_token)

        if not after_indices:
            failures.append(f"Precedence check failed (missing apply_set_overrides) in {rel}")
            continue

        if not any(idx > before_idx for idx in after_indices):
            failures.append(
                f"Precedence check failed in {rel}: apply_set_overrides must appear after '{before_token}'"
            )

    return (len(failures) == 0, failures)


def _detect_set_support(repo_root: Path) -> SetSupportStatus:
    cli_ok, cli_failures = _check_cli_set_parser_support(repo_root)
    tui_ok, tui_failures = _check_tui_set_support(repo_root)
    precedence_ok, precedence_failures = _check_set_precedence(repo_root)
    failures = cli_failures + tui_failures + precedence_failures
    enabled = cli_ok and tui_ok and precedence_ok
    return SetSupportStatus(
        enabled=enabled,
        cli_parser_support=cli_ok,
        tui_support=tui_ok,
        precedence_ok=precedence_ok,
        failed_checks=failures,
    )


def compute_coverage(repo_root: Path) -> CoverageData:
    backend_override_keys = _extract_backend_override_keys(
        _resolve_paths(repo_root, BACKEND_OVERRIDE_SOURCES)
    )
    backend_runtime_keys = _extract_backend_runtime_keys(repo_root)
    backend_keys = backend_override_keys | backend_runtime_keys

    tui_hydration_keys = _extract_tui_hydration_keys(repo_root / TUI_HYDRATION_SOURCE)
    set_support = _detect_set_support(repo_root)

    covered_directly = sorted(backend_keys & tui_hydration_keys)
    if set_support.enabled:
        covered_via_set = sorted(backend_keys - tui_hydration_keys)
        missing = []
    else:
        covered_via_set = []
        missing = sorted(backend_keys - tui_hydration_keys)

    return CoverageData(
        backend_override_key_count=len(backend_override_keys),
        backend_runtime_key_count=len(backend_runtime_keys),
        backend_union_key_count=len(backend_keys),
        tui_hydration_key_count=len(tui_hydration_keys),
        set_support=set_support,
        backend_keys_covered_directly=covered_directly,
        backend_keys_covered_via_set=covered_via_set,
        backend_keys_missing_in_tui=missing,
    )


def render_markdown(data: CoverageData) -> str:
    counts = data.as_dict()["counts"]
    lines = [
        "# TUI Config Coverage Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Generic `--set` fallback fully supported: `{data.set_support.enabled}`",
        f"- CLI parser `--set` support: `{data.set_support.cli_parser_support}`",
        f"- TUI `Config Overrides` support: `{data.set_support.tui_support}`",
        f"- `--set` precedence checks: `{data.set_support.precedence_ok}`",
        f"- Backend override keys audited: `{counts['backend_override_key_count']}`",
        f"- Backend runtime keys audited: `{counts['backend_runtime_key_count']}`",
        f"- Backend union keys audited: `{counts['backend_union_key_count']}`",
        f"- Directly hydrated in TUI: `{counts['backend_keys_covered_directly']}`",
        f"- Covered via `--set` fallback: `{counts['backend_keys_covered_via_set']}`",
        f"- Missing from TUI coverage: `{counts['backend_keys_missing_in_tui']}`",
        "",
    ]

    def _render_list_section(title: str, values: Sequence[str]) -> None:
        lines.append(f"### {title}")
        lines.append("")
        if not values:
            lines.append("- None")
        else:
            for item in values:
                lines.append(f"- `{item}`")
        lines.append("")

    _render_list_section("Set Support Check Failures", data.set_support.failed_checks)
    _render_list_section(
        "Backend Keys Missing In TUI Coverage",
        data.backend_keys_missing_in_tui,
    )
    _render_list_section(
        "Backend Keys Covered Via Generic --set",
        data.backend_keys_covered_via_set,
    )

    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current working directory)",
    )
    parser.add_argument("--json-out", default="", help="Optional path to write JSON output.")
    parser.add_argument(
        "--markdown-out",
        default="",
        help="Optional path to write Markdown report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when backend keys are missing from TUI coverage "
            "or generic --set support/precedence checks fail."
        ),
    )
    return parser.parse_args(argv)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    repo_root = Path(args.repo_root).resolve()
    coverage = compute_coverage(repo_root)
    payload = coverage.as_dict()

    if args.json_out:
        _write_json(Path(args.json_out), payload)
    if args.markdown_out:
        _write_text(Path(args.markdown_out), render_markdown(coverage))

    if args.strict and (
        coverage.backend_keys_missing_in_tui
        or not coverage.set_support.enabled
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
