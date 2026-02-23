#!/usr/bin/env python3
"""Smoke-check CLI pipeline entrypoints for the TUI.

This script is intentionally lightweight and safe to run without data.
It validates that key command parsers are wired and callable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SmokeCase:
    name: str
    argv: List[str]


def _cases(task: str) -> List[SmokeCase]:
    # Help checks cover parser wiring for each pipeline family.
    # A single runtime check validates command dispatch end-to-end.
    return [
        SmokeCase("preprocessing", ["-m", "eeg_pipeline.cli.main", "preprocessing", "full", "--help"]),
        SmokeCase("features", ["-m", "eeg_pipeline.cli.main", "features", "compute", "--help"]),
        SmokeCase("behavior", ["-m", "eeg_pipeline.cli.main", "behavior", "compute", "--help"]),
        SmokeCase("machine_learning", ["-m", "eeg_pipeline.cli.main", "ml", "regression", "--help"]),
        SmokeCase("plotting", ["-m", "eeg_pipeline.cli.main", "plotting", "visualize", "--help"]),
        SmokeCase("fmri_preprocessing", ["-m", "eeg_pipeline.cli.main", "fmri", "preprocess", "--help"]),
        SmokeCase("fmri_analysis", ["-m", "eeg_pipeline.cli.main", "fmri-analysis", "first-level", "--help"]),
        SmokeCase("validate", ["-m", "eeg_pipeline.cli.main", "validate", "quick", "--help"]),
        SmokeCase("info", ["-m", "eeg_pipeline.cli.main", "info", "subjects", "--help"]),
        SmokeCase("stats", ["-m", "eeg_pipeline.cli.main", "stats", "--help"]),
        SmokeCase("runtime_version", ["-m", "eeg_pipeline.cli.main", "info", "version", "--json", "--task", task]),
    ]


def _parse_pipeline_tokens(raw: str) -> List[str]:
    tokens: List[str] = []
    for chunk in str(raw or "").replace(";", ",").split(","):
        value = chunk.strip()
        if value:
            tokens.append(value)
    deduped: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _emit_json(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=True), flush=True)


def _emit_log(progress_json: bool, message: str, *, subject: str = "", level: str = "info") -> None:
    if progress_json:
        _emit_json(
            {
                "event": "log",
                "level": level,
                "subject": subject,
                "message": message,
            }
        )
        return
    print(message, flush=True)


def _emit_subject_done(progress_json: bool, subject: str, success: bool) -> None:
    if progress_json:
        _emit_json({"event": "subject_done", "subject": subject, "success": bool(success)})


def _preview_output(text: str, max_lines: int = 6) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    preview = lines[:max_lines]
    suffix = "" if len(lines) <= max_lines else f" ... (+{len(lines) - max_lines} lines)"
    return " | ".join(preview) + suffix


def _candidate_python_paths(repo_root: Path) -> List[str]:
    candidates = [
        repo_root / "eeg_pipeline" / ".venv311" / "bin" / "python",
        repo_root / ".venv311" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python",
        repo_root / "venv" / "bin" / "python",
    ]
    out: List[str] = [str(path) for path in candidates if path.exists()]
    if sys.executable:
        out.append(sys.executable)
    seen = set()
    deduped: List[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _has_core_deps(py_cmd: str, repo_root: Path) -> bool:
    probe = subprocess.run(
        [py_cmd, "-c", "import yaml"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def _resolve_python(repo_root: Path) -> str:
    for candidate in _candidate_python_paths(repo_root):
        if _has_core_deps(candidate, repo_root):
            return candidate
    return sys.executable or "python3"


def _run_case(case: SmokeCase, *, python_cmd: str, repo_root: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python_cmd, *case.argv],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )


def run(progress_json: bool, task: str, timeout_s: float, pipelines: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python_cmd = _resolve_python(repo_root)
    all_cases = _cases(task)
    by_name = {case.name: case for case in all_cases}
    if pipelines:
        unknown = [name for name in pipelines if name not in by_name]
        if unknown:
            _emit_log(
                progress_json,
                "Unknown pipeline smoke IDs: " + ", ".join(unknown),
                level="error",
            )
            _emit_log(
                progress_json,
                "Valid IDs: " + ", ".join(case.name for case in all_cases),
                level="error",
            )
            return 2
        cases = [by_name[name] for name in pipelines]
    else:
        cases = all_cases
    names = [case.name for case in cases]

    if progress_json:
        _emit_json(
            {
                "event": "start",
                "operation": "tui_pipeline_smoke",
                "subjects": names,
                "total_subjects": len(cases),
            }
        )
    else:
        print(f"Running {len(cases)} pipeline smoke checks", flush=True)
    _emit_log(progress_json, f"Using Python interpreter: {python_cmd}")

    if not _has_core_deps(python_cmd, repo_root):
        _emit_log(
            progress_json,
            "Missing required Python dependency 'yaml' for CLI bootstrap. "
            "Install project dependencies (for example: pip install -e \".[dev]\").",
            level="error",
        )
        return 2

    failures: List[str] = []
    started = time.time()

    for case in cases:
        if progress_json:
            _emit_json({"event": "subject_start", "subject": case.name})
        _emit_log(progress_json, f"[run] {case.name}", subject=case.name)

        try:
            proc = _run_case(case, python_cmd=python_cmd, repo_root=repo_root, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            failures.append(case.name)
            _emit_log(progress_json, f"[fail] {case.name}: timed out after {timeout_s:.0f}s", subject=case.name, level="error")
            _emit_subject_done(progress_json, case.name, False)
            continue

        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        preview = _preview_output(output)
        if proc.returncode == 0:
            if preview:
                _emit_log(progress_json, f"[ok] {case.name}: {preview}", subject=case.name)
            else:
                _emit_log(progress_json, f"[ok] {case.name}", subject=case.name)
            _emit_subject_done(progress_json, case.name, True)
            continue

        failures.append(case.name)
        if preview:
            _emit_log(
                progress_json,
                f"[fail] {case.name}: exit={proc.returncode}; {preview}",
                subject=case.name,
                level="error",
            )
        else:
            _emit_log(
                progress_json,
                f"[fail] {case.name}: exit={proc.returncode}",
                subject=case.name,
                level="error",
            )
        _emit_subject_done(progress_json, case.name, False)

    elapsed = time.time() - started
    if failures:
        _emit_log(
            progress_json,
            f"Smoke checks finished with failures ({len(failures)}/{len(cases)}): {', '.join(failures)} in {elapsed:.1f}s",
            level="error",
        )
        return 1

    _emit_log(progress_json, f"Smoke checks passed ({len(cases)}/{len(cases)}) in {elapsed:.1f}s")
    return 0


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TUI pipeline smoke checks")
    parser.add_argument("--task", default="thermalactive", help="Task label for runtime smoke commands")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Per-command timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="",
        help=(
            "Comma-separated smoke IDs to run "
            "(example: preprocessing,features,plotting). "
            "Default runs all."
        ),
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit progress events as JSON lines for the TUI execution view",
    )
    args, _unknown = parser.parse_known_args(list(argv))
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    pipelines = _parse_pipeline_tokens(str(getattr(args, "pipelines", "") or ""))
    return run(
        progress_json=bool(args.progress_json),
        task=str(args.task),
        timeout_s=float(args.timeout_s),
        pipelines=pipelines or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
