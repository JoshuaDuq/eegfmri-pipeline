"""``eeg-pipeline data-chain`` -- say which generation of the source chain is live.

``source_data/EEG_SOURCE_LAYOUT.md`` documents the chain, but a document cannot notice when
the data stops matching it. On 2026-08-02 three ``step2_*`` directories existed where the
layout named one, and the one it named was not the one that fed step 3.

``manifest`` fingerprints every stage on disk and writes the table; ``verify`` asserts that
the stage the delivered BIDS runs are supposed to come from actually reproduces their
R-marker counts, and exits non-zero when it does not.

The fingerprint is the R-marker count per recording, read from ``.vmrk`` text. It separates
otherwise identical 13 GB directories without opening a binary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

MODES = ("manifest", "verify")

#: Stage directories, in chain order. Anything named ``_superseded*`` is reported but never
#: treated as live.
CHAIN = (
    ("step1", "source_data/step1_scanner_artifact_pulse_marked"),
    ("step2", "source_data/step2_pulse_markers_recovered_v2"),
    ("step3", "source_data/step3_bcg_corrected"),
)
REFERENCE = ("reference_bcg_pre_recovery", "source_data/reference_bcg_pre_recovery")

#: The stage the delivered BIDS runs are built from.
EXPECTED_SOURCE = "step3"


def setup_data_chain(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the data-chain parser."""
    parser = subparsers.add_parser(
        "data-chain",
        help="Fingerprint the source chain and check the delivered data came from it",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=list(MODES), help="Stage to run")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root holding source_data/ and bids_output/ (default: from paths.bids_root)",
    )
    parser.add_argument(
        "--bids-root", type=str, default=None, help="Delivered BIDS EEG root to check against"
    )
    parser.add_argument("--output", type=str, default=None, help="manifest: where the table goes")
    return parser


def _resolve_roots(args: Any, config: Any) -> tuple[Path, Path]:
    bids_root = Path(args.bids_root or str(config.get("paths.bids_root"))).expanduser()
    if args.data_root:
        data_root = Path(args.data_root).expanduser()
    else:
        # bids_output/<name> -> the directory holding both source_data and bids_output
        data_root = bids_root.parent.parent
    return data_root, bids_root


def run_data_chain(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Fingerprint the chain, or check the delivered data against it."""
    from studies.pain_study.analysis import data_chain as dc

    data_root, bids_root = _resolve_roots(args, config)

    stages: dict[str, dict] = {}
    present: dict[str, str] = {}
    for name, relative in (*CHAIN, REFERENCE):
        path = data_root / relative
        if path.is_dir():
            stages[name] = dc.stage_marker_counts(path)
            present[name] = str(path)

    superseded = sorted(
        str(p) for p in (data_root / "source_data").glob("_superseded*") if p.is_dir()
    )
    partial = sorted(
        str(p)
        for p in (data_root / "source_data").glob("step2_pulse_markers_recovered_v*")
        if p.is_dir() and p.name != Path(CHAIN[1][1]).name
    )

    delivered = dc.delivered_marker_counts(bids_root)
    print(f"delivered BIDS runs: {len(delivered)} ({bids_root})")
    for name, path in present.items():
        counts = stages[name]
        score = dc.agreement(counts, delivered)
        print(
            f"  {name:26} {len(counts):4d} recordings  "
            f"matches delivered {score.matched:3d}/{score.shared:3d}  {path}"
        )
    for path in superseded:
        print(f"  {'(superseded)':26} {'':4}              {path}")
    for path in partial:
        print(f"  {'(partial/other)':26} {'':4}              {path}")

    if args.mode == "manifest":
        destination = Path(args.output or (data_root / "source_data" / "DATA_CHAIN_MANIFEST.json"))
        payload = {
            "delivered_bids_root": str(bids_root),
            "delivered_recordings": len(delivered),
            "expected_source": EXPECTED_SOURCE,
            "stages": {
                name: {
                    "path": present[name],
                    "recordings": len(stages[name]),
                    "matches_delivered": dc.agreement(stages[name], delivered).matched,
                    "shared_with_delivered": dc.agreement(stages[name], delivered).shared,
                    "marker_counts": {f"{s}/{r}": n for (s, r), n in sorted(stages[name].items())},
                }
                for name in present
            },
            "superseded": superseded,
            "partial_or_other": partial,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {destination}")
        return

    problems = dc.verify_chain(stages, delivered, expected_source=EXPECTED_SOURCE)
    if problems:
        print()
        for problem in problems:
            print(f"MISMATCH  {problem}")
        likely = dc.best_match(stages, delivered)
        if likely and likely != EXPECTED_SOURCE:
            print(
                f"\nThe delivered data matches '{likely}' more closely than "
                f"'{EXPECTED_SOURCE}'. Either the chain moved or the layout is stale."
            )
        raise SystemExit(1)
    print(f"\nOK: every delivered run's marker count is reproduced by '{EXPECTED_SOURCE}'.")


__all__ = ["CHAIN", "EXPECTED_SOURCE", "MODES", "run_data_chain", "setup_data_chain"]
