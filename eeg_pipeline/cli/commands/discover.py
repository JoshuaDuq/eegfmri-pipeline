"""Column discovery CLI command for TUI integration."""

from __future__ import annotations

import argparse
import json as json_module
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import add_task_arg, resolve_task


def setup_discover(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the discover command parser."""
    parser = subparsers.add_parser(
        "discover",
        help="Discover available columns and values from data files",
        description="Scan events files or trial tables to find available columns and their values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        choices=["events", "trial-table", "all"],
        nargs="?",
        default="all",
        help="Where to discover columns from (default: all)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--subject",
        default=None,
        help="Specific subject to scan (default: auto-detect from first available)",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Get values for a specific column only",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format (for TUI integration)",
    )
    return parser


def run_discover(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the discover command."""
    from eeg_pipeline.cli.commands.base import (
        discover_event_columns,
        discover_trial_table_columns,
    )
    from eeg_pipeline.infra.paths import resolve_deriv_root
    
    task = resolve_task(args.task, config)
    bids_root = Path(config.bids_root) if hasattr(config, "bids_root") else None
    deriv_root = resolve_deriv_root(config=config)
    
    result = {
        "columns": [],
        "values": {},
        "source": None,
        "sources_checked": [],
    }
    
    subject = args.subject
    if not subject and subjects:
        subject = subjects[0]
    
    if args.source in ["events", "all"] and bids_root:
        events_data = discover_event_columns(bids_root, task=task, subject=subject)
        if events_data["columns"]:
            result["sources_checked"].append("events")
            if not result["columns"]:
                result["columns"] = events_data["columns"]
                result["values"] = events_data["values"]
                result["source"] = events_data["source"]
                result["file"] = events_data.get("file")
            else:
                for col, vals in events_data["values"].items():
                    if col not in result["values"]:
                        result["values"][col] = vals
    
    if args.source in ["trial-table", "all"]:
        trial_data = discover_trial_table_columns(deriv_root, subject=subject)
        if trial_data["columns"]:
            result["sources_checked"].append("trial_table")
            if not result["columns"] or args.source == "trial-table":
                result["columns"] = trial_data["columns"]
                result["values"] = trial_data["values"]
                result["source"] = trial_data["source"]
                result["file"] = trial_data.get("file")
            else:
                for col, vals in trial_data["values"].items():
                    if col not in result["values"]:
                        result["values"][col] = vals
    
    if args.column:
        if args.column in result["values"]:
            result["values"] = {args.column: result["values"][args.column]}
        else:
            result["values"] = {}
    
    if args.output_json:
        print(json_module.dumps(result, indent=2))
    else:
        print("=" * 50)
        print("       COLUMN DISCOVERY REPORT")
        print("=" * 50)
        print()
        
        if not result["columns"]:
            print("  No columns discovered.")
            print("  Make sure you have events files in your BIDS directory")
            print("  or have run behavior compute to create trial tables.")
            return
        
        print(f"  Source: {result.get('source', 'unknown')}")
        if result.get("file"):
            print(f"  File: {result['file']}")
        print()
        
        print("  AVAILABLE COLUMNS")
        print("  " + "-" * 30)
        for col in result["columns"]:
            has_values = col in result["values"]
            val_indicator = f" ({len(result['values'][col])} values)" if has_values else ""
            print(f"    • {col}{val_indicator}")
        print()
        
        if result["values"]:
            print("  COLUMN VALUES")
            print("  " + "-" * 30)
            for col, vals in sorted(result["values"].items()):
                if len(vals) <= 10:
                    vals_str = ", ".join(str(v) for v in vals)
                else:
                    vals_str = ", ".join(str(v) for v in vals[:8]) + f", ... (+{len(vals) - 8} more)"
                print(f"    {col}: {vals_str}")
        print()
