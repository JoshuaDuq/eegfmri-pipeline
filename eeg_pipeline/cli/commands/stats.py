"""Stats/analytics CLI command."""

from __future__ import annotations

import argparse
import json as json_module
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import add_task_arg, resolve_task


def setup_stats(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the stats command parser."""
    parser = subparsers.add_parser(
        "stats",
        help="Project statistics: subjects, features, storage",
        description="Show project-wide statistics and analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["summary", "subjects", "features", "storage", "timeline"],
        nargs="?",
        default="summary",
        help="What to show (default: summary)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown",
    )
    return parser


def run_stats(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the stats command."""
    from datetime import datetime
    from eeg_pipeline.utils.data.subjects import (
        _collect_subjects_from_bids,
        _collect_subjects_from_derivatives_epochs,
        _collect_subjects_from_features,
    )
    from eeg_pipeline.infra.paths import resolve_deriv_root, deriv_features_path
    
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)
    
    try:
        bids_subjects = set(_collect_subjects_from_bids())
    except Exception:
        bids_subjects = set()
    
    epochs_subjects = set(_collect_subjects_from_derivatives_epochs(deriv_root, task, config))
    features_subjects = set(_collect_subjects_from_features(deriv_root))
    
    all_subjects = bids_subjects | epochs_subjects | features_subjects
    
    if args.mode == "summary" or args.mode == "subjects":
        n_bids = len(bids_subjects)
        n_epochs = len(epochs_subjects)
        n_features = len(features_subjects)
        n_total = len(all_subjects)
        
        feature_categories = ["power", "connectivity", "aperiodic", "dynamics", "complexity", "itpc", "microstates"]
        category_counts = {cat: 0 for cat in feature_categories}
        
        for subj in features_subjects:
            features_dir = deriv_features_path(deriv_root, subj)
            if features_dir.exists():
                for cat in feature_categories:
                    tsv_files = list(features_dir.glob(f"features_{cat}*.tsv"))
                    if tsv_files:
                        category_counts[cat] += 1
        
        if n_total > 0:
            pct_epochs = (n_epochs / n_total) * 100
            pct_features = (n_features / n_total) * 100
        else:
            pct_epochs = pct_features = 0
        
        if args.mode == "summary":
            stats = {
                "total_subjects": n_total,
                "bids_subjects": n_bids,
                "epochs_subjects": n_epochs,
                "features_subjects": n_features,
                "epochs_pct": round(pct_epochs, 1),
                "features_pct": round(pct_features, 1),
                "feature_categories": category_counts,
                "task": task,
                "deriv_root": str(deriv_root),
            }
            
            if args.output_json:
                print(json_module.dumps(stats, indent=2))
            else:
                print("=" * 50)
                print("           EEG PIPELINE PROJECT STATS")
                print("=" * 50)
                print()
                print(f"  Task:           {task}")
                print(f"  Derivatives:    {deriv_root}")
                print()
                print("  SUBJECTS")
                print("  " + "-" * 30)
                print(f"  Total Discovered:   {n_total:>5}")
                print(f"  With BIDS:          {n_bids:>5}")
                print(f"  With Epochs:        {n_epochs:>5} ({pct_epochs:.0f}%)")
                print(f"  With Features:      {n_features:>5} ({pct_features:.0f}%)")
                print()
                print("  FEATURE CATEGORIES")
                print("  " + "-" * 30)
                for cat, count in category_counts.items():
                    bar_len = int((count / n_total) * 20) if n_total > 0 else 0
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    print(f"  {cat:15} {bar} {count}/{n_total}")
                print()
        else:
            if args.output_json:
                print(json_module.dumps({
                    "bids_only": list(bids_subjects - epochs_subjects),
                    "epochs_only": list(epochs_subjects - features_subjects),
                    "complete": list(features_subjects),
                    "counts": {
                        "bids": n_bids,
                        "epochs": n_epochs,
                        "features": n_features,
                    }
                }, indent=2))
            else:
                print("SUBJECTS BY PIPELINE STAGE")
                print("=" * 40)
                
                complete = features_subjects
                epochs_only = epochs_subjects - features_subjects
                bids_only = bids_subjects - epochs_subjects
                
                if complete:
                    print(f"\n✓ Complete ({len(complete)})")
                    for s in sorted(complete):
                        print(f"    sub-{s}")
                
                if epochs_only:
                    print(f"\n◐ Epochs Only ({len(epochs_only)})")
                    for s in sorted(epochs_only):
                        print(f"    sub-{s}")
                
                if bids_only:
                    print(f"\n○ BIDS Only ({len(bids_only)})")
                    for s in sorted(bids_only):
                        print(f"    sub-{s}")
    
    elif args.mode == "features":
        import pandas as pd
        
        feature_stats = []
        for subj in sorted(features_subjects):
            features_dir = deriv_features_path(deriv_root, subj)
            if not features_dir.exists():
                continue
            
            subj_stats = {"subject": subj, "files": 0, "total_features": 0, "categories": []}
            
            for tsv in features_dir.glob("features_*.tsv"):
                try:
                    df = pd.read_csv(tsv, sep="\t", nrows=1)
                    n_cols = len([c for c in df.columns if c not in ["subject", "epoch", "condition", "task"]])
                    subj_stats["files"] += 1
                    subj_stats["total_features"] += n_cols
                    
                    cat = tsv.stem.replace("features_", "").split("_")[0]
                    if cat not in subj_stats["categories"]:
                        subj_stats["categories"].append(cat)
                except Exception:
                    pass
            
            feature_stats.append(subj_stats)
        
        if args.output_json:
            print(json_module.dumps({"feature_stats": feature_stats}, indent=2))
        else:
            print("FEATURE STATISTICS BY SUBJECT")
            print("=" * 50)
            print(f"{'Subject':15} {'Files':>8} {'Features':>10} {'Categories'}")
            print("-" * 50)
            for s in feature_stats:
                cats = ", ".join(s["categories"][:3])
                if len(s["categories"]) > 3:
                    cats += f"... +{len(s['categories']) - 3}"
                print(f"sub-{s['subject']:11} {s['files']:>8} {s['total_features']:>10} {cats}")
            
            total_files = sum(s["files"] for s in feature_stats)
            total_features = sum(s["total_features"] for s in feature_stats)
            print("-" * 50)
            print(f"{'TOTAL':15} {total_files:>8} {total_features:>10}")
    
    elif args.mode == "storage":
        def get_dir_size(path: Path) -> int:
            total = 0
            if path.exists():
                for f in path.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
            return total
        
        def format_size(size_bytes: int) -> str:
            for unit in ["B", "KB", "MB", "GB"]:
                if size_bytes < 1024:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024
            return f"{size_bytes:.1f} TB"
        
        storage_stats = {}
        
        features_total = 0
        for subj in features_subjects:
            features_dir = deriv_features_path(deriv_root, subj)
            features_total += get_dir_size(features_dir)
        storage_stats["features"] = features_total
        
        epochs_dir = deriv_root / "epochs"
        storage_stats["epochs"] = get_dir_size(epochs_dir) if epochs_dir.exists() else 0
        
        plots_dir = deriv_root / "plots"
        storage_stats["plots"] = get_dir_size(plots_dir) if plots_dir.exists() else 0
        
        behavior_dir = deriv_root / "behavior"
        storage_stats["behavior"] = get_dir_size(behavior_dir) if behavior_dir.exists() else 0
        
        total_storage = sum(storage_stats.values())
        
        if args.output_json:
            print(json_module.dumps({
                "storage_bytes": storage_stats,
                "storage_formatted": {k: format_size(v) for k, v in storage_stats.items()},
                "total_bytes": total_storage,
                "total_formatted": format_size(total_storage),
            }, indent=2))
        else:
            print("STORAGE USAGE")
            print("=" * 40)
            for category, size in sorted(storage_stats.items(), key=lambda x: -x[1]):
                pct = (size / total_storage * 100) if total_storage > 0 else 0
                bar_len = int(pct / 5)
                bar = "█" * bar_len
                print(f"  {category:12} {format_size(size):>10}  {bar}")
            print("-" * 40)
            print(f"  {'TOTAL':12} {format_size(total_storage):>10}")
    
    elif args.mode == "timeline":
        feature_times = []
        for subj in features_subjects:
            features_dir = deriv_features_path(deriv_root, subj)
            if not features_dir.exists():
                continue
            
            tsvs = list(features_dir.glob("*.tsv"))
            if tsvs:
                latest_mtime = max(f.stat().st_mtime for f in tsvs)
                feature_times.append({
                    "subject": subj,
                    "last_modified": datetime.fromtimestamp(latest_mtime).isoformat(),
                    "n_files": len(tsvs),
                })
        
        feature_times.sort(key=lambda x: x["last_modified"], reverse=True)
        
        if args.output_json:
            print(json_module.dumps({"timeline": feature_times}, indent=2))
        else:
            print("PROCESSING TIMELINE (most recent first)")
            print("=" * 50)
            for entry in feature_times[:20]:
                print(f"  sub-{entry['subject']:11}  {entry['last_modified'][:19]}  ({entry['n_files']} files)")
            if len(feature_times) > 20:
                print(f"  ... and {len(feature_times) - 20} more")
