"""Selection helpers for plotting command."""

from __future__ import annotations

import argparse
from typing import Dict, List, Set


def resolve_plot_ids(
    args: argparse.Namespace,
    *,
    plot_ids: Set[str],
    plot_groups: Dict[str, List[str]],
) -> List[str]:
    selected: Set[str] = set()
    if args.plots:
        selected.update(args.plots)
    if args.groups:
        for group in args.groups:
            selected.update(plot_groups.get(group, []))
    if args.all_plots or not selected:
        selected.update(plot_ids)
    return sorted(selected)


def unique_in_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result
