"""Plot catalog loading for plotting CLI command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PlotDefinition:
    plot_id: str
    group: str
    label: str
    description: str
    required_files: List[str]
    feature_categories: Optional[List[str]] = None
    feature_plot_patterns: Optional[List[str]] = None
    behavior_plots: Optional[List[str]] = None
    tfr_plots: Optional[List[str]] = None
    erp_plots: Optional[List[str]] = None
    requires_epochs: bool = False
    requires_features: bool = False
    requires_stats: bool = False


def _load_plot_catalog() -> List[PlotDefinition]:
    catalog_path = Path(__file__).resolve().parents[2] / "plotting" / "plot_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    plots: List[PlotDefinition] = []
    for entry in payload.get("plots", []):
        plots.append(
            PlotDefinition(
                plot_id=str(entry["id"]),
                group=str(entry["group"]),
                label=str(entry.get("label", "")),
                description=str(entry.get("description", "")),
                required_files=list(entry.get("required_files", [])),
                feature_categories=entry.get("feature_categories"),
                feature_plot_patterns=entry.get("feature_plot_patterns"),
                behavior_plots=entry.get("behavior_plots"),
                tfr_plots=entry.get("tfr_plots"),
                erp_plots=entry.get("erp_plots"),
                requires_epochs=bool(entry.get("requires_epochs", False)),
                requires_features=bool(entry.get("requires_features", False)),
                requires_stats=bool(entry.get("requires_stats", False)),
            )
        )
    return plots


PLOT_CATALOG: List[PlotDefinition] = _load_plot_catalog()
PLOT_BY_ID: Dict[str, PlotDefinition] = {plot.plot_id: plot for plot in PLOT_CATALOG}
PLOT_GROUPS: Dict[str, List[str]] = {}
for plot in PLOT_CATALOG:
    PLOT_GROUPS.setdefault(plot.group, []).append(plot.plot_id)
