"""What each run contributes to a first-level contrast.

Two measurements the report could not otherwise make, both keyed by run and therefore
both belonging to one table rather than two panels.

The whole-mask mean of a run's own estimate answers whether the contrast carries a
brain-wide offset. A contrast that differences two conditions has no reason to: a
non-zero value is signal shared across the whole mask rather than anatomy. It matters
because the fitted null's centre *is* this quantity, and a map centred away from zero
produces large clusters of the offset's sign and shifts survival toward whichever
tissue class the offset reaches most -- both of which arrive in the cluster table
looking like a result.

Leave-one-run-out survivor counts answer a different question: which run the map rests
on. The forest panel already shows each run's estimate at the chosen peaks, and that is
not the same measurement -- a run can carry the largest peak estimates while a different
run moves the map more. On this study's data exactly that happens.

Nothing here is scored. A run differing from the others is a measurement; a task with
habituation should produce one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def run_offsets(
    run_effect_img: Any, mask_img: Any = None
) -> Optional[List[float]]:
    """Whole-mask mean of each run's own estimate of the contrast.

    Restricted to the analysis mask when one is supplied. Over the whole volume the
    mean is dominated by background zeros and reads near zero however large the real
    offset is.

    ``None`` when the maps cannot be read, which is a property of the derivatives
    rather than a fault.
    """
    try:
        data = np.asanyarray(run_effect_img.dataobj).astype(float)
    except Exception as exc:
        logger.warning("Could not read the run-level effect maps (%s)", exc)
        return None
    if data.ndim != 4:
        logger.warning("Run-level effects must be 4D, got shape %s.", data.shape)
        return None

    if mask_img is not None:
        try:
            mask = np.asanyarray(mask_img.dataobj).astype(bool)
        except Exception as exc:
            logger.warning("Could not read the analysis mask (%s)", exc)
            mask = None
        else:
            if mask.shape != data.shape[:3]:
                logger.warning(
                    "Mask shape %s does not match the run-level maps' %s.",
                    mask.shape,
                    data.shape[:3],
                )
                mask = None
    else:
        mask = None

    offsets: List[float] = []
    for index in range(data.shape[3]):
        volume = data[..., index]
        selected = volume[mask] if mask is not None else volume[volume != 0]
        finite = selected[np.isfinite(selected)]
        offsets.append(float(finite.mean()) if finite.size else float("nan"))
    return offsets


def read_run_influence(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load the leave-one-run-out table the analysis wrote, keyed by dropped run.

    An empty mapping when no table was written -- a single-run contrast has nothing
    to drop, and a derivatives tree built before the measurement existed carries none.
    """
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        import pandas as pd

        frame = pd.read_csv(path, sep="\t")
    except Exception as exc:
        logger.warning("Could not read %s (%s)", path.name, exc)
        return {}
    return {str(row["dropped_run"]): dict(row) for _index, row in frame.iterrows()}


def contribution_rows(
    *,
    run_labels: Sequence[str],
    offsets: Optional[Sequence[float]] = None,
    influence: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """One row per run, carrying whichever measurements are available.

    Keyed by run label rather than by position, so a table missing a run leaves that
    cell empty instead of shifting every later run's numbers up by one.
    """
    influence = influence or {}
    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(run_labels):
        row: Dict[str, Any] = {"Run": str(label)}
        if offsets is not None and index < len(offsets):
            row["Mean effect over mask"] = offsets[index]
        found = influence.get(str(label))
        if found is not None:
            row["Voxels surviving without this run"] = found.get("survivors")
            row["Change"] = found.get("delta")
            row["r with the combined map"] = found.get("correlation")
        rows.append(row)
    return rows


__all__ = ["contribution_rows", "read_run_influence", "run_offsets"]
