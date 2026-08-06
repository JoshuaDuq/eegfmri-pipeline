"""Anatomical names for cluster peaks.

A cluster table of bare X/Y/Z coordinates makes the reader do the lookup, and a
reader who does not bother reads a result they cannot locate. Naming the peak is what
turns a row of the table into a finding.

The atlas is supplied by the study, as a label volume plus an optional index-to-name
table -- the same convention ``fmri_resting_state`` already uses for its parcellation.
Deliberately not ``nilearn.datasets.fetch_atlas_*``: that downloads on first use, which
would put a network round-trip inside a report path that otherwise runs offline from a
derivatives tree, and would silently pin the report to whatever version the fetcher
resolves.

Labelling is gated on space. An MNI atlas read at a native-space coordinate returns the
name of whatever structure happens to sit at those millimetres in a different brain,
and nothing in the output distinguishes that from a correct answer -- so an unlabelled
table says why it is unlabelled instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: What ``manifest.space`` has to say before an MNI atlas may be read.
_MNI = "mni"


def atlas_applies_to(space: str) -> bool:
    """Whether a standard-space atlas may be read at this contrast's coordinates."""
    return str(space or "").strip().lower() == _MNI


def space_refusal(space: str) -> str:
    """Say why a contrast got no anatomical labels."""
    text = str(space or "").strip() or "native"
    return (
        f"no anatomical labels: the atlas is defined in MNI and these coordinates are "
        f"in {text} space"
    )


@dataclass(frozen=True)
class AtlasLabeller:
    """Names the structure a world coordinate falls in.

    ``names`` maps a label volume's integer values to their names. When no table is
    supplied the integer itself is reported, which is still more than a coordinate:
    two peaks carrying the same index are in the same parcel.
    """

    label_img: Any
    names: Dict[int, str]
    source: str

    def label_at(self, coord: Sequence[float]) -> Optional[str]:
        """Return the structure at a world coordinate, or ``None`` outside the atlas.

        ``None`` for a coordinate outside the atlas or falling in its background,
        which is a real answer: a peak in a ventricle or outside the parcellated
        volume has no anatomical name, and inventing the nearest one would be a
        guess the table could not be checked against.
        """
        try:
            inverse = np.linalg.inv(np.asarray(self.label_img.affine))
            voxel = np.rint(
                (np.append(np.asarray(coord, dtype=float), 1.0) @ inverse.T)[:3]
            ).astype(int)
        except Exception as exc:  # pragma: no cover - depends on a singular affine
            logger.debug("Could not map %s into the atlas (%s)", coord, exc)
            return None

        shape = np.asarray(self.label_img.shape[:3])
        if np.any(voxel < 0) or np.any(voxel >= shape):
            return None

        value = int(np.asanyarray(self.label_img.dataobj)[tuple(voxel)])
        if value == 0:
            return None
        return self.names.get(value, str(value))

    def label_all(
        self, coords: Sequence[Sequence[float]]
    ) -> Tuple[Optional[str], ...]:
        """Label a sequence of peaks, preserving order and gaps."""
        return tuple(self.label_at(coord) for coord in coords)


def _read_names(path: Path) -> Dict[int, str]:
    """Read an index-to-name table.

    Accepts the BIDS ``dseg.tsv`` convention -- an ``index`` column and a ``name`` or
    ``label`` column -- and falls back to the first two columns, which is what most
    hand-rolled lookup tables are.
    """
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    index_column = lowered.get("index") or lowered.get("id") or frame.columns[0]
    name_column = (
        lowered.get("name")
        or lowered.get("label")
        or lowered.get("region")
        or (frame.columns[1] if len(frame.columns) > 1 else frame.columns[0])
    )
    names: Dict[int, str] = {}
    for raw_index, raw_name in zip(frame[index_column], frame[name_column]):
        try:
            names[int(raw_index)] = str(raw_name)
        except (TypeError, ValueError):
            continue
    return names


def load_atlas(
    *, labels_img: Optional[Path | str], labels_tsv: Optional[Path | str] = None
) -> Optional[AtlasLabeller]:
    """Load a configured atlas, or ``None`` when none is configured or readable.

    Best-effort throughout. An atlas that cannot be read costs the report a column,
    not the cluster table -- and the caption says the column is absent rather than
    leaving a reader to assume the peaks had no anatomy.
    """
    if not labels_img:
        return None

    path = Path(labels_img)
    if not path.exists():
        logger.warning("Configured atlas %s does not exist; peaks stay unlabelled.", path)
        return None

    try:
        import nibabel as nib

        img = nib.load(str(path))
    except Exception as exc:
        logger.warning("Could not read the atlas %s (%s)", path, exc)
        return None

    names: Dict[int, str] = {}
    if labels_tsv:
        tsv_path = Path(labels_tsv)
        if tsv_path.exists():
            try:
                names = _read_names(tsv_path)
            except Exception as exc:
                logger.warning(
                    "Could not read the atlas label table %s (%s); reporting raw "
                    "indices instead.",
                    tsv_path,
                    exc,
                )
        else:
            logger.warning(
                "Configured atlas label table %s does not exist; reporting raw indices.",
                tsv_path,
            )

    return AtlasLabeller(label_img=img, names=names, source=path.name)


__all__ = [
    "AtlasLabeller",
    "atlas_applies_to",
    "load_atlas",
    "space_refusal",
]
