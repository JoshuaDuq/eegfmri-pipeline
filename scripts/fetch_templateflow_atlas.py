#!/usr/bin/env python3
"""
Fetch an MNI atlas from TemplateFlow and copy it into this repo's external atlas directory.

Example:
  ./.venv/bin/python scripts/fetch_templateflow_atlas.py --schaefer 100 7 --resolution 2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_out_dir() -> Path:
    return _repo_root() / "eeg_pipeline" / "data" / "external" / "atlases"


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch an atlas from TemplateFlow and copy it into eeg_pipeline/data/external/atlases.")
    p.add_argument("--out-dir", type=str, default=str(_default_out_dir()), help="Output directory for atlas files")
    p.add_argument("--template", type=str, default="MNI152NLin2009cAsym", help="TemplateFlow template (default: MNI152NLin2009cAsym)")
    p.add_argument("--resolution", type=int, default=2, help="Atlas resolution (e.g., 1 or 2 for Schaefer)")

    # For now we implement the exact atlas you discussed; extend later as needed.
    p.add_argument(
        "--schaefer",
        nargs=2,
        metavar=("PARCELS", "NETWORKS"),
        help="Fetch Schaefer2018 atlas (e.g., --schaefer 100 7)",
    )

    args = p.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tpl = str(args.template).strip()

    if args.schaefer is None:
        raise SystemExit("No atlas specified. Example: --schaefer 100 7")

    parcels = int(args.schaefer[0])
    networks = int(args.schaefer[1])
    desc = f"{parcels}Parcels{networks}Networks"

    try:
        import templateflow.api as tflow
    except Exception as exc:
        raise SystemExit(
            "templateflow is not installed in this environment. "
            "Install it into your venv (pip install templateflow) and re-run."
        ) from exc

    atlas_nii = tflow.get(
        tpl,
        atlas="Schaefer2018",
        desc=desc,
        resolution=int(args.resolution),
        suffix="dseg",
        extension="nii.gz",
    )

    labels_tsv = tflow.get(
        tpl,
        atlas="Schaefer2018",
        desc=desc,
        suffix="dseg",
        extension="tsv",
    )

    atlas_src = Path(str(atlas_nii))
    labels_src = Path(str(labels_tsv))

    res_tag = f"res-0{int(args.resolution)}"
    atlas_dst = out_dir / f"tpl-{tpl}_atlas-Schaefer2018_desc-{desc}_{res_tag}_dseg.nii.gz"
    labels_dst = out_dir / f"tpl-{tpl}_atlas-Schaefer2018_desc-{desc}_dseg.tsv"

    _copy(atlas_src, atlas_dst)
    _copy(labels_src, labels_dst)

    print("Wrote:")
    print(f"  atlas : {atlas_dst}")
    print(f"  labels: {labels_dst}")


if __name__ == "__main__":
    main()
