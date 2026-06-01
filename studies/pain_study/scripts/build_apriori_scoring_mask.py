"""Materialize the fixed a-priori signature scoring mask.

The scoring mask defines, independently of the analyzed sample, which voxels enter the
NPS/SIIPS1 dot product. Using a standard MNI152NLin2009cAsym brain mask keeps the target
definition stable as subjects are added and prevents a single truncated field of view from
shrinking the scored extent for the whole cohort.

The mask is fetched from TemplateFlow, written into the signature directory, and verified to
retain at least ``--min-coverage`` of each signature's nonzero weight support.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from nilearn.image import resample_to_img


def _nonzero_fraction_within_mask(signature_path: Path, mask_img: nib.Nifti1Image) -> float:
    signature_img = nib.load(str(signature_path))
    weights = np.asanyarray(signature_img.dataobj, dtype=float)
    nonzero = np.isfinite(weights) & (np.abs(weights) > 0.0)
    total = int(nonzero.sum())
    if total == 0:
        raise ValueError(f"Signature map has no nonzero weights: {signature_path}")
    mask_on_grid = resample_to_img(
        mask_img, signature_img, interpolation="nearest", force_resample=True, copy_header=True
    )
    covered = nonzero & (np.asanyarray(mask_on_grid.dataobj, dtype=float) > 0)
    return float(covered.sum() / total)


def _signature_paths(signature_dir: Path) -> dict[str, Path]:
    manifest_path = signature_dir / "signature_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Signature manifest not found: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    signatures = manifest["signatures"]
    return {name: signature_dir / entry["path"] for name, entry in signatures.items()}


def build_mask(*, signature_dir: Path, output_name: str, min_coverage: float) -> Path:
    import templateflow.api as templateflow

    source_mask = Path(
        templateflow.get(
            "MNI152NLin2009cAsym", resolution=2, desc="brain", suffix="mask", extension=".nii.gz"
        )
    )
    output_path = signature_dir / output_name
    shutil.copyfile(source_mask, output_path)

    mask_img = nib.load(str(output_path))
    print(f"Scoring mask written to {output_path} (grid {mask_img.shape})")
    for name, signature_path in _signature_paths(signature_dir).items():
        coverage = _nonzero_fraction_within_mask(signature_path, mask_img)
        print(f"  {name}: {coverage:.4f} of nonzero support retained")
        if coverage < min_coverage:
            raise ValueError(
                f"{name} support coverage {coverage:.4f} is below the required {min_coverage:.2f}."
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signature-dir",
        required=True,
        type=Path,
        help="Directory holding the signature maps and signature_manifest.yaml.",
    )
    parser.add_argument(
        "--output-name",
        default="tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
        help="Filename for the scoring mask, written under the signature directory.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.90)
    args = parser.parse_args()
    build_mask(
        signature_dir=args.signature_dir.expanduser(),
        output_name=args.output_name,
        min_coverage=args.min_coverage,
    )


if __name__ == "__main__":
    main()
