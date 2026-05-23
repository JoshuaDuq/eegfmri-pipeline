from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from studies.pain_study.study1.targets import _sha256, _signature_support_summary


MNI152NLIN2009CASYM = "mni152nlin2009casym"

STUDY1_SIGNATURE_SPECS = (
    {
        "name": "NPS",
        "path": "NPS/weights_NSF_grouppred_cvpcr.nii.gz",
        "source_publication": "Wager et al. 2013, Neurologic Pain Signature",
        "source_repository_or_access_record": "Local KINGSTON external signature map",
    },
    {
        "name": "SIIPS1",
        "path": "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
        "source_publication": "Woo et al. 2017, SIIPS1",
        "source_repository_or_access_record": "Local KINGSTON external signature map",
    },
)


def build_signature_manifest(
    *,
    signature_root: Path,
    signature_specs: Sequence[dict[str, str]],
    space: str = MNI152NLIN2009CASYM,
) -> dict[str, Any]:
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:
        raise RuntimeError("Study 1 signature manifest generation requires nibabel.") from exc

    root = Path(signature_root).expanduser()
    signatures: dict[str, Any] = {}
    for spec in signature_specs:
        name = str(spec["name"]).strip()
        relative_path = str(spec["path"]).strip()
        image_path = root / relative_path
        if not image_path.exists():
            raise FileNotFoundError(f"Study 1 signature map does not exist for {name}: {image_path}")

        image = nib.load(str(image_path))
        affine = np.asarray(image.affine, dtype=float)
        signatures[name] = {
            "path": relative_path,
            "space": str(space).strip().lower(),
            "source_publication": str(spec["source_publication"]).strip(),
            "source_repository_or_access_record": str(
                spec["source_repository_or_access_record"]
            ).strip(),
            "sha256": _sha256(image_path),
            "shape": [int(value) for value in image.shape],
            "affine": affine.tolist(),
            "support": _signature_support_summary(image_path),
        }
    return {"signatures": signatures}


def write_signature_manifest(
    *,
    signature_root: Path,
    output_path: Path,
    space: str = MNI152NLIN2009CASYM,
) -> Path:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("Study 1 signature manifest generation requires PyYAML.") from exc

    manifest = build_signature_manifest(
        signature_root=signature_root,
        signature_specs=STUDY1_SIGNATURE_SPECS,
        space=space,
    )
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=True)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the Study 1 frozen signature manifest.")
    parser.add_argument("signature_root", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--space", default=MNI152NLIN2009CASYM)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    write_signature_manifest(
        signature_root=args.signature_root,
        output_path=args.output_path,
        space=args.space,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
