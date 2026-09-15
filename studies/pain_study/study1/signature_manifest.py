from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from studies.pain_study.study1.targets import (
    _sha256,
    _signature_support_summary,
    _validate_spatial_provenance,
)


def build_signature_manifest(
    *,
    signature_root: Path,
    signature_specs: Sequence[dict[str, Any]],
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
            raise FileNotFoundError(
                f"Study 1 signature map does not exist for {name}: {image_path}"
            )

        checksum = _sha256(image_path)
        _validate_spatial_provenance(name=name, entry=spec, checksum=checksum)
        image = nib.load(str(image_path))
        affine = np.asarray(image.affine, dtype=float)
        signatures[name] = {
            "path": relative_path,
            "space": str(spec["space"]).strip().lower(),
            "source_space": str(spec["source_space"]).strip().lower(),
            "spatial_reference": str(spec["spatial_reference"]).strip(),
            "source_publication": str(spec["source_publication"]).strip(),
            "source_repository_or_access_record": str(
                spec["source_repository_or_access_record"]
            ).strip(),
            "sha256": checksum,
            "shape": [int(value) for value in image.shape],
            "affine": affine.tolist(),
            "support": _signature_support_summary(image_path),
        }
        if "transform" in spec:
            signatures[name]["transform"] = dict(spec["transform"])
    return {"signatures": signatures}


def write_signature_manifest(
    *,
    signature_root: Path,
    output_path: Path,
    provenance_path: Path,
) -> Path:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("Study 1 signature manifest generation requires PyYAML.") from exc

    with open(provenance_path, encoding="utf-8") as handle:
        provenance = yaml.safe_load(handle)
    if not isinstance(provenance, dict) or not isinstance(provenance.get("signatures"), list):
        raise ValueError("Signature provenance YAML must define a signatures list.")
    manifest = build_signature_manifest(
        signature_root=signature_root,
        signature_specs=provenance["signatures"],
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
    parser.add_argument("--provenance", type=Path, required=True, dest="provenance_path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    write_signature_manifest(
        signature_root=args.signature_root,
        output_path=args.output_path,
        provenance_path=args.provenance_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
