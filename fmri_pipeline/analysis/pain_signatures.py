from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PainSignatureResult:
    name: str
    weight_path: Path
    n_voxels: int
    dot: float
    cosine: Optional[float]
    pearson_r: Optional[float]


def discover_pain_signature_files(signature_root: Path) -> Dict[str, Path]:
    """
    Discover expected multivariate pain signature weight maps.

    Expected layout (as provided by this repo):
      <root>/NPS/weights_NSF_grouppred_cvpcr.nii.gz
      <root>/SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz
    """
    # Avoid .resolve() here: on macOS it can rewrite /var -> /private/var,
    # which is inconvenient for reproducible path comparisons/logging.
    root = Path(signature_root).expanduser()
    nps = root / "NPS" / "weights_NSF_grouppred_cvpcr.nii.gz"
    siips = root / "SIIPS1" / "nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
    out: Dict[str, Path] = {}
    if nps.exists():
        out["NPS"] = nps
    if siips.exists():
        out["SIIPS1"] = siips
    return out


def _maybe_import_nibabel():
    try:
        import nibabel as nib  # type: ignore

        return nib
    except Exception:
        return None


def _maybe_import_nilearn_image():
    try:
        from nilearn import image  # type: ignore

        return image
    except Exception:
        return None


def _maybe_resample_to_img(
    *,
    moving_img: Any,
    target_img: Any,
    interpolation: str,
) -> Any:
    """
    Best-effort resampling of a NIfTI image onto a target image grid.

    Uses nilearn when available; falls back to nibabel when possible.
    Raises ValueError on failure (to avoid silent scientific invalidity).
    """
    nilearn_image = _maybe_import_nilearn_image()
    if nilearn_image is not None:
        return nilearn_image.resample_to_img(
            moving_img,
            target_img,
            interpolation=interpolation,
            force_resample=True,
            copy_header=True,
        )

    try:
        from nibabel.processing import resample_from_to  # type: ignore

        order = 0 if interpolation == "nearest" else 1
        return resample_from_to(moving_img, (target_img.shape, target_img.affine), order=order)
    except Exception as exc:
        raise ValueError(
            "Could not resample image to target grid (missing nilearn and/or resampling backend)."
        ) from exc


def _flatten_masked_pairs(
    *,
    img_data: Any,
    w_data: Any,
    mask_data: Optional[Any] = None,
) -> Tuple[List[float], List[float]]:
    """
    Flatten (image, weights) into paired vectors with finite values.
    """
    x: List[float] = []
    w: List[float] = []
    it = zip(img_data.ravel(), w_data.ravel())
    if mask_data is None:
        for a, b in it:
            try:
                fa = float(a)
                fb = float(b)
            except Exception:
                continue
            if math.isfinite(fa) and math.isfinite(fb):
                x.append(fa)
                w.append(fb)
        return x, w

    for (a, b), m in zip(it, mask_data.ravel()):
        try:
            if not bool(m):
                continue
            fa = float(a)
            fb = float(b)
        except Exception:
            continue
        if math.isfinite(fa) and math.isfinite(fb):
            x.append(fa)
            w.append(fb)
    return x, w


def _dot(x: Sequence[float], w: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(x, w)))


def _norm(x: Sequence[float]) -> float:
    return math.sqrt(sum(a * a for a in x))


def _pearson_r(x: Sequence[float], w: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    mw = sum(w) / n
    num = sum((a - mx) * (b - mw) for a, b in zip(x, w))
    dx = sum((a - mx) ** 2 for a in x)
    dw = sum((b - mw) ** 2 for b in w)
    den = math.sqrt(dx * dw)
    if den == 0:
        return None
    return float(num / den)


def compute_pain_signature_expression(
    *,
    stat_or_effect_img: Any,
    signature_root: Path,
    mask_img: Optional[Any] = None,
    signatures: Optional[Sequence[str]] = None,
    resampling: str = "image_to_weights",
) -> List[PainSignatureResult]:
    """
    Compute multivariate pain signature expression (best-effort).

    Scientific notes:
    - Intended for MNI-space images when using the provided NPS/SIIPS1 weight maps.
    - Returns both dot-product (pattern expression) and Pearson correlation (scale-invariant).
    - Uses intersection of finite voxels and an optional analysis mask (resampled as needed).
    - Resampling strategy matters for comparability across signatures/resolutions:
        * resampling="image_to_weights" (default): resample the target image (and mask) to each signature's grid.
        * resampling="weights_to_image": resample each signature's weights (and mask) to the target image grid.
    """
    nib = _maybe_import_nibabel()
    if nib is None:
        return []

    files = discover_pain_signature_files(signature_root)
    if signatures:
        files = {k: v for k, v in files.items() if k in set(signatures)}
    if not files:
        return []

    resampling = str(resampling or "image_to_weights").strip().lower().replace("-", "_")
    if resampling not in {"image_to_weights", "weights_to_image"}:
        raise ValueError("resampling must be one of: image_to_weights, weights_to_image")

    # Load target image
    img = stat_or_effect_img
    if isinstance(img, (str, Path)):
        img = nib.load(str(img))

    results: List[PainSignatureResult] = []

    m = None
    if mask_img is not None:
        m = mask_img
        if isinstance(m, (str, Path)):
            m = nib.load(str(m))

    for name, w_path in files.items():
        try:
            w_img = nib.load(str(w_path))
            if resampling == "image_to_weights":
                # Preferred: resample subject/stat image to signature grid.
                x_img = img
                if tuple(getattr(x_img, "shape", ())) != tuple(getattr(w_img, "shape", ())):
                    x_img = _maybe_resample_to_img(moving_img=x_img, target_img=w_img, interpolation="continuous")
                else:
                    try:
                        import numpy as np

                        if not np.allclose(x_img.affine, w_img.affine):
                            x_img = _maybe_resample_to_img(moving_img=x_img, target_img=w_img, interpolation="continuous")
                    except Exception:
                        x_img = _maybe_resample_to_img(moving_img=x_img, target_img=w_img, interpolation="continuous")

                mask_data = None
                if m is not None:
                    mask_on_ref = m
                    if tuple(getattr(mask_on_ref, "shape", ())) != tuple(getattr(w_img, "shape", ())):
                        mask_on_ref = _maybe_resample_to_img(moving_img=mask_on_ref, target_img=w_img, interpolation="nearest")
                    else:
                        try:
                            import numpy as np

                            if not np.allclose(mask_on_ref.affine, w_img.affine):
                                mask_on_ref = _maybe_resample_to_img(moving_img=mask_on_ref, target_img=w_img, interpolation="nearest")
                        except Exception:
                            mask_on_ref = _maybe_resample_to_img(moving_img=mask_on_ref, target_img=w_img, interpolation="nearest")
                    try:
                        mask_data = (mask_on_ref.get_fdata() > 0).astype(bool)
                    except Exception:
                        mask_data = None

                img_data = x_img.get_fdata()
                w_data = w_img.get_fdata()
            else:
                # Backward-compatible: resample signature weights to the target image grid.
                w_on_ref = w_img
                if tuple(getattr(w_on_ref, "shape", ())) != tuple(getattr(img, "shape", ())):
                    w_on_ref = _maybe_resample_to_img(moving_img=w_on_ref, target_img=img, interpolation="continuous")
                else:
                    try:
                        import numpy as np

                        if not np.allclose(w_on_ref.affine, img.affine):
                            w_on_ref = _maybe_resample_to_img(moving_img=w_on_ref, target_img=img, interpolation="continuous")
                    except Exception:
                        w_on_ref = _maybe_resample_to_img(moving_img=w_on_ref, target_img=img, interpolation="continuous")

                mask_data = None
                if m is not None:
                    mask_on_ref = m
                    if tuple(getattr(mask_on_ref, "shape", ())) != tuple(getattr(img, "shape", ())):
                        mask_on_ref = _maybe_resample_to_img(moving_img=mask_on_ref, target_img=img, interpolation="nearest")
                    else:
                        try:
                            import numpy as np

                            if not np.allclose(mask_on_ref.affine, img.affine):
                                mask_on_ref = _maybe_resample_to_img(moving_img=mask_on_ref, target_img=img, interpolation="nearest")
                        except Exception:
                            mask_on_ref = _maybe_resample_to_img(moving_img=mask_on_ref, target_img=img, interpolation="nearest")
                    try:
                        mask_data = (mask_on_ref.get_fdata() > 0).astype(bool)
                    except Exception:
                        mask_data = None

                img_data = img.get_fdata()
                w_data = w_on_ref.get_fdata()

            if tuple(getattr(w_data, "shape", ())) != tuple(getattr(img_data, "shape", ())):
                # Can't align => skip (avoids scientifically invalid comparisons)
                continue

            if mask_data is not None and tuple(getattr(mask_data, "shape", ())) != tuple(getattr(img_data, "shape", ())):
                raise ValueError(
                    f"Mask grid mismatch for signature {name}: "
                    f"mask_shape={getattr(mask_data,'shape',None)} img_shape={getattr(img_data,'shape',None)}"
                )

            x_vec, w_vec = _flatten_masked_pairs(img_data=img_data, w_data=w_data, mask_data=mask_data)
            if not x_vec:
                continue

            dot = _dot(x_vec, w_vec)
            nx = _norm(x_vec)
            nw = _norm(w_vec)
            cosine = float(dot / (nx * nw)) if nx > 0 and nw > 0 else None
            r = _pearson_r(x_vec, w_vec)

            results.append(
                PainSignatureResult(
                    name=name,
                    weight_path=w_path,
                    n_voxels=len(x_vec),
                    dot=float(dot),
                    cosine=cosine,
                    pearson_r=r,
                )
            )
        except Exception as exc:
            # Avoid silently masking/mixing mismatched grids.
            if isinstance(exc, ValueError) and "Mask grid mismatch" in str(exc):
                raise
            continue

    return results
