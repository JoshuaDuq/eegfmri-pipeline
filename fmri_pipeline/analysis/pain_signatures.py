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
) -> List[PainSignatureResult]:
    """
    Compute multivariate pain signature expression (best-effort).

    Scientific notes:
    - Intended for MNI-space images when using the provided NPS/SIIPS1 weight maps.
    - Returns both dot-product (pattern expression) and Pearson correlation (scale-invariant).
    - Uses intersection of finite voxels and an optional analysis mask.
    """
    nib = _maybe_import_nibabel()
    if nib is None:
        return []

    files = discover_pain_signature_files(signature_root)
    if signatures:
        files = {k: v for k, v in files.items() if k in set(signatures)}
    if not files:
        return []

    # Load target image data
    img = stat_or_effect_img
    if isinstance(img, (str, Path)):
        img = nib.load(str(img))

    nilearn_image = _maybe_import_nilearn_image()
    results: List[PainSignatureResult] = []

    mask_data = None
    if mask_img is not None:
        m = mask_img
        if isinstance(m, (str, Path)):
            m = nib.load(str(m))
        try:
            mask_data = m.get_fdata().astype(bool)
        except Exception:
            mask_data = None

    img_data = img.get_fdata()

    for name, w_path in files.items():
        try:
            w_img = nib.load(str(w_path))
            # Resample weights to the target image grid when possible
            if nilearn_image is not None:
                try:
                    w_img = nilearn_image.resample_to_img(w_img, img, interpolation="continuous")
                except Exception:
                    pass
            w_data = w_img.get_fdata()

            if w_data.shape != img_data.shape:
                # Can't align => skip (avoids scientifically invalid comparisons)
                continue

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
        except Exception:
            continue

    return results
