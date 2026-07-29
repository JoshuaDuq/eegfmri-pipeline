from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SignatureResult:
    name: str
    weight_path: Path
    n_voxels: int
    dot: float
    cosine: Optional[float]
    pearson_r: Optional[float]
    nonzero_support_fraction: Optional[float] = None
    positive_support_fraction: Optional[float] = None
    negative_support_fraction: Optional[float] = None
    positive_weight_mass_change_fraction: Optional[float] = None
    negative_weight_mass_change_fraction: Optional[float] = None
    coverage_nonzero_support_fraction: Optional[float] = None
    coverage_positive_support_fraction: Optional[float] = None
    coverage_negative_support_fraction: Optional[float] = None
    coverage_positive_weight_mass_loss_fraction: Optional[float] = None
    coverage_negative_weight_mass_loss_fraction: Optional[float] = None
    scoring_mask_sha256: Optional[str] = None


def discover_signature_files(
    signature_root: Path,
    signature_specs: Sequence[Dict[str, str]],
) -> Dict[str, Path]:
    """
    Discover signature weight maps from a config-driven spec list.

    Each entry in ``signature_specs`` must have ``name`` and ``path`` keys,
    where ``path`` is relative to ``signature_root``.

    Raises when any configured signature is malformed or missing on disk.
    """
    root = Path(signature_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Signature root does not exist: {root}")
    out: Dict[str, Path] = {}
    for spec in signature_specs:
        name = str(spec.get("name", "")).strip()
        rel_path = str(spec.get("path", "")).strip()
        if not name or not rel_path:
            raise ValueError("Each signature spec must define non-empty 'name' and 'path'.")
        if name in out:
            raise ValueError(f"Duplicate signature name: {name!r}")
        candidate = root / rel_path
        if not candidate.exists():
            raise FileNotFoundError(f"Signature weight map not found for {name!r}: {candidate}")
        out[name] = candidate
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


def _validate_resampling_input(
    moving_img: Any,
    *,
    interpolation: str,
) -> Any:
    """
    Fail fast when resampling would mix unsupported voxels into valid data.

    Some upstream maps encode unsupported voxels as NaN. Replacing those values with zeros
    before continuous interpolation changes the boundary voxels instead of preserving the
    original support. Signature expression therefore rejects continuous resampling of images
    that contain non-finite voxels and requires callers to align grids up front.
    """
    import numpy as np  # type: ignore

    data = np.asanyarray(moving_img.dataobj, dtype=np.float32)
    if np.isfinite(data).all():
        return moving_img

    if interpolation == "continuous":
        raise ValueError(
            "Signature expression continuous resampling does not support non-finite voxels. "
            "Align the image and signature grids before computing pattern expression."
        )
    return moving_img


def _maybe_resample_to_img(
    *,
    moving_img: Any,
    target_img: Any,
    interpolation: str,
) -> Any:
    """
    Resample a NIfTI image onto a target image grid.

    Prefers nilearn when available; falls back to nibabel resampling.
    Raises ValueError on failure to prevent silent scientific invalidity.
    """
    moving_img = _validate_resampling_input(
        moving_img,
        interpolation=interpolation,
    )
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


def _image_grids_match(left_img: Any, right_img: Any) -> bool:
    if tuple(getattr(left_img, "shape", ())) != tuple(getattr(right_img, "shape", ())):
        return False
    import numpy as np  # type: ignore

    return bool(np.allclose(left_img.affine, right_img.affine))


def _mask_on_image_grid(*, mask_img: Any, image_img: Any) -> Any:
    if _image_grids_match(mask_img, image_img):
        return mask_img
    return _maybe_resample_to_img(
        moving_img=mask_img,
        target_img=image_img,
        interpolation="nearest",
    )


def _mask_data_on_grid(*, mask_img: Optional[Any], reference_img: Any, mask_name: str) -> Any:
    if mask_img is None:
        return None

    import numpy as np  # type: ignore

    mask_on_reference = _mask_on_image_grid(mask_img=mask_img, image_img=reference_img)
    mask_data = np.asanyarray(mask_on_reference.get_fdata(), dtype=float) > 0
    reference_shape = tuple(getattr(reference_img, "shape", ()))
    if mask_data.shape != reference_shape:
        raise ValueError(
            f"{mask_name} grid mismatch: mask_shape={mask_data.shape}, "
            f"reference_shape={reference_shape}."
        )
    return mask_data


def _fill_nonfinite_background_for_resampling(*, image_img: Any, mask_img: Optional[Any]) -> Any:
    import numpy as np  # type: ignore

    data = np.asanyarray(image_img.dataobj, dtype=np.float32)
    nonfinite = ~np.isfinite(data)
    if not bool(np.any(nonfinite)):
        return image_img
    if mask_img is None:
        return image_img

    mask_on_image = _mask_on_image_grid(mask_img=mask_img, image_img=image_img)
    mask = np.asanyarray(mask_on_image.get_fdata(), dtype=float) > 0
    if mask.shape != data.shape:
        raise ValueError(
            "Mask grid mismatch while preparing image for signature resampling: "
            f"mask_shape={mask.shape}, image_shape={data.shape}."
        )
    if bool(np.any(nonfinite & mask)):
        raise ValueError(
            "Non-finite image values were found inside the analysis mask before signature "
            "resampling."
        )

    nib = _maybe_import_nibabel()
    if nib is None:
        raise RuntimeError("Signature expression requires nibabel to prepare masked images.")
    filled = data.copy()
    filled[nonfinite] = 0.0
    return nib.Nifti1Image(filled, image_img.affine, image_img.header)


def _flatten_masked_pairs(
    *,
    img_data: Any,
    w_data: Any,
    mask_data: Optional[Any] = None,
) -> Tuple[List[float], List[float]]:
    """Flatten image and signature weights inside a fixed finite-weight mask."""
    import numpy as np  # type: ignore

    image = np.asanyarray(img_data, dtype=float)
    weights = np.asanyarray(w_data, dtype=float)
    fixed_mask = np.isfinite(weights)

    if mask_data is not None:
        mask = np.asanyarray(mask_data, dtype=bool)
        fixed_mask &= mask

    if not bool(np.any(fixed_mask)):
        raise ValueError("No voxels remain in the fixed signature mask.")

    invalid_image = fixed_mask & ~np.isfinite(image)
    if bool(np.any(invalid_image)):
        raise ValueError(
            "Non-finite image values were found inside the fixed signature mask; "
            "the trial cannot be scored on a smaller trial-specific voxel set."
        )

    x = image[fixed_mask].ravel().tolist()
    w = weights[fixed_mask].ravel().tolist()
    return x, w


def _scoring_mask(
    *,
    w_data: Any,
    mask_data: Optional[Any],
) -> Any:
    import numpy as np  # type: ignore

    weights = np.asanyarray(w_data, dtype=float)
    mask = np.isfinite(weights)
    if mask_data is not None:
        mask &= np.asanyarray(mask_data, dtype=bool)
    return mask


def _scoring_mask_sha256(*, scoring_mask: Any, affine: Any) -> str:
    import numpy as np  # type: ignore

    mask = np.ascontiguousarray(np.asanyarray(scoring_mask, dtype=np.uint8))
    grid_shape = np.ascontiguousarray(np.asarray(mask.shape, dtype=np.int64))
    grid_affine = np.ascontiguousarray(np.asarray(affine, dtype=np.float64))

    digest = hashlib.sha256()
    digest.update(grid_shape.tobytes())
    digest.update(grid_affine.tobytes())
    digest.update(mask.tobytes())
    return digest.hexdigest()


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


def _validate_fraction(value: Optional[float], *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be a finite fraction in [0, 1], got {value!r}.")
    return numeric


def _support_fraction(*, original_count: int, retained_count: int) -> Optional[float]:
    if original_count <= 0:
        return None
    return float(retained_count) / float(original_count)


def _mass_change_fraction(*, original_mass: float, retained_mass: float) -> Optional[float]:
    if original_mass <= 0:
        return None
    return abs(float(retained_mass) - float(original_mass)) / float(original_mass)


def _mass_loss_fraction(*, fixed_mass: float, retained_mass: float) -> Optional[float]:
    if fixed_mass <= 0:
        return None
    return 1.0 - float(retained_mass) / float(fixed_mass)


def _signature_support_summary(
    *,
    original_weights: Any,
    scored_weights: Any,
    mask_data: Optional[Any],
) -> Dict[str, Optional[float]]:
    import numpy as np  # type: ignore

    original = np.asanyarray(original_weights, dtype=float)
    scored = np.asanyarray(scored_weights, dtype=float)
    original_finite = np.isfinite(original)
    scored_mask = np.isfinite(scored)
    if mask_data is not None:
        scored_mask &= np.asanyarray(mask_data, dtype=bool)

    original_nonzero = original_finite & (np.abs(original) > 0.0)
    original_positive = original_finite & (original > 0.0)
    original_negative = original_finite & (original < 0.0)

    scored_nonzero = scored_mask & (np.abs(scored) > 0.0)
    scored_positive = scored_mask & (scored > 0.0)
    scored_negative = scored_mask & (scored < 0.0)

    positive_mass_original = float(np.sum(np.abs(original[original_positive])))
    negative_mass_original = float(np.sum(np.abs(original[original_negative])))
    positive_mass_scored = float(np.sum(np.abs(scored[scored_positive])))
    negative_mass_scored = float(np.sum(np.abs(scored[scored_negative])))

    return {
        "nonzero_support_fraction": _support_fraction(
            original_count=int(np.count_nonzero(original_nonzero)),
            retained_count=int(np.count_nonzero(scored_nonzero)),
        ),
        "positive_support_fraction": _support_fraction(
            original_count=int(np.count_nonzero(original_positive)),
            retained_count=int(np.count_nonzero(scored_positive)),
        ),
        "negative_support_fraction": _support_fraction(
            original_count=int(np.count_nonzero(original_negative)),
            retained_count=int(np.count_nonzero(scored_negative)),
        ),
        "positive_weight_mass_change_fraction": _mass_change_fraction(
            original_mass=positive_mass_original,
            retained_mass=positive_mass_scored,
        ),
        "negative_weight_mass_change_fraction": _mass_change_fraction(
            original_mass=negative_mass_original,
            retained_mass=negative_mass_scored,
        ),
    }


def _coverage_support_summary(
    *,
    weights: Any,
    scoring_mask: Any,
    coverage_mask: Any,
) -> Dict[str, Optional[float]]:
    import numpy as np  # type: ignore

    weight_data = np.asanyarray(weights, dtype=float)
    fixed_mask = np.asanyarray(scoring_mask, dtype=bool)
    coverage = np.asanyarray(coverage_mask, dtype=bool)
    if coverage.shape != fixed_mask.shape:
        raise ValueError(
            "Coverage grid mismatch while summarizing signature support: "
            f"coverage_shape={coverage.shape}, scoring_shape={fixed_mask.shape}."
        )

    fixed_nonzero = fixed_mask & (np.abs(weight_data) > 0.0)
    fixed_positive = fixed_mask & (weight_data > 0.0)
    fixed_negative = fixed_mask & (weight_data < 0.0)
    covered_nonzero = fixed_nonzero & coverage
    covered_positive = fixed_positive & coverage
    covered_negative = fixed_negative & coverage

    positive_mass_fixed = float(np.sum(np.abs(weight_data[fixed_positive])))
    negative_mass_fixed = float(np.sum(np.abs(weight_data[fixed_negative])))
    positive_mass_covered = float(np.sum(np.abs(weight_data[covered_positive])))
    negative_mass_covered = float(np.sum(np.abs(weight_data[covered_negative])))

    return {
        "coverage_nonzero_support_fraction": _support_fraction(
            original_count=int(np.count_nonzero(fixed_nonzero)),
            retained_count=int(np.count_nonzero(covered_nonzero)),
        ),
        "coverage_positive_support_fraction": _support_fraction(
            original_count=int(np.count_nonzero(fixed_positive)),
            retained_count=int(np.count_nonzero(covered_positive)),
        ),
        "coverage_negative_support_fraction": _support_fraction(
            original_count=int(np.count_nonzero(fixed_negative)),
            retained_count=int(np.count_nonzero(covered_negative)),
        ),
        "coverage_positive_weight_mass_loss_fraction": _mass_loss_fraction(
            fixed_mass=positive_mass_fixed,
            retained_mass=positive_mass_covered,
        ),
        "coverage_negative_weight_mass_loss_fraction": _mass_loss_fraction(
            fixed_mass=negative_mass_fixed,
            retained_mass=negative_mass_covered,
        ),
    }


def _raise_if_coverage_thresholds_fail(
    *,
    name: str,
    summary: Dict[str, Optional[float]],
    min_support_fraction: Optional[float],
    max_weight_mass_change_fraction: Optional[float],
) -> None:
    support_fields = (
        "coverage_nonzero_support_fraction",
        "coverage_positive_support_fraction",
        "coverage_negative_support_fraction",
    )
    support_failures = [
        field
        for field in support_fields
        if summary[field] is not None
        and min_support_fraction is not None
        and float(summary[field]) < min_support_fraction
    ]
    if support_failures:
        raise ValueError(
            f"{name} failed coverage signature support retention: "
            f"{support_failures} below {min_support_fraction:.3f}."
        )

    mass_loss_fields = (
        "coverage_positive_weight_mass_loss_fraction",
        "coverage_negative_weight_mass_loss_fraction",
    )
    mass_loss_failures = [
        field
        for field in mass_loss_fields
        if summary[field] is not None
        and max_weight_mass_change_fraction is not None
        and float(summary[field]) > max_weight_mass_change_fraction
    ]
    if mass_loss_failures:
        raise ValueError(
            f"{name} failed coverage signature weight-mass loss threshold: "
            f"{mass_loss_failures} above {max_weight_mass_change_fraction:.3f}."
        )


def _raise_if_support_thresholds_fail(
    *,
    name: str,
    summary: Dict[str, Optional[float]],
    min_support_fraction: Optional[float],
    max_weight_mass_change_fraction: Optional[float],
) -> None:
    support_fields = (
        "nonzero_support_fraction",
        "positive_support_fraction",
        "negative_support_fraction",
    )
    support_failures = [
        field
        for field in support_fields
        if summary[field] is not None
        and min_support_fraction is not None
        and float(summary[field]) < min_support_fraction
    ]
    if support_failures:
        raise ValueError(
            f"{name} failed positive/negative signature support retention: "
            f"{support_failures} below {min_support_fraction:.3f}."
        )

    mass_fields = (
        "positive_weight_mass_change_fraction",
        "negative_weight_mass_change_fraction",
    )
    mass_failures = [
        field
        for field in mass_fields
        if summary[field] is not None
        and max_weight_mass_change_fraction is not None
        and float(summary[field]) > max_weight_mass_change_fraction
    ]
    if mass_failures:
        raise ValueError(
            f"{name} failed positive/negative signature weight-mass stability: "
            f"{mass_failures} above {max_weight_mass_change_fraction:.3f}."
        )


def compute_signature_expression(
    *,
    stat_or_effect_img: Any,
    signature_root: Path,
    signature_specs: Sequence[Dict[str, str]],
    mask_img: Optional[Any] = None,
    coverage_mask_img: Optional[Any] = None,
    signatures: Optional[Sequence[str]] = None,
    resampling: str = "image_to_weights",
    min_support_fraction: Optional[float] = None,
    max_weight_mass_change_fraction: Optional[float] = None,
) -> List[SignatureResult]:
    """
    Compute multivariate signature expression (dot product and Pearson correlation).

    Scientific notes:

    - Images must be in the same space as the signature weight maps (typically MNI).
    - Both dot-product (pattern expression) and Pearson correlation (scale-invariant) are returned.
    - Uses intersection of finite voxels and an optional analysis mask (resampled as needed).
    - ``resampling="image_to_weights"`` (default): resample the target image to each
      signature's grid.
    - ``resampling="weights_to_image"``: resample each signature's weights to the
      target image grid.
    """
    files = discover_signature_files(signature_root, signature_specs)
    if signatures:
        requested = {str(name) for name in signatures}
        missing = sorted(requested - set(files))
        if missing:
            raise FileNotFoundError(
                f"Requested signatures were not found in signature_specs/signature_root: {missing}"
            )
        files = {k: v for k, v in files.items() if k in requested}
    if not files:
        raise ValueError("No signature files were resolved for signature expression.")

    nib = _maybe_import_nibabel()
    if nib is None:
        raise RuntimeError("Signature expression requires nibabel to load NIfTI images.")

    resampling = str(resampling or "image_to_weights").strip().lower().replace("-", "_")
    if resampling not in {"image_to_weights", "weights_to_image"}:
        raise ValueError("resampling must be one of: image_to_weights, weights_to_image")
    min_support_fraction = _validate_fraction(
        min_support_fraction,
        field_name="min_support_fraction",
    )
    max_weight_mass_change_fraction = _validate_fraction(
        max_weight_mass_change_fraction,
        field_name="max_weight_mass_change_fraction",
    )

    img = stat_or_effect_img
    if isinstance(img, (str, Path)):
        img = nib.load(str(img))

    results: List[SignatureResult] = []

    m = None
    if mask_img is not None:
        m = mask_img
        if isinstance(m, (str, Path)):
            m = nib.load(str(m))

    coverage = coverage_mask_img
    if isinstance(coverage, (str, Path)):
        coverage = nib.load(str(coverage))

    for name, w_path in files.items():
        try:
            w_img = nib.load(str(w_path))
            original_w_data = w_img.get_fdata()
            if resampling == "image_to_weights":
                x_img = _fill_nonfinite_background_for_resampling(
                    image_img=img,
                    mask_img=coverage if coverage is not None else m,
                )
                if not _image_grids_match(x_img, w_img):
                    x_img = _maybe_resample_to_img(
                        moving_img=x_img, target_img=w_img, interpolation="continuous"
                    )

                mask_data = _mask_data_on_grid(
                    mask_img=m,
                    reference_img=w_img,
                    mask_name="Fixed scoring mask",
                )
                coverage_data = _mask_data_on_grid(
                    mask_img=coverage,
                    reference_img=w_img,
                    mask_name="Coverage mask",
                )

                img_data = x_img.get_fdata()
                w_data = original_w_data
                scoring_affine = w_img.affine
            else:
                x_img = img
                if coverage is not None:
                    x_img = _fill_nonfinite_background_for_resampling(
                        image_img=img,
                        mask_img=coverage,
                    )
                w_on_ref = w_img
                if tuple(getattr(w_on_ref, "shape", ())) != tuple(getattr(img, "shape", ())):
                    w_on_ref = _maybe_resample_to_img(
                        moving_img=w_on_ref, target_img=img, interpolation="continuous"
                    )
                else:
                    try:
                        import numpy as np

                        if not np.allclose(w_on_ref.affine, img.affine):
                            w_on_ref = _maybe_resample_to_img(
                                moving_img=w_on_ref, target_img=img, interpolation="continuous"
                            )
                    except Exception:
                        w_on_ref = _maybe_resample_to_img(
                            moving_img=w_on_ref, target_img=img, interpolation="continuous"
                        )

                mask_data = _mask_data_on_grid(
                    mask_img=m,
                    reference_img=img,
                    mask_name="Fixed scoring mask",
                )
                coverage_data = _mask_data_on_grid(
                    mask_img=coverage,
                    reference_img=img,
                    mask_name="Coverage mask",
                )

                img_data = x_img.get_fdata()
                w_data = w_on_ref.get_fdata()
                scoring_affine = img.affine

            if tuple(getattr(w_data, "shape", ())) != tuple(getattr(img_data, "shape", ())):
                raise ValueError(
                    f"Resampled data grid mismatch for signature {name}: "
                    f"image_shape={getattr(img_data, 'shape', None)} "
                    f"weights_shape={getattr(w_data, 'shape', None)}"
                )

            if mask_data is not None and tuple(getattr(mask_data, "shape", ())) != tuple(
                getattr(img_data, "shape", ())
            ):
                raise ValueError(
                    f"Mask grid mismatch for signature {name}: "
                    f"mask_shape={getattr(mask_data,'shape',None)} img_shape={getattr(img_data,'shape',None)}"
                )

            scoring_mask = _scoring_mask(w_data=w_data, mask_data=mask_data)
            coverage_summary: Dict[str, Optional[float]] = {}
            if coverage_data is not None:
                coverage_summary = _coverage_support_summary(
                    weights=w_data,
                    scoring_mask=scoring_mask,
                    coverage_mask=coverage_data,
                )
                _raise_if_coverage_thresholds_fail(
                    name=name,
                    summary=coverage_summary,
                    min_support_fraction=min_support_fraction,
                    max_weight_mass_change_fraction=max_weight_mass_change_fraction,
                )

            support_summary = _signature_support_summary(
                original_weights=original_w_data,
                scored_weights=w_data,
                mask_data=mask_data,
            )
            _raise_if_support_thresholds_fail(
                name=name,
                summary=support_summary,
                min_support_fraction=min_support_fraction,
                max_weight_mass_change_fraction=max_weight_mass_change_fraction,
            )

            x_vec, w_vec = _flatten_masked_pairs(
                img_data=img_data, w_data=w_data, mask_data=mask_data
            )
            if not x_vec:
                raise ValueError(
                    f"No overlapping finite voxels remained for signature {name} after masking/resampling."
                )

            dot = _dot(x_vec, w_vec)
            nx = _norm(x_vec)
            nw = _norm(w_vec)
            cosine = float(dot / (nx * nw)) if nx > 0 and nw > 0 else None
            r = _pearson_r(x_vec, w_vec)

            results.append(
                SignatureResult(
                    name=name,
                    weight_path=w_path,
                    n_voxels=len(x_vec),
                    dot=float(dot),
                    cosine=cosine,
                    pearson_r=r,
                    **support_summary,
                    **coverage_summary,
                    scoring_mask_sha256=_scoring_mask_sha256(
                        scoring_mask=scoring_mask,
                        affine=scoring_affine,
                    ),
                )
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to compute signature expression for {name!r} using {w_path}: {exc}"
            ) from exc

    return results


def write_signature_expression_tsv(
    results: Sequence[SignatureResult],
    path: Path,
) -> Path:
    """Write signature expression beside the contrast's maps, and return the path.

    This is an analysis output, not a rendering artifact: computing it needs the
    weight maps and the study's signature configuration, neither of which the report
    path should reach for. The report reads this file.

    A metric that could not be computed is written as an empty field rather than a
    zero, because zero is a meaningful similarity and "not measured" is not.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _number(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.6g}"

    lines = ["\t".join(("signature", "dot", "cosine", "pearson_r", "n_voxels", "weight_path"))]
    for result in results:
        lines.append(
            "\t".join(
                (
                    result.name,
                    _number(result.dot),
                    _number(result.cosine),
                    _number(result.pearson_r),
                    str(int(result.n_voxels)),
                    str(result.weight_path),
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
