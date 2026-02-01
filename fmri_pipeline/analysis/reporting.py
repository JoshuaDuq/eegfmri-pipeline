from __future__ import annotations

import html
import logging
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
from fmri_pipeline.analysis.pain_signatures import discover_pain_signature_files, compute_pain_signature_expression, PainSignatureResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportImage:
    title: str
    path: Path
    caption: str = ""


@dataclass(frozen=True)
class ReportTable:
    title: str
    tsv_path: Optional[Path] = None
    html_table: str = ""
    caption: str = ""


@dataclass(frozen=True)
class ReportSpaceSection:
    space: str  # "native" | "mni"
    images: Tuple[ReportImage, ...] = ()
    tables: Tuple[ReportTable, ...] = ()
    summary: Optional[Dict[str, Any]] = None


def _safe_relpath(base_dir: Path, target_path: Path) -> str:
    try:
        return str(target_path.relative_to(base_dir))
    except Exception:
        return str(target_path)


def _discover_design_matrix_qc(contrast_dir: Path) -> List[Tuple[Optional[Path], Optional[Path]]]:
    """
    Return list of (png, tsv) pairs for design matrix QC files, best-effort.
    """
    qc_dir = contrast_dir / "qc"
    if not qc_dir.exists():
        return []

    pngs = sorted(qc_dir.glob("*design_matrix.png"))
    tsvs = sorted(qc_dir.glob("*design_matrix.tsv"))

    def _run_key(p: Path) -> str:
        # Match filenames like ..._run-06_design_matrix.png
        parts = p.name.split("_")
        for part in parts:
            if part.startswith("run-"):
                return part
        return p.stem

    png_by_run = {_run_key(p): p for p in pngs}
    tsv_by_run = {_run_key(p): p for p in tsvs}
    runs = sorted(set(png_by_run) | set(tsv_by_run))

    out: List[Tuple[Optional[Path], Optional[Path]]] = []
    for r in runs:
        out.append((png_by_run.get(r), tsv_by_run.get(r)))
    return out


def _vif_from_design(X: "np.ndarray") -> "np.ndarray":
    """
    Variance inflation factor per column of design matrix X (n_samples, n_features).

    VIF_j = 1 / (1 - R²_j), where R²_j is from regressing column j on all other columns.
    Returns inf where R² >= 1 (perfect collinearity) or computation fails.
    """
    import numpy as np

    n, p = X.shape
    if p < 2:
        return np.array([], dtype=np.float64)

    vif = np.full(p, np.nan, dtype=np.float64)
    for j in range(p):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        Z_const = np.column_stack([np.ones(n, dtype=X.dtype), Z])
        try:
            beta, residuals, rank, _ = np.linalg.lstsq(Z_const, y, rcond=None)
            if residuals.size:
                ss_res = float(residuals.flat[0])
            else:
                ss_res = float(np.sum((y - Z_const @ beta) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 0:
                vif[j] = np.inf
                continue
            r_sq = 1.0 - (ss_res / ss_tot)
            if r_sq >= 1.0 or np.isnan(r_sq):
                vif[j] = np.inf
            else:
                vif[j] = 1.0 / (1.0 - r_sq)
        except Exception:
            vif[j] = np.inf
    return vif


def generate_carpet_qc_images(
    *,
    contrast_dir: Path,
    cfg: FmriPlottingConfig,
    run_meta: Dict[str, Any],
    mask_img_path: Optional[Path] = None,
    max_voxels: int = 6000,
) -> List[ReportImage]:
    """
    Generate a "carpet plot" QC image (best-effort).

    Uses the included BOLD run paths from run_meta["included_bold_paths"].
    """
    cfg = cfg.normalized()
    if not cfg.enabled or not cfg.include_carpet_qc:
        return []

    bold_paths = run_meta.get("included_bold_paths") if isinstance(run_meta, dict) else None
    if not isinstance(bold_paths, list) or not bold_paths:
        return []

    try:
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        nib = _maybe_import_nibabel()
        if nib is None:
            return []

        qc_dir = contrast_dir / "plots" / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        mask_img = None
        if mask_img_path and mask_img_path.exists():
            try:
                mask_img = nib.load(str(mask_img_path))
            except Exception:
                mask_img = None

        mats = []
        run_breaks = [0]

        for bp in bold_paths:
            if not bp:
                continue
            p = Path(str(bp))
            if not p.exists():
                continue
            img = nib.load(str(p))
            data = np.asanyarray(img.dataobj)
            if data.ndim != 4:
                continue

            m = None
            if mask_img is not None:
                try:
                    mdata = np.asanyarray(mask_img.dataobj).astype(bool)
                    if mdata.shape != data.shape[:3]:
                        try:
                            from nibabel.processing import resample_from_to  # type: ignore

                            mask_res = resample_from_to(mask_img, (data.shape[:3], img.affine), order=0)
                            mdata = np.asanyarray(mask_res.dataobj).astype(bool)
                        except Exception:
                            mdata = None
                    if mdata is not None and mdata.shape == data.shape[:3]:
                        m = mdata
                except Exception:
                    m = None

            if m is None:
                # fall back to non-zero voxels in mean image
                mean_img = np.mean(data, axis=3)
                m = np.isfinite(mean_img) & (mean_img != 0)

            vox_ts = data[m]  # (n_voxels, T)
            if vox_ts.size == 0:
                continue

            # Downsample voxels for readability and performance
            if vox_ts.shape[0] > max_voxels:
                idx = np.linspace(0, vox_ts.shape[0] - 1, max_voxels).astype(int)
                vox_ts = vox_ts[idx, :]

            mean = np.mean(vox_ts, axis=1, keepdims=True)
            std = np.std(vox_ts, axis=1, keepdims=True)
            std[std == 0] = 1.0
            vox_ts = (vox_ts - mean) / std
            vox_ts = np.clip(vox_ts, -3, 3)

            mats.append(vox_ts)
            run_breaks.append(run_breaks[-1] + int(vox_ts.shape[1]))

        if not mats:
            return []

        carpet = np.concatenate(mats, axis=1)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        im = ax.imshow(
            carpet,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-2.5,
            vmax=2.5,
        )
        ax.set_title("Carpet plot (standardized voxel time series; concatenated runs)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Voxels (subsampled)")
        for b in run_breaks[1:-1]:
            ax.axvline(b, color="#000000", linewidth=0.7, alpha=0.35)
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("z (per voxel)")
        fig.tight_layout()

        images: List[ReportImage] = []
        for fmt in cfg.formats:
            out_path = qc_dir / f"carpet_qc.{fmt}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
            images.append(ReportImage(title="QC: Carpet plot", path=out_path))
        plt.close(fig)
        return images
    except Exception as exc:
        logger.warning("Failed to generate carpet QC (%s)", exc)
        return []


def generate_tsnr_qc_images(
    *,
    contrast_dir: Path,
    cfg: FmriPlottingConfig,
    run_meta: Dict[str, Any],
    mask_img_path: Optional[Path] = None,
) -> List[ReportImage]:
    """
    Generate tSNR QC images (best-effort).

    Uses the included BOLD run paths from run_meta["included_bold_paths"].
    """
    cfg = cfg.normalized()
    if not cfg.enabled or not cfg.include_tsnr_qc:
        return []

    bold_paths = run_meta.get("included_bold_paths") if isinstance(run_meta, dict) else None
    if not isinstance(bold_paths, list) or not bold_paths:
        return []

    try:
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        nib = _maybe_import_nibabel()
        if nib is None:
            return []

        qc_dir = contrast_dir / "plots" / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        mask_img = None
        if mask_img_path and mask_img_path.exists():
            try:
                mask_img = nib.load(str(mask_img_path))
            except Exception:
                mask_img = None

        tsnr_sum = None
        tsnr_n = 0
        ref_affine = None

        for bp in bold_paths:
            if not bp:
                continue
            p = Path(str(bp))
            if not p.exists():
                continue
            img = nib.load(str(p))
            data = np.asanyarray(img.dataobj)
            if data.ndim != 4:
                continue
            ref_affine = img.affine if ref_affine is None else ref_affine

            m = None
            if mask_img is not None:
                try:
                    mdata = np.asanyarray(mask_img.dataobj).astype(bool)
                    if mdata.shape != data.shape[:3]:
                        try:
                            from nibabel.processing import resample_from_to  # type: ignore

                            mask_res = resample_from_to(mask_img, (data.shape[:3], img.affine), order=0)
                            mdata = np.asanyarray(mask_res.dataobj).astype(bool)
                        except Exception:
                            mdata = None
                    if mdata is not None and mdata.shape == data.shape[:3]:
                        m = mdata
                except Exception:
                    m = None
            if m is None:
                mean_img = np.mean(data, axis=3)
                m = np.isfinite(mean_img) & (mean_img != 0)

            mean = np.mean(data, axis=3)
            std = np.std(data, axis=3)
            std = np.where(std == 0, np.nan, std)
            tsnr = mean / std
            tsnr = np.nan_to_num(tsnr, nan=0.0, posinf=0.0, neginf=0.0)
            tsnr = np.where(m, tsnr, 0.0)

            if tsnr_sum is None:
                tsnr_sum = tsnr.astype(float)
            else:
                if tsnr_sum.shape != tsnr.shape:
                    continue
                tsnr_sum += tsnr
            tsnr_n += 1

        if tsnr_sum is None or tsnr_n == 0:
            return []

        tsnr_mean = tsnr_sum / float(tsnr_n)

        # Write a NIfTI for downstream use (best-effort)
        try:
            if ref_affine is not None:
                tsnr_img = nib.Nifti1Image(tsnr_mean.astype("float32"), affine=ref_affine)
                nii_path = qc_dir / "tsnr_mean.nii.gz"
                nib.save(tsnr_img, str(nii_path))
        except Exception:
            pass

        # Pick informative slices (max mask coverage)
        m = tsnr_mean > 0
        if m.any():
            x_idx = int(np.argmax(m.sum(axis=(1, 2))))
            y_idx = int(np.argmax(m.sum(axis=(0, 2))))
            z_idx = int(np.argmax(m.sum(axis=(0, 1))))
        else:
            x_idx, y_idx, z_idx = (tsnr_mean.shape[0] // 2, tsnr_mean.shape[1] // 2, tsnr_mean.shape[2] // 2)

        vals = tsnr_mean[m]
        vmax = float(np.percentile(vals, 98)) if vals.size else float(np.max(tsnr_mean))
        vmax = max(vmax, 1.0)

        # Map montage
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), dpi=150)
        im0 = axes[0].imshow(tsnr_mean[x_idx, :, :].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axes[0].set_title("Sagittal")
        axes[1].imshow(tsnr_mean[:, y_idx, :].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axes[1].set_title("Coronal")
        axes[2].imshow(tsnr_mean[:, :, z_idx].T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axes[2].set_title("Axial")
        for ax in axes:
            ax.axis("off")
        cbar = fig.colorbar(im0, ax=axes, fraction=0.02, pad=0.06)
        cbar.set_label("tSNR")
        fig.suptitle(f"tSNR (mean across runs, n={tsnr_n})")
        fig.tight_layout(rect=[0, 0, 0.88, 0.96])

        images: List[ReportImage] = []
        for fmt in cfg.formats:
            out_path = qc_dir / f"tsnr_map.{fmt}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
            images.append(ReportImage(title="QC: tSNR map", path=out_path))
        plt.close(fig)

        # Histogram
        if vals.size:
            fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
            ax.hist(vals, bins=80, color="#2CA02C", alpha=0.9, edgecolor="none")
            ax.set_title("tSNR distribution (masked voxels)")
            ax.set_xlabel("tSNR")
            ax.set_ylabel("voxels")
            fig.tight_layout()
            for fmt in cfg.formats:
                out_path = qc_dir / f"tsnr_hist.{fmt}"
                fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
                images.append(ReportImage(title="QC: tSNR histogram", path=out_path))
            plt.close(fig)

        return images
    except Exception as exc:
        logger.warning("Failed to generate tSNR QC (%s)", exc)
        return []


def generate_pain_signature_tables(
    *,
    contrast_dir: Path,
    cfg: FmriPlottingConfig,
    mni_effect_img: Optional[Any],
    mni_mask_img: Optional[Any],
    signature_root: Optional[Path],
) -> List[ReportTable]:
    """
    Compute NPS/SIIPS1 expression on the unthresholded MNI effect-size map and return report tables (best-effort).
    """
    cfg = cfg.normalized()
    if not cfg.enabled or not bool(getattr(cfg, "include_signatures", True)):
        return []
    if mni_effect_img is None:
        return []
    if signature_root is None:
        return []

    sig_files = discover_pain_signature_files(signature_root)
    if not sig_files:
        return []

    results: List[PainSignatureResult] = []
    try:
        results = compute_pain_signature_expression(
            stat_or_effect_img=mni_effect_img,
            signature_root=signature_root,
            mask_img=mni_mask_img,
            signatures=sorted(sig_files.keys()),
        )
    except Exception as exc:
        logger.warning("Failed to compute pain signatures (%s)", exc)
        return []

    if not results:
        return []

    qc_dir = contrast_dir / "plots" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = qc_dir / "pain_signature_expression.tsv"
    try:
        header = ["signature", "dot", "cosine", "pearson_r", "n_voxels", "weight_path"]
        lines = ["\t".join(header)]
        for r in results:
            lines.append(
                "\t".join(
                    [
                        r.name,
                        f"{r.dot:.6g}",
                        "" if r.cosine is None else f"{r.cosine:.6g}",
                        "" if r.pearson_r is None else f"{r.pearson_r:.6g}",
                        str(int(r.n_voxels)),
                        str(r.weight_path),
                    ]
                )
            )
        tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        tsv_path = None

    def _fmt(x: Any) -> str:
        if x is None:
            return ""
        try:
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    th = "".join(
        f"<th>{html.escape(k)}</th>"
        for k in ["signature", "dot", "cosine", "pearson_r", "n_voxels"]
    )
    trs = []
    for r in results:
        trs.append(
            "<tr>"
            + f"<td>{html.escape(r.name)}</td>"
            + f"<td>{html.escape(_fmt(r.dot))}</td>"
            + f"<td>{html.escape(_fmt(r.cosine))}</td>"
            + f"<td>{html.escape(_fmt(r.pearson_r))}</td>"
            + f"<td>{html.escape(str(int(r.n_voxels)))}</td>"
            + "</tr>"
        )
    html_table = f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"

    return [
        ReportTable(
            title="Multivariate Pain Signature Expression (MNI effect-size map)",
            tsv_path=tsv_path,
            html_table=html_table,
            caption=(
                "Dot product is the raw pattern expression; cosine and Pearson r are scale-invariant. "
                "Computed on the unthresholded effect-size map, after resampling the image (and mask) to each signature's grid."
            ),
        )
    ]


def build_fmri_report_html(
    *,
    report_path: Path,
    subject: str,
    task: str,
    contrast_name: str,
    z_threshold: float,
    include_unthresholded: bool,
    sections: Sequence[ReportSpaceSection],
    extra_notes: Optional[Sequence[str]] = None,
    embed_images: bool = True,
    methods_payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a self-contained HTML report.
    """
    base_dir = report_path.parent

    def esc(s: Any) -> str:
        return html.escape("" if s is None else str(s))

    notes = list(extra_notes or [])
    notes.append(f"Thresholded panels use |z| > {z_threshold:.2f}.")
    if include_unthresholded:
        notes.append("Unthresholded panels are also included.")

    qc_pairs = _discover_design_matrix_qc(base_dir)

    def _mime_for_path(p: Path) -> str:
        suf = p.suffix.lower()
        if suf == ".png":
            return "image/png"
        if suf == ".svg":
            return "image/svg+xml"
        return "application/octet-stream"

    def _image_src(p: Path) -> str:
        if not embed_images:
            return _safe_relpath(base_dir, p)
        try:
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{_mime_for_path(p)};base64,{b64}"
        except Exception:
            return _safe_relpath(base_dir, p)

    css = """
    :root { --fg:#111; --muted:#555; --bg:#fff; --card:#f7f7f9; --border:#e6e6ea; }
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
           background: var(--bg); color: var(--fg); margin: 0; padding: 24px; line-height: 1.35; }
    .container { max-width: 1180px; margin: 0 auto; }
    h1 { font-size: 22px; margin: 0 0 6px 0; }
    .subhead { color: var(--muted); margin: 0 0 16px 0; font-size: 13px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin: 14px 0; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .fig { background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
    .fig-title { font-weight: 650; margin: 0 0 6px 0; }
    .fig-cap { color: var(--muted); font-size: 12px; margin-top: 6px; }
    img { width: 100%; height: auto; border-radius: 8px; display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid var(--border); padding: 6px 8px; text-align: left; }
    th { background: #fafafa; position: sticky; top: 0; }
    .kvs { display: grid; grid-template-columns: 160px 1fr; gap: 4px 12px; font-size: 13px; }
    .k { color: var(--muted); }
    a { color: #0a58ca; text-decoration: none; }
    a:hover { text-decoration: underline; }
    """

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head>")
    parts.append('<meta charset="utf-8" />')
    parts.append(f"<title>{esc(subject)} · {esc(task)} · {esc(contrast_name)}</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head><body><div class='container'>")

    parts.append(f"<h1>{esc(subject)} · task-{esc(task)} · contrast-{esc(contrast_name)}</h1>")
    parts.append("<p class='subhead'>First-level GLM contrast report</p>")

    parts.append("<div class='card'><div class='kvs'>")
    parts.append(f"<div class='k'>Threshold</div><div>|z| &gt; {z_threshold:.2f}</div>")
    parts.append(f"<div class='k'>Unthresholded</div><div>{'Yes' if include_unthresholded else 'No'}</div>")
    parts.append("</div></div>")

    if notes:
        parts.append("<div class='card'><ul>")
        for n in notes:
            parts.append(f"<li>{esc(n)}</li>")
        parts.append("</ul></div>")

    for section in sections:
        if section.space == "native":
            space_title = "Native (subject space)"
        elif section.space == "mni":
            space_title = "MNI (standard space)"
        else:
            space_title = str(section.space).strip() or "Section"
        parts.append(f"<div class='card'><h2>{esc(space_title)}</h2>")

        if section.summary:
            parts.append("<div class='kvs'>")
            for k, v in section.summary.items():
                parts.append(f"<div class='k'>{esc(k)}</div><div>{esc(v)}</div>")
            parts.append("</div>")

        parts.append("<div class='grid'>")

        for img in section.images:
            rel = esc(_image_src(img.path))
            parts.append("<div class='fig'>")
            parts.append(f"<div class='fig-title'>{esc(img.title)}</div>")
            parts.append(f"<img src='{rel}' loading='lazy' />")
            if img.caption:
                parts.append(f"<div class='fig-cap'>{esc(img.caption)}</div>")
            parts.append("</div>")

        for tbl in section.tables:
            parts.append("<div class='fig'>")
            parts.append(f"<div class='fig-title'>{esc(tbl.title)}</div>")
            if tbl.tsv_path is not None:
                rel_tsv = esc(_safe_relpath(base_dir, tbl.tsv_path))
                parts.append(f"<div class='fig-cap'><a href='{rel_tsv}'>Download TSV</a></div>")
            if tbl.html_table:
                parts.append(tbl.html_table)
            if tbl.caption:
                parts.append(f"<div class='fig-cap'>{esc(tbl.caption)}</div>")
            parts.append("</div>")

        parts.append("</div></div>")

    if qc_pairs:
        parts.append("<div class='card'><h2>Design Matrices (QC)</h2><div class='grid'>")
        for png_path, tsv_path in qc_pairs:
            parts.append("<div class='fig'>")
            title = "Design matrix"
            if png_path is not None:
                title = png_path.name.replace("_design_matrix.png", "").replace("_", " ")
            parts.append(f"<div class='fig-title'>{esc(title)}</div>")

            if png_path is not None:
                rel_png = esc(_image_src(png_path))
                parts.append(f"<img src='{rel_png}' loading='lazy' />")
            else:
                parts.append("<div class='fig-cap'>PNG not found</div>")

            if tsv_path is not None:
                rel_tsv = esc(_safe_relpath(base_dir, tsv_path))
                parts.append(f"<div class='fig-cap'><a href='{rel_tsv}'>Open design matrix TSV</a></div>")
            parts.append("</div>")
        parts.append("</div></div>")

    if methods_payload:
        parts.append("<div class='card'><h2>Methods / Provenance</h2>")
        parts.append("<details><summary>Show details</summary>")
        parts.append("<pre style='white-space:pre-wrap; font-size:12px; margin-top:10px;'>")
        try:
            parts.append(esc(json.dumps(methods_payload, indent=2, sort_keys=True)))
        except Exception:
            parts.append(esc(repr(methods_payload)))
        parts.append("</pre></details></div>")

    parts.append("</div></body></html>")
    return "\n".join(parts)


def _maybe_import_nilearn_plotting():
    try:
        from nilearn import plotting  # type: ignore

        return plotting
    except Exception:
        return None


def _maybe_import_nilearn_reporting():
    try:
        from nilearn import reporting  # type: ignore

        return reporting
    except Exception:
        return None


def _maybe_import_nibabel():
    try:
        import nibabel as nib  # type: ignore

        return nib
    except Exception:
        return None


def _save_nilearn_display(display: Any, out_path: Path, *, dpi: int = 300) -> None:
    fig = getattr(display, "figure", None) or getattr(display, "_fig", None)
    if fig is None and hasattr(display, "frame_axes"):
        fig = getattr(display.frame_axes, "figure", None)
    if fig is None:
        raise RuntimeError("Could not resolve matplotlib figure from nilearn display object")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def _stat_summary_from_img(img: Any, mask_img: Optional[Any] = None) -> Dict[str, Any]:
    nib = _maybe_import_nibabel()
    if nib is None:
        return {}

    import numpy as np

    try:
        data = np.asarray(img.get_fdata())
        if mask_img is not None:
            mask = np.asarray(mask_img.get_fdata()).astype(bool)
            if mask.shape != data.shape:
                try:
                    from nilearn.image import resample_to_img  # type: ignore

                    mask_res = resample_to_img(
                        mask_img, img, interpolation="nearest",
                        force_resample=True, copy_header=True,
                    )
                    mask = np.asarray(mask_res.get_fdata()).astype(bool)
                except Exception:
                    mask = None
            if mask is not None and mask.shape == data.shape:
                data = data[mask]
            else:
                # Fallback mask: exclude zeros (common outside-brain background in z-maps).
                data = data[data != 0]
        data = data[np.isfinite(data)]
        if data.size == 0:
            return {"n_voxels": 0}
        return {
            "n_voxels": int(data.size),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "p99_abs": float(np.percentile(np.abs(data), 99)),
        }
    except Exception:
        return {}


def _robust_vmax_abs(img: Any, mask_img: Optional[Any] = None, *, pct: float = 99.0) -> Optional[float]:
    try:
        import numpy as np

        data = np.asarray(img.get_fdata())
        if mask_img is not None:
            m = np.asarray(mask_img.get_fdata()).astype(bool)
            if m.shape == data.shape:
                data = data[m]
            else:
                data = data[data != 0]
        data = data[np.isfinite(data)]
        if data.size == 0:
            return None
        return float(np.percentile(np.abs(data), pct))
    except Exception:
        return None


def _compute_threshold_for_cfg(stat_img: Any, cfg: FmriPlottingConfig) -> Tuple[Optional[Any], Optional[float], str]:
    """
    Returns (thresholded_img_or_none, threshold_value_or_none, label).
    """
    cfg = cfg.normalized()
    if cfg.threshold_mode == "none":
        return None, None, "none"

    if cfg.threshold_mode == "fdr":
        try:
            from nilearn.glm import threshold_stats_img  # type: ignore

            thr_img, thr = threshold_stats_img(
                stat_img,
                alpha=float(cfg.fdr_q),
                height_control="fdr",
                cluster_threshold=int(cfg.cluster_min_voxels) if cfg.cluster_min_voxels > 0 else 0,
                two_sided=bool(cfg.two_sided),
            )
            return thr_img, float(thr), f"fdr q={cfg.fdr_q:.3f}"
        except Exception:
            # Fall back to plain z-threshold if FDR is unavailable.
            return None, float(cfg.z_threshold), f"z |z|>{cfg.z_threshold:.2f}"

    # cfg.threshold_mode == "z"
    return None, float(cfg.z_threshold), f"z |z|>{cfg.z_threshold:.2f}"


def _apply_cluster_min_voxels(
    stat_img: Any,
    *,
    threshold: float,
    min_voxels: int,
    two_sided: bool,
) -> Any:
    """
    Apply a cluster-extent filter to a thresholded stat image (best-effort).

    Returns a new image with small clusters zeroed out.
    """
    if min_voxels <= 0:
        return stat_img

    try:
        import numpy as np
        from scipy import ndimage
        import nibabel as nib  # type: ignore

        data = np.asarray(stat_img.get_fdata())
        if two_sided:
            mask = np.abs(data) > float(threshold)
        else:
            mask = data > float(threshold)

        structure = np.ones((3, 3, 3), dtype=bool)
        labels, n = ndimage.label(mask, structure=structure)
        if n == 0:
            return stat_img
        counts = np.bincount(labels.ravel())
        keep = counts >= int(min_voxels)
        keep[0] = False
        keep_mask = keep[labels]
        out = np.zeros_like(data)
        out[keep_mask] = data[keep_mask]
        return nib.Nifti1Image(out, stat_img.affine, stat_img.header)
    except Exception:
        return stat_img


def generate_fmri_space_section(
    *,
    space: str,
    stat_img_path: Optional[Path] = None,
    stat_img: Optional[Any] = None,
    out_base_dir: Path,
    formats: Sequence[str],
    z_threshold: float,
    include_unthresholded: bool,
    plot_types: Sequence[str],
    bg_img_path: Optional[Path] = None,
    bg_img: Optional[Any] = None,
    mask_img_path: Optional[Path] = None,
    mask_img: Optional[Any] = None,
    title_prefix: str = "",
    cfg: Optional[FmriPlottingConfig] = None,
    vmax: Optional[float] = None,
    effect_img: Optional[Any] = None,
    variance_img: Optional[Any] = None,
) -> ReportSpaceSection:
    """
    Generate plots for one space into out_base_dir/plots/<space>/, best-effort.
    """
    space = (space or "").strip().lower()
    formats = [f.lower() for f in formats]
    plot_types = [p.lower() for p in plot_types]

    plotting = _maybe_import_nilearn_plotting()
    reporting = _maybe_import_nilearn_reporting()
    nib = _maybe_import_nibabel()
    if plotting is None or nib is None:
        logger.warning("nilearn/nibabel not available; skipping fMRI plotting outputs")
        return ReportSpaceSection(space=space, images=(), tables=(), summary={"plots": "skipped (missing nilearn/nibabel)"})

    if stat_img is None:
        if stat_img_path is None:
            raise ValueError("Either stat_img or stat_img_path must be provided")
        stat_img = nib.load(str(stat_img_path))
    if bg_img is None and bg_img_path and bg_img_path.exists():
        bg_img = nib.load(str(bg_img_path))
    if mask_img is None and mask_img_path and mask_img_path.exists():
        mask_img = nib.load(str(mask_img_path))

    out_dir = out_base_dir / "plots" / space
    out_dir.mkdir(parents=True, exist_ok=True)

    images: List[ReportImage] = []
    tables: List[ReportTable] = []

    def _add_image(title: str, display: Any, stem: str, caption: str = "") -> None:
        for fmt in formats:
            out_path = out_dir / f"{stem}.{fmt}"
            _save_nilearn_display(display, out_path)
            images.append(ReportImage(title=title, path=out_path, caption=caption))

    cfg_obj = cfg.normalized() if cfg is not None else FmriPlottingConfig(enabled=True)
    thr_img, thr_val, thr_label = _compute_threshold_for_cfg(stat_img, cfg_obj)
    if cfg_obj.threshold_mode == "z" and cfg_obj.cluster_min_voxels > 0 and thr_val is not None:
        thr_img = _apply_cluster_min_voxels(
            stat_img,
            threshold=float(thr_val),
            min_voxels=int(cfg_obj.cluster_min_voxels),
            two_sided=bool(cfg_obj.two_sided),
        )

    # Determine vmax for the z-stat panels (robust scaling unless user provided manual).
    z_vmax = vmax
    if z_vmax is None:
        z_vmax = _robust_vmax_abs(stat_img, mask_img=mask_img) or None

    # Slices (mosaic): Z-stat
    if "slices" in plot_types:
        try:
            if include_unthresholded:
                disp = plotting.plot_stat_map(
                    stat_img,
                    bg_img=bg_img,
                    title=f"{title_prefix}Z map (unthresholded)".strip(),
                    display_mode="mosaic",
                    threshold=None,
                    colorbar=True,
                    vmax=z_vmax,
                    symmetric_cbar=True,
                    annotate=False,
                )
                _add_image("Stat map (slices) · unthresholded", disp, "stat_slices_unthresholded")
            if thr_label != "none":
                disp = plotting.plot_stat_map(
                    thr_img if thr_img is not None else stat_img,
                    bg_img=bg_img,
                    title=f"{title_prefix}Z map (thresholded: {thr_label})".strip(),
                    display_mode="mosaic",
                    threshold=None if thr_img is not None else (float(thr_val) if thr_val is not None else None),
                    colorbar=True,
                    vmax=z_vmax,
                    symmetric_cbar=True,
                    annotate=False,
                )
                _add_image("Stat map (slices) · thresholded", disp, "stat_slices_thresholded", caption=thr_label)
        except Exception as exc:
            logger.warning("Failed to generate stat-map slices (%s)", exc)

    # Glass brain: Z-stat
    if "glass" in plot_types:
        try:
            if include_unthresholded:
                disp = plotting.plot_glass_brain(
                    stat_img,
                    title=f"{title_prefix}Glass brain (unthresholded)".strip(),
                    threshold=None,
                    colorbar=True,
                    vmax=z_vmax,
                )
                _add_image("Glass brain · unthresholded", disp, "glass_unthresholded")
            if thr_label != "none":
                disp = plotting.plot_glass_brain(
                    thr_img if thr_img is not None else stat_img,
                    title=f"{title_prefix}Glass brain (thresholded: {thr_label})".strip(),
                    threshold=None if thr_img is not None else (float(thr_val) if thr_val is not None else None),
                    colorbar=True,
                    vmax=z_vmax,
                )
                _add_image("Glass brain · thresholded", disp, "glass_thresholded", caption=thr_label)
        except Exception as exc:
            logger.warning("Failed to generate glass brain (%s)", exc)

    # Histogram
    if "hist" in plot_types:
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            data = np.asarray(stat_img.get_fdata())
            if mask_img is not None:
                m = np.asarray(mask_img.get_fdata()).astype(bool)
                if m.shape != data.shape:
                    try:
                        from nilearn.image import resample_to_img  # type: ignore

                        m_res = resample_to_img(
                        mask_img, stat_img, interpolation="nearest",
                        force_resample=True, copy_header=True,
                    )
                        m = np.asarray(m_res.get_fdata()).astype(bool)
                    except Exception:
                        m = None
                if m is not None and m.shape == data.shape:
                    data = data[m]
                else:
                    data = data[data != 0]
            data = data[np.isfinite(data)]
            if data.size:
                fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
                ax.hist(data, bins=80, color="#1F77B4", alpha=0.9, edgecolor="none")
                if thr_val is not None and thr_label != "none":
                    ax.axvline(float(thr_val), color="#E31A1C", linestyle="--", linewidth=1.5, label=f"+thr")
                    ax.axvline(-float(thr_val), color="#E31A1C", linestyle="--", linewidth=1.5)
                ax.set_title("Z-statistic distribution")
                ax.set_xlabel("z")
                ax.set_ylabel("voxels")
                ax.legend(frameon=False, fontsize=9)
                fig.tight_layout()
                for fmt in formats:
                    out_path = out_dir / f"z_hist.{fmt}"
                    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
                    images.append(ReportImage(title="Z histogram", path=out_path))
                plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to generate z histogram (%s)", exc)

    # Cluster/peak table
    if "clusters" in plot_types and reporting is not None:
        try:
            if thr_val is None:
                raise ValueError("No threshold available for clusters table")
            tbl = reporting.get_clusters_table(
                stat_img,
                stat_threshold=float(thr_val),
                cluster_threshold=int(cfg_obj.cluster_min_voxels) if cfg_obj.cluster_min_voxels > 0 else 0,
                two_sided=bool(cfg_obj.two_sided),
            )  # type: ignore[attr-defined]
            tsv_path = out_dir / "clusters.tsv"
            try:
                tbl.to_csv(tsv_path, sep="\t", index=False)
            except Exception:
                tsv_path = None
            html_table = ""
            try:
                html_table = tbl.to_html(index=False, border=0, classes="")
            except Exception:
                html_table = ""
            tables.append(
                ReportTable(
                    title="Clusters / Peaks",
                    tsv_path=tsv_path,
                    html_table=html_table,
                    caption=f"{'two-sided' if cfg_obj.two_sided else 'one-sided'}, {thr_label}",
                )
            )
        except Exception as exc:
            logger.warning("Failed to generate clusters table (%s)", exc)

    summary = _stat_summary_from_img(stat_img, mask_img=mask_img)
    if summary:
        summary = {
            "n_voxels (masked)": summary.get("n_voxels", ""),
            "min z": f"{summary.get('min', float('nan')):.3f}" if "min" in summary else "",
            "max z": f"{summary.get('max', float('nan')):.3f}" if "max" in summary else "",
            "p99(|z|)": f"{summary.get('p99_abs', float('nan')):.3f}" if "p99_abs" in summary else "",
        }

    # Optional: effect size + standard error (from variance)
    try:
        if bool(getattr(cfg_obj, "include_effect_size", True)) and effect_img is not None:
            eff_vmax = _robust_vmax_abs(effect_img, mask_img=mask_img) or None
            if "slices" in plot_types:
                disp = plotting.plot_stat_map(
                    effect_img,
                    bg_img=bg_img,
                    title=f"{title_prefix}Effect size (unthresholded)".strip(),
                    display_mode="mosaic",
                    threshold=None,
                    colorbar=True,
                    vmax=eff_vmax,
                    symmetric_cbar=True,
                    cmap="cold_hot",
                    annotate=False,
                )
                _add_image("Effect size (slices)", disp, "effect_slices")
            if "glass" in plot_types:
                disp = plotting.plot_glass_brain(
                    effect_img,
                    title=f"{title_prefix}Effect size (glass)".strip(),
                    threshold=None,
                    colorbar=True,
                    vmax=eff_vmax,
                    cmap="cold_hot",
                )
                _add_image("Effect size (glass)", disp, "effect_glass")
    except Exception as exc:
        logger.warning("Failed to generate effect size panels (%s)", exc)

    try:
        if bool(getattr(cfg_obj, "include_standard_error", True)) and variance_img is not None:
            import numpy as np
            import nibabel as nib  # type: ignore

            var = np.asarray(variance_img.get_fdata())
            se = np.sqrt(np.clip(var, 0, None))
            se_img = nib.Nifti1Image(se, variance_img.affine, variance_img.header)
            se_vmax = float(np.percentile(se[np.isfinite(se)], 99)) if np.isfinite(se).any() else None
            if "slices" in plot_types:
                disp = plotting.plot_stat_map(
                    se_img,
                    bg_img=bg_img,
                    title=f"{title_prefix}Std. error".strip(),
                    display_mode="mosaic",
                    threshold=None,
                    colorbar=True,
                    vmax=se_vmax,
                    symmetric_cbar=False,
                    cmap="viridis",
                    annotate=False,
                )
                _add_image("Std. error (slices)", disp, "se_slices")
    except Exception as exc:
        logger.warning("Failed to generate standard error panels (%s)", exc)

    return ReportSpaceSection(space=space, images=tuple(images), tables=tuple(tables), summary=summary)


def write_fmri_report(
    *,
    contrast_dir: Path,
    subject: str,
    task: str,
    contrast_name: str,
    z_threshold: float,
    include_unthresholded: bool,
    sections: Sequence[ReportSpaceSection],
    report_filename: str = "report.html",
    extra_notes: Optional[Sequence[str]] = None,
    embed_images: bool = True,
    methods_payload: Optional[Dict[str, Any]] = None,
) -> Path:
    report_path = contrast_dir / report_filename
    html_text = build_fmri_report_html(
        report_path=report_path,
        subject=subject,
        task=task,
        contrast_name=contrast_name,
        z_threshold=z_threshold,
        include_unthresholded=include_unthresholded,
        sections=sections,
        extra_notes=extra_notes,
        embed_images=embed_images,
        methods_payload=methods_payload,
    )
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def run_fmri_plotting_and_report(
    *,
    contrast_dir: Path,
    subject: str,
    task: str,
    contrast_name: str,
    cfg: FmriPlottingConfig,
    run_meta: Optional[Dict[str, Any]] = None,
    native_stat_map_path: Optional[Path] = None,
    mni_stat_map_path: Optional[Path] = None,
    native_stat_img: Optional[Any] = None,
    mni_stat_img: Optional[Any] = None,
    native_effect_img: Optional[Any] = None,
    native_variance_img: Optional[Any] = None,
    mni_effect_img: Optional[Any] = None,
    mni_variance_img: Optional[Any] = None,
    native_bg_img_path: Optional[Path] = None,
    mni_bg_img_path: Optional[Path] = None,
    native_bg_img: Optional[Any] = None,
    mni_bg_img: Optional[Any] = None,
    native_mask_img_path: Optional[Path] = None,
    mni_mask_img_path: Optional[Path] = None,
    native_mask_img: Optional[Any] = None,
    mni_mask_img: Optional[Any] = None,
    signature_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Best-effort plotting + report for a contrast directory.

    Returns metadata about generated outputs.
    """
    cfg = cfg.normalized()
    if not cfg.enabled:
        return {"enabled": False}

    cfg.validate()

    sections: List[ReportSpaceSection] = []
    meta: Dict[str, Any] = {"enabled": True, "spaces": [], "formats": list(cfg.formats)}

    want_native = cfg.space in {"native", "both"}
    want_mni = cfg.space in {"mni", "both"}

    # Robust vmax across spaces (optional shared scaling)
    native_vmax = _robust_vmax_abs(native_stat_img, mask_img=native_mask_img) if native_stat_img is not None else None
    mni_vmax = _robust_vmax_abs(mni_stat_img, mask_img=mni_mask_img) if mni_stat_img is not None else None
    shared_vmax = None
    if cfg.vmax_mode == "shared_robust":
        vals = [v for v in [native_vmax, mni_vmax] if v is not None]
        shared_vmax = max(vals) if vals else None

    if want_native and (
        (native_stat_img is not None)
        or (native_stat_map_path is not None and native_stat_map_path.exists())
    ):
        vmax = None
        if cfg.vmax_mode == "manual":
            vmax = cfg.vmax_manual
        elif cfg.vmax_mode == "shared_robust":
            vmax = shared_vmax
        else:
            vmax = native_vmax
        sec = generate_fmri_space_section(
            space="native",
            stat_img_path=native_stat_map_path,
            stat_img=native_stat_img,
            out_base_dir=contrast_dir,
            formats=cfg.formats,
            z_threshold=cfg.z_threshold,
            include_unthresholded=cfg.include_unthresholded,
            plot_types=cfg.plot_types,
            bg_img_path=native_bg_img_path,
            bg_img=native_bg_img,
            mask_img_path=native_mask_img_path,
            mask_img=native_mask_img,
            cfg=cfg,
            vmax=vmax,
            effect_img=native_effect_img if cfg.include_effect_size else None,
            variance_img=native_variance_img if cfg.include_standard_error else None,
        )
        sections.append(sec)
        meta["spaces"].append("native")

    if want_mni and (
        (mni_stat_img is not None)
        or (mni_stat_map_path is not None and mni_stat_map_path.exists())
    ):
        vmax = None
        if cfg.vmax_mode == "manual":
            vmax = cfg.vmax_manual
        elif cfg.vmax_mode == "shared_robust":
            vmax = shared_vmax
        else:
            vmax = mni_vmax
        sec = generate_fmri_space_section(
            space="mni",
            stat_img_path=mni_stat_map_path,
            stat_img=mni_stat_img,
            out_base_dir=contrast_dir,
            formats=cfg.formats,
            z_threshold=cfg.z_threshold,
            include_unthresholded=cfg.include_unthresholded,
            plot_types=cfg.plot_types,
            bg_img_path=mni_bg_img_path,
            bg_img=mni_bg_img,
            mask_img_path=mni_mask_img_path,
            mask_img=mni_mask_img,
            cfg=cfg,
            vmax=vmax,
            effect_img=mni_effect_img if cfg.include_effect_size else None,
            variance_img=mni_variance_img if cfg.include_standard_error else None,
        )
        sections.append(sec)
        meta["spaces"].append("mni")

    # QC sections (not space-specific)
    qc_images: List[ReportImage] = []
    qc_tables: List[ReportTable] = []
    try:
        if cfg.include_motion_qc and isinstance(run_meta, dict):
            conf_paths = run_meta.get("included_confounds_paths") or []
            if isinstance(conf_paths, list) and conf_paths:
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt

                dfs = []
                run_breaks = [0]
                for p in conf_paths:
                    if not p:
                        continue
                    df = pd.read_csv(str(p), sep="\t")
                    dfs.append(df)
                    run_breaks.append(run_breaks[-1] + len(df))
                if dfs:
                    df_all = pd.concat(dfs, ignore_index=True)
                    x = np.arange(len(df_all))
                    fd = df_all.get("framewise_displacement")
                    dvars = df_all.get("dvars") if "dvars" in df_all.columns else df_all.get("std_dvars")

                    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, dpi=150)
                    if fd is not None:
                        axes[0].plot(x, fd.fillna(0).to_numpy(), color="#444444", linewidth=0.8)
                        axes[0].set_ylabel("FD (mm)")
                    if dvars is not None:
                        axes[1].plot(x, dvars.fillna(0).to_numpy(), color="#1F77B4", linewidth=0.8)
                        axes[1].set_ylabel("DVARS")
                    axes[1].set_xlabel("Frame (concatenated runs)")
                    for b in run_breaks[1:-1]:
                        for ax in axes:
                            ax.axvline(b, color="#999999", linewidth=0.7, alpha=0.6)
                    fig.suptitle("Motion QC (concatenated across runs)")
                    fig.tight_layout()

                    qc_dir = contrast_dir / "plots" / "qc"
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    for fmt in cfg.formats:
                        out_path = qc_dir / f"motion_qc.{fmt}"
                        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
                        qc_images.append(ReportImage(title="Motion QC (FD/DVARS)", path=out_path))
                    plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to generate motion QC (%s)", exc)

    try:
        if cfg.include_carpet_qc and isinstance(run_meta, dict):
            qc_images.extend(
                generate_carpet_qc_images(
                    contrast_dir=contrast_dir,
                    cfg=cfg,
                    run_meta=run_meta,
                    mask_img_path=native_mask_img_path,
                )
            )
    except Exception as exc:
        logger.warning("Failed to generate carpet QC (%s)", exc)

    try:
        if cfg.include_tsnr_qc and isinstance(run_meta, dict):
            qc_images.extend(
                generate_tsnr_qc_images(
                    contrast_dir=contrast_dir,
                    cfg=cfg,
                    run_meta=run_meta,
                    mask_img_path=native_mask_img_path,
                )
            )
    except Exception as exc:
        logger.warning("Failed to generate tSNR QC (%s)", exc)

    try:
        if cfg.include_design_qc:
            qc_dir = contrast_dir / "qc"
            if qc_dir.exists():
                import pandas as pd
                import numpy as np

                rows: List[Dict[str, Any]] = []
                for tsv in sorted(qc_dir.glob("*design_matrix.tsv")):
                    dm = pd.read_csv(tsv, sep="\t")
                    # Drop frame index if present
                    if "frame" in dm.columns:
                        dm = dm.drop(columns=["frame"])
                    cols = [c for c in dm.columns if c.lower() not in {"constant", "intercept"}]
                    X = dm[cols].to_numpy() if cols else dm.to_numpy()
                    # Correlation summary (best-effort)
                    max_corr = None
                    top_pair = ""
                    if X.shape[1] >= 2:
                        corr = np.corrcoef(X.T)
                        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                        np.fill_diagonal(corr, 0)
                        idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
                        max_corr = float(corr[idx])
                        try:
                            top_pair = f"{cols[idx[0]]} vs {cols[idx[1]]}"
                        except Exception:
                            top_pair = ""
                    # Condition number
                    cond = None
                    try:
                        u, s, vh = np.linalg.svd(X, full_matrices=False)
                        if s.size and s.min() > 0:
                            cond = float(s.max() / s.min())
                    except Exception:
                        cond = None
                    # Variance inflation factor (max over regressors)
                    max_vif = None
                    max_vif_regressor = ""
                    if X.shape[1] >= 2:
                        vifs = _vif_from_design(X)
                        finite = np.isfinite(vifs)
                        if np.any(finite):
                            idx_max = int(np.nanargmax(np.where(finite, vifs, -1)))
                            max_vif = float(vifs[idx_max])
                            max_vif_regressor = cols[idx_max] if idx_max < len(cols) else ""
                        elif np.any(~np.isnan(vifs)):
                            max_vif = np.inf
                            max_vif_regressor = ""
                    rows.append(
                        {
                            "design_matrix": tsv.name,
                            "n_regressors": int(X.shape[1]),
                            "max_abs_corr": float(abs(max_corr)) if max_corr is not None else None,
                            "top_corr_pair": top_pair,
                            "condition_number": cond,
                            "max_vif": max_vif,
                            "max_vif_regressor": max_vif_regressor,
                        }
                    )
                if rows:
                    df = pd.DataFrame(rows)
                    html_table = df.to_html(index=False, border=0, classes="")
                    tsv_out = contrast_dir / "plots" / "qc" / "design_qc_summary.tsv"
                    tsv_out.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        df.to_csv(tsv_out, sep="\t", index=False)
                    except Exception:
                        tsv_out = None
                    qc_tables.append(
                        ReportTable(
                            title="Design Matrix Sanity (per run)",
                            tsv_path=tsv_out,
                            html_table=html_table,
                            caption="High max_abs_corr, condition_number, or max_vif (variance inflation factor) indicate collinearity/instability.",
                        )
                    )
    except Exception as exc:
        logger.warning("Failed to generate design QC summary (%s)", exc)

    if qc_images or qc_tables:
        sections.append(ReportSpaceSection(space="QC", images=tuple(qc_images), tables=tuple(qc_tables), summary=None))

    # Signatures section (computed on the MNI effect-size map)
    try:
        sig_tables = generate_pain_signature_tables(
            contrast_dir=contrast_dir,
            cfg=cfg,
            mni_effect_img=mni_effect_img,
            mni_mask_img=mni_mask_img,
            signature_root=signature_root,
        )
        if sig_tables:
            sections.append(ReportSpaceSection(space="Signatures", images=(), tables=tuple(sig_tables), summary=None))
            meta["signatures"] = {"enabled": True, "root": str(signature_root) if signature_root else None}
    except Exception as exc:
        logger.warning("Failed to generate signatures section (%s)", exc)

    if cfg.html_report:
        methods_payload: Dict[str, Any] = {
            "subject": subject,
            "task": task,
            "contrast": contrast_name,
            "plotting_cfg": {
                k: v for k, v in cfg.__dict__.items()
            },
            "run_meta": run_meta,
            "signature_root": str(signature_root) if signature_root else None,
        }
        try:
            import nilearn  # type: ignore
            import nibabel  # type: ignore
            import numpy  # type: ignore

            methods_payload["versions"] = {
                "nilearn": getattr(nilearn, "__version__", None),
                "nibabel": getattr(nibabel, "__version__", None),
                "numpy": getattr(numpy, "__version__", None),
            }
        except Exception:
            pass

        report_path = write_fmri_report(
            contrast_dir=contrast_dir,
            subject=subject,
            task=task,
            contrast_name=contrast_name,
            z_threshold=cfg.z_threshold,
            include_unthresholded=cfg.include_unthresholded,
            sections=sections,
            embed_images=bool(cfg.embed_images),
            methods_payload=methods_payload,
        )
        meta["report_html"] = str(report_path)

    # Write provenance JSON for reproducibility (best-effort)
    try:
        prov = {
            "subject": subject,
            "task": task,
            "contrast": contrast_name,
            "plotting_cfg": {k: v for k, v in cfg.__dict__.items()},
            "spaces_rendered": meta.get("spaces", []),
            "report_html": meta.get("report_html"),
        }
        prov_path = contrast_dir / "plots" / "plot_provenance.json"
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
        meta["provenance_json"] = str(prov_path)
    except Exception:
        pass

    return meta
