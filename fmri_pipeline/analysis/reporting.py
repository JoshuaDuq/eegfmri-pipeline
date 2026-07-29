from __future__ import annotations

import html
import logging
import base64
import json
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
from fmri_pipeline.analysis.multivariate_signatures import (
    SignatureResult,
    compute_signature_expression,
    discover_signature_files,
)
from fmri_pipeline.analysis.report.figures import (
    carpet as carpet_figures,
    coverage as coverage_figures,
    distributions as distribution_figures,
    stat_maps as stat_map_figures,
    volumes as volume_figures,
)
from fmri_pipeline.analysis.report.manifest import (
    sample_masks_from_confounds as _sample_masks_from_confounds,
)
from fmri_pipeline.analysis.report.figures.design import (
    vif_from_design as _vif_from_design,
)
from fmri_pipeline.analysis.report.style import (
    MAGNITUDE_CMAP,
    plot_context,
    savefig_kwargs,
)

logger = logging.getLogger(__name__)

_SUPPORTED_REPORT_STAT_MAP_TYPES = frozenset({"z_score"})


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


def _discover_design_matrix_qc(
    contrast_dir: Path,
    run_meta: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Optional[Path], Optional[Path]]]:
    """
    Return list of (png, tsv) pairs for design matrix QC files, best-effort.
    """
    qc_dir = contrast_dir / "qc"
    if not qc_dir.exists():
        return []

    pngs: List[Path] = []
    tsvs: List[Path] = []
    if isinstance(run_meta, dict):
        raw_pngs = run_meta.get("design_matrix_png_paths")
        raw_tsvs = run_meta.get("design_matrix_tsv_paths")
        if isinstance(raw_pngs, list):
            pngs = [Path(str(p)) for p in raw_pngs if p]
            pngs = [p for p in pngs if p.exists()]
        if isinstance(raw_tsvs, list):
            tsvs = [Path(str(p)) for p in raw_tsvs if p]
            tsvs = [p for p in tsvs if p.exists()]

    if not pngs and not tsvs:
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


def _save_figure(
    figure: Any,
    *,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    title: str,
    caption: str = "",
) -> List[ReportImage]:
    """Write one figure to every requested format and return a single report entry.

    One entry, not one per format. ``formats`` says what to put on disk; the report
    embeds a figure once. Emitting one entry per format previously rendered every
    figure twice in a report configured for both PNG and SVG.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary: Optional[Path] = None
        # Saving happens inside the style context, not just drawing. svg.hashsalt,
        # savefig.dpi, and savefig.bbox are all read at save time, so a figure drawn
        # under the context but written outside it gets none of them -- and without
        # a fixed hashsalt the SVG element ids differ on every render.
        with plot_context():
            for fmt in formats:
                path = out_dir / f"{stem}.{fmt}"
                # Deterministic metadata: without it an SVG carries a timestamp and
                # the same figure differs on every render.
                figure.savefig(path, **savefig_kwargs(path))
                if primary is None:
                    primary = path
        if primary is None:
            return []
        return [ReportImage(title=title, path=primary, caption=caption)]
    finally:
        # Closed here rather than after a successful save: a figure leaked on the
        # error path grows without bound across a cohort.
        with suppress(Exception):
            plt.close(figure)


@contextmanager
def _panel(description: str) -> Iterator[None]:
    """Log and swallow one panel's failure.

    A panel that cannot be drawn is a gap in the report, not a reason to lose the
    rest of it. This is the policy the QC blocks already used; the space sections
    previously logged and then re-raised, so one bad panel cost the whole contrast.
    """
    try:
        yield
    except Exception as exc:
        logger.warning("Failed to generate %s (%s)", description, exc)


def _effect_units(run_meta: Optional[Dict[str, Any]]) -> str:
    """Name the units of a GLM effect size, or decline to.

    A contrast effect is in arbitrary BOLD units unless the model applied signal
    scaling. Printing "% signal change" on a map that is not in those units is worse
    than printing nothing, because it invites a quantitative reading the number
    cannot support.
    """
    scaling = (run_meta or {}).get("signal_scaling")
    return "% signal change" if scaling else "effect (arbitrary BOLD units)"


def generate_carpet_qc_images(
    *,
    contrast_dir: Path,
    cfg: FmriPlottingConfig,
    run_meta: Dict[str, Any],
    mask_img_path: Optional[Path] = None,
    max_voxels: int = 6000,
) -> List[ReportImage]:
    """Generate the carpet QC panel with motion traces on a shared time axis."""
    cfg = cfg.normalized()
    if not cfg.enabled or not cfg.include_carpet_qc:
        return []

    bold_paths = run_meta.get("included_bold_paths") if isinstance(run_meta, dict) else None
    if not isinstance(bold_paths, list) or not bold_paths:
        return []

    images: List[ReportImage] = []
    with _panel("carpet QC"):
        series, _imgs, _mask, labels = _load_run_series(bold_paths, mask_img_path)
        if not series:
            return []

        confound_paths = run_meta.get("included_confounds_paths") or []
        sample_masks: List[np.ndarray] = []
        if isinstance(confound_paths, list) and len(confound_paths) == len(series):
            with suppress(Exception):
                sample_masks = _sample_masks_from_confounds(confound_paths)

        standardised: List[np.ndarray] = []
        run_breaks = [0]
        for index, voxels in enumerate(series):
            mask = sample_masks[index] if index < len(sample_masks) else None
            if mask is not None and mask.size != voxels.shape[1]:
                mask = None
            standardised.append(
                carpet_figures.standardise_carpet(voxels, sample_mask=mask)
            )
            run_breaks.append(run_breaks[-1] + int(voxels.shape[1]))

        carpet = np.concatenate(standardised, axis=1)
        fd_values, dvars_values, dvars_label = _motion_traces(
            confound_paths, n_frames=carpet.shape[1]
        )

        codes, source = (None, "none")
        with suppress(Exception):
            reference = _imgs[0] if _imgs else None
            if reference is not None:
                from fmri_pipeline.analysis.report.assets import PlotAssets

                assets = run_meta.get("plot_assets")
                if isinstance(assets, PlotAssets):
                    volume_codes, source = carpet_figures.resolve_tissue_codes(
                        np.asanyarray(reference.dataobj).shape[:3],
                        assets=assets,
                        reference_img=reference,
                    )
                    if volume_codes is not None:
                        codes = volume_codes[
                            np.isfinite(np.mean(np.asanyarray(reference.dataobj), axis=3))
                        ]

        tr = run_meta.get("t_r") or run_meta.get("repetition_time") or 1.0
        figure = carpet_figures.carpet_figure(
            carpet,
            tissue_codes=codes,
            tissue_source=source,
            tr=float(tr),
            run_boundaries=run_breaks[1:-1],
            run_labels=labels,
            fd=fd_values,
            dvars=dvars_values,
            dvars_label=dvars_label,
            title="Carpet (as modelled)",
        )
        images.extend(_save_figure(
            figure,
            out_dir=contrast_dir / "plots" / "qc",
            stem="carpet_qc",
            formats=cfg.formats,
            title="QC: Carpet",
            caption=f"Voxel order: {source}. Motion shares the carpet's time axis.",
        ))
    return images


def _motion_traces(
    confound_paths: Sequence[Any],
    *,
    n_frames: int,
) -> Tuple[Optional["np.ndarray"], Optional["np.ndarray"], str]:
    """Concatenate FD and DVARS across runs, preserving undefined samples.

    ``framewise_displacement`` is undefined for the first frame of every run.
    Filling it with zero draws a dip to "no motion" at each run boundary, which is
    a fabricated measurement; NaN is left in place and Matplotlib gaps it.

    The DVARS label follows the column that was actually found, because ``dvars``
    and ``std_dvars`` live on different scales and mislabelling one as the other
    makes the axis unreadable.
    """
    if not confound_paths:
        return None, None, "DVARS"

    import pandas as pd

    fd_parts: List[np.ndarray] = []
    dvars_parts: List[np.ndarray] = []
    label = "DVARS"
    for path in confound_paths:
        if not path:
            continue
        try:
            frame = pd.read_csv(str(path), sep="\t")
        except Exception:
            return None, None, label
        fd_parts.append(
            frame["framewise_displacement"].to_numpy(dtype=float)
            if "framewise_displacement" in frame.columns
            else np.full(len(frame), np.nan)
        )
        if "dvars" in frame.columns:
            dvars_parts.append(frame["dvars"].to_numpy(dtype=float))
        elif "std_dvars" in frame.columns:
            dvars_parts.append(frame["std_dvars"].to_numpy(dtype=float))
            label = "std DVARS"
        else:
            dvars_parts.append(np.full(len(frame), np.nan))

    fd = np.concatenate(fd_parts) if fd_parts else None
    dvars = np.concatenate(dvars_parts) if dvars_parts else None
    if fd is not None and fd.size != n_frames:
        fd = None
    if dvars is not None and dvars.size != n_frames:
        dvars = None
    return fd, dvars, label


def generate_tsnr_qc_images(
    *,
    contrast_dir: Path,
    cfg: FmriPlottingConfig,
    run_meta: Dict[str, Any],
    mask_img_path: Optional[Path] = None,
) -> List[ReportImage]:
    """Generate tSNR QC panels: the mean map, per-run medians, and a histogram.

    Non-steady-state frames are excluded from the computation. They sit far above
    steady state before longitudinal magnetisation saturates, so including them
    inflates the temporal standard deviation and biases every tSNR value low.
    """
    cfg = cfg.normalized()
    if not cfg.enabled or not cfg.include_tsnr_qc:
        return []

    bold_paths = run_meta.get("included_bold_paths") if isinstance(run_meta, dict) else None
    if not isinstance(bold_paths, list) or not bold_paths:
        return []

    images: List[ReportImage] = []
    qc_dir = contrast_dir / "plots" / "qc"

    with _panel("tSNR QC"):
        nib = _maybe_import_nibabel()
        if nib is None:
            return []
        _series, imgs, mask_img, labels = _load_run_series(bold_paths, mask_img_path)
        if not imgs:
            return []

        confound_paths = run_meta.get("included_confounds_paths") or []
        sample_masks = None
        if isinstance(confound_paths, list) and len(confound_paths) == len(imgs):
            with suppress(Exception):
                candidate = _sample_masks_from_confounds(confound_paths)
                if all(
                    m.size == np.asanyarray(img.dataobj).shape[3]
                    for m, img in zip(candidate, imgs)
                ):
                    sample_masks = candidate

        result = volume_figures.compute_tsnr(
            imgs, mask_img=mask_img, sample_masks=sample_masks
        )

        with suppress(Exception):
            qc_dir.mkdir(parents=True, exist_ok=True)
            nib.save(result.mean_img, str(qc_dir / "tsnr_mean.nii.gz"))

        figure = volume_figures.tsnr_volume(
            result, title=f"tSNR (mean of {len(imgs)} run(s), as modelled)"
        )
        images.extend(_save_figure(
            figure, out_dir=qc_dir, stem="tsnr_map", formats=cfg.formats,
            title="QC: tSNR map",
        ))

        if len(result.per_run_median) > 1:
            figure = volume_figures.per_run_tsnr_figure(
                result, run_labels=labels, title="tSNR by run"
            )
            images.extend(_save_figure(
                figure, out_dir=qc_dir, stem="tsnr_by_run", formats=cfg.formats,
                title="QC: tSNR by run",
                caption="Shown per run because averaging maps hides a single bad run.",
            ))

        data = np.asarray(result.mean_img.get_fdata())
        values = data[np.isfinite(data) & (data > 0)]
        if values.size:
            figure = distribution_figures.magnitude_histogram(
                values, xlabel="tSNR", title="tSNR distribution (masked voxels)"
            )
            images.extend(_save_figure(
                figure, out_dir=qc_dir, stem="tsnr_hist", formats=cfg.formats,
                title="QC: tSNR histogram",
            ))
    return images


def generate_signature_tables(
    *,
    contrast_dir: Path,
    cfg: FmriPlottingConfig,
    mni_effect_img: Optional[Any],
    mni_mask_img: Optional[Any],
    signature_root: Optional[Path],
    signature_specs: Optional[List[Any]] = None,
) -> List[ReportTable]:
    """
    Compute multivariate signature expression on the unthresholded MNI effect-size map.

    Returns report tables for configured signatures.
    Signatures are read from ``signature_specs`` (config-driven list of {name, path} dicts).
    """
    cfg = cfg.normalized()
    if not cfg.enabled or not bool(getattr(cfg, "include_signatures", True)):
        return []
    if mni_effect_img is None:
        raise ValueError(
            "Signature report generation requires an MNI effect-size map."
        )
    if signature_root is None or not signature_specs:
        raise ValueError(
            "Signature report generation requires signature_root and signature_specs."
        )

    sig_files = discover_signature_files(signature_root, signature_specs)
    if not sig_files:
        raise ValueError("No configured signature weight maps were discovered.")

    results: List[SignatureResult] = compute_signature_expression(
        stat_or_effect_img=mni_effect_img,
        signature_root=signature_root,
        signature_specs=signature_specs,
        mask_img=mni_mask_img,
        signatures=sorted(sig_files.keys()),
    )

    if not results:
        raise ValueError("Signature expression returned no results.")

    qc_dir = contrast_dir / "plots" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = qc_dir / "signature_expression.tsv"
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
            title="Multivariate Signature Expression (MNI effect-size map)",
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
    run_meta: Optional[Dict[str, Any]] = None,
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

    qc_pairs = _discover_design_matrix_qc(base_dir, run_meta=run_meta)

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


def _normalize_report_stat_map_type(stat_map_type: str) -> str:
    normalized = str(stat_map_type or "").strip().lower().replace("-", "_")
    if normalized not in _SUPPORTED_REPORT_STAT_MAP_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_REPORT_STAT_MAP_TYPES))
        raise ValueError(
            "fMRI plotting/reporting requires z-score statistic maps because thresholded "
            "report panels are calibrated only for z-statistics. "
            f"Got stat_map_type={stat_map_type!r}; supported values: {supported}."
        )
    return normalized


def _build_mean_bold_background_from_run_meta(run_meta: Optional[Dict[str, Any]]) -> Optional[Any]:
    """
    Build a 3D mean-BOLD background image from the first included run.

    This keeps stat overlays anatomically interpretable even when boldref
    discovery misses derivative naming variants.
    """
    if not isinstance(run_meta, dict):
        return None

    bold_paths = run_meta.get("included_bold_paths")
    if not isinstance(bold_paths, list) or not bold_paths:
        return None

    nib = _maybe_import_nibabel()
    if nib is None:
        return None

    for raw_path in bold_paths:
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.exists():
            continue
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        if data.ndim == 4:
            mean_data = np.mean(data, axis=3)
            return nib.Nifti1Image(mean_data, img.affine, img.header)
        if data.ndim == 3:
            return img
    return None


def _load_mni_template_background() -> Optional[Any]:
    """Load Nilearn's canonical MNI template for anatomical background plotting."""
    try:
        from nilearn.datasets import load_mni152_template  # type: ignore

        return load_mni152_template()
    except Exception:
        return None


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
        from nilearn.glm import threshold_stats_img  # type: ignore

        thr_img, thr = threshold_stats_img(
            stat_img,
            alpha=float(cfg.fdr_q),
            height_control="fdr",
            cluster_threshold=int(cfg.cluster_min_voxels) if cfg.cluster_min_voxels > 0 else 0,
            two_sided=bool(cfg.two_sided),
        )
        return thr_img, float(thr), f"fdr q={cfg.fdr_q:.3f}"

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
        raise RuntimeError(
            "fMRI plotting requires nilearn plotting and nibabel."
        )

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

    two_sided = bool(cfg_obj.two_sided)
    radiological = bool(getattr(cfg_obj, "radiological", False))

    # Dual-coded panel leads the results: hue carries the effect, opacity carries
    # the evidence, so sub-threshold structure fades instead of vanishing. Needs
    # both an effect map and a statistic map, so it is skipped when only one exists.
    if "slices" in plot_types and effect_img is not None and thr_val is not None:
        with _panel("dual-coded stat map"):
            figure = stat_map_figures.dual_coded_mosaic(
                effect_img,
                stat_img=stat_img,
                bg_img=bg_img,
                threshold=float(thr_val),
                two_sided=two_sided,
                radiological=radiological,
                cbar_label=_effect_units(None),
                title=f"{title_prefix}Effect, opacity-coded by evidence".strip(),
            )
            images.extend(_save_figure(
                figure, out_dir=out_dir, stem="dual_coded", formats=formats,
                title="Effect map · dual-coded",
                caption=(
                    "Colour is effect magnitude; opacity is statistical evidence, "
                    f"ramped from |z| {0.5 * float(thr_val):.2f} to "
                    f"{float(thr_val):.2f}. No voxels are hidden."
                ),
            ))

    if "slices" in plot_types:
        if include_unthresholded:
            with _panel("unthresholded stat-map slices"):
                figure = stat_map_figures.stat_map_mosaic(
                    stat_img, bg_img=bg_img, threshold=None, vmax=z_vmax,
                    two_sided=two_sided, radiological=radiological,
                    title=f"{title_prefix}Z map (unthresholded)".strip(),
                )
                images.extend(_save_figure(
                    figure, out_dir=out_dir, stem="stat_slices_unthresholded",
                    formats=formats, title="Stat map (slices) · unthresholded",
                ))
        if thr_label != "none":
            with _panel("thresholded stat-map slices"):
                figure = stat_map_figures.stat_map_mosaic(
                    stat_img if thr_img is None else thr_img, bg_img=bg_img,
                    threshold=float(thr_val) if thr_val is not None else None,
                    two_sided=two_sided, radiological=radiological,
                    title=f"{title_prefix}Z map (thresholded)".strip(),
                )
                images.extend(_save_figure(
                    figure, out_dir=out_dir, stem="stat_slices_thresholded",
                    formats=formats, title="Stat map (slices) · thresholded",
                    caption=thr_label,
                ))

    # Only the thresholded glass brain is drawn. An unthresholded projection is a
    # saturated blob at any threshold setting and carries no information.
    if "glass" in plot_types and thr_label != "none":
        with _panel("thresholded glass brain"):
            figure = stat_map_figures.glass_brain(
                stat_img if thr_img is None else thr_img,
                threshold=float(thr_val) if thr_val is not None else None,
                two_sided=two_sided, radiological=radiological,
                title=f"{title_prefix}Glass brain (thresholded)".strip(),
            )
            images.extend(_save_figure(
                figure, out_dir=out_dir, stem="glass_thresholded", formats=formats,
                title="Glass brain · thresholded", caption=thr_label,
            ))

    if "hist" in plot_types:
        with _panel("z histogram"):
            data = np.asarray(stat_img.get_fdata())
            if mask_img is not None:
                m = np.asarray(mask_img.get_fdata()).astype(bool)
                data = data[m] if m.shape == data.shape else data[data != 0]
            figure = distribution_figures.z_histogram(
                data,
                threshold=thr_val if thr_label != "none" else None,
                title="Z-statistic distribution",
            )
            images.extend(_save_figure(
                figure, out_dir=out_dir, stem="z_hist", formats=formats,
                title="Z histogram",
            ))

    if mask_img is not None:
        with _panel("coverage"):
            figure = coverage_figures.coverage_figure(
                mask_img, bg_img=bg_img,
                title=f"{title_prefix}Analysis mask".strip(),
            )
            images.extend(_save_figure(
                figure, out_dir=out_dir, stem="coverage", formats=formats,
                title="Coverage (analysis mask)",
                caption="Voxels outside this mask were not tested.",
            ))

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
            caption_parts = [
                "two-sided" if cfg_obj.two_sided else "one-sided",
                f"height threshold: {thr_label}",
            ]
            if cfg_obj.cluster_min_voxels > 0:
                # Eklund, Nichols & Knutsson (2016) measured familywise false-positive
                # rates up to 70% for parametric cluster-extent inference. This
                # pipeline performs no cluster-level correction at all, so the caption
                # must not let extent read as inferential.
                caption_parts.append(
                    f"clusters smaller than {cfg_obj.cluster_min_voxels} voxels removed "
                    "for display; this is an extent filter, not familywise-error-"
                    "corrected cluster-level inference"
                )
            tables[-1] = ReportTable(
                title=tables[-1].title,
                tsv_path=tables[-1].tsv_path,
                html_table=tables[-1].html_table,
                caption="; ".join(caption_parts),
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

    # Effect size and standard error are diagnostics, kept but drawn through the
    # tested figure layer. The effect map uses the same diverging colormap as the z
    # map: same kind of data, so a second colormap would imply a difference that is
    # not there. Standard error is an unsigned magnitude and takes the sequential ramp.
    if bool(getattr(cfg_obj, "include_effect_size", True)) and effect_img is not None:
        if "slices" in plot_types:
            with _panel("effect size slices"):
                figure = stat_map_figures.stat_map_mosaic(
                    effect_img, bg_img=bg_img, threshold=None,
                    two_sided=two_sided, radiological=radiological,
                    cbar_label=_effect_units(None),
                    title=f"{title_prefix}Effect size".strip(),
                )
                images.extend(_save_figure(
                    figure, out_dir=out_dir, stem="effect_slices", formats=formats,
                    title="Effect size (slices)",
                ))
        if "glass" in plot_types:
            with _panel("effect size glass brain"):
                figure = stat_map_figures.glass_brain(
                    effect_img, threshold=None,
                    two_sided=two_sided, radiological=radiological,
                    cbar_label=_effect_units(None),
                    title=f"{title_prefix}Effect size (glass)".strip(),
                )
                images.extend(_save_figure(
                    figure, out_dir=out_dir, stem="effect_glass", formats=formats,
                    title="Effect size (glass)",
                ))

    if (
        bool(getattr(cfg_obj, "include_standard_error", True))
        and variance_img is not None
        and "slices" in plot_types
    ):
        with _panel("standard error slices"):
            import nibabel as nib_se  # type: ignore

            var = np.asarray(variance_img.get_fdata())
            se = np.sqrt(np.clip(var, 0, None))
            se_img = nib_se.Nifti1Image(se, variance_img.affine, variance_img.header)
            figure = stat_map_figures.stat_map_mosaic(
                se_img, bg_img=bg_img, threshold=None,
                two_sided=True, radiological=radiological,
                cmap=MAGNITUDE_CMAP, cbar_label="standard error",
                title=f"{title_prefix}Standard error".strip(),
            )
            images.extend(_save_figure(
                figure, out_dir=out_dir, stem="se_slices", formats=formats,
                title="Std. error (slices)",
                caption="Where the model is least certain; distinguishes a true null "
                        "from dropout-driven absence of effect.",
            ))

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
    run_meta: Optional[Dict[str, Any]] = None,
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
        run_meta=run_meta,
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
    stat_map_type: str,
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
    signature_specs: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Generate plotting outputs and report artifacts for a contrast directory.

    Returns metadata about generated outputs.
    """
    cfg = cfg.normalized()
    if not cfg.enabled:
        return {"enabled": False}

    cfg.validate()
    normalized_stat_map_type = _normalize_report_stat_map_type(stat_map_type)

    sections: List[ReportSpaceSection] = []
    meta: Dict[str, Any] = {
        "enabled": True,
        "spaces": [],
        "formats": list(cfg.formats),
        "stat_map_type": normalized_stat_map_type,
    }

    want_native = cfg.space in {"native", "both"}
    want_mni = cfg.space in {"mni", "both"}

    # Robust vmax across spaces (optional shared scaling)
    native_vmax = _robust_vmax_abs(native_stat_img, mask_img=native_mask_img) if native_stat_img is not None else None
    mni_vmax = _robust_vmax_abs(mni_stat_img, mask_img=mni_mask_img) if mni_stat_img is not None else None
    shared_vmax = None
    if cfg.vmax_mode == "shared_robust":
        vals = [v for v in [native_vmax, mni_vmax] if v is not None]
        shared_vmax = max(vals) if vals else None

    if want_native:
        have_native_bg_path = native_bg_img_path is not None and native_bg_img_path.exists()
        if native_bg_img is None and not have_native_bg_path:
            native_bg_img = _build_mean_bold_background_from_run_meta(run_meta)
    if want_mni:
        have_mni_bg_path = mni_bg_img_path is not None and mni_bg_img_path.exists()
        if mni_bg_img is None and not have_mni_bg_path:
            mni_bg_img = _load_mni_template_background()

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
    # Motion no longer has a figure of its own: FD and DVARS are drawn on the
    # carpet's time axis, which is the only arrangement in which either is
    # diagnostic. A spike beside a carpet band is evidence; a spike on a separate
    # axis with its own x-scale is not.


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

                design_tsv_paths: List[Path] = []
                if isinstance(run_meta, dict):
                    raw_tsv_paths = run_meta.get("design_matrix_tsv_paths")
                    if isinstance(raw_tsv_paths, list):
                        design_tsv_paths = [Path(str(p)) for p in raw_tsv_paths if p]
                        design_tsv_paths = [p for p in design_tsv_paths if p.exists()]
                if not design_tsv_paths:
                    design_tsv_paths = sorted(qc_dir.glob("*design_matrix.tsv"))

                rows: List[Dict[str, Any]] = []
                for tsv in design_tsv_paths:
                    dm = pd.read_csv(tsv, sep="\t", encoding="utf-8")
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
                        _u, s, _vh = np.linalg.svd(X, full_matrices=False)
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
                        df.to_csv(tsv_out, sep="\t", index=False, encoding="utf-8")
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
        sig_tables = generate_signature_tables(
            contrast_dir=contrast_dir,
            cfg=cfg,
            mni_effect_img=mni_effect_img,
            mni_mask_img=mni_mask_img,
            signature_root=signature_root,
            signature_specs=signature_specs,
        )
        if sig_tables:
            sections.append(ReportSpaceSection(space="Signatures", images=(), tables=tuple(sig_tables), summary=None))
            meta["signatures"] = {"enabled": True, "root": str(signature_root) if signature_root else None}
    except Exception as exc:
        logger.warning("Failed to generate signatures section (%s)", exc)
        raise

    if cfg.html_report:
        methods_payload: Dict[str, Any] = {
            "subject": subject,
            "task": task,
            "contrast": contrast_name,
            "plotting_cfg": {
                k: v for k, v in cfg.__dict__.items()
            },
            "stat_map_type": normalized_stat_map_type,
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
        except Exception as exc:
            logger.debug("Failed to collect package version metadata for report: %s", exc)

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
            run_meta=run_meta if isinstance(run_meta, dict) else None,
        )
        meta["report_html"] = str(report_path)

    prov = {
        "subject": subject,
        "task": task,
        "contrast": contrast_name,
        "stat_map_type": normalized_stat_map_type,
        "plotting_cfg": {k: v for k, v in cfg.__dict__.items()},
        "spaces_rendered": meta.get("spaces", []),
        "report_html": meta.get("report_html"),
    }
    prov_path = contrast_dir / "plots" / "plot_provenance.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    meta["provenance_json"] = str(prov_path)

    return meta
