from __future__ import annotations

import csv
import inspect
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fmri_pipeline.analysis.pain_signatures import compute_pain_signature_expression
from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm
from fmri_pipeline.analysis.confounds_selection import select_fmriprep_confounds_columns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialSignatureExtractionConfig:
    # Data selection
    input_source: str  # "fmriprep" | "bids_raw"
    fmriprep_space: str  # e.g. "MNI152NLin2009cAsym" or "T1w"
    require_fmriprep: bool
    runs: Optional[List[int]]
    task: str

    # Trial selection
    name: str
    condition_a_column: str
    condition_a_value: str
    condition_b_column: str
    condition_b_value: str

    # GLM
    hrf_model: str
    drift_model: Optional[str]
    high_pass_hz: float
    low_pass_hz: Optional[float]
    smoothing_fwhm: Optional[float]
    confounds_strategy: str

    # Signature extraction
    method: str  # "beta-series" | "lss"
    include_other_events: bool = True
    lss_other_regressors: str = "per_condition"  # "per_condition" | "all"
    max_trials_per_run: Optional[int] = None
    fixed_effects_weighting: str = "variance"  # "variance" | "mean"
    signatures: Optional[Tuple[str, ...]] = None  # default: all discovered
    roi_atlas: Optional[str] = None  # atlas label image (MNI) path or alias
    roi_labels: Optional[str] = None  # labels table path (TSV/CSV) mapping label -> name
    roi_names: Optional[Tuple[str, ...]] = None  # ROI names to extract (or ("all",))

    # Outputs
    write_trial_betas: bool = False
    write_trial_variances: bool = False
    write_condition_betas: bool = True

    def normalized(self) -> "TrialSignatureExtractionConfig":
        input_source = (self.input_source or "fmriprep").strip().lower()
        if input_source not in {"fmriprep", "bids_raw"}:
            input_source = "fmriprep"

        drift_model = (self.drift_model or "").strip().lower() or None
        if drift_model == "none":
            drift_model = None

        low_pass_hz = self.low_pass_hz
        try:
            if low_pass_hz is not None and float(low_pass_hz) <= 0:
                low_pass_hz = None
        except Exception:
            low_pass_hz = None

        smoothing_fwhm = normalize_smoothing_fwhm(self.smoothing_fwhm)

        method = (self.method or "beta-series").strip().lower()
        if method not in {"beta-series", "lss"}:
            method = "beta-series"

        lss_other = (self.lss_other_regressors or "per_condition").strip().lower().replace("-", "_")
        if lss_other not in {"per_condition", "all"}:
            lss_other = "per_condition"

        weight = (self.fixed_effects_weighting or "variance").strip().lower()
        if weight not in {"variance", "mean"}:
            weight = "variance"

        roi_atlas = (self.roi_atlas or "").strip() or None
        roi_labels = (self.roi_labels or "").strip() or None
        roi_names = None
        if self.roi_names:
            # Accept comma/semicolon-separated values and normalize special "all".
            raw: List[str] = []
            for item in self.roi_names:
                if item is None:
                    continue
                s = str(item).replace(";", ",").strip()
                if not s:
                    continue
                raw.extend([p.strip() for p in s.split(",") if p.strip()])
            if any(s.strip().lower() in {"all", "*", "@all"} for s in raw):
                roi_names = ("all",)
            else:
                dedup: List[str] = []
                seen = set()
                for s in raw:
                    key = s.strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    dedup.append(s.strip())
                roi_names = tuple(dedup) if dedup else None

        return TrialSignatureExtractionConfig(
            **{
                **asdict(self),
                "input_source": input_source,
                "drift_model": drift_model,
                "low_pass_hz": low_pass_hz,
                "smoothing_fwhm": smoothing_fwhm,
                "method": method,
                "lss_other_regressors": lss_other,
                "fixed_effects_weighting": weight,
                "roi_atlas": roi_atlas,
                "roi_labels": roi_labels,
                "roi_names": roi_names,
            }
        )


@dataclass(frozen=True)
class TrialInfo:
    run: int
    run_label: str
    trial_index: int
    condition: str  # "A" | "B"
    regressor: str
    onset: float
    duration: float
    original_trial_type: str
    source_events_path: Path
    source_row: int
    extra: Dict[str, str]


def _safe_slug(value: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "item"


def _resolve_path_with_search(
    value: str,
    *,
    search_roots: Sequence[Path],
    suffixes: Sequence[str] = (".nii.gz", ".nii", ".tsv", ".csv", ".txt"),
) -> Path:
    """
    Resolve a file path from an explicit path or by searching roots.

    - If `value` exists as-is, return it.
    - Else try `<root>/<value>` and `<root>/<value><suffix>` for common suffixes.
    """
    p = Path(str(value)).expanduser()
    if p.exists():
        return p
    for root in search_roots:
        root = Path(root)
        if not root:
            continue
        cand = root / str(value)
        if cand.exists():
            return cand
        for suf in suffixes:
            cand2 = root / f"{value}{suf}"
            if cand2.exists():
                return cand2
    roots_str = ", ".join(str(r) for r in search_roots[:4])
    raise FileNotFoundError(
        f"Could not resolve path: {value} (searched: {roots_str}). "
        "Run: python scripts/fetch_templateflow_atlas.py --schaefer 100 7 --resolution 2"
    )


def _infer_atlas_labels_sidecar(atlas_path: Path) -> Optional[Path]:
    """
    Best-effort inference for an atlas labels sidecar file.
    Tries: <atlas_stem>{,.tsv|.csv|.txt}, <atlas_stem>_labels.*, and (TemplateFlow)
    <atlas_stem> with _res-XX_ stripped + .tsv (labels often omit resolution in the name).
    """
    import re

    p = Path(atlas_path)
    stem = p.name
    if stem.endswith(".nii.gz"):
        base = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        base = stem[: -len(".nii")]
    else:
        base = p.stem

    for ext in (".tsv", ".csv", ".txt"):
        cand = p.with_name(base + ext)
        if cand.exists():
            return cand
        cand = p.with_name(base + "_labels" + ext)
        if cand.exists():
            return cand
    # TemplateFlow: labels file often has same stem but without _res-XX_
    base_no_res = re.sub(r"_res-\d+_", "_", base)
    if base_no_res != base:
        for ext in (".tsv", ".csv", ".txt"):
            cand = p.with_name(base_no_res + ext)
            if cand.exists():
                return cand
    return None


def _read_atlas_labels_table(labels_path: Path) -> Dict[int, str]:
    """
    Read a labels table mapping integer label -> ROI name.

    Supports TSV/CSV with either:
      - header columns like (label|index|id) and (name|roi|region)
      - two-column rows: <label> <name>
    """
    import csv

    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Atlas labels file does not exist: {path}")

    # Heuristic delimiter
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    text = path.read_text(errors="ignore").splitlines()
    if not text:
        raise ValueError(f"Empty atlas labels file: {path}")

    # Fall back to whitespace splitting for .txt if needed
    sample = text[0]
    if path.suffix.lower() == ".txt" and ("\t" not in sample and "," not in sample):
        # We'll treat it as a 2-column whitespace table
        out: Dict[int, str] = {}
        for line in text:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(float(parts[0]))
            except Exception:
                continue
            name = " ".join(parts[1:]).strip()
            if name:
                out[idx] = name
        if not out:
            raise ValueError(f"Could not parse atlas labels from: {path}")
        return out

    # CSV/TSV parsing
    out: Dict[int, str] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"Empty atlas labels file: {path}")

        # Detect header if first field is not numeric
        def _is_num(v: str) -> bool:
            try:
                float(v)
                return True
            except Exception:
                return False

        header = None
        rows_iter = reader
        if first and any(not _is_num(v) for v in first[:1]):
            header = [str(c).strip().lower() for c in first]
        else:
            # No header; treat as row
            rows_iter = iter([first] + list(reader))

        if header is not None:
            # Re-read with DictReader for easier mapping
            f.seek(0)
            d = csv.DictReader(f, delimiter=delim)
            fieldnames = [str(c).strip().lower() for c in (d.fieldnames or [])]
            idx_key = next((k for k in fieldnames if k in {"label", "index", "id", "roi_id"}), None)
            name_key = next((k for k in fieldnames if k in {"name", "roi", "region", "label_name"}), None)
            if idx_key is None or name_key is None:
                raise ValueError(
                    f"Atlas labels file must have columns like (label/index) and (name/roi/region): {path}"
                )
            for row in d:
                try:
                    idx = int(float(str(row.get(idx_key, "")).strip()))
                except Exception:
                    continue
                name = str(row.get(name_key, "")).strip()
                if name:
                    out[idx] = name
            if not out:
                raise ValueError(f"No valid labels parsed from: {path}")
            return out

        # Two-column table parsing
        for row in rows_iter:
            if not row:
                continue
            if len(row) < 2:
                continue
            try:
                idx = int(float(str(row[0]).strip()))
            except Exception:
                continue
            name = str(row[1]).strip()
            if name:
                out[idx] = name
        if not out:
            raise ValueError(f"Could not parse atlas labels from: {path}")
    return out


def _select_roi_labels(labels: Dict[int, str], roi_names: Tuple[str, ...]) -> List[Tuple[int, str]]:
    if not labels:
        raise ValueError("No atlas labels available.")
    if len(roi_names) == 1 and str(roi_names[0]).strip().lower() == "all":
        return sorted(labels.items(), key=lambda t: int(t[0]))

    # Build reverse lookup by lowercased label name
    name_to_label: Dict[str, int] = {}
    for idx, name in labels.items():
        key = str(name).strip().lower()
        if key and key not in name_to_label:
            name_to_label[key] = int(idx)

    selected: List[Tuple[int, str]] = []
    missing: List[str] = []
    for req in roi_names:
        q = str(req).strip()
        if not q:
            continue
        key = q.lower()
        if key in name_to_label:
            idx = name_to_label[key]
            selected.append((idx, labels[idx]))
            continue

        # Try unique substring match
        matches = [(idx, name) for idx, name in labels.items() if key in str(name).strip().lower()]
        if len(matches) == 1:
            selected.append((int(matches[0][0]), str(matches[0][1])))
            continue
        missing.append(q)

    if missing:
        examples = ", ".join([str(n) for _, n in list(sorted(labels.items()))[:15]])
        raise ValueError(
            "Unknown/ambiguous ROI name(s): "
            + ", ".join(missing)
            + f". Examples of available names: {examples} ..."
        )

    # Deduplicate by label index
    out: List[Tuple[int, str]] = []
    seen = set()
    for idx, name in selected:
        if int(idx) in seen:
            continue
        seen.add(int(idx))
        out.append((int(idx), str(name)))
    return out


def _build_roi_masks_from_atlas(
    *,
    atlas_path: Path,
    labels: List[Tuple[int, str]],
    target_img: Any,
) -> Dict[str, Any]:
    """
    Return mapping ROI name -> NIfTI mask aligned to `target_img` grid.
    """
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore

    try:
        from nilearn import image as nilearn_image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ROI masking requires nilearn to resample the atlas to the target image grid.") from exc

    atlas_img = nib.load(str(atlas_path))
    atlas_res = nilearn_image.resample_to_img(atlas_img, target_img, interpolation="nearest")
    data = atlas_res.get_fdata()
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError(f"Atlas has no finite voxels after resampling: {atlas_path}")
    # Ensure integer-like labels
    vals = data[finite]
    if not np.all(np.isclose(vals, np.round(vals), atol=1e-3)):
        raise ValueError(f"Atlas does not look like an integer label image: {atlas_path}")
    lab = np.round(data).astype(int)

    masks: Dict[str, Any] = {}
    for idx, name in labels:
        idx = int(idx)
        if idx == 0:
            continue
        m = lab == idx
        if int(np.sum(m)) == 0:
            continue
        masks[str(name)] = nib.Nifti1Image(m.astype(np.uint8), atlas_res.affine, atlas_res.header)
    if not masks:
        raise ValueError("All requested ROI masks were empty after resampling to the target image grid.")
    return masks


def _mask_hash_and_count(mask_img: Any) -> Tuple[str, int]:
    import hashlib

    import numpy as np  # type: ignore

    data = np.asanyarray(mask_img.get_fdata())
    mask = np.isfinite(data) & (data > 0)
    n_vox = int(np.sum(mask))

    h = hashlib.sha256()
    h.update(mask.astype(np.uint8).tobytes())
    h.update(np.asanyarray(mask_img.affine, dtype=np.float64).tobytes())
    h.update(np.asanyarray(mask.shape, dtype=np.int64).tobytes())
    return h.hexdigest(), n_vox


def _resample_mask_to_target(mask_img: Any, target_img: Any) -> Any:
    import numpy as np  # type: ignore

    try:
        from nilearn import image as nilearn_image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ROI masking requires nilearn to resample masks to the target image grid.") from exc

    # Best-effort: if already aligned, avoid resampling. (nilearn will still handle it if needed.)
    try:
        same_shape = tuple(getattr(mask_img, "shape", ())) == tuple(getattr(target_img, "shape", ()))
        same_affine = np.allclose(np.asanyarray(mask_img.affine), np.asanyarray(target_img.affine))
        if same_shape and same_affine:
            return mask_img
    except Exception:
        pass
    return nilearn_image.resample_to_img(mask_img, target_img, interpolation="nearest")


def _intersect_masks_to_target(*, roi_mask_img: Any, brain_mask_img: Any, target_img: Any) -> Any:
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore

    roi = _resample_mask_to_target(roi_mask_img, target_img)
    brain = _resample_mask_to_target(brain_mask_img, target_img)

    r = np.asanyarray(roi.get_fdata())
    b = np.asanyarray(brain.get_fdata())
    m = np.isfinite(r) & np.isfinite(b) & (r > 0) & (b > 0)
    return nib.Nifti1Image(m.astype(np.uint8), roi.affine, roi.header)


def _union_masks_to_target(mask_imgs: Sequence[Any], target_img: Any) -> Any:
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore

    if not mask_imgs:
        raise ValueError("No masks provided.")
    resampled = [_resample_mask_to_target(m, target_img) for m in mask_imgs]
    datas = [np.asanyarray(m.get_fdata()) for m in resampled]
    m0 = datas[0]
    union = np.isfinite(m0) & (m0 > 0)
    for d in datas[1:]:
        union = union | (np.isfinite(d) & (d > 0))
    ref = resampled[0]
    return nib.Nifti1Image(union.astype(np.uint8), ref.affine, ref.header)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _get_tr_from_bold(bold_path: Path) -> float:
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    if sidecar.exists():
        meta = _read_json(sidecar)
        if "RepetitionTime" in meta:
            return float(meta["RepetitionTime"])

    import nibabel as nib  # type: ignore

    img = nib.load(str(bold_path))
    zooms = img.header.get_zooms()
    if len(zooms) >= 4:
        return float(zooms[3])
    raise ValueError(f"Could not determine TR for {bold_path}")


def _discover_fmriprep_preproc_bold(
    *,
    bids_derivatives: Path,
    sub_label: str,
    task: str,
    run_num: int,
    space: Optional[str],
) -> Optional[Path]:
    search_dirs = [
        bids_derivatives / "preprocessed" / "fmri" / sub_label / "func",
        bids_derivatives / "preprocessed" / "fmri" / "fmriprep" / sub_label / "func",
        bids_derivatives / "fmriprep" / sub_label / "func",
    ]

    run_tokens = [f"run-{run_num:02d}", f"run-{run_num}"]
    patterns: List[str] = []
    for run_tok in run_tokens:
        if space:
            patterns.append(f"{sub_label}_task-{task}_{run_tok}_space-{space}_desc-preproc_bold.nii.gz")
        patterns.append(f"{sub_label}_task-{task}_{run_tok}_desc-preproc_bold.nii.gz")

    for func_dir in search_dirs:
        if not func_dir.exists():
            continue
        for pat in patterns:
            p = func_dir / pat
            if p.exists():
                return p
    return None


def _discover_confounds(
    *,
    bids_derivatives: Path,
    sub_label: str,
    task: str,
    run_num: int,
) -> Optional[Path]:
    search_dirs = [
        bids_derivatives / "preprocessed" / "fmri" / sub_label / "func",
        bids_derivatives / "preprocessed" / "fmri" / "fmriprep" / sub_label / "func",
        bids_derivatives / "fmriprep" / sub_label / "func",
        bids_derivatives / sub_label / "func",
    ]
    patterns = [
        f"{sub_label}_task-{task}_run-{run_num}_desc-confounds_timeseries.tsv",
        f"{sub_label}_task-{task}_run-{run_num:02d}_desc-confounds_timeseries.tsv",
        f"{sub_label}_task-{task}_run-{run_num}_desc-confounds_regressors.tsv",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            p = d / pat
            if p.exists():
                return p
    return None


def _discover_brain_mask_for_bold(bold_path: Path) -> Optional[Path]:
    name = bold_path.name
    suffix = "_desc-preproc_bold.nii.gz"
    if name.endswith(suffix):
        candidate = bold_path.with_name(name.replace(suffix, "_desc-brain_mask.nii.gz"))
        if candidate.exists():
            return candidate
    return None


def _discover_runs(
    *,
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    subject: str,
    task: str,
    runs: Optional[List[int]],
    input_source: str,
    fmriprep_space: Optional[str],
) -> List[Tuple[int, Path, Path, Optional[Path]]]:
    """
    Return list of (run_num, bold_path, events_path, confounds_path).
    """
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    func_dir = bids_fmri_root / sub_label / "func"
    if not func_dir.exists():
        raise FileNotFoundError(f"fMRI func directory not found: {func_dir}")

    # Discover run numbers from events files first.
    events_glob = [
        p for p in sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_events.tsv")) if not p.name.endswith("_bold_events.tsv")
    ]
    if not events_glob:
        events_glob = sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_bold_events.tsv"))
    run_nums: List[int] = []
    for p in events_glob:
        try:
            run_str = p.name.split("_run-")[1].split("_")[0]
            run_nums.append(int(run_str))
        except Exception:
            continue

    if runs is not None:
        want = {int(r) for r in runs}
        run_nums = [r for r in sorted(set(run_nums)) if r in want]
    else:
        run_nums = sorted(set(run_nums))

    if not run_nums:
        raise FileNotFoundError(f"No runs found for subject {subject}, task {task} in {func_dir}")

    out: List[Tuple[int, Path, Path, Optional[Path]]] = []
    for run_num in run_nums:
        events_patterns = [
            f"{sub_label}_task-{task}_run-{run_num:02d}_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num}_events.tsv",
            f"{sub_label}_task-{task}_run-0{run_num}_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num:02d}_bold_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num}_bold_events.tsv",
            f"{sub_label}_task-{task}_run-0{run_num}_bold_events.tsv",
        ]
        events_path = next((func_dir / n for n in events_patterns if (func_dir / n).exists()), None)
        if events_path is None:
            continue

        bold_path: Optional[Path] = None
        confounds_path: Optional[Path] = None

        if input_source == "fmriprep" and bids_derivatives is not None:
            bold_path = _discover_fmriprep_preproc_bold(
                bids_derivatives=bids_derivatives,
                sub_label=sub_label,
                task=task,
                run_num=run_num,
                space=fmriprep_space,
            )
            confounds_path = _discover_confounds(
                bids_derivatives=bids_derivatives,
                sub_label=sub_label,
                task=task,
                run_num=run_num,
            )

        if bold_path is None:
            raw_patterns = [
                f"{sub_label}_task-{task}_run-{run_num:02d}_bold.nii.gz",
                f"{sub_label}_task-{task}_run-{run_num}_bold.nii.gz",
            ]
            bold_path = next((func_dir / n for n in raw_patterns if (func_dir / n).exists()), None)

        if bold_path is None:
            continue

        out.append((int(run_num), bold_path, events_path, confounds_path))

    if not out:
        raise FileNotFoundError(f"No usable runs found for subject {subject}, task {task}")
    return out


def _coerce_condition_value(value: str, series: Any) -> Any:
    """
    Coerce CLI string values to the dtype of a pandas Series (best-effort).
    """
    try:
        import pandas as pd  # type: ignore

        if isinstance(series, pd.Series):
            if pd.api.types.is_integer_dtype(series):
                try:
                    return int(value)
                except Exception:
                    return value
            if pd.api.types.is_float_dtype(series):
                try:
                    return float(value)
                except Exception:
                    return value
            if pd.api.types.is_bool_dtype(series):
                return str(value).strip().lower() in ("true", "1", "yes")
    except Exception:
        pass
    return value


def _select_confounds(confounds_path: Optional[Path], strategy: str) -> tuple[Optional[Any], List[str]]:
    if confounds_path is None or not confounds_path.exists():
        return None, []

    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(confounds_path, sep="\t")
        cols = select_fmriprep_confounds_columns(list(df.columns), strategy=strategy)
        if not cols:
            return None, []
        selected = df[cols].copy().fillna(0)
        return selected, list(selected.columns)
    except Exception as exc:
        logger.warning("Failed to read/select confounds from %s (%s)", confounds_path, exc)
        return None, []


def _build_first_level_model(
    *,
    tr: float,
    cfg: TrialSignatureExtractionConfig,
    mask_img: Optional[Any],
) -> Any:
    from nilearn.glm.first_level import FirstLevelModel  # type: ignore

    high_pass = cfg.high_pass_hz if float(cfg.high_pass_hz) > 0 else None
    low_pass = cfg.low_pass_hz

    kwargs: Dict[str, Any] = dict(
        t_r=float(tr),
        hrf_model=cfg.hrf_model,
        drift_model=cfg.drift_model,
        high_pass=high_pass,
        noise_model="ar1",
        standardize=True,
        signal_scaling=0,
        minimize_memory=False,
    )

    sig = inspect.signature(FirstLevelModel)
    if "low_pass" in sig.parameters:
        kwargs["low_pass"] = low_pass
    if cfg.smoothing_fwhm is not None and "smoothing_fwhm" in sig.parameters:
        kwargs["smoothing_fwhm"] = cfg.smoothing_fwhm
    if mask_img is not None and "mask_img" in sig.parameters:
        kwargs["mask_img"] = mask_img

    return FirstLevelModel(**kwargs)


def _make_trial_regressor(run_label: str, trial_index: int, condition: str) -> str:
    return f"trial_{_safe_slug(run_label)}_{trial_index:03d}_{condition.lower()}"


def _extract_trials_for_run(
    *,
    events_df: Any,
    cfg: TrialSignatureExtractionConfig,
    events_path: Path,
    run_num: int,
) -> Tuple[List[TrialInfo], Any]:
    """
    Returns (trial_infos, modeled_events_df).
    """
    import pandas as pd  # type: ignore

    required = {"onset", "duration"}
    if not required.issubset(set(events_df.columns)):
        raise ValueError(f"Events file missing required columns {required}: {events_path}")

    col_a = str(cfg.condition_a_column).strip()
    col_b = str(cfg.condition_b_column).strip()
    if col_a not in events_df.columns or col_b not in events_df.columns:
        raise ValueError(f"Events file missing selection columns: {col_a}, {col_b} in {events_path}")

    val_a = _coerce_condition_value(cfg.condition_a_value, events_df[col_a])
    val_b = _coerce_condition_value(cfg.condition_b_value, events_df[col_b])

    mask_a = events_df[col_a] == val_a
    mask_b = events_df[col_b] == val_b
    sel_mask = mask_a | mask_b

    selected = events_df.loc[sel_mask].copy()
    if selected.empty:
        raise ValueError(f"No trials matched cond A/B selection in {events_path}")

    selected = selected.sort_values("onset").reset_index(drop=False)

    run_label = f"run-{run_num:02d}"
    trials: List[TrialInfo] = []

    max_trials = cfg.max_trials_per_run
    if max_trials is not None and int(max_trials) > 0:
        selected = selected.iloc[: int(max_trials)].copy()

    modeled_rows: List[Dict[str, Any]] = []

    for idx, row in selected.iterrows():
        condition = "A" if bool(mask_a.loc[row["index"]]) else "B"
        reg = _make_trial_regressor(run_label, int(idx) + 1, condition)

        onset = float(row["onset"])
        duration = float(row["duration"])
        orig_tt = str(row.get("trial_type", "n/a"))

        extra: Dict[str, str] = {}
        for k, v in row.items():
            if k in {"onset", "duration", "trial_type", "index"}:
                continue
            if v is None:
                continue
            extra[str(k)] = str(v)

        trials.append(
            TrialInfo(
                run=int(run_num),
                run_label=run_label,
                trial_index=int(idx) + 1,
                condition=condition,
                regressor=reg,
                onset=onset,
                duration=duration,
                original_trial_type=orig_tt,
                source_events_path=events_path,
                source_row=int(row["index"]),
                extra=extra,
            )
        )
        modeled_rows.append({"onset": onset, "duration": duration, "trial_type": reg})

    if cfg.include_other_events:
        others = events_df.loc[~sel_mask].copy()
        if not others.empty:
            # Group non-selected events by their original trial_type (best-effort).
            for _idx, row in others.iterrows():
                tt = str(row.get("trial_type", "other"))
                modeled_rows.append(
                    {
                        "onset": float(row["onset"]),
                        "duration": float(row["duration"]),
                        "trial_type": f"nuis_{_safe_slug(tt)}",
                    }
                )

    modeled_events = pd.DataFrame(modeled_rows, columns=["onset", "duration", "trial_type"])
    return trials, modeled_events


def _build_lss_events(
    *,
    trial: TrialInfo,
    all_trials: Sequence[TrialInfo],
    original_events_df: Any,
    cfg: TrialSignatureExtractionConfig,
) -> Any:
    import pandas as pd  # type: ignore

    rows: List[Dict[str, Any]] = []
    # Target trial regressor
    rows.append({"onset": trial.onset, "duration": trial.duration, "trial_type": "target"})

    # Other selected trials
    others = [t for t in all_trials if not (t.run == trial.run and t.trial_index == trial.trial_index)]
    if cfg.lss_other_regressors == "all":
        for t in others:
            rows.append({"onset": t.onset, "duration": t.duration, "trial_type": "other_trials"})
    else:
        for t in others:
            rows.append(
                {
                    "onset": t.onset,
                    "duration": t.duration,
                    "trial_type": "other_cond_a" if t.condition == "A" else "other_cond_b",
                }
            )

    if cfg.include_other_events:
        # Add non-selected events (those not represented in all_trials) as nuisance regressors.
        represented_rows = {int(t.source_row) for t in all_trials}
        for idx, row in original_events_df.iterrows():
            try:
                onset = float(row["onset"])
                duration = float(row["duration"])
            except Exception:
                continue
            row_id = None
            try:
                row_id = int(idx)
            except Exception:
                row_id = None
            if row_id is not None and row_id in represented_rows:
                continue
            tt = str(row.get("trial_type", "other"))
            rows.append({"onset": onset, "duration": duration, "trial_type": f"nuis_{_safe_slug(tt)}"})

    return pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])


def _contrast_vector_for_column(columns: Sequence[str], col: str) -> Any:
    # Nilearn's FirstLevelModel.compute_contrast treats Python lists as "one contrast per run".
    # For single-run models, passing a list of floats (i.e., a 1D contrast vector) is ambiguous
    # and can be misinterpreted as multiple per-run contrasts. Use a numpy array to force the
    # intended 1D vector semantics.
    import numpy as np  # type: ignore

    try:
        idx = list(columns).index(col)
    except Exception as exc:
        raise KeyError(f"Column not found in design matrix: {col}") from exc
    vec = np.zeros(len(columns), dtype=float)
    vec[idx] = 1.0
    return vec


def _fixed_effects_combine_effects(
    *,
    effects: Sequence[Any],
    variances: Optional[Sequence[Any]],
    method: str,
) -> Any:
    """
    Combine effect images using either simple mean or inverse-variance weighting.
    """
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore

    if not effects:
        raise ValueError("No effect images provided.")

    ref = effects[0]
    ref_img = nib.load(str(ref)) if isinstance(ref, (str, Path)) else ref
    eff_imgs = [nib.load(str(e)) if isinstance(e, (str, Path)) else e for e in effects]
    eff = np.stack([np.asanyarray(img.dataobj) for img in eff_imgs], axis=0)

    method = (method or "variance").strip().lower()
    if method == "mean" or not variances:
        out = np.nanmean(eff, axis=0)
        return nib.Nifti1Image(out, ref_img.affine, ref_img.header)

    var_imgs = [nib.load(str(v)) if isinstance(v, (str, Path)) else v for v in variances]
    var = np.stack([np.asanyarray(img.dataobj) for img in var_imgs], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / var
        w[~np.isfinite(w)] = 0.0
        num = np.sum(w * eff, axis=0)
        den = np.sum(w, axis=0)
        out = np.zeros_like(num, dtype=float)
        m = den > 0
        out[m] = num[m] / den[m]
        out[~m] = 0.0
    return nib.Nifti1Image(out, ref_img.affine, ref_img.header)


def _write_tsv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def run_trial_signature_extraction_for_subject(
    *,
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    deriv_root: Path,
    subject: str,
    cfg: TrialSignatureExtractionConfig,
    signature_root: Optional[Path],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute trial-wise beta maps (beta-series or LSS) and multivariate signature expression.

    Outputs (per subject) under:
      <deriv_root>/sub-XX/fmri/<beta_series|lss>/task-<task>/contrast-<name>/
        - trials.tsv
        - signatures/trial_signature_expression.tsv
        - signatures/trial_signature_expression_rois.tsv (optional)
        - condition_betas/*.nii.gz (optional)
        - signatures/condition_signature_expression.tsv
        - signatures/condition_signature_expression_rois.tsv (optional)
        - provenance.json
        - trial_betas/**/*.nii.gz (optional)
    """
    cfg = cfg.normalized()
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    if signature_root is not None:
        # Scientific validity guardrail: the bundled NPS/SIIPS1 weights are in MNI space.
        # Resampling them into subject-native/T1w grids is not a valid comparison.
        space = str(cfg.fmriprep_space or "").strip().lower()
        if "mni" not in space:
            raise ValueError(
                "Trial-wise signature extraction requires MNI-space images for NPS/SIIPS1. "
                "Re-run with --input-source fmriprep --fmriprep-space MNI152NLin2009cAsym "
                "(or provide MNI-space inputs)."
            )
    if cfg.method == "beta-series":
        root = deriv_root / sub_label / "fmri" / "beta_series" / f"task-{cfg.task}" / f"contrast-{_safe_slug(cfg.name)}"
    else:
        root = deriv_root / sub_label / "fmri" / "lss" / f"task-{cfg.task}" / f"contrast-{_safe_slug(cfg.name)}"
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else root
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=cfg.task,
        runs=cfg.runs,
        input_source=cfg.input_source,
        fmriprep_space=cfg.fmriprep_space,
    )

    provenance = {
        "subject": sub_label,
        "task": cfg.task,
        "method": cfg.method,
        "cfg": asdict(cfg),
        "signature_root": str(signature_root) if signature_root else None,
        "n_runs": len(runs),
    }

    trial_infos: List[TrialInfo] = []
    trial_rows_out: List[Dict[str, Any]] = []
    trial_sig_rows: List[Dict[str, Any]] = []
    trial_sig_roi_rows: List[Dict[str, Any]] = []
    run_brain_masks: List[Any] = []

    roi_enabled = bool(signature_root is not None and cfg.roi_atlas and cfg.roi_names)
    roi_atlas_path: Optional[Path] = None
    roi_labels_path: Optional[Path] = None
    roi_label_map: Dict[int, str] = {}
    roi_selected: List[Tuple[int, str]] = []
    roi_name_to_label: Dict[str, int] = {}
    atlas_display: Optional[str] = None

    if roi_enabled:
        atlas_search_roots: List[Path] = []
        if signature_root is not None:
            atlas_search_roots.append(Path(signature_root))
            atlas_search_roots.append(Path(signature_root) / "atlases")
        external_root = Path(deriv_root).expanduser().resolve().parent / "external"
        atlas_search_roots.append(external_root)
        atlas_search_roots.append(external_root / "atlases")
        atlas_search_roots.append(Path(deriv_root))

        roi_atlas_path = _resolve_path_with_search(
            str(cfg.roi_atlas),
            search_roots=atlas_search_roots,
            suffixes=(".nii.gz", ".nii"),
        )
        if cfg.roi_labels:
            roi_labels_path = _resolve_path_with_search(
                str(cfg.roi_labels),
                search_roots=atlas_search_roots,
                suffixes=(".tsv", ".csv", ".txt"),
            )
        else:
            roi_labels_path = _infer_atlas_labels_sidecar(roi_atlas_path)
        if roi_labels_path is None:
            raise ValueError(
                f"--signature-roi-atlas provided but no labels file could be inferred. "
                f"Pass --signature-roi-labels explicitly. atlas={roi_atlas_path}"
            )

        roi_label_map = _read_atlas_labels_table(roi_labels_path)
        roi_selected = _select_roi_labels(roi_label_map, tuple(cfg.roi_names or ()))
        roi_name_to_label = {name: int(idx) for idx, name in roi_selected}
        atlas_display = roi_atlas_path.name

    # Per-condition maps (optional)
    cond_a_effects: List[Any] = []
    cond_a_vars: List[Any] = []
    cond_b_effects: List[Any] = []
    cond_b_vars: List[Any] = []

    for run_num, bold_path, events_path, confounds_path in runs:
        import pandas as pd  # type: ignore
        import nibabel as nib  # type: ignore

        events_df = pd.read_csv(events_path, sep="\t")
        confounds, conf_cols = _select_confounds(confounds_path, cfg.confounds_strategy)

        mask_img = None
        mask_path = _discover_brain_mask_for_bold(bold_path)
        if mask_path is not None and mask_path.exists():
            try:
                mask_img = nib.load(str(mask_path))
            except Exception:
                mask_img = None
        if mask_img is not None:
            run_brain_masks.append(mask_img)

        tr = _get_tr_from_bold(bold_path)

        # Determine trial list and build run-level modeled events.
        trials, modeled_events = _extract_trials_for_run(
            events_df=events_df,
            cfg=cfg,
            events_path=events_path,
            run_num=run_num,
        )
        if not trials:
            continue

        run_roi_masks: Optional[Dict[str, Any]] = None
        run_roi_masks_final: Optional[Dict[str, Any]] = None
        run_roi_mask_meta: Optional[Dict[str, Tuple[str, int]]] = None

        if cfg.method == "beta-series":
            flm = _build_first_level_model(tr=tr, cfg=cfg, mask_img=mask_img)
            flm.fit(bold_path, events=modeled_events, confounds=confounds)

            dm = flm.design_matrices_[0]
            dm_cols = list(getattr(dm, "columns", []))

            for t in trials:
                trial_infos.append(t)
                trial_rows_out.append(
                    {
                        "subject": sub_label,
                        "task": cfg.task,
                        "run": t.run_label,
                        "trial_index": t.trial_index,
                        "condition": t.condition,
                        "regressor": t.regressor,
                        "onset": f"{t.onset:.6f}",
                        "duration": f"{t.duration:.6f}",
                        "original_trial_type": t.original_trial_type,
                        "events_file": str(t.source_events_path),
                        "events_row": t.source_row,
                        **{f"events_{k}": v for k, v in sorted(t.extra.items())},
                    }
                )

                con = _contrast_vector_for_column(dm_cols, t.regressor)
                beta_img = flm.compute_contrast(con, output_type="effect_size")
                var_img = None
                try:
                    var_img = flm.compute_contrast(con, output_type="effect_variance")
                except Exception:
                    var_img = None

                if t.condition == "A":
                    cond_a_effects.append(beta_img)
                    if var_img is not None:
                        cond_a_vars.append(var_img)
                else:
                    cond_b_effects.append(beta_img)
                    if var_img is not None:
                        cond_b_vars.append(var_img)

                if cfg.write_trial_betas:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_{t.regressor}_beta.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(beta_img, str(p))
                if cfg.write_trial_variances and var_img is not None:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_{t.regressor}_var.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(var_img, str(p))

                if signature_root is not None:
                    sigs = compute_pain_signature_expression(
                        stat_or_effect_img=beta_img,
                        signature_root=signature_root,
                        mask_img=mask_img,
                        signatures=cfg.signatures,
                    )
                    for s in sigs:
                        trial_sig_rows.append(
                            {
                                "subject": sub_label,
                                "task": cfg.task,
                                "method": cfg.method,
                                "run": t.run_label,
                                "run_num": t.run,
                                "trial_index": t.trial_index,
                                "condition": t.condition,
                                "regressor": t.regressor,
                                "onset": f"{t.onset:.6f}",
                                "duration": f"{t.duration:.6f}",
                                "signature": s.name,
                                "dot": f"{s.dot:.8g}",
                                "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                "n_voxels": s.n_voxels,
                                "weights": str(s.weight_path),
                            }
                        )

                if roi_enabled and signature_root is not None and roi_atlas_path is not None:
                    if run_roi_masks_final is None:
                        run_roi_masks = _build_roi_masks_from_atlas(
                            atlas_path=roi_atlas_path,
                            labels=roi_selected,
                            target_img=beta_img,
                        )
                        run_roi_masks_final = {}
                        run_roi_mask_meta = {}
                        for roi_name, roi_mask in run_roi_masks.items():
                            final_mask = (
                                _intersect_masks_to_target(
                                    roi_mask_img=roi_mask, brain_mask_img=mask_img, target_img=beta_img
                                )
                                if mask_img is not None
                                else roi_mask
                            )
                            h, n = _mask_hash_and_count(final_mask)
                            run_roi_masks_final[str(roi_name)] = final_mask
                            run_roi_mask_meta[str(roi_name)] = (h, n)

                    for roi_name, roi_mask in (run_roi_masks_final or {}).items():
                        sigs = compute_pain_signature_expression(
                            stat_or_effect_img=beta_img,
                            signature_root=signature_root,
                            mask_img=roi_mask,
                            signatures=cfg.signatures,
                        )
                        for s in sigs:
                            mh, mn = ("", 0)
                            if run_roi_mask_meta and str(roi_name) in run_roi_mask_meta:
                                mh, mn = run_roi_mask_meta[str(roi_name)]
                            trial_sig_roi_rows.append(
                                {
                                    "subject": sub_label,
                                    "task": cfg.task,
                                    "method": cfg.method,
                                    "run": t.run_label,
                                    "run_num": t.run,
                                    "trial_index": t.trial_index,
                                    "condition": t.condition,
                                    "regressor": t.regressor,
                                    "onset": f"{t.onset:.6f}",
                                    "duration": f"{t.duration:.6f}",
                                    "atlas": atlas_display or "",
                                    "roi": str(roi_name),
                                    "roi_label": roi_name_to_label.get(str(roi_name), ""),
                                    "roi_mask_hash": mh,
                                    "roi_mask_n_voxels": mn,
                                    "signature": s.name,
                                    "dot": f"{s.dot:.8g}",
                                    "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                    "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                    "n_voxels": s.n_voxels,
                                    "weights": str(s.weight_path),
                                }
                            )

        else:
            # LSS: one model per trial within each run
            # Build the run-level trial list once, then fit per-trial models.
            for t in trials:
                trial_infos.append(t)
                trial_rows_out.append(
                    {
                        "subject": sub_label,
                        "task": cfg.task,
                        "run": t.run_label,
                        "trial_index": t.trial_index,
                        "condition": t.condition,
                        "regressor": "target",
                        "onset": f"{t.onset:.6f}",
                        "duration": f"{t.duration:.6f}",
                        "original_trial_type": t.original_trial_type,
                        "events_file": str(t.source_events_path),
                        "events_row": t.source_row,
                        **{f"events_{k}": v for k, v in sorted(t.extra.items())},
                    }
                )

                lss_events = _build_lss_events(trial=t, all_trials=trials, original_events_df=events_df, cfg=cfg)
                flm = _build_first_level_model(tr=tr, cfg=cfg, mask_img=mask_img)
                flm.fit(bold_path, events=lss_events, confounds=confounds)

                dm = flm.design_matrices_[0]
                dm_cols = list(getattr(dm, "columns", []))
                con = _contrast_vector_for_column(dm_cols, "target")
                beta_img = flm.compute_contrast(con, output_type="effect_size")
                var_img = None
                try:
                    var_img = flm.compute_contrast(con, output_type="effect_variance")
                except Exception:
                    var_img = None

                if t.condition == "A":
                    cond_a_effects.append(beta_img)
                    if var_img is not None:
                        cond_a_vars.append(var_img)
                else:
                    cond_b_effects.append(beta_img)
                    if var_img is not None:
                        cond_b_vars.append(var_img)

                if cfg.write_trial_betas:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_trial-{t.trial_index:03d}_beta.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(beta_img, str(p))
                if cfg.write_trial_variances and var_img is not None:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_trial-{t.trial_index:03d}_var.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(var_img, str(p))

                if signature_root is not None:
                    sigs = compute_pain_signature_expression(
                        stat_or_effect_img=beta_img,
                        signature_root=signature_root,
                        mask_img=mask_img,
                        signatures=cfg.signatures,
                    )
                    for s in sigs:
                        trial_sig_rows.append(
                            {
                                "subject": sub_label,
                                "task": cfg.task,
                                "method": cfg.method,
                                "run": t.run_label,
                                "run_num": t.run,
                                "trial_index": t.trial_index,
                                "condition": t.condition,
                                "signature": s.name,
                                "dot": f"{s.dot:.8g}",
                                "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                "n_voxels": s.n_voxels,
                                "weights": str(s.weight_path),
                                "onset": f"{t.onset:.6f}",
                                "duration": f"{t.duration:.6f}",
                            }
                        )

                if roi_enabled and signature_root is not None and roi_atlas_path is not None:
                    if run_roi_masks_final is None:
                        run_roi_masks = _build_roi_masks_from_atlas(
                            atlas_path=roi_atlas_path,
                            labels=roi_selected,
                            target_img=beta_img,
                        )
                        run_roi_masks_final = {}
                        run_roi_mask_meta = {}
                        for roi_name, roi_mask in run_roi_masks.items():
                            final_mask = (
                                _intersect_masks_to_target(
                                    roi_mask_img=roi_mask, brain_mask_img=mask_img, target_img=beta_img
                                )
                                if mask_img is not None
                                else roi_mask
                            )
                            h, n = _mask_hash_and_count(final_mask)
                            run_roi_masks_final[str(roi_name)] = final_mask
                            run_roi_mask_meta[str(roi_name)] = (h, n)

                    for roi_name, roi_mask in (run_roi_masks_final or {}).items():
                        sigs = compute_pain_signature_expression(
                            stat_or_effect_img=beta_img,
                            signature_root=signature_root,
                            mask_img=roi_mask,
                            signatures=cfg.signatures,
                        )
                        for s in sigs:
                            mh, mn = ("", 0)
                            if run_roi_mask_meta and str(roi_name) in run_roi_mask_meta:
                                mh, mn = run_roi_mask_meta[str(roi_name)]
                            trial_sig_roi_rows.append(
                                {
                                    "subject": sub_label,
                                    "task": cfg.task,
                                    "method": cfg.method,
                                    "run": t.run_label,
                                    "run_num": t.run,
                                    "trial_index": t.trial_index,
                                    "condition": t.condition,
                                    "regressor": "target",
                                    "onset": f"{t.onset:.6f}",
                                    "duration": f"{t.duration:.6f}",
                                    "atlas": atlas_display or "",
                                    "roi": str(roi_name),
                                    "roi_label": roi_name_to_label.get(str(roi_name), ""),
                                    "roi_mask_hash": mh,
                                    "roi_mask_n_voxels": mn,
                                    "signature": s.name,
                                    "dot": f"{s.dot:.8g}",
                                    "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                    "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                    "n_voxels": s.n_voxels,
                                    "weights": str(s.weight_path),
                                }
                            )

    # Write metadata tables
    _write_tsv(out_dir / "trials.tsv", trial_rows_out)
    _write_tsv(out_dir / "signatures" / "trial_signature_expression.tsv", trial_sig_rows)
    if roi_enabled:
        _write_tsv(out_dir / "signatures" / "trial_signature_expression_rois.tsv", trial_sig_roi_rows)

    # Condition averages (fixed-effects) + signatures
    cond_rows: List[Dict[str, Any]] = []
    cond_roi_rows: List[Dict[str, Any]] = []
    if (cond_a_effects or cond_b_effects) and (cfg.write_condition_betas or signature_root is not None or roi_enabled):
        import nibabel as nib  # type: ignore

        cond_dir = out_dir / "condition_betas"
        if cfg.write_condition_betas:
            cond_dir.mkdir(parents=True, exist_ok=True)

        a_img = None
        b_img = None
        if cond_a_effects:
            a_img = _fixed_effects_combine_effects(
                effects=cond_a_effects,
                variances=cond_a_vars if cond_a_vars else None,
                method=cfg.fixed_effects_weighting,
            )
            if cfg.write_condition_betas:
                nib.save(a_img, str(cond_dir / f"{sub_label}_task-{cfg.task}_cond-a_beta.nii.gz"))
        if cond_b_effects:
            b_img = _fixed_effects_combine_effects(
                effects=cond_b_effects,
                variances=cond_b_vars if cond_b_vars else None,
                method=cfg.fixed_effects_weighting,
            )
            if cfg.write_condition_betas:
                nib.save(b_img, str(cond_dir / f"{sub_label}_task-{cfg.task}_cond-b_beta.nii.gz"))

        diff_img = None
        if a_img is not None and b_img is not None:
            import numpy as np  # type: ignore

            a = np.asanyarray(a_img.dataobj)
            b = np.asanyarray(b_img.dataobj)
            diff_img = nib.Nifti1Image(a - b, a_img.affine, a_img.header)
            if cfg.write_condition_betas:
                nib.save(diff_img, str(cond_dir / f"{sub_label}_task-{cfg.task}_cond-a_minus_b_beta.nii.gz"))

        if signature_root is not None:
            for label, img in [
                ("cond_a", a_img),
                ("cond_b", b_img),
                ("cond_a_minus_b", diff_img),
            ]:
                if img is None:
                    continue
                sigs = compute_pain_signature_expression(
                    stat_or_effect_img=img,
                    signature_root=signature_root,
                    mask_img=None,
                    signatures=cfg.signatures,
                )
                for s in sigs:
                    cond_rows.append(
                        {
                            "subject": sub_label,
                            "task": cfg.task,
                            "method": cfg.method,
                            "map": label,
                            "signature": s.name,
                            "dot": f"{s.dot:.8g}",
                            "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                            "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                            "n_voxels": s.n_voxels,
                                "weights": str(s.weight_path),
                        }
                    )
        if roi_enabled and signature_root is not None and roi_atlas_path is not None:
            target_for_rois = a_img or b_img or diff_img
            if target_for_rois is not None:
                cond_roi_masks = _build_roi_masks_from_atlas(
                    atlas_path=roi_atlas_path,
                    labels=roi_selected,
                    target_img=target_for_rois,
                )
                cond_roi_masks_final: Dict[str, Any] = {}
                cond_roi_mask_meta: Dict[str, Tuple[str, int]] = {}
                brain_union = None
                if run_brain_masks:
                    brain_union = _union_masks_to_target(run_brain_masks, target_for_rois)
                for roi_name, roi_mask in cond_roi_masks.items():
                    final_mask = (
                        _intersect_masks_to_target(
                            roi_mask_img=roi_mask, brain_mask_img=brain_union, target_img=target_for_rois
                        )
                        if brain_union is not None
                        else roi_mask
                    )
                    h, n = _mask_hash_and_count(final_mask)
                    cond_roi_masks_final[str(roi_name)] = final_mask
                    cond_roi_mask_meta[str(roi_name)] = (h, n)

                for map_label, img in [
                    ("cond_a", a_img),
                    ("cond_b", b_img),
                    ("cond_a_minus_b", diff_img),
                ]:
                    if img is None:
                        continue
                    for roi_name, roi_mask in cond_roi_masks_final.items():
                        sigs = compute_pain_signature_expression(
                            stat_or_effect_img=img,
                            signature_root=signature_root,
                            mask_img=roi_mask,
                            signatures=cfg.signatures,
                        )
                        for s in sigs:
                            mh, mn = cond_roi_mask_meta.get(str(roi_name), ("", 0))
                            cond_roi_rows.append(
                                {
                                    "subject": sub_label,
                                    "task": cfg.task,
                                    "method": cfg.method,
                                    "map": map_label,
                                    "atlas": atlas_display or "",
                                    "roi": str(roi_name),
                                    "roi_label": roi_name_to_label.get(str(roi_name), ""),
                                    "roi_mask_hash": mh,
                                    "roi_mask_n_voxels": mn,
                                    "signature": s.name,
                                    "dot": f"{s.dot:.8g}",
                                    "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                    "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                    "n_voxels": s.n_voxels,
                                    "weights": str(s.weight_path),
                                }
                            )
    _write_tsv(out_dir / "signatures" / "condition_signature_expression.tsv", cond_rows)
    if roi_enabled:
        _write_tsv(out_dir / "signatures" / "condition_signature_expression_rois.tsv", cond_roi_rows)

    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return {
        "output_dir": str(out_dir),
        "n_trials": len(trial_infos),
        "n_trial_signature_rows": len(trial_sig_rows),
        "n_condition_signature_rows": len(cond_rows),
        "n_trial_signature_roi_rows": len(trial_sig_roi_rows),
        "n_condition_signature_roi_rows": len(cond_roi_rows),
    }
