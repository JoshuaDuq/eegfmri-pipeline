"""Raw fMRI (DICOM) to BIDS conversion utilities.

This module converts per-series DICOM directories into a BIDS-compliant fMRI
dataset using `dcm2niix` (external dependency). It also optionally generates
`*_events.tsv` files from PsychoPy TrialSummary CSVs for the thermal pain task.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import pandas as pd

from eeg_pipeline.utils.data.preprocessing import ensure_dataset_description, find_behavior_csv_for_run

logger = logging.getLogger(__name__)


class Dcm2NiixNotFoundError(RuntimeError):
    pass


def _sidecar_json_for_nifti(nifti_path: Path) -> Path:
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return nifti_path.with_name(name[:-7] + ".json")
    if name.endswith(".nii"):
        return nifti_path.with_suffix(".json")
    raise ValueError(f"Not a NIfTI file: {nifti_path}")


def _run_dcm2niix(
    dicom_dir: Path,
    out_dir: Path,
    *,
    dcm2niix_path: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    _logger: Optional[logging.Logger] = None,
) -> None:
    log = _logger or logger
    exe = dcm2niix_path or shutil.which("dcm2niix")
    if not exe:
        raise Dcm2NiixNotFoundError(
            "dcm2niix not found on PATH. Install it (recommended) or pass --dcm2niix-path."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # -b y: write BIDS sidecars (JSON)
    # -z y: gzip NIfTI output
    # -f: predictable filename (protocol + series number)
    cmd = [
        str(exe),
        "-b",
        "y",
        "-z",
        "y",
        "-f",
        "%p_%s",
        "-o",
        str(out_dir),
        str(dicom_dir),
    ]
    if extra_args:
        cmd[1:1] = list(extra_args)

    log.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        log.info(proc.stdout.strip())
    if proc.returncode != 0:
        stderr = proc.stderr.strip() if proc.stderr else ""
        raise RuntimeError(f"dcm2niix failed (exit {proc.returncode}): {stderr}")


def _choose_primary_nifti(out_dir: Path) -> Path:
    niftis = list(out_dir.glob("*.nii")) + list(out_dir.glob("*.nii.gz"))
    if not niftis:
        raise FileNotFoundError(f"No NIfTI output found in: {out_dir}")
    niftis.sort(key=lambda p: p.stat().st_size, reverse=True)
    return niftis[0]


def _list_niftis(out_dir: Path) -> list[Path]:
    niftis = list(out_dir.glob("*.nii")) + list(out_dir.glob("*.nii.gz"))
    niftis = [p for p in niftis if p.is_file()]
    niftis.sort(key=lambda p: p.stat().st_size, reverse=True)
    return niftis


def _looks_like_phasediff(meta: dict[str, Any], filename: str) -> bool:
    if "EchoTime1" in meta and "EchoTime2" in meta:
        return True
    image_type = meta.get("ImageType")
    if isinstance(image_type, list) and any("PHASE" in str(x).upper() for x in image_type):
        return True
    if "phase" in filename.lower():
        return True
    return False


def _classify_fieldmap_outputs(tmp_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[Path]]:
    """Return ([(suffix, nifti, json)], json_paths_to_update_intendedfor)."""
    niftis = _list_niftis(tmp_dir)
    if not niftis:
        raise FileNotFoundError(f"No NIfTI output found in: {tmp_dir}")

    phase_candidates: list[tuple[Path, Path, dict[str, Any]]] = []
    mag_candidates: list[tuple[Path, Path, dict[str, Any]]] = []

    for nifti in niftis:
        js = _sidecar_json_for_nifti(nifti)
        meta = _load_json(js) if js.exists() else {}
        if _looks_like_phasediff(meta, nifti.name):
            phase_candidates.append((nifti, js, meta))
        else:
            mag_candidates.append((nifti, js, meta))

    # If we have no obvious phase output, fall back to treating the smallest file as phase.
    if not phase_candidates and len(mag_candidates) >= 2:
        mag_candidates.sort(key=lambda t: t[0].stat().st_size)
        phase_candidates.append(mag_candidates.pop(0))

    outputs: list[tuple[str, Path, Path]] = []
    intended_jsons: list[Path] = []

    if phase_candidates:
        phase_candidates.sort(key=lambda t: t[0].stat().st_size, reverse=True)
        nifti, js, _ = phase_candidates[0]
        outputs.append(("phasediff", nifti, js))
        intended_jsons.append(js)

    if mag_candidates:
        # Sort magnitudes by EchoTime if present, else by size descending.
        def mag_key(t: tuple[Path, Path, dict[str, Any]]) -> tuple[float, int]:
            et = t[2].get("EchoTime")
            try:
                et_val = float(et) if et is not None else float("inf")
            except (TypeError, ValueError):
                et_val = float("inf")
            return (et_val, -t[0].stat().st_size)

        mag_candidates.sort(key=mag_key)
        outputs.append(("magnitude1", mag_candidates[0][0], mag_candidates[0][1]))
        intended_jsons.append(mag_candidates[0][1])
        if len(mag_candidates) > 1:
            outputs.append(("magnitude2", mag_candidates[1][0], mag_candidates[1][1]))
            intended_jsons.append(mag_candidates[1][1])

    if not outputs:
        # As a last resort, keep the primary output as magnitude1.
        primary = _choose_primary_nifti(tmp_dir)
        outputs.append(("magnitude1", primary, _sidecar_json_for_nifti(primary)))

    return outputs, intended_jsons

def _safe_write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _copy_or_link_tree(
    source_dir: Path,
    target_dir: Path,
    *,
    mode: str,
    overwrite: bool,
) -> None:
    if mode == "skip":
        return
    if target_dir.exists():
        if not overwrite:
            return
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        target_dir.symlink_to(source_dir, target_is_directory=True)
        return
    if mode == "copy":
        shutil.copytree(source_dir, target_dir)
        return
    raise ValueError(f"Unknown dicom_mode: {mode}")


@dataclass(frozen=True)
class FmriSeries:
    kind: str  # "t1w" | "bold" | "rest" | "fmap"
    dicom_dir: Path
    run: Optional[int] = None


def discover_series(dicom_root: Path) -> list[FmriSeries]:
    """Discover series directories for a subject (simple name-based heuristics)."""
    series: list[FmriSeries] = []
    if not dicom_root.exists():
        return series

    fmri_dirs = [p for p in sorted(dicom_root.iterdir()) if p.is_dir()]

    fmap_dirs: list[Path] = []
    for d in fmri_dirs:
        name = d.name.lower()
        if "mprage" in name or "t1w" in name or "t1" in name:
            series.append(FmriSeries(kind="t1w", dicom_dir=d))
            continue

        m = re.search(r"painr(\d+)", name)
        if m:
            series.append(FmriSeries(kind="bold", dicom_dir=d, run=int(m.group(1))))
            continue

        if "rs_" in name or "rest" in name or "rs " in name:
            series.append(FmriSeries(kind="rest", dicom_dir=d, run=1))
            continue

        if "field" in name and "map" in name:
            fmap_dirs.append(d)
            continue

    # Fieldmaps: assign run numbers by discovery order
    for idx, d in enumerate(fmap_dirs, 1):
        series.append(FmriSeries(kind="fmap", dicom_dir=d, run=idx))

    # Sort: T1 first, then fmap, then rest, then bold
    sort_order = {"t1w": 0, "fmap": 1, "rest": 2, "bold": 3}
    series.sort(key=lambda s: (sort_order.get(s.kind, 99), s.run or 0, s.dicom_dir.name))
    return series


def _bids_subject_dir(bids_root: Path, subject: str, session: Optional[str]) -> Path:
    subj = bids_root / f"sub-{subject}"
    if session:
        return subj / f"ses-{session}"
    return subj


def _format_run(run: Optional[int]) -> Optional[str]:
    if run is None:
        return None
    return f"{run:02d}"


def _ensure_participants_tsv(bids_root: Path, subjects: Iterable[str]) -> None:
    path = bids_root / "participants.tsv"
    existing: set[str] = set()
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            if line.strip():
                existing.add(line.split("\t")[0].strip())
    else:
        path.write_text("participant_id\n", encoding="utf-8")

    new_rows = []
    for s in sorted(set(subjects)):
        pid = f"sub-{s}"
        if pid not in existing:
            new_rows.append(pid)
    if new_rows:
        with path.open("a", encoding="utf-8") as handle:
            for pid in new_rows:
                handle.write(f"{pid}\n")


def _write_events_tsv_for_run(
    behavior_csv: Path,
    out_tsv: Path,
    *,
    run_id: int,
    onset_reference: str,
    onset_offset_s: float,
    event_granularity: str,
    bold_nifti: Optional[Path] = None,
    bold_json: Optional[Path] = None,
) -> None:
    df = pd.read_csv(behavior_csv)
    if "run_id" not in df.columns:
        raise ValueError(f"Missing 'run_id' column in: {behavior_csv}")
    run_df = df[df["run_id"] == run_id].copy()
    if run_df.empty:
        raise ValueError(f"No rows for run_id={run_id} in: {behavior_csv}")

    required = [
        "trial_number",
        "stimulus_temp",
        "selected_surface",
        "pain_binary_coded",
        "vas_final_coded_rating",
        "iti_start_time",
        "iti_end_time",
        "stim_start_time",
        "stim_end_time",
        "pain_q_start_time",
        "pain_q_end_time",
        "vas_start_time",
        "vas_end_time",
    ]
    missing = [c for c in required if c not in run_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {behavior_csv.name}: {missing}")

    # Determine onset reference per run (leave as-is by default).
    base = 0.0
    if onset_reference == "first_iti_start":
        base = float(run_df["iti_start_time"].min())
    elif onset_reference == "first_stim_start":
        base = float(run_df["stim_start_time"].min())
    elif onset_reference == "as_is":
        base = 0.0
    else:
        raise ValueError("onset_reference must be one of: as_is, first_iti_start, first_stim_start")

    def t(x: float) -> float:
        return float(x) - base + float(onset_offset_s)

    rows: list[dict[str, Any]] = []
    for _, tr in run_df.iterrows():
        trial = int(tr["trial_number"])
        stim_start = float(tr["stim_start_time"])
        stim_end = float(tr["stim_end_time"])
        iti_start = float(tr["iti_start_time"])
        iti_end = float(tr["iti_end_time"])
        q_start = float(tr["pain_q_start_time"])
        q_end = float(tr["pain_q_end_time"])
        vas_start = float(tr["vas_start_time"])
        vas_end = float(tr["vas_end_time"])

        pain_binary = int(tr["pain_binary_coded"])
        rating = float(tr["vas_final_coded_rating"])
        temp = float(tr["stimulus_temp"])
        surface = int(tr["selected_surface"])

        scale_min, scale_max = (100, 200) if pain_binary == 1 else (0, 99)

        common = {
            "run_id": int(run_id),
            "trial_number": trial,
            "stimulus_temp": temp,
            "selected_surface": surface,
            "pain_binary_coded": pain_binary,
            "vas_final_coded_rating": rating,
            "vas_scale_min": scale_min,
            "vas_scale_max": scale_max,
        }

        def add_event(
            trial_type: str,
            onset: float,
            duration: float,
            _common: dict[str, Any] = common,
            **extra: Any,
        ) -> None:
            row = {
                "onset": round(t(onset), 6),
                "duration": round(float(duration), 6),
                "trial_type": trial_type,
            }
            row.update(_common)
            row.update(extra)
            rows.append(row)

        # Baseline fixation (rest interval) and post-stim fixation are explicitly modeled.
        add_event("fixation_rest", iti_start, iti_end - iti_start)
        add_event("fixation_poststim", stim_end, max(0.0, q_start - stim_end))

        if event_granularity == "trial":
            add_event("stimulation", stim_start, stim_end - stim_start)
        elif event_granularity == "phases":
            add_event("stimulation", stim_start, 3.0, stim_phase="ramp_up")
            add_event("stimulation", stim_start + 3.0, 7.5, stim_phase="plateau")
            add_event("stimulation", stim_start + 10.5, max(0.0, stim_end - (stim_start + 10.5)), stim_phase="ramp_down")
        else:
            raise ValueError("event_granularity must be one of: trial, phases")

        add_event("pain_question", q_start, max(0.0, q_end - q_start))
        add_event("vas_rating", vas_start, max(0.0, vas_end - vas_start))

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    ev_df = pd.DataFrame(rows)
    ev_df = ev_df.sort_values(["onset", "trial_number", "trial_type"], kind="mergesort")

    # Scientific validity guardrail: ensure events are plausible for this run.
    # If onsets are not "seconds from BOLD run start", the GLM will be invalid.
    try:
        if bold_nifti is not None and Path(bold_nifti).exists():
            try:
                import nibabel as nib  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Validating events.tsv against BOLD duration requires nibabel. "
                    "Install nibabel or disable event generation with --no-events."
                ) from exc

            img = nib.load(str(bold_nifti))
            shape = getattr(img, "shape", None)
            n_vols = int(shape[3]) if shape and len(shape) >= 4 else 0
            tr = None
            if bold_json is not None and Path(bold_json).exists():
                meta = _load_json(Path(bold_json))
                if "RepetitionTime" in meta:
                    try:
                        tr = float(meta["RepetitionTime"])
                    except Exception:
                        tr = None
            if tr is None:
                try:
                    zooms = img.header.get_zooms()
                    if len(zooms) >= 4:
                        tr = float(zooms[3])
                except Exception:
                    tr = None

            if n_vols > 0 and tr is not None and tr > 0:
                run_dur_s = float(n_vols) * float(tr)
                max_end = float((ev_df["onset"] + ev_df["duration"]).max())
                min_onset = float(ev_df["onset"].min())
                tol = max(2.0 * float(tr), 1.0)
                if min_onset < -tol or max_end > (run_dur_s + tol):
                    raise ValueError(
                        "Events are out of bounds for the BOLD run. "
                        f"min_onset={min_onset:.3f}s, max_end={max_end:.3f}s, "
                        f"run_duration≈{run_dur_s:.3f}s (n_vols={n_vols}, TR={float(tr):.3f}). "
                        "This usually means PsychoPy timestamps are not aligned to BOLD run start. "
                        "Try --onset-reference first_iti_start or first_stim_start (and/or --onset-offset-s)."
                    )
    except Exception:
        # Raise to prevent silently generating scientifically invalid events.tsv.
        raise

    ev_df.to_csv(out_tsv, sep="\t", index=False)


def ensure_task_events_json(bids_root: Path, task: str) -> Path:
    """Write (or update) a dataset-level events.json for a task."""
    out = bids_root / f"task-{task}_events.json"
    if out.exists():
        return out
    schema = {
        "onset": {"Description": "Event onset in seconds from the start of the BOLD run."},
        "duration": {"Description": "Event duration in seconds."},
        "trial_type": {
            "Description": "Event label.",
            "Levels": {
                "fixation_rest": "Fixation cross during pre-stim rest interval (35°C baseline).",
                "fixation_poststim": "Fixation cross between stimulation end and pain question onset (random 4.5–8.5 s).",
                "stimulation": "Thermal stimulation (total 12.5 s: 3 s ramp-up, 7.5 s plateau, 2 s ramp-down).",
                "pain_question": "Binary pain question window (max 4 s; can end early on response).",
                "vas_rating": "Visual analogue scale rating window (max 7 s; can end early on response).",
            },
        },
        "stim_phase": {
            "Description": "Stimulation sub-phase (only when event_granularity=phases).",
            "Levels": {"ramp_up": "3 s ramp", "plateau": "7.5 s plateau", "ramp_down": "2 s ramp down"},
        },
        "stimulus_temp": {"Description": "Thermode target temperature (°C)."},
        "selected_surface": {"Description": "Stimulus surface index (experiment-defined)."},
        "pain_binary_coded": {"Description": "Pain yes/no response (1=yes, 0=no)."},
        "vas_final_coded_rating": {
            "Description": "Final VAS-coded rating (non-pain: 0–99 heat; pain: 100–200 pain)."
        },
        "vas_scale_min": {"Description": "VAS scale minimum for this trial."},
        "vas_scale_max": {"Description": "VAS scale maximum for this trial."},
        "trial_number": {"Description": "Within-run trial index (1-based)."},
        "run_id": {"Description": "Run index (1-based; matches PsychoPy run_id)."},
    }
    _safe_write_json(out, schema)
    return out


def run_fmri_raw_to_bids(
    *,
    source_root: Path,
    bids_fmri_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    session: Optional[str] = None,
    rest_task: str = "rest",
    include_rest: bool = True,
    include_fieldmaps: bool = True,
    dicom_mode: str = "symlink",  # "symlink" | "copy" | "skip"
    overwrite: bool = False,
    create_events: bool = True,
    event_granularity: str = "phases",  # "trial" | "phases"
    onset_reference: str = "as_is",  # "as_is" | "first_iti_start" | "first_stim_start"
    onset_offset_s: float = 0.0,
    dcm2niix_path: Optional[str] = None,
    dcm2niix_extra_args: Optional[Sequence[str]] = None,
    _logger: Optional[logging.Logger] = None,
) -> int:
    """Convert raw DICOM series under `<source_root>/sub-*/fmri` to BIDS fMRI."""
    log = _logger or logger

    ensure_dataset_description(bids_fmri_root, name=f"{task} fMRI")

    if subjects:
        subj_dirs = [source_root / f"sub-{s}" for s in subjects]
    else:
        subj_dirs = sorted([p for p in source_root.glob("sub-*") if p.is_dir()])

    subject_labels = [p.name.replace("sub-", "") for p in subj_dirs]
    _ensure_participants_tsv(bids_fmri_root, subject_labels)

    n_written = 0
    for subj_dir in subj_dirs:
        subject = subj_dir.name.replace("sub-", "")
        fmri_dicom_root = subj_dir / "fmri"
        if not fmri_dicom_root.exists():
            log.warning("No fMRI DICOM directory for %s: %s", subj_dir.name, fmri_dicom_root)
            continue

        bids_subj_dir = _bids_subject_dir(bids_fmri_root, subject, session)
        sourcedata_target = bids_fmri_root / "sourcedata" / subj_dir.name / "fmri"
        _copy_or_link_tree(fmri_dicom_root, sourcedata_target, mode=dicom_mode, overwrite=overwrite)

        series = discover_series(fmri_dicom_root)
        if not series:
            log.warning("No recognized series under: %s", fmri_dicom_root)
            continue

        created_func_relpaths: list[str] = []
        fmap_json_paths: list[Path] = []

        for s in series:
            if s.kind == "rest" and not include_rest:
                continue
            if s.kind == "fmap" and not include_fieldmaps:
                continue

            with tempfile.TemporaryDirectory(prefix="dcm2niix_") as tmp:
                tmp_dir = Path(tmp)
                _run_dcm2niix(
                    s.dicom_dir,
                    tmp_dir,
                    dcm2niix_path=dcm2niix_path,
                    extra_args=dcm2niix_extra_args,
                    _logger=log,
                )
                if s.kind == "fmap":
                    dest_dir = bids_subj_dir / "fmap"
                    run_label = _format_run(s.run)
                    dest_prefix = f"sub-{subject}"
                    if session:
                        dest_prefix += f"_ses-{session}"
                    if run_label:
                        dest_prefix += f"_run-{run_label}"

                    mapped, _jsons = _classify_fieldmap_outputs(tmp_dir)
                    all_tes: List[float] = []
                    for _suf, _n, _js in mapped:
                        if _js.exists():
                            _m = _load_json(_js)
                            for _k in ("EchoTime", "EchoTime1", "EchoTime2"):
                                _v = _m.get(_k)
                                if _v is not None:
                                    try:
                                        all_tes.append(float(_v))
                                    except (TypeError, ValueError):
                                        pass
                    all_tes = sorted(set(all_tes))

                    for suffix, nifti, js in mapped:
                        dest_stem = f"{dest_prefix}_{suffix}"
                        dest_nifti = dest_dir / f"{dest_stem}.nii.gz"
                        dest_json = dest_dir / f"{dest_stem}.json"

                        if dest_nifti.exists() and not overwrite:
                            log.info("Exists (skip): %s", dest_nifti)
                            continue
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(nifti), str(dest_nifti))
                        if js.exists():
                            shutil.move(str(js), str(dest_json))
                        else:
                            _safe_write_json(dest_json, {})
                        if suffix == "phasediff":
                            meta = _load_json(dest_json)
                            if ("EchoTime1" not in meta or "EchoTime2" not in meta) and len(all_tes) >= 2:
                                meta["EchoTime1"] = all_tes[0]
                                meta["EchoTime2"] = all_tes[-1]
                                _safe_write_json(dest_json, meta)
                        fmap_json_paths.append(dest_json)
                        n_written += 1
                    continue

                primary = _choose_primary_nifti(tmp_dir)
                primary_json = _sidecar_json_for_nifti(primary)

                if s.kind == "t1w":
                    dest_dir = bids_subj_dir / "anat"
                    dest_stem = f"sub-{subject}"
                    if session:
                        dest_stem += f"_ses-{session}"
                    dest_stem += "_T1w"
                    dest_nifti = dest_dir / f"{dest_stem}.nii.gz"
                    dest_json = dest_dir / f"{dest_stem}.json"

                elif s.kind in {"bold", "rest"}:
                    dest_dir = bids_subj_dir / "func"
                    run_label = _format_run(s.run)
                    dest_stem = f"sub-{subject}"
                    if session:
                        dest_stem += f"_ses-{session}"
                    task_label = task if s.kind == "bold" else rest_task
                    dest_stem += f"_task-{task_label}"
                    if run_label:
                        dest_stem += f"_run-{run_label}"
                    dest_stem += "_bold"
                    dest_nifti = dest_dir / f"{dest_stem}.nii.gz"
                    dest_json = dest_dir / f"{dest_stem}.json"

                else:
                    log.warning("Skipping unhandled series kind=%s: %s", s.kind, s.dicom_dir)
                    continue

                if dest_nifti.exists() and not overwrite:
                    log.info("Exists (skip): %s", dest_nifti)
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(primary), str(dest_nifti))
                if primary_json.exists():
                    shutil.move(str(primary_json), str(dest_json))
                else:
                    _safe_write_json(dest_json, {})

                # Update JSON metadata for BIDS downstream tools.
                meta = _load_json(dest_json)
                if s.kind in {"bold", "rest"}:
                    meta["TaskName"] = task if s.kind == "bold" else rest_task
                    _safe_write_json(dest_json, meta)
                    created_func_relpaths.append(str(dest_nifti.relative_to(bids_subj_dir)))
                n_written += 1

                # Events for pain runs: use PsychoPy TrialSummary for matching run.
                if create_events and s.kind == "bold" and s.run is not None:
                    beh = find_behavior_csv_for_run(
                        subj_dir,
                        run=s.run,
                        behavior_dir_name="PsychoPy_Data",
                        glob_pattern="*TrialSummary.csv",
                    )
                    if beh and beh.exists():
                        ev_tsv = dest_dir / f"{dest_stem}_events.tsv"
                        _write_events_tsv_for_run(
                            behavior_csv=beh,
                            out_tsv=ev_tsv,
                            run_id=int(s.run),
                            onset_reference=onset_reference,
                            onset_offset_s=float(onset_offset_s),
                            event_granularity=event_granularity,
                            bold_nifti=dest_nifti,
                            bold_json=dest_json,
                        )
                    else:
                        log.warning("No TrialSummary.csv found for %s run %s", subj_dir.name, s.run)

        # Add IntendedFor to fieldmap JSON (best-effort: link to all func in this session).
        if created_func_relpaths and fmap_json_paths:
            for fmap_json in fmap_json_paths:
                meta = _load_json(fmap_json)
                meta["IntendedFor"] = created_func_relpaths
                _safe_write_json(fmap_json, meta)

    if create_events:
        ensure_task_events_json(bids_fmri_root, task=task)

    log.info("Done. Converted %d series into BIDS: %s", n_written, bids_fmri_root)
    return n_written
