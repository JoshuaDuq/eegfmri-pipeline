from __future__ import annotations

import re
from typing import Iterable, List, Sequence


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


_COMP_COR_RE = re.compile(r"^(?P<prefix>[atcw]_comp_cor)_(?P<idx>\d+)$")


def _pick_compcor_components(
    available_columns: Sequence[str],
    *,
    n: int,
) -> List[str]:
    """
    Choose CompCor components in a stable, fMRIPrep-friendly way.

    Preference order:
    1) a_comp_cor_XX (anatomical)
    2) t_comp_cor_XX (temporal)
    3) c_comp_cor_XX
    4) w_comp_cor_XX
    """
    by_prefix = {"a_comp_cor": [], "t_comp_cor": [], "c_comp_cor": [], "w_comp_cor": []}
    for c in available_columns:
        m = _COMP_COR_RE.match(c)
        if not m:
            continue
        pref = m.group("prefix")
        idx = int(m.group("idx"))
        if pref in by_prefix:
            by_prefix[pref].append((idx, c))

    for pref in ["a_comp_cor", "t_comp_cor", "c_comp_cor", "w_comp_cor"]:
        items = sorted(by_prefix[pref], key=lambda t: t[0])
        if items:
            return [c for _i, c in items[: max(0, int(n))]]
    return []


def select_fmriprep_confounds_columns(
    available_columns: Sequence[str],
    *,
    strategy: str = "auto",
    auto_compcor_n: int = 5,
) -> List[str]:
    """
    Select confound columns (by name) from an fMRIPrep confounds TSV header.

    This function is stdlib-only so it can be unit-tested without numpy/pandas.
    """
    strategy = str(strategy or "auto").strip().lower()
    if strategy in {"", "default"}:
        strategy = "auto"

    if strategy in {"none", "no", "off"}:
        return []

    avail = set(available_columns)

    motion6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    motion_derivs = [f"{c}_derivative1" for c in motion6]
    motion_power2 = [f"{c}_power2" for c in motion6]
    motion_derivs_power2 = [f"{c}_derivative1_power2" for c in motion6]

    base_cols: List[str] = []
    if strategy in {"motion6"}:
        base_cols = motion6
    elif strategy in {"motion12", "motion+derivs"}:
        base_cols = motion6 + motion_derivs
    elif strategy in {"motion24"}:
        base_cols = motion6 + motion_derivs + motion_power2 + motion_derivs_power2
    elif strategy in {"motion24+wmcsf", "motion24+wm_csf"}:
        base_cols = motion6 + motion_derivs + motion_power2 + motion_derivs_power2 + ["white_matter", "csf"]
    elif strategy in {"motion24+wmcsf+fd", "motion24+wm_csf+fd"}:
        base_cols = motion6 + motion_derivs + motion_power2 + motion_derivs_power2 + [
            "white_matter",
            "csf",
            "framewise_displacement",
        ]
    elif strategy in {"auto"}:
        # Prefer a widely used "24p + WM/CSF + FD" if present, otherwise fall back.
        if all(c in avail for c in motion6):
            base_cols = motion6 + motion_derivs
            if all(c in avail for c in motion_power2 + motion_derivs_power2):
                base_cols += motion_power2 + motion_derivs_power2
            if "white_matter" in avail:
                base_cols.append("white_matter")
            if "csf" in avail:
                base_cols.append("csf")
            if "framewise_displacement" in avail:
                base_cols.append("framewise_displacement")

            # Upgrade: include CompCor components by default (if present).
            if int(auto_compcor_n) > 0:
                base_cols += _pick_compcor_components(available_columns, n=int(auto_compcor_n))
        else:
            # Non-fMRIPrep-like: conservative fallback.
            base_cols = [c for c in ["csf", "white_matter", "framewise_displacement"] if c in avail]
    else:
        raise ValueError(
            f"Unsupported confounds_strategy '{strategy}'. "
            "Use one of: none, motion6, motion12, motion24, motion24+wmcsf, motion24+wmcsf+fd, auto."
        )

    outlier_cols = [
        c
        for c in available_columns
        if c.startswith("motion_outlier") or c.startswith("non_steady_state_outlier") or c.startswith("outlier")
    ]

    cols = [c for c in base_cols if c in avail]
    cols += [c for c in outlier_cols if c in avail and c not in cols]
    return _unique_preserve_order(cols)

