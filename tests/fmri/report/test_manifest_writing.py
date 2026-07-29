"""The analysis run must leave behind everything the report needs.

This is the seam. Until the fitting path writes a manifest, a report can only be
produced from inside the GLM, which is what made subject QC recompute per contrast
and what made iterating on a figure mean re-running the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fmri_pipeline.analysis.report.manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    write_report_manifest,
)


def _run_meta(**overrides) -> dict:
    base = dict(
        analysis_space="T1w",
        tr=2.0,
        confounds_strategy="24HMP+aCompCor",
        included_bold_paths=["/d/sub-01_run-01_bold.nii.gz"],
        included_confounds_paths=["/d/sub-01_run-01_confounds.tsv"],
        n_runs_included=1,
        skipped_runs=[{"run_index": 2, "reason": "mean FD 0.9 mm"}],
    )
    base.update(overrides)
    return base


def test_a_fitted_contrast_leaves_a_manifest_beside_its_stat_maps(
    tmp_path: Path,
) -> None:
    contrast_dir = tmp_path / "contrast-heatgtwarm"
    contrast_dir.mkdir(parents=True)
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="heat>warm",
        stat_map=stat_map,
        run_meta=_run_meta(),
        smoothing_fwhm=6.0,
        signal_scaling=True,
    )

    assert written.name == MANIFEST_FILENAME
    assert written.parent == contrast_dir

    restored = read_manifest(written)
    assert restored.subject == "sub-01"
    assert restored.task == "heat"
    assert restored.contrast_name == "heat>warm"
    assert restored.t_r == 2.0
    assert restored.smoothing_fwhm == 6.0
    assert restored.signal_scaling is True
    assert restored.confound_strategy == "24HMP+aCompCor"


def test_the_manifest_records_why_a_run_was_excluded(tmp_path: Path) -> None:
    """A report that says a run was dropped without saying why gives nothing to act on."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        run_meta=_run_meta(
            skipped_runs=[
                {"run_index": 3, "reason": "no events"},
                {"run_index": 4, "reason": "confounds file missing"},
            ]
        ),
    )
    excluded = read_manifest(written).excluded_runs
    assert len(excluded) == 2
    assert excluded[0][1] == "no events"
    assert "run" in excluded[0][0]


def test_the_space_comes_from_what_was_actually_fit(tmp_path: Path) -> None:
    """A manifest that misreports its space sends the glass brain the wrong way."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    mni = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        run_meta=_run_meta(analysis_space="MNI152NLin2009cAsym"),
    )
    assert read_manifest(mni).space == "mni"

    native = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        run_meta=_run_meta(analysis_space="T1w"),
    )
    assert read_manifest(native).space == "native"


def test_included_run_labels_are_derived_from_the_bold_filenames(
    tmp_path: Path,
) -> None:
    """The header names runs, so they need names a reader recognises."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        run_meta=_run_meta(
            included_bold_paths=[
                "/d/sub-01_task-heat_run-01_bold.nii.gz",
                "/d/sub-01_task-heat_run-02_bold.nii.gz",
            ],
            included_confounds_paths=[None, None],
        ),
    )
    assert read_manifest(written).included_runs == ("run-01", "run-02")


def test_a_missing_confounds_entry_does_not_become_a_path(tmp_path: Path) -> None:
    """nilearn allows a run without confounds; None must not serialise as 'None'."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        run_meta=_run_meta(included_confounds_paths=[None]),
    )
    assert read_manifest(written).confounds_paths == ()


def test_the_manifest_is_written_even_without_a_tr(tmp_path: Path) -> None:
    """An unknown TR is a gap in the metadata, not a reason to write nothing."""
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        run_meta=_run_meta(tr=None),
    )
    assert read_manifest(written).t_r is None


def test_writing_a_manifest_never_raises_out_of_the_analysis_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed manifest must not lose a fitted GLM.

    The model has already run and its maps are on disk by this point. Losing that
    to a reporting-metadata problem would be the most expensive possible failure.
    """
    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    stat_map = contrast_dir / "z.nii.gz"
    stat_map.touch()

    written = write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="c",
        stat_map=stat_map,
        run_meta="not a mapping at all",  # type: ignore[arg-type]
    )
    assert written is None
