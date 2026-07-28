from __future__ import annotations

from pathlib import Path

from fmri_pipeline.analysis.report.assets import PlotAssets, discover_plot_assets


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_anatomical_background_is_preferred_over_the_boldref(tmp_path: Path) -> None:
    func = tmp_path / "preprocessed" / "fmri" / "sub-01" / "func"
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    _touch(func / "sub-01_task-rest_run-01_space-T1w_desc-preproc_boldref.nii.gz")
    expected = _touch(anat / "sub-01_desc-preproc_T1w.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.background == expected


def test_boldref_is_used_when_no_anatomical_is_present(tmp_path: Path) -> None:
    func = tmp_path / "preprocessed" / "fmri" / "sub-01" / "func"
    expected = _touch(
        func / "sub-01_task-rest_run-01_space-T1w_desc-preproc_boldref.nii.gz"
    )

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.background == expected


def test_mni_space_selects_the_mni_anatomical(tmp_path: Path) -> None:
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    _touch(anat / "sub-01_desc-preproc_T1w.nii.gz")
    expected = _touch(anat / "sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="mni"
    )
    assert assets.background == expected


def test_probseg_tissue_maps_are_collected_by_class(tmp_path: Path) -> None:
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    gm = _touch(anat / "sub-01_label-GM_probseg.nii.gz")
    wm = _touch(anat / "sub-01_label-WM_probseg.nii.gz")
    csf = _touch(anat / "sub-01_label-CSF_probseg.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.probseg == {"GM": gm, "WM": wm, "CSF": csf}


def test_dseg_is_found_when_probseg_is_absent(tmp_path: Path) -> None:
    anat = tmp_path / "preprocessed" / "fmri" / "sub-01" / "anat"
    expected = _touch(anat / "sub-01_desc-aseg_dseg.nii.gz")

    assets = discover_plot_assets(
        deriv_root=tmp_path, subject="sub-01", task="rest", space="native"
    )
    assert assets.probseg == {}
    assert assets.dseg == expected


def test_a_missing_derivative_tree_yields_empty_assets_rather_than_raising(
    tmp_path: Path,
) -> None:
    assets = discover_plot_assets(
        deriv_root=tmp_path / "absent", subject="sub-01", task="rest", space="native"
    )
    assert assets == PlotAssets(background=None, mask=None, probseg={}, dseg=None)
