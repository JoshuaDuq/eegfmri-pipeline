from __future__ import annotations

import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest
import subprocess
import sys
import warnings
from xml.etree import ElementTree

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.fmri_construct_models import GroupMapResult
from studies.pain_study.study1.figures.fmri_construct_validity import (
    FmriConstructValiditySummary,
)


def test_fmri_construct_figure_has_fixed_publication_structure() -> None:
    from studies.pain_study.study1.figures.fmri_construct_validity_plot import (
        build_fmri_construct_validity_figure,
    )

    summary = synthetic_summary()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        figure = build_fmri_construct_validity_figure(
            summary,
            load_study1_config(),
        )

    try:
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 112.0 / 25.4))
        text = "\n".join(label.get_text() for label in figure.texts)
        assert "a  Delivered temperature" in text
        assert "b  Subjective intensity beyond temperature" in text
        assert "z = −12, 0, 12, 24, 36, 48 mm" in text
        assert "two-sided voxelwise max-T FWE p < 0.05" in text
        assert "Preliminary cohort" in text
        colorbar_labels = {axis.get_xlabel() for axis in figure.axes if axis.get_xlabel()}
        assert "Mean BOLD signal change (% per 1 °C)" in colorbar_labels
        assert "Mean BOLD signal change (% per 10 rating points)" in colorbar_labels
        assert not any(
            label.get_text().startswith("z=") for axis in figure.axes for label in axis.texts
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        footnote = next(
            label
            for label in figure.texts
            if label.get_text().startswith("Unthresholded participant-mean")
        )
        footnote_box = footnote.get_window_extent(renderer)
        for axis in figure.axes:
            if axis.get_xlabel().startswith("Mean BOLD signal change"):
                assert not footnote_box.overlaps(axis.xaxis.label.get_window_extent(renderer))
    finally:
        plt.close(figure)


def test_fmri_construct_display_limits_are_symmetric_and_robust() -> None:
    from studies.pain_study.study1.figures.fmri_construct_validity_plot import (
        symmetric_display_limit,
    )

    image = nib.Nifti1Image(
        np.asarray([[[0.0, -1.0, 1.0, 2.0, 100.0]]], dtype=np.float32),
        np.eye(4),
    )

    limit = symmetric_display_limit(image, percentile=80.0)

    assert 1.0 < limit < 100.0
    assert np.isfinite(limit)


def test_fmri_construct_writer_creates_exact_reproducibility_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_fmri_construct_validity as module

    summary = synthetic_summary()
    monkeypatch.setattr(
        module,
        "build_fmri_construct_validity_summary",
        lambda **kwargs: summary,
    )

    outputs = module.write_fmri_construct_validity(
        task="thermalactive",
        config=load_study1_config(),
        output_path=tmp_path / "fmri_construct_validity.svg",
    )

    expected_root_files = {
        "fmri_construct_validity.svg",
        "fmri_temperature_mean_effect.nii.gz",
        "fmri_temperature_neg_log10_fwe_p.nii.gz",
        "fmri_rating_mean_effect.nii.gz",
        "fmri_rating_neg_log10_fwe_p.nii.gz",
        "fmri_construct_validity_subjects.tsv",
        "fmri_construct_validity_subjects.parquet",
        "fmri_construct_validity_design_audit.tsv",
        "fmri_construct_validity_design_audit.parquet",
        "fmri_construct_validity_peaks.tsv",
        "fmri_construct_validity_peaks.parquet",
        "fmri_construct_validity_provenance.json",
    }
    assert {path.name for path in tmp_path.iterdir() if path.is_file()} == expected_root_files
    assert len(list(outputs.subject_maps_dir.glob("*.nii.gz"))) == 8
    root = ElementTree.parse(outputs.svg).getroot()
    assert float(root.attrib["width"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        183.0, abs=0.01
    )
    assert float(root.attrib["height"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        112.0, abs=0.01
    )
    assert outputs.svg.read_text(encoding="utf-8").count("<text") > 0
    assert pd.read_csv(outputs.subjects_tsv, sep="\t")["subject_id"].tolist() == (
        summary.subjects["subject_id"].tolist()
    )
    provenance = json.loads(outputs.provenance_json.read_text(encoding="utf-8"))
    assert provenance["inference"]["n_permutations"] == 10000
    assert provenance["inference"]["random_state"] == 20260711
    assert set(provenance["output_sha256"]) == {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file() and path != outputs.provenance_json
    }


def test_fmri_construct_cli_help_has_no_runtime_warning(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MNE_DONTWRITE_HOME"] = "true"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "studies.pain_study.study1.figures.plot_fmri_construct_validity",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def synthetic_summary() -> FmriConstructValiditySummary:
    from nilearn.datasets import load_mni152_template

    template = load_mni152_template(resolution=2)
    shape = template.shape
    affine = template.affine
    temperature = _spatial_field(
        shape,
        affine,
        peaks=((40.0, -20.0, 30.0, 0.8), (-38.0, -18.0, 28.0, 0.7)),
    )
    rating = _spatial_field(
        shape,
        affine,
        peaks=((34.0, 18.0, 8.0, 0.6), (0.0, 18.0, 34.0, 0.5)),
    )
    group_maps = {
        "temperature": _group_result("temperature", temperature, affine),
        "rating": _group_result("rating", rating, affine),
    }
    subject_ids = [f"sub-{index:04d}" for index in range(4)]
    subjects = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "n_runs": 6,
            "temperature_estimable": True,
            "rating_estimable": True,
        }
    )
    design_audit = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "estimand": estimand,
                "target_column": (
                    "temperature_linear"
                    if estimand == "temperature"
                    else "rating_within_temperature"
                ),
            }
            for subject_id in subject_ids
            for estimand in ("temperature", "rating")
        ]
    )
    return FmriConstructValiditySummary(
        subjects=subjects,
        design_audit=design_audit,
        subject_effects={
            "temperature": tuple(group_maps["temperature"].mean_effect for _ in range(4)),
            "rating": tuple(group_maps["rating"].mean_effect for _ in range(4)),
        },
        group_maps=group_maps,
        n_subjects=4,
        article_ready=False,
    )


def _group_result(estimand: str, values: np.ndarray, affine: np.ndarray) -> GroupMapResult:
    significant = np.abs(values) >= 0.22
    corrected = np.where(significant, 2.0, 0.0)
    return GroupMapResult(
        estimand=estimand,
        mean_effect=nib.Nifti1Image(values.astype(np.float32), affine),
        neg_log10_fwe_p=nib.Nifti1Image(corrected.astype(np.float32), affine),
        significance_mask=nib.Nifti1Image(significant.astype(np.uint8), affine),
        peaks=pd.DataFrame(
            columns=(
                "estimand",
                "cluster_id",
                "sign",
                "peak_effect",
                "peak_x_mm",
                "peak_y_mm",
                "peak_z_mm",
                "n_voxels",
            )
        ),
        n_subjects=4,
    )


def _spatial_field(
    shape: tuple[int, ...],
    affine: np.ndarray,
    *,
    peaks: tuple[tuple[float, float, float, float], ...],
) -> np.ndarray:
    grid = np.indices(shape, dtype=float).reshape(3, -1).T
    assert np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])))
    world = grid * np.diag(affine[:3, :3]) + affine[:3, 3]
    values = np.zeros(len(world), dtype=float)
    for x, y, z, amplitude in peaks:
        distance_squared = np.sum((world - np.asarray([x, y, z])) ** 2, axis=1)
        values += amplitude * np.exp(-distance_squared / (2.0 * 8.0**2))
    return values.reshape(shape)
