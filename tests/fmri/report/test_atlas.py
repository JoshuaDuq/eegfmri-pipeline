"""Anatomical names for cluster peaks, and the space that makes them meaningful.

An MNI atlas read at a native-space coordinate returns the name of whatever structure
sits at those millimetres in a different brain. The output is indistinguishable from a
correct label, so the gate has to be tested rather than assumed.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report import atlas


@pytest.fixture()
def parcels(tmp_path):
    """Two parcels split at x = 0 on a 3 mm grid, with a name table."""
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[:5] = 1
    data[5:] = 2
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    affine[:3, 3] = [-15.0, -15.0, -15.0]
    image_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), image_path)

    table_path = tmp_path / "labels.tsv"
    pd.DataFrame({"index": [1, 2], "name": ["Left thing", "Right thing"]}).to_csv(
        table_path, sep="\t", index=False
    )
    return image_path, table_path


# --- the gate -------------------------------------------------------------


def test_only_mni_coordinates_may_be_labelled() -> None:
    assert atlas.atlas_applies_to("mni")
    assert atlas.atlas_applies_to("MNI")
    assert not atlas.atlas_applies_to("native")
    assert not atlas.atlas_applies_to("T1w")
    assert not atlas.atlas_applies_to("")


def test_an_unlabelled_table_says_why_it_is_unlabelled() -> None:
    # Silence is indistinguishable from an atlas that failed to load.
    message = atlas.space_refusal("native")
    assert "MNI" in message and "native" in message


# --- labelling ------------------------------------------------------------


def test_a_peak_gets_the_name_of_the_parcel_it_falls_in(parcels) -> None:
    image_path, table_path = parcels
    labeller = atlas.load_atlas(labels_img=image_path, labels_tsv=table_path)
    assert labeller.label_at((-9.0, 0.0, 0.0)) == "Left thing"
    assert labeller.label_at((9.0, 0.0, 0.0)) == "Right thing"


def test_a_peak_outside_the_atlas_has_no_name(parcels) -> None:
    # Not the nearest parcel: a guess the table could not be checked against.
    image_path, table_path = parcels
    labeller = atlas.load_atlas(labels_img=image_path, labels_tsv=table_path)
    assert labeller.label_at((900.0, 0.0, 0.0)) is None


def test_a_peak_in_the_atlas_background_has_no_name(tmp_path) -> None:
    data = np.zeros((6, 6, 6), dtype=np.int16)
    path = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    labeller = atlas.load_atlas(labels_img=path)
    assert labeller.label_at((1.0, 1.0, 1.0)) is None


def test_without_a_name_table_the_parcel_index_is_reported(parcels) -> None:
    # Still more than a coordinate: two peaks sharing an index are in one parcel.
    image_path, _table = parcels
    labeller = atlas.load_atlas(labels_img=image_path)
    assert labeller.label_at((9.0, 0.0, 0.0)) == "2"


def test_labelling_a_sequence_preserves_order_and_gaps(parcels) -> None:
    image_path, table_path = parcels
    labeller = atlas.load_atlas(labels_img=image_path, labels_tsv=table_path)
    labels = labeller.label_all([(-9.0, 0.0, 0.0), (900.0, 0.0, 0.0), (9.0, 0.0, 0.0)])
    assert labels == ("Left thing", None, "Right thing")


def test_a_bids_style_table_with_a_label_column_is_accepted(parcels, tmp_path) -> None:
    image_path, _table = parcels
    bids = tmp_path / "dseg.tsv"
    pd.DataFrame({"index": [1, 2], "label": ["A", "B"]}).to_csv(
        bids, sep="\t", index=False
    )
    labeller = atlas.load_atlas(labels_img=image_path, labels_tsv=bids)
    assert labeller.label_at((-9.0, 0.0, 0.0)) == "A"


# --- declining to label ---------------------------------------------------


def test_no_configured_atlas_is_not_an_error() -> None:
    assert atlas.load_atlas(labels_img=None) is None
    assert atlas.load_atlas(labels_img="") is None


def test_a_missing_atlas_costs_the_column_and_not_the_report(tmp_path) -> None:
    assert atlas.load_atlas(labels_img=tmp_path / "absent.nii.gz") is None


def test_an_unreadable_name_table_falls_back_to_indices(parcels, tmp_path) -> None:
    # The atlas still names parcels by number; losing the table must not lose them.
    image_path, _table = parcels
    labeller = atlas.load_atlas(
        labels_img=image_path, labels_tsv=tmp_path / "absent.tsv"
    )
    assert labeller is not None
    assert labeller.label_at((-9.0, 0.0, 0.0)) == "1"


def test_the_atlas_names_its_own_source(parcels) -> None:
    # The caption states which atlas produced the column; a name that came from
    # nowhere cannot be reproduced or disputed.
    image_path, table_path = parcels
    labeller = atlas.load_atlas(labels_img=image_path, labels_tsv=table_path)
    assert labeller.source == "atlas.nii.gz"


def test_the_gate_follows_the_coordinates_not_the_fit():
    """A native fit with an MNI companion yields MNI coordinates, so labels apply.

    The gate is on the space the *table's coordinates* are in, which is the space of
    whichever map the table was built from -- not on the space the report as a whole
    was fitted in.
    """
    assert atlas.atlas_applies_to("mni")
    assert not atlas.atlas_applies_to("native")
