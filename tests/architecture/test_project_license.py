from __future__ import annotations

import tomllib

from tests import REPO_ROOT


def _project_metadata() -> dict[str, object]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_project_is_consistently_licensed_under_gpl_v3() -> None:
    metadata = _project_metadata()
    project = metadata["project"]
    classifiers = project["classifiers"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert license_text.startswith("GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007")
    assert project["license"] == {"text": "GPL-3.0-only"}
    assert "License :: OSI Approved :: GNU General Public License v3 (GPLv3)" in classifiers
    assert all("MIT" not in classifier for classifier in classifiers)
    assert "license-GPL--3.0" in readme
    assert "GPL-3.0-only. See [LICENSE](LICENSE)." in readme


def test_neuxus_attribution_survives_the_code_it_described() -> None:
    """The adapted GPL source is gone from the tree but not from the history.

    The native EEG-fMRI correction package that vendored NeuXus was removed on 2026-07-30.
    Nothing may re-declare its model assets as shipped package data, and the notice must
    still carry the upstream provenance, because published history distributes that code.
    """
    metadata = _project_metadata()
    package_data = metadata["tool"]["setuptools"]["package-data"]["eeg_pipeline"]
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert not (REPO_ROOT / "eeg_pipeline" / "preprocessing" / "eeg_fmri").exists()
    assert all("eeg_fmri" not in entry for entry in package_data)
    assert "NeuXus" in notices
    assert "v0.0.4" in notices
    assert "10.1016/j.neuroimage.2023.120353" in notices
    assert "GPL-3.0-only" in notices
