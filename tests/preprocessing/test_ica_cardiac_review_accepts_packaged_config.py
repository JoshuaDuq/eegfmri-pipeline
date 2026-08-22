# The reader of ica.cardiac_review must accept the config the package ships.
#
# ``from_mapping`` rejects any key it does not name, which catches typos. But
# ``marker_ctps_qc`` is a real sibling key under the same mapping, read by
# ``PreprocessingPipeline._get_marker_ctps_qc_config`` and required there, so the two
# readers have to agree about what may appear. When they drifted apart, every
# ``preprocessing ica`` run reached the cardiac review and died on the packaged
# defaults.

from __future__ import annotations

from eeg_pipeline.preprocessing.ica_cardiac_review import CardiacReviewSettings
from eeg_pipeline.utils.config.loader import load_config

from tests import REPO_ROOT

PACKAGED_CONFIG = REPO_ROOT / "eeg_pipeline" / "utils" / "config" / "eeg_config.yaml"


def test_from_mapping_accepts_the_packaged_cardiac_review_settings():
    config = load_config(PACKAGED_CONFIG)

    settings = CardiacReviewSettings.from_mapping(config.get("ica.cardiac_review"))

    assert settings.ecg_channel == "ECG"


def test_marker_ctps_qc_is_present_to_be_accepted():
    """Guards the test above from passing because the key quietly disappeared."""
    config = load_config(PACKAGED_CONFIG)

    assert config.get("ica.cardiac_review.marker_ctps_qc")
