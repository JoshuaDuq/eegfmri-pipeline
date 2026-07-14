from unittest.mock import Mock

from eeg_pipeline.utils.data.preprocessing import set_channel_types


def test_set_channel_types_explicitly_accepts_expected_unit_changes() -> None:
    raw = Mock()
    raw.ch_names = ["Cz", "ECG"]

    set_channel_types(raw)

    raw.set_channel_types.assert_called_once_with(
        {"ECG": "ecg"},
        on_unit_change="ignore",
    )
