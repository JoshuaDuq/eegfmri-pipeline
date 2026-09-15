import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "trim_brainvision_to_volume_bounds.py"


def _load_trim_module():
    spec = importlib.util.spec_from_file_location("trim_brainvision_to_volume_bounds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_triplet(
    directory: Path,
    stem: str,
    *,
    n_samples: int,
    volume_samples: list[int],
    extra_markers: list[tuple[str, str, int]] | None = None,
    n_channels: int = 2,
) -> Path:
    header = directory / f"{stem}.vhdr"
    marker = directory / f"{stem}.vmrk"
    data = directory / f"{stem}.eeg"
    header.write_text(
        "\n".join(
            [
                "BrainVision Data Exchange Header File Version 1.0",
                "",
                "[Common Infos]",
                "Codepage=UTF-8",
                f"DataFile={data.name}",
                f"MarkerFile={marker.name}",
                "DataFormat=BINARY",
                "DataOrientation=MULTIPLEXED",
                f"NumberOfChannels={n_channels}",
                "SamplingInterval=1000",
                "",
                "[Binary Infos]",
                "BinaryFormat=INT_16",
                "",
                "[Channel Infos]",
                "Ch1=Fz,,1,µV",
                "Ch2=Cz,,1,µV",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "BrainVision Data Exchange Marker File, Version 1.0",
        "",
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={data.name}",
        "",
        "[Marker Infos]",
        "Mk1=New Segment,,1,1,0,20260209111144706000",
    ]
    number = 2
    for position in volume_samples:
        lines.append(f"Mk{number}=Volume,V  1,{position},1,0")
        number += 1
    for marker_type, description, position in extra_markers or []:
        lines.append(f"Mk{number}={marker_type},{description},{position},1,0")
        number += 1
    marker.write_text("\n".join(lines) + "\n", encoding="utf-8")
    data.write_bytes(b"\x00" * (n_samples * n_channels * 2))
    return header


def test_trim_keeps_one_tr_after_the_last_volume(tmp_path: Path) -> None:
    module = _load_trim_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    header = _write_triplet(
        source,
        "run",
        n_samples=3000,
        volume_samples=[501, 1401],
        extra_markers=[("Stimulus", "S  2", 2001)],
    )
    plan = module.create_trim_plan(header, source, output)
    module.trim_recording(plan)

    assert plan.first_volume_sample == 501
    assert plan.last_volume_sample == 1401
    assert plan.tr_samples == 900
    assert plan.last_inclusive_sample == 2300
    assert plan.output_samples == 1800
    assert plan.output_data.stat().st_size == 1800 * 2 * 2

    markers = module.parse_markers(plan.output_marker, module.read_utf8_brainvision(plan.output_marker).text)
    volumes = [marker for marker in markers if marker.marker_type == "Volume"]
    assert volumes[0].position == 1
    assert volumes[-1].position == 901
    assert volumes[-1].position != plan.output_samples
    stimuli = [marker for marker in markers if marker.marker_type == "Stimulus"]
    assert stimuli[0].position == 1501


def test_trim_clips_when_recording_ends_inside_the_last_tr(tmp_path: Path) -> None:
    module = _load_trim_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    header = _write_triplet(source, "run", n_samples=1600, volume_samples=[501, 1401])
    plan = module.create_trim_plan(header, source, output)
    module.trim_recording(plan)

    assert plan.last_inclusive_sample == 1600
    assert plan.output_samples == 1100
    markers = module.parse_markers(plan.output_marker, module.read_utf8_brainvision(plan.output_marker).text)
    volumes = [marker for marker in markers if marker.marker_type == "Volume"]
    assert volumes[-1].position == 901
    assert plan.output_samples - volumes[-1].position + 1 == 200


def test_trim_keeps_only_the_first_scanner_block(tmp_path: Path) -> None:
    module = _load_trim_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    header = _write_triplet(
        source,
        "run",
        n_samples=8000,
        volume_samples=[501, 1401, 2301, 7001, 7901],
    )
    plan = module.create_trim_plan(header, source, output)
    module.trim_recording(plan)

    assert plan.last_volume_sample == 2301
    assert plan.last_inclusive_sample == 3200
    markers = module.parse_markers(plan.output_marker, module.read_utf8_brainvision(plan.output_marker).text)
    volumes = [marker for marker in markers if marker.marker_type == "Volume"]
    assert [marker.position for marker in volumes] == [1, 901, 1801]


def test_volume_blocks_split_on_a_scanner_gap() -> None:
    module = _load_trim_module()
    blocks = module.volume_blocks([1, 901, 1801, 10001, 10901], 1.5)
    assert blocks == [[1, 901, 1801], [10001, 10901]]
