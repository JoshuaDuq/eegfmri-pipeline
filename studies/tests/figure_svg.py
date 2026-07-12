"""Assertions for raster layers embedded in publication SVG files."""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from xml.etree import ElementTree

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def embedded_raster_dpi(svg_path: Path) -> tuple[tuple[float, float], ...]:
    """Return effective x/y DPI for every PNG embedded in an SVG."""

    root = ElementTree.parse(svg_path).getroot()
    resolutions = []
    for image in root.findall(f".//{{{SVG_NAMESPACE}}}image"):
        payload = base64.b64decode(image.attrib[XLINK_HREF].split(",", 1)[1])
        if not payload.startswith(PNG_SIGNATURE):
            raise ValueError("SVG test helper requires embedded PNG images.")
        width_px, height_px = struct.unpack(">II", payload[16:24])
        width_pt = float(image.attrib["width"])
        height_pt = float(image.attrib["height"])
        resolutions.append(
            (
                width_px / (width_pt / 72.0),
                height_px / (height_pt / 72.0),
            )
        )
    return tuple(resolutions)


__all__ = ["embedded_raster_dpi"]
