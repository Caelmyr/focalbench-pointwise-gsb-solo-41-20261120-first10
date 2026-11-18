"""Localize holes and unreliable patches in an archived reconstruction."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .imaging import percentile_summary
from .models import RUN_ASSETS, Region
from .surface import Inspection


@dataclass(frozen=True)
class Deficit:
    label: int
    count: int
    left: int
    top: int
    right: int
    bottom: int
    center: tuple[float, float]
    reason: str
    contributors: dict
    confidence: dict

    def public(self, frame) -> dict:
        return {
            "id": self.label,
            "pixels": self.count,
            "left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom,
            "fraction_of_output": self.count / (frame.width * frame.height),
            "area_um2": None if frame.pixel_area_um2 is None else self.count * frame.pixel_area_um2,
            "center": frame.center(*self.center),
            "touches_output_edge": (
                self.left == 0 or self.top == 0 or self.right == frame.width or self.bottom == frame.height
            ),
            "reason": self.reason,
            "contributors": self.contributors,
            "confidence": self.confidence,
        }


def _region_slices(inspection: Inspection, region: Region | None):
    frame = inspection.frame
    if region is None:
        return slice(0, frame.height), slice(0, frame.width)
    left, top, right, bottom = region.pixels(frame.width, frame.height)
    return slice(top, bottom), slice(left, right)


def _type_mask(inspection: Inspection, viewport, kind: str) -> np.ndarray:
    present = inspection.valid[viewport]
    good = inspection.reliable[viewport]
    if kind == "missing":
        return ~present
    if kind == "weak":
        return present & ~good
    return ~good


def _dominant_reason(present: np.ndarray, contributors: np.ndarray) -> str:
    if not np.any(present):
        return "Outside aligned coverage"
    if float(np.mean(~present)) >= 0.5:
        return "Mostly outside aligned coverage"
    if float(np.mean(contributors <= 1)) >= 0.5:
        return "Too few overlapping captures"
    return "Weak or ambiguous focus response"


def deficit_components(
    inspection: Inspection, kind: str, minimum: int, *, region: Region | None = None,
    limit: int = 16,
) -> tuple[dict, np.ndarray, tuple[slice, slice]]:
    viewport = _region_slices(inspection, region)
    mask = _type_mask(inspection, viewport, kind)
    groups, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1))
    boxes = ndimage.find_objects(groups, max_label=count)
    totals = np.bincount(groups.ravel(), minlength=count + 1)
    indices = np.flatnonzero(totals >= minimum)
    kept = [int(index) for index in indices if index]
    kept.sort(key=lambda index: (-totals[index], index))
    launch = []
    for index in kept[:limit]:
        vertical, horizontal = boxes[index - 1]
        section = groups[vertical, horizontal] == index
        canvas = (
            slice(viewport[0].start + vertical.start, viewport[0].start + vertical.stop),
            slice(viewport[1].start + horizontal.start, viewport[1].start + horizontal.stop),
        )
        present = inspection.valid[canvas] & section
        counts = inspection.contributors[canvas]
        details = inspection.confidence[canvas]
        cy, cx = ndimage.center_of_mass(section)
        coordinates = (canvas[1].start + cx, canvas[0].start + cy)
        deficit = Deficit(
            index, int(totals[index]), canvas[1].start, canvas[0].start,
            canvas[1].stop, canvas[0].stop, coordinates,
            _dominant_reason(present[section], counts[section]),
            percentile_summary(counts, section), percentile_summary(details, present),
        )
        launch.append(deficit.public(inspection.frame))
    return ({
        "kind": kind,
        "region": None if region is None else region.public(),
        "selected_pixels": int(mask.sum()),
        "analyzed_pixels": int(mask.size),
        "kept_regions": len(kept),
        "kept_pixels": int(totals[kept].sum()) if kept else 0,
        "minimum_region_pixels": minimum,
        "components": launch,
    }, groups, viewport)


def coverage_report(inspection: Inspection, kind: str, minimum: int, region: Region | None = None) -> dict:
    report, _, _ = deficit_components(inspection, kind, minimum, region=region)
    return report


def show_deficits(
    inspection: Inspection, kind: str, minimum: int, region: Region | None = None,
) -> Image.Image:
    report, groups, viewport = deficit_components(inspection, kind, minimum, region=region)
    img_path = inspection.directory / RUN_ASSETS["composite"]
    with Image.open(img_path) as original:
        samples = np.array(original.convert("RGB"), dtype=np.float32)
    selected = report["components"]
    lum = (0.2126 * samples[..., 0] + 0.7152 * samples[..., 1] + 0.0722 * samples[..., 2])
    dimmed = np.broadcast_to(lum[..., None], samples.shape) * 0.58
    palette = ((231, 145, 80), (222, 125, 108), (219, 157, 212), (231, 195, 109))
    canvas = dimmed[viewport]
    for index, component in enumerate(selected):
        masked = groups == component["id"]
        color = np.asarray(palette[index % len(palette)])
        canvas[masked] = canvas[masked] * 0.44 + color * 0.56
    image = Image.fromarray(np.clip(dimmed, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for rank, area in enumerate(selected, 1):
        color = palette[(rank - 1) % len(palette)]
        box = (area["left"], area["top"], area["right"] - 1, area["bottom"] - 1)
        draw.rectangle(box, outline=color, width=1)
        x, y = area["left"], max(0, area["top"] - 14)
        draw.rectangle((x, y, x + 21, y + 14), fill=(20, 22, 28))
        draw.text((x + 3, y), str(rank), fill=color)
    return image
