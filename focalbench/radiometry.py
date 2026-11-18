"""Inspect clipping and luminance distribution directly in an oriented capture."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .imaging import linear_luminance, linear_to_srgb, percentile_summary, read_pixels
from .models import Region

CLIP_THRESHOLD = 0.985
SHADOW_THRESHOLD = 0.003
HISTOGRAM_BINS = 64


@dataclass(frozen=True)
class MeasurementPatch:
    rgb: np.ndarray
    support: np.ndarray
    left: int
    top: int

    @property
    def width(self) -> int:
        return self.support.shape[1]

    @property
    def height(self) -> int:
        return self.support.shape[0]

    @property
    def size(self) -> int:
        return self.support.size


def _inspection_size(width: int, height: int, budget: int = 1_600_000):
    area = width * height
    scale = 1 if area <= budget else (budget / area) ** 0.5
    return max(1, round(width * scale)), max(1, round(height * scale))


def source_patch(library, capture: dict, area: Region | None):
    size = _inspection_size(capture["width"], capture["height"])
    source = read_pixels(library, capture, size)
    left, top, right, bottom = area.pixels(*size) if area else (0, 0, *size)
    rows = slice(top, bottom)
    cols = slice(left, right)
    sampled = MeasurementPatch(source.rgb[rows, cols], source.support[rows, cols], left, top)
    return sampled, size


def histogram(values: np.ndarray) -> list[int]:
    bucket = np.clip((values * HISTOGRAM_BINS).astype(np.int32), 0, HISTOGRAM_BINS - 1)
    return np.bincount(bucket, minlength=HISTOGRAM_BINS).tolist()


def _clipping_grain(clipped: np.ndarray) -> tuple[int, int]:
    labels, count = ndimage.label(clipped, structure=ndimage.generate_binary_structure(2, 1))
    if count == 0:
        return 0, 0
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return int(areas.max()), int(np.count_nonzero(areas >= 6))


def capture_histogram(library, capture: dict, area: Region | None) -> dict:
    patch, size = source_patch(library, capture, area)
    luminance = linear_luminance(patch.rgb)
    included = patch.support
    max_channel = np.max(patch.rgb, axis=2)
    high = (max_channel >= CLIP_THRESHOLD) & included
    low = (luminance <= SHADOW_THRESHOLD) & included
    diffuse_clipping = np.all(patch.rgb >= CLIP_THRESHOLD, axis=2) & included
    span = percentile_summary(luminance, included)
    count = int(included.sum())
    biggest, patches = _clipping_grain(high)
    channel_rows = []
    for band, label in enumerate(("red", "green", "blue")):
        channel = patch.rgb[..., band]
        values = channel[included]
        channel_rows.append({
            "color": label,
            "bins": histogram(values),
            "stats": percentile_summary(channel, included),
            "clipped_pixels": int(np.count_nonzero(values >= CLIP_THRESHOLD)),
        })
    return {
        "capture": capture["name"],
        "oriented_size": [capture["width"], capture["height"]],
        "measurement_size": list(size),
        "selection": None if area is None else area.public(),
        "bounds": [patch.left, patch.top, patch.left + patch.width, patch.top + patch.height],
        "total_pixels": patch.size,
        "covered_pixels": count,
        "luminance_histogram": histogram(luminance[included]),
        "luminance": span,
        "channels": channel_rows,
        "clipping_pixels": int(high.sum()),
        "clipping_all_channels": int(diffuse_clipping.sum()),
        "blocked_shadows": int(low.sum()),
        "clipping_area": biggest,
        "clipping_patches": patches,
        "histogram_space": "linear sRGB",
        "white_signal": CLIP_THRESHOLD,
        "shadow_signal": SHADOW_THRESHOLD,
    }


def exposure_map(library, capture: dict, mode: str, area: Region | None) -> Image.Image:
    patch, _ = source_patch(library, capture, area)
    luminance = linear_luminance(patch.rgb)
    neutral = linear_to_srgb(luminance * 0.56)
    output = np.stack((neutral, neutral, neutral), axis=2)
    if mode in ("clipped", "both"):
        saturated = patch.rgb >= CLIP_THRESHOLD
        for band, color in enumerate(((229, 111, 114), (116, 220, 144), (114, 141, 230))):
            single = saturated[..., band] & (saturated.sum(axis=2) == 1)
            output[single] = color
        mixed = saturated.sum(axis=2) == 2
        output[mixed] = (244, 177, 95)
        output[mixed & ~saturated[..., 0]] = (113, 218, 228)
        output[mixed & ~saturated[..., 1]] = (220, 126, 222)
        output[saturated.all(axis=2)] = (255, 244, 190)
    if mode in ("shadow", "both"):
        output[luminance <= SHADOW_THRESHOLD] = (103, 110, 234)
    alpha = (patch.support * 255).astype(np.uint8)
    return Image.fromarray(np.concatenate((output, alpha[..., None]), axis=2))
