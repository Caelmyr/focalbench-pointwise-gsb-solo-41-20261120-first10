"""Raw-capture preflight separated from result-space focus measurements."""

from collections import Counter

import numpy as np
from scipy import ndimage

from .acquisition import read_capture_positions
from .focus import image_response
from .imaging import fit_size, percentile_summary, read_pixels
from .models import MAX_ACTIVE_FRAMES, MIN_FRAMES

INSPECTION_EDGE = 420
LIGHT_BLOCKS = (4, 6)


def _region_slices(shape: tuple[int, int], grid: tuple[int, int]):
    height, width = shape
    rows, columns = grid
    vertical = np.round(np.linspace(0, height, rows + 1)).astype(np.int32)
    horizontal = np.round(np.linspace(0, width, columns + 1)).astype(np.int32)
    for row in range(rows):
        for column in range(columns):
            yield row, column, (
                slice(vertical[row], vertical[row + 1]),
                slice(horizontal[column], horizontal[column + 1]),
            )


def _tile_statistics(
    plane: np.ndarray, response: np.ndarray, valid: np.ndarray, rgb: np.ndarray,
) -> list[dict]:
    tiles = []
    for row, column, viewport in _region_slices(plane.shape, LIGHT_BLOCKS):
        mask = valid[viewport]
        luminance = plane[viewport][mask]
        sharpness = response[viewport][mask]
        sample_rgb = rgb[viewport][mask]
        total = int(mask.size)
        tiles.append({
            "row": row,
            "column": column,
            "fraction": float(mask.sum() / max(1, total)),
            "median_light": float(np.median(luminance)) if luminance.size else None,
            "median_response": float(np.median(sharpness)) if sharpness.size else None,
            "white_fraction": float(np.mean(np.min(sample_rgb, axis=1) > 0.982)) if luminance.size else None,
        })
    return tiles


def capture_metrics(library, capture: dict) -> dict:
    size = fit_size((capture["width"], capture["height"]), INSPECTION_EDGE)
    pixels = read_pixels(library, capture, size)
    gray = pixels.luminance
    evidence, supported = image_response(gray, pixels.support, "balanced", 2)
    edge = np.sqrt(
        ndimage.sobel(gray, axis=0, mode="nearest") ** 2
        + ndimage.sobel(gray, axis=1, mode="nearest") ** 2
    ) / 8
    valid = int(pixels.support.sum())
    coverage = float(pixels.support.mean())
    lights = percentile_summary(gray, pixels.support)
    responses = percentile_summary(evidence, supported)
    gradients = percentile_summary(edge, supported)
    if valid:
        clipped = np.max(pixels.rgb[pixels.support], axis=1) >= 0.987
        deep = gray[pixels.support] <= 0.003
        clipping = float(clipped.mean())
        shadows = float(deep.mean())
    else:
        clipping = None
        shadows = None
    tiles = _tile_statistics(gray, evidence, supported, pixels.rgb)
    return {
        "eligible": bool(valid),
        "capture_id": capture["id"],
        "name": capture["name"],
        "position": capture["position"],
        "included": capture["included"],
        "size": [capture["width"], capture["height"]],
        "profile": capture["information"]["profile"],
        "coverage": coverage,
        "brightness": lights,
        "focus_response": responses,
        "gradient": gradients,
        "overexposed": clipping,
        "blocked_shadow": shadows,
        "tile_rows": LIGHT_BLOCKS[0],
        "tile_columns": LIGHT_BLOCKS[1],
        "tiles": tiles,
    }


def _metric_flags(
    metrics: dict, reference_brightness: float | None,
    canonical_size: tuple[int, int] | None,
) -> list[dict]:
    flags = []

    def note(level, code, detail):
        flags.append({"level": level, "code": code, "detail": detail})

    if not metrics["eligible"]:
        note("error", "no-coverage", "No opaque image samples")
        return flags
    if canonical_size is not None and tuple(metrics["size"]) != canonical_size:
        note("error", "geometry", "Oriented dimensions differ from the stack")
    if metrics["coverage"] < 0.8:
        note("warning", "alpha", "More than one fifth of the capture is transparent")
    if metrics["overexposed"] is not None and metrics["overexposed"] > 0.12:
        note("warning", "clipped", "More than 12% of visible pixels clip in at least one channel")
    if metrics["blocked_shadow"] is not None and metrics["blocked_shadow"] > 0.18:
        note("warning", "shadow", "More than 18% of visible pixels contain near-black luminance")
    p95 = metrics["focus_response"]["p95"]
    if p95 is None or p95 < 0.009:
        note("warning", "soft", "Few strong detail responses in this capture")
    brightness = metrics["brightness"]["median"]
    if reference_brightness is not None and brightness is not None:
        relative = (max(brightness, 0.001) / max(reference_brightness, 0.001))
        if relative < 0.65 or relative > 1.55:
            note("warning", "lighting", "Median luminance is far from the stack's middle value")
    return flags


def _typical_brightness(rows: list[dict]) -> float | None:
    samples = [row["brightness"]["median"] for row in rows if row["included"]
               and row["brightness"]["median"] is not None]
    return float(np.median(samples)) if samples else None


def _dimension_consensus(captures: list[dict]) -> tuple[int, int] | None:
    sizes = [(row["width"], row["height"]) for row in captures if row["included"]]
    if not sizes:
        return None
    domains = Counter(sizes)
    return domains.most_common(1)[0][0]


def stack_readiness(
    summary: dict, flagged: list[dict], enabled_count: int, size_consensus: tuple | None,
) -> list[dict]:
    blockers = []
    if enabled_count < MIN_FRAMES:
        blockers.append({"code": "too-few", "detail": f"Enable at least {MIN_FRAMES} captures."})
    if enabled_count > MAX_ACTIVE_FRAMES:
        blockers.append({"code": "too-many", "detail": f"Enable at most {MAX_ACTIVE_FRAMES} captures."})
    if summary["duplicates"]:
        blockers.append({"code": "position", "detail": "Set a distinct rail position for each enabled capture."})
    if size_consensus is not None and any(
        tuple(row["size"]) != size_consensus for row in flagged if row["included"]
    ):
        blockers.append({"code": "dimensions", "detail": "Enabled captures need matching oriented dimensions."})
    if any(row["included"] and not row["eligible"] for row in flagged):
        blockers.append({"code": "coverage", "detail": "At least one enabled capture has no opaque sample."})
    return blockers


def inspect_stack(library, captures: list[dict]) -> dict:
    active = [capture for capture in captures if not capture.get("archived")]
    rows = [capture_metrics(library, capture) for capture in active]
    mid = _typical_brightness(rows)
    reference_size = _dimension_consensus(active)
    for row in rows:
        row["flags"] = _metric_flags(row, mid, reference_size)
    summary = read_capture_positions(active)
    blockers = stack_readiness(summary, rows, summary["count"], reference_size)
    return {
        "can_reconstruct": not blockers,
        "blockers": blockers,
        "rail": summary,
        "reference_size": list(reference_size) if reference_size else None,
        "typical_brightness": mid,
        "warning_count": sum(len(row["flags"]) for row in rows),
        "captures": rows,
    }
