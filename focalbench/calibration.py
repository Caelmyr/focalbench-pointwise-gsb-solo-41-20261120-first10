"""Measure image-plane scale from known spans, keeping archived results immutable."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .models import Conflict, RunSettings, Unprocessable, choice, number, object_value, parse_points_query
from .models import short_text, validate_keys
from .surface import CoordinateFrame

MIN_SPAN_PIXELS = 3.0


@dataclass(frozen=True)
class KnownSpan:
    name: str
    points: tuple
    length_um: float
    tolerance_um: float

    @classmethod
    def parse(cls, value: dict, index: int):
        raw = object_value(value, "scale span")
        validate_keys(raw, {"name", "points", "length", "unit", "tolerance"}, "scale span")
        label = short_text(raw.get("name", f"Span {index + 1}"), "span label", maximum=64)
        unit = choice(raw.get("unit", "µm"), "known length unit", ("µm", "mm"))
        factor = 1000 if unit == "mm" else 1
        length = number(raw.get("length"), "known length", 0.001, 1_000_000)
        tolerance = number(raw.get("tolerance", 0), "length tolerance", 0, length * 0.5)
        return cls(
            label, tuple(parse_points_query(raw.get("points"))),
            length * factor, tolerance * factor,
        )


def parse_spans(value: object) -> list[KnownSpan]:
    if not isinstance(value, list) or not 1 <= len(value) <= 12:
        raise Unprocessable("Use between 1 and 12 marked calibration spans.")
    return [KnownSpan.parse(span, index) for index, span in enumerate(value)]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    accumulated = np.cumsum(weights[order])
    pivot = np.searchsorted(accumulated, accumulated[-1] * 0.5)
    return float(values[order[pivot]])


def assess_span(span: KnownSpan, frame: CoordinateFrame) -> dict:
    distance = frame.distance(*span.points)
    if distance < MIN_SPAN_PIXELS:
        raise Unprocessable(f"{span.name} spans fewer than {MIN_SPAN_PIXELS:g} original pixels.")
    value = span.length_um / distance
    endpoint_error = math.hypot(frame.height_ratio, frame.width_ratio) * 0.5
    coordinate_error = math.sqrt(2) * endpoint_error * value
    combined = math.hypot(span.tolerance_um, coordinate_error) / distance
    start, end = (frame.reference_point(point) for point in span.points)
    orientation_deg = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    return {
        "name": span.name,
        "original_start": start,
        "original_end": end,
        "angle_deg": orientation_deg,
        "original_distance_px": distance,
        "known_um": span.length_um,
        "tolerance_um": span.tolerance_um,
        "estimated_um_per_px": value,
        "single_span_uncertainty_um_per_px": combined,
    }


def project_estimate(run: dict, spans: list[KnownSpan]) -> dict:
    frame = CoordinateFrame.from_summary(run["summary"])
    estimates = [assess_span(span, frame) for span in spans]
    ratios = np.asarray([row["estimated_um_per_px"] for row in estimates], dtype=np.float64)
    errors = np.asarray([row["single_span_uncertainty_um_per_px"] for row in estimates], dtype=np.float64)
    weights = 1.0 / np.maximum(errors, 1e-5) ** 2
    weights = np.minimum(weights, np.median(weights) * 16)
    median = _weighted_median(ratios, weights)
    mad = float(np.median(np.abs(ratios - median)))
    accepted = np.abs(ratios - median) <= np.maximum(3 * errors, max(2.8 * mad, 0.018 * median))
    combined = float(np.average(ratios[accepted], weights=weights[accepted]))
    independent = math.sqrt(1 / float(weights[accepted].sum()))
    sampling = float(np.sqrt(np.average((ratios[accepted] - combined) ** 2, weights=weights[accepted])))
    uncertain = math.hypot(independent, sampling)
    old = frame.pixel_um
    warnings = []
    rejected = int((~accepted).sum())
    if rejected:
        warnings.append(f"{rejected} span(s) disagree with the other calibration marks.")
    if combined < 0.001 or combined > 100_000:
        warnings.append("The estimated scale is outside the project range.")
    if uncertain > combined * 0.08:
        warnings.append("The scale estimate is sensitive to point placement or disagreement.")
    for index, row in enumerate(estimates):
        row["retained"] = bool(accepted[index])
        row["residual_um"] = row["original_distance_px"] * combined - row["known_um"]
    return {
        "value": combined,
        "uncertainty": uncertain,
        "reference_capture_id": run["summary"]["reference_id"],
        "reference_capture_name": run["summary"]["reference_name"],
        "scale_change_percent": None if old is None else ((combined / old) - 1) * 100,
        "using_original_coordinates": True,
        "can_apply": bool(0.001 <= combined <= 100_000),
        "spans": estimates,
        "warnings": warnings,
    }


def commit_estimate(library, project_id: str, estimate: dict, revision: int | None) -> dict:
    if not estimate["can_apply"]:
        raise Unprocessable("Use a calibrated size inside the supported range.")
    active_snapshot = library.snapshot(project_id, revision)
    published = active_snapshot["captures"]
    source_id = estimate["reference_capture_id"]
    current = RunSettings.parse(active_snapshot["settings"])
    ordered = sorted(published, key=lambda item: (item["position"], item["id"]))
    selected = current.reference_id or (ordered[len(ordered) // 2]["id"] if ordered else None)
    if selected != source_id:
        raise Conflict("This calibration belongs to a different reference capture. Reconstruct the current stack first.")
    scale = replace(current.scale, pixel_um=estimate["value"])
    return library.update_project(
        project_id, {"settings": replace(current, scale=scale)}, active_snapshot["revision"],
    )
