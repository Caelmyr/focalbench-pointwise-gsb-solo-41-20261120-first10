"""Bench capture schedules and observed rail coverage."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .focus import rail_statistics
from .models import CapturePlan


@dataclass(frozen=True)
class Schedule:
    positions: tuple[float, ...]
    required_frames: int
    ideal_step: float
    actual_step: float
    overlap: float
    reserve: int
    settle_ms: int
    interval_ms: int

    @property
    def budget_limited(self) -> bool:
        return self.required_frames > len(self.positions)

    @property
    def time_ms(self) -> int:
        return self.settle_ms * (len(self.positions) - 1) + self.interval_ms * len(self.positions)

    def public(self) -> dict:
        return {
            "required_frames": self.required_frames,
            "frames": len(self.positions),
            "ideal_step": self.ideal_step,
            "actual_step": self.actual_step,
            "overlap_percent": self.overlap,
            "budget_limited": self.budget_limited,
            "extra_frames_needed": max(0, self.required_frames - len(self.positions)),
            "remaining_frame_slots": self.reserve,
            "estimated_ms": self.time_ms,
            "direction": "ascending" if self.positions[0] < self.positions[-1] else "descending",
            "positions": list(self.positions),
        }


def make_schedule(plan: CapturePlan) -> Schedule:
    travel = abs(plan.last - plan.first)
    spacing = plan.in_focus_width * (1 - plan.overlap_percent / 100)
    required = max(2, math.ceil(travel / spacing - 1e-10) + 1)
    usable = min(required, plan.usable_frames)
    positions = np.linspace(plan.first, plan.last, usable).tolist()
    span = travel / (usable - 1)
    actual_overlap = min(100.0, 100 * (1 - span / plan.in_focus_width))
    return Schedule(
        positions=tuple(positions),
        required_frames=required,
        ideal_step=spacing,
        actual_step=span,
        overlap=actual_overlap,
        reserve=plan.usable_frames - usable,
        settle_ms=plan.settle_ms,
        interval_ms=plan.interval_ms,
    )


def read_capture_positions(captures: list[dict]) -> dict:
    enabled = [capture for capture in captures if capture["included"] and not capture.get("archived")]
    groups = defaultdict(list)
    for capture in enabled:
        groups[capture["position"]].append(capture["id"])
    values = sorted(groups)
    duplicates = [
        {"position": value, "capture_ids": group}
        for value, group in groups.items() if len(group) > 1
    ]
    gaps = []
    sorted_positions = np.asarray(values, dtype=np.float64)
    stats = rail_statistics(sorted_positions)
    typical_gap = stats["median_step"]
    for index in range(len(values) - 1):
        span = values[index + 1] - values[index]
        if typical_gap and span > typical_gap * 1.75:
            gaps.append({"start": values[index], "end": values[index + 1], "gap": span})
    return {
        "count": len(enabled),
        "distinct_count": len(values),
        "positions": values,
        "duplicates": duplicates,
        "gaps": gaps,
        "statistics": stats,
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
    }


def _match_targets(
    positions: tuple[float, ...],
    captures: list[dict],
    tolerance: float,
) -> list[tuple[int, int]]:
    """Prefer maximum targets matched, then minimum total stage error."""
    targets = sorted(enumerate(positions), key=lambda item: (item[1], item[0]))
    scans = sorted(enumerate(captures), key=lambda item: (item[1]["position"], item[0]))
    rows, cols = len(targets) + 1, len(scans) + 1
    score = [[(0, 0.0)] * cols for _ in range(rows)]
    path = [[0] * cols for _ in range(rows)]
    for row in range(1, rows):
        for col in range(1, cols):
            contenders = ((score[row - 1][col], 1), (score[row][col - 1], 2))
            best, choice = max(contenders, key=lambda pair: pair[0])
            error = abs(targets[row - 1][1] - scans[col - 1][1]["position"])
            if error <= tolerance:
                previous, cost = score[row - 1][col - 1]
                linked = previous + 1, cost - error
                if linked >= best:
                    best, choice = linked, 3
            score[row][col] = best
            path[row][col] = choice
    matches = []
    row, col = rows - 1, cols - 1
    while row > 0 and col > 0:
        direction = path[row][col]
        if direction == 3:
            matches.append((targets[row - 1][0], scans[col - 1][0]))
            row -= 1
            col -= 1
        elif direction == 1:
            row -= 1
        else:
            col -= 1
    return matches


def reconcile_schedule(schedule: Schedule, captures: list[dict], tolerance: float | None = None) -> dict:
    active = [capture for capture in captures if capture["included"] and not capture.get("archived")]
    if tolerance is None:
        tolerance = schedule.actual_step * 0.24
    matches = _match_targets(schedule.positions, active, tolerance)
    by_target = {target: active[source] for target, source in matches}
    used = {source for _, source in matches}
    upcoming = []
    for index, position in enumerate(schedule.positions):
        capture = by_target.get(index)
        upcoming.append({
            "index": index + 1,
            "target": position,
            "recorded": capture["position"] if capture else None,
            "capture_id": capture["id"] if capture else None,
            "offset": capture["position"] - position if capture else None,
        })
    extras = [
        {"id": capture["id"], "position": capture["position"], "name": capture["name"]}
        for index, capture in enumerate(active) if index not in used
    ]
    missing = [row["target"] for row in upcoming if row["recorded"] is None]
    return {
        "tolerance": tolerance,
        "matched": len(matches),
        "missing": missing,
        "unmatched": extras,
        "targets": upcoming,
    }


def plan_warnings(schedule: Schedule, coverage: dict) -> list[str]:
    warnings = []
    if schedule.overlap < 0:
        warnings.append("The frame budget leaves an estimated gap between sharp intervals.")
    elif schedule.budget_limited:
        warnings.append("The frame budget reduces the estimated overlap below your target.")
    if coverage["duplicates"]:
        warnings.append("Multiple enabled captures share a focus position.")
    if coverage["gaps"]:
        warnings.append("Some observed rail jumps are larger than the typical step.")
    return warnings


def inspect_schedule(plan: CapturePlan, captures: list[dict]) -> dict:
    schedule = make_schedule(plan)
    observed = read_capture_positions(captures)
    return {
        "schedule": schedule.public(),
        "observed": observed,
        "matches": reconcile_schedule(schedule, captures),
        "warnings": plan_warnings(schedule, observed),
    }
