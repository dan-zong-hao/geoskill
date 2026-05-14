"""Spatial locator parsing and reward helpers for GeoSkill-RL."""
from __future__ import annotations

import math
import re
from typing import Any, Optional


LOCATOR_PATTERNS = {
    "top": [
        r"\btop[- ]?most\b",
        r"\bupper[- ]?most\b",
        r"\bnorthern[- ]?most\b",
        r"\bnorth[- ]?most\b",
        r"\btop\b",
        r"\bupper\b",
        r"\bnorthern\b",
        r"\bnorth\b",
    ],
    "bottom": [
        r"\bbottom[- ]?most\b",
        r"\blower[- ]?most\b",
        r"\bsouthern[- ]?most\b",
        r"\bsouth[- ]?most\b",
        r"\bbottom\b",
        r"\blower\b",
        r"\bsouthern\b",
        r"\bsouth\b",
    ],
    "left": [
        r"\bleft[- ]?most\b",
        r"\bwestern[- ]?most\b",
        r"\bwest[- ]?most\b",
        r"\bleft\b",
        r"\bwestern\b",
        r"\bwest\b",
    ],
    "right": [
        r"\bright[- ]?most\b",
        r"\beastern[- ]?most\b",
        r"\beast[- ]?most\b",
        r"\bright\b",
        r"\beastern\b",
        r"\beast\b",
    ],
}

BACKLOG_PATTERNS = {
    "largest": [r"\blargest\b", r"\bbiggest\b"],
    "smallest": [r"\bsmallest\b", r"\btiniest\b"],
    "nearest": [r"\bnearest\b", r"\bclosest\b"],
    "farthest": [r"\bfarthest\b", r"\bfurthest\b"],
}


def _matches(text: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            hits.append(m.group(0))
    return hits


def parse_locator(question: str) -> dict[str, Any]:
    """Parse explicit spatial locator words from a question.

    Returns a small structured dict used by both reward and failure mining.
    The v1 reward acts only on top/bottom/left/right axes. Other high-value
    locators are recorded as backlog families for later skill evolution.
    """
    q = (question or "").lower()
    axes: list[str] = []
    triggers: dict[str, list[str]] = {}
    for family, patterns in LOCATOR_PATTERNS.items():
        hits = _matches(q, patterns)
        if hits:
            axes.append(family)
            triggers[family] = hits

    backlog: dict[str, list[str]] = {}
    for family, patterns in BACKLOG_PATTERNS.items():
        hits = _matches(q, patterns)
        if hits:
            backlog[family] = hits

    vertical = [x for x in axes if x in {"top", "bottom"}]
    horizontal = [x for x in axes if x in {"left", "right"}]
    if vertical and horizontal:
        family = "corner"
    elif axes:
        family = axes[0]
    elif backlog:
        family = next(iter(backlog))
    else:
        family = "none"

    return {
        "has_locator": bool(axes),
        "family": family,
        "axes": axes,
        "triggers": triggers,
        "backlog_families": list(backlog.keys()),
        "backlog_triggers": backlog,
    }


def canonical_bbox_1024(bbox: Optional[list[float]]) -> Optional[list[float]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in [x1, y1, x2, y2]):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(1024.0, x1))
    y1 = max(0.0, min(1024.0, y1))
    x2 = max(0.0, min(1024.0, x2))
    y2 = max(0.0, min(1024.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def spatial_violation_type(
    pred_bbox: Optional[list[float]],
    gt_bbox: Optional[list[float]],
    locator: dict[str, Any] | str | None,
    margin: float = 32.0,
) -> str:
    if isinstance(locator, str):
        locator = parse_locator(locator)
    locator = locator or {}
    axes = list(locator.get("axes") or [])
    if not axes:
        return "none"

    pred = canonical_bbox_1024(pred_bbox)
    gt = canonical_bbox_1024(gt_bbox)
    if pred is None:
        return "missing_bbox"
    if gt is None:
        return "missing_gt"

    px, py = center(pred)
    gx, gy = center(gt)
    violations: list[str] = []
    if "top" in axes and py > gy + margin:
        violations.append("top_violated")
    if "bottom" in axes and py < gy - margin:
        violations.append("bottom_violated")
    if "left" in axes and px > gx + margin:
        violations.append("left_violated")
    if "right" in axes and px < gx - margin:
        violations.append("right_violated")
    if not violations:
        return "none"
    if len(axes) > 1 and len(violations) < len(axes):
        return "corner_partial:" + "+".join(violations)
    return "+".join(violations)


def spatial_reward(
    pred_bbox: Optional[list[float]],
    gt_bbox: Optional[list[float]],
    locator: dict[str, Any] | str | None,
    margin: float = 32.0,
) -> dict[str, Any]:
    if isinstance(locator, str):
        locator = parse_locator(locator)
    locator = locator or {}
    axes = list(locator.get("axes") or [])
    pred = canonical_bbox_1024(pred_bbox)
    gt = canonical_bbox_1024(gt_bbox)
    violation = spatial_violation_type(pred, gt, locator, margin=margin)
    applicable = bool(axes)
    ok = applicable and violation == "none" and pred is not None and gt is not None

    dx = dy = None
    if pred is not None and gt is not None:
        px, py = center(pred)
        gx, gy = center(gt)
        dx = px - gx
        dy = py - gy

    return {
        "spatial_applicable": 1.0 if applicable else 0.0,
        "spatial_reward": 1.0 if ok else 0.0,
        "spatial_ok": 1.0 if ok else 0.0,
        "spatial_penalty": 1.0 if applicable and violation != "none" else 0.0,
        "spatial_violation": violation,
        "locator_family": locator.get("family", "none"),
        "locator_axes": axes,
        "center_delta_x": float(dx) if dx is not None else 0.0,
        "center_delta_y": float(dy) if dy is not None else 0.0,
        "backlog_families": locator.get("backlog_families", []),
    }
