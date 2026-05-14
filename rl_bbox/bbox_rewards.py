"""BBox-focused rewards for Turn1-only GRPO.

Model-emitted and dataset bboxes live in the 1024 coordinate domain, but the
paper/eval APO IoU is computed after mapping bboxes back to original pixels and
expanding both predicted and GT boxes to a fixed 512 crop. The main IoU reward
therefore follows the eval definition when image_size is provided; raw 1024 IoU
is kept only as a diagnostic.
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

from geoskill.spatial import parse_locator, spatial_reward, spatial_violation_type


ZOOM_RE = re.compile(r"<zoom>\s*(\[.*?\])\s*</zoom>", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[(.*?)\]', re.DOTALL)


def extract_first_bbox(text: str) -> Optional[list[float]]:
    zm = ZOOM_RE.search(text or "")
    if zm is None:
        return None
    bm = BBOX_RE.search(zm.group(1))
    if bm is None:
        return None
    try:
        nums = [float(x.strip()) for x in bm.group(1).split(",")]
    except ValueError:
        return None
    if len(nums) < 4:
        return None
    return nums[:4]


def canonical_bbox_1024(bbox: Optional[list[float]]) -> Optional[list[float]]:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if not all(math.isfinite(v) for v in [x1, y1, x2, y2]):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    # Reward uses clamped boxes so a partially out-of-range prediction can still
    # receive a shaped signal. Format/validity logs keep track of raw validity.
    x1 = max(0.0, min(1024.0, x1))
    y1 = max(0.0, min(1024.0, y1))
    x2 = max(0.0, min(1024.0, x2))
    y2 = max(0.0, min(1024.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def _center(b: list[float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _scale_1024_to_orig(bbox: list[float], image_size: tuple[int, int]) -> list[float]:
    scale = max(float(image_size[0]), float(image_size[1])) / 1024.0
    return [float(v) * scale for v in bbox]


def _shift_inside(x1: float, x2: float, low: float, high: float) -> tuple[float, float]:
    span = x2 - x1
    if high - low <= span:
        return low, high
    if x1 < low:
        x2 += low - x1
        x1 = low
    if x2 > high:
        x1 -= x2 - high
        x2 = high
    return max(low, x1), min(high, x2)


def _expand_fixed512_orig(bbox_orig: list[float], image_size: tuple[int, int], size: int = 512) -> list[float]:
    width, height = image_size
    side_x = min(float(size), float(width))
    side_y = min(float(size), float(height))
    cx = (float(bbox_orig[0]) + float(bbox_orig[2])) / 2.0
    cy = (float(bbox_orig[1]) + float(bbox_orig[3])) / 2.0
    x1, x2 = cx - side_x / 2.0, cx + side_x / 2.0
    y1, y2 = cy - side_y / 2.0, cy + side_y / 2.0
    x1, x2 = _shift_inside(x1, x2, 0.0, float(width))
    y1, y2 = _shift_inside(y1, y2, 0.0, float(height))
    return [x1, y1, x2, y2]


def apo_iou_fixed512_from_1024(
    pred_1024: Optional[list[float]],
    gt_1024: Optional[list[float]],
    image_size: Optional[tuple[int, int]],
) -> float:
    pred = canonical_bbox_1024(pred_1024)
    gt = canonical_bbox_1024(gt_1024)
    if pred is None or gt is None or image_size is None:
        return 0.0
    pred_orig = _scale_1024_to_orig(pred, image_size)
    gt_orig = _scale_1024_to_orig(gt, image_size)
    pred_512 = _expand_fixed512_orig(pred_orig, image_size, 512)
    gt_512 = _expand_fixed512_orig(gt_orig, image_size, 512)
    return _iou(pred_512, gt_512)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def center_distance_1024(pred: Optional[list[float]], gt: Optional[list[float]]) -> Optional[float]:
    pred = canonical_bbox_1024(pred)
    gt = canonical_bbox_1024(gt)
    if pred is None or gt is None:
        return None
    px, py = _center(pred)
    gx, gy = _center(gt)
    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2)


def center_distance_orig(
    pred: Optional[list[float]],
    gt: Optional[list[float]],
    image_size: Optional[tuple[int, int]],
) -> Optional[float]:
    pred = canonical_bbox_1024(pred)
    gt = canonical_bbox_1024(gt)
    if pred is None or gt is None or image_size is None:
        return None
    pred_orig = _scale_1024_to_orig(pred, image_size)
    gt_orig = _scale_1024_to_orig(gt, image_size)
    px, py = _center(pred_orig)
    gx, gy = _center(gt_orig)
    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2)


def region_guided_reward_orig(
    pred: Optional[list[float]],
    gt: Optional[list[float]],
    image_size: Optional[tuple[int, int]],
    alpha: float = 200.0,
    eps: float = 0.2,
    shifted: bool = False,
) -> float:
    dist = center_distance_orig(pred, gt, image_size)
    if dist is None:
        return 0.0
    r = _sigmoid(alpha / (dist + eps))
    if shifted:
        r = 2.0 * r - 1.0
    return float(max(0.0, min(1.0, r)))


def region_guided_reward_1024(
    pred: Optional[list[float]],
    gt: Optional[list[float]],
    alpha: float = 200.0,
    eps: float = 0.2,
    shifted: bool = False,
) -> float:
    dist = center_distance_1024(pred, gt)
    if dist is None:
        return 0.0
    r = _sigmoid(alpha / (dist + eps))
    if shifted:
        r = 2.0 * r - 1.0
    return float(max(0.0, min(1.0, r)))


def format_reward_bbox(text: str) -> float:
    bbox = canonical_bbox_1024(extract_first_bbox(text))
    if bbox is None:
        return 0.0
    # Keep this small-weighted reward forgiving; the main goal of this stage is
    # bbox quality, not long CoT style. Think presence is still useful.
    return 1.0 if THINK_RE.search(text or "") else 0.8


def bbox_reward(
    trajectory: str,
    pred_bbox_1024: Optional[list[float]],
    gt_bbox_1024: Optional[list[float]],
    *,
    image_size: Optional[tuple[int, int]] = None,
    question: Optional[str] = None,
    w_format: float = 0.05,
    w_iou: float = 1.0,
    w_rg: float = 2.0,
    w_spatial: float = 0.0,
    spatial_penalty: float = 0.0,
    spatial_margin: float = 32.0,
    rg_alpha: float = 200.0,
    rg_shifted: bool = False,
) -> dict:
    pred = canonical_bbox_1024(pred_bbox_1024)
    gt = canonical_bbox_1024(gt_bbox_1024)
    fmt = format_reward_bbox(trajectory)
    raw_iou_1024 = _iou(pred, gt) if pred is not None and gt is not None else 0.0
    apo_iou_512 = apo_iou_fixed512_from_1024(pred, gt, image_size)
    # Main IoU reward follows eval when image_size is available. The fallback is
    # only for unit/debug calls that do not know the original image size.
    iou = apo_iou_512 if image_size is not None else raw_iou_1024
    rg = region_guided_reward_orig(pred, gt, image_size, alpha=rg_alpha, shifted=rg_shifted)
    rg_1024 = region_guided_reward_1024(pred, gt, alpha=rg_alpha, shifted=rg_shifted)
    dist = center_distance_1024(pred, gt)
    dist_orig = center_distance_orig(pred, gt, image_size)
    locator = parse_locator(question or "")
    spatial = spatial_reward(pred, gt, locator, margin=spatial_margin)
    total = (
        w_format * fmt
        + w_iou * iou
        + w_rg * rg
        + w_spatial * float(spatial["spatial_reward"])
        - spatial_penalty * float(spatial["spatial_penalty"])
    )
    return {
        "format": fmt,
        "iou": iou,
        "iou_apo512": apo_iou_512,
        "iou_1024": raw_iou_1024,
        "region_guided": rg,
        "region_guided_1024": rg_1024,
        "center_distance_1024": dist if dist is not None else 1024.0 * math.sqrt(2.0),
        "center_distance_orig": dist_orig if dist_orig is not None else 1024.0 * math.sqrt(2.0),
        "bbox_valid": 1.0 if pred is not None else 0.0,
        "spatial_applicable": float(spatial["spatial_applicable"]),
        "spatial_reward": float(spatial["spatial_reward"]),
        "spatial_ok": float(spatial["spatial_ok"]),
        "spatial_penalty": float(spatial["spatial_penalty"]),
        "spatial_violation": str(spatial["spatial_violation"]),
        "locator_family": str(spatial["locator_family"]),
        "center_delta_x": float(spatial["center_delta_x"]),
        "center_delta_y": float(spatial["center_delta_y"]),
        "total": float(total),
    }
