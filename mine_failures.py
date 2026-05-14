from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Any
from speedup.geoskill.spatial import parse_locator, spatial_reward
BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[(.*?)\]', re.DOTALL)
def _bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            m = BBOX_RE.search(value)
            if not m:
                return None
            value = [x.strip() for x in m.group(1).split(",")]
    if isinstance(value, dict):
        for key in ["pred_bbox_1024", "bbox_pred_1024", "prediction_bbox", "gt_bbox_1024", "bbox_gt_1024", "bbox_1024", "bbox", "bbox_ref"]:
            if key in value:
                return _bbox(value[key])
    if isinstance(value, list) and len(value) == 4:
        try:
            return [float(x) for x in value]
        except Exception:
            return None
    return None
def _first(row: dict, keys: list[str], default=None):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default
def signature(row: dict) -> dict:
    question = str(_first(row, ["question", "prompt"], ""))
    gt = _bbox(_first(row, ["gt_bbox_1024", "bbox_gt_1024", "bbox_1024", "bbox", "bbox_ref", "solution"], None))
    pred = _bbox(_first(row, ["pred_bbox_1024", "bbox_pred_1024", "pred_bbox", "prediction_bbox", "completion", "trajectory", "output", "response"], None))
    locator = parse_locator(question)
    spatial = spatial_reward(pred, gt, locator)
    qid = _first(row, ["question_id", "qid", "id"], "")
    answer_correct = _first(row, ["answer_correct", "correct_final", "correct"], None)
    if isinstance(answer_correct, str):
        answer_correct = answer_correct.lower() in {"1", "true", "yes", "correct"}
    actions = _first(row, ["actions", "action_path"], None)
    if actions is None:
        text = str(_first(row, ["trajectory", "completion", "output", "response"], ""))
        fired = []
        if "<zoom>" in text:
            fired.append("zoom")
        if "<seg>" in text:
            fired.append("seg")
        actions = "+".join(fired) if fired else "none"
    return {
        "question_id": qid,
        "question": question,
        "type": _first(row, ["type", "qtype"], ""),
        "category": _first(row, ["category"], ""),
        "gt_bbox_1024": gt,
        "pred_bbox_1024": pred,
        "locator_family": spatial["locator_family"],
        "locator_axes": spatial["locator_axes"],
        "violation_type": spatial["spatial_violation"],
        "spatial_applicable": bool(spatial["spatial_applicable"]),
        "spatial_ok": bool(spatial["spatial_ok"]),
        "center_delta": {"dx": spatial["center_delta_x"], "dy": spatial["center_delta_y"]},
        "apo_iou": float(_first(row, ["apo_iou", "iou_apo512", "iou"], 0.0) or 0.0),
        "answer_correct": answer_correct,
        "action_path": actions,
        "backlog_families": spatial["backlog_families"],
    }
def mine_failures(predictions_jsonl: str, out_jsonl: str, out_report: str) -> None:
    residuals = []
    with Path(predictions_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                residuals.append(signature(json.loads(line)))
    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in residuals:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = len(residuals)
    applicable = [r for r in residuals if r["spatial_applicable"]]
    violations = [r for r in applicable if not r["spatial_ok"]]
    by_violation = Counter(r["violation_type"] for r in violations)
    by_family = Counter(r["locator_family"] for r in applicable)
    backlog = Counter(x for r in residuals for x in r.get("backlog_families", []))
    report = {
        "n": total,
        "spatial_applicable": len(applicable),
        "spatial_violation_rate": len(violations) / max(len(applicable), 1),
        "by_locator_family": dict(by_family),
        "by_violation_type": dict(by_violation),
        "backlog_families": dict(backlog),
    }
    Path(out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_report", required=True)
    args = ap.parse_args()
    mine_failures(args.predictions_jsonl, args.out_jsonl, args.out_report)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
