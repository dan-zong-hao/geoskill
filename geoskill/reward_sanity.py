import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from rl_bbox.bbox_rewards import bbox_reward

q = "What color is the surface of the left-most vertical breakwater?"
gt = [163, 78, 196, 601]
cases = {
    "good": [120, 70, 180, 610],
    "bad": [500, 70, 560, 610],
    "missing": None,
}
for name, bbox in cases.items():
    if bbox is None:
        text = "<think>x</think><answer>unknown</answer>"
    else:
        text = '<think>x</think><zoom>[{"bbox_2d": %s, "label": "x"}]</zoom>' % json.dumps(bbox)
    r = bbox_reward(
        text,
        bbox,
        gt,
        question=q,
        image_size=(1024, 1024),
        w_format=0.05,
        w_iou=1.0,
        w_rg=1.0,
        w_spatial=1.5,
        spatial_penalty=0.5,
    )
    print(name, round(r["total"], 4), r["spatial_reward"], r["spatial_violation"])
