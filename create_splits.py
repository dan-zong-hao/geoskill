from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import random
from collections import defaultdict
from speedup.geoskill.spatial import parse_locator
def strat_key(r: dict) -> tuple[str, str, str, str]:
    loc = parse_locator(r.get("question", ""))
    return (
        str(r.get("type", "")),
        str(r.get("category", "")),
        "bbox" if (r.get("bbox") or r.get("bbox_ref")) else "no_bbox",
        str(loc.get("family", "none")),
    )
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/root/autodl-tmp/VQA/json_data/zoom_seg_json/rl_level/rl-00000-of-00001.1.zoom_seg.think.jsonl")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/VQA/speedup/geoskill")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rl_train", type=int, default=1700)
    ap.add_argument("--evo_val", type=int, default=400)
    ap.add_argument("--dev_val", type=int, default=400)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    total_target = args.rl_train + args.evo_val + args.dev_val
    if total_target != len(rows):
        raise ValueError(f"split counts {total_target} do not match input rows {len(rows)}")
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[strat_key(r)].append(r)
    rng = random.Random(args.seed)
    for g in groups.values():
        rng.shuffle(g)
    ratios = {"rl_train": args.rl_train / len(rows), "evo_val": args.evo_val / len(rows), "dev_val": args.dev_val / len(rows)}
    split_rows = {"rl_train": [], "evo_val": [], "dev_val": []}
    remainders: list[tuple[float, str, list[dict]]] = []
    for group_rows in groups.values():
        n = len(group_rows)
        used = 0
        for split in ["rl_train", "evo_val"]:
            exact = n * ratios[split]
            take = int(exact)
            split_rows[split].extend(group_rows[used:used + take])
            used += take
            remainders.append((exact - take, split, group_rows[used:used]))
        split_rows["dev_val"].extend(group_rows[used:])
    # Exact-size correction while preserving deterministic order.
    all_rows = [r for part in split_rows.values() for r in part]
    by_qid = {str(r["question_id"]): r for r in all_rows}
    assigned = {}
    for split, part in split_rows.items():
        for r in part:
            assigned[str(r["question_id"])] = split
    def move_one(src: str, dst: str) -> bool:
        for r in list(split_rows[src]):
            split_rows[src].remove(r)
            split_rows[dst].append(r)
            assigned[str(r["question_id"])] = dst
            return True
        return False
    targets = {"rl_train": args.rl_train, "evo_val": args.evo_val, "dev_val": args.dev_val}
    changed = True
    while changed:
        changed = False
        for dst, target in targets.items():
            while len(split_rows[dst]) < target:
                src = max(split_rows, key=lambda s: len(split_rows[s]) - targets[s])
                if len(split_rows[src]) <= targets[src] or not move_one(src, dst):
                    break
                changed = True
        for src, target in targets.items():
            while len(split_rows[src]) > target:
                dst = min(split_rows, key=lambda s: len(split_rows[s]) - targets[s])
                if len(split_rows[dst]) >= targets[dst] or not move_one(src, dst):
                    break
                changed = True
    out_dir = Path(args.out_dir)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": args.input,
        "seed": args.seed,
        "counts": {k: len(v) for k, v in split_rows.items()},
        "splits": {str(r["question_id"]): split for split, part in split_rows.items() for r in part},
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, part in split_rows.items():
        with (split_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
