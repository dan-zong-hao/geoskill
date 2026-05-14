# GeoSkill-RL

Utilities for eval-driven spatial skill evolution in remote-sensing active-perception VQA.

This package contains the v1 GeoSkill-RL scaffolding used with the existing
`speedup/rl_bbox` Turn-1 GRPO trainer:

- spatial locator parsing for `top/bottom/left/right/corner` questions
- spatial compliance reward helpers
- a seed spatial SkillBank and retrieval/formatting utilities
- RL data split generation for `rl_train/evo_val/dev_val`
- failure mining from prediction JSONL files
- launchers for 4-GPU GeoSkill bbox GRPO and smoke testing

## Typical Usage

Create data splits and run a 4-GPU Round-1 training job:

```bash
cd /root/autodl-tmp/VQA
python speedup/geoskill/create_splits.py
bash speedup/geoskill/run_geoskill_bbox_4gpu.sh --max_new_tokens_turn1 512
```

Run lightweight checks:

```bash
cd /root/autodl-tmp/VQA
python speedup/geoskill/test_spatial.py
python speedup/geoskill/reward_sanity.py
```

Mine spatial residuals from predictions:

```bash
python speedup/geoskill/mine_failures.py \
  --predictions_jsonl <predictions.jsonl> \
  --out_jsonl speedup/geoskill/evo_val_residuals_round1.jsonl \
  --out_report speedup/geoskill/failure_report_round1.json
```

Generated splits, residual files, logs, and checkpoints are intentionally ignored
by git to avoid leaking dataset-derived artifacts or large model files.
