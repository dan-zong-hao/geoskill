# rl_bbox

Turn1-only bbox GRPO trainer used by GeoSkill-RL.

This directory contains the actual trainer that `geoskill/run_geoskill_bbox_4gpu.sh`
calls in the original server layout:

- `train_grpo_bbox.py`: GRPO training loop for Turn-1 `<zoom>` bbox generation.
- `bbox_rewards.py`: bbox/APO/region-guided reward plus GeoSkill spatial reward hooks.
- `run_rl_bbox_4gpu.sh`: vanilla bbox-GRPO launcher.
- `run_rl_bbox_smoke.sh`: smoke launcher.
- `run_rl_bbox_4gpu_zoomearth_steps.sh`: alternate longer-step launcher.

The GeoSkill additions are exposed through:

```bash
--enable_spatial_reward
--skillbank_path
--split_manifest
--train_split
--w_spatial
--spatial_penalty
```

In this GitHub repository, imports use the top-level `geoskill` package. On the
training server, the same code corresponds to `/root/autodl-tmp/VQA/speedup/rl_bbox`.
