# SFT Scripts

SFT launchers and training entrypoint used before GeoSkill-RL Round 1.

Included files:

- 	rain_sft_full_qwen35.py: Qwen3.5-VL SFT training entrypoint.
- un_sft_full_qwen35.sh: single-node launcher.
- un_sft_full_qwen35_4gpu.sh: 4-GPU launcher for the full SFT baseline.
- un_sft_spatial_qwen35_4gpu.sh: 4-GPU launcher for the spatial SFT variant.

Checkpoints, logs, pycache files, and dataset-derived outputs are intentionally not tracked.
