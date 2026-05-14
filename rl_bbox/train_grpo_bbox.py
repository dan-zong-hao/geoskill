"""Turn1-only bbox GRPO training.

Purpose:
  Improve bbox localization before returning to the full zoom->seg->answer RL.

Compared with /root/autodl-tmp/VQA/rl/train_grpo.py:
  * filters to samples with GT bbox;
  * samples only Turn1 under the same dispatcher-style prompt used by eval;
  * stops generation at </zoom> or </answer>;
  * rewards only bbox format + direct IoU + region-guided center reward;
  * computes r_R-G in 1024 coordinate space with alpha=200;
  * supports real GRPO effective batch via gradient accumulation;
  * optionally adds the paper-style KL term against the SFT reference model;
  * loss focuses on the <zoom>...</zoom> span when present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bbox_rewards import bbox_reward, extract_first_bbox
from geoskill.skillbank import format_skill_block, load_skillbank, retrieve_skills


os.environ.setdefault("HF_HUB_VERBOSITY", "error")
Image.MAX_IMAGE_PIXELS = None

VQA_ROOT = Path("/root/autodl-tmp/VQA")
RL_BBOX_ROOT = Path(__file__).resolve().parent

SYSTEM_PROMPT = """You are an intelligent remote sensing analyst. Given a question about a \
satellite image, you MAY use two tools to focus before answering:

  1. <zoom>[{"bbox_2d":[x1,y1,x2,y2],"label":"<short>"}]</zoom>
     Crop the image to the referenced region. Use when the target occupies a small
     fraction of the global view.
  2. <seg>{"prompt":"<text>"}</seg>
     Segment the target within the crop. Use when the answer depends on shape or
     coverage or precise boundary.

Protocol:
  - Wrap reasoning in <think>...</think>.
  - At most ONE <zoom> per trajectory; <seg> may only appear AFTER <zoom>.
  - End with exactly one <answer>...</answer> (single word or short phrase).
  - If the whole image is enough, skip zoom/seg and answer directly.
  - Never say "uncertain" — make your best guess.
"""

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
LRS_IMG_DIR_CANDIDATES = [
    Path("/root/autodl-tmp/dataset/lrs_gro/image"),
    Path("/root/autodl-tmp/dataset/lrs_gro/images"),
    Path("/root/autodl-tmp/dataset/lrs_gro"),
]


@dataclass
class BBoxGRPOConfig:
    model_path: str = str(VQA_ROOT / "sft" / "ckpt_sft_full_qwen35_think1024_4gpu" / "final_hf")
    train_jsonl: str = str(VQA_ROOT / "json_data" / "zoom_seg_json" / "rl_level" / "rl-00000-of-00001.1.zoom_seg.think.jsonl")
    output_dir: str = str(RL_BBOX_ROOT / "ckpt_bbox_grpo_4gpu")
    num_train_epochs: int = 1
    batch_size_per_device: int = 1
    num_generations: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-7
    warmup_steps: int = 20
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    beta: float = 0.04
    seed: int = 42
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.8
    rollout_top_k: int = 20
    rollout_min_p: float = 0.0
    rollout_repetition_penalty: float = 1.0
    max_new_tokens_turn1: int = 384
    max_pixels: int = 64 * 64 * 28 * 28
    w_format: float = 0.05
    w_iou: float = 1.0
    w_rg: float = 2.0
    rg_alpha: float = 200.0
    rg_shifted: bool = False
    enable_spatial_reward: bool = False
    w_spatial: float = 1.5
    spatial_penalty: float = 0.5
    spatial_margin: float = 32.0
    skillbank_path: str = ""
    split_manifest: str = ""
    train_split: str = "rl_train"
    skip_final_save: bool = False
    loss_on_zoom_only: bool = True
    save_steps: int = 25
    log_steps: int = 1


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=os.environ.get("DDP_BACKEND", "nccl"), init_method="env://")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    return distributed, rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def rank_print(rank: int, *args, **kwargs):
    if is_main(rank):
        print(*args, **kwargs)


def barrier(distributed: bool, local_rank: int):
    if not distributed:
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def reduce_mean(x: float, device: str, distributed: bool) -> float:
    t = torch.tensor(float(x), dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t.item()


def resize_image(image: Image.Image, max_size: int = 512) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        return image, 1.0 / scale
    return image, 1.0


def find_image(name: str) -> Optional[Path]:
    for d in LRS_IMG_DIR_CANDIDATES:
        p = d / name
        if p.exists():
            return p
    return None


class BBoxPromptDataset(Dataset):
    def __init__(self, jsonl_path: str, split_manifest: str = "", train_split: str = "rl_train"):
        self.records: list[dict] = []
        allowed_qids: set[str] | None = None
        if split_manifest:
            with open(split_manifest, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            splits = manifest.get("splits", {})
            allowed_qids = {str(qid) for qid, split in splits.items() if split == train_split}
            print(f"[data] split={train_split} allowed_qids={len(allowed_qids)} from {split_manifest}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if allowed_qids is not None and str(r.get("question_id")) not in allowed_qids:
                    continue
                bbox = r.get("bbox") or r.get("bbox_ref") or []
                if not bbox or len(bbox) != 4:
                    continue
                image_name = r.get("image_name") or r.get("image")
                if image_name and find_image(image_name) is not None:
                    self.records.append(r)
        print(f"[data] bbox prompts={len(self.records)} from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image_name = r.get("image_name") or r.get("image")
        bbox = r.get("bbox") or r.get("bbox_ref") or []
        return {
            "question_id": r["question_id"],
            "image_path": str(find_image(image_name)),
            "question": r["question"],
            "bbox_1024": [float(v) for v in bbox[:4]],
            "ground_truth": r.get("ground_truth", ""),
            "type": r.get("type", ""),
            "category": r.get("category", ""),
        }


def collate(batch):
    return batch


def build_turn1_prompt(question: str, skills: Optional[list[dict]] = None) -> str:
    skill_block = format_skill_block(skills or [])
    user_text = f"{VISION_TOKEN}{question}"
    if skill_block:
        user_text += "\n\n" + skill_block
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class BBoxSampler:
    def __init__(self, model, processor, device: str, cfg: BBoxGRPOConfig):
        self.model = model
        self.processor = processor
        self.device = device
        self.cfg = cfg
        self.skillbank = load_skillbank(cfg.skillbank_path) if cfg.skillbank_path else []
        tok = processor.tokenizer
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.model.generation_config.pad_token_id = pad_id
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = pad_id

    def generate(self, prompt: str, images: list[Image.Image], n: int = 1) -> list[str]:
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt", padding="longest").to(self.device)
        tok = self.processor.tokenizer
        kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens_turn1,
            do_sample=True,
            num_beams=1,
            num_return_sequences=n,
            temperature=self.cfg.rollout_temperature,
            top_p=self.cfg.rollout_top_p,
            top_k=self.cfg.rollout_top_k,
            min_p=self.cfg.rollout_min_p,
            repetition_penalty=self.cfg.rollout_repetition_penalty,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            stop_strings=["</zoom>", "</answer>"],
            tokenizer=tok,
        )
        if tok.pad_token_id is not None:
            kwargs["bad_words_ids"] = [[tok.pad_token_id]]
        with torch.inference_mode():
            gen = self.model.generate(**inputs, **kwargs)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        return [tok.decode(gen[i], skip_special_tokens=True).strip() for i in range(gen.shape[0])]


def rollout_batch(batch: list[dict], sampler: BBoxSampler, cfg: BBoxGRPOConfig):
    model = sampler.model
    was_training = model.training
    was_grad_ckpt = bool(getattr(model, "is_gradient_checkpointing", False))
    model.gradient_checkpointing_disable()
    for m in model.modules():
        if getattr(m, "gradient_checkpointing", False):
            m.gradient_checkpointing = False
    prev_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    model.eval()

    all_rollouts: list[dict] = []
    try:
        for sample in batch:
            image = Image.open(sample["image_path"]).convert("RGB")
            image_size = image.size
            global_small, _ = resize_image(image)
            skills = retrieve_skills(sample["question"], sampler.skillbank) if sampler.skillbank else []
            prompt = build_turn1_prompt(sample["question"], skills)
            outs = sampler.generate(prompt, [global_small], n=cfg.num_generations)
            sample_rollouts = []
            for out in outs:
                trajectory = f"[TURN1]\n{out}\n"
                pred_bbox = extract_first_bbox(out)
                reward = bbox_reward(
                    trajectory,
                    pred_bbox,
                    sample["bbox_1024"],
                    image_size=image_size,
                    question=sample["question"] if cfg.enable_spatial_reward else None,
                    w_format=cfg.w_format,
                    w_iou=cfg.w_iou,
                    w_rg=cfg.w_rg,
                    w_spatial=cfg.w_spatial if cfg.enable_spatial_reward else 0.0,
                    spatial_penalty=cfg.spatial_penalty if cfg.enable_spatial_reward else 0.0,
                    spatial_margin=cfg.spatial_margin,
                    rg_alpha=cfg.rg_alpha,
                    rg_shifted=cfg.rg_shifted,
                )
                sample_rollouts.append({
                    "question_id": sample["question_id"],
                    "prompt": prompt,
                    "completion": out + "<|im_end|>",
                    "images": [global_small],
                    "trajectory": trajectory,
                    "pred_bbox_1024": pred_bbox,
                    "gt_bbox_1024": sample["bbox_1024"],
                    "reward": reward,
                })
            totals = torch.tensor([r["reward"]["total"] for r in sample_rollouts], dtype=torch.float32)
            adv = (totals - totals.mean()) / (totals.std(unbiased=False) + 1e-4)
            for r, a in zip(sample_rollouts, adv):
                r["advantage"] = float(a.item())
            all_rollouts.extend(sample_rollouts)
    finally:
        model.config.use_cache = False if was_training else prev_use_cache
        if was_grad_ckpt:
            model.gradient_checkpointing_enable()
        model.train(was_training)
    return all_rollouts


def special_token_ids(tokenizer) -> list[int]:
    ids = []
    for tok in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.append(int(tid))
    return ids


def compute_logps_and_mask(model, processor, prompt: str, completion: str, images: list[Image.Image], *, zoom_only: bool):
    full = prompt + completion
    device = next(model.parameters()).device
    enc = processor(text=[full], images=images, return_tensors="pt", padding="longest").to(device)
    input_ids = enc["input_ids"]
    prompt_len = len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    out = model(**enc, use_cache=False)
    logits = out.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    completion_logps = token_logps[:, prompt_len - 1:].squeeze(0)
    completion_labels = shift_labels[:, prompt_len - 1:].squeeze(0)
    mask = torch.ones_like(completion_logps, dtype=torch.float32)

    ids = special_token_ids(processor.tokenizer)
    if ids:
        special = torch.tensor(ids, device=completion_labels.device)
        mask = mask * (~torch.isin(completion_labels, special)).float()

    if zoom_only:
        start = completion.find("<zoom>")
        end = completion.find("</zoom>")
        if start >= 0 and end >= start:
            end += len("</zoom>")
            start_len = len(processor.tokenizer(prompt + completion[:start], add_special_tokens=False)["input_ids"])
            end_len = len(processor.tokenizer(prompt + completion[:end], add_special_tokens=False)["input_ids"])
            s = max(0, start_len - prompt_len)
            e = min(mask.numel(), max(s + 1, end_len - prompt_len))
            focus = torch.zeros_like(mask)
            focus[s:e] = 1.0
            if (focus * mask).sum().item() > 0:
                mask = mask * focus
    return completion_logps, mask


def grpo_loss(model, ref_model, processor, rollouts: list[dict], cfg: BBoxGRPOConfig):
    losses = []
    kl_values = []
    for r in rollouts:
        logps, mask = compute_logps_and_mask(
            model, processor, r["prompt"], r["completion"], r["images"], zoom_only=cfg.loss_on_zoom_only
        )
        denom = mask.sum().clamp_min(1.0)
        old_logps = logps.detach()
        adv = torch.tensor(float(r["advantage"]), dtype=logps.dtype, device=logps.device)
        ratio = torch.exp(logps - old_logps)
        per_token_loss = -(ratio * adv)
        if ref_model is not None and cfg.beta > 0:
            with torch.no_grad():
                ref_logps, _ = compute_logps_and_mask(
                    ref_model, processor, r["prompt"], r["completion"], r["images"], zoom_only=cfg.loss_on_zoom_only
                )
            delta = ref_logps.detach() - logps
            kl = torch.exp(delta) - delta - 1.0
            per_token_loss = per_token_loss + cfg.beta * kl
            kl_values.append(((kl * mask).sum() / denom).detach())
        losses.append((per_token_loss * mask).sum() / denom)
    if not losses:
        return None, None
    loss = torch.stack(losses).mean()
    kl_mean = torch.stack(kl_values).mean() if kl_values else torch.tensor(0.0, device=loss.device)
    return loss, kl_mean


def metric_mean(rollouts: list[dict], key: str) -> float:
    return sum(float(r["reward"][key]) for r in rollouts) / max(len(rollouts), 1)


def metric_std_total(rollouts: list[dict]) -> float:
    vals = torch.tensor([float(r["reward"]["total"]) for r in rollouts], dtype=torch.float32)
    return vals.std(unbiased=False).item() if len(vals) else 0.0


def train(args):
    distributed, rank, local_rank, world_size, device = init_distributed()
    cfg = BBoxGRPOConfig(
        model_path=args.model_path,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size_per_device=args.batch_size_per_device,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        seed=args.seed,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        rollout_top_k=args.rollout_top_k,
        rollout_min_p=args.rollout_min_p,
        rollout_repetition_penalty=args.rollout_repetition_penalty,
        max_new_tokens_turn1=args.max_new_tokens_turn1,
        w_format=args.w_format,
        w_iou=args.w_iou,
        w_rg=args.w_rg,
        rg_alpha=args.rg_alpha,
        rg_shifted=args.rg_shifted,
        enable_spatial_reward=args.enable_spatial_reward,
        w_spatial=args.w_spatial,
        spatial_penalty=args.spatial_penalty,
        spatial_margin=args.spatial_margin,
        skillbank_path=args.skillbank_path,
        split_manifest=args.split_manifest,
        train_split=args.train_split,
        skip_final_save=args.skip_final_save,
        loss_on_zoom_only=not args.loss_on_full_completion,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )
    random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    rank_print(rank, f"[ddp] enabled={distributed} rank={rank}/{world_size} local_rank={local_rank} device={device}")
    rank_print(rank, f"[load] policy={cfg.model_path}")
    processor = AutoProcessor.from_pretrained(cfg.model_path, max_pixels=cfg.max_pixels)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.train()
    model.config.use_cache = False
    policy_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if distributed else model
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ref_model = None
    if cfg.beta > 0:
        rank_print(rank, f"[load] ref_model for KL beta={cfg.beta}")
        ref_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)
        ref_model.config.use_cache = False

    sampler_obj = BBoxSampler(model, processor, device, cfg)
    dataset = BBoxPromptDataset(cfg.train_jsonl, cfg.split_manifest, cfg.train_split)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed, drop_last=False) if distributed else None
    loader = DataLoader(dataset, batch_size=cfg.batch_size_per_device, shuffle=(sampler is None), sampler=sampler, num_workers=0, collate_fn=collate)

    total_updates = max(1, (len(loader) * cfg.num_train_epochs) // cfg.gradient_accumulation_steps)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_updates)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_prompts = cfg.batch_size_per_device * world_size * cfg.gradient_accumulation_steps
    rank_print(rank, f"[model] trainable={n_params/1e9:.2f}B")
    rank_print(rank, f"[sched] updates={total_updates} effective_prompt_batch={eff_prompts} rollouts/update={eff_prompts * cfg.num_generations}")
    rank_print(rank, f"[reward] w_format={cfg.w_format} w_iou={cfg.w_iou} w_rg={cfg.w_rg} w_spatial={cfg.w_spatial if cfg.enable_spatial_reward else 0.0} spatial_penalty={cfg.spatial_penalty if cfg.enable_spatial_reward else 0.0} rg_alpha={cfg.rg_alpha} rg_domain=orig beta={cfg.beta}")

    log_f = None
    tb = None
    if is_main(rank):
        log_f = (Path(cfg.output_dir) / "train.log").open("a", encoding="utf-8")
        log_f.write(f"\n=== bbox run start {datetime.now().isoformat()} world_size={world_size} ===\n")
        tb_dir = Path(cfg.output_dir) / "tb" / datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(str(tb_dir))
        print(f"[tb] {tb_dir}")

    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    global_step = 0
    stop_training = False

    for epoch in range(cfg.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for it, batch in enumerate(loader):
            if args.max_steps > 0 and it >= args.max_steps:
                stop_training = True
                break
            t0 = time.time()
            rollouts = rollout_batch(batch, sampler_obj, cfg)
            loss, kl_mean = grpo_loss(policy_model, ref_model, processor, rollouts, cfg)
            has_loss = torch.tensor(0 if loss is None else 1, dtype=torch.int32, device=device)
            if distributed:
                dist.all_reduce(has_loss, op=dist.ReduceOp.MIN)
            if has_loss.item() == 0:
                rank_print(rank, f"[ep{epoch} it{it}] skip no loss")
                continue
            (loss / cfg.gradient_accumulation_steps).backward()
            accumulated += 1
            if accumulated % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            local = {
                "reward_total": metric_mean(rollouts, "total"),
                "reward_std": metric_std_total(rollouts),
                "reward_format": metric_mean(rollouts, "format"),
                "reward_iou": metric_mean(rollouts, "iou"),
                "reward_rg": metric_mean(rollouts, "region_guided"),
                "reward_spatial": metric_mean(rollouts, "spatial_reward"),
                "spatial_applicable_rate": metric_mean(rollouts, "spatial_applicable"),
                "spatial_ok_rate": metric_mean(rollouts, "spatial_ok"),
                "spatial_penalty": metric_mean(rollouts, "spatial_penalty"),
                "center_distance_orig": metric_mean(rollouts, "center_distance_orig"),
                "bbox_valid_rate": metric_mean(rollouts, "bbox_valid"),
                "loss": float(loss.detach().item()),
                "kl": float(kl_mean.detach().item()) if kl_mean is not None else 0.0,
                "dt": time.time() - t0,
            }
            reduced = {k: reduce_mean(v, device, distributed) for k, v in local.items()}
            qids = ",".join(str(x["question_id"]) for x in batch)
            msg = (
                f"[ep{epoch} it{it} step{global_step}] qids={qids} "
                f"total={reduced['reward_total']:.3f} std={reduced['reward_std']:.3f} "
                f"fmt={reduced['reward_format']:.2f} iou={reduced['reward_iou']:.3f} "
                f"rg={reduced['reward_rg']:.3f} spatial={reduced['reward_spatial']:.3f} "
                f"sp_ok={reduced['spatial_ok_rate']:.2f}/{reduced['spatial_applicable_rate']:.2f} "
                f"dist_orig={reduced['center_distance_orig']:.1f} "
                f"bbox_rate={reduced['bbox_valid_rate']:.2f} loss={reduced['loss']:.4f} "
                f"kl={reduced['kl']:.5f} dt={reduced['dt']:.1f}s"
            )
            if is_main(rank) and (it % cfg.log_steps == 0):
                print(msg)
                log_f.write(msg + "\n")
                if it < 3 or it % 50 == 0:
                    for gi, r in enumerate(rollouts[: min(len(rollouts), 8)]):
                        log_f.write(
                            f"  r{gi} qid={r['question_id']} total={r['reward']['total']:.3f} "
                            f"apo={r['reward']['iou']:.3f} rg={r['reward']['region_guided']:.3f} "
                            f"sp={r['reward'].get('spatial_reward', 0.0):.3f} viol={r['reward'].get('spatial_violation', 'none')} "
                            f"dist_orig={r['reward']['center_distance_orig']:.1f} "
                            f"pred={r['pred_bbox_1024']} gt={r['gt_bbox_1024']}\n"
                        )
                        log_f.write("    out=" + repr(r["trajectory"][:700]) + "\n")
                log_f.flush()
                if tb is not None:
                    tb.add_scalar("reward/total", reduced["reward_total"], global_step)
                    tb.add_scalar("reward/total_std", reduced["reward_std"], global_step)
                    tb.add_scalar("reward/format", reduced["reward_format"], global_step)
                    tb.add_scalar("reward/iou_apo512", reduced["reward_iou"], global_step)
                    tb.add_scalar("reward/region_guided_orig", reduced["reward_rg"], global_step)
                    tb.add_scalar("reward/spatial", reduced["reward_spatial"], global_step)
                    tb.add_scalar("debug/spatial_applicable_rate", reduced["spatial_applicable_rate"], global_step)
                    tb.add_scalar("debug/spatial_ok_rate", reduced["spatial_ok_rate"], global_step)
                    tb.add_scalar("debug/spatial_penalty", reduced["spatial_penalty"], global_step)
                    tb.add_scalar("debug/center_distance_orig", reduced["center_distance_orig"], global_step)
                    tb.add_scalar("debug/bbox_valid_rate", reduced["bbox_valid_rate"], global_step)
                    tb.add_scalar("loss/policy_plus_kl", reduced["loss"], global_step)
                    tb.add_scalar("loss/kl", reduced["kl"], global_step)
                    tb.add_scalar("opt/lr", scheduler.get_last_lr()[0], global_step)
                    tb.add_scalar("opt/step_seconds", reduced["dt"], global_step)

            if is_main(rank) and global_step > 0 and global_step % cfg.save_steps == 0 and accumulated % cfg.gradient_accumulation_steps == 0:
                ckpt = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt, safe_serialization=True)
                processor.save_pretrained(ckpt)
                print(f"[save] {ckpt}")

        if stop_training:
            break

    barrier(distributed, local_rank)
    if is_main(rank):
        if cfg.skip_final_save:
            print("[done] skip_final_save enabled; final checkpoint not written")
        else:
            final = Path(cfg.output_dir) / "final_hf"
            final.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final, safe_serialization=True)
            processor.save_pretrained(final)
            print(f"[done] final ckpt -> {final}")
        if log_f:
            log_f.close()
        if tb:
            tb.close()
    barrier(distributed, local_rank)
    if distributed:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(VQA_ROOT / "sft" / "ckpt_sft_full_qwen35_think1024_4gpu" / "final_hf"))
    p.add_argument("--train_jsonl", default=str(VQA_ROOT / "json_data" / "zoom_seg_json" / "rl_level" / "rl-00000-of-00001.1.zoom_seg.think.jsonl"))
    p.add_argument("--output_dir", default=str(RL_BBOX_ROOT / "ckpt_bbox_grpo_4gpu"))
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--batch_size_per_device", type=int, default=1)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-7)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollout_temperature", type=float, default=0.7)
    p.add_argument("--rollout_top_p", type=float, default=0.8)
    p.add_argument("--rollout_top_k", type=int, default=20)
    p.add_argument("--rollout_min_p", type=float, default=0.0)
    p.add_argument("--rollout_repetition_penalty", type=float, default=1.0)
    p.add_argument("--max_new_tokens_turn1", type=int, default=384)
    p.add_argument("--w_format", type=float, default=0.05)
    p.add_argument("--w_iou", type=float, default=1.0)
    p.add_argument("--w_rg", type=float, default=2.0)
    p.add_argument("--rg_alpha", type=float, default=200.0)
    p.add_argument("--rg_shifted", action="store_true")
    p.add_argument("--enable_spatial_reward", action="store_true")
    p.add_argument("--w_spatial", type=float, default=1.5)
    p.add_argument("--spatial_penalty", type=float, default=0.5)
    p.add_argument("--spatial_margin", type=float, default=32.0)
    p.add_argument("--skillbank_path", default="")
    p.add_argument("--split_manifest", default="")
    p.add_argument("--train_split", default="rl_train")
    p.add_argument("--loss_on_full_completion", action="store_true")
    p.add_argument("--skip_final_save", action="store_true")
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--log_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
