"""Full-parameter SFT for Qwen3.5-4B on the v0502 zoom_seg dataset.

Variant of train_sft_full.py with model class swapped from
`Qwen2_5_VLForConditionalGeneration` to `Qwen3_5ForConditionalGeneration`.
Probe (probe_qwen35.py 2026-05-03) confirmed:
  - vision tokens identical (`<|vision_start|>`/`<|image_pad|>`/`<|vision_end|>`)
  - chat template structure identical (only adds `<think>\\n` after the
    `<|im_start|>assistant\\n` when add_generation_prompt=True; we bypass
    this by hand-constructing the prompt — same as the Qwen2.5 variant)
  - Qwen3VLProcessor loads via AutoProcessor, accepts list[list[Image]]

Memory expectation (4B params, bf16, full-FT, grad_ckpt on):
  weights 8GB + grads 8GB + Adam fp32 32GB + activations ~10-15GB ~= 60GB
  fits in a single 96GB Blackwell, tighter than the 3B variant.

Single-GPU launch:
  CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/VQA/pilot_0502/train_sft_full_qwen35.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

# LRS-GRO has up to 5000x5000+ images.
Image.MAX_IMAGE_PIXELS = None

VQA_ROOT = Path("/root/autodl-tmp/VQA")

DEFAULTS = dict(
    model_name=os.environ.get(
        "PILOT_MODEL_PATH",
        str(VQA_ROOT / "models" / "Qwen3.5-4B"),
    ),
    train_jsonl=str(
        VQA_ROOT / "json_data" / "zoom_seg_json" / "sft_level"
        / "sft-00000-of-00001.zoom_seg.jsonl"
    ),
    overlay_dir=str(VQA_ROOT / "json_data" / "zoom_seg_json" / "sft_level" / "overlays"),
    output_dir=str(VQA_ROOT / "sft" / "ckpt_sft_full_qwen35_think1024"),
    img_dir_candidates=[
        "/root/autodl-tmp/dataset/lrs_gro/image",
        "/root/autodl-tmp/dataset/lrs_gro/images",
        "/root/autodl-tmp/dataset/lrs_gro",
    ],
)

SYSTEM_PROMPT = (
    "You are an intelligent remote sensing analyst. Given a question about a "
    "satellite image, you MAY use two tools to focus before answering:\n"
    '  1. <zoom>[{"bbox_2d":[x1,y1,x2,y2],"label":"<short>"}]</zoom>\n'
    '  2. <seg>{"prompt":"<text>"}</seg>\n'
    "Protocol: wrap reasoning in <think>...</think>. At most ONE <zoom> per "
    "trajectory; <seg> may only appear AFTER <zoom>. End with exactly one "
    "<answer>...</answer> (single word or short phrase). If the whole image "
    "is enough, skip zoom/seg. Never say 'uncertain'."
)

VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


# ---------------------------------------------------------------------------
# Image utils — inlined from ZoomEarth/src/eval/infer.py to avoid extra deps.
# ---------------------------------------------------------------------------
def cut_image(image: Image.Image, bbox, min_size: int = 512) -> Image.Image:
    x1, y1, x2, y2 = map(int, bbox)
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1, ny1 = cx - min_size // 2, cy - min_size // 2
        nx2, ny2 = nx1 + min_size, ny1 + min_size
        if nx1 < 0: nx2 += -nx1; nx1 = 0
        if ny1 < 0: ny2 += -ny1; ny1 = 0
        if nx2 > image.width: nx1 -= nx2 - image.width; nx2 = image.width
        if ny2 > image.height: ny1 -= ny2 - image.height; ny2 = image.height
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(image.width, nx1 + min_size), min(image.height, ny1 + min_size)
        return image.crop((int(nx1), int(ny1), int(nx2), int(ny2)))
    return image.crop((x1, y1, x2, y2))


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return image


def _find_image(image_name: str, cands: List[str]) -> Optional[Path]:
    for d in cands:
        p = Path(d) / image_name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------
class ZoomSegDataset(Dataset):
    """Streams (global, crop, overlay) image triplets + assistant_text.

    The image count per record is determined by what `assistant_text`
    actually contains (presence of `</zoom>` / `</seg>`), so missing
    overlays gracefully degrade to a 2-image record.
    """

    def __init__(self, records: list, img_dir_candidates: List[str]):
        self.records = records
        self.cands = img_dir_candidates

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        path = _find_image(r["image_name"], self.cands)
        if path is None:
            raise FileNotFoundError(
                f"image not found: {r['image_name']} (searched: {self.cands})"
            )
        img_full = Image.open(path).convert("RGB")
        assistant_text = r["assistant_text"]

        images: List[Image.Image] = [resize_image(img_full)]

        has_zoom = "</zoom>" in assistant_text
        has_seg = "</seg>" in assistant_text

        if has_zoom and r.get("bbox"):
            scale = r.get("scale", 1.0)
            bbox_orig = [scale * v for v in r["bbox"]]
            crop = resize_image(cut_image(img_full, bbox_orig))
            images.append(crop)

        if has_seg:
            overlay_path = r.get("overlay_path")
            if overlay_path and Path(overlay_path).exists():
                images.append(Image.open(overlay_path).convert("RGB"))
            else:
                # The chain wants 3 images but overlay is missing — drop the
                # `</seg>` block from assistant_text so image count stays consistent.
                assistant_text = _strip_seg_block(assistant_text)

        return {
            "question": r["question"],
            "assistant_text": assistant_text,
            "images": images,
            "question_id": r["question_id"],
        }


def _strip_seg_block(text: str) -> str:
    """Remove the `<seg>...</seg>` segment and the following `<think>...</think>`
    that describes the mask, since both are meaningless without an overlay image."""
    import re
    text = re.sub(
        r"<seg>.*?</seg>\s*<think>.*?</think>\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return text


# ---------------------------------------------------------------------------
# Collator.
# ---------------------------------------------------------------------------
def build_collator(processor, max_length: int):
    pad_id = processor.tokenizer.pad_token_id
    ignore_index = -100

    def collate(examples):
        texts_prompt: List[str] = []
        texts_full: List[str] = []
        images_flat: List[List[Image.Image]] = []

        for ex in examples:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{VISION_TOKEN}{ex['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            assistant = ex["assistant_text"]
            # Inject vision slots inline: each image after the global one is
            # attached at the dispatcher hand-off points.
            if "</zoom>" in assistant:
                assistant = assistant.replace(
                    "</zoom>\n<think>",
                    f"</zoom>\n{VISION_TOKEN}\n<think>",
                    1,
                )
            if "</seg>" in assistant:
                assistant = assistant.replace(
                    "</seg>\n<think>",
                    f"</seg>\n{VISION_TOKEN}\n<think>",
                    1,
                )

            full = prompt + assistant + "<|im_end|>"
            texts_prompt.append(prompt)
            texts_full.append(full)
            images_flat.append(ex["images"])

        tok = processor(
            text=texts_full,
            images=images_flat,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )

        labels = tok["input_ids"].clone()
        for i, p in enumerate(texts_prompt):
            plen = len(processor.tokenizer(p, add_special_tokens=False)["input_ids"])
            labels[i, :plen] = ignore_index
        labels[labels == pad_id] = ignore_index
        tok["labels"] = labels
        return tok

    return collate


# ---------------------------------------------------------------------------
# Config / model preparation.
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    model_name: str = DEFAULTS["model_name"]
    train_jsonl: str = DEFAULTS["train_jsonl"]
    output_dir: str = DEFAULTS["output_dir"]
    img_folders: List[str] = field(default_factory=lambda: list(DEFAULTS["img_dir_candidates"]))
    wandb_project: Optional[str] = None

    num_train_epochs: int = 3
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 4
    lr: float = 3e-5  # match ZoomEarth/run_scripts/train_sft.sh
    warmup_steps: int = 500  # match ZoomEarth/run_scripts/train_sft.sh
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    seed: int = 42
    dtype: torch.dtype = torch.bfloat16
    resume_from_checkpoint: bool = True
    save_steps: int = 200
    log_steps: int = 10
    print_steps: int = 20
    max_pixels: int = 64 * 64 * 28 * 28
    max_length: int = 4096

    freeze_vision: bool = False
    gradient_checkpointing: bool = True


def prepare_model_and_processor(config: TrainingConfig):
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        max_pixels=config.max_pixels,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required when gradient_checkpointing is on without PEFT — the input
        # embedding's requires_grad must be set so backprop reaches all
        # parameters through the checkpointed blocks.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if config.freeze_vision:
        # transformers 5.x nests visual under model.model.visual
        visual = getattr(getattr(model, "model", model), "visual", None) \
            or getattr(model, "visual", None)
        if visual is None:
            raise RuntimeError("Could not locate vision tower under model.[model.]visual")
        for p in visual.parameters():
            p.requires_grad = False

    return model, processor


def prepare_dataloader(config: TrainingConfig, collate):
    with open(config.train_jsonl) as f:
        records = [json.loads(line) for line in f]

    before = len(records)
    records = [
        r for r in records
        if _find_image(r["image_name"], config.img_folders) is not None
    ]
    print(f"[data] {len(records)}/{before} records have resolvable images")
    if not records:
        raise RuntimeError(
            f"no images resolvable under {config.img_folders} — check paths"
        )

    dataset = ZoomSegDataset(records, config.img_folders)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_per_gpu,
        collate_fn=collate,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    return loader, len(dataset)


def prepare_optimizer_and_scheduler(config: TrainingConfig, model, num_training_steps: int):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Checkpoint helpers.
# ---------------------------------------------------------------------------
def save_checkpoint(accelerator, model, processor, epoch, step, config, loss):
    checkpoint_dir = f"{config.output_dir}/checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    info = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "latest_checkpoint": checkpoint_dir,
    }
    if accelerator.is_main_process:
        with open(f"{config.output_dir}/training_info.json", "w") as f:
            json.dump(info, f)
    accelerator.save_state(checkpoint_dir, safe_serialization=False)


def save_hf_format(accelerator, model, processor, config):
    """Save HF-compatible weights for downstream inference."""
    if not accelerator.is_main_process:
        return
    out = f"{config.output_dir}/final_hf"
    os.makedirs(out, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        out,
        is_main_process=True,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    processor.save_pretrained(out)
    accelerator.print(f"[done] HF-format model saved to {out}")


def load_checkpoint(accelerator, checkpoint_dir):
    accelerator.print(f"[resume] loading {checkpoint_dir}")
    accelerator.load_state(checkpoint_dir)


# ---------------------------------------------------------------------------
# Train loop.
# ---------------------------------------------------------------------------
def train(args):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    config = TrainingConfig(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        img_folders=(args.img_dir or []) + DEFAULTS["img_dir_candidates"],
        wandb_project=args.wandb_project,
        num_train_epochs=args.num_train_epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        dtype=dtype,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        print_steps=args.print_steps,
        max_pixels=args.max_pixels,
        max_length=args.max_length,
        freeze_vision=args.freeze_vision,
        gradient_checkpointing=not args.no_grad_ckpt,
    )

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=["tensorboard"],
        project_dir=config.output_dir,
    )

    model, processor = prepare_model_and_processor(config)
    collate = build_collator(processor, config.max_length)
    dataloader, dataset_len = prepare_dataloader(config, collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    dataloader, model, optimizer = accelerator.prepare(
        dataloader, model, optimizer
    )
    # Compute schedule length after accelerator.prepare(): in multi-GPU mode the
    # dataloader is sharded per process. Computing it before prepare() overstates
    # total updates by world_size and makes the progress bar / LR warmup too long.
    micro_steps_per_epoch = len(dataloader)
    update_steps_per_epoch = math.ceil(micro_steps_per_epoch / config.gradient_accumulation_steps)
    num_training_steps = update_steps_per_epoch * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(config.warmup_steps, max(1, num_training_steps // 2)),
        num_training_steps=num_training_steps,
    )
    scheduler = accelerator.prepare(scheduler)
    accelerator.print(
        f"[sched] dataset={dataset_len} per-rank_micro_steps/epoch={micro_steps_per_epoch} "
        f"update_steps/epoch={update_steps_per_epoch} total_updates={num_training_steps} "
        f"world_size={accelerator.num_processes}"
    )

    progress_bar = tqdm(total=num_training_steps, disable=not accelerator.is_local_main_process)
    starting_epoch = 1
    global_step = 0
    skipped_dataloader = dataloader

    info_path = f"{config.output_dir}/training_info.json"
    if config.resume_from_checkpoint and os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        starting_epoch = info["epoch"]
        global_step = info["step"]
        load_checkpoint(accelerator, info["latest_checkpoint"])
        progress_bar.update(global_step)
        accelerator.print(f"[resume] from epoch {starting_epoch} step {global_step}")
        skip = global_step * config.gradient_accumulation_steps % micro_steps_per_epoch
        skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=skip)

    proj = config.wandb_project or (
        f"VQA-pilot0502-fullsft-{date.today()}_"
        f"{datetime.now().strftime('%H_%M_%S')}"
    )
    accelerator.init_trackers(
        project_name=proj,
        init_kwargs={
            "wandb": {"name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")},
            "tensorboard": {},
        },
        config={k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
                for k, v in vars(config).items()},
    )

    unwrapped = accelerator.unwrap_model(model)
    total_params = sum(p.numel() for p in unwrapped.parameters())
    trainable_params = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
    accelerator.print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100*trainable_params/total_params:.4f}%)"
    )

    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_loss_count = 0
    dataloader_step = 0
    grad_norm = None

    for epoch in range(1, config.num_train_epochs + 1):
        if epoch < starting_epoch:
            continue
        loader = skipped_dataloader if epoch == starting_epoch else dataloader
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss

                total_loss += loss.detach()
                total_loss_count += 1
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                dataloader_step += 1

                if dataloader_step % config.gradient_accumulation_steps == 0:
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % config.print_steps == 0:
                        accelerator.print(
                            f"epoch={epoch} step={global_step} loss={loss.item():.4f}"
                        )

                    if global_step % config.log_steps == 0:
                        log_data = {
                            "train/loss": (
                                accelerator.gather(total_loss).detach().sum().item()
                                / accelerator.num_processes / max(1, total_loss_count)
                            ),
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/global_step": global_step,
                            "train/epoch": global_step / num_training_steps * config.num_train_epochs,
                            "train/grad_norm": (
                                grad_norm.detach().item()
                                if isinstance(grad_norm, torch.Tensor) else (grad_norm or 0.0)
                            ),
                        }
                        accelerator.log(log_data, step=global_step)
                        total_loss = torch.tensor(0.0, device=accelerator.device)
                        total_loss_count = 0
                        accelerator.wait_for_everyone()

                    if global_step % config.save_steps == 0:
                        save_checkpoint(accelerator, model, processor, epoch, global_step, config, loss.item())

    save_checkpoint(accelerator, model, processor, epoch, global_step, config, loss.item())
    save_hf_format(accelerator, model, processor, config)
    accelerator.end_training()


def parse_args():
    p = argparse.ArgumentParser(description="Full-parameter SFT for v0502 zoom_seg dataset")
    p.add_argument("--model_name", default=DEFAULTS["model_name"])
    p.add_argument("--train_jsonl", default=DEFAULTS["train_jsonl"])
    p.add_argument("--output_dir", default=DEFAULTS["output_dir"])
    p.add_argument("--img_dir", action="append", default=None,
                   help="extra dirs to search for image files (repeatable)")
    p.add_argument("--wandb_project", default=None)

    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size_per_gpu", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--resume_from_checkpoint", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume_from_checkpoint", action="store_false")

    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--print_steps", type=int, default=20)
    p.add_argument("--max_pixels", type=int, default=64 * 64 * 28 * 28)
    p.add_argument("--max_length", type=int, default=4096)

    p.add_argument("--freeze_vision", action="store_true",
                   help="freeze vision tower; default off to match ZoomEarth full-FT")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="disable gradient checkpointing (faster but uses more memory)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
