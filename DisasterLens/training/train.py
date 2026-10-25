"""
DisasterLens AI -- Training Pipeline
=====================================
Trains a U-Net + ResNet-34 segmentation model on the xBD disaster damage
dataset.  Run from the project root:

    python training/train.py --config config.yaml

Key design decisions
--------------------
- Loss: 50% FocalLoss (γ=2) + 50% DiceLoss — combats class imbalance.
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4).
- Scheduler: CosineAnnealingLR over T_max epochs.
- Weighted sampler: tiles with more damaged buildings are sampled more often.
- Checkpoint: saved whenever val_mean_iou improves (top-k=3 kept).
- Artifacts: training_history.json updated after every epoch so the
  Analytics page can display live-ish progress.
- Splits: disaster-event-level (no geographic leakage).
- Reproducible: all random state seeded.

Usage
-----
    python training/train.py                      # uses config.yaml defaults
    python training/train.py --epochs 10          # override epochs
    python training/train.py --dry-run            # 1 batch, no checkpoint saved
    python training/train.py --resume path/to/ckpt.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path so src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DisasterLens AI damage segmentation model"
    )
    p.add_argument("--config",    default="config.yaml", help="Path to config.yaml")
    p.add_argument("--epochs",    type=int,   default=None, help="Override epochs")
    p.add_argument("--batch-size",type=int,   default=None, help="Override batch size")
    p.add_argument("--lr",        type=float, default=None, help="Override learning rate")
    p.add_argument("--resume",    default=None,             help="Checkpoint to resume from")
    p.add_argument("--dry-run",   action="store_true",
                   help="Train 1 batch only — smoke-test without saving checkpoints")
    p.add_argument("--no-amp",    action="store_true",
                   help="Disable automatic mixed precision (AMP)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed: %d", seed)


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

def compute_sample_weights(pairs: list) -> list:
    """Assign a sampling weight to each training pair.

    Tiles with more damaged buildings are sampled more frequently to
    counteract the heavy class imbalance (most pixels are background).

    Weight formula:
        w = 1.0 + log(1 + n_damaged)
    where n_damaged = number of polygons with subtype != "no-damage".

    Falls back to uniform weight 1.0 if the JSON cannot be read.

    Args:
        pairs: List of pair dicts (output of _discover_pairs).

    Returns:
        List of float weights, same length as pairs.
    """
    weights = []
    for pair in pairs:
        label_path = pair.get("post_label")
        n_damaged  = 0
        if label_path and Path(label_path).exists():
            try:
                with open(label_path, encoding="utf-8") as f:
                    data = json.load(f)
                features = data.get("features", {}).get("xy", [])
                for feat in features:
                    sub = feat.get("properties", {}).get("subtype", "no-damage")
                    if sub not in ("no-damage", "un-classified"):
                        n_damaged += 1
            except Exception:
                pass
        weights.append(1.0 + np.log1p(n_damaged))
    return weights


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_criterion(config: dict, ignore_index: int, device):
    """Construct combined Focal + Dice loss."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch required. "
            "Install: pip install segmentation-models-pytorch"
        )
    import torch.nn as nn

    tc = config.get("training", {})
    gamma      = tc.get("focal_loss_gamma", 2.0)
    lw         = tc.get("loss_weights", {})
    focal_w    = float(lw.get("focal", 0.5))
    dice_w     = float(lw.get("dice",  0.5))

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.focal = smp.losses.FocalLoss(
                mode="multiclass",
                gamma=gamma,
                ignore_index=ignore_index,
                reduction="mean",
            )
            self.dice  = smp.losses.DiceLoss(
                mode="multiclass",
                ignore_index=ignore_index,
                from_logits=True,
            )
            self.fw = focal_w
            self.dw = dice_w

        def forward(self, logits, targets):
            return self.fw * self.focal(logits, targets) + \
                   self.dw * self.dice(logits, targets)

    return CombinedLoss().to(device)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_loaders(config: dict, train_pairs: list, val_pairs: list):
    """Build train and val DataLoaders from pair lists.

    Args:
        config:      Loaded config.yaml dict.
        train_pairs: Training pair dicts.
        val_pairs:   Validation pair dicts.

    Returns:
        Tuple (train_loader, val_loader).
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from src.data.dataset import XBDDataset, collate_batch
    from src.data.transforms import get_train_transform, get_val_transform

    tc        = config.get("training", {})
    mc        = config.get("model",    {})
    img_size  = mc.get("image_size",  512)
    batch_sz  = tc.get("batch_size",  8)
    n_workers = tc.get("num_workers", 0)  # 0 on Windows to avoid spawn issues

    train_tfm = get_train_transform(img_size)
    val_tfm   = get_val_transform(img_size)

    train_ds = XBDDataset(train_pairs, image_size=img_size, transform=train_tfm)
    val_ds   = XBDDataset(val_pairs,   image_size=img_size, transform=val_tfm)

    logger.info("Train dataset: %d samples", len(train_ds))
    logger.info("Val   dataset: %d samples", len(val_ds))

    # Weighted sampler for class imbalance
    weights = compute_sample_weights(train_pairs)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_sz,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_iou: float,
    ckpt_dir: Path,
    tag: str,
    save_top_k: int = 3,
) -> Path:
    """Save a model checkpoint and prune old ones beyond save_top_k."""
    import torch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:03d}_iou{val_iou:.4f}.pth"
    torch.save({
        "epoch":            epoch,
        "val_mean_iou":     val_iou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, ckpt_path)
    logger.info("Saved checkpoint: %s", ckpt_path.name)

    # Keep only top-k checkpoints (highest IoU)
    all_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pth"))
    if len(all_ckpts) > save_top_k:
        # Sort by IoU in filename (descending); remove lowest
        def _iou_from_name(p: Path) -> float:
            try:
                return float(p.stem.split("iou")[-1])
            except ValueError:
                return 0.0
        ranked = sorted(all_ckpts, key=_iou_from_name, reverse=True)
        for old in ranked[save_top_k:]:
            old.unlink(missing_ok=True)
            logger.debug("Removed old checkpoint: %s", old.name)

    # Always maintain a symlink/copy named best_model.pth
    best_path = ckpt_dir / "best_model.pth"
    if best_path.exists():
        best_path.unlink()
    import shutil
    shutil.copy2(ckpt_path, best_path)

    return ckpt_path


def save_last_checkpoint(model, optimizer, scheduler, epoch: int, val_iou: float,
                         best_iou: float, ckpt_dir: Path) -> Path:
    """Overwrite ``last_checkpoint.pth`` every epoch so a dropped Colab session
    can be resumed with ``--resume <ckpt_dir>/last_checkpoint.pth``."""
    import torch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp  = ckpt_dir / "last_checkpoint.pth.tmp"
    path = ckpt_dir / "last_checkpoint.pth"
    torch.save({
        "epoch":                epoch,
        "val_mean_iou":         val_iou,
        "best_val_mean_iou":    best_iou,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(model, optimizer, scheduler, path: str, device) -> int:
    """Load a checkpoint; returns the epoch to resume from."""
    import torch
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # old PyTorch without weights_only
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    iou   = ckpt.get("val_mean_iou", 0.0)
    logger.info("Resumed from %s  (epoch %d, val_iou=%.4f)", path, epoch, iou)
    # ``epoch`` is the 1-based count of completed epochs == next 0-based index
    return epoch


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def history_resume_best(path: str, device) -> float:
    """Best val IoU stored in a checkpoint (so resuming does not overwrite a better model)."""
    import torch
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        return float(ckpt.get("best_val_mean_iou", ckpt.get("val_mean_iou", 0.0)))
    except Exception:
        return 0.0


def load_history_file(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"train_loss": [], "val_loss": [], "val_mean_iou": [], "epochs_trained": 0}


def save_history_file(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler, dry_run: bool
) -> float:
    """Run one training epoch; returns mean loss."""
    import torch

    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = model(images)
            loss   = criterion(logits, masks)

        # A batch made only of ignore-index pixels yields a NaN loss; skip it so
        # one degenerate tile cannot poison the weights of a long run.
        if not torch.isfinite(loss):
            logger.warning("  batch %d: non-finite loss, skipped", batch_idx + 1)
            continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            logger.info("  batch %4d/%d  loss=%.4f",
                        batch_idx + 1, len(loader), loss.item())

        if dry_run:
            break

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

def validate_one_epoch(
    model, loader, criterion, device, num_classes: int, ignore_index: int,
    dry_run: bool
):
    """Run one validation epoch; returns (mean_val_loss, results_dict)."""
    import torch
    from src.evaluation.metrics import accumulate_confusion_matrix, evaluate_dataset

    model.eval()
    total_loss = 0.0
    n_batches  = 0
    running_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device,  non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, masks)

            preds = logits.argmax(dim=1)  # (B, H, W)

            # Accumulate CM on CPU
            preds_np = preds.cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)
            for p, t in zip(preds_np, masks_np):
                accumulate_confusion_matrix(running_cm, p, t,
                                            num_classes, ignore_index)

            total_loss += loss.item()
            n_batches  += 1

            if dry_run:
                break

    val_loss = total_loss / max(n_batches, 1)
    results  = evaluate_dataset(running_cm, num_classes=num_classes)
    return val_loss, results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    tc          = config.get("training", {})
    mc          = config.get("model",    {})
    pc          = config.get("paths",    {})
    ignore_idx  = config.get("ignore_index", 255)
    num_classes = mc.get("num_classes", 5)

    # Apply CLI overrides
    epochs    = args.epochs    or tc.get("epochs",       50)
    lr        = args.lr        or tc.get("learning_rate", 1e-4)
    wd        = tc.get("weight_decay", 1e-4)
    seed      = tc.get("seed", 42)
    save_top_k= tc.get("save_top_k", 3)

    seed_everything(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    import torch
    pref   = config.get("inference", {}).get("device", "auto")
    device = torch.device(
        "cuda" if (pref == "auto" and torch.cuda.is_available()) else
        "cpu"  if pref == "auto" else pref
    )
    logger.info("Device: %s", device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    from src.data.dataset import (
        DatasetNotFoundError, _discover_pairs, split_pairs_by_event
    )

    dataset_root_str = pc.get("dataset_root")
    if not dataset_root_str:
        raise DatasetNotFoundError(
            "'dataset_root' is not set in config.yaml. "
            "Download xBD from https://xview2.org"
        )

    logger.info("Discovering pairs in: %s", dataset_root_str)
    pairs = _discover_pairs(Path(dataset_root_str))

    train_pairs, val_pairs, test_pairs = split_pairs_by_event(
        pairs,
        train_frac = tc.get("train_split", 0.70),
        val_frac   = tc.get("val_split",   0.15),
        seed       = seed,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    train_loader, val_loader = build_loaders(config, train_pairs, val_pairs)

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.models.unet_resnet import build_model

    model = build_model(
        encoder_name    = mc.get("encoder",          "resnet34"),
        encoder_weights = mc.get("encoder_weights",  "imagenet"),
        in_channels     = mc.get("in_channels",      6),
        num_classes     = num_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: U-Net + %s  (%s params)",
                mc.get("encoder", "resnet34"), f"{total_params:,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = args.epochs or tc.get("scheduler_T_max", epochs),
        eta_min = lr * 0.01,
    )

    # AMP scaler (CUDA only)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("AMP enabled")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_criterion(config, ignore_idx, device)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_iou    = 0.0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        best_iou = max(history_resume_best(args.resume, device), 0.0)

    # ── Artifact paths ────────────────────────────────────────────────────────
    artifacts_dir  = Path(pc.get("artifacts_dir",           "artifacts"))
    ckpt_dir_str   = pc.get("checkpoint_dir")
    ckpt_dir       = Path(ckpt_dir_str) if ckpt_dir_str else artifacts_dir / "checkpoints"
    history_path   = Path(pc.get("training_history_file",   "artifacts/training_history.json"))

    history = load_history_file(history_path)
    if start_epoch > 0:  # resuming: drop any epochs recorded after the checkpoint
        for key in ("train_loss", "val_loss", "val_mean_iou"):
            history[key] = history.get(key, [])[:start_epoch]
        history["epochs_trained"] = start_epoch
    elif not args.dry_run:
        history = {"train_loss": [], "val_loss": [], "val_mean_iou": [], "epochs_trained": 0}

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting training: %d epochs, lr=%.1e, batch=%d",
                epochs, lr, config["training"].get("batch_size", 8))

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        logger.info("── Epoch %d/%d ──────────────────────────", epoch + 1, epochs)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            dry_run=args.dry_run,
        )

        val_loss, val_results = validate_one_epoch(
            model, val_loader, criterion, device, num_classes, ignore_idx,
            dry_run=args.dry_run,
        )

        scheduler.step()

        val_iou  = val_results.get("mean_iou")  or 0.0
        val_f1   = val_results.get("mean_f1")   or 0.0
        elapsed  = time.time() - epoch_start
        cur_lr   = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | "
            "val_mIoU=%.4f | val_mF1=%.4f | lr=%.2e | %.1fs",
            epoch + 1, train_loss, val_loss, val_iou, val_f1, cur_lr, elapsed,
        )

        # Save checkpoint on improvement
        if not args.dry_run and val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, val_iou, ckpt_dir, tag="best",
                save_top_k=save_top_k,
            )

        # Append to training history
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss,   6))
        history["val_mean_iou"].append(round(val_iou, 6))
        history["epochs_trained"] = epoch + 1
        if not args.dry_run:
            save_history_file(history_path, history)

        if not args.dry_run:
            save_last_checkpoint(model, optimizer, scheduler, epoch + 1,
                                 val_iou, best_iou, ckpt_dir)

        if args.dry_run:
            logger.info("Dry-run complete — stopping after 1 epoch.")
            break

    logger.info("Training complete. Best val_mIoU: %.4f", best_iou)
    if not args.dry_run:
        logger.info("Checkpoint directory: %s", ckpt_dir)
        logger.info("Training history:     %s", history_path)


if __name__ == "__main__":
    main()
