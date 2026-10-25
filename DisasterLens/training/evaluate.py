"""
DisasterLens AI -- Evaluation Script
=====================================
Loads a trained checkpoint, runs inference on the held-out test split,
and saves ``artifacts/evaluation_results.json`` for the Analytics page.

Usage
-----
    python training/evaluate.py --config config.yaml --checkpoint path/to/best_model.pth

The script auto-discovers the best checkpoint in checkpoint_dir if
``--checkpoint`` is not supplied.

Output
------
Saves ``evaluation_results.json`` in the path specified by
``paths.evaluation_results_file`` in config.yaml. This file is consumed
directly by ``pages/2_Analytics.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

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
        description="Evaluate a trained DisasterLens AI checkpoint on the test set"
    )
    p.add_argument("--config",     default="config.yaml",
                   help="Path to config.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pth checkpoint (auto-detected from checkpoint_dir if omitted)")
    p.add_argument("--split",      default="test",
                   choices=["test", "val"],
                   help="Which split to evaluate on (default: test)")
    p.add_argument("--output",     default=None,
                   help="Override output JSON path")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dry-run",    action="store_true",
                   help="Evaluate 1 batch only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoint(config: dict) -> Optional[Path]:
    """Auto-discover best_model.pth or any .pth in checkpoint_dir."""
    ckpt_dir_str = config.get("paths", {}).get("checkpoint_dir")
    if not ckpt_dir_str:
        return None
    ckpt_dir = Path(ckpt_dir_str)
    if not ckpt_dir.exists():
        return None
    for name in ("best_model.pth", "best.pth", "checkpoint_best.pth"):
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    candidates = sorted(ckpt_dir.glob("*.pth"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    loader,
    device,
    num_classes: int,
    ignore_index: int,
    dry_run: bool,
) -> dict:
    """Run inference on the loader and return an evaluation results dict."""
    import torch
    from src.evaluation.metrics import accumulate_confusion_matrix, evaluate_dataset

    model.eval()
    running_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_batches  = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device,  non_blocking=True)

            logits = model(images)
            preds  = logits.argmax(dim=1)

            preds_np = preds.cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)

            for p, t in zip(preds_np, masks_np):
                accumulate_confusion_matrix(running_cm, p, t,
                                            num_classes, ignore_index)

            n_batches += 1
            if (batch_idx + 1) % 10 == 0:
                logger.info("  evaluated %d batches", batch_idx + 1)

            if dry_run:
                break

    logger.info("Total batches evaluated: %d", n_batches)
    return evaluate_dataset(running_cm, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tc          = config.get("training", {})
    mc          = config.get("model",    {})
    pc          = config.get("paths",    {})
    ignore_idx  = config.get("ignore_index", 255)
    num_classes = mc.get("num_classes", 5)
    img_size    = mc.get("image_size",  512)
    batch_sz    = args.batch_size or tc.get("batch_size", 8)
    n_workers   = tc.get("num_workers", 0)

    # ── Device ────────────────────────────────────────────────────────────────
    import torch
    pref   = config.get("inference", {}).get("device", "auto")
    device = torch.device(
        "cuda" if (pref == "auto" and torch.cuda.is_available()) else "cpu"
    )
    logger.info("Device: %s", device)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint(config)
    if ckpt_path is None or not ckpt_path.exists():
        logger.error(
            "No checkpoint found. Set 'checkpoint_dir' in config.yaml or "
            "pass --checkpoint <path>"
        )
        sys.exit(1)
    logger.info("Checkpoint: %s", ckpt_path)

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.models.unet_resnet import build_model

    model = build_model(
        encoder_name    = mc.get("encoder",         "resnet34"),
        encoder_weights = None,          # weights come from checkpoint
        in_channels     = mc.get("in_channels", 6),
        num_classes     = num_classes,
    )
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    logger.info("Model loaded from %s  (epoch %s, val_mIoU=%s)",
                ckpt_path.name,
                ckpt.get("epoch", "?"),
                ckpt.get("val_mean_iou", "?"))

    # ── Dataset ───────────────────────────────────────────────────────────────
    from src.data.dataset import (
        DatasetNotFoundError, _discover_pairs, split_pairs_by_event
    )
    from src.data.transforms import get_val_transform
    from src.data.dataset import XBDDataset, collate_batch
    from torch.utils.data import DataLoader

    dataset_root_str = pc.get("dataset_root")
    if not dataset_root_str:
        raise DatasetNotFoundError("'dataset_root' not set in config.yaml")

    pairs = _discover_pairs(Path(dataset_root_str))
    _, val_pairs, test_pairs = split_pairs_by_event(
        pairs,
        train_frac = tc.get("train_split", 0.70),
        val_frac   = tc.get("val_split",   0.15),
        seed       = tc.get("seed",        42),
    )
    eval_pairs = test_pairs if args.split == "test" else val_pairs
    logger.info("Evaluating on %s split: %d samples", args.split, len(eval_pairs))

    val_tfm = get_val_transform(img_size)

    eval_ds = XBDDataset(eval_pairs, image_size=img_size, transform=val_tfm)
    eval_loader = DataLoader(
        eval_ds,
        batch_size  = batch_sz,
        shuffle     = False,
        num_workers = n_workers,
        pin_memory  = torch.cuda.is_available(),
        collate_fn  = collate_batch,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Running evaluation...")
    results = evaluate(model, eval_loader, device, num_classes, ignore_idx,
                       dry_run=args.dry_run)

    # Annotate with checkpoint info
    results["checkpoint"] = str(ckpt_path.name)
    results["split"]      = args.split
    results["num_samples"]= len(eval_ds)

    # ── Log summary ───────────────────────────────────────────────────────────
    logger.info("─── Evaluation Results ──────────────────────────────────")
    logger.info("  Pixel Accuracy : %s", results.get("accuracy"))
    logger.info("  Mean IoU       : %s", results.get("mean_iou"))
    logger.info("  Mean F1        : %s", results.get("mean_f1"))
    for cls_name, m in results.get("per_class", {}).items():
        if m.get("support", 0) > 0:
            logger.info("  %-15s  IoU=%.4f  F1=%.4f  support=%d",
                        cls_name,
                        m.get("iou")  or 0.0,
                        m.get("f1")   or 0.0,
                        m.get("support", 0))
    logger.info("──────────────────────────────────────────────────────────")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path_str = (
        args.output or
        pc.get("evaluation_results_file", "artifacts/evaluation_results.json")
    )
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
