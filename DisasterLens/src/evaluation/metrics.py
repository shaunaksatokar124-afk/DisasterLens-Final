"""
DisasterLens AI -- Evaluation Metrics
=======================================
Computes segmentation metrics for the 5-class damage assessment task.

Public API
----------
compute_confusion_matrix(pred, true, num_classes, ignore_index)
    Build a (num_classes x num_classes) confusion matrix from two flat masks.

per_class_metrics_from_cm(cm)
    Derive precision, recall, F1, and IoU per class from a confusion matrix.

pixel_accuracy(cm)
    Overall pixel accuracy from a confusion matrix.

evaluate_masks(pred_mask, true_mask, num_classes, ignore_index, class_names)
    One-stop function: returns a results dict compatible with
    ``artifacts/evaluation_results.json`` and ``pages/2_Analytics.py``.

accumulate_confusion_matrix(running_cm, pred, true, num_classes, ignore_index)
    Accumulates per-batch confusion matrices for epoch-level evaluation.

Output format (matches Analytics page expectations)
---------------------------------------------------
{
    "mean_iou":        float,
    "mean_f1":         float,
    "mean_precision":  float,
    "mean_recall":     float,
    "accuracy":        float,
    "class_names":     ["Background", "No Damage", ...],
    "per_class": {
        "Background": {"precision": f, "recall": f, "f1": f, "iou": f, "support": n},
        ...
    },
    "confusion_matrix": [[int, ...], ...],  # num_classes x num_classes, row=true
}
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLASS_NAMES: List[str] = [
    "Background",
    "No Damage",
    "Minor Damage",
    "Major Damage",
    "Destroyed",
]

NUM_CLASSES: int = 5
IGNORE_INDEX: int = 255


# ---------------------------------------------------------------------------
# Core: Confusion Matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    pred: np.ndarray,
    true: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Build a confusion matrix from predicted and ground-truth masks.

    Pixels where ``true == ignore_index`` are excluded from the matrix.
    Predicted pixels with value ``ignore_index`` are also excluded.

    Args:
        pred:         Flat or multi-dim uint8/int array of predicted class indices.
        true:         Flat or multi-dim uint8/int array of ground-truth class indices.
        num_classes:  Number of classes (matrix side length).
        ignore_index: Pixel value to exclude from all computations.

    Returns:
        ``(num_classes, num_classes)`` int64 array.
        Row index = true class, column index = predicted class.

    Raises:
        ValueError: If ``pred`` and ``true`` have different shapes.
    """
    pred = np.asarray(pred).ravel().astype(np.int64)
    true = np.asarray(true).ravel().astype(np.int64)

    if pred.shape != true.shape:
        raise ValueError(
            f"pred and true must have the same shape; "
            f"got pred={pred.shape}, true={true.shape}"
        )

    # Mask out ignore pixels in both pred and true
    valid = (true != ignore_index) & (pred != ignore_index)
    pred  = pred[valid]
    true  = true[valid]

    # Fast histogram-based confusion matrix
    # Encode as a single integer: true * num_classes + pred
    k = num_classes
    encoded = true * k + pred
    # Only keep entries where both indices are in [0, num_classes)
    in_range = (true >= 0) & (true < k) & (pred >= 0) & (pred < k)
    encoded  = encoded[in_range]

    cm = np.bincount(encoded, minlength=k * k).reshape(k, k).astype(np.int64)
    return cm


def accumulate_confusion_matrix(
    running_cm: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Add a single-sample confusion matrix to a running accumulator.

    Args:
        running_cm:   Accumulated ``(num_classes, num_classes)`` matrix.
                      Pass ``np.zeros((num_classes, num_classes), dtype=np.int64)``
                      for the first batch.
        pred:         Predicted mask for the current sample/batch.
        true:         Ground-truth mask for the current sample/batch.
        num_classes:  Number of classes.
        ignore_index: Pixel value to exclude.

    Returns:
        Updated ``running_cm`` (in-place addition, also returned for chaining).
    """
    running_cm += compute_confusion_matrix(pred, true, num_classes, ignore_index)
    return running_cm


# ---------------------------------------------------------------------------
# Per-class metrics from confusion matrix
# ---------------------------------------------------------------------------

def per_class_metrics_from_cm(
    cm: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute per-class Precision, Recall, F1, IoU from a confusion matrix.

    Classes with zero ground-truth support (TP + FN == 0) still appear in the
    output but have NaN values for metrics, so they can be excluded from mean
    calculations.

    Args:
        cm: ``(num_classes, num_classes)`` int64 confusion matrix.
            Rows = true, columns = predicted.

    Returns:
        Dict keyed by class index (0 … num_classes-1), each value is::

            {
                "precision": float,  # NaN if TP+FP == 0
                "recall":    float,  # NaN if TP+FN == 0
                "f1":        float,  # NaN if P+R == 0
                "iou":       float,  # NaN if TP+FP+FN == 0
                "support":   int,    # total true pixels for this class (TP+FN)
            }
    """
    num_classes = cm.shape[0]
    metrics: Dict[int, Dict[str, float]] = {}

    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp   # predicted as i but not truly i
        fn = int(cm[i, :].sum()) - tp   # truly i but not predicted as i
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        iou_denom = tp + fp + fn
        iou = tp / iou_denom if iou_denom > 0 else float("nan")

        metrics[i] = {
            "precision": round(float(precision), 6),
            "recall":    round(float(recall),    6),
            "f1":        round(float(f1),        6),
            "iou":       round(float(iou),       6),
            "support":   support,
        }

    return metrics


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def pixel_accuracy(cm: np.ndarray) -> float:
    """Overall pixel accuracy: fraction of correctly classified valid pixels.

    Args:
        cm: ``(num_classes, num_classes)`` confusion matrix.

    Returns:
        Float in [0, 1], or ``nan`` if the matrix is all zeros.
    """
    total   = cm.sum()
    correct = np.trace(cm)
    return float(correct / total) if total > 0 else float("nan")


def mean_metric(
    per_class: Dict[int, Dict[str, float]],
    key: str,
    include_background: bool = False,
    require_support: bool = True,
) -> float:
    """Average a metric across classes, optionally excluding background (idx 0).

    Args:
        per_class:          Output of ``per_class_metrics_from_cm``.
        key:                Metric name: ``"precision"``, ``"recall"``, ``"f1"``,
                            or ``"iou"``.
        include_background: If False (default), class 0 is excluded from the mean.
                            Background dominates pixel counts and including it
                            inflates overall scores.
        require_support:    If True (default), classes with zero ground-truth
                            support (support == 0) are excluded from the mean.

    Returns:
        Mean float value, or ``nan`` if no valid classes remain.
    """
    values: List[float] = []
    for class_idx, m in per_class.items():
        if not include_background and class_idx == 0:
            continue
        if require_support and m["support"] == 0:
            continue
        v = m[key]
        if not np.isnan(v):
            values.append(v)

    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# One-stop evaluation function
# ---------------------------------------------------------------------------

def evaluate_masks(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    class_names: Optional[Sequence[str]] = None,
    include_background_in_mean: bool = False,
) -> dict:
    """Evaluate a single predicted mask against ground truth.

    Args:
        pred_mask:    ``(H, W)`` predicted class index mask.
        true_mask:    ``(H, W)`` ground-truth class index mask.
        num_classes:  Number of classes (default 5).
        ignore_index: Pixel value excluded from all metrics (default 255).
        class_names:  Optional list of class name strings.  Defaults to
                      ``DEFAULT_CLASS_NAMES``.
        include_background_in_mean: Whether to include class 0 when computing
                      mean IoU / F1.  Defaults to False (standard practice).

    Returns:
        Results dict compatible with ``artifacts/evaluation_results.json``.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES[:num_classes]

    cm          = compute_confusion_matrix(pred_mask, true_mask, num_classes, ignore_index)
    per_class_i = per_class_metrics_from_cm(cm)

    # Build per_class dict keyed by name (for JSON serialisation)
    per_class_named: Dict[str, dict] = {}
    for idx, m in per_class_i.items():
        name = class_names[idx] if idx < len(class_names) else str(idx)
        # Replace NaN with None for JSON compatibility
        per_class_named[name] = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in m.items()
        }

    def _safe(v: float) -> Optional[float]:
        return None if np.isnan(v) else round(v, 6)

    return {
        "mean_iou":       _safe(mean_metric(per_class_i, "iou",       include_background=include_background_in_mean)),
        "mean_f1":        _safe(mean_metric(per_class_i, "f1",        include_background=include_background_in_mean)),
        "mean_precision": _safe(mean_metric(per_class_i, "precision", include_background=include_background_in_mean)),
        "mean_recall":    _safe(mean_metric(per_class_i, "recall",    include_background=include_background_in_mean)),
        "accuracy":       _safe(pixel_accuracy(cm)),
        "class_names":    list(class_names),
        "per_class":      per_class_named,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_dataset(
    running_cm: np.ndarray,
    num_classes: int = NUM_CLASSES,
    class_names: Optional[Sequence[str]] = None,
    include_background_in_mean: bool = False,
) -> dict:
    """Compute final metrics from an accumulated confusion matrix.

    This is the companion to ``accumulate_confusion_matrix`` for use in
    training evaluation loops where the CM is built batch-by-batch.

    Args:
        running_cm:   Accumulated ``(num_classes, num_classes)`` confusion matrix.
        num_classes:  Number of classes.
        class_names:  Optional list of class name strings.
        include_background_in_mean: See ``evaluate_masks``.

    Returns:
        Same results dict format as ``evaluate_masks``.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES[:num_classes]

    per_class_i = per_class_metrics_from_cm(running_cm)

    per_class_named: Dict[str, dict] = {}
    for idx, m in per_class_i.items():
        name = class_names[idx] if idx < len(class_names) else str(idx)
        per_class_named[name] = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in m.items()
        }

    def _safe(v: float) -> Optional[float]:
        return None if np.isnan(v) else round(v, 6)

    return {
        "mean_iou":       _safe(mean_metric(per_class_i, "iou",       include_background=include_background_in_mean)),
        "mean_f1":        _safe(mean_metric(per_class_i, "f1",        include_background=include_background_in_mean)),
        "mean_precision": _safe(mean_metric(per_class_i, "precision", include_background=include_background_in_mean)),
        "mean_recall":    _safe(mean_metric(per_class_i, "recall",    include_background=include_background_in_mean)),
        "accuracy":       _safe(pixel_accuracy(running_cm)),
        "class_names":    list(class_names),
        "per_class":      per_class_named,
        "confusion_matrix": running_cm.tolist(),
    }
