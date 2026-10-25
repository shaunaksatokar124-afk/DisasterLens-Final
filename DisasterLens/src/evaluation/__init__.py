"""DisasterLens AI — evaluation sub-package."""

from .metrics import (
    compute_confusion_matrix,
    accumulate_confusion_matrix,
    per_class_metrics_from_cm,
    pixel_accuracy,
    mean_metric,
    evaluate_masks,
    evaluate_dataset,
    DEFAULT_CLASS_NAMES,
    NUM_CLASSES,
    IGNORE_INDEX,
)

__all__ = [
    "compute_confusion_matrix",
    "accumulate_confusion_matrix",
    "per_class_metrics_from_cm",
    "pixel_accuracy",
    "mean_metric",
    "evaluate_masks",
    "evaluate_dataset",
    "DEFAULT_CLASS_NAMES",
    "NUM_CLASSES",
    "IGNORE_INDEX",
]
