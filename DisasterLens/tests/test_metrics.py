"""
DisasterLens AI -- Evaluation Metrics Unit Tests
==================================================
All tests use synthetic masks constructed from numpy arrays.
No dataset or model checkpoint is required.

Coverage:
  - compute_confusion_matrix: shape, dtype, diagonal (perfect pred), off-diagonal
  - ignore_index: excluded from matrix; ignore in pred also excluded
  - per_class_metrics_from_cm: known TP/FP/FN values, edge cases (zero support, etc.)
  - pixel_accuracy: perfect, empty, partial
  - mean_metric: background exclusion, zero-support exclusion, NaN handling
  - evaluate_masks: end-to-end with known synthetic masks
  - accumulate_confusion_matrix: batch accumulation matches single full call
  - evaluate_dataset: consistency with evaluate_masks output format
  - JSON serialisability: no raw NaN values in output dict

Run with:
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    DEFAULT_CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    accumulate_confusion_matrix,
    compute_confusion_matrix,
    evaluate_dataset,
    evaluate_masks,
    mean_metric,
    per_class_metrics_from_cm,
    pixel_accuracy,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _perfect_pred(size: int = 100, num_classes: int = NUM_CLASSES) -> tuple:
    """Return (pred, true) where pred == true (perfect segmentation)."""
    true = np.arange(size, dtype=np.uint8) % num_classes
    pred = true.copy()
    return pred, true


def _single_class_masks(
    true_class: int, pred_class: int, size: int = 50
) -> tuple:
    """Both masks contain only one class each (may differ)."""
    true = np.full(size, true_class, dtype=np.uint8)
    pred = np.full(size, pred_class, dtype=np.uint8)
    return pred, true


# ===========================================================================
# compute_confusion_matrix
# ===========================================================================

class TestComputeConfusionMatrix:

    def test_shape(self):
        pred, true = _perfect_pred()
        cm = compute_confusion_matrix(pred, true)
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)

    def test_dtype(self):
        pred, true = _perfect_pred()
        cm = compute_confusion_matrix(pred, true)
        assert cm.dtype == np.int64

    def test_perfect_prediction_is_diagonal(self):
        pred, true = _perfect_pred(size=500)
        cm = compute_confusion_matrix(pred, true)
        off_diag = cm - np.diag(np.diag(cm))
        assert off_diag.sum() == 0, "Off-diagonal should be zero for perfect prediction"

    def test_perfect_prediction_diagonal_sums(self):
        """Each diagonal entry should equal the number of true pixels of that class."""
        size = 500
        pred, true = _perfect_pred(size=size)
        cm = compute_confusion_matrix(pred, true)
        for cls in range(NUM_CLASSES):
            expected = int(np.sum(true == cls))
            assert cm[cls, cls] == expected

    def test_total_counts_match_valid_pixels(self):
        true = np.array([0, 1, 2, IGNORE_INDEX, 3, 4], dtype=np.uint8)
        pred = np.array([0, 0, 2, 0,            3, 4], dtype=np.uint8)
        cm = compute_confusion_matrix(pred, true)
        # Total valid pixels = 5 (ignore excluded)
        assert cm.sum() == 5

    def test_ignore_index_excluded_from_true(self):
        size = 10
        true = np.full(size, IGNORE_INDEX, dtype=np.uint8)
        pred = np.zeros(size, dtype=np.uint8)
        cm = compute_confusion_matrix(pred, true)
        assert cm.sum() == 0, "All-ignore true mask should produce empty CM"

    def test_ignore_index_excluded_from_pred(self):
        true = np.array([1, 2, 3], dtype=np.uint8)
        pred = np.array([IGNORE_INDEX, 2, 3], dtype=np.uint8)
        cm = compute_confusion_matrix(pred, true)
        # First pixel is excluded (pred == ignore), so 2 valid pixels
        assert cm.sum() == 2

    def test_known_off_diagonal(self):
        """Class 1 predicted as class 2: expect cm[1, 2] == 1."""
        true = np.array([1], dtype=np.uint8)
        pred = np.array([2], dtype=np.uint8)
        cm = compute_confusion_matrix(pred, true)
        assert cm[1, 2] == 1
        assert cm.sum() == 1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_confusion_matrix(
                np.array([0, 1]), np.array([0, 1, 2])
            )

    def test_custom_num_classes(self):
        pred = np.array([0, 1, 2], dtype=np.uint8)
        true = np.array([0, 1, 2], dtype=np.uint8)
        cm = compute_confusion_matrix(pred, true, num_classes=3)
        assert cm.shape == (3, 3)
        assert np.trace(cm) == 3

    def test_2d_input_flattened(self):
        """2D arrays should be accepted and flattened automatically."""
        pred = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        true = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        cm = compute_confusion_matrix(pred, true)
        assert cm.sum() == 4  # 4 valid pixels


# ===========================================================================
# per_class_metrics_from_cm
# ===========================================================================

class TestPerClassMetricsFromCm:

    def test_perfect_iou_is_one(self):
        pred, true = _perfect_pred(size=500)
        cm = compute_confusion_matrix(pred, true)
        m  = per_class_metrics_from_cm(cm)
        for cls in range(NUM_CLASSES):
            if m[cls]["support"] > 0:
                assert abs(m[cls]["iou"] - 1.0) < 1e-9, \
                    f"Expected IoU=1 for class {cls}, got {m[cls]['iou']}"

    def test_perfect_precision_recall_f1(self):
        pred, true = _perfect_pred(size=500)
        cm = compute_confusion_matrix(pred, true)
        m  = per_class_metrics_from_cm(cm)
        for cls in range(NUM_CLASSES):
            if m[cls]["support"] > 0:
                assert abs(m[cls]["precision"] - 1.0) < 1e-9
                assert abs(m[cls]["recall"]    - 1.0) < 1e-9
                assert abs(m[cls]["f1"]        - 1.0) < 1e-9

    def test_known_tp_fp_fn(self):
        """Manually construct CM and verify arithmetic.

        true=[1,1,1], pred=[1,1,2]:
          class 1: TP=2, FP=0, FN=1 → P=1, R=2/3, F1=0.8, IoU=2/3
          class 2: TP=0, FP=1, FN=0 → P=0, R=nan, F1=nan, IoU=0
        """
        true = np.array([1, 1, 1], dtype=np.uint8)
        pred = np.array([1, 1, 2], dtype=np.uint8)
        cm   = compute_confusion_matrix(pred, true)
        m    = per_class_metrics_from_cm(cm)

        assert abs(m[1]["precision"] - 1.0)           < 1e-6
        assert abs(m[1]["recall"]    - 2/3)            < 1e-6
        assert abs(m[1]["f1"]        - 2*(1*(2/3))/(1 + 2/3)) < 1e-6
        assert abs(m[1]["iou"]       - 2/3)            < 1e-6
        assert m[1]["support"] == 3

        assert abs(m[2]["precision"] - 0.0) < 1e-6  # TP=0, FP=1 → P=0
        assert m[2]["support"] == 0                  # no true class-2 pixels

    def test_zero_support_class_has_nan_recall(self):
        """A class absent from ground truth should have NaN recall."""
        # Only class 0 in ground truth
        true = np.zeros(20, dtype=np.uint8)
        pred = np.zeros(20, dtype=np.uint8)
        cm   = compute_confusion_matrix(pred, true)
        m    = per_class_metrics_from_cm(cm)
        # Classes 1-4 have zero support → recall = NaN
        for cls in range(1, NUM_CLASSES):
            assert math.isnan(m[cls]["recall"]), \
                f"Class {cls} recall should be NaN (zero support)"

    def test_returns_all_classes(self):
        pred, true = _perfect_pred()
        cm = compute_confusion_matrix(pred, true)
        m  = per_class_metrics_from_cm(cm)
        assert set(m.keys()) == set(range(NUM_CLASSES))

    def test_metrics_in_zero_one_range(self):
        """Non-NaN metrics should be in [0, 1]."""
        pred = np.array([0, 1, 1, 2, 3, 4], dtype=np.uint8)
        true = np.array([0, 1, 2, 2, 3, 4], dtype=np.uint8)
        cm   = compute_confusion_matrix(pred, true)
        m    = per_class_metrics_from_cm(cm)
        for cls, metrics in m.items():
            for key in ("precision", "recall", "f1", "iou"):
                v = metrics[key]
                if not math.isnan(v):
                    assert 0.0 <= v <= 1.0, \
                        f"Class {cls} {key}={v} out of [0, 1]"


# ===========================================================================
# pixel_accuracy
# ===========================================================================

class TestPixelAccuracy:

    def test_perfect_prediction(self):
        pred, true = _perfect_pred(size=200)
        cm  = compute_confusion_matrix(pred, true)
        acc = pixel_accuracy(cm)
        assert abs(acc - 1.0) < 1e-9

    def test_all_wrong(self):
        """Every pixel misclassified: all true-class-1, predicted as class-2."""
        pred, true = _single_class_masks(true_class=1, pred_class=2, size=50)
        cm  = compute_confusion_matrix(pred, true)
        acc = pixel_accuracy(cm)
        assert abs(acc - 0.0) < 1e-9

    def test_half_right(self):
        """50 pixels correct, 50 wrong."""
        true = np.array([0] * 50 + [1] * 50, dtype=np.uint8)
        pred = np.array([0] * 50 + [0] * 50, dtype=np.uint8)
        cm  = compute_confusion_matrix(pred, true)
        acc = pixel_accuracy(cm)
        assert abs(acc - 0.5) < 1e-9

    def test_all_ignore_returns_nan(self):
        true = np.full(10, IGNORE_INDEX, dtype=np.uint8)
        pred = np.zeros(10, dtype=np.uint8)
        cm  = compute_confusion_matrix(pred, true)
        acc = pixel_accuracy(cm)
        assert math.isnan(acc)

    def test_empty_cm_returns_nan(self):
        cm  = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        acc = pixel_accuracy(cm)
        assert math.isnan(acc)


# ===========================================================================
# mean_metric
# ===========================================================================

class TestMeanMetric:

    def _make_per_class(self, values: dict) -> dict:
        """Build a minimal per_class dict with given iou values keyed by class int."""
        result = {}
        for idx in range(NUM_CLASSES):
            iou = values.get(idx, float("nan"))
            result[idx] = {"precision": iou, "recall": iou, "f1": iou,
                           "iou": iou, "support": 0 if math.isnan(iou) else 10}
        return result

    def test_excludes_background_by_default(self):
        per_class = self._make_per_class({0: 1.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5})
        m = mean_metric(per_class, "iou", include_background=False)
        # Background excluded → mean of [0.5, 0.5, 0.5, 0.5]
        assert abs(m - 0.5) < 1e-9

    def test_includes_background_when_requested(self):
        per_class = self._make_per_class({0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
        m = mean_metric(per_class, "iou", include_background=True)
        # Average of [0.0, 1.0, 1.0, 1.0, 1.0] = 0.8
        assert abs(m - 0.8) < 1e-9

    def test_excludes_zero_support_classes(self):
        per_class = self._make_per_class({1: 0.6, 2: 0.8})
        # Classes 0, 3, 4 have NaN iou and support=0 → excluded
        m = mean_metric(per_class, "iou")
        assert abs(m - 0.7) < 1e-9  # mean(0.6, 0.8)

    def test_all_nan_returns_nan(self):
        per_class = self._make_per_class({})  # all NaN
        m = mean_metric(per_class, "iou")
        assert math.isnan(m)

    def test_single_valid_class(self):
        per_class = self._make_per_class({2: 0.75})
        m = mean_metric(per_class, "iou")
        assert abs(m - 0.75) < 1e-9


# ===========================================================================
# evaluate_masks (end-to-end)
# ===========================================================================

class TestEvaluateMasks:

    def test_output_keys_present(self):
        pred, true = _perfect_pred(size=200)
        result = evaluate_masks(pred, true)
        required_keys = {
            "mean_iou", "mean_f1", "mean_precision", "mean_recall",
            "accuracy", "class_names", "per_class", "confusion_matrix",
        }
        assert required_keys.issubset(result.keys())

    def test_perfect_prediction_accuracy_one(self):
        pred, true = _perfect_pred(size=500)
        result = evaluate_masks(pred, true)
        assert abs(result["accuracy"] - 1.0) < 1e-9

    def test_perfect_prediction_mean_iou_one(self):
        pred, true = _perfect_pred(size=500)
        result = evaluate_masks(pred, true)
        assert result["mean_iou"] is not None
        assert abs(result["mean_iou"] - 1.0) < 1e-6

    def test_class_names_default(self):
        pred, true = _perfect_pred()
        result = evaluate_masks(pred, true)
        assert result["class_names"] == DEFAULT_CLASS_NAMES

    def test_class_names_custom(self):
        pred, true = _perfect_pred()
        names = ["BG", "C1", "C2", "C3", "C4"]
        result = evaluate_masks(pred, true, class_names=names)
        assert result["class_names"] == names

    def test_confusion_matrix_is_list_of_lists(self):
        pred, true = _perfect_pred()
        result = evaluate_masks(pred, true)
        cm = result["confusion_matrix"]
        assert isinstance(cm, list)
        assert isinstance(cm[0], list)
        assert len(cm) == NUM_CLASSES
        assert len(cm[0]) == NUM_CLASSES

    def test_per_class_keys(self):
        pred, true = _perfect_pred()
        result = evaluate_masks(pred, true)
        for name in DEFAULT_CLASS_NAMES:
            assert name in result["per_class"], f"'{name}' missing from per_class"
        for name, m in result["per_class"].items():
            assert "precision" in m
            assert "recall"    in m
            assert "f1"        in m
            assert "iou"       in m
            assert "support"   in m

    def test_json_serialisable(self):
        """Output must be serialisable to JSON without raw NaN/Inf values."""
        true = np.array([0, 1, 2, 3, 4, IGNORE_INDEX], dtype=np.uint8)
        pred = np.array([0, 1, 2, 3, 0, 1],            dtype=np.uint8)
        result = evaluate_masks(pred, true)
        # Should not raise
        serialised = json.dumps(result)
        # Should not contain raw 'NaN' (Python's float repr; JSON uses null)
        assert "NaN" not in serialised

    def test_ignore_pixels_excluded(self):
        """Ignore-index pixels in true mask must not affect metrics."""
        true_with_ignore = np.array([0, 0, IGNORE_INDEX], dtype=np.uint8)
        true_without     = np.array([0, 0],               dtype=np.uint8)
        pred             = np.array([0, 0, 1],             dtype=np.uint8)

        r_with    = evaluate_masks(pred,   true_with_ignore)
        r_without = evaluate_masks(pred[:2], true_without)

        assert r_with["accuracy"] == r_without["accuracy"]

    def test_2d_masks_accepted(self):
        pred = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        true = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = evaluate_masks(pred, true)
        assert abs(result["accuracy"] - 1.0) < 1e-9

    def test_mean_excludes_background_by_default(self):
        """With background-only mask, mean_iou should be NaN (no foreground support)."""
        true = np.zeros(50, dtype=np.uint8)
        pred = np.zeros(50, dtype=np.uint8)
        result = evaluate_masks(pred, true)
        # Background excluded, all other classes have zero support → NaN mean
        assert result["mean_iou"] is None


# ===========================================================================
# accumulate_confusion_matrix
# ===========================================================================

class TestAccumulateConfusionMatrix:

    def test_accumulation_matches_single_call(self):
        """Accumulating two batches should equal processing both at once."""
        rng  = np.random.default_rng(42)
        batch_a_pred = rng.integers(0, NUM_CLASSES, size=100, dtype=np.uint8)
        batch_a_true = rng.integers(0, NUM_CLASSES, size=100, dtype=np.uint8)
        batch_b_pred = rng.integers(0, NUM_CLASSES, size=150, dtype=np.uint8)
        batch_b_true = rng.integers(0, NUM_CLASSES, size=150, dtype=np.uint8)

        # Accumulate
        running = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        accumulate_confusion_matrix(running, batch_a_pred, batch_a_true)
        accumulate_confusion_matrix(running, batch_b_pred, batch_b_true)

        # Single combined call
        combined_pred = np.concatenate([batch_a_pred, batch_b_pred])
        combined_true = np.concatenate([batch_a_true, batch_b_true])
        expected = compute_confusion_matrix(combined_pred, combined_true)

        np.testing.assert_array_equal(running, expected)

    def test_accumulate_inplace(self):
        """accumulate_confusion_matrix modifies running_cm in place."""
        running = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        original_id = id(running)
        returned = accumulate_confusion_matrix(
            running,
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
        )
        assert id(returned) == original_id


# ===========================================================================
# evaluate_dataset
# ===========================================================================

class TestEvaluateDataset:

    def test_output_format_matches_evaluate_masks(self):
        """evaluate_dataset and evaluate_masks should produce same keys."""
        pred, true = _perfect_pred(size=200)
        cm = compute_confusion_matrix(pred, true)

        result_masks   = evaluate_masks(pred, true)
        result_dataset = evaluate_dataset(cm)

        assert set(result_masks.keys()) == set(result_dataset.keys())

    def test_perfect_cm_gives_full_score(self):
        pred, true = _perfect_pred(size=500)
        cm     = compute_confusion_matrix(pred, true)
        result = evaluate_dataset(cm)
        assert abs(result["accuracy"] - 1.0) < 1e-9
        assert abs(result["mean_iou"] - 1.0) < 1e-6

    def test_json_serialisable(self):
        pred, true = _perfect_pred(size=100)
        cm     = compute_confusion_matrix(pred, true)
        result = evaluate_dataset(cm)
        json.dumps(result)  # must not raise
