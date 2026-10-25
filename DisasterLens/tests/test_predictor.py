"""
DisasterLens AI -- Predictor Unit Tests
=========================================
Tests for src/inference/predictor.py.

All tests run WITHOUT a real model checkpoint.  The predictor's
"not ready" state (no checkpoint) is the primary path exercised here.
Behaviour when a checkpoint IS loaded is covered by the end-to-end
integration test (run separately when a checkpoint is available).

Run with:
    pytest tests/test_predictor.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import DamagePredictor


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

def _base_config(checkpoint_dir: str | None = None) -> dict:
    return {
        "paths": {
            "checkpoint_dir": checkpoint_dir,
            "dataset_root":   None,
        },
        "model": {
            "encoder":         "resnet34",
            "encoder_weights": "imagenet",
            "in_channels":     6,
            "num_classes":     5,
            "image_size":      64,   # tiny size for fast tests
        },
        "inference": {
            "device": "cpu",
        },
    }


# ---------------------------------------------------------------------------
# No-checkpoint state (expected behaviour for users without a trained model)
# ---------------------------------------------------------------------------

class TestPredictorNotReady:

    def test_ready_is_false_when_no_checkpoint_dir(self):
        p = DamagePredictor(_base_config(checkpoint_dir=None))
        assert p.ready is False

    def test_ready_is_false_when_dir_missing(self, tmp_path):
        missing = str(tmp_path / "no_such_dir")
        p = DamagePredictor(_base_config(checkpoint_dir=missing))
        assert p.ready is False

    def test_ready_is_false_when_dir_empty(self, tmp_path):
        empty = tmp_path / "empty_ckpts"
        empty.mkdir()
        p = DamagePredictor(_base_config(checkpoint_dir=str(empty)))
        assert p.ready is False

    def test_predict_raises_runtime_error_when_not_ready(self):
        p = DamagePredictor(_base_config(checkpoint_dir=None))
        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not ready"):
            p.predict(fake_img)

    def test_predict_raises_with_pre_image_when_not_ready(self):
        p = DamagePredictor(_base_config(checkpoint_dir=None))
        fake = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            p.predict(fake, pre_image=fake)

    def test_status_message_mentions_checkpoint_dir(self):
        p = DamagePredictor(_base_config(checkpoint_dir=None))
        assert "checkpoint" in p.status_message.lower() or "not ready" in p.status_message.lower()

    def test_status_message_is_string(self):
        p = DamagePredictor(_base_config(checkpoint_dir=None))
        assert isinstance(p.status_message, str)
        assert len(p.status_message) > 0

    def test_device_is_cpu_when_configured(self):
        import torch
        p = DamagePredictor(_base_config(checkpoint_dir=None))
        assert p.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Preprocessing helper (_preprocess)
# ---------------------------------------------------------------------------

class TestPreprocess:
    """Test the internal _preprocess static method directly."""

    def test_output_shape(self):
        import torch
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        t   = DamagePredictor._preprocess(img, size=64)
        assert t.shape == (1, 3, 64, 64), f"Unexpected shape: {t.shape}"

    def test_output_dtype_float32(self):
        import torch
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        t   = DamagePredictor._preprocess(img, size=64)
        assert t.dtype == torch.float32

    def test_output_range_is_normalised(self):
        """After ImageNet normalisation output should be roughly in [-3, 3]."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        t   = DamagePredictor._preprocess(img, size=64)
        assert t.min().item() > -5.0
        assert t.max().item() <  5.0

    def test_all_black_image(self):
        """All-zero input should not crash."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        t   = DamagePredictor._preprocess(img, size=64)
        assert t.shape == (1, 3, 64, 64)

    def test_all_white_image(self):
        """All-255 input should not crash."""
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        t   = DamagePredictor._preprocess(img, size=64)
        assert t.shape == (1, 3, 64, 64)

    def test_non_square_input_resized(self):
        """Non-square images should be resized to (size, size)."""
        import torch
        img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        t   = DamagePredictor._preprocess(img, size=64)
        assert t.shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing:

    def test_auto_device_selects_cpu_on_colab_cpu_only(self):
        """When CUDA is unavailable, 'auto' should resolve to 'cpu'."""
        import torch
        config = _base_config()
        config["inference"]["device"] = "auto"
        p = DamagePredictor(config)
        if not torch.cuda.is_available():
            assert p.device.type == "cpu"

    def test_explicit_cpu_device(self):
        import torch
        config = _base_config()
        config["inference"]["device"] = "cpu"
        p = DamagePredictor(config)
        assert p.device == torch.device("cpu")

    def test_image_size_read_from_config(self):
        """The predictor should use image_size from model config during inference."""
        config = _base_config()
        config["model"]["image_size"] = 32
        p = DamagePredictor(config)
        # Just ensure no crash during init — inference is blocked (not ready)
        assert not p.ready
