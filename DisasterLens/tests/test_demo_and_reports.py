"""
Tests for demo-mode inference, PDF reports, history storage, and the
transform/collate path used for training.  All data is synthetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.demo_predictor import DemoPredictor
from src.inference.predictor import DamagePredictor
from src.visualization.overlay import blend_overlay, compute_class_statistics


def _scene(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (90, 110, 80), np.uint8)
    img[20:50, 20:60] = (200, 200, 205)      # a "building"
    img[60:85, 70:115] = (150, 90, 80)       # another
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def _config(demo: bool, ckpt=None):
    return {
        "paths": {"checkpoint_dir": ckpt},
        "model": {"encoder": "resnet34", "in_channels": 6, "num_classes": 5, "image_size": 64},
        "inference": {"device": "cpu", "demo_mode": demo},
    }


class TestDemoPredictor:
    def test_mask_shape_dtype_and_range(self):
        post = _scene()
        m = DemoPredictor().predict(post, None)
        assert m.shape == post.shape[:2] and m.dtype == np.uint8
        assert m.max() <= 4

    def test_deterministic(self):
        post = _scene()
        assert np.array_equal(DemoPredictor().predict(post), DemoPredictor().predict(post))

    def test_mismatched_pre_image_is_resized(self):
        post, pre = _scene(), _scene(h=40, w=40, seed=1)
        m = DemoPredictor().predict(post, pre)   # must not raise
        assert m.shape == post.shape[:2]

    def test_flat_image_does_not_crash(self):
        flat = np.zeros((64, 64, 3), np.uint8)
        assert DemoPredictor().predict(flat).shape == (64, 64)


class TestPredictorDemoMode:
    def test_ready_in_demo_mode_without_checkpoint(self):
        p = DamagePredictor(_config(demo=True))
        assert p.ready and p.is_demo
        assert "demo" in p.status_message.lower()

    def test_predict_works_with_different_size_pre(self):
        p = DamagePredictor(_config(demo=True))
        m = p.predict(_scene(), _scene(h=50, w=70))
        assert m.shape == (96, 128)

    def test_not_ready_when_demo_disabled(self):
        assert DamagePredictor(_config(demo=False)).ready is False

    def test_corrupt_checkpoint_falls_back_only_if_demo_allowed(self, tmp_path):
        (tmp_path / "best_model.pth").write_bytes(b"not a checkpoint")
        assert DamagePredictor(_config(False, str(tmp_path))).ready is False
        p = DamagePredictor(_config(True, str(tmp_path)))
        assert p.ready and p.is_demo

    def test_real_checkpoint_roundtrip(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("segmentation_models_pytorch")
        from src.models.unet_resnet import build_model
        model = build_model(encoder_weights=None, in_channels=6, num_classes=5)
        torch.save({"model_state_dict": model.state_dict()}, tmp_path / "best_model.pth")
        p = DamagePredictor(_config(False, str(tmp_path)))
        assert p.ready and not p.is_demo
        m = p.predict(_scene(), _scene(h=30, w=30))
        assert m.shape == (96, 128) and m.max() <= 4


class TestOverlayRegression:
    def test_blend_with_smaller_mask_keeps_foreground(self):
        img = np.zeros((128, 128, 3), np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[:32] = 4
        out = blend_overlay(img, mask, alpha=1.0)
        assert out[:64].sum() > 0 and out[64:].sum() == 0


@pytest.fixture()
def entry_and_config(tmp_path):
    from PIL import Image
    img = _scene()
    Image.fromarray(img).save(tmp_path / "post.png")
    Image.fromarray(blend_overlay(img, DemoPredictor().predict(img), 0.5)).save(tmp_path / "overlay.png")
    entry = {
        "id": "abc12345", "timestamp": "2026-01-01T10:00:00",
        "filename": "Tom & Jerry <test>.png",   # characters that break Paragraph markup
        "post_img": str(tmp_path / "post.png"), "overlay_img": str(tmp_path / "overlay.png"),
        "stats": compute_class_statistics(DemoPredictor().predict(img)),
        "model_device": "cpu", "mode": "demo",
    }
    return entry, {"model": {"encoder": "resnet34"}, "dataset": {"name": "xBD"}}


class TestPdfReport:
    def test_generates_valid_pdf(self, entry_and_config):
        pytest.importorskip("reportlab")
        from src.reports.pdf_generator import generate_report_pdf
        pdf = generate_report_pdf(*entry_and_config)
        assert pdf.startswith(b"%PDF") and len(pdf) > 2000

    def test_missing_images_and_stats_do_not_crash(self):
        pytest.importorskip("reportlab")
        from src.reports.pdf_generator import generate_report_pdf
        pdf = generate_report_pdf({"id": "x", "post_img": "nope.png"}, {})
        assert pdf.startswith(b"%PDF")


class TestHistoryStorage:
    def test_roundtrip_and_corrupt_file(self, tmp_path):
        pytest.importorskip("streamlit")
        from src.ui.components import load_history, save_history_entry
        cfg = {"paths": {"analysis_history_file": str(tmp_path / "h.json")}}
        assert load_history(cfg) == []
        save_history_entry(cfg, {"id": "1"})
        save_history_entry(cfg, {"id": "2"})
        assert [e["id"] for e in load_history(cfg)] == ["1", "2"]
        (tmp_path / "h.json").write_text("{broken")
        assert load_history(cfg) == []
        save_history_entry(cfg, {"id": "3"})       # recovers
        assert json.loads((tmp_path / "h.json").read_text())[0]["id"] == "3"


class TestTrainingInputPipeline:
    def _dataset(self, tmp_path, transform):
        from PIL import Image
        from src.data.dataset import XBDDataset
        img_dir, lbl_dir = tmp_path / "images", tmp_path / "labels"
        img_dir.mkdir(); lbl_dir.mkdir()
        stem = "ev_000001"
        for kind in ("pre", "post"):
            Image.fromarray(_scene(64, 64)).save(img_dir / f"{stem}_{kind}_disaster.png")
        json.dump({"features": {"xy": [{"properties": {"subtype": "destroyed"},
                   "wkt": "POLYGON ((100 100, 500 100, 500 500, 100 500, 100 100))"}]}},
                  open(lbl_dir / f"{stem}_post_disaster.json", "w"))
        pair = {"pre_image": img_dir / f"{stem}_pre_disaster.png",
                "post_image": img_dir / f"{stem}_post_disaster.png",
                "post_label": lbl_dir / f"{stem}_post_disaster.json",
                "disaster": "ev", "tile_id": "000001"}
        return XBDDataset([pair, pair], image_size=32, transform=transform)

    @pytest.mark.parametrize("train", [False, True])
    def test_transformed_batches_collate(self, tmp_path, train):
        torch = pytest.importorskip("torch")
        pytest.importorskip("albumentations")
        from src.data.dataset import collate_batch
        from src.data.transforms import get_train_transform, get_val_transform
        tfm = get_train_transform(32) if train else get_val_transform(32)
        ds = self._dataset(tmp_path, tfm)
        batch = collate_batch([ds[0], ds[1]])
        assert batch["image"].shape == (2, 6, 32, 32) and batch["image"].dtype == torch.float32
        assert batch["mask"].shape == (2, 32, 32) and batch["mask"].dtype == torch.int64

    def test_raw_numpy_samples_collate(self, tmp_path):
        torch = pytest.importorskip("torch")
        from src.data.dataset import collate_batch
        ds = self._dataset(tmp_path, None)
        batch = collate_batch([ds[0], ds[1]])
        assert batch["image"].shape == (2, 6, 32, 32)
        assert batch["image"].abs().max() < 10          # normalised, not 0-255
        assert (batch["mask"] == 4).any()
