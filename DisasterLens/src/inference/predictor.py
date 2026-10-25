"""
DisasterLens AI -- Inference Predictor
Loads a trained U-Net checkpoint and runs pixel-level damage segmentation.

When no usable checkpoint exists the predictor can fall back to a
heuristic *demo* predictor (``src/inference/demo_predictor.py``).  Demo mode is
opt-in via ``inference.demo_mode: true`` in ``config.yaml`` so that unit tests
and production deployments can still require a real model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:  # torch is only needed for real-model inference; demo mode works without it
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only in torch-less installs
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

from src.paths import resolve_path

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_checkpoint(config: dict) -> Optional[Path]:
    """Locate a checkpoint file inside ``paths.checkpoint_dir`` (or ``None``)."""
    paths = config.get("paths", {})
    p = resolve_path(paths.get("checkpoint_dir"))
    if p is None and paths.get("artifacts_dir"):
        # train.py saves here when checkpoint_dir is left unset
        p = resolve_path(paths["artifacts_dir"]) / "checkpoints"
    if p is None or not p.is_dir():
        return None
    for name in ("best_model.pth", "best.pth", "checkpoint_best.pth", "checkpoint.pth"):
        if (p / name).exists():
            return p / name
    candidates = sorted(p.glob("*.pth")) + sorted(p.glob("*.pt"))
    return candidates[0] if candidates else None


class DamagePredictor:
    """Wraps the segmentation model (or demo predictor) for Streamlit inference.

    Usage::

        predictor = DamagePredictor(config)
        if predictor.ready:
            mask = predictor.predict(post_image, pre_image)

    Attributes:
        ready (bool): True when a checkpoint is loaded OR demo mode is active.
        is_demo (bool): True when running in image-processing demo mode.
        status_message (str): Human-readable description of the current state.
    """

    NUM_CLASSES: int = 5
    DEFAULT_IMAGE_SIZE: int = 512

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = self._select_device()
        self.model = None
        self.ready: bool = False
        self.is_demo: bool = False
        self.checkpoint_path: Optional[Path] = None
        self._demo_predictor = None
        self._load_error: Optional[str] = None
        self._try_load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def allow_demo(self) -> bool:
        """Whether the heuristic demo predictor may be used as a fallback."""
        return bool(self.config.get("inference", {}).get("demo_mode", False))

    def predict(
        self,
        post_image: np.ndarray,
        pre_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run damage segmentation on a post-disaster image.

        Args:
            post_image: (H, W, 3) uint8 RGB post-disaster image.
            pre_image:  (H, W, 3) uint8 RGB pre-disaster image (optional).
                        If its size differs from ``post_image`` it is resized
                        to match.

        Returns:
            (H, W) uint8 predicted class mask, values 0-4.

        Raises:
            RuntimeError: If the predictor is not ready.
        """
        if not self.ready:
            raise RuntimeError(
                "DamagePredictor is not ready. "
                "Configure 'checkpoint_dir' in config.yaml and restart."
            )

        orig_h, orig_w = post_image.shape[:2]
        if pre_image is not None and pre_image.shape[:2] != (orig_h, orig_w):
            pre_image = cv2.resize(
                pre_image, (orig_w, orig_h), interpolation=cv2.INTER_AREA
            )

        # -- Demo mode path ---------------------------------------------------
        if self.is_demo and self._demo_predictor is not None:
            return self._demo_predictor.predict(post_image, pre_image)

        # -- Real model path --------------------------------------------------
        image_size = self.config.get("model", {}).get("image_size", self.DEFAULT_IMAGE_SIZE)

        post_t = self._preprocess(post_image, image_size)
        pre_t  = self._preprocess(
            pre_image if pre_image is not None else post_image, image_size
        )
        x = torch.cat([pre_t, post_t], dim=1).to(self.device)  # (1, 6, H, W)

        with torch.no_grad():
            logits = self.model(x)                             # (1, 5, H, W)
            pred   = torch.argmax(logits, dim=1).squeeze(0)   # (H, W)

        mask = pred.cpu().numpy().astype(np.uint8)

        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(
                mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

        return mask

    @property
    def status_message(self) -> str:
        if self.is_demo:
            return (
                "Demo Mode — image-processing predictor active  |  "
                "Results are heuristic, not from a trained model  |  Device: CPU"
            )
        if self.ready:
            ckpt = self.checkpoint_path.name if self.checkpoint_path else "unknown"
            return f"Ready  |  Checkpoint: {ckpt}  |  Device: {self.device}"
        if self._load_error:
            return f"Not ready — checkpoint could not be loaded: {self._load_error}"
        ckpt_dir = self.config.get("paths", {}).get("checkpoint_dir")
        if not ckpt_dir:
            return "Not ready — 'checkpoint_dir' is not set in config.yaml"
        return f"Not ready — no checkpoint found in: {ckpt_dir}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_device(self):
        pref = self.config.get("inference", {}).get("device", "auto")
        if torch is None:
            return "cpu"
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(pref)

    def _find_checkpoint(self) -> Optional[Path]:
        return find_checkpoint(self.config)

    def _try_load_model(self) -> None:
        """Try loading a real checkpoint; fall back to demo mode if allowed."""
        self.checkpoint_path = self._find_checkpoint()

        if self.checkpoint_path is None:
            logger.info("No checkpoint found.")
            if self.allow_demo:
                self._activate_demo_mode()
            return

        try:
            if torch is None:
                raise ImportError("PyTorch is not installed")
            from src.models.unet_resnet import build_model

            model_cfg = self.config.get("model", {})
            model = build_model(
                encoder_name=model_cfg.get("encoder", "resnet34"),
                encoder_weights=None,
                in_channels=model_cfg.get("in_channels", 6),
                num_classes=model_cfg.get("num_classes", 5),
            )
            ckpt = self._load_checkpoint_file(self.checkpoint_path)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict)
            self.model = model.to(self.device).eval()
            self.ready = True
            logger.info("Model loaded: %s on %s", self.checkpoint_path, self.device)
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to load checkpoint: %s", self._load_error)
            self.model = None
            if self.allow_demo:
                self._activate_demo_mode()

    def _load_checkpoint_file(self, path: Path):
        """``torch.load`` that works across PyTorch versions (safe by default)."""
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch < 1.13 has no weights_only argument
            return torch.load(path, map_location=self.device)

    def _activate_demo_mode(self) -> None:
        """Activate the image-processing demo predictor."""
        try:
            from src.inference.demo_predictor import DemoPredictor
            self._demo_predictor = DemoPredictor()
            self.is_demo = True
            self.ready   = True
            logger.info("Demo predictor activated.")
        except Exception as exc:
            logger.error("Could not activate demo predictor: %s", exc)
            self.ready = False

    @staticmethod
    def _preprocess(image: np.ndarray, size: int):
        """Resize, normalise (ImageNet stats), and convert to NCHW tensor."""
        img = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)  # (1, 3, H, W)
        return t
