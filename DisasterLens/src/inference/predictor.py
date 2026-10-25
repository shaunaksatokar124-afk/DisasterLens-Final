"""
DisasterLens AI -- Inference Predictor
Loads a trained U-Net checkpoint and runs pixel-level damage segmentation.

Checkpoint loading order
------------------------
1. Try to find a checkpoint in the local ``checkpoint_dir`` / ``artifacts/checkpoints/``.
2. If nothing is found locally, attempt to download ``best_model.pth`` from
   Hugging Face (``Shaunak-006/disasterlens-checkpoint``) via
   ``src.inference.hf_loader.download_checkpoint``.
3. If the download fails, the predictor is marked **not ready** and a clear
   error message is surfaced.  No silent fallback to a heuristic predictor is
   performed unless ``inference.demo_mode: true`` is explicitly set in
   ``config.yaml``.

The model expects:
    * Six input channels  -- 3-ch pre-disaster RGB + 3-ch post-disaster RGB
    * Five output classes -- background + 4 damage levels (No Damage → Destroyed)
    * Checkpoint key       -- ``model_state_dict``

This module is intentionally conservative: if loading fails, ``ready`` is
``False`` and ``predict()`` raises ``RuntimeError`` rather than returning
silently wrong results.
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
    """Locate a checkpoint file inside ``paths.checkpoint_dir`` (or ``None``).

    Searches the configured ``checkpoint_dir``; if that is unset, also checks
    ``artifacts/checkpoints/`` (the default HF download location).
    """
    paths = config.get("paths", {})
    candidates: list[Optional[Path]] = []

    # 1. Explicitly configured directory
    p = resolve_path(paths.get("checkpoint_dir"))
    if p is not None:
        candidates.append(p)

    # 2. Default HF download cache (always checked as a fallback)
    artifacts = resolve_path(paths.get("artifacts_dir") or "artifacts")
    if artifacts:
        candidates.append(artifacts / "checkpoints")

    for p in candidates:
        if p is None or not p.is_dir():
            continue
        for name in ("best_model.pth", "best.pth", "checkpoint_best.pth", "checkpoint.pth"):
            if (p / name).exists():
                return p / name
        found = sorted(p.glob("*.pth")) + sorted(p.glob("*.pt"))
        if found:
            return found[0]

    return None


class DamagePredictor:
    """Wraps the segmentation model (or demo predictor) for Streamlit inference.

    Usage::

        predictor = DamagePredictor(config)
        if predictor.ready:
            mask = predictor.predict(post_image, pre_image)

    Attributes:
        ready (bool): True when a real checkpoint is loaded (or demo mode is active).
        is_demo (bool): True when running in image-processing demo mode.
        status_message (str): Human-readable description of the current state.

    Notes:
        This predictor is **experimental**. The model was trained on the xBD
        dataset and has not been validated for operational use. Always apply
        independent human verification to all predictions.
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
        self._hf_attempted: bool = False
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
                + (self._load_error or "Configure the model and restart.")
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
            return (
                f"Model Ready (experimental)  |  Checkpoint: {ckpt}  |  Device: {self.device}  |  "
                "Predictions require human verification"
            )
        if self._load_error:
            return f"Not Ready — {self._load_error}"
        return "Not Ready — checkpoint could not be located or downloaded"

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
        """Try loading a real checkpoint.

        Loading order:
          1. Local checkpoint (``checkpoint_dir`` or ``artifacts/checkpoints/``).
          2. Hugging Face download.
          3. If both fail → not ready (error surfaced; demo mode only if allowed).
        """
        # Step 1: local search
        self.checkpoint_path = self._find_checkpoint()

        # Step 2: HF download if nothing found locally
        if self.checkpoint_path is None:
            self.checkpoint_path = self._try_hf_download()

        if self.checkpoint_path is None:
            # No checkpoint and no successful download
            if not self._load_error:
                self._load_error = (
                    "No checkpoint found locally and Hugging Face download was not attempted "
                    "or was skipped. Set 'checkpoint_dir' in config.yaml or ensure network access."
                )
            logger.warning("No checkpoint available: %s", self._load_error)
            if self.allow_demo:
                self._activate_demo_mode()
            return

        # Step 3: load the checkpoint file into the model
        self._load_from_checkpoint(self.checkpoint_path)

    def _try_hf_download(self) -> Optional[Path]:
        """Attempt to download the checkpoint from Hugging Face.

        Returns the local path on success, ``None`` on failure.
        Sets ``self._load_error`` on failure.
        """
        self._hf_attempted = True
        try:
            from src.inference.hf_loader import download_checkpoint
            path = download_checkpoint()
            logger.info("HF download succeeded: %s", path)
            return path
        except RuntimeError as exc:
            self._load_error = str(exc)
            logger.error("HF download failed: %s", self._load_error)
            return None
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Unexpected error during HF download: %s", self._load_error)
            return None

    def _load_from_checkpoint(self, path: Path) -> None:
        """Load model weights from *path*.  Sets ``self.ready`` on success."""
        try:
            if torch is None:
                raise ImportError("PyTorch is not installed")
            from src.models.unet_resnet import DisasterLensModel

            model_cfg = self.config.get("model", {})
            model = DisasterLensModel(
                encoder_name=model_cfg.get("encoder", "resnet34"),
                encoder_weights=None,          # weights come from the checkpoint
                in_channels=model_cfg.get("in_channels", 6),
                num_classes=model_cfg.get("num_classes", 5),
            )

            ckpt = self._load_checkpoint_file(path)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict", ckpt)
            else:
                state_dict = ckpt

            model.load_state_dict(state_dict)
            self.model = model.to(self.device).eval()
            self.ready = True
            self._load_error = None
            logger.info("Model loaded successfully from %s on %s", path, self.device)

        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to load checkpoint %s: %s", path, self._load_error)
            self.model = None
            self.ready = False
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
