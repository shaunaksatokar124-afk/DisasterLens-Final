"""
DisasterLens AI -- Demo Predictor
==================================
Provides realistic-looking damage segmentation using image-processing
heuristics, requiring NO trained model checkpoint.

Algorithm
---------
1. Detect "built-up" areas via brightness + saturation thresholding.
2. Compute a *damage signal* from Laplacian texture and (if available)
   pre/post frame difference.
3. Threshold the damage signal into the 4 damage levels (1–4) within
   detected building areas.
4. If fewer than 3 classes are detected, inject synthetic spatial damage
   zones seeded deterministically from the image content.
5. Apply median blur for spatial realism.

The predictor is reproducible: identical input → identical output.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DemoPredictor:
    """
    Image-processing based damage predictor for demo / no-checkpoint mode.

    Attributes
    ----------
    is_demo : bool
        Always True – signals that this is a demonstration predictor.
    ready : bool
        Always True – no external resource is needed.
    status_message : str
        Human-readable status string.
    """

    is_demo: bool = True
    ready: bool   = True
    status_message: str = "Demo Mode — image-processing predictor (no checkpoint required)"

    # Damage-signal thresholds (0-255)
    THRESH_MINOR     = 35
    THRESH_MAJOR     = 85
    THRESH_DESTROYED = 145

    # Minimum fraction of image that must be flagged as "built area"
    # before we fall back to the full-image treatment.
    MIN_BUILT_FRACTION = 0.04

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        post_image: np.ndarray,
        pre_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a (H, W) uint8 damage mask (values 0–4).

        Parameters
        ----------
        post_image : np.ndarray
            Post-disaster RGB image (H, W, 3), uint8.
        pre_image : np.ndarray or None
            Pre-disaster RGB image.  When provided the per-pixel
            difference is used as an additional damage signal.
        """
        h, w = post_image.shape[:2]

        # The pre-disaster frame must match the post frame for change detection
        if pre_image is not None and pre_image.shape[:2] != (h, w):
            pre_image = cv2.resize(pre_image, (w, h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(post_image, cv2.COLOR_RGB2GRAY)
        hsv  = cv2.cvtColor(post_image, cv2.COLOR_RGB2HSV)

        damage_signal = self._compute_damage_signal(gray, pre_image)
        built_mask    = self._detect_built_areas(hsv, h, w)

        mask = self._classify_damage(built_mask, damage_signal)
        mask = cv2.medianBlur(mask, 9)

        # Guarantee at least 3 damage classes are visible
        unique_fg = np.unique(mask[mask > 0])
        if len(unique_fg) < 3:
            logger.debug("DemoPredictor: few classes detected – injecting synthetic zones")
            synthetic = self._synthetic_damage_zones(h, w, gray)
            # Prefer synthetic only where mask is background
            mask = np.where((mask == 0) & (synthetic > 0), synthetic, mask)
            mask = cv2.medianBlur(mask, 7)

        return mask.astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_damage_signal(
        self,
        gray: np.ndarray,
        pre_image: Optional[np.ndarray],
    ) -> np.ndarray:
        """Produce a 0-255 damage-intensity map."""
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

        # Laplacian texture (high texture ↔ debris / destroyed surfaces)
        lap     = cv2.Laplacian(smoothed.astype(np.float64), cv2.CV_64F, ksize=3)
        texture = cv2.GaussianBlur(np.abs(lap).astype(np.float32), (25, 25), 0)
        texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if pre_image is not None:
            pre_gray = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY)
            pre_sm   = cv2.GaussianBlur(pre_gray, (5, 5), 0)
            diff     = cv2.absdiff(smoothed, pre_sm)
            diff     = cv2.GaussianBlur(diff, (19, 19), 0)
            diff     = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Weight change detection more heavily than texture
            return cv2.addWeighted(diff, 0.65, texture, 0.35, 0)

        return texture

    def _detect_built_areas(
        self,
        hsv: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Return a binary (H, W) uint8 mask of likely building footprints.

        Buildings are compact, edge-bounded blobs.  We close the Canny edge
        map, keep filled blobs whose size and rectangularity look like
        footprints, and drop long thin structures (roads) and huge regions
        (open terrain).
        """
        gray = hsv[:, :, 2]
        sat  = hsv[:, :, 1]
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 40, 110)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        )

        # Fill enclosed regions
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        built = np.zeros((h, w), np.uint8)
        img_area = float(h * w)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.0002 * img_area or area > 0.04 * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            fill = area / float(bw * bh)
            if aspect > 4.0 or fill < 0.45:
                continue
            cv2.drawContours(built, [c], -1, 255, thickness=-1)

        # Low-saturation / bright blobs (grey & white roofs) in textured areas
        if np.sum(built > 0) < self.MIN_BUILT_FRACTION * h * w:
            logger.debug("DemoPredictor: sparse built area - falling back to colour rule")
            raw = ((gray > 70) & (gray < 245) & (sat < 90)).astype(np.uint8) * 255
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            built = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        # Fallback: still nothing -> treat whole image as built-up
        if np.sum(built > 0) < 0.01 * h * w:
            return np.ones((h, w), dtype=np.uint8) * 255

        return built

    @staticmethod
    def _classify_damage(
        built: np.ndarray,
        damage_signal: np.ndarray,
    ) -> np.ndarray:
        """Map damage signal values to class indices within built areas."""
        mask = np.zeros_like(damage_signal, dtype=np.uint8)
        B = built > 0
        D = damage_signal

        mask[B & (D <= DemoPredictor.THRESH_MINOR)]                                 = 1  # No damage
        mask[B & (D > DemoPredictor.THRESH_MINOR)  & (D <= DemoPredictor.THRESH_MAJOR)]     = 2  # Minor
        mask[B & (D > DemoPredictor.THRESH_MAJOR)  & (D <= DemoPredictor.THRESH_DESTROYED)] = 3  # Major
        mask[B & (D > DemoPredictor.THRESH_DESTROYED)]                              = 4  # Destroyed

        return mask

    @staticmethod
    def _synthetic_damage_zones(
        h: int,
        w: int,
        gray: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a spatially-coherent synthetic damage field seeded from
        the image content so that results are reproducible.
        """
        seed = int(gray.mean() * 137 + gray.std() * 31) % (2 ** 31)
        rng  = np.random.default_rng(seed)

        # 3-6 damage epicentres
        n_ep = int(rng.integers(3, 7))
        field = np.zeros((h, w), dtype=np.float32)

        for _ in range(n_ep):
            cx     = int(rng.integers(w // 5, 4 * w // 5))
            cy     = int(rng.integers(h // 5, 4 * h // 5))
            radius = float(rng.uniform(min(h, w) * 0.08, min(h, w) * 0.35))
            amp    = float(rng.uniform(0.5, 1.0))

            yy, xx = np.ogrid[:h, :w]
            dist   = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            field += amp * np.exp(-dist ** 2 / (2 * (radius * 0.5) ** 2))

        field = cv2.normalize(field, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[field > 0.12] = 1   # No damage halo
        mask[field > 0.40] = 2   # Minor
        mask[field > 0.62] = 3   # Major
        mask[field > 0.80] = 4   # Destroyed

        return mask
