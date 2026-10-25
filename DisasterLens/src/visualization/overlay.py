"""
DisasterLens AI -- Visualization Utilities
Colorizes segmentation masks, blends overlays onto images,
creates side-by-side comparisons, and computes class statistics.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Class colour definitions
# ---------------------------------------------------------------------------

CLASS_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (45, 45, 45),     # Background       -- dark grey
    1: (0, 200, 81),     # No Damage        -- green
    2: (255, 214, 0),    # Minor Damage     -- yellow
    3: (255, 109, 0),    # Major Damage     -- orange
    4: (255, 23, 68),    # Destroyed        -- red
}

CLASS_NAMES: Dict[int, str] = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}

IGNORE_INDEX: int = 255


def _make_lut() -> np.ndarray:
    """Build a 256x3 uint8 colour lookup table for fast mask colorisation."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for idx, rgb in CLASS_COLORS_RGB.items():
        lut[idx] = rgb
    lut[IGNORE_INDEX] = CLASS_COLORS_RGB[0]
    return lut


_LUT = _make_lut()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask to an RGB colour image.

    Args:
        mask: (H, W) uint8 array, values 0-4 plus 255 for ignore.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")
    return _LUT[mask]


def blend_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a colour mask overlay onto the original image.

    Only foreground pixels (class > 0 and != 255) are blended; background
    pixels remain unchanged so the original imagery stays visible.

    Args:
        original: (H, W, 3) uint8 RGB image.
        mask:     (H, W) uint8 class-index mask.
        alpha:    Overlay opacity in [0, 1].  0 = original only.

    Returns:
        (H, W, 3) uint8 blended RGB image.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Resize the *class-index* mask (nearest-neighbour keeps labels intact)
    # so that the foreground selector and the colour image share the
    # original image's shape.
    if mask.shape[:2] != original.shape[:2]:
        mask = cv2.resize(
            mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    colored = colorize_mask(mask)

    fg = (mask > 0) & (mask != IGNORE_INDEX)
    blended = original.copy()
    blended[fg] = (
        (1.0 - alpha) * original[fg] + alpha * colored[fg]
    ).astype(np.uint8)
    return blended


def create_comparison(
    left: np.ndarray,
    right: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Damage Prediction",
    gap: int = 4,
    header_h: int = 28,
) -> np.ndarray:
    """Create a side-by-side comparison image with labelled panels.

    Args:
        left:        (H, W, 3) uint8 left panel (original image).
        right:       (H, W, 3) uint8 right panel (overlay).
        label_left:  Caption for the left panel.
        label_right: Caption for the right panel.
        gap:         Pixel gap between panels.
        header_h:    Height of the label header bar in pixels.

    Returns:
        (H + header_h, W*2 + gap, 3) uint8 RGB image.
    """
    h, w = left.shape[:2]
    total_w = w * 2 + gap
    total_h = h + header_h

    canvas = np.full((total_h, total_w, 3), 10, dtype=np.uint8)
    canvas[header_h:, :w] = left
    canvas[header_h:, w + gap:] = right

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    bg_col = (15, 22, 41)
    draw.rectangle([0, 0, w - 1, header_h - 1], fill=bg_col)
    draw.rectangle([w + gap, 0, total_w - 1, header_h - 1], fill=bg_col)

    draw.text((w // 2, header_h // 2), label_left,
              fill=(184, 196, 200), anchor="mm")
    draw.text((w + gap + w // 2, header_h // 2), label_right,
              fill=(0, 212, 255), anchor="mm")

    return np.array(pil)


def compute_class_statistics(mask: np.ndarray) -> Dict[str, dict]:
    """Compute per-class pixel counts and percentages.

    Args:
        mask: (H, W) uint8 class-index mask. Pixels == 255 are excluded.

    Returns:
        Ordered dict {class_name: {class_idx, count, percentage, color_hex}}.
    """
    valid = mask[mask != IGNORE_INDEX]
    total = int(valid.size) if valid.size > 0 else 1
    stats: Dict[str, dict] = {}

    for idx, name in CLASS_NAMES.items():
        count = int(np.sum(mask == idx))
        pct = count / total * 100.0
        r, g, b = CLASS_COLORS_RGB[idx]
        stats[name] = {
            "class_idx": idx,
            "count": count,
            "percentage": round(pct, 2),
            "color_hex": f"#{r:02x}{g:02x}{b:02x}",
        }
    return stats


# ---------------------------------------------------------------------------
# Image utility helpers
# ---------------------------------------------------------------------------

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to (H, W, 3) uint8 numpy array (RGB)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert (H, W, 3) uint8 numpy array to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def resize_keep_aspect(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize so the longest edge equals max_size, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
