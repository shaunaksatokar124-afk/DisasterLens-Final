"""
DisasterLens AI -- Visualization Unit Tests
=============================================
Tests for src/visualization/overlay.py.

No dataset, model, or image files required — all inputs are synthetic numpy arrays.

Run with:
    pytest tests/test_visualization.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.overlay import (
    CLASS_COLORS_RGB,
    CLASS_NAMES,
    IGNORE_INDEX,
    blend_overlay,
    colorize_mask,
    compute_class_statistics,
    create_comparison,
    numpy_to_pil,
    pil_to_numpy,
    resize_keep_aspect,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a synthetic mask with each class in a horizontal band."""
    mask = np.zeros((h, w), dtype=np.uint8)
    band = h // 5
    for cls in range(5):
        mask[cls * band : (cls + 1) * band, :] = cls
    return mask


def _make_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a random uint8 RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ===========================================================================
# colorize_mask
# ===========================================================================

class TestColorizeMask:

    def test_output_shape(self):
        mask = _make_mask()
        rgb  = colorize_mask(mask)
        assert rgb.shape == (64, 64, 3)

    def test_output_dtype(self):
        mask = _make_mask()
        rgb  = colorize_mask(mask)
        assert rgb.dtype == np.uint8

    def test_known_class_colour(self):
        """Class 1 (No Damage) pixels should match CLASS_COLORS_RGB[1]."""
        mask = np.full((16, 16), 1, dtype=np.uint8)
        rgb  = colorize_mask(mask)
        expected = CLASS_COLORS_RGB[1]
        assert tuple(rgb[0, 0]) == expected

    def test_background_colour(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        rgb  = colorize_mask(mask)
        assert tuple(rgb[0, 0]) == CLASS_COLORS_RGB[0]

    def test_destroyed_colour(self):
        mask = np.full((16, 16), 4, dtype=np.uint8)
        rgb  = colorize_mask(mask)
        assert tuple(rgb[0, 0]) == CLASS_COLORS_RGB[4]

    def test_ignore_index_does_not_crash(self):
        mask = np.full((8, 8), IGNORE_INDEX, dtype=np.uint8)
        rgb  = colorize_mask(mask)
        assert rgb.shape == (8, 8, 3)

    def test_1d_mask_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            colorize_mask(np.zeros(64, dtype=np.uint8))

    def test_all_classes_produce_distinct_colours(self):
        """Each foreground class should produce a distinct colour."""
        colours = set()
        for cls in range(1, 5):
            mask = np.full((8, 8), cls, dtype=np.uint8)
            rgb  = colorize_mask(mask)
            colours.add(tuple(rgb[0, 0]))
        assert len(colours) == 4, "Expected 4 distinct damage class colours"


# ===========================================================================
# blend_overlay
# ===========================================================================

class TestBlendOverlay:

    def test_output_shape_matches_original(self):
        img    = _make_image()
        mask   = _make_mask()
        result = blend_overlay(img, mask, alpha=0.5)
        assert result.shape == img.shape

    def test_output_dtype(self):
        result = blend_overlay(_make_image(), _make_mask(), alpha=0.5)
        assert result.dtype == np.uint8

    def test_alpha_zero_returns_original(self):
        img    = _make_image()
        mask   = _make_mask()
        result = blend_overlay(img, mask, alpha=0.0)
        np.testing.assert_array_equal(result, img)

    def test_background_pixels_unchanged(self):
        """Pixels where mask==0 (background) should not be blended."""
        img  = _make_image(32, 32)
        mask = np.zeros((32, 32), dtype=np.uint8)  # all background
        result = blend_overlay(img, mask, alpha=0.9)
        np.testing.assert_array_equal(result, img)

    def test_alpha_clamp_below_zero(self):
        """alpha < 0 should be clamped to 0 (no blend)."""
        img    = _make_image()
        mask   = _make_mask()
        result = blend_overlay(img, mask, alpha=-1.0)
        np.testing.assert_array_equal(result, img)

    def test_alpha_clamp_above_one(self):
        """alpha > 1 should be clamped to 1."""
        img    = _make_image()
        mask   = _make_mask()
        result = blend_overlay(img, mask, alpha=2.0)
        assert result.shape == img.shape  # should not crash

    def test_different_size_mask_resized(self):
        """If mask and image differ in size, mask should be resized."""
        img  = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.full((64, 64), 2, dtype=np.uint8)
        result = blend_overlay(img, mask, alpha=0.5)
        assert result.shape == (128, 128, 3)


# ===========================================================================
# create_comparison
# ===========================================================================

class TestCreateComparison:

    def test_output_shape(self):
        left  = _make_image(64, 64)
        right = _make_image(64, 64)
        comp  = create_comparison(left, right, gap=4, header_h=28)
        # height = h + header_h = 64 + 28 = 92
        # width  = w*2 + gap   = 64*2 + 4 = 132
        assert comp.shape == (92, 132, 3)

    def test_output_dtype(self):
        comp = create_comparison(_make_image(), _make_image())
        assert comp.dtype == np.uint8

    def test_output_is_numpy(self):
        comp = create_comparison(_make_image(), _make_image())
        assert isinstance(comp, np.ndarray)


# ===========================================================================
# compute_class_statistics
# ===========================================================================

class TestComputeClassStatistics:

    def test_returns_all_class_names(self):
        mask  = _make_mask()
        stats = compute_class_statistics(mask)
        for name in CLASS_NAMES.values():
            assert name in stats, f"'{name}' missing from stats"

    def test_percentages_sum_to_100(self):
        mask  = _make_mask(100, 100)
        stats = compute_class_statistics(mask)
        total = sum(s["percentage"] for s in stats.values())
        assert abs(total - 100.0) < 0.1, f"Percentages sum to {total}, expected 100"

    def test_counts_are_non_negative(self):
        stats = compute_class_statistics(_make_mask())
        for name, s in stats.items():
            assert s["count"] >= 0, f"Negative count for {name}"

    def test_ignore_index_excluded(self):
        mask = np.full((10, 10), IGNORE_INDEX, dtype=np.uint8)
        stats = compute_class_statistics(mask)
        for name, s in stats.items():
            assert s["count"] == 0, f"Expected 0 for {name}, got {s['count']}"

    def test_single_class_mask(self):
        mask  = np.full((8, 8), 3, dtype=np.uint8)  # all major-damage
        stats = compute_class_statistics(mask)
        assert stats["Major Damage"]["count"] == 64
        assert abs(stats["Major Damage"]["percentage"] - 100.0) < 0.1

    def test_color_hex_format(self):
        stats = compute_class_statistics(_make_mask())
        for name, s in stats.items():
            hex_color = s["color_hex"]
            assert hex_color.startswith("#"), f"Invalid hex for {name}: {hex_color}"
            assert len(hex_color) == 7, f"Invalid hex length for {name}: {hex_color}"

    def test_class_idx_matches(self):
        stats = compute_class_statistics(_make_mask())
        for cls_idx, cls_name in CLASS_NAMES.items():
            assert stats[cls_name]["class_idx"] == cls_idx


# ===========================================================================
# pil_to_numpy / numpy_to_pil
# ===========================================================================

class TestConversions:

    def test_pil_to_numpy_shape(self):
        img = Image.new("RGB", (32, 48))
        arr = pil_to_numpy(img)
        assert arr.shape == (48, 32, 3)

    def test_pil_to_numpy_dtype(self):
        img = Image.new("RGB", (16, 16))
        arr = pil_to_numpy(img)
        assert arr.dtype == np.uint8

    def test_pil_to_numpy_converts_to_rgb(self):
        """RGBA input should be converted to RGB."""
        img = Image.new("RGBA", (16, 16))
        arr = pil_to_numpy(img)
        assert arr.shape[2] == 3

    def test_numpy_to_pil_and_back(self):
        """Round-trip: numpy → PIL → numpy should preserve values."""
        arr  = _make_image(32, 32)
        pil  = numpy_to_pil(arr)
        arr2 = pil_to_numpy(pil)
        np.testing.assert_array_equal(arr, arr2)

    def test_numpy_to_pil_returns_image(self):
        assert isinstance(numpy_to_pil(_make_image()), Image.Image)


# ===========================================================================
# resize_keep_aspect
# ===========================================================================

class TestResizeKeepAspect:

    def test_no_resize_when_already_small(self):
        img    = _make_image(64, 64)
        result = resize_keep_aspect(img, max_size=1024)
        np.testing.assert_array_equal(result, img)

    def test_square_image_resized_correctly(self):
        img    = _make_image(2048, 2048)
        result = resize_keep_aspect(img, max_size=512)
        assert result.shape == (512, 512, 3)

    def test_landscape_image_preserves_aspect(self):
        img    = _make_image(1000, 2000)  # w=2000, h=1000
        result = resize_keep_aspect(img, max_size=1000)
        # longest edge = 2000 → scale = 0.5 → (500, 1000)
        assert result.shape[1] == 1000   # width
        assert result.shape[0] == 500    # height

    def test_portrait_image_preserves_aspect(self):
        img    = _make_image(2000, 500)   # h=2000, w=500
        result = resize_keep_aspect(img, max_size=1000)
        assert result.shape[0] == 1000   # height
        assert result.shape[1] == 250    # width

    def test_output_dtype(self):
        img    = _make_image(128, 128)
        result = resize_keep_aspect(img, max_size=64)
        assert result.dtype == np.uint8
