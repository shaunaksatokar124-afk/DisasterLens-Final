"""
DisasterLens AI -- Dataset Unit Tests
======================================
Tests for src/data/dataset.py and src/data/transforms.py.

All tests use **synthetic data only** -- no real xBD dataset required.
Synthetic annotations and images are created in-memory or in a temporary
directory using pytest's ``tmp_path`` fixture.

Test categories:
  - Polygon-to-mask conversion (WKT parsing, rasterisation)
  - Mask class values and ignore-index handling
  - Pair discovery with a synthetic dataset directory
  - Disaster-event-level split correctness
  - XBDDataset __getitem__ output shapes and types
  - DatasetNotFoundError when root is missing
  - Malformed annotation graceful handling

Run with:
    pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    IGNORE_INDEX,
    SUBTYPE_TO_CLASS,
    DatasetNotFoundError,
    XBDDataset,
    _discover_pairs,
    polygons_to_mask,
    split_pairs_by_event,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_wkt_rect(x0: int, y0: int, x1: int, y1: int) -> str:
    """Return a WKT POLYGON string for an axis-aligned rectangle."""
    return (
        f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"
    )


def _make_label_json(features: list) -> dict:
    """Return a minimal xBD-format label dict."""
    return {
        "metadata": {"img_name": "test", "disaster": "hurricane-test"},
        "features": {"xy": features},
    }


def _write_png(path: Path, size: int = 64) -> None:
    """Write a tiny solid-colour PNG for tests that need real image files."""
    from PIL import Image
    img = Image.new("RGB", (size, size), color=(100, 150, 200))
    img.save(path)


@pytest.fixture()
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal xBD directory structure with 6 image pairs across 3 events."""
    events = [
        ("hurricane-harvey", ["000001", "000002"]),
        ("california-fire",  ["000003", "000004"]),
        ("nepal-flooding",   ["000005", "000006"]),
    ]

    root = tmp_path / "xbd"
    img_dir   = root / "train" / "images"
    label_dir = root / "train" / "labels"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for disaster, tile_ids in events:
        for tile_id in tile_ids:
            stem_post = f"{disaster}_{tile_id}_post_disaster"
            stem_pre  = f"{disaster}_{tile_id}_pre_disaster"

            _write_png(img_dir / f"{stem_post}.png", size=64)
            _write_png(img_dir / f"{stem_pre}.png",  size=64)

            # Label JSON with one building polygon per tile
            features = [{
                "properties": {"subtype": "major-damage"},
                "wkt": _make_wkt_rect(10, 10, 40, 40),
            }]
            label_data = _make_label_json(features)
            with open(label_dir / f"{stem_post}.json", "w") as f:
                json.dump(label_data, f)

    return root


# ===========================================================================
# polygons_to_mask
# ===========================================================================

class TestPolygonsToMask:
    def test_background_only_when_no_features(self):
        mask = polygons_to_mask([], image_size=64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert np.all(mask == 0), "All pixels should be background"

    def test_single_building_correct_class(self):
        features = [{
            "properties": {"subtype": "major-damage"},
            "wkt": _make_wkt_rect(10, 10, 50, 50),
        }]
        mask = polygons_to_mask(features, image_size=64)
        # At least some pixels inside the rectangle should be class 3
        assert np.any(mask == 3), "Expected class 3 (major-damage) pixels"
        # Pixels far outside should be background
        assert mask[0, 0] == 0

    def test_all_damage_classes_present(self):
        subtypes = ["no-damage", "minor-damage", "major-damage", "destroyed"]
        expected = [1, 2, 3, 4]
        features = []
        for i, sub in enumerate(subtypes):
            x0, y0 = i * 60, 0
            x1, y1 = x0 + 50, 50
            features.append({
                "properties": {"subtype": sub},
                "wkt": _make_wkt_rect(x0, y0, x1, y1),
            })
        mask = polygons_to_mask(features, image_size=256)
        for cls in expected:
            assert np.any(mask == cls), f"Class {cls} not found in mask"

    def test_unclassified_becomes_ignore_index(self):
        features = [{
            "properties": {"subtype": "un-classified"},
            "wkt": _make_wkt_rect(5, 5, 60, 60),
        }]
        mask = polygons_to_mask(features, image_size=64)
        assert np.any(mask == IGNORE_INDEX), \
            f"Expected ignore index ({IGNORE_INDEX}) for un-classified"

    def test_higher_damage_overwrites_lower(self):
        """Two overlapping polygons: major-damage should overwrite no-damage.

        WKT coordinates are in 1024-pixel (xBD) space, as the function
        always treats input coordinates as xBD-native and scales to image_size.
        """
        features = [
            {
                "properties": {"subtype": "no-damage"},
                "wkt": _make_wkt_rect(0, 0, 1023, 1023),   # full 1024x1024 canvas
            },
            {
                "properties": {"subtype": "major-damage"},
                "wkt": _make_wkt_rect(300, 300, 700, 700),  # centre square
            },
        ]
        mask = polygons_to_mask(features, image_size=1024)
        # Centre of inner square: should be major-damage (class 3)
        assert mask[500, 500] == 3, f"Expected class 3, got {mask[500, 500]}"
        # Corner (outer region only): should be no-damage (class 1)
        assert mask[10, 10] == 1, f"Expected class 1, got {mask[10, 10]}"

    def test_malformed_wkt_is_skipped_gracefully(self):
        features = [
            {"properties": {"subtype": "destroyed"}, "wkt": "NOT_A_WKT"},
            {
                "properties": {"subtype": "no-damage"},
                "wkt": _make_wkt_rect(5, 5, 60, 60),
            },
        ]
        # Should not raise; valid polygon still rendered
        mask = polygons_to_mask(features, image_size=64)
        assert np.any(mask == 1), "Valid polygon should still be rasterized"

    def test_output_shape_matches_image_size(self):
        for size in (128, 256, 512):
            mask = polygons_to_mask([], image_size=size)
            assert mask.shape == (size, size)

    def test_subtype_mapping_completeness(self):
        """Every subtype in SUBTYPE_TO_CLASS must produce valid class values."""
        for subtype, expected_class in SUBTYPE_TO_CLASS.items():
            features = [{
                "properties": {"subtype": subtype},
                "wkt": _make_wkt_rect(10, 10, 50, 50),
            }]
            mask = polygons_to_mask(features, image_size=64)
            assert np.any(mask == expected_class), \
                f"subtype '{subtype}' expected class {expected_class}"


# ===========================================================================
# _discover_pairs
# ===========================================================================

class TestDiscoverPairs:
    def test_finds_correct_number_of_pairs(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        assert len(pairs) == 6

    def test_pair_keys_present(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        required_keys = {"pre_image", "post_image", "post_label", "disaster", "tile_id"}
        for pair in pairs:
            assert required_keys.issubset(pair.keys())

    def test_all_paths_exist(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        for pair in pairs:
            assert pair["pre_image"].exists(),  f"Missing: {pair['pre_image']}"
            assert pair["post_image"].exists(), f"Missing: {pair['post_image']}"
            assert pair["post_label"].exists(), f"Missing: {pair['post_label']}"

    def test_disaster_names_extracted(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        disasters = {p["disaster"] for p in pairs}
        assert "hurricane-harvey" in disasters
        assert "california-fire"  in disasters
        assert "nepal-flooding"   in disasters

    def test_raises_when_root_missing(self, tmp_path):
        with pytest.raises(DatasetNotFoundError):
            _discover_pairs(tmp_path / "nonexistent_dir")

    def test_raises_when_root_has_no_pairs(self, tmp_path):
        empty = tmp_path / "empty_xbd"
        empty.mkdir()
        (empty / "images").mkdir()
        with pytest.raises(DatasetNotFoundError):
            _discover_pairs(empty)


# ===========================================================================
# split_pairs_by_event
# ===========================================================================

class TestSplitPairsByEvent:
    def _make_pairs(self, events_and_counts: list) -> list:
        pairs = []
        for disaster, count in events_and_counts:
            for i in range(count):
                pairs.append({"disaster": disaster, "tile_id": str(i)})
        return pairs

    def test_splits_are_disjoint_by_event(self):
        pairs = self._make_pairs([
            ("event-A", 10), ("event-B", 8), ("event-C", 6),
            ("event-D", 5),  ("event-E", 4),
        ])
        train, val, test = split_pairs_by_event(pairs, seed=42)

        train_events = {p["disaster"] for p in train}
        val_events   = {p["disaster"] for p in val}
        test_events  = {p["disaster"] for p in test}

        # No event should appear in more than one split
        assert train_events.isdisjoint(val_events),  "Train/val event overlap"
        assert train_events.isdisjoint(test_events), "Train/test event overlap"
        assert val_events.isdisjoint(test_events),   "Val/test event overlap"

    def test_all_pairs_accounted_for(self):
        pairs = self._make_pairs([("A", 5), ("B", 5), ("C", 5), ("D", 5)])
        train, val, test = split_pairs_by_event(pairs, seed=0)
        assert len(train) + len(val) + len(test) == len(pairs)

    def test_reproducibility(self):
        pairs = self._make_pairs([("A", 4), ("B", 4), ("C", 4), ("D", 4)])
        t1, v1, ts1 = split_pairs_by_event(pairs, seed=7)
        t2, v2, ts2 = split_pairs_by_event(pairs, seed=7)
        assert [p["disaster"] for p in t1] == [p["disaster"] for p in t2]

    def test_different_seeds_may_differ(self):
        pairs = self._make_pairs([("A", 3), ("B", 3), ("C", 3), ("D", 3), ("E", 3)])
        t1, _, _ = split_pairs_by_event(pairs, seed=0)
        t2, _, _ = split_pairs_by_event(pairs, seed=99)
        # Not guaranteed to differ, but extremely likely with 5 events
        disasters_1 = sorted(p["disaster"] for p in t1)
        disasters_2 = sorted(p["disaster"] for p in t2)
        # They might differ; just ensure both produce valid non-empty splits
        assert len(t1) > 0 and len(t2) > 0

    def test_single_event_still_produces_splits(self):
        pairs = [{"disaster": "only-event", "tile_id": str(i)} for i in range(10)]
        # Should not crash; one split gets everything
        train, val, test = split_pairs_by_event(pairs, seed=0)
        assert len(train) + len(val) + len(test) == 10


# ===========================================================================
# XBDDataset
# ===========================================================================

class TestXBDDataset:
    def test_len(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        assert len(ds) == 6

    def test_getitem_output_types(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        item = ds[0]
        assert "image" in item
        assert "mask"  in item
        assert "metadata" in item
        assert isinstance(item["image"], np.ndarray)
        assert isinstance(item["mask"],  np.ndarray)

    def test_getitem_image_shape(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        item = ds[0]
        # (H, W, 6) -- pre+post stacked
        assert item["image"].shape == (64, 64, 6), \
            f"Unexpected image shape: {item['image'].shape}"

    def test_getitem_mask_shape(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        item = ds[0]
        assert item["mask"].shape == (64, 64), \
            f"Unexpected mask shape: {item['mask'].shape}"

    def test_getitem_image_dtype(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        item = ds[0]
        assert item["image"].dtype == np.float32

    def test_getitem_mask_dtype(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        item = ds[0]
        assert item["mask"].dtype == np.uint8

    def test_mask_values_in_valid_range(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        for i in range(len(ds)):
            mask = ds[i]["mask"]
            valid = mask[(mask != IGNORE_INDEX)]
            assert valid.min() >= 0 and valid.max() <= 4, \
                f"Mask values out of range [0, 4]: {valid.min()}-{valid.max()}"

    def test_metadata_keys_present(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        meta = ds[0]["metadata"]
        assert "disaster" in meta
        assert "tile_id"  in meta

    def test_repr(self, synthetic_dataset_dir):
        pairs = _discover_pairs(synthetic_dataset_dir)
        ds = XBDDataset(pairs, image_size=64)
        r = repr(ds)
        assert "XBDDataset" in r

    def test_empty_dataset_has_zero_len(self):
        ds = XBDDataset([], image_size=64)
        assert len(ds) == 0

    def test_malformed_label_produces_background_mask(self, tmp_path):
        """A JSON with invalid content should yield a background-only mask, not crash."""
        img_dir   = tmp_path / "images"
        label_dir = tmp_path / "labels"
        img_dir.mkdir(); label_dir.mkdir()

        _write_png(img_dir / "ev_001_post_disaster.png", size=64)
        _write_png(img_dir / "ev_001_pre_disaster.png",  size=64)

        bad_label = label_dir / "ev_001_post_disaster.json"
        bad_label.write_text("THIS IS NOT JSON", encoding="utf-8")

        pairs = [{
            "pre_image":  img_dir / "ev_001_pre_disaster.png",
            "post_image": img_dir / "ev_001_post_disaster.png",
            "post_label": bad_label,
            "disaster":   "ev",
            "tile_id":    "001",
        }]
        ds = XBDDataset(pairs, image_size=64, label_required=False)
        item = ds[0]
        # Should not raise; mask should be all zeros (background)
        assert np.all(item["mask"] == 0), "Malformed label should produce background mask"


# ===========================================================================
# DatasetNotFoundError
# ===========================================================================

class TestDatasetNotFoundError:
    def test_raised_on_missing_root(self, tmp_path):
        with pytest.raises(DatasetNotFoundError):
            _discover_pairs(tmp_path / "does_not_exist")

    def test_error_message_contains_path(self, tmp_path):
        import re
        missing = tmp_path / "no_such_dir"
        with pytest.raises(DatasetNotFoundError, match=re.escape(str(missing))):
            _discover_pairs(missing)

    @staticmethod
    def test_build_datasets_raises_when_config_null():
        """build_datasets() must raise when dataset_root is null in config."""
        from src.data.dataset import build_datasets
        config = {"paths": {"dataset_root": None}}
        with pytest.raises(DatasetNotFoundError):
            build_datasets(config)
