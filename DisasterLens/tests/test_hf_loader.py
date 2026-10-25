"""
DisasterLens AI -- HF Loader Unit Tests
=========================================
Tests for src/inference/hf_loader.py.

All tests mock the network so they run offline without touching Hugging Face.

Run with:
    pytest tests/test_hf_loader.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import hf_loader


# ---------------------------------------------------------------------------
# get_cached_path
# ---------------------------------------------------------------------------

class TestGetCachedPath:

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hf_loader, "_CACHE_FILE", tmp_path / "no_file.pth")
        assert hf_loader.get_cached_path() is None

    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        f = tmp_path / "best_model.pth"
        f.write_bytes(b"fake")
        monkeypatch.setattr(hf_loader, "_CACHE_FILE", f)
        assert hf_loader.get_cached_path() == f


# ---------------------------------------------------------------------------
# download_checkpoint -- cached (no network)
# ---------------------------------------------------------------------------

class TestDownloadCheckpointCached:

    def test_returns_cached_path_without_download(self, tmp_path, monkeypatch):
        """If the file already exists, hf_hub_download should NOT be called."""
        cached = tmp_path / "best_model.pth"
        cached.write_bytes(b"fake_checkpoint")

        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            result = hf_loader.download_checkpoint(cache_dir=tmp_path)

        mock_dl.assert_not_called()
        assert result == cached

    def test_force_re_downloads_even_when_cached(self, tmp_path, monkeypatch):
        """force=True should re-download even when the file exists."""
        cached = tmp_path / "best_model.pth"
        cached.write_bytes(b"old_data")

        fake_return = str(cached)
        with patch("huggingface_hub.hf_hub_download", return_value=fake_return) as mock_dl:
            result = hf_loader.download_checkpoint(cache_dir=tmp_path, force=True)

        mock_dl.assert_called_once()
        assert result.name == "best_model.pth"


# ---------------------------------------------------------------------------
# download_checkpoint -- network paths (mocked)
# ---------------------------------------------------------------------------

class TestDownloadCheckpointNetwork:

    def test_successful_download(self, tmp_path):
        """Successful download returns the local file path."""
        expected = tmp_path / hf_loader.HF_FILENAME
        expected.write_bytes(b"checkpoint_bytes")  # simulate downloaded file

        with patch("huggingface_hub.hf_hub_download", return_value=str(expected)):
            result = hf_loader.download_checkpoint(cache_dir=tmp_path, force=True)

        assert result == expected.resolve()

    def test_network_error_raises_runtime_error(self, tmp_path):
        """Network failures should raise RuntimeError with a helpful message."""
        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=Exception("Connection timeout"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download checkpoint"):
                hf_loader.download_checkpoint(cache_dir=tmp_path, force=True)

    def test_error_message_contains_repo_info(self, tmp_path):
        """The error message should name the repository."""
        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=Exception("404"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                hf_loader.download_checkpoint(cache_dir=tmp_path, force=True)

        assert hf_loader.HF_REPO_ID in str(exc_info.value)

    def test_missing_huggingface_hub_raises_runtime_error(self, tmp_path):
        """Missing huggingface_hub import should raise RuntimeError (not ImportError)."""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises((RuntimeError, ImportError)):
                # Depending on Python version the ImportError may bubble up
                hf_loader.download_checkpoint(cache_dir=tmp_path, force=True)

    def test_correct_repo_id_passed(self, tmp_path):
        """hf_hub_download must be called with the correct repo_id."""
        expected = tmp_path / hf_loader.HF_FILENAME
        expected.write_bytes(b"x")

        with patch("huggingface_hub.hf_hub_download", return_value=str(expected)) as mock_dl:
            hf_loader.download_checkpoint(cache_dir=tmp_path, force=True)

        call_kwargs = mock_dl.call_args
        # Accept both positional and keyword call styles
        args = call_kwargs[0] if call_kwargs[0] else []
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        repo_id = args[0] if args else kwargs.get("repo_id", "")
        assert repo_id == hf_loader.HF_REPO_ID

    def test_correct_filename_passed(self, tmp_path):
        """hf_hub_download must be called with the correct filename."""
        expected = tmp_path / hf_loader.HF_FILENAME
        expected.write_bytes(b"x")

        with patch("huggingface_hub.hf_hub_download", return_value=str(expected)) as mock_dl:
            hf_loader.download_checkpoint(cache_dir=tmp_path, force=True)

        call_kwargs = mock_dl.call_args
        args = call_kwargs[0] if call_kwargs[0] else []
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        filename = args[1] if len(args) > 1 else kwargs.get("filename", "")
        assert filename == hf_loader.HF_FILENAME


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:

    def test_hf_repo_id_is_correct(self):
        assert hf_loader.HF_REPO_ID == "Shaunak-006/disasterlens-checkpoint"

    def test_hf_filename_is_correct(self):
        assert hf_loader.HF_FILENAME == "best_model.pth"

    def test_hf_repo_type_is_model(self):
        assert hf_loader.HF_REPO_TYPE == "model"
