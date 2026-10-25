"""
DisasterLens AI -- Hugging Face checkpoint loader
==================================================
Downloads ``best_model.pth`` from the Hugging Face Hub repository
``Shaunak-006/disasterlens-checkpoint`` and caches it under
``artifacts/checkpoints/`` so that subsequent runs skip the download.

Usage::

    from src.inference.hf_loader import download_checkpoint
    local_path = download_checkpoint()          # returns pathlib.Path
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ── Hugging Face repository coordinates ─────────────────────────────────────
HF_REPO_ID   = "Shaunak-006/disasterlens-checkpoint"
HF_FILENAME  = "best_model.pth"
HF_REPO_TYPE = "model"

# ── Local cache location (inside the project tree) ───────────────────────────
_CACHE_DIR  = PROJECT_ROOT / "artifacts" / "checkpoints"
_CACHE_FILE = _CACHE_DIR / HF_FILENAME


def get_cached_path() -> Optional[Path]:
    """Return the cached checkpoint path if it already exists, else ``None``."""
    return _CACHE_FILE if _CACHE_FILE.exists() else None


def download_checkpoint(
    *,
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download ``best_model.pth`` from Hugging Face and return its local path.

    The file is stored at ``artifacts/checkpoints/best_model.pth`` inside the
    project tree.  If the file is already present and ``force`` is False, the
    download is skipped and the cached path is returned immediately.

    Args:
        force:     Re-download even if a local copy already exists.
        cache_dir: Override the default cache directory (tests use ``tmp_path``).

    Returns:
        Absolute :class:`~pathlib.Path` to the downloaded checkpoint.

    Raises:
        RuntimeError: If ``huggingface_hub`` is not installed.
        RuntimeError: If the download fails for any reason (network error,
                      authentication error, file not found on Hub, etc.).
    """
    dest_dir  = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    dest_file = dest_dir / HF_FILENAME

    # ── Fast path: already cached ─────────────────────────────────────────────
    if dest_file.exists() and not force:
        logger.info("Checkpoint already cached at %s", dest_file)
        return dest_file

    # ── Verify huggingface_hub is available ───────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "The 'huggingface_hub' package is required to download the "
            "DisasterLens checkpoint.\n"
            "Install it with:  pip install huggingface_hub\n"
            f"Original error: {exc}"
        ) from exc

    # ── Download ──────────────────────────────────────────────────────────────
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading checkpoint '%s' from '%s' ...", HF_FILENAME, HF_REPO_ID
    )
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type=HF_REPO_TYPE,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,  # copy real file, not a symlink
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download checkpoint '{HF_FILENAME}' from "
            f"'{HF_REPO_ID}'.\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            "Possible causes:\n"
            "  - No internet access (check your connection)\n"
            "  - The repository or file does not exist on Hugging Face\n"
            "  - A private repository requires HF_TOKEN to be set\n"
            "  - Transient network issue (try again in a moment)"
        ) from exc

    result = Path(downloaded).resolve()
    logger.info("Checkpoint saved to %s", result)
    return result
