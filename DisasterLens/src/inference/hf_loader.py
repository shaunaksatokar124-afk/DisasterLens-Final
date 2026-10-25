```python
"""
DisasterLens AI -- Hugging Face checkpoint loader
Downloads and caches best_model.pth from Hugging Face.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

HF_REPO_ID = "Shaunak-006/disasterlens-checkpoint"
HF_FILENAME = "best_model.pth"
HF_REPO_TYPE = "model"

_CACHE_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
_CACHE_FILE = _CACHE_DIR / HF_FILENAME


def get_cached_path() -> Optional[Path]:
    """Return the cached checkpoint if it exists."""
    return _CACHE_FILE if _CACHE_FILE.is_file() else None


def download_checkpoint(
    *,
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download the checkpoint and return its local path."""

    dest_dir = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    dest_file = dest_dir / HF_FILENAME

    if dest_file.is_file() and not force:
        logger.info("Using cached checkpoint: %s", dest_file)
        return dest_file

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Add "
            "'huggingface_hub' to requirements.txt."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type=HF_REPO_TYPE,
            local_dir=str(dest_dir),
        )

        result = Path(downloaded).resolve()

        if not result.is_file():
            raise FileNotFoundError(
                f"Downloaded checkpoint was not found at {result}"
            )

        logger.info("Checkpoint available at %s", result)
        return result

    except Exception as exc:
        logger.exception("Checkpoint download failed")
        raise RuntimeError(
            f"Could not download {HF_FILENAME} from {HF_REPO_ID}. "
            f"{type(exc).__name__}: {exc}"
        ) from exc
```
