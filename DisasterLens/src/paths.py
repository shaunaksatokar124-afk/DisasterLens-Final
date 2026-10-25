"""
DisasterLens AI -- Path helpers
===============================
Relative paths written in ``config.yaml`` (e.g. ``artifacts/analysis_history.json``)
must resolve against the *project root*, not against whatever directory the
user happened to launch Streamlit from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

#: Absolute path of the project root (the directory containing ``config.yaml``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    """Return ``p`` as an absolute path (relative paths anchor at PROJECT_ROOT).

    ``None`` / empty strings return ``None``.  ``~`` is expanded.
    """
    if p is None or str(p).strip() == "":
        return None
    path = Path(str(p)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
