"""
DisasterLens AI -- Streamlit compatibility helpers
==================================================
Streamlit replaced ``use_container_width=True`` with ``width="stretch"`` and
eventually removed the old argument.  These helpers pick whichever one the
installed Streamlit version understands so the app works on both old and new
releases.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Callable, Dict

import streamlit as st


@lru_cache(maxsize=None)
def _supports_width_stretch(fn: Callable) -> bool:
    try:
        param = inspect.signature(fn).parameters.get("width")
    except (TypeError, ValueError):
        return False
    if param is None:
        return False
    # Old releases: width is an ``int | None`` pixel value (images).
    # New releases: annotation mentions "stretch" (``Literal["stretch", ...]``).
    return "stretch" in str(param.annotation)


def stretch(fn: Callable) -> Dict[str, object]:
    """Keyword arguments that make widget ``fn`` fill its container."""
    if _supports_width_stretch(fn):
        return {"width": "stretch"}
    try:
        if "use_container_width" in inspect.signature(fn).parameters:
            return {"use_container_width": True}
    except (TypeError, ValueError):
        pass
    return {}


def button(label: str, **kwargs):
    """``st.button`` that fills its container."""
    return st.button(label, **stretch(st.button), **kwargs)


def image(data, **kwargs):
    """``st.image`` that fills its container."""
    return st.image(data, **stretch(st.image), **kwargs)


def plotly_chart(fig, **kwargs):
    """``st.plotly_chart`` that fills its container."""
    kwargs.setdefault("config", {"displayModeBar": False})
    return st.plotly_chart(fig, **stretch(st.plotly_chart), **kwargs)


def download_button(label: str, **kwargs):
    """``st.download_button`` that fills its container."""
    return st.download_button(label, **stretch(st.download_button), **kwargs)


def page_link(page: str, label: str, icon: str | None = None):
    """``st.page_link`` that fills its container."""
    return st.page_link(page, label=label, icon=icon, **stretch(st.page_link))
