"""
DisasterLens AI -- Shared UI Components
Reusable Streamlit components used across all pages.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml

from src.paths import PROJECT_ROOT, resolve_path
from src.ui import compat


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_config() -> dict:
    """Load and cache config.yaml from the project root."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_model_ready(config: dict) -> bool:
    """Return True if a trained checkpoint is available (local or HF cache).

    Checks both the configured ``checkpoint_dir`` AND the default Hugging Face
    download cache (``artifacts/checkpoints/``) so that the dashboard reflects
    the real state after a first-run download without requiring a page reload.
    """
    from src.inference.predictor import find_checkpoint
    if find_checkpoint(config) is not None:
        return True
    # Also check the HF download cache directly (avoids importing hf_loader
    # when not needed, but still reflects a previously downloaded file).
    try:
        from src.inference.hf_loader import get_cached_path
        return get_cached_path() is not None
    except Exception:
        return False


def demo_mode_enabled(config: dict) -> bool:
    """Return True if the heuristic demo predictor may be used as a fallback."""
    return bool(config.get("inference", {}).get("demo_mode", False))


def inference_mode(config: dict) -> str:
    """Return ``"model"`` (real checkpoint), ``"demo"`` (heuristic) or ``"none"``."""
    if check_model_ready(config):
        return "model"
    return "demo" if demo_mode_enabled(config) else "none"


def check_inference_ready(config: dict) -> bool:
    """True when the Image Analysis page can run (real model *or* demo mode)."""
    return inference_mode(config) != "none"


def check_dataset_ready(config: dict) -> bool:
    """Return True if dataset_root exists and is non-empty."""
    root = resolve_path(config.get("paths", {}).get("dataset_root"))
    if root is None:
        return False
    try:
        return root.is_dir() and any(root.iterdir())
    except OSError:
        return False


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(config_key: str, _config: dict):
    """Create (once per config) and cache the ``DamagePredictor``."""
    from src.inference.predictor import DamagePredictor
    return DamagePredictor(_config)


def config_key(config: dict) -> str:
    """Cache key that changes whenever the settings affecting the model change."""
    return json.dumps(
        {
            "paths": config.get("paths", {}),
            "model": config.get("model", {}),
            "inference": config.get("inference", {}),
        },
        sort_keys=True,
        default=str,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(config: dict, model_ready: bool, dataset_ready: bool) -> None:
    """Render the shared navigation sidebar."""
    mode = inference_mode(config)
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <span class="brand-mark"></span>
            <div style="font-size:1.05rem;font-weight:700;color:#00D4FF;margin-top:0.25rem">
                DisasterLens AI
            </div>
            <div style="font-size:0.68rem;color:#546E7A;margin-top:0.15rem">
                v1.0.0 &mdash; Damage Assessment
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**System Status**")
        if mode == "model":
            mb = '<span class="badge-ready">&#9679; Model Ready</span>'
        elif mode == "demo":
            mb = '<span class="badge-warning">&#9679; Demo Mode (no checkpoint)</span>'
        else:
            mb = '<span class="badge-notready">&#9679; Model Not Ready</span>'
        db = '<span class="badge-ready">&#9679; Dataset Found</span>' if dataset_ready \
            else '<span class="badge-warning">&#9679; Dataset Not Configured</span>'
        st.markdown(mb, unsafe_allow_html=True)
        st.markdown(db, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Navigate**")
        compat.page_link("app.py", label="Overview", icon=":material/home:")
        compat.page_link("pages/1_Image_Analysis.py", label="Image Analysis", icon=":material/biotech:")
        compat.page_link("pages/2_Analytics.py", label="Analytics", icon=":material/monitoring:")
        compat.page_link("pages/3_Model_Information.py", label="Model Information", icon=":material/neurology:")
        compat.page_link("pages/4_Assessment_Reports.py", label="Assessment Reports", icon=":material/description:")

        st.divider()
        st.markdown("""
        <div class="disclaimer-box">
            <strong>Disclaimer:</strong> Model predictions require human
            verification and must not be used as the sole basis for emergency
            response decisions.
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Analysis History
# ---------------------------------------------------------------------------

def _history_path(config: dict) -> Path:
    raw = config.get("paths", {}).get("analysis_history_file", "artifacts/analysis_history.json")
    return resolve_path(raw) or (PROJECT_ROOT / "artifacts" / "analysis_history.json")


def load_history(config: dict) -> list:
    """Load the analysis history JSON. Returns empty list if missing/corrupt."""
    path = _history_path(config)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def save_history_entry(config: dict, entry: dict) -> None:
    """Append an analysis entry to the history JSON file (atomic write)."""
    path = _history_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(config)
    history.append(entry)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Training / Evaluation Artifacts
# ---------------------------------------------------------------------------

def _load_json_file(raw_path: str) -> Optional[dict]:
    path = resolve_path(raw_path)
    if path is None or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_training_history(config: dict) -> Optional[dict]:
    """Load training history JSON if it exists (and is valid)."""
    data = _load_json_file(
        config.get("paths", {}).get("training_history_file", "artifacts/training_history.json")
    )
    if data is not None and not data.get("train_loss"):
        return None
    return data


def load_evaluation_results(config: dict) -> Optional[dict]:
    """Load evaluation results JSON if it exists (and is valid)."""
    return _load_json_file(
        config.get("paths", {}).get("evaluation_results_file", "artifacts/evaluation_results.json")
    )


# ---------------------------------------------------------------------------
# Common UI elements
# ---------------------------------------------------------------------------

def model_not_ready_banner() -> None:
    """Display a warning banner when no checkpoint is loaded and demo mode is off."""
    st.markdown("""
    <div class="warning-box">
        <div style="font-weight:700;color:#FFB800;margin-bottom:0.4rem">
            Model Checkpoint Not Loaded
        </div>
        <div style="font-size:0.87rem;color:#B0BEC5;line-height:1.65">
            The checkpoint will be downloaded automatically from Hugging Face
            (<code style="color:#00D4FF">Shaunak-006/disasterlens-checkpoint</code>)
            on first use &mdash; an internet connection is required.<br><br>
            If the automatic download failed, you can also:<br>
            1. Download <code style="color:#00D4FF">best_model.pth</code> manually and place it in
               <code style="color:#00D4FF">artifacts/checkpoints/</code><br>
            2. Or set <code style="color:#00D4FF">checkpoint_dir</code> in
               <code style="color:#00D4FF">config.yaml</code> and restart Streamlit.<br>
            <em>The model is experimental &mdash; always verify predictions independently.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


def demo_mode_banner() -> None:
    """Display a banner explaining that results come from the demo predictor."""
    st.markdown("""
    <div class="warning-box">
        <div style="font-weight:700;color:#FFB800;margin-bottom:0.3rem">
            Demo Mode &mdash; no trained checkpoint loaded
        </div>
        <div style="font-size:0.85rem;color:#B0BEC5;line-height:1.65">
            Predictions are produced by an image-processing heuristic
            (texture + pre/post change detection), <strong>not</strong> by the
            trained U-Net. They illustrate the workflow only and must not be
            interpreted as real damage assessments. To use the real model, train
            it with the Colab notebook and set
            <code style="color:#00D4FF">checkpoint_dir</code> in
            <code style="color:#00D4FF">config.yaml</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)


def dataset_not_ready_banner() -> None:
    """Display a warning banner when dataset path is not configured."""
    st.markdown("""
    <div class="warning-box">
        <div style="font-weight:700;color:#FFB800;margin-bottom:0.4rem">
            Dataset Not Configured
        </div>
        <div style="font-size:0.87rem;color:#B0BEC5;line-height:1.65">
            Set <code style="color:#00D4FF">dataset_root</code> in
            <code style="color:#00D4FF">config.yaml</code> to the root of your
            downloaded xBD dataset. Download from
            <a href="https://xview2.org" target="_blank" style="color:#00D4FF">
            xview2.org</a>.
        </div>
    </div>
    """, unsafe_allow_html=True)
