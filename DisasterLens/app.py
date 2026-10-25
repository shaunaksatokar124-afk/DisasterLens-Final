"""
DisasterLens AI -- Overview Page (app.py)
Main entry point for the Streamlit multipage application.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for src/ imports
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.ui.styles import CSS, FOOTER_HTML
from src.ui import compat
from src.ui.components import (
    load_config,
    check_model_ready,
    check_dataset_ready,
    inference_mode,
    load_history,
    render_sidebar,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DisasterLens AI — Overview",
    page_icon=":material/satellite_alt:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    config      = load_config()
    model_ready = check_model_ready(config)
    mode        = inference_mode(config)
    ds_ready    = check_dataset_ready(config)
    history     = load_history(config)

    render_sidebar(config, model_ready, ds_ready)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div style="display:flex;align-items:center;gap:1.1rem;margin-bottom:1.2rem">
            <span class="brand-mark brand-mark-lg"></span>
            <div>
                <div class="hero-title">DisasterLens AI</div>
                <div class="hero-subtitle">
                    AI-Powered Disaster Damage Assessment &nbsp;·&nbsp;
                    Deep Learning &nbsp;·&nbsp; Satellite Imagery
                </div>
            </div>
        </div>
        <div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:1rem">
            <span class="tag-pill">U-Net Architecture</span>
            <span class="tag-pill">ResNet-34 Encoder</span>
            <span class="tag-pill">xBD Dataset</span>
            <span class="tag-pill">PyTorch</span>
            <span class="tag-pill">Semantic Segmentation</span>
            <span class="tag-pill">4 Damage Levels + Background</span>
            <span class="tag-pill">6-Channel Pre+Post Input</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── System status row ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        sc, si, st_txt = {
            "model": ("#00FF9F", "", "Ready for Inference"),
            "demo":  ("#FFB800", "", "Demo Mode (no checkpoint)"),
            "none":  ("#FF6B6B", "", "Checkpoint Not Loaded"),
        }[mode]
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size:0.7rem;color:#546E7A;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.45rem">Model Status</div>
            <div style="font-size:1rem;font-weight:600;color:{sc}">{si} {st_txt}</div>
            <div style="font-size:0.78rem;color:#546E7A;margin-top:.35rem">
                U-Net + ResNet-34 &nbsp;|&nbsp; 5 classes
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        dc = "#00FF9F" if ds_ready else "#FFB800"
        di = "" if ds_ready else ""
        ds_txt = "Dataset Configured" if ds_ready else "Dataset Not Configured"
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size:0.7rem;color:#546E7A;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.45rem">Dataset Status</div>
            <div style="font-size:1rem;font-weight:600;color:{dc}">{di} {ds_txt}</div>
            <div style="font-size:0.78rem;color:#546E7A;margin-top:.35rem">
                xBD &nbsp;|&nbsp; 11,034 image pairs
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        hc = len(history)
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size:0.7rem;color:#546E7A;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.45rem">Analysis History</div>
            <div style="font-size:1rem;font-weight:600;color:#E8EAF6">
                 {hc} {"Analyses" if hc != 1 else "Analysis"} Completed
            </div>
            <div style="font-size:0.78rem;color:#546E7A;margin-top:.35rem">
                Stored locally in artifacts/
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <span class="metric-value">11,034</span>
            <div class="metric-label">Pre/Post Image Pairs</div>
        </div>
        <div class="metric-card">
            <span class="metric-value">850K+</span>
            <div class="metric-label">Building Annotations</div>
        </div>
        <div class="metric-card">
            <span class="metric-value">19</span>
            <div class="metric-label">Disaster Events</div>
        </div>
        <div class="metric-card">
            <span class="metric-value">4</span>
            <div class="metric-label">Damage Categories</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column content ────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <p style="margin:0;font-size:.91rem;line-height:1.75;color:#B0BEC5 !important">
            <strong style="color:#00D4FF">DisasterLens AI</strong> is a deep-learning application that
            analyses post-disaster satellite and aerial imagery to classify visible building damage at the
            pixel level. It is built on the
            <strong style="color:#E8EAF6">xBD dataset</strong> — the largest publicly available
            disaster damage benchmark — using a
            <strong style="color:#E8EAF6">U-Net + ResNet-34</strong> encoder for semantic segmentation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Damage Categories</div>', unsafe_allow_html=True)

        cats = [
            ("#00C851", "No Damage",    "Structure intact — no visible damage"),
            ("#FFD600", "Minor Damage", "Partial debris, roof or wall compromise"),
            ("#FF6D00", "Major Damage", "Severe structural failure, partial collapse"),
            ("#FF1744", "Destroyed",    "Total collapse or complete destruction"),
        ]
        for color, name, desc in cats:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.75rem;
                        padding:.55rem 0;border-bottom:1px solid rgba(255,255,255,.05)">
                <div style="width:11px;height:11px;border-radius:3px;
                            background:{color};flex-shrink:0"></div>
                <div>
                    <span style="font-weight:600;color:#E8EAF6;font-size:.9rem">{name}</span>
                    <span style="color:#546E7A;font-size:.81rem;margin-left:.45rem">— {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Quick Start</div>', unsafe_allow_html=True)

        if not model_ready:
            _title = ("Running in Demo Mode" if mode == "demo"
                      else " Model Setup Required")
            st.markdown(f"""
            <div class="warning-box">
                <div style="font-weight:700;color:#FFB800;margin-bottom:.4rem">
                    {_title}
                </div>
                <div style="font-size:.84rem;color:#B0BEC5 !important;line-height:1.65">
                    1. Obtain the xBD dataset from
                       <a href="https://xview2.org" target="_blank" style="color:#00D4FF">xview2.org</a><br>
                    2. Set <code style="color:#00D4FF">dataset_root</code> in config.yaml<br>
                    3. Train using the Colab notebook<br>
                    4. Set <code style="color:#00D4FF">checkpoint_dir</code> and restart
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <div style="font-weight:700;color:#00FF9F;margin-bottom:.3rem">
                     Model is ready
                </div>
                <div style="font-size:.84rem;color:#B0BEC5 !important">
                    Upload a post-disaster image to begin analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if compat.button("Start Image Analysis"):
            st.switch_page("pages/1_Image_Analysis.py")

        st.markdown('<div class="section-header" style="margin-top:1.5rem">Resources</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        -  [xView2 Dataset Portal](https://xview2.org)
        -  [xBD Research Paper](https://arxiv.org/abs/1911.09296)
        -  Training notebook: `notebooks/DisasterLens_AI_Training.ipynb`
        -  Configuration file: `config.yaml`
        """)

    # ── Recent history ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recent Analyses</div>', unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:2rem;
                    background:rgba(15,22,41,.4);
                    border:1px dashed rgba(0,212,255,.15);border-radius:12px;color:#546E7A">
            <div>No analyses yet. Upload an image to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        shown = list(reversed(history))[:5]
        for entry in shown:
            ts   = entry.get("timestamp", "Unknown")
            fn   = entry.get("filename", "Unknown file")
            aid  = entry.get("id", "—")
            st.markdown(f"""
            <div class="history-row">
                <div>
                    <div style="font-weight:600;color:#E8EAF6;font-size:.88rem"> {fn}</div>
                    <div style="font-size:.73rem;color:#546E7A">ID: {aid} · {ts}</div>
                </div>
                <span class="badge-ready">Completed</span>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
