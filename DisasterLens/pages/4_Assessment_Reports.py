"""
DisasterLens AI -- Assessment Reports Page
==========================================
Displays the local analysis history, allows users to review past
damage assessments, and generates / downloads a PDF report for
any selected analysis.

Navigation: app.py  pages/4_Assessment_Reports.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import streamlit as st
from PIL import Image

from src.paths import resolve_path
from src.ui import compat
from src.ui.styles import CSS, FOOTER_HTML
from src.ui.components import (
    check_dataset_ready,
    check_model_ready,
    load_config,
    load_history,
    render_sidebar,
)

st.set_page_config(
    page_title="DisasterLens AI — Assessment Reports",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_timestamp(ts: str) -> str:
    """Format ISO timestamp to human-readable string."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        return ts


def _load_image_safe(path: str | None):
    """Return PIL Image or None if path is missing."""
    if not path:
        return None
    p = resolve_path(path)
    if p is None or not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def _generate_pdf(entry: dict, config: dict) -> bytes | None:
    """Call PDF generator and surface any import / generation error."""
    try:
        from src.reports.pdf_generator import generate_report_pdf
        return generate_report_pdf(entry, config)
    except ImportError as exc:
        st.error(f"PDF generation requires reportlab: {exc}")
        return None
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
        return None


def _stats_donut_chart(stats: dict):
    """Build a Plotly donut chart from per-class pixel stats."""
    import plotly.graph_objects as go

    names  = [n for n in stats if n != "Background"]
    pcts   = [stats[n]["percentage"] for n in names]
    colors = [stats[n]["color_hex"]  for n in names]

    fig = go.Figure(go.Pie(
        labels=names,
        values=pcts,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Coverage: %{value:.2f}%<extra></extra>",
        textfont=dict(color="#E8EAF6", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#B0BEC5"),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        height=230,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    config      = load_config()
    model_ready = check_model_ready(config)
    ds_ready    = check_dataset_ready(config)
    render_sidebar(config, model_ready, ds_ready)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="hero-title" style="font-size:1.8rem">Assessment Reports</div>
        <div class="hero-subtitle">
            Review past damage analyses and download PDF reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    history = load_history(config)

    # ── Empty state ───────────────────────────────────────────────────────────
    if not history:
        st.markdown("""
        <div style="text-align:center;padding:3.5rem;
                    background:rgba(15,22,41,.4);
                    border:1px dashed rgba(0,212,255,.15);
                    border-radius:14px;margin-top:1rem">
            <div style="color:#E8EAF6;font-size:1.05rem;font-weight:600;
                        margin-bottom:.5rem">No analyses yet</div>
            <div style="color:#546E7A;font-size:.88rem;max-width:420px;
                        margin:0 auto;line-height:1.7">
                Complete an analysis on the
                <strong style="color:#00D4FF">Image Analysis</strong> page
                and click <em>Save Analysis</em> to store results here.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Go to Image Analysis"):
            st.switch_page("pages/1_Image_Analysis.py")
        return

    # ── History list + detail split ───────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">Analysis History '
        f'<span style="color:#546E7A;font-size:.8rem;font-weight:400">'
        f'({len(history)} {"record" if len(history)==1 else "records"})</span></div>',
        unsafe_allow_html=True,
    )

    # Reverse so most-recent is first
    history_rev = list(reversed(history))

    # Build selection labels
    labels = [
        f"[{_fmt_timestamp(e.get('timestamp',''))}]  {e.get('filename','unknown')}  "
        f"(ID: {e.get('id','—')})"
        for e in history_rev
    ]

    selected_label = st.selectbox(
        "Select an analysis to review",
        options=labels,
        index=0,
        label_visibility="collapsed",
    )
    selected_idx  = labels.index(selected_label)
    selected_entry = history_rev[selected_idx]

    st.markdown('<div class="section-header">Analysis Details</div>', unsafe_allow_html=True)

    # ── Metadata row ──────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.4rem">Analysis ID</div>
            <div style="font-size:.95rem;font-weight:600;color:#E8EAF6">
                {selected_entry.get('id','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.4rem">Timestamp</div>
            <div style="font-size:.95rem;font-weight:600;color:#E8EAF6">
                {_fmt_timestamp(selected_entry.get('timestamp','—'))}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="status-card">
            <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.4rem">Source File</div>
            <div style="font-size:.95rem;font-weight:600;color:#E8EAF6;
                        word-break:break-all">
                {selected_entry.get('filename','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Image panels ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Imagery</div>', unsafe_allow_html=True)

    post_img    = _load_image_safe(selected_entry.get("post_img"))
    overlay_img = _load_image_safe(selected_entry.get("overlay_img"))

    img_col1, img_col2 = st.columns(2, gap="medium")
    with img_col1:
        st.markdown("**Post-Disaster Image**")
        if post_img:
            compat.image(post_img)
        else:
            st.markdown("""
            <div style="height:180px;display:flex;align-items:center;
                        justify-content:center;background:rgba(15,22,41,.6);
                        border:1px dashed rgba(0,212,255,.2);border-radius:8px;
                        color:#546E7A;font-size:.84rem">
                Image file not found on disk
            </div>
            """, unsafe_allow_html=True)

    with img_col2:
        st.markdown("**Damage Prediction Overlay**")
        if overlay_img:
            compat.image(overlay_img)
        else:
            st.markdown("""
            <div style="height:180px;display:flex;align-items:center;
                        justify-content:center;background:rgba(15,22,41,.6);
                        border:1px dashed rgba(0,212,255,.2);border-radius:8px;
                        color:#546E7A;font-size:.84rem">
                Overlay file not found on disk
            </div>
            """, unsafe_allow_html=True)

    # ── Statistics ────────────────────────────────────────────────────────────
    stats: dict = selected_entry.get("stats", {})

    if stats:
        st.markdown('<div class="section-header">Damage Statistics</div>',
                    unsafe_allow_html=True)

        chart_col, table_col = st.columns([1, 1], gap="large")

        with chart_col:
            try:
                fig = _stats_donut_chart(stats)
                compat.plotly_chart(fig)
            except Exception as exc:
                st.info(f"Chart unavailable: {exc}")

        with table_col:
            st.markdown("<br>", unsafe_allow_html=True)
            # Colour legend rows
            for class_name, s in stats.items():
                color = s.get("color_hex", "#555")
                count = s.get("count", 0)
                pct   = s.get("percentage", 0.0)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.7rem;
                            padding:.45rem 0;
                            border-bottom:1px solid rgba(255,255,255,.05)">
                    <div style="width:12px;height:12px;border-radius:3px;
                                background:{color};flex-shrink:0"></div>
                    <div style="flex:1">
                        <span style="font-weight:600;color:#E8EAF6;
                                     font-size:.88rem">{class_name}</span>
                    </div>
                    <div style="color:#546E7A;font-size:.82rem;min-width:70px;
                                text-align:right">{count:,} px</div>
                    <div style="color:#00D4FF;font-size:.88rem;font-weight:600;
                                min-width:54px;text-align:right">{pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No statistics stored for this analysis.")

    # ── PDF Report ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Download Report</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-bottom:1rem">
        <div style="font-size:.86rem;color:#B0BEC5 !important;line-height:1.65">
            <strong style="color:#00D4FF">PDF Report</strong> includes the
            original imagery, damage overlay, per-class statistics, model
            information, and a required human-verification disclaimer.
        </div>
    </div>
    """, unsafe_allow_html=True)

    btn_col, msg_col = st.columns([1, 3])
    with btn_col:
        generate_clicked = compat.button("Generate PDF")

    pdf_key = f"pdf_{selected_entry.get('id', 'unknown')}"
    if generate_clicked:
        with st.spinner("Generating PDF report..."):
            pdf_bytes = _generate_pdf(selected_entry, config)
        if pdf_bytes:
            st.session_state[pdf_key] = pdf_bytes
        else:
            st.session_state.pop(pdf_key, None)

    pdf_bytes = st.session_state.get(pdf_key)
    if pdf_bytes:
        analysis_id = selected_entry.get("id", "unknown")
        filename    = f"DisasterLens_Report_{analysis_id}.pdf"
        with btn_col:
            compat.download_button(
                " Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"dl_{pdf_key}",
            )
        with msg_col:
            st.success(f"PDF ready — {len(pdf_bytes) / 1024:.1f} KB  ({filename})")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-box" style="margin-top:2rem">
         <strong>Disclaimer:</strong> These predictions are produced by a
        deep learning model trained on the xBD dataset. They reflect
        pixel-level damage indicators visible in the imagery and have not
        been validated by domain experts for this specific analysis. Results
        must not be used as the sole basis for emergency response, insurance
        claims, or any operational decision without independent human expert
        verification.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
