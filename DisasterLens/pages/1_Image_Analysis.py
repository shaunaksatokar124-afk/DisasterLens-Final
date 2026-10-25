"""
DisasterLens AI -- Image Analysis Page
Upload pre/post-disaster images, run inference, and visualise results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import uuid
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from src.paths import PROJECT_ROOT
from src.ui import compat
from src.ui.styles import CSS, FOOTER_HTML
from src.ui.components import (
    check_dataset_ready,
    check_inference_ready,
    check_model_ready,
    config_key,
    demo_mode_banner,
    get_predictor,
    inference_mode,
    load_config,
    model_not_ready_banner,
    render_sidebar,
    save_history_entry,
)
from src.visualization.overlay import (
    CLASS_COLORS_RGB,
    CLASS_NAMES,
    blend_overlay,
    colorize_mask,
    compute_class_statistics,
    create_comparison,
    numpy_to_pil,
    pil_to_numpy,
    resize_keep_aspect,
)

st.set_page_config(
    page_title="DisasterLens AI — Image Analysis",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_SIZE = 1024


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_DIR = PROJECT_ROOT / "sample_data"
SAMPLE_PRE  = SAMPLE_DIR / "pre_disaster_sample.png"
SAMPLE_POST = SAMPLE_DIR / "post_disaster_sample.png"
ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def validate_image(uploaded_file) -> tuple[bool, str]:
    """Return (valid, error_message)."""
    if uploaded_file is None:
        return False, ""
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return False, f"Unsupported format: '{ext}'. Use PNG, JPG, or TIF."
    size_mb = uploaded_file.size / 1_048_576
    if size_mb > 50:
        return False, f"File too large ({size_mb:.1f} MB). Maximum is 50 MB."
    return True, ""


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL image (16-bit, palette, RGBA, multiband TIFF) to 8-bit RGB."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    return img.convert("RGB")


def load_image(source) -> np.ndarray:
    """Load an uploaded file or a path as an (H, W, 3) uint8 array (<= MAX_SIZE)."""
    with Image.open(source) as img:
        arr = np.array(_to_rgb(img))
    return resize_keep_aspect(arr, MAX_SIZE)


def file_signature(uploaded_file) -> str:
    """Stable id for an upload so cached results can be tied to their inputs."""
    if uploaded_file is None:
        return "-"
    data = uploaded_file.getvalue()
    return f"{uploaded_file.name}:{len(data)}:{hashlib.md5(data).hexdigest()[:12]}"


def build_stats_chart(stats: dict) -> go.Figure:
    names  = [n for n in stats if n != "Background"]
    counts = [stats[n]["count"] for n in names]
    pcts   = [stats[n]["percentage"] for n in names]
    colors = [stats[n]["color_hex"] for n in names]

    fig = go.Figure(go.Bar(
        x=pcts,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{p:.1f}%" for p in pcts],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Pixels: %{customdata:,}<br>Coverage: %{x:.2f}%<extra></extra>",
        customdata=counts,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#B0BEC5"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="Coverage (%)", gridcolor="rgba(255,255,255,0.06)",
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=220,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def _empty_state() -> None:
    st.markdown("""
    <div style="text-align:center;padding:3rem;
                background:rgba(15,22,41,.4);
                border:1px dashed rgba(0,212,255,.15);border-radius:12px">
        <div style="color:#E8EAF6;font-size:1rem;font-weight:600;margin-bottom:.4rem">
            Upload a post-disaster image to begin
        </div>
        <div style="color:#546E7A;font-size:.86rem">
            Supported formats: PNG, JPG, TIF &nbsp;|&nbsp; Max 50 MB
        </div>
    </div>
    """, unsafe_allow_html=True)


def _stats_table_html(stats: dict) -> str:
    rows_html = ""
    for name, s_ in stats.items():
        r, g, b = CLASS_COLORS_RGB[s_["class_idx"]]
        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,.06)">
            <td style="padding:.5rem .75rem;display:flex;align-items:center;gap:.5rem">
                <div style="width:10px;height:10px;border-radius:2px;
                            background:rgb({r},{g},{b});flex-shrink:0"></div>
                <span style="color:#E8EAF6;font-size:.86rem">{name}</span>
            </td>
            <td style="padding:.5rem .75rem;color:#B0BEC5;font-size:.86rem">{s_["count"]:,}</td>
            <td style="padding:.5rem .75rem;color:#00D4FF;font-size:.86rem;font-weight:600">
                {s_["percentage"]:.2f}%</td>
        </tr>"""
    th = ('padding:.5rem .75rem;text-align:left;color:#546E7A;font-size:.72rem;'
          'text-transform:uppercase;letter-spacing:.06em')
    return f"""
    <table style="width:100%;border-collapse:collapse;margin-top:.5rem">
        <thead><tr style="border-bottom:1px solid rgba(0,212,255,.2)">
            <th style="{th}">Class</th><th style="{th}">Pixels</th><th style="{th}">Coverage</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>"""


def _render_results(config: dict, result: dict) -> None:
    mask     = result["mask"]
    post_arr = result["post"]

    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
    if result.get("is_demo"):
        st.caption("Demo-mode result: heuristic output, not a trained-model prediction.")

    alpha = st.slider(
        "Overlay opacity", min_value=0.10, max_value=0.90,
        value=0.45, step=0.05,
        help="Adjust the transparency of the damage mask overlay.",
    )

    overlay_arr    = blend_overlay(post_arr, mask, alpha=alpha)
    colored_arr    = colorize_mask(mask)
    comparison_arr = create_comparison(post_arr, overlay_arr)
    stats          = compute_class_statistics(mask)

    tab1, tab2, tab3 = st.tabs(["Side-by-Side", "Mask Only", "Statistics"])

    with tab1:
        compat.image(comparison_arr, caption="Original vs. Damage Prediction Overlay")

    with tab2:
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            compat.image(overlay_arr, caption="Damage Overlay")
        with c_col2:
            compat.image(colored_arr, caption="Class Mask")

        st.markdown('<div class="section-header">Colour Legend</div>',
                    unsafe_allow_html=True)
        legend_cols = st.columns(5)
        for i, (name, col_) in enumerate(zip(CLASS_NAMES.values(), CLASS_COLORS_RGB.values())):
            r, g, b = col_
            with legend_cols[i]:
                st.markdown(f"""
                <div style="text-align:center">
                    <div style="width:100%;height:18px;border-radius:4px;
                                background:rgb({r},{g},{b});margin-bottom:.3rem"></div>
                    <div style="font-size:.72rem;color:#B0BEC5">{name}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        compat.plotly_chart(build_stats_chart(stats))
        st.markdown(_stats_table_html(stats), unsafe_allow_html=True)

    # ── Save to history ────────────────────────────────────────────────────
    st.divider()
    col_sv, col_msg = st.columns([1, 3])
    with col_sv:
        already = st.session_state.get("saved_result_id") == result["result_id"]
        if compat.button("Save Analysis", disabled=already):
            analysis_id  = uuid.uuid4().hex[:8]
            artifact_dir = PROJECT_ROOT / config.get("paths", {}).get("artifacts_dir", "artifacts")
            img_dir      = artifact_dir / analysis_id
            img_dir.mkdir(parents=True, exist_ok=True)

            numpy_to_pil(post_arr).save(img_dir / "post.png")
            numpy_to_pil(overlay_arr).save(img_dir / "overlay.png")

            def _rel(path: Path) -> str:
                try:
                    return path.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    return str(path)

            entry = {
                "id":          analysis_id,
                "timestamp":   datetime.now().isoformat(timespec="seconds"),
                "filename":    result["filename"],
                "post_img":    _rel(img_dir / "post.png"),
                "overlay_img": _rel(img_dir / "overlay.png"),
                "stats":       stats,
                "model_device": result["device"],
                "mode":        "demo" if result.get("is_demo") else "model",
            }
            save_history_entry(config, entry)
            st.session_state["saved_result_id"] = result["result_id"]
            st.success(f"Analysis saved — ID: {analysis_id}")
        elif already:
            st.caption("Already saved ")
    with col_msg:
        st.markdown("""
        <div class="disclaimer-box" style="margin-top:.4rem">
             These predictions are generated by a deep learning model and require
            human expert verification before use in any operational context.
        </div>
        """, unsafe_allow_html=True)


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    config      = load_config()
    model_ready = check_model_ready(config)
    ds_ready    = check_dataset_ready(config)
    can_run     = check_inference_ready(config)
    mode        = inference_mode(config)
    render_sidebar(config, model_ready, ds_ready)

    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="hero-title" style="font-size:1.8rem">Image Analysis</div>
        <div class="hero-subtitle">
            Upload pre/post-disaster satellite images and run damage segmentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if mode == "demo":
        demo_mode_banner()
    elif mode == "none":
        model_not_ready_banner()
        st.divider()

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Upload Images</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="margin-bottom:1rem">
        <div style="font-size:.86rem;color:#B0BEC5 !important;line-height:1.6">
            <strong style="color:#00D4FF">Tip:</strong>
            Upload both a <em>pre-disaster</em> and a <em>post-disaster</em> image for the best accuracy.
            The model uses change information between the two images.
            If only the post-disaster image is available, it will be used for both input channels
            (accuracy will be lower). Images of different sizes are resized automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)

    have_samples = SAMPLE_PRE.exists() and SAMPLE_POST.exists()
    use_sample = False
    if have_samples:
        sc1, sc2 = st.columns([1, 3])
        with sc1:
            if compat.button("Load sample images", key="load_sample"):
                st.session_state["use_sample"] = True
        with sc2:
            if st.session_state.get("use_sample"):
                st.caption("Using the bundled synthetic sample pair. Upload files below to override.")
        use_sample = bool(st.session_state.get("use_sample"))

    col_pre, col_post = st.columns(2)

    with col_pre:
        st.markdown("**Pre-Disaster Image** *(optional but recommended)*")
        pre_file = st.file_uploader(
            "Pre-disaster image", type=[e[1:] for e in ALLOWED_EXT],
            key="pre_uploader", label_visibility="collapsed",
        )

    with col_post:
        st.markdown("**Post-Disaster Image** *(required)*")
        post_file = st.file_uploader(
            "Post-disaster image", type=[e[1:] for e in ALLOWED_EXT],
            key="post_uploader", label_visibility="collapsed",
        )

    pre_valid,  pre_err  = validate_image(pre_file)
    post_valid, post_err = validate_image(post_file)

    if pre_err:
        st.markdown(f'<div class="danger-box">Pre-disaster: {pre_err}</div>',
                    unsafe_allow_html=True)
    if post_err:
        st.markdown(f'<div class="danger-box">Post-disaster: {post_err}</div>',
                    unsafe_allow_html=True)

    # ── Resolve the active inputs (uploads win over the sample pair) ──────────
    post_arr = pre_arr = None
    post_name = ""
    sig = ""
    try:
        if post_file is not None and post_valid:
            post_arr  = load_image(post_file)
            post_name = post_file.name
            pre_arr   = load_image(pre_file) if (pre_file is not None and pre_valid) else None
            sig = file_signature(post_file) + "|" + file_signature(pre_file if pre_valid else None)
        elif use_sample and post_file is None:
            post_arr  = load_image(SAMPLE_POST)
            pre_arr   = load_image(SAMPLE_PRE)
            post_name = SAMPLE_POST.name
            sig = "sample"
    except Exception as exc:  # corrupt / unreadable image
        st.markdown(f'<div class="danger-box">Could not read image: {exc}</div>',
                    unsafe_allow_html=True)
        return

    if post_arr is None:
        if not post_err:
            _empty_state()
        return

    # ── Preview ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Image Preview</div>', unsafe_allow_html=True)
    prev_c1, prev_c2 = st.columns(2)
    with prev_c1:
        st.markdown("**Pre-Disaster**" if pre_arr is not None else "**Pre-Disaster** *(not provided)*")
        if pre_arr is not None:
            compat.image(pre_arr)
        else:
            st.markdown("""
            <div style="height:200px;display:flex;align-items:center;justify-content:center;
                        background:rgba(15,22,41,.6);border:1px dashed rgba(0,212,255,.2);
                        border-radius:8px;color:#546E7A">
                No pre-disaster image provided
            </div>
            """, unsafe_allow_html=True)
    with prev_c2:
        st.markdown("**Post-Disaster**")
        compat.image(post_arr)

    h, w = post_arr.shape[:2]
    st.markdown(f"""
    <div style="font-size:.78rem;color:#546E7A;margin-top:.3rem">
         Dimensions: {w} × {h} px &nbsp;|&nbsp;
         File: {post_name} &nbsp;|&nbsp;
        {' Using post image for pre channel' if pre_arr is None else ' Pre+Post pair loaded'}
    </div>
    """, unsafe_allow_html=True)

    # ── Run Analysis ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Run Analysis</div>', unsafe_allow_html=True)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = compat.button("Run Damage Assessment", disabled=not can_run)
    with col_info:
        if not can_run:
            st.markdown("""
            <div style="padding:.6rem;font-size:.84rem;color:#FFB800">
                 Model checkpoint required to run inference.
            </div>
            """, unsafe_allow_html=True)

    if run_clicked and can_run:
        with st.spinner("Running damage segmentation..."):
            try:
                predictor = get_predictor(config_key(config), config)
                if not predictor.ready:
                    st.error(f"Predictor unavailable: {predictor.status_message}")
                    return
                mask = predictor.predict(post_arr, pre_arr)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return
            st.session_state["result"] = {
                "result_id": uuid.uuid4().hex,
                "input_sig": sig,
                "mask":      mask,
                "post":      post_arr,
                "filename":  post_name,
                "device":    "cpu (demo)" if predictor.is_demo else str(predictor.device),
                "is_demo":   predictor.is_demo,
            }

    # ── Results (only if they belong to the images currently loaded) ──────────
    result = st.session_state.get("result")
    if result and result.get("input_sig") == sig:
        _render_results(config, result)
    elif result:
        st.info("Inputs changed — click **Run Damage Assessment** to analyse the new images.")


if __name__ == "__main__":
    main()
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
