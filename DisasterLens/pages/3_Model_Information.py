"""
DisasterLens AI -- Model Information Page
Architecture details, dataset overview, training methodology,
limitations, and real evaluation results if available.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.ui.styles import CSS, FOOTER_HTML
from src.ui.components import (
    check_dataset_ready,
    check_model_ready,
    load_config,
    load_evaluation_results,
    render_sidebar,
)

st.set_page_config(
    page_title="DisasterLens AI — Model Information",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    config      = load_config()
    model_ready = check_model_ready(config)
    ds_ready    = check_dataset_ready(config)
    render_sidebar(config, model_ready, ds_ready)

    eval_res = load_evaluation_results(config)
    mcfg     = config.get("model", {})
    tcfg     = config.get("training", {})
    dcfg     = config.get("dataset", {})
    classes  = config.get("classes", [])

    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="hero-title" style="font-size:1.8rem">Model Information</div>
        <div class="hero-subtitle">
            Architecture, dataset, training methodology, evaluation results, and limitations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_arch, tab_data, tab_train, tab_eval, tab_limits = st.tabs([
        " Architecture",
        " Dataset",
        " Training",
        " Evaluation Results",
        " Limitations",
    ])

    # ── Architecture ──────────────────────────────────────────────────────────
    with tab_arch:
        st.markdown('<div class="section-header">Model Architecture</div>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 3], gap="large")
        with col_a:
            st.markdown(f"""
            <div class="status-card">
                <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem">Architecture</div>
                <div style="font-size:1.1rem;font-weight:700;color:#00D4FF">
                    U-Net + {mcfg.get("encoder","ResNet-34").upper()}
                </div>
            </div>
            <div class="status-card">
                <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem">Input Channels</div>
                <div style="font-size:1.05rem;font-weight:600;color:#E8EAF6">
                    {mcfg.get("in_channels",6)} (3-ch pre-disaster + 3-ch post-disaster)
                </div>
            </div>
            <div class="status-card">
                <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem">Output Classes</div>
                <div style="font-size:1.05rem;font-weight:600;color:#E8EAF6">
                    {mcfg.get("num_classes",5)} (Background + 4 damage levels)
                </div>
            </div>
            <div class="status-card">
                <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem">Training Resolution</div>
                <div style="font-size:1.05rem;font-weight:600;color:#E8EAF6">
                    {mcfg.get("image_size",512)} × {mcfg.get("image_size",512)} px
                </div>
            </div>
            <div class="status-card">
                <div style="font-size:.7rem;color:#546E7A;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem">Encoder Weights</div>
                <div style="font-size:1.05rem;font-weight:600;color:#E8EAF6">
                    ImageNet pretrained
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="info-box">
                <strong style="color:#00D4FF">Architecture Design Rationale</strong>
                <p style="color:#B0BEC5 !important;font-size:.87rem;line-height:1.75;margin:.5rem 0 0 0">
                <strong style="color:#E8EAF6">U-Net</strong> is the gold standard for semantic
                segmentation tasks with limited data. Its encoder-decoder structure with skip
                connections preserves fine spatial details lost during downsampling, making it
                ideal for precise building footprint localisation.
                </p>
                <p style="color:#B0BEC5 !important;font-size:.87rem;line-height:1.75;margin:.75rem 0 0 0">
                <strong style="color:#E8EAF6">ResNet-34</strong> is used as the encoder backbone,
                pretrained on ImageNet. This provides rich low-level feature representations from
                the start and is computationally feasible for training on a single GPU.
                </p>
                <p style="color:#B0BEC5 !important;font-size:.87rem;line-height:1.75;margin:.75rem 0 0 0">
                <strong style="color:#E8EAF6">6-channel input</strong> (pre + post stacked) gives the
                model change information — crucial for distinguishing pre-existing damage from
                disaster-induced damage. The pretrained first-layer weights are averaged across the
                additional channels (a well-established approach).
                </p>
                <p style="color:#B0BEC5 !important;font-size:.87rem;line-height:1.75;margin:.75rem 0 0 0">
                <strong style="color:#E8EAF6">Squeeze-and-Excitation (scSE) attention</strong>
                is applied in the decoder to improve feature recalibration across both channels and
                spatial dimensions.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-family:monospace;background:rgba(0,0,0,.3);
                        border:1px solid rgba(0,212,255,.1);border-radius:8px;
                        padding:1rem;font-size:.78rem;color:#78909C;margin-top:1rem">
                <span style="color:#00D4FF">Input</span>: [B, 6, 512, 512]<br>
                &nbsp;&nbsp;&nbsp; Encoder (ResNet-34)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Block 1: [B, 64,  256, 256]<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Block 2: [B, 128, 128, 128]<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Block 3: [B, 256,  64,  64]<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ Block 4: [B, 512,  32,  32]<br>
                &nbsp;&nbsp;&nbsp; Decoder (U-Net + scSE attention)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ + skip connections from encoder<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ Upsample × 4<br>
                <span style="color:#00FF9F">Output</span>: [B, 5, 512, 512] (logits)
            </div>
            """, unsafe_allow_html=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="section-header">Dataset: xBD (xView2)</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown(f"""
            <div class="metric-grid" style="grid-template-columns:repeat(2,1fr)">
                <div class="metric-card">
                    <span class="metric-value">11,034</span>
                    <div class="metric-label">Pre/Post Image Pairs</div>
                </div>
                <div class="metric-card">
                    <span class="metric-value">850K+</span>
                    <div class="metric-label">Building Polygons</div>
                </div>
                <div class="metric-card">
                    <span class="metric-value">19</span>
                    <div class="metric-label">Disaster Events</div>
                </div>
                <div class="metric-card">
                    <span class="metric-value">&le;0.8 m</span>
                    <div class="metric-label">Ground Sample Distance</div>
                </div>
            </div>
            <div class="info-box" style="margin-top:1rem">
                <div style="font-size:.83rem;color:#B0BEC5 !important;line-height:1.7">
                    <strong style="color:#E8EAF6">Sensor:</strong>
                        {dcfg.get("sensor","WorldView-2/3")}<br>
                    <strong style="color:#E8EAF6">License:</strong>
                        {dcfg.get("license","CC BY-NC-SA 4.0")}<br>
                    <strong style="color:#E8EAF6">Annotation:</strong>
                        {dcfg.get("annotation_format","WKT Polygon")}<br>
                    <strong style="color:#E8EAF6">Image size:</strong>
                        {dcfg.get("image_resolution","1024×1024")} PNG<br>
                    <strong style="color:#E8EAF6">Source:</strong>
                        <a href="https://xview2.org" target="_blank"
                           style="color:#00D4FF">xview2.org</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="info-box">
                <strong style="color:#00D4FF">Annotation Format</strong>
                <p style="color:#B0BEC5 !important;font-size:.85rem;line-height:1.75;margin:.4rem 0 0">
                Each image is paired with a JSON sidecar file. Building polygons are stored
                as <strong>WKT (Well-Known Text)</strong> strings in pixel coordinates:
                </p>
            </div>
            <div style="font-family:monospace;background:rgba(0,0,0,.35);
                        border:1px solid rgba(0,212,255,.1);border-radius:8px;
                        padding:.8rem;font-size:.76rem;color:#78909C;margin-top:.5rem">
                {<br>
                &nbsp;&nbsp;<span style="color:#00D4FF">"features"</span>: {<br>
                &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#00D4FF">"xy"</span>: [{<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#00FF9F">"wkt"</span>:
                    <span style="color:#FFB800">"POLYGON ((312 400, ...))"</span>,<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#00FF9F">"properties"</span>: {<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    <span style="color:#00D4FF">"subtype"</span>:
                    <span style="color:#FF6B6B">"destroyed"</span>,<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    <span style="color:#00D4FF">"feature_type"</span>:
                    <span style="color:#FFB800">"building"</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;}]<br>
                &nbsp;&nbsp;}<br>
                }
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem">Damage Classes</div>',
                    unsafe_allow_html=True)
        class_colors = {
            "Background": "#2D2D2D", "No Damage": "#00C851",
            "Minor Damage": "#FFD600", "Major Damage": "#FF6D00", "Destroyed": "#FF1744",
        }
        class_descs = {
            "Background":   "Non-building pixels; excluded from damage statistics",
            "No Damage":    "Building intact; no visible structural change from pre-disaster",
            "Minor Damage": "Partial roof or wall damage; minor debris; building still functional",
            "Major Damage": "Severe structural compromise; partial collapse; requires inspection",
            "Destroyed":    "Total collapse, flattened, or complete debris field",
        }
        for cname, ccolor in class_colors.items():
            desc = class_descs.get(cname, "")
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:.75rem;
                        padding:.6rem 0;border-bottom:1px solid rgba(255,255,255,.05)">
                <div style="width:14px;height:14px;border-radius:3px;
                            background:{ccolor};flex-shrink:0;margin-top:.1rem"></div>
                <div>
                    <span style="font-weight:600;color:#E8EAF6;font-size:.9rem">{cname}</span><br>
                    <span style="color:#546E7A;font-size:.81rem">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Training ──────────────────────────────────────────────────────────────
    with tab_train:
        st.markdown('<div class="section-header">Training Configuration</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")

        with c1:
            rows = [
                ("Optimizer",       "AdamW"),
                ("Learning Rate",   f"{tcfg.get('learning_rate', 0.0001)}"),
                ("Weight Decay",    f"{tcfg.get('weight_decay', 0.0001)}"),
                ("Batch Size",      f"{tcfg.get('batch_size', 8)}"),
                ("Epochs",          f"{tcfg.get('epochs', 50)}"),
                ("Loss",            "FocalLoss (γ=2) + DiceLoss (50/50)"),
                ("Scheduler",       f"{tcfg.get('scheduler', 'CosineAnnealingLR')}"),
                ("Augmentation",    "Albumentations (flip, rotate, colour jitter)"),
            ]
            tbl = "".join(
                f"""<tr style="border-bottom:1px solid rgba(255,255,255,.05)">
                    <td style="padding:.5rem .75rem;color:#546E7A;font-size:.84rem">{k}</td>
                    <td style="padding:.5rem .75rem;color:#E8EAF6;font-size:.84rem;
                               font-weight:500">{v}</td>
                </tr>"""
                for k, v in rows
            )
            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse">
                <thead>
                    <tr style="border-bottom:1px solid rgba(0,212,255,.2)">
                        <th style="padding:.5rem .75rem;text-align:left;color:#546E7A;
                                   font-size:.72rem;text-transform:uppercase;
                                   letter-spacing:.06em">Parameter</th>
                        <th style="padding:.5rem .75rem;text-align:left;color:#546E7A;
                                   font-size:.72rem;text-transform:uppercase;
                                   letter-spacing:.06em">Value</th>
                    </tr>
                </thead>
                <tbody>{tbl}</tbody>
            </table>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="info-box">
                <strong style="color:#00D4FF">Data Splitting Strategy</strong>
                <p style="color:#B0BEC5 !important;font-size:.85rem;line-height:1.75;margin:.4rem 0 0">
                Splits are performed at the <strong>disaster-event level</strong>, not the image
                level. All images from the same hurricane-florence or mexico-earthquake event are
                assigned to the same split. This prevents geographic data leakage between sets.
                </p>
                <ul style="color:#B0BEC5 !important;font-size:.84rem;line-height:1.8;margin:.5rem 0 0 1rem">
                    <li>Train: 70% of disaster events</li>
                    <li>Validation: 15% of disaster events</li>
                    <li>Test: 15% of disaster events (held out)</li>
                </ul>
            </div>
            <div class="info-box" style="margin-top:.75rem">
                <strong style="color:#00D4FF">Class Imbalance Handling</strong>
                <p style="color:#B0BEC5 !important;font-size:.85rem;line-height:1.75;margin:.4rem 0 0">
                The xBD dataset is heavily imbalanced — background pixels dominate, and
                "destroyed" buildings are rare. Mitigations applied:
                </p>
                <ul style="color:#B0BEC5 !important;font-size:.84rem;line-height:1.8;margin:.5rem 0 0 1rem">
                    <li>Focal Loss (γ=2) — down-weights easy background pixels</li>
                    <li>Dice Loss — optimises overlap directly</li>
                    <li>ignore_index=255 — un-classified pixels excluded from loss</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ── Evaluation Results ────────────────────────────────────────────────────
    with tab_eval:
        st.markdown('<div class="section-header">Evaluation Results</div>',
                    unsafe_allow_html=True)
        if eval_res is None:
            st.markdown("""
            <div class="warning-box">
                <div style="font-weight:700;color:#FFB800;margin-bottom:.4rem">
                     No Evaluation Results Yet
                </div>
                <div style="font-size:.86rem;color:#B0BEC5 !important">
                    Evaluation results will appear here after training and running
                    <code style="color:#00D4FF">training/evaluate.py</code> on the held-out test set.
                    Results are never fabricated or estimated.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.json(eval_res)

    # ── Limitations ───────────────────────────────────────────────────────────
    with tab_limits:
        st.markdown('<div class="section-header">Known Limitations</div>',
                    unsafe_allow_html=True)
        limits = [
            ("Pixel-level only",
             "DisasterLens AI produces pixel-level segmentation masks. It does not independently "
             "detect individual buildings or classify damage at the building level."),
            ("No geographic area",
             "Pixel counts cannot be converted to geographic area (m²) without georeferenced "
             "imagery and known ground sample distance (GSD)."),
            ("Domain distribution",
             "The xBD dataset covers 19 disaster events across six hazard types. Performance may degrade on "
             "imagery from sensors, resolutions, or geographies not represented in training."),
            ("No structural assessment",
             "Predictions are based solely on visual appearance in satellite imagery. Structural "
             "engineering assessment requires ground-based inspection."),
            ("Single-date post-disaster",
             "The model expects post-disaster imagery. Multi-temporal analysis or change detection "
             "beyond two dates is not currently supported."),
            ("Human verification required",
             "Model outputs must be verified by qualified human analysts before being used in "
             "operational emergency response, insurance assessment, or policy decisions."),
        ]
        for title, desc in limits:
            st.markdown(f"""
            <div class="status-card" style="margin-bottom:.6rem">
                <div style="font-weight:600;color:#FFB800;font-size:.9rem;
                            margin-bottom:.3rem"> {title}</div>
                <div style="font-size:.84rem;color:#B0BEC5 !important;line-height:1.65">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box" style="margin-top:1.5rem">
            <strong>Official Disclaimer:</strong> DisasterLens AI is a research and educational
            tool. Predictions generated by this system are based on a deep learning model and are
            subject to errors, biases, and limitations inherent to any automated image analysis
            system. These predictions must not be used as the sole basis for emergency response
            decisions, structural safety assessments, insurance claims, or any official capacity
            without human expert verification.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
