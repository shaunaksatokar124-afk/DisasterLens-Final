"""
DisasterLens AI -- Analytics Page
Displays training curves, per-class metrics, confusion matrix,
and overall model performance. Shows placeholder state when no
training has been completed yet.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.ui import compat
from src.ui.styles import CSS, FOOTER_HTML
from src.ui.components import (
    check_dataset_ready,
    check_model_ready,
    load_config,
    load_evaluation_results,
    load_training_history,
    render_sidebar,
)

st.set_page_config(
    page_title="DisasterLens AI — Analytics",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#B0BEC5"),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ── Chart builders ────────────────────────────────────────────────────────────

def build_training_curves(history: dict) -> go.Figure:
    epochs      = list(range(1, len(history["train_loss"]) + 1))
    train_loss  = history["train_loss"]
    val_loss    = history.get("val_loss", [])
    val_iou     = history.get("val_mean_iou", [])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss Curves", "Validation Mean IoU"),
    )
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, mode="lines", name="Train Loss",
        line=dict(color="#00D4FF", width=2),
    ), row=1, col=1)
    if val_loss:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_loss, mode="lines", name="Val Loss",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
        ), row=1, col=1)
    if val_iou:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_iou, mode="lines", name="Val Mean IoU",
            line=dict(color="#00FF9F", width=2),
            fill="tozeroy", fillcolor="rgba(0,255,159,0.06)",
        ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True,
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", title_text="Epoch")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def build_class_metrics_chart(results: dict) -> go.Figure:
    classes = results.get("per_class", {})
    names   = list(classes.keys())
    metrics = ["precision", "recall", "f1", "iou"]
    colors  = ["#00D4FF", "#00FF9F", "#FFB800", "#FF6B6B"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        vals = [classes[n].get(metric) for n in names]  # None -> gap in chart
        fig.add_trace(go.Bar(name=metric.upper(), x=names, y=vals,
                              marker_color=color))

    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group", height=320,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_confusion_matrix(results: dict) -> go.Figure:
    cm     = results.get("confusion_matrix", [])
    labels = results.get("class_names",
                         ["Background", "No Damage", "Minor", "Major", "Destroyed"])

    if not cm:
        return None

    cm_arr = np.array(cm, dtype=float)
    # Row-normalise for better visualisation
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_arr / row_sums

    fig = px.imshow(
        cm_norm,
        x=labels, y=labels,
        color_continuous_scale=[[0, "#0A0E1A"], [0.5, "#004060"], [1, "#00D4FF"]],
        text_auto=".2f",
        labels=dict(x="Predicted", y="True", color="Normalised"),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        coloraxis_colorbar=dict(tickcolor="#B0BEC5",
                                title=dict(font=dict(color="#B0BEC5"))),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    config      = load_config()
    model_ready = check_model_ready(config)
    ds_ready    = check_dataset_ready(config)
    render_sidebar(config, model_ready, ds_ready)

    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="hero-title" style="font-size:1.8rem">Analytics</div>
        <div class="hero-subtitle">
            Training history, evaluation metrics, and model performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    train_hist = load_training_history(config)
    eval_res   = load_evaluation_results(config)
    has_data   = (train_hist is not None) or (eval_res is not None)

    if not has_data:
        st.markdown("""
        <div class="warning-box">
            <div style="font-weight:700;color:#FFB800;margin-bottom:.5rem">
                 No Training Data Available
            </div>
            <div style="font-size:.86rem;color:#B0BEC5 !important;line-height:1.65">
                Analytics become available after the model has been trained. Complete these steps:<br>
                1. Configure the xBD dataset path in <code style="color:#00D4FF">config.yaml</code><br>
                2. Run training with <code style="color:#00D4FF">notebooks/DisasterLens_AI_Training.ipynb</code><br>
                3. The training script saves <code style="color:#00D4FF">artifacts/training_history.json</code>
                   and <code style="color:#00D4FF">artifacts/evaluation_results.json</code> automatically
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Placeholder metric cards (clearly labelled as placeholders)
        st.markdown('<div class="section-header">Expected Metrics (after training)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-grid">
            <div class="metric-card">
                <span class="metric-value" style="color:#546E7A">—</span>
                <div class="metric-label">Mean IoU</div>
            </div>
            <div class="metric-card">
                <span class="metric-value" style="color:#546E7A">—</span>
                <div class="metric-label">Mean F1</div>
            </div>
            <div class="metric-card">
                <span class="metric-value" style="color:#546E7A">—</span>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <span class="metric-value" style="color:#546E7A">—</span>
                <div class="metric-label">Epochs Trained</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:1rem">
            <strong style="color:#00D4FF">What you will see here after training:</strong>
            <ul style="color:#B0BEC5 !important;margin:.5rem 0 0 1rem;font-size:.87rem;line-height:1.8">
                <li>Loss curves (train vs. validation)</li>
                <li>Validation mean IoU over epochs</li>
                <li>Per-class Precision, Recall, F1, IoU</li>
                <li>Normalised confusion matrix</li>
                <li>Class coverage distribution pie chart</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    if eval_res:
        st.markdown('<div class="section-header">Evaluation Summary</div>',
                    unsafe_allow_html=True)
        mean_iou = eval_res.get("mean_iou", None)
        mean_f1  = eval_res.get("mean_f1", None)
        acc      = eval_res.get("accuracy", None)
        epochs   = None
        if train_hist:
            epochs = train_hist.get("epochs_trained") or len(train_hist.get("train_loss", []))

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <span class="metric-value">
                    {f"{mean_iou:.3f}" if mean_iou is not None else "—"}
                </span>
                <div class="metric-label">Mean IoU</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">
                    {f"{mean_f1:.3f}" if mean_f1 is not None else "—"}
                </span>
                <div class="metric-label">Mean F1</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">
                    {f"{acc:.3f}" if acc is not None else "—"}
                </span>
                <div class="metric-label">Pixel Accuracy</div>
            </div>
            <div class="metric-card">
                <span class="metric-value">
                    {epochs if epochs is not None else "—"}
                </span>
                <div class="metric-label">Epochs Trained</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Training curves ───────────────────────────────────────────────────────
    if train_hist:
        st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
        compat.plotly_chart(build_training_curves(train_hist))

    # ── Per-class metrics ─────────────────────────────────────────────────────
    if eval_res and eval_res.get("per_class"):
        st.markdown('<div class="section-header">Per-Class Metrics</div>', unsafe_allow_html=True)
        compat.plotly_chart(build_class_metrics_chart(eval_res))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if eval_res and eval_res.get("confusion_matrix"):
        st.markdown('<div class="section-header">Confusion Matrix (Row-Normalised)</div>',
                    unsafe_allow_html=True)
        cm_fig = build_confusion_matrix(eval_res)
        if cm_fig:
            compat.plotly_chart(cm_fig)


if __name__ == "__main__":
    main()
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
