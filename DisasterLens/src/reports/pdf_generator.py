"""
DisasterLens AI -- PDF Assessment Report Generator
====================================================
Generates a professional PDF report for a completed damage analysis session.

The report is assembled entirely in memory using ReportLab PLATYPUS and
returned as ``bytes``, making it directly usable with ``st.download_button``.

Report structure
----------------
1. Header bar        -- project name, analysis ID, timestamp
2. Metadata block    -- filename, device, damage summary badge
3. Image panels      -- original satellite image + damage overlay (side-by-side)
4. Statistics table  -- per-class pixel count and coverage %
5. Model info table  -- architecture, encoder, checkpoint
6. Disclaimer        -- mandatory human-verification notice
7. Footer            -- page number + generation timestamp

Public API
----------
generate_report_pdf(entry, config) -> bytes
    Build and return the PDF for a single history entry dict.

image_path_to_rl_image(path, max_width, max_height) -> RLImage | None
    Helper: load an image file into a ReportLab Image flowable.

Dependencies
------------
- reportlab >= 4.1  (in requirements.txt)
- Pillow             (in requirements.txt)
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as _xml_escape

from src.paths import resolve_path

logger = logging.getLogger(__name__)

# ReportLab imports -- guarded so the module can be imported even if reportlab
# is not yet installed (the app shows a clear error instead of crashing).
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


def _esc(value: Any) -> str:
    """Escape text for ReportLab Paragraph markup (handles &, <, >)."""
    return _xml_escape(str(value))


# ---------------------------------------------------------------------------
# Colour palette (matches the Streamlit dark theme)
# ---------------------------------------------------------------------------

if _REPORTLAB_AVAILABLE:
    _NAVY       = colors.HexColor("#0A1628")
    _CYAN       = colors.HexColor("#00D4FF")
    _CYAN_DARK  = colors.HexColor("#0098BA")
    _GREEN      = colors.HexColor("#00C851")
    _YELLOW     = colors.HexColor("#FFD600")
    _ORANGE     = colors.HexColor("#FF6D00")
    _RED        = colors.HexColor("#FF1744")
    _GREY_DARK  = colors.HexColor("#263238")
    _GREY_MID   = colors.HexColor("#546E7A")
    _GREY_LIGHT = colors.HexColor("#B0BEC5")
    _WHITE      = colors.white
    _BLACK      = colors.HexColor("#1A1A2E")

    # Damage category colours (for table cell backgrounds)
    _CLASS_COLORS = {
        "Background":   colors.HexColor("#2D2D2D"),
        "No Damage":    _GREEN,
        "Minor Damage": _YELLOW,
        "Major Damage": _ORANGE,
        "Destroyed":    _RED,
    }


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def image_path_to_rl_image(
    path: str | Path,
    max_width: float,
    max_height: float,
) -> Optional[Any]:
    """Load an image file and return a ReportLab Image flowable.

    Preserves the image's aspect ratio within the given bounding box.
    Returns ``None`` if the file does not exist or cannot be read.

    Args:
        path:       Filesystem path to a PNG/JPG image.
        max_width:  Maximum width in points.
        max_height: Maximum height in points.

    Returns:
        ``RLImage`` flowable, or ``None`` on failure.
    """
    if not _REPORTLAB_AVAILABLE:
        return None

    try:
        from PIL import Image as PILImage
        p = resolve_path(path)
        if p is None or not p.exists():
            return None

        with PILImage.open(p) as img:
            w, h = img.size
            # Fit within bounding box while preserving aspect ratio
            scale = min(max_width / w, max_height / h)
            rw, rh = w * scale, h * scale

            # Convert to RGB PNG bytes for ReportLab
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            buf.seek(0)

        return RLImage(buf, width=rw, height=rh)
    except Exception as exc:
        logger.warning("Could not load image for PDF: %s — %s", path, exc)
        return None


def _image_bytes_to_rl_image(
    img_bytes: bytes,
    max_width: float,
    max_height: float,
) -> Optional[Any]:
    """Like ``image_path_to_rl_image`` but accepts raw PNG bytes."""
    if not _REPORTLAB_AVAILABLE:
        return None
    try:
        from PIL import Image as PILImage
        buf_in = io.BytesIO(img_bytes)
        with PILImage.open(buf_in) as img:
            w, h = img.size
            scale = min(max_width / w, max_height / h)
            rw, rh = w * scale, h * scale
            buf_out = io.BytesIO()
            img.convert("RGB").save(buf_out, format="PNG")
            buf_out.seek(0)
        return RLImage(buf_out, width=rw, height=rh)
    except Exception as exc:
        logger.warning("Could not process image bytes for PDF: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    """Build and return a dict of named ParagraphStyle objects."""
    base = getSampleStyleSheet()

    styles = {
        "report_title": ParagraphStyle(
            "report_title",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=_WHITE,
            alignment=TA_LEFT,
        ),
        "report_subtitle": ParagraphStyle(
            "report_subtitle",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=_CYAN,
            alignment=TA_LEFT,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=_BLACK,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body_text": ParagraphStyle(
            "body_text",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=_GREY_DARK,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=_WHITE,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=9,
            textColor=_BLACK,
            alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=12,
            textColor=_GREY_DARK,
            borderColor=_YELLOW,
            borderWidth=1,
            borderPadding=6,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=7,
            textColor=_GREY_MID,
            alignment=TA_CENTER,
        ),
        "meta_label": ParagraphStyle(
            "meta_label",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=_GREY_MID,
            spaceBefore=2,
        ),
        "meta_value": ParagraphStyle(
            "meta_value",
            fontName="Helvetica",
            fontSize=9,
            textColor=_BLACK,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _header_table(entry: dict, styles: dict, page_width: float) -> Table:
    """Build the dark navy header bar with logo, title and analysis ID."""
    analysis_id = entry.get("id", "—")
    timestamp   = entry.get("timestamp", "—")

    title_para = Paragraph("DisasterLens AI", styles["report_title"])
    sub_para   = Paragraph(
        "AI-Powered Disaster Damage Assessment Report",
        styles["report_subtitle"],
    )
    id_style = ParagraphStyle(
        "id_right",
        fontName="Helvetica",
        fontSize=9,
        textColor=_GREY_LIGHT,
        alignment=TA_RIGHT,
    )
    id_para = Paragraph(
        f"<b>Analysis ID:</b> {_esc(analysis_id)}<br/>"
        f"<b>Analyzed:</b> {_esc(timestamp)}",
        id_style,
    )

    col_w = page_width - 2 * cm
    tbl = Table(
        [[title_para, id_para], [sub_para, ""]],
        colWidths=[col_w * 0.65, col_w * 0.35],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), _NAVY),
        ("TOPPADDING",  (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN",          (0, 1), (0, 1)),
    ]))
    return tbl


def _metadata_table(entry: dict, styles: dict, page_width: float) -> Table:
    """Build a 2-column metadata grid (filename, device, date)."""
    col_w = (page_width - 2 * cm) / 2

    filename  = entry.get("filename", "—")
    device    = entry.get("model_device", "—")
    timestamp = entry.get("timestamp", "—")
    analysis_mode = (
        "DEMO heuristic (no trained checkpoint)"
        if entry.get("mode") == "demo" else "Trained U-Net checkpoint"
    )

    rows = [
        [
            Paragraph("Source File",           styles["meta_label"]),
            Paragraph("Model Device",          styles["meta_label"]),
        ],
        [
            Paragraph(_esc(filename),          styles["meta_value"]),
            Paragraph(_esc(device),            styles["meta_value"]),
        ],
        [
            Paragraph("Analysis Timestamp",    styles["meta_label"]),
            Paragraph("Prediction Source",     styles["meta_label"]),
        ],
        [
            Paragraph(_esc(timestamp),         styles["meta_value"]),
            Paragraph(_esc(analysis_mode),     styles["meta_value"]),
        ],
    ]
    tbl = Table(rows, colWidths=[col_w, col_w])
    tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


def _image_panel(
    entry: dict,
    styles: dict,
    page_width: float,
) -> Optional[Table]:
    """Build the side-by-side image panel (original + overlay)."""
    available = page_width - 2 * cm
    img_w = (available - 0.5 * cm) / 2
    img_h = img_w * 0.75  # 4:3 aspect

    post_img_path    = entry.get("post_img")
    overlay_img_path = entry.get("overlay_img")

    post_rl    = image_path_to_rl_image(post_img_path,    img_w, img_h) if post_img_path    else None
    overlay_rl = image_path_to_rl_image(overlay_img_path, img_w, img_h) if overlay_img_path else None

    missing_style = ParagraphStyle(
        "missing",
        fontName="Helvetica",
        fontSize=8,
        textColor=_GREY_MID,
        alignment=TA_CENTER,
    )

    left_cell  = post_rl    or Paragraph("[Image file not found]", missing_style)
    right_cell = overlay_rl or Paragraph("[Overlay file not found]", missing_style)

    label_style = ParagraphStyle(
        "img_label",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=_GREY_MID,
        alignment=TA_CENTER,
    )

    tbl = Table(
        [
            [Paragraph("Post-Disaster Imagery", label_style),
             Paragraph("Damage Prediction Overlay", label_style)],
            [left_cell, right_cell],
        ],
        colWidths=[img_w, img_w],
        rowHeights=[None, img_h],
    )
    tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.5, _GREY_LIGHT),
    ]))
    return tbl


def _stats_table(entry: dict, styles: dict, page_width: float) -> Optional[Table]:
    """Build the per-class damage statistics table."""
    stats: Dict[str, dict] = entry.get("stats", {})
    if not stats:
        return None

    col_w = (page_width - 2 * cm) / 3

    header_row = [
        Paragraph("Damage Class",   styles["table_header"]),
        Paragraph("Pixel Count",    styles["table_header"]),
        Paragraph("Coverage (%)",   styles["table_header"]),
    ]
    rows: List[list] = [header_row]

    for class_name, s in stats.items():
        count      = s.get("count", 0)
        percentage = s.get("percentage", 0.0)
        rows.append([
            Paragraph(_esc(class_name),   styles["table_cell"]),
            Paragraph(f"{count:,}",        styles["table_cell"]),
            Paragraph(f"{percentage:.2f}", styles["table_cell"]),
        ])

    tbl = Table(rows, colWidths=[col_w, col_w, col_w])

    # Build per-row background colours
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),          # header row
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F5F5")]),
    ]

    # Colour-dot in first column for each data row
    for row_i, class_name in enumerate(stats.keys(), start=1):
        cls_color = _CLASS_COLORS.get(class_name)
        if cls_color:
            style_cmds.append(("BACKGROUND", (0, row_i), (0, row_i), cls_color))
            # Make class name white text on coloured background
            rows[row_i][0] = Paragraph(
                _esc(class_name),
                ParagraphStyle(
                    f"cell_{row_i}",
                    fontName="Helvetica-Bold",
                    fontSize=9,
                    textColor=_BLACK if class_name in ("No Damage", "Minor Damage") else _WHITE,
                    alignment=TA_CENTER,
                ),
            )

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _model_info_table(config: dict, styles: dict, page_width: float) -> Table:
    """Build the model architecture info table."""
    mcfg = config.get("model", {})
    dcfg = config.get("dataset", {})

    col_label = (page_width - 2 * cm) * 0.35
    col_value = (page_width - 2 * cm) * 0.65

    lbl = ParagraphStyle("mi_lbl", fontName="Helvetica-Bold", fontSize=9,
                         textColor=_GREY_DARK)
    val = ParagraphStyle("mi_val", fontName="Helvetica",      fontSize=9,
                         textColor=_BLACK)

    rows = [
        [Paragraph("Architecture", lbl), Paragraph("U-Net with SCSE Attention Decoder", val)],
        [Paragraph("Encoder",      lbl), Paragraph(_esc(mcfg.get("encoder", "resnet34")), val)],
        [Paragraph("Input",        lbl), Paragraph(f"{mcfg.get('in_channels', 6)}-channel (pre + post imagery)", val)],
        [Paragraph("Classes",      lbl), Paragraph(f"{mcfg.get('num_classes', 5)} (Background + 4 damage levels)", val)],
        [Paragraph("Image Size",   lbl), Paragraph(f"{mcfg.get('image_size', 512)} × {mcfg.get('image_size', 512)} px", val)],
        [Paragraph("Dataset",      lbl), Paragraph(_esc(dcfg.get("name", "xBD (xView2)")), val)],
        [Paragraph("Framework",    lbl), Paragraph("PyTorch + segmentation-models-pytorch", val)],
    ]

    tbl = Table(rows, colWidths=[col_label, col_value])
    tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.white, colors.HexColor("#F5F5F5")]),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Page footer callback
# ---------------------------------------------------------------------------

class _FooterCanvas:
    """Mixin that adds a footer to each page via a canvas callback."""

    def __init__(self, generation_time: str):
        self._gen_time = generation_time

    def __call__(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(_GREY_MID)
        footer_text = (
            f"DisasterLens AI  |  Page {doc.page}  |  "
            f"Generated {self._gen_time}  |  "
            "Predictions require human expert verification."
        )
        canvas.drawCentredString(doc.pagesize[0] / 2, 0.6 * cm, footer_text)
        # Footer rule
        canvas.setStrokeColor(_CYAN_DARK)
        canvas.setLineWidth(0.5)
        canvas.line(1 * cm, 1.1 * cm, doc.pagesize[0] - 1 * cm, 1.1 * cm)
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_report_pdf(entry: dict, config: dict) -> bytes:
    """Generate a PDF assessment report from a history entry.

    Args:
        entry:  Analysis history dict as stored in ``artifacts/analysis_history.json``.
                Expected keys: ``id``, ``timestamp``, ``filename``, ``post_img``,
                ``overlay_img``, ``stats``, ``model_device``.
        config: Loaded ``config.yaml`` dict.

    Returns:
        PDF file contents as ``bytes``, suitable for ``st.download_button``.

    Raises:
        ImportError: If reportlab is not installed.
        RuntimeError: If PDF generation fails for any other reason.
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )

    buf = io.BytesIO()
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1 * cm,
        rightMargin=1 * cm,
        topMargin=0.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"DisasterLens AI - Assessment Report {entry.get('id', '')}",
        author="DisasterLens AI",
        subject="Disaster Damage Assessment Report",
    )

    page_w = A4[0]
    styles = _build_styles()
    footer_cb = _FooterCanvas(gen_time)
    story = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    story.append(_header_table(entry, styles, page_w))
    story.append(Spacer(1, 0.4 * cm))

    # ── 2. Metadata ───────────────────────────────────────────────────────────
    story.append(Paragraph("Analysis Metadata", styles["section_heading"]))
    story.append(_metadata_table(entry, styles, page_w))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_CYAN_DARK))
    story.append(Spacer(1, 0.3 * cm))

    # ── 3. Images ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Satellite Imagery & Damage Prediction", styles["section_heading"]))
    img_panel = _image_panel(entry, styles, page_w)
    if img_panel:
        story.append(img_panel)
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_CYAN_DARK))
    story.append(Spacer(1, 0.3 * cm))

    # ── 4. Statistics table ───────────────────────────────────────────────────
    story.append(Paragraph("Damage Category Statistics", styles["section_heading"]))
    stats_tbl = _stats_table(entry, styles, page_w)
    if stats_tbl:
        story.append(stats_tbl)
    else:
        story.append(Paragraph("No statistics available.", styles["body_text"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_CYAN_DARK))
    story.append(Spacer(1, 0.3 * cm))

    # ── 5. Model info ─────────────────────────────────────────────────────────
    story.append(Paragraph("Model Information", styles["section_heading"]))
    story.append(_model_info_table(config, styles, page_w))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_CYAN_DARK))
    story.append(Spacer(1, 0.3 * cm))

    # ── 6. Disclaimer ─────────────────────────────────────────────────────────
    story.append(Paragraph("Important Limitations & Disclaimer", styles["section_heading"]))
    disclaimer_text = (
        "<b>Important: this report is generated by an experimental deep learning model "
        "and is intended for research and educational purposes only.</b> "
        "Damage predictions are based on pixel-level segmentation of satellite "
        "imagery and have not been validated by domain experts for this specific "
        "analysis. Predictions may contain errors due to image quality, "
        "occlusion, model limitations, or scenes outside the training distribution. "
        "<b>These results must not be used as the sole basis for emergency response, "
        "insurance claims, or any operational decisions without independent "
        "human expert verification.</b> The xBD dataset and this model focus on "
        "building damage; other disaster effects (e.g. road damage, flooding) "
        "are outside the scope of this assessment."
    )
    story.append(Paragraph(disclaimer_text, styles["disclaimer"]))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    try:
        doc.build(
            story,
            onFirstPage=footer_cb,
            onLaterPages=footer_cb,
        )
    except Exception as exc:
        raise RuntimeError(f"PDF generation failed: {exc}") from exc

    buf.seek(0)
    return buf.read()
