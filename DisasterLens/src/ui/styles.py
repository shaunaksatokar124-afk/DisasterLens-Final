"""
DisasterLens AI — Shared CSS Styles
Injected into every Streamlit page via st.markdown(CSS, unsafe_allow_html=True).
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ─ Hero ─ */
.hero-banner {
    background: linear-gradient(135deg, #0F1629 0%, #0D1B2A 50%, #0A1628 100%);
    border: 1px solid rgba(0,212,255,0.25); border-radius: 16px;
    padding: 2.5rem 2.5rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero-banner::before {
    content:''; position:absolute; top:-50%; right:-20%;
    width:400px; height:400px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 0 0 0.4rem 0;
    background: linear-gradient(135deg, #00D4FF 0%, #0080FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-subtitle { font-size: 1rem; color: #78909C !important; margin: 0; font-weight: 400; }

/* ─ Status Cards ─ */
.status-card {
    background: linear-gradient(135deg, #0F1629 0%, #141E3A 100%);
    border: 1px solid rgba(0,212,255,0.15); border-radius: 12px;
    padding: 1.1rem 1.4rem; margin-bottom: 0.75rem;
    transition: border-color 0.25s ease;
}
.status-card:hover { border-color: rgba(0,212,255,0.35); }

/* ─ Status Badges ─ */
.badge-ready {
    display:inline-flex; align-items:center; gap:0.35rem;
    background: rgba(0,255,159,0.10); color:#00FF9F;
    border:1px solid rgba(0,255,159,0.30); border-radius:20px;
    padding:0.18rem 0.7rem; font-size:0.78rem; font-weight:600;
}
.badge-notready {
    display:inline-flex; align-items:center; gap:0.35rem;
    background: rgba(255,68,68,0.10); color:#FF6B6B;
    border:1px solid rgba(255,68,68,0.30); border-radius:20px;
    padding:0.18rem 0.7rem; font-size:0.78rem; font-weight:600;
}
.badge-warning {
    display:inline-flex; align-items:center; gap:0.35rem;
    background: rgba(255,184,0,0.10); color:#FFB800;
    border:1px solid rgba(255,184,0,0.30); border-radius:20px;
    padding:0.18rem 0.7rem; font-size:0.78rem; font-weight:600;
}

/* ─ Metric Grid ─ */
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.5rem 0; }
.metric-card {
    background: linear-gradient(135deg, #0F1629 0%, #141E3A 100%);
    border:1px solid rgba(0,212,255,0.12); border-radius:12px;
    padding:1.2rem; text-align:center; transition:all 0.25s ease;
}
.metric-card:hover { border-color:rgba(0,212,255,0.35); transform:translateY(-2px); }
.metric-value { font-size:1.9rem; font-weight:700; color:#00D4FF; display:block; line-height:1; }
.metric-label { font-size:0.72rem; color:#546E7A !important; margin-top:0.35rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }

/* ─ Section Headers ─ */
.section-header {
    font-size:0.72rem; font-weight:700; color:#00D4FF !important; text-transform:uppercase;
    letter-spacing:0.12em; margin:2rem 0 1rem 0;
    display:flex; align-items:center; gap:0.5rem;
}
.section-header::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,rgba(0,212,255,0.3) 0%,transparent 100%); }

/* ─ Info / Warning Boxes ─ */
.info-box {
    background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.20);
    border-left:3px solid #00D4FF; border-radius:8px; padding:0.9rem 1.2rem; margin:0.75rem 0;
}
.warning-box {
    background:rgba(255,184,0,0.05); border:1px solid rgba(255,184,0,0.20);
    border-left:3px solid #FFB800; border-radius:8px; padding:0.9rem 1.2rem; margin:0.75rem 0;
}
.danger-box {
    background:rgba(255,68,68,0.05); border:1px solid rgba(255,68,68,0.20);
    border-left:3px solid #FF4444; border-radius:8px; padding:0.9rem 1.2rem; margin:0.75rem 0;
}

/* ─ Tag Pills ─ */
.tag-pill {
    display:inline-block; background:rgba(0,212,255,0.10); color:#00D4FF;
    border:1px solid rgba(0,212,255,0.20); border-radius:20px;
    padding:0.15rem 0.6rem; font-size:0.72rem; font-weight:500; margin:0.15rem;
}

/* ─ History Row ─ */
.history-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:0.7rem 1rem; background:rgba(15,22,41,0.6);
    border:1px solid rgba(0,212,255,0.08); border-radius:8px;
    margin-bottom:0.45rem; transition:border-color 0.2s ease;
}
.history-row:hover { border-color:rgba(0,212,255,0.22); }

/* ─ Disclaimer ─ */
.disclaimer-box {
    background:rgba(255,109,0,0.05); border:1px solid rgba(255,109,0,0.20);
    border-radius:8px; padding:0.65rem 0.9rem; font-size:0.78rem; color:#FF8C42 !important;
}

/* ─ Divider ─ */
.custom-divider {
    height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.3),transparent);
    margin:2rem 0; border:none;
}

/* ─ Sidebar ─ */
/* Hide Streamlit's auto-generated page list; the custom navigation below replaces it */
[data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #0A0E1A 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
}
.sidebar-logo {
    text-align:center; padding:1.5rem 1rem 1rem;
    border-bottom:1px solid rgba(0,212,255,0.10); margin-bottom:1rem;
}

/* ─ Buttons ─ */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0080FF 100%) !important;
    color: #0A0E1A !important; font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(0,212,255,0.25) !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,212,255,0.40) !important;
}

/* ─ Upload Widget ─ */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.25) !important;
    border-radius: 12px !important;
    background: rgba(0,212,255,0.03) !important;
    transition: border-color 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,212,255,0.50) !important; }

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    background: rgba(15,22,41,0.8) !important; border-radius: 8px !important;
    border: 1px solid rgba(0,212,255,0.12) !important; color: #78909C !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.15) !important;
    border-color: rgba(0,212,255,0.40) !important; color: #00D4FF !important;
}

/* ─ Scrollbar ─ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.28); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.50); }
/* ─ Brand mark (inline SVG, no emoji) ─ */
.brand-mark {
    display:inline-block; width:40px; height:40px; border-radius:10px;
    background:#0F1A30 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'><circle cx='20' cy='20' r='11' fill='none' stroke='%2300D4FF' stroke-width='2'/><circle cx='20' cy='20' r='5' fill='none' stroke='%2300D4FF' stroke-width='2' opacity='.7'/><circle cx='20' cy='20' r='1.8' fill='%2300D4FF'/><path d='M20 4v6M20 30v6M4 20h6M30 20h6' stroke='%2300D4FF' stroke-width='2' stroke-linecap='round'/></svg>") center/80% no-repeat;
    border:1px solid rgba(0,212,255,0.35);
}
.brand-mark-lg { width:56px; height:56px; border-radius:14px; }

/* ─ Professional refinements ─ */
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
.hero-banner { box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 12px 32px rgba(0,0,0,0.25); }
.hero-title { letter-spacing: -0.02em; }
.metric-card:hover, .status-card:hover { transform: none; }
.metric-card, .status-card { box-shadow: 0 1px 2px rgba(0,0,0,0.25); }
.stButton > button { letter-spacing: .01em; box-shadow: none !important; }
.stButton > button:hover { transform: none !important; filter: brightness(1.08); box-shadow: none !important; }
.stButton > button:disabled { background:#1B2540 !important; color:#5C6B8A !important; }
[data-testid="stDownloadButton"] button {
    background: transparent !important; color:#00D4FF !important;
    border: 1px solid rgba(0,212,255,0.6) !important; font-weight:600 !important;
}
[data-testid="stCaptionContainer"] { color:#78909C !important; }
code { color:#00D4FF !important; background:rgba(0,212,255,0.08) !important; border-radius:4px; }
.app-footer {
    margin-top:3rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.06);
    color:#546E7A; font-size:0.75rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem;
}
</style>
"""


FOOTER_HTML = """
<div class="app-footer">
    <span>DisasterLens AI v1.0.0 &middot; Research &amp; educational use</span>
    <span>Predictions require human expert verification before any operational use.</span>
</div>
"""
