# app.py — Barrio Pizza · Dashboard de Compras
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import base64

# ================================
# Carga de variables de entorno (opcional)
# ================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ================================
# Importación de módulos propios
# ================================
from modules import data_loader
from modules import forecasting
from modules import alerts
from modules import anomaly
from modules import insights
from modules import chat_engine
from modules import history

# ================================
# 0. SISTEMA DE DISEÑO — TOKENS DE COLOR
# ================================
# Paleta cálida anclada en el mundo de la pizzería: tinta (negro cálido),
# espresso (marca), y estados semánticos que evocan ingredientes —
# tomate (crítico), queso/corteza (atención) y albahaca (correcto).
COLORS = {
    # Claves heredadas (se mantienen para compatibilidad con el resto del código)
    "primary": "#1C1917",
    "secondary": "#7A5237",
    "red": "#C0453D",
    "yellow": "#C08A2B",
    "green": "#5E7A4F",
    "white": "#FFFFFF",
    "light": "#FBFAF8",
    # Tokens semánticos
    "ink": "#1C1917",
    "espresso": "#7A5237",
    "tomato": "#C0453D",
    "cheese": "#C08A2B",
    "basil": "#5E7A4F",
    "muted": "#78716C",
    "line": "#ECE8E1",
    "canvas": "#FBFAF8",
    "surface": "#FFFFFF",
}

# ================================
# RUTA DEL LOGO (CON ESPACIOS)
# ================================
LOGO_PATH = 'assets/logo barrio pizza.jpg'

def get_logo():
    if os.path.exists(LOGO_PATH):
        return LOGO_PATH
    return None

LOGO_VALIDO = get_logo()

# ================================
# Configuración de página (Favicon en pestaña)
# ================================
page_icon = LOGO_VALIDO if LOGO_VALIDO else "🍕"
st.set_page_config(
    page_title="Barrio Pizza - Dashboard de Compras",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Inyección de CSS (sistema de diseño)
# ================================
def local_css():
    # 1) Variables de diseño (:root) — único bloque interpolado
    st.markdown(f"""
    <style>
    :root {{
      --ink: {COLORS['ink']};
      --espresso: {COLORS['espresso']};
      --tomato: {COLORS['tomato']};
      --cheese: {COLORS['cheese']};
      --basil: {COLORS['basil']};
      --muted: {COLORS['muted']};
      --line: {COLORS['line']};
      --canvas: {COLORS['canvas']};
      --surface: {COLORS['surface']};
      --tint-tomato: #FAF0EE;
      --tint-cheese: #FBF4E8;
      --tint-basil: #F1F5EE;
      --tint-neutral: #F6F4F0;
      --tint-espresso: #F3EEE8;
      --r-lg: 16px;
      --r-md: 10px;
      --r-sm: 8px;
      --shadow-sm: 0 1px 2px rgba(28,25,23,0.04), 0 1px 3px rgba(28,25,23,0.05);
      --shadow-md: 0 6px 20px rgba(28,25,23,0.08), 0 2px 6px rgba(28,25,23,0.05);
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-serif: 'Fraunces', Georgia, 'Times New Roman', serif;
      --font-mono: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
    }}
    </style>
    """, unsafe_allow_html=True)

    # 2) Reglas (sin interpolación → usa var(--x); las llaves son literales)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- Base ---------- */
    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
    button, input, textarea, select, .stMarkdown {
      font-family: var(--font-sans);
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }
    body, .stApp { color: var(--ink); }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--canvas); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] .block-container { padding-top: 2.25rem; padding-bottom: 3rem; max-width: 1200px; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .sidebar-brand { font-family: var(--font-serif); font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); text-align: center; margin-top: 0.85rem; line-height: 1.1; }
    .sidebar-sub { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); text-align: center; margin-top: 0.4rem; }
    .nav-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); margin: 1.4rem 0 0.5rem; }

    /* Nav (radio como lista de navegación) */
    section[data-testid="stSidebar"] .stRadio > div { gap: 0.15rem; }
    section[data-testid="stSidebar"] .stRadio label {
      display: flex; align-items: center; gap: 0.55rem;
      padding: 0.5rem 0.7rem; border-radius: var(--r-md);
      font-size: 0.92rem; font-weight: 500; color: var(--ink);
      cursor: pointer; transition: background 140ms ease, color 140ms ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover { background: var(--tint-neutral); }
    section[data-testid="stSidebar"] .stRadio label:has(input:checked) { background: var(--tint-espresso); font-weight: 600; }

    /* ---------- Header ---------- */
    .app-header__title { font-family: var(--font-serif); font-size: 2rem; font-weight: 600; letter-spacing: -0.02em; color: var(--ink); line-height: 1.05; }
    .app-header__meta { font-size: 0.78rem; letter-spacing: 0.02em; color: var(--muted); margin-top: 0.35rem; }
    .header-rule { height: 1px; background: var(--line); margin: 0.75rem 0 1.9rem; }

    /* ---------- Encabezados de sección ---------- */
    .sec-header { margin: 1.9rem 0 1.05rem; }
    .sec-kicker { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--espresso); margin-bottom: 0.35rem; }
    .sec-title { font-size: 1.22rem; font-weight: 600; letter-spacing: -0.015em; color: var(--ink); line-height: 1.25; }
    .sec-sub { font-size: 0.86rem; color: var(--muted); margin-top: 0.32rem; line-height: 1.55; max-width: 72ch; }

    /* ---------- Tarjetas KPI ---------- */
    .kpi-card {
      background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg);
      padding: 1.15rem 1.2rem; height: 100%;
      transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }
    .kpi-card:hover { border-color: #DED8CF; box-shadow: var(--shadow-md); transform: translateY(-2px); }
    .kpi-top { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.7rem; }
    .kpi-dot { width: 8px; height: 8px; border-radius: 999px; flex: 0 0 auto; }
    .kpi-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
    .kpi-value { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; font-feature-settings: 'tnum' 1, 'lnum' 1; }
    .kpi-value--text { font-size: 1.3rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .kpi-sub { font-size: 0.77rem; color: var(--muted); margin-top: 0.5rem; }

    /* ---------- Botones ---------- */
    .stButton > button {
      background: var(--ink); color: #FFFFFF; border: 1px solid var(--ink);
      border-radius: var(--r-md); font-weight: 500; font-size: 0.9rem;
      padding: 0.55rem 1.15rem;
      transition: background 150ms ease, transform 120ms ease, box-shadow 150ms ease;
    }
    .stButton > button:hover { background: #2E2622; border-color: #2E2622; transform: translateY(-1px); box-shadow: var(--shadow-md); }
    .stButton > button:active { transform: translateY(0); box-shadow: var(--shadow-sm); }
    .stButton > button:focus, .stButton > button:focus-visible { outline: 2px solid var(--espresso); outline-offset: 2px; box-shadow: none; }

    /* ---------- Entradas ---------- */
    [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, [data-testid="stChatInput"] textarea {
      border-radius: var(--r-md) !important; border-color: var(--line) !important; font-family: var(--font-sans) !important;
    }
    [data-baseweb="select"] > div:focus-within, .stTextInput input:focus, [data-testid="stChatInput"] > div:focus-within {
      border-color: var(--espresso) !important; box-shadow: 0 0 0 3px var(--tint-espresso) !important;
    }
    [data-testid="stChatInput"] { border-radius: var(--r-md); }

    /* ---------- Tablas / editor ---------- */
    [data-testid="stDataFrame"], [data-testid="stDataEditor"], [data-testid="stTable"] {
      border: 1px solid var(--line); border-radius: var(--r-lg); overflow: hidden;
    }

    /* ---------- Expander ---------- */
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: var(--r-lg); overflow: hidden; background: var(--surface); }
    [data-testid="stExpander"] summary { font-weight: 600; font-size: 0.92rem; }
    [data-testid="stExpander"] summary:hover { color: var(--espresso); }

    /* ---------- Alertas Streamlit (suavizadas) ---------- */
    [data-testid="stAlert"], [data-testid="stNotification"], .stAlert {
      border-radius: var(--r-md); border: 1px solid var(--line); box-shadow: none; font-size: 0.87rem;
    }

    /* ---------- Métricas ---------- */
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 1rem 1.1rem; }
    [data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.01em; font-feature-settings: 'tnum' 1; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.7rem; }

    /* ---------- Callouts / hints ---------- */
    .hint { display: block; position: relative; background: var(--tint-neutral); border: 1px solid var(--line); border-radius: var(--r-md); padding: 0.72rem 0.9rem 0.72rem 1.7rem; font-size: 0.85rem; color: var(--ink); line-height: 1.55; margin: 0.25rem 0 0.75rem; }
    .hint::before { content: ""; position: absolute; left: 0.9rem; top: 0.82rem; width: 6px; height: 6px; border-radius: 999px; background: var(--espresso); }
    .hint code { background: #ECE8E1; border-radius: 5px; padding: 0.05rem 0.35rem; font-family: var(--font-mono); font-size: 0.82em; }
    .hint-warn { background: var(--tint-cheese); border-color: #EEDFC2; }
    .hint-warn::before { background: var(--cheese); }

    /* ---------- Tarjetas de anomalías ---------- */
    .anomaly-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 1rem 1.1rem; margin-bottom: 0.75rem; }
    .anomaly-card__head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
    .anomaly-dot { width: 8px; height: 8px; border-radius: 999px; flex: 0 0 auto; }
    .anomaly-card__body { font-size: 0.92rem; color: var(--ink); line-height: 1.55; }
    .chip { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em; padding: 0.16rem 0.6rem; border-radius: 999px; }
    .chip-warn { background: var(--tint-cheese); color: #8A6318; }

    /* ---------- Footer ---------- */
    .footer { margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--line); text-align: center; font-size: 0.8rem; color: var(--muted); }

    /* ---------- Varios ---------- */
    hr { border: none; border-top: 1px solid var(--line); margin: 1rem 0; }
    code, pre { font-family: var(--font-mono); }
    [data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
    a { color: var(--espresso); }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: #E2DCD3; border-radius: 999px; border: 2px solid var(--canvas); }
    ::-webkit-scrollbar-thumb:hover { background: #D3CBBF; }

    /* ---------- Reduced motion ---------- */
    @media (prefers-reduced-motion: reduce) {
      * { transition: none !important; animation: none !important; }
      .kpi-card:hover, .stButton > button:hover { transform: none !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # 3) Capa "con vida": animaciones, hero, iconos y profundidad
    st.markdown("""
    <style>
    /* ===== Motion + profundidad ===== */
    @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideGradient { to { background-position: 200% 0; } }
    @keyframes livePulse {
      0% { box-shadow: 0 0 0 0 rgba(94,122,79,0.45); }
      70% { box-shadow: 0 0 0 8px rgba(94,122,79,0); }
      100% { box-shadow: 0 0 0 0 rgba(94,122,79,0); }
    }
    @keyframes pulseRing {
      0% { box-shadow: 0 0 0 0 rgba(192,69,61,0.42); }
      70% { box-shadow: 0 0 0 12px rgba(192,69,61,0); }
      100% { box-shadow: 0 0 0 0 rgba(192,69,61,0); }
    }

    /* Fondo ambiental cálido */
    .stApp::before {
      content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background:
        radial-gradient(1100px 480px at 12% -8%, rgba(122,82,55,0.055), transparent 60%),
        radial-gradient(950px 460px at 102% -4%, rgba(192,69,61,0.035), transparent 55%);
    }
    [data-testid="stMain"] .block-container { position: relative; z-index: 1; }

    /* ---------- Hero ---------- */
    .hero {
      display: flex; align-items: center; justify-content: space-between; gap: 1.25rem;
      padding: 1.15rem 1.4rem; margin-bottom: 1.9rem;
      background: linear-gradient(135deg, #FFFFFF 0%, #FBF7F2 100%);
      border: 1px solid var(--line); border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm); position: relative; overflow: hidden;
    }
    .hero::before {
      content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, var(--espresso), var(--tomato), var(--cheese), var(--basil), var(--espresso));
      background-size: 200% 100%; animation: slideGradient 7s linear infinite;
    }
    .hero__brand { display: flex; align-items: center; gap: 1rem; }
    .hero__logo { height: 52px; width: auto; border-radius: 11px; display: block; }
    .hero__logo--fallback {
      height: 52px; width: 52px; border-radius: 11px; display: flex; align-items: center; justify-content: center;
      background: var(--ink); color: #fff; font-family: var(--font-serif); font-weight: 600; font-size: 1.3rem;
    }
    .hero__title { font-family: var(--font-serif); font-size: 1.85rem; font-weight: 600; letter-spacing: -0.02em; color: var(--ink); line-height: 1.05; }
    .hero__meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; letter-spacing: 0.01em; }
    .hero__status { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0.85rem; border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,0.65); }
    .live-dot { width: 9px; height: 9px; border-radius: 999px; background: var(--basil); flex: 0 0 auto; animation: livePulse 2s ease-out infinite; }
    .hero__status-label { font-size: 0.78rem; font-weight: 600; color: var(--ink); line-height: 1.1; }
    .hero__status-sub { font-size: 0.7rem; color: var(--muted); }

    /* ---------- Iconos de sección ---------- */
    .sec-header--icon { display: flex; gap: 0.85rem; align-items: flex-start; }
    .sec-ic { width: 38px; height: 38px; border-radius: 10px; background: var(--tint-espresso); color: var(--espresso); display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; margin-top: 1px; }
    .sec-ic svg { width: 19px; height: 19px; }
    .sec-head-text { display: block; }
    .sec-title, .sec-sub { display: block; }

    /* ---------- KPI: profundidad + icono ---------- */
    .kpi-card { background: linear-gradient(180deg, #FFFFFF, #FDFCFA); animation: fadeIn 0.45s ease both; }
    .kpi-ic { width: 34px; height: 34px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; }
    .kpi-ic svg { width: 18px; height: 18px; }
    .kpi-value { color: var(--ink); }
    .kpi-card--alert { background: linear-gradient(180deg, #FFF6F4, #FFFDFC); border-color: #F3D9D4; }
    .kpi-card--alert .kpi-ic { animation: pulseRing 2.2s ease-out infinite; }

    /* ---------- Reveal de bloques ---------- */
    .sec-header { animation: fadeUp 0.45s ease both; }
    .anomaly-card { animation: fadeUp 0.45s ease both; }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stDataEditor"], [data-testid="stTable"] { animation: fadeUp 0.45s ease both; }
    [data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: var(--r-lg); background: var(--surface); padding: 0.35rem 0.65rem; margin-bottom: 0.5rem; animation: fadeUp 0.4s ease both; }

    /* ---------- Botones con más presencia ---------- */
    .stButton > button { background: linear-gradient(180deg, #2A211E 0%, #1C1917 100%); box-shadow: var(--shadow-sm); }
    .stButton > button:hover { background: linear-gradient(180deg, #3A2E29 0%, #2A211E 100%); }

    /* ---------- Nav: barra de acento animada ---------- */
    section[data-testid="stSidebar"] .stRadio label { position: relative; }
    section[data-testid="stSidebar"] .stRadio label::before {
      content: ""; position: absolute; left: 2px; top: 50%; transform: translateY(-50%);
      width: 3px; height: 0; border-radius: 999px; background: var(--espresso);
      transition: height 160ms ease;
    }
    section[data-testid="stSidebar"] .stRadio label:has(input:checked)::before { height: 58%; }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
      .stApp::before { display: none; }
      .hero::before, .live-dot, .kpi-card--alert .kpi-ic { animation: none !important; }
      .kpi-card, .sec-header, .anomaly-card, [data-testid="stPlotlyChart"],
      [data-testid="stDataFrame"], [data-testid="stDataEditor"], [data-testid="stChatMessage"] { animation: none !important; }
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ================================
# Constantes de navegación
# ================================
PAGE_PANEL = "Panel"
PAGE_PROV = "Proveedores"
PAGE_ANOM = "Anomalías"
PAGE_SIM = "Simulador"
PAGE_HIST = "Historial"
PAGE_CHAT = "Chat"

# ================================
# Helpers de presentación
# ================================
# Iconos de línea (estilo Lucide), heredan color con currentColor
_ICONS = {
    "dashboard": '<rect x="3" y="3" width="7" height="8" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="11" width="7" height="10" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/>',
    "list": '<line x1="9" y1="6" x2="20" y2="6"/><line x1="9" y1="12" x2="20" y2="12"/><line x1="9" y1="18" x2="20" y2="18"/><circle cx="4.5" cy="6" r="1.4" fill="currentColor" stroke="none"/><circle cx="4.5" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="4.5" cy="18" r="1.4" fill="currentColor" stroke="none"/>',
    "chart": '<line x1="5" y1="20" x2="5" y2="12"/><line x1="12" y1="20" x2="12" y2="5"/><line x1="19" y1="20" x2="19" y2="15"/><line x1="3.2" y1="20.4" x2="20.8" y2="20.4"/>',
    "package": '<path d="M12 2.6l8.4 4.7v9.4L12 21.4 3.6 16.7V7.3L12 2.6z"/><path d="M3.8 7.4L12 12l8.2-4.6"/><line x1="12" y1="12" x2="12" y2="21.2"/>',
    "activity": '<path d="M3 12h3.5l2.4 7 4-15 2.5 8H21"/>',
    "sliders": '<line x1="4" y1="8" x2="20" y2="8"/><line x1="4" y1="16" x2="20" y2="16"/><circle cx="10" cy="8" r="2.6" fill="#FFFFFF"/><circle cx="15" cy="16" r="2.6" fill="#FFFFFF"/>',
    "chat": '<path d="M20.5 15.3a2 2 0 0 1-2 2H8l-4.5 3.4V6a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z"/>',
    "alert-triangle": '<path d="M12 3.6l8.4 14.8H3.6L12 3.6z"/><line x1="12" y1="10" x2="12" y2="13.6"/><circle cx="12" cy="16.4" r="0.9" fill="currentColor" stroke="none"/>',
    "alert-circle": '<circle cx="12" cy="12" r="8.5"/><line x1="12" y1="8" x2="12" y2="12.4"/><circle cx="12" cy="15.8" r="0.9" fill="currentColor" stroke="none"/>',
    "check": '<circle cx="12" cy="12" r="8.5"/><path d="M8.4 12.4l2.4 2.4 4.6-5"/>',
    "store": '<path d="M4 10l1.3-5.5h13.4L20 10"/><path d="M4.6 10v9.4h14.8V10"/><path d="M9.6 19.4V14h4.8v5.4"/>',
    "star": '<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>',
    "history": '<path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l3.5 2"/>',
    "upload": '<path d="M12 15V3"/><path d="M7.5 7.5L12 3l4.5 4.5"/><path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
}

def icon(name, size=19):
    p = _ICONS.get(name, "")
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{p}</svg>')

def section_header(title, subtitle=None, icon_name=None):
    sub_html = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ''
    if icon_name:
        st.markdown(
            f'<div class="sec-header sec-header--icon">'
            f'<span class="sec-ic">{icon(icon_name)}</span>'
            f'<span class="sec-head-text"><span class="sec-title">{title}</span>{sub_html}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="sec-header"><div class="sec-title">{title}</div>{sub_html}</div>',
            unsafe_allow_html=True,
        )

def hint(text, tone="neutral"):
    cls = "hint hint-warn" if tone == "warn" else "hint"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def kpi_card(value, label, sub, color, tint, icon_name, delay=0.0, pulse=False, is_text=False):
    value_cls = "kpi-value kpi-value--text" if is_text else "kpi-value"
    card_cls = "kpi-card kpi-card--alert" if pulse else "kpi-card"
    st.markdown(
        f'<div class="{card_cls}" style="animation-delay:{delay}s;">'
        f'<div class="kpi-top">'
        f'<span class="kpi-ic" style="background:{tint};color:{color};">{icon(icon_name)}</span>'
        f'<span class="kpi-label">{label}</span>'
        f'</div>'
        f'<div class="{value_cls}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def style_fig(fig):
    """Aplica una estética coherente a todas las gráficas Plotly."""
    fig.update_layout(
        font_family="Inter, sans-serif",
        font_size=13,
        font_color=COLORS["ink"],
        title_font_family="Inter, sans-serif",
        title_font_size=15,
        title_font_color=COLORS["ink"],
        title_x=0,
        title_xanchor="left",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="", font_size=12),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=COLORS["line"], font_family="Inter, sans-serif", font_size=12, font_color=COLORS["ink"]),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=COLORS["line"], ticks="",
                     title_font_color=COLORS["muted"], title_font_size=12,
                     tickfont_color=COLORS["muted"], tickfont_size=11)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["line"], zeroline=False,
                     title_font_color=COLORS["muted"], title_font_size=12,
                     tickfont_color=COLORS["muted"], tickfont_size=11)
    return fig

# ================================
# 1. CARGA DE DATOS
# ================================
@st.cache_resource
def load_data():
    try:
        return data_loader.load_all("data")
    except FileNotFoundError as e:
        st.error(f"❌ No se encontraron los archivos CSV en la carpeta 'data/'. Verifica que existan los siguientes archivos: ingredientes.csv, inventario_actual.csv, orden_compra_semana.csv, consumo_historico.csv.")
        st.stop()

# ================================
# 2. FUNCIONES DE NEGOCIO
# ================================
def obtener_alertas(data):
    return alerts.calcular_todas_las_alertas(data)

def obtener_anomalias(data):
    return anomaly.detectar_anomalias(data, umbral_desvio_pct=10.0)

def obtener_pedido_corregido(data, df_alertas):
    return insights.pedido_corregido(data, df_alertas)

# ================================
# 3. FUNCIONES DE VISUALIZACIÓN
# ================================
def mostrar_header():
    hora = datetime.now().strftime('%d/%m/%Y %H:%M')
    if LOGO_VALIDO:
        with open(LOGO_VALIDO, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img class="hero__logo" src="data:image/jpeg;base64,{img_base64}" alt="Barrio Pizza">'
    else:
        logo_html = '<div class="hero__logo--fallback">BP</div>'
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero__brand">{logo_html}'
        f'<div><div class="hero__title">Dashboard de Compras</div>'
        f'<div class="hero__meta">Barrio Pizza · Panamá · desde 2015</div></div>'
        f'</div>'
        f'<div class="hero__status"><span class="live-dot"></span>'
        f'<div><div class="hero__status-label">Sistema activo</div>'
        f'<div class="hero__status-sub">Actualizado {hora}</div></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def mostrar_kpis(df_alertas):
    section_header(
        "Resumen ejecutivo",
        "Vista general de las alertas por sucursal, con KPIs y una lectura rápida del estado de los pedidos.",
        icon_name="dashboard",
    )

    total_criticas = len(df_alertas[df_alertas['severidad'] == 'CRITICO'])
    total_atencion = len(df_alertas[df_alertas['severidad'] == 'ATENCION'])
    total_ok = len(df_alertas[df_alertas['severidad'] == 'OK'])
    criticas_por_suc = df_alertas[df_alertas['severidad'] == 'CRITICO'].groupby('sucursal').size()
    if not criticas_por_suc.empty:
        peor_suc = criticas_por_suc.idxmax()
        peor_count = criticas_por_suc.max()
    else:
        peor_suc = 'Ninguna'
        peor_count = 0

    # Mejor sucursal: la que tiene MENOS alertas críticas. Ojo: hay que incluir
    # las sucursales con 0 críticas (que ni aparecen en criticas_por_suc), así
    # que reindexamos sobre TODAS las sucursales rellenando con 0. Es el espejo
    # de "peor sucursal".
    todas_sucursales = df_alertas['sucursal'].unique()
    if len(todas_sucursales) > 0:
        criticas_todas = (
            df_alertas[df_alertas['severidad'] == 'CRITICO']
            .groupby('sucursal').size()
            .reindex(todas_sucursales, fill_value=0)
        )
        mejor_suc = criticas_todas.idxmin()
        mejor_count = int(criticas_todas.min())
    else:
        mejor_suc = 'Ninguna'
        mejor_count = 0
    mejor_sub = "Sin alertas críticas" if mejor_count == 0 else f"{mejor_count} críticas"

    crit_pulse = total_criticas > 0
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        kpi_card(total_criticas, "Críticas", "Riesgo de quiebre", COLORS['tomato'], "var(--tint-tomato)", "alert-triangle", 0.0, pulse=crit_pulse)
    with col2:
        kpi_card(total_atencion, "Atención", "Sobrepedido o subpedido leve", COLORS['cheese'], "var(--tint-cheese)", "alert-circle", 0.06)
    with col3:
        kpi_card(total_ok, "Correctos", "Dentro del rango", COLORS['basil'], "var(--tint-basil)", "check", 0.12)
    with col4:
        kpi_card(mejor_suc, "Mejor sucursal", mejor_sub, COLORS['basil'], "var(--tint-basil)", "star", 0.18, is_text=True)
    with col5:
        kpi_card(peor_suc, "Peor sucursal", f"{peor_count} críticas", COLORS['espresso'], "var(--tint-espresso)", "store", 0.24, is_text=True)

def mostrar_tabla_alertas(df_alertas):
    section_header(
        "Detalle de alertas",
        "Lista completa de las alertas generadas. Filtra por sucursal, severidad o tipo.",
        icon_name="list",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sucursal_filter = st.selectbox("Filtrar por sucursal", ['Todas'] + sorted(df_alertas['sucursal'].unique()))
    with col2:
        severidad_filter = st.selectbox("Filtrar por severidad", ['Todos', 'CRITICO', 'ATENCION', 'OK'])
    with col3:
        tipo_filter = st.selectbox("Filtrar por tipo", ['Todos'] + df_alertas['tipo'].unique().tolist())
    df_filtrado = df_alertas.copy()
    if sucursal_filter != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['sucursal'] == sucursal_filter]
    if severidad_filter != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['severidad'] == severidad_filter]
    if tipo_filter != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['tipo'] == tipo_filter]
    cols_mostrar = ['sucursal', 'nombre_ingrediente', 'tipo', 'severidad', 'proyeccion', 'stock_actual', 'pedido_formatos', 'necesidad_formatos', 'mensaje', 'sugerencia']
    df_mostrar = df_filtrado[cols_mostrar].copy()
    df_mostrar.columns = ['Sucursal', 'Ingrediente', 'Tipo', 'Severidad', 'Proyección', 'Stock', 'Pedido actual', 'Pedido recomendado', 'Mensaje', 'Sugerencia']

    # Redondear a 1 decimal para Proyección y Stock
    for col in ['Proyección', 'Stock']:
        df_mostrar[col] = df_mostrar[col].round(1)
    # Convertir pedidos a enteros
    for col in ['Pedido actual', 'Pedido recomendado']:
        df_mostrar[col] = df_mostrar[col].astype(int)

    df_mostrar.reset_index(drop=True, inplace=True)
    df_mostrar.index = df_mostrar.index + 1
    def color_row(row):
        if row['Severidad'] == 'CRITICO':
            return ['background-color: #FAF0EE; color: #1C1917;'] * len(row)
        elif row['Severidad'] == 'ATENCION':
            return ['background-color: #FBF4E8; color: #1C1917;'] * len(row)
        else:
            return ['background-color: #F1F5EE; color: #1C1917;'] * len(row)

    st.dataframe(
        df_mostrar.style.apply(color_row, axis=1),
        use_container_width=True,
        height=400,
        column_config={
            "Proyección": st.column_config.NumberColumn(format="%.1f"),
            "Stock": st.column_config.NumberColumn(format="%.1f"),
            "Pedido actual": st.column_config.NumberColumn(format="%.0f"),
            "Pedido recomendado": st.column_config.NumberColumn(format="%.0f"),
            "Mensaje": st.column_config.TextColumn("Mensaje", width="large"),
            "Sugerencia": st.column_config.TextColumn("Sugerencia", width="large"),
        }
    )

def mostrar_graficos(df_alertas):
    section_header(
        "Análisis visual",
        "Identifica de un vistazo las sucursales con más problemas y los ingredientes más críticos.",
        icon_name="chart",
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df_alertas.groupby(['sucursal', 'severidad']).size().reset_index(name='count'),
            x='sucursal', y='count', color='severidad', title='Alertas por sucursal',
            color_discrete_map={'CRITICO': COLORS['red'], 'ATENCION': COLORS['yellow'], 'OK': COLORS['green']},
            template="plotly_white",
        )
        fig.update_traces(marker_line_width=0)
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with col2:
        criticas = df_alertas[df_alertas['severidad'] == 'CRITICO']
        if not criticas.empty:
            top = criticas.groupby('nombre_ingrediente').size().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top.values, y=top.index, orientation='h',
                title='Ingredientes con faltantes críticos',
                color_discrete_sequence=[COLORS['tomato']],
                template="plotly_white",
            )
            fig.update_traces(marker_line_width=0)
            style_fig(fig)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            hint("No hay ingredientes con alertas críticas.")

def mostrar_resumen_proveedor(df_pedido):
    section_header(
        "Pedido corregido por proveedor",
        "Pedido recomendado agrupado por proveedor, listo para copiar y enviar.",
        icon_name="package",
    )

    if df_pedido.empty:
        hint("No hay pedidos recomendados para esta semana.")
        return
    resumen = df_pedido.groupby(['proveedor', 'nombre_ingrediente']).agg({'pedido_recomendado_formatos': 'sum', 'formato_compra': 'first', 'sucursal': lambda x: ', '.join(x.unique())}).reset_index()
    resumen = resumen.sort_values(['proveedor', 'nombre_ingrediente'])
    resumen.columns = ['Proveedor', 'Ingrediente', 'Cantidad a pedir', 'Formato', 'Sucursales']
    st.dataframe(resumen, use_container_width=True)
    if st.button("Copiar texto para proveedores", use_container_width=True):
        textos = insights.texto_por_proveedor(df_pedido)
        texto_completo = "\n\n".join(textos.values())
        st.code(texto_completo, language='text')
        st.success("Texto generado. Selecciona y copia.")

# ================================
# FUNCIONES DE ANOMALÍAS
# ================================
def mostrar_anomalias_ratio(df_anomalias):
    section_header(
        "Anomalías por consumo/caja",
        "Compara el consumo de cada ingrediente por caja vendida entre sucursales. Detecta si una sucursal usa demasiado o muy poco insumo por pizza.",
        icon_name="activity",
    )

    if df_anomalias.empty:
        hint("No se detectaron anomalías significativas en el consumo normalizado por caja.")
        return
    st.dataframe(df_anomalias[['sucursal', 'nombre', 'ratio_por_caja', 'mediana_otras_sucursales', 'desvio_pct', 'direccion']], column_config={'ratio_por_caja': st.column_config.NumberColumn("Ratio (consumo/caja)", format="%.2f"), 'desvio_pct': st.column_config.NumberColumn("Desviación %", format="%.1f%%")}, use_container_width=True)
    fig = px.bar(df_anomalias, x='sucursal', y='desvio_pct', color='direccion', title='Desviación porcentual respecto a la mediana de otras sucursales', color_discrete_map={'por encima': COLORS['red'], 'por debajo': COLORS['green']}, barmode='group', template="plotly_white")
    fig.update_traces(marker_line_width=0)
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def mostrar_anomalias_cobertura(df):
    section_header(
        "Anomalías por cobertura en semanas",
        "Compara cuántas semanas cubre el pedido de cada sucursal. Detecta excesos de stock (compras de más) u olvidos (compras de menos).",
        icon_name="activity",
    )

    if df.empty:
        hint("No se detectaron anomalías en la cobertura de semanas (todas las sucursales piden en rangos similares).")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hallazgos", len(df))
    with col2:
        st.metric("Sin alerta propia", len(df[df['tipo_alerta_original'].isna()]))
    with col3:
        st.metric("Sucursales involucradas", df['sucursal'].nunique())
    st.markdown("---")

    # Función auxiliar para pluralizar correctamente la primera palabra
    def pluralizar_formato(cantidad, formato):
        if cantidad == 1:
            return formato
        partes = formato.split(maxsplit=1)
        if len(partes) == 1:
            return partes[0] + 's'
        primera_palabra = partes[0]
        resto = partes[1]
        if primera_palabra.endswith('s'):
            return formato
        elif primera_palabra.endswith('z'):
            return primera_palabra[:-1] + 'ces' + ' ' + resto
        elif primera_palabra.endswith('e') or primera_palabra.endswith('a') or primera_palabra.endswith('o') or primera_palabra.endswith('i') or primera_palabra.endswith('u'):
            return primera_palabra + 's' + ' ' + resto
        else:
            return primera_palabra + 'es' + ' ' + resto

    for _, row in df.iterrows():
        sucursal = row['sucursal']
        nombre = row['nombre_ingrediente']
        formato = row['formato_compra']
        pedido_actual = row['pedido_actual_formatos']
        multiplo = row['multiplo']
        pedido_recomendado = row['pedido_recomendado_formatos']
        diferencia = row['diferencia_formatos']
        direccion = row['direccion']
        formato_plural = pluralizar_formato(pedido_actual, formato)

        if pedido_actual == 0:
            parte1 = f"{sucursal} no pidió {nombre} esta semana (0 {formato_plural})."
        else:
            if direccion == 'EXCESO':
                comparativo = f"{multiplo}x más de lo que piden"
            else:
                comparativo = f"{multiplo}x menos de lo que piden"
            parte1 = f"{sucursal} pidió {pedido_actual} {formato_plural} de {nombre}: {comparativo} las otras sucursales para su propio consumo."

        if direccion == 'EXCESO':
            if diferencia > 0:
                diferencia_abs = abs(diferencia)
                if diferencia_abs == 0:
                    parte2 = ""
                else:
                    formato_plural_diff = pluralizar_formato(diferencia_abs, formato)
                    if pedido_recomendado == 0:
                        parte2 = f"Con el criterio del resto de la cadena, no debería pedir nada (se excedió por {diferencia_abs} {formato_plural_diff})."
                    else:
                        parte2 = f"Con el criterio del resto de la cadena le alcanzarían con {pedido_recomendado} ({diferencia_abs} {formato_plural_diff} de más)."
            else:
                parte2 = f"Con el criterio del resto de la cadena, el pedido está alineado."
        else:
            if diferencia < 0:
                diferencia_abs = abs(diferencia)
                if diferencia_abs == 0:
                    parte2 = ""
                else:
                    formato_plural_diff = pluralizar_formato(diferencia_abs, formato)
                    if pedido_recomendado == 0:
                        parte2 = f"Con el criterio del resto de la cadena, no debería pedir nada (no pidió, pero las otras sí)."
                    else:
                        parte2 = f"Con el criterio del resto de la cadena pediría {pedido_recomendado} ({diferencia_abs} {formato_plural_diff} de menos)."
            else:
                parte2 = f"Con el criterio del resto de la cadena, el pedido está alineado."

        if parte2:
            mensaje_completo = parte1 + " " + parte2
        else:
            mensaje_completo = parte1
        etiqueta = ""
        if not pd.isna(row['tipo_alerta_original']):
            if row['tipo_alerta_original'] == 'SOBRE_PEDIDO':
                etiqueta = "También tiene alerta: Exceso"
            elif row['tipo_alerta_original'] == 'OLVIDO':
                etiqueta = "También tiene alerta: Olvido"
            elif row['tipo_alerta_original'] == 'SUB_PEDIDO':
                etiqueta = "También tiene alerta: Subpedido"
            else:
                etiqueta = "También tiene alerta"

        status_color = COLORS['tomato'] if pedido_actual == 0 else COLORS['muted']
        chip = f'<span class="chip chip-warn">{etiqueta}</span>' if etiqueta else ''
        st.markdown(
            f'<div class="anomaly-card">'
            f'<div class="anomaly-card__head"><span class="anomaly-dot" style="background:{status_color};"></span>{chip}</div>'
            f'<div class="anomaly-card__body">{mensaje_completo}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ================================
# 4. SIMULADOR
# ================================
def simulador_ordenes(data, df_alertas_original):
    section_header(
        "Simulador de órdenes",
        "Subí un archivo de órdenes o edita las cantidades a mano, y observa cómo se actualizan las alertas al instante.",
        icon_name="sliders",
    )
    hint("Los cambios no se guardan hasta que pulses «Recalcular alertas».")

    # --- Cargar órdenes desde un archivo (acerca la herramienta a su visión final) ---
    with st.expander("Cargar órdenes desde un archivo CSV", expanded=False):
        plantilla = data.orden[['sucursal', 'ingrediente_id', 'cantidad_formatos']].copy()
        nombres_map = data.ingredientes[['ingrediente_id', 'nombre']]
        plantilla = plantilla.merge(nombres_map, on='ingrediente_id', how='left')
        plantilla = plantilla[['sucursal', 'ingrediente_id', 'nombre', 'cantidad_formatos']]
        st.download_button(
            "Descargar plantilla (órdenes actuales)",
            data=plantilla.to_csv(index=False).encode('utf-8'),
            file_name="plantilla_ordenes.csv",
            mime="text/csv",
            use_container_width=True,
        )
        hint("Subí un CSV con las columnas <code>sucursal</code>, <code>ingrediente_id</code> y <code>cantidad_formatos</code> (la columna <code>nombre</code> es opcional). Descargá la plantilla para ver el formato exacto.")
        archivo = st.file_uploader("Archivo de órdenes (CSV)", type=["csv"], key="upload_ordenes")
        if archivo is not None:
            try:
                df_subido = pd.read_csv(archivo)
            except Exception as e:
                st.error(f"No pude leer el CSV: {e}")
                df_subido = None
            if df_subido is not None:
                requeridas = {'sucursal', 'ingrediente_id', 'cantidad_formatos'}
                faltantes = requeridas - set(df_subido.columns)
                if faltantes:
                    st.error(f"Al CSV le faltan columnas: {', '.join(sorted(faltantes))}. Descargá la plantilla para ver el formato.")
                else:
                    df_subido = df_subido[['sucursal', 'ingrediente_id', 'cantidad_formatos']].copy()
                    df_subido['cantidad_formatos'] = pd.to_numeric(df_subido['cantidad_formatos'], errors='coerce')
                    invalidas = int(df_subido['cantidad_formatos'].isna().sum())
                    df_subido = df_subido.dropna(subset=['cantidad_formatos'])
                    df_subido = df_subido[df_subido['cantidad_formatos'] > 0]
                    msg = f"CSV válido: {len(df_subido)} líneas de pedido."
                    if invalidas:
                        msg += f" ({invalidas} fila(s) con cantidad no numérica fueron ignoradas.)"
                    st.success(msg)
                    if st.button("Aplicar órdenes cargadas y recalcular", use_container_width=True):
                        data.set_orden(df_subido[['sucursal', 'ingrediente_id', 'cantidad_formatos']])
                        st.session_state['df_alertas'] = obtener_alertas(data)
                        st.success("Órdenes cargadas. Actualizando…")
                        st.rerun()

    orden_df = data.orden[['sucursal', 'ingrediente_id', 'cantidad_formatos']]
    orden_pivot = orden_df.pivot_table(index='ingrediente_id', columns='sucursal', values='cantidad_formatos', fill_value=0).reset_index()
    nombres = data.ingredientes[['ingrediente_id', 'nombre']].set_index('ingrediente_id')
    orden_pivot['nombre'] = orden_pivot['ingrediente_id'].map(nombres['nombre'])
    orden_pivot = orden_pivot[['ingrediente_id', 'nombre'] + [col for col in orden_pivot.columns if col not in ['ingrediente_id', 'nombre']]]
    edited_df = st.data_editor(orden_pivot, use_container_width=True, num_rows="fixed", key="orden_editor")
    if st.button("Recalcular alertas", use_container_width=True):
        id_vars = ['ingrediente_id', 'nombre']
        value_vars = [col for col in edited_df.columns if col not in id_vars]
        nueva_orden = edited_df.melt(id_vars=id_vars, var_name='sucursal', value_name='cantidad_formatos')
        nueva_orden = nueva_orden[nueva_orden['cantidad_formatos'] > 0]
        nueva_orden = nueva_orden[['sucursal', 'ingrediente_id', 'cantidad_formatos']]
        data.set_orden(nueva_orden)
        df_alertas_nuevas = obtener_alertas(data)
        st.session_state['df_alertas'] = df_alertas_nuevas
        st.success("Alertas recalculadas. Actualizando vista…")
        st.rerun()

# ================================
# 5. CHAT CON IA
# ================================
def configurar_chat(data, df_alertas, df_anomalias, df_pedido):
    section_header(
        "Chat con los datos",
        "Pregunta en lenguaje natural sobre las compras y recibe respuestas basadas en la información real del sistema.",
        icon_name="chat",
    )
    hint("Ejemplo: «¿Qué sucursal está pidiendo demasiado queso esta semana?»")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        # Solo mostramos mensajes de usuario y respuestas de texto del asistente.
        # Los mensajes internos (llamadas a herramientas y sus resultados) se
        # mantienen en el historial para la IA, pero no se dibujan en pantalla.
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if role in ("user", "assistant") and content:
            with st.chat_message(role):
                st.markdown(content)
    if prompt := st.chat_input("Haz una pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # ---- CAMBIO AQUÍ: AIMLAPI_KEY -> GROQ_API_KEY ----
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            engine = chat_engine.ChatEngine(data, df_alertas, df_anomalias, df_pedido)
            respuesta, historial = engine.responder(prompt, st.session_state.messages[:-1])
            st.session_state.messages = historial
        else:
            respuesta = procesar_pregunta_simple(prompt, df_alertas)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)

def procesar_pregunta_simple(pregunta, df_alertas):
    pregunta_lower = pregunta.lower()
    if "queso" in pregunta_lower:
        queso_rows = df_alertas[df_alertas['nombre_ingrediente'].str.contains('mozzarella|burrata|parmesano', case=False)]
        if queso_rows.empty:
            return "No hay alertas relacionadas con quesos en este momento."
        else:
            resumen = "Aquí están las alertas relacionadas con quesos:\n\n"
            for _, row in queso_rows.iterrows():
                resumen += f"- {row['mensaje']}\n"
            return resumen
    elif "sucursal" in pregunta_lower and ("más" in pregunta_lower or "demasiado" in pregunta_lower):
        criticas_por_suc = df_alertas[df_alertas['severidad'] == 'CRITICO'].groupby('sucursal').size()
        if criticas_por_suc.empty:
            return "No hay alertas críticas en ninguna sucursal. ¡Todo está bien!"
        else:
            peor_suc = criticas_por_suc.idxmax()
            return f"La sucursal con más problemas es **{peor_suc}** con {criticas_por_suc.max()} alertas críticas."
    else:
        total_criticas = len(df_alertas[df_alertas['severidad'] == 'CRITICO'])
        total_atencion = len(df_alertas[df_alertas['severidad'] == 'ATENCION'])
        return f"""Resumen general:
- {total_criticas} alertas críticas
- {total_atencion} alertas de atención
- El ingrediente más crítico es **{df_alertas[df_alertas['severidad']=='CRITICO'].iloc[0]['nombre_ingrediente'] if not df_alertas[df_alertas['severidad']=='CRITICO'].empty else 'ninguno'}**.
- Revisa la pestaña 'Panel' para más detalles."""

# ================================
# 5.b HISTORIAL Y COMPARACIÓN ENTRE SEMANAS
# ================================
def _tabla_cambios(titulo, items, color):
    if not items:
        return
    st.markdown(
        f'<div style="margin:0.6rem 0 0.35rem;"><span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:999px;background:{color};margin-right:8px;"></span>'
        f'<span style="font-weight:600;font-size:0.92rem;">{titulo} ({len(items)})</span></div>',
        unsafe_allow_html=True,
    )
    df_c = pd.DataFrame([
        {"Sucursal": it["sucursal"], "Ingrediente": it["ingrediente"],
         "Antes": it["antes"], "Ahora": it["ahora"]}
        for it in items
    ])
    st.dataframe(df_c, use_container_width=True, hide_index=True)


def mostrar_historial(df_alertas):
    section_header(
        "Historial y comparación entre semanas",
        "Guardá una foto de las alertas de cada semana para ver la evolución y comparar la semana actual contra semanas anteriores.",
        icon_name="history",
    )

    resumen_actual = history.resumen_semana(df_alertas)

    # --- Guardar la semana actual ---
    st.markdown('<div class="nav-label" style="margin-top:0.5rem;">Guardar la semana actual en el historial</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        etiqueta = st.text_input("Etiqueta", placeholder="Ej: Semana 24 · junio", label_visibility="collapsed")
    with col_b:
        guardar = st.button("Guardar semana", use_container_width=True)
    if guardar:
        history.guardar_snapshot(df_alertas, etiqueta)
        st.success("Semana guardada en el historial.")
        st.rerun()

    snaps = history.cargar_historial()
    if not snaps:
        hint("Todavía no hay semanas guardadas. Guardá la semana actual para empezar a construir el historial y poder comparar la evolución.")
        return

    # --- Evolución en el tiempo ---
    section_header("Evolución de las alertas", "Cantidad de alertas por semana guardada.")
    df_trend = pd.DataFrame([
        {"Semana": s["etiqueta"], "Críticas": s.get("criticas", 0),
         "Atención": s.get("atencion", 0), "Correctas": s.get("ok", 0)}
        for s in snaps
    ])
    fig = px.line(
        df_trend, x="Semana", y=["Críticas", "Atención", "Correctas"], markers=True,
        title="Alertas por semana",
        color_discrete_map={"Críticas": COLORS['tomato'], "Atención": COLORS['cheese'], "Correctas": COLORS['basil']},
        template="plotly_white",
    )
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with st.expander("Ver semanas guardadas / borrar"):
        df_lista = pd.DataFrame([
            {"Fecha": s["fecha"], "Semana": s["etiqueta"], "Críticas": s.get("criticas", 0),
             "Atención": s.get("atencion", 0), "Correctas": s.get("ok", 0)}
            for s in reversed(snaps)
        ])
        st.dataframe(df_lista, use_container_width=True, hide_index=True)
        opciones_borrar = {f'{s["etiqueta"]} ({s["fecha"]})': s["id"] for s in reversed(snaps)}
        a_borrar = st.selectbox("Borrar una semana", ["—"] + list(opciones_borrar.keys()))
        if a_borrar != "—" and st.button("Borrar semana seleccionada"):
            history.borrar_snapshot(opciones_borrar[a_borrar])
            st.success("Semana borrada.")
            st.rerun()

    # --- Comparación semana actual vs. una guardada ---
    section_header("Comparar la semana actual con una anterior", "Qué se resolvió, qué empeoró y qué sigue en alerta respecto a la semana elegida.")
    opciones = {f'{s["etiqueta"]} ({s["fecha"]})': s for s in reversed(snaps)}
    elegido = st.selectbox("Comparar contra:", list(opciones.keys()))
    previo = opciones[elegido]
    comp = history.comparar(resumen_actual, previo)

    c1, c2, c3 = st.columns(3)
    # En críticas/atención, más = peor -> delta en rojo cuando sube (delta_color inverse)
    c1.metric("Críticas", resumen_actual["criticas"], delta=comp["delta_criticas"], delta_color="inverse")
    c2.metric("Atención", resumen_actual["atencion"], delta=comp["delta_atencion"], delta_color="inverse")
    c3.metric("Correctas", resumen_actual["ok"], delta=comp["delta_ok"])

    if not (comp["resueltas"] or comp["empeoraron"] or comp["mejoraron"]):
        hint("No hubo cambios de estado entre ambas semanas: las mismas líneas siguen igual.")
    else:
        _tabla_cambios("Resueltas (pasaron a OK)", comp["resueltas"], COLORS['basil'])
        _tabla_cambios("Empeoraron o son nuevas", comp["empeoraron"], COLORS['tomato'])
        _tabla_cambios("Mejoraron (siguen en alerta)", comp["mejoraron"], COLORS['cheese'])


# ================================
# 6. FUNCIÓN PRINCIPAL
# ================================
def main():
    mostrar_header()

    with st.sidebar:
        # Logo centrado con base64 (evita problemas con espacios en la ruta)
        if LOGO_VALIDO:
            with open(LOGO_VALIDO, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" width="140" style="max-width: 100%; border-radius: 12px;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; font-size: 3rem;">🍕</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-brand">Barrio Pizza</div>
        <div class="sidebar-sub">Panamá · desde 2015</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-label">Navegación</div>', unsafe_allow_html=True)
        paginas = [PAGE_PANEL, PAGE_PROV, PAGE_ANOM, PAGE_SIM, PAGE_HIST, PAGE_CHAT]
        seleccion = st.radio("Ir a", paginas, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Barrio Pizza © 2026")
        st.caption(f"Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    data = load_data()

    # Sección de ingredientes desconocidos con sugerencia
    if not data.ingredientes_desconocidos.empty:
        with st.expander("Ingredientes desconocidos en órdenes", expanded=True):
            st.dataframe(data.ingredientes_desconocidos, use_container_width=True)
            hint("Revisa si el nombre del ingrediente está mal escrito o si es un insumo nuevo. Si es nuevo, agrégalo al archivo maestro <code>ingredientes.csv</code> con su factor de conversión para que el sistema pueda calcular su proyección y sus alertas.", tone="warn")

    if 'df_alertas' not in st.session_state:
        st.session_state['df_alertas'] = obtener_alertas(data)
    df_alertas = st.session_state['df_alertas']
    df_anomalias = obtener_anomalias(data)
    df_pedido = obtener_pedido_corregido(data, df_alertas)

    if seleccion == PAGE_PANEL:
        mostrar_kpis(df_alertas)
        mostrar_tabla_alertas(df_alertas)
        mostrar_graficos(df_alertas)
    elif seleccion == PAGE_PROV:
        mostrar_resumen_proveedor(df_pedido)
    elif seleccion == PAGE_ANOM:
        section_header(
            "Análisis de anomalías entre sucursales",
            "Comparación entre sucursales para detectar comportamientos de compra inusuales. Elige entre análisis por consumo/caja o por cobertura en semanas.",
            icon_name="activity",
        )
        tipo_anomalia = st.radio("Selecciona el tipo de análisis:", ["Por consumo/caja (Receta)", "Por cobertura en semanas (Stock)"], index=0, horizontal=True)
        if tipo_anomalia == "Por consumo/caja (Receta)":
            mostrar_anomalias_ratio(df_anomalias)
        else:
            df_anomalias_cobertura = anomaly.detectar_anomalias_por_cobertura(data, df_alertas, umbral_multiplo=1.3)
            mostrar_anomalias_cobertura(df_anomalias_cobertura)
    elif seleccion == PAGE_SIM:
        simulador_ordenes(data, df_alertas)
    elif seleccion == PAGE_HIST:
        mostrar_historial(df_alertas)
    elif seleccion == PAGE_CHAT:
        configurar_chat(data, df_alertas, df_anomalias, df_pedido)

    # Footer con logo en lugar de emoji
    if LOGO_VALIDO:
        with open(LOGO_VALIDO, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div class="footer">
            <img src="data:image/jpeg;base64,{img_base64}" style="height:22px; vertical-align:middle; margin-right:8px; border-radius:5px;" />
            Barrio Pizza — Dashboard de Compras · Optimización basada en datos
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="footer">
            Barrio Pizza — Dashboard de Compras · Optimización basada en datos
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()