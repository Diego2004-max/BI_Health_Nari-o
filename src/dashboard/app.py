"""
SentinelaIA Nariño — Dashboard Streamlit (Volcanic Dark Theme)
Ejecutar desde BI_Health_Nari-o/:
    streamlit run src/dashboard/app.py
"""

import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FINAL = BASE_DIR / "data" / "final" / "dataset_final_eda_municipio_semana.csv"
PRED_PATH  = BASE_DIR / "data" / "final" / "predicciones_riesgo.csv"
MODEL_PATH = BASE_DIR / "src" / "models" / "modelo_riesgo_xgboost.pkl"

st.set_page_config(
    page_title="SentinelaIA Nariño",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME CONSTANTS ─────────────────────────────────────────────────────────
COLORES = {"alto": "#C0392B", "medio": "#E67E22", "bajo": "#2ECC71"}
EMOJI_RIESGO = {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}

COORDS = {
    "PASTO":       (1.2136, -77.2811),
    "SANDONA":     (1.2889, -77.4656),
    "CONSACA":     (1.2333, -77.4667),
    "LA FLORIDA":  (1.3000, -77.4000),
    "YACUANQUER":  (1.1667, -77.3833),
    "NARINO":      (1.3500, -77.2000),
}

GALERAS_IMG = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Galeras_volcano.jpg/1280px-Galeras_volcano.jpg"
GALERAS_IMG2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Galeras_desde_Pasto.jpg/1280px-Galeras_desde_Pasto.jpg"

VOLCANO_THEME = dict(
    paper_bgcolor="rgba(15,15,15,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="rgba(255,255,255,0.8)", family="Inter, sans-serif"),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.1)",
        tickcolor="rgba(255,255,255,0.3)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.1)",
        tickcolor="rgba(255,255,255,0.3)",
    ),
    colorway=["#E67E22", "#C0392B", "#2ECC71", "#3498DB", "#9B59B6"],
)

# ─── CSS GLOBAL OSCURO ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
    font-family: 'Inter', sans-serif !important;
}
.main .block-container {
    background: #0a0a0a;
    padding-top: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f0f 0%, #1a0505 100%) !important;
    border-right: 1px solid rgba(192,57,43,0.3) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.45) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #C0392B, #E67E22) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(192,57,43,0.4) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.08);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.45) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(192,57,43,0.25) !important;
    color: white !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.45) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: white !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Headings ── */
h1, h2, h3 { color: white !important; }
h2 { font-size: 18px !important; font-weight: 700 !important; margin-top: 1.2rem !important; }
h3 { font-size: 15px !important; font-weight: 600 !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Info / Warning / Error boxes ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.8) !important;
}

/* ── Caption / small text ── */
.stCaption, .stCaption * {
    color: rgba(255,255,255,0.35) !important;
    font-size: 11px !important;
}

/* ── Selectbox / Slider controls ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] .stSlider {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 8px !important;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(8px); }
}
@keyframes float-particle {
    0%   { transform: translateY(100vh) rotate(0deg);   opacity: 0; }
    10%  { opacity: 0.6; }
    90%  { opacity: 0.2; }
    100% { transform: translateY(-20px) rotate(360deg); opacity: 0; }
}
@keyframes pulse-ring {
    0%   { transform: scale(0.8); opacity: 1; }
    100% { transform: scale(2);   opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# ─── CARGA DE DATOS ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def cargar_datos():
    if DATA_FINAL.exists():
        df = pd.read_csv(DATA_FINAL)
        df["municipio_upper"] = (
            df["municipio"].str.upper().str.strip()
            .str.replace("Á", "A").str.replace("É", "E")
            .str.replace("Í", "I").str.replace("Ó", "O")
            .str.replace("Ú", "U").str.replace("Ñ", "N")
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii")
        )
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def cargar_predicciones():
    if PRED_PATH.exists():
        return pd.read_csv(PRED_PATH)
    return pd.DataFrame()

# ─── HELPERS ─────────────────────────────────────────────────────────────────

_DEFAULT_LEGEND = dict(
    bgcolor="rgba(255,255,255,0.04)",
    bordercolor="rgba(255,255,255,0.08)",
    borderwidth=1,
)

def apply_volcano_theme(fig, height=400, **extra):
    legend = {**_DEFAULT_LEGEND, **extra.pop("legend", {})}
    fig.update_layout(height=height, legend=legend, **VOLCANO_THEME, **extra)
    return fig

def render_alerta_card(municipio, nivel, tasa, casos):
    colores = {"alto": "#C0392B", "medio": "#E67E22", "bajo": "#2ECC71"}
    emojis  = {"alto": "🔴",      "medio": "🟡",       "bajo": "🟢"}
    color   = colores.get(nivel, "#888")
    emoji   = emojis.get(nivel, "⚪")
    rgb     = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    rgb_str = f"{rgb[0]},{rgb[1]},{rgb[2]}"
    return f"""
    <div style="
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid {color};
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        transition: all 0.3s ease;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
            <div>
                <div style="font-size:14px;font-weight:700;color:white;line-height:1.2">
                    {emoji} {municipio}
                </div>
                <div style="font-size:11px;color:rgba(255,255,255,0.4);margin-top:3px;letter-spacing:0.02em">
                    Tasa: {tasa:.1f} x 100k &nbsp;·&nbsp; {int(casos)} casos
                </div>
            </div>
            <div style="
                background: rgba({rgb_str},0.15);
                border: 1px solid {color};
                border-radius: 999px;
                padding: 3px 12px;
                font-size: 10px;
                font-weight: 700;
                color: {color};
                text-transform: uppercase;
                letter-spacing: 0.1em;
                white-space: nowrap;
                flex-shrink: 0;
            ">{nivel}</div>
        </div>
    </div>
    """

def section_header(title):
    st.markdown(f"""
    <div style="
        border-left: 3px solid #E67E22;
        padding: 6px 0 6px 14px;
        margin: 20px 0 12px 0;
    ">
        <div style="font-size:16px;font-weight:700;color:white">{title}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo + Brand
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:20px">
        <div style="
            position:relative;
            border-radius:12px;
            overflow:hidden;
            height:80px;
            margin-bottom:10px;
            border:1px solid rgba(255,255,255,0.1);
        ">
            <img
                src="{GALERAS_IMG}"
                onerror="this.style.display='none'"
                style="width:100%;height:100%;object-fit:cover;filter:saturate(0.6) brightness(0.45)"
            />
            <div style="
                position:absolute;inset:0;
                background:linear-gradient(to top, rgba(15,15,15,0.9) 0%, transparent 60%);
                display:flex;align-items:flex-end;justify-content:center;padding:8px;
            ">
                <span style="color:rgba(255,255,255,0.6);font-size:10px;letter-spacing:0.08em;text-transform:uppercase">
                    Urcunina — Montaña de Fuego
                </span>
            </div>
        </div>
        <div style="font-size:20px;font-weight:900;color:white;letter-spacing:-0.02em">
            SentinelaIA <span style="color:#E67E22">Nariño</span>
        </div>
        <div style="font-size:11px;color:rgba(255,255,255,0.4);margin-top:2px">
            Volcán Galeras · 6 municipios
        </div>
        <div style="
            display:inline-block;
            background:rgba(230,126,34,0.15);
            border:1px solid rgba(230,126,34,0.4);
            border-radius:999px;
            padding:3px 12px;
            margin-top:8px;
            font-size:10px;font-weight:700;
            color:#E67E22;letter-spacing:0.08em;text-transform:uppercase
        ">🌋 Actividad: Medio</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.08);margin:8px 0 16px"></div>', unsafe_allow_html=True)

    df_full = cargar_datos()

    if df_full.empty:
        st.error("Sin datos. Ejecuta el pipeline ETL.")
        st.stop()

    municipios_disp = ["Todos"] + sorted(df_full["municipio"].unique().tolist())
    municipio_sel   = st.selectbox("🏙️ Municipio", municipios_disp)

    anios_disp = sorted(df_full["anio"].dropna().unique().astype(int).tolist(), reverse=True)
    anio_sel   = st.selectbox("📅 Año", anios_disp)

    evento_sel    = st.selectbox("🦠 Enfermedad", ["IRA + EDA", "IRA", "EDA"])
    semanas_rango = st.slider("📊 Semanas epidemiológicas", 1, 52, (1, 52))

    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.08);margin:16px 0"></div>', unsafe_allow_html=True)

    if st.button("🔄 Actualizar datos", use_container_width=True, type="primary"):
        with st.spinner("Ejecutando pipeline ETL..."):
            for script in [
                "src/etl/download_data.py",
                "src/etl/clean_salud.py",
                "src/etl/clean_dane.py",
                "src/etl/build_dataset_final.py",
            ]:
                subprocess.run(
                    ["python", script], cwd=str(BASE_DIR),
                    capture_output=True, text=True
                )
        st.cache_data.clear()
        st.success("✅ Datos actualizados")
        st.rerun()

    if st.button("🤖 Re-entrenar modelo", use_container_width=True):
        with st.spinner("Entrenando XGBoost..."):
            r = subprocess.run(
                ["python", "src/models/modelo_xgboost.py"],
                cwd=str(BASE_DIR), capture_output=True, text=True
            )
        if r.returncode == 0:
            st.cache_data.clear()
            st.success("✅ Modelo re-entrenado")
            st.rerun()
        else:
            st.error(r.stderr[-500:] if r.stderr else "Error desconocido")

    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.08);margin:16px 0 8px"></div>', unsafe_allow_html=True)
    st.caption(f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("Concurso Datos al Ecosistema 2026 — MinTIC")

# ─── HERO SECTION ────────────────────────────────────────────────────────────

import random as _rnd
_rnd.seed(42)
_particles_html = ""
for _i in range(25):
    _sz  = round(_rnd.uniform(2, 5), 1)
    _lft = round(_rnd.uniform(2, 98), 1)
    _dly = round(_rnd.uniform(0, 5), 1)
    _dur = round(_rnd.uniform(8, 18), 1)
    _alp = round(_rnd.uniform(0.1, 0.45), 2)
    _col = f"rgba(230,126,34,{_alp})" if _i % 2 == 0 else f"rgba(192,57,43,{_alp})"
    _blr = round(_rnd.uniform(0, 1.2), 1)
    _particles_html += (
        f'<div style="position:absolute;width:{_sz}px;height:{_sz}px;'
        f'background:{_col};border-radius:50%;left:{_lft}%;bottom:-10px;'
        f'animation:float-particle {_dur}s linear {_dly}s infinite;'
        f'filter:blur({_blr}px)"></div>'
    )

st.markdown(f"""
<div style="
    position: relative;
    width: calc(100% + 2rem);
    margin: -1rem -1rem 0 -1rem;
    min-height: 520px;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0505 40%, #2d0f0f 70%, #1a0a0a 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
">
    <!-- Imagen de fondo -->
    <div style="
        position: absolute; top:0; left:0; right:0; bottom:0;
        background-image: url('{GALERAS_IMG}');
        background-size: cover;
        background-position: center 60%;
        opacity: 0.28;
        filter: saturate(0.5);
    "></div>

    <!-- Overlay degradado vertical -->
    <div style="
        position: absolute; top:0; left:0; right:0; bottom:0;
        background: linear-gradient(to bottom,
            rgba(10,10,10,0.35) 0%,
            rgba(192,57,43,0.08) 45%,
            rgba(10,10,10,0.85) 100%);
    "></div>

    <!-- Partículas de ceniza / lava (CSS-only, generadas en Python) -->
    <div style="position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;overflow:hidden">
        {_particles_html}
    </div>

    <!-- Contenido principal -->
    <div style="
        position: relative; z-index: 2;
        text-align: center;
        padding: 3rem 2rem;
        max-width: 900px;
        width: 100%;
    ">
        <!-- Badge de alerta volcánica -->
        <div style="
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(192,57,43,0.18);
            border: 1px solid rgba(192,57,43,0.45);
            border-radius: 999px;
            padding: 6px 20px;
            margin-bottom: 28px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        ">
            <span style="
                width:8px;height:8px;border-radius:50%;
                background:#E67E22;display:inline-block;
                box-shadow:0 0 0 0 rgba(230,126,34,0.6);
                animation:pulse-ring 1.8s ease-out infinite;
            "></span>
            <span style="color:#E67E22;font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase">
                🌋 Volcán Galeras &nbsp;·&nbsp; Nivel de Actividad: MEDIO
            </span>
        </div>

        <!-- Título principal -->
        <h1 style="
            font-size: clamp(38px, 7vw, 76px);
            font-weight: 900;
            color: white;
            margin: 0 0 14px 0;
            line-height: 1.05;
            letter-spacing: -0.03em;
            text-shadow: 0 0 80px rgba(192,57,43,0.4);
        ">
            SentinelaIA<br>
            <span style="color:#E67E22">Nariño</span>
        </h1>

        <!-- Subtítulo -->
        <p style="
            font-size: clamp(14px, 2.2vw, 20px);
            color: rgba(255,255,255,0.6);
            margin: 0 auto 36px;
            font-weight: 300;
            max-width: 580px;
            line-height: 1.6;
        ">
            Sistema predictivo de alertas tempranas de enfermedades<br>
            en el área de influencia del <strong style="color:rgba(255,255,255,0.85)">Volcán Galeras</strong>
        </p>

        <!-- KPI Pills dinámicos (placeholders — se actualizan abajo con Python) -->
        <div id="hero-kpis" style="display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-bottom:36px">
            <div style="background:rgba(255,255,255,0.07);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:16px 24px;min-width:130px">
                <div style="font-size:30px;font-weight:900;color:white" id="kpi-casos">—</div>
                <div style="font-size:10px;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px">Casos totales</div>
            </div>
            <div style="background:rgba(255,255,255,0.07);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:16px 24px;min-width:130px">
                <div style="font-size:30px;font-weight:900;color:#E67E22" id="kpi-municipios">6</div>
                <div style="font-size:10px;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px">Municipios</div>
            </div>
            <div style="background:rgba(255,255,255,0.07);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:16px 24px;min-width:130px">
                <div style="font-size:30px;font-weight:900;color:#C0392B" id="kpi-so2">SO₂</div>
                <div style="font-size:10px;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px">Monitoreo activo</div>
            </div>
            <div style="background:rgba(255,255,255,0.07);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:16px 24px;min-width:130px">
                <div style="font-size:30px;font-weight:900;color:#2ECC71" id="kpi-ia">IA</div>
                <div style="font-size:10px;color:rgba(255,255,255,0.45);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px">XGBoost predictor</div>
            </div>
        </div>

        <!-- Scroll hint -->
        <div style="animation:bounce 2.2s ease-in-out infinite;color:rgba(255,255,255,0.3)">
            <div style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px">Explorar datos</div>
            <div style="font-size:20px">↓</div>
        </div>
    </div>
</div>

""", unsafe_allow_html=True)

# ─── FILTRADO ────────────────────────────────────────────────────────────────

df = df_full.copy()
if municipio_sel != "Todos":
    df = df[df["municipio"] == municipio_sel]
df = df[df["anio"] == anio_sel]
if evento_sel != "IRA + EDA":
    df = df[df["evento_estandar"] == evento_sel]
df = df[df["semana_epidemiologica"].between(semanas_rango[0], semanas_rango[1])]

# ─── KPIs ────────────────────────────────────────────────────────────────────

total_casos      = int(df["casos"].sum())
tasa_prom        = df["tasa_x_100k"].mean() if not df.empty else 0
semanas_con_datos = df[df["casos"] > 0]["semana_epidemiologica"].nunique()
so2_prom         = df["so2_flux_ton_dia"].mean() if "so2_flux_ton_dia" in df.columns else 0

anio_ant = anio_sel - 1
df_ant   = df_full[df_full["anio"] == anio_ant]
if evento_sel != "IRA + EDA":
    df_ant = df_ant[df_ant["evento_estandar"] == evento_sel]
if municipio_sel != "Todos":
    df_ant = df_ant[df_ant["municipio"] == municipio_sel]
df_ant    = df_ant[df_ant["semana_epidemiologica"].between(semanas_rango[0], semanas_rango[1])]
casos_ant = int(df_ant["casos"].sum()) if not df_ant.empty else None
delta_casos = int(total_casos - casos_ant) if casos_ant else None

st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🦠 Total casos", f"{total_casos:,}", delta=delta_casos, delta_color="inverse")
with col2:
    st.metric("📊 Tasa x 100k hab.", f"{tasa_prom:.1f}")
with col3:
    st.metric("📅 Semanas con casos", semanas_con_datos)
with col4:
    so2_txt  = f"{so2_prom:.0f} t/día" if so2_prom > 0 else "Sin dato"
    nivel_so2 = "🔴 Alto" if so2_prom > 700 else ("🟡 Medio" if so2_prom > 400 else "🟢 Normal")
    st.metric("🌋 SO₂ Galeras", so2_txt, delta=nivel_so2, delta_color="off")

st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.07);margin:20px 0 16px"></div>', unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Mapa de Riesgo",
    "📈 Canal Endémico",
    "📊 Comparativo Municipal",
    "🌋 Galeras + SO₂",
    "🤖 Predicciones IA",
])

# ════════ TAB 1 — MAPA ═══════════════════════════════════════════════════════
with tab1:
    col_mapa, col_alertas = st.columns([3, 1])

    with col_mapa:
        section_header("Distribución de riesgo — Área Galeras")

        if df.empty:
            st.info("Sin datos para los filtros seleccionados.")
        else:
            df_mapa = (
                df.groupby("municipio_upper")
                .agg(casos=("casos", "sum"), tasa=("tasa_x_100k", "mean"))
                .reset_index()
            )
            q33, q66 = df_mapa["tasa"].quantile(0.33), df_mapa["tasa"].quantile(0.66)
            df_mapa["nivel_riesgo"] = df_mapa["tasa"].apply(
                lambda x: "alto" if x >= q66 else ("medio" if x >= q33 else "bajo")
            )
            df_mapa["lat"]      = df_mapa["municipio_upper"].map(lambda m: COORDS.get(m, (1.2, -77.3))[0])
            df_mapa["lon"]      = df_mapa["municipio_upper"].map(lambda m: COORDS.get(m, (1.2, -77.3))[1])
            df_mapa["size_plot"] = df_mapa["casos"].clip(lower=1)

            fig_mapa = px.scatter_mapbox(
                df_mapa, lat="lat", lon="lon",
                size="size_plot", color="nivel_riesgo",
                color_discrete_map=COLORES,
                hover_name="municipio_upper",
                hover_data={"casos": True, "tasa": ":.1f", "lat": False, "lon": False, "size_plot": False},
                size_max=50, zoom=9.5,
                mapbox_style="carto-darkmatter",
            )
            fig_mapa.add_trace(go.Scattermapbox(
                lat=[1.2217], lon=[-77.3597],
                mode="markers+text",
                marker=dict(size=16, color="#E67E22", symbol="triangle"),
                text=["🌋 Galeras"], textposition="top right",
                name="Volcán Galeras",
                hovertext="Volcán Galeras (4.276 m)",
            ))
            fig_mapa.update_layout(
                height=430,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="h", y=-0.06,
                    bgcolor="rgba(255,255,255,0.05)",
                    bordercolor="rgba(255,255,255,0.1)",
                    font=dict(color="rgba(255,255,255,0.7)"),
                ),
            )
            st.plotly_chart(fig_mapa, use_container_width=True)

    with col_alertas:
        section_header("Panel de alertas")
        if not df.empty:
            df_al = (
                df.groupby("municipio")
                .agg(tasa=("tasa_x_100k", "mean"), casos=("casos", "sum"))
                .reset_index()
                .sort_values("tasa", ascending=False)
            )
            q33a = df_al["tasa"].quantile(0.33)
            q66a = df_al["tasa"].quantile(0.66)
            df_al["nivel"] = df_al["tasa"].apply(
                lambda x: "alto" if x >= q66a else ("medio" if x >= q33a else "bajo")
            )
            cards_html = "".join(
                render_alerta_card(row["municipio"], row["nivel"], row["tasa"], row["casos"])
                for _, row in df_al.iterrows()
            )
            st.markdown(cards_html, unsafe_allow_html=True)

# ════════ TAB 2 — CANAL ENDÉMICO ═════════════════════════════════════════════
with tab2:
    section_header("Canal endémico — Semanas epidemiológicas")

    df_hist = df_full.copy()
    if evento_sel != "IRA + EDA":
        df_hist = df_hist[df_hist["evento_estandar"] == evento_sel]
    if municipio_sel != "Todos":
        df_hist = df_hist[df_hist["municipio"] == municipio_sel]

    canal = (
        df_hist.groupby(["semana_epidemiologica", "anio"])["casos"]
        .sum().reset_index()
        .groupby("semana_epidemiologica")["casos"]
        .agg(mediana="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
    )

    df_actual = (
        df[df["anio"] == anio_sel]
        .groupby("semana_epidemiologica")["casos"].sum().reset_index()
    )

    fig_canal = go.Figure()
    fig_canal.add_trace(go.Scatter(
        x=canal["semana_epidemiologica"].tolist() + canal["semana_epidemiologica"].tolist()[::-1],
        y=canal["p75"].tolist() + canal["p25"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(230,126,34,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Corredor histórico (P25-P75)",
    ))
    fig_canal.add_trace(go.Scatter(
        x=canal["semana_epidemiologica"], y=canal["mediana"],
        mode="lines",
        line=dict(color="rgba(230,126,34,0.6)", dash="dash", width=1.5),
        name="Mediana histórica",
    ))

    alerta_alta = pd.DataFrame()
    if not df_actual.empty:
        fig_canal.add_trace(go.Scatter(
            x=df_actual["semana_epidemiologica"], y=df_actual["casos"],
            mode="lines+markers",
            line=dict(color="#E67E22", width=2.5),
            marker=dict(size=5, color="#E67E22"),
            name=f"Casos {anio_sel}",
        ))
        alerta = df_actual.merge(canal, on="semana_epidemiologica")
        alerta_alta = alerta[alerta["casos"] > alerta["p75"]]
        if not alerta_alta.empty:
            fig_canal.add_trace(go.Scatter(
                x=alerta_alta["semana_epidemiologica"], y=alerta_alta["casos"],
                mode="markers",
                marker=dict(size=11, color="#C0392B", symbol="circle-open", line=dict(width=2, color="#C0392B")),
                name="⚠ Sobre corredor",
            ))

    apply_volcano_theme(fig_canal, height=420,
                        xaxis_title="Semana epidemiológica",
                        yaxis_title="Casos",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=-0.22,
                                    bgcolor="rgba(255,255,255,0.04)",
                                    bordercolor="rgba(255,255,255,0.08)"))
    st.plotly_chart(fig_canal, use_container_width=True)

    semanas_alerta = len(alerta_alta) if not alerta_alta.empty else 0
    if semanas_alerta > 0:
        st.warning(f"⚠ {semanas_alerta} semana(s) superaron el corredor histórico en {anio_sel}")

# ════════ TAB 3 — COMPARATIVO MUNICIPAL ══════════════════════════════════════
with tab3:
    section_header("Comparativo por municipio")

    col_bar, col_heat = st.columns(2)

    with col_bar:
        df_muni = (
            df.groupby(["municipio", "evento_estandar"])
            .agg(casos=("casos", "sum"), tasa=("tasa_x_100k", "mean"))
            .reset_index()
            .sort_values("tasa", ascending=True)
        )
        fig_bar = px.bar(
            df_muni, x="tasa", y="municipio", color="evento_estandar",
            color_discrete_map={"IRA": "#3498DB", "EDA": "#E67E22"},
            orientation="h", barmode="group",
            text="tasa",
            title=f"Tasa x 100k hab — {anio_sel}",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                              textfont_color="rgba(255,255,255,0.7)")
        apply_volcano_theme(fig_bar, height=370, xaxis_title="Tasa x 100k")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_heat:
        evento_heat = "EDA" if evento_sel == "IRA + EDA" else evento_sel
        df_heat = (
            df_full[
                (df_full["anio"] == anio_sel) &
                (df_full["evento_estandar"] == evento_heat)
            ]
            .pivot_table(index="municipio", columns="semana_epidemiologica",
                         values="casos", aggfunc="sum")
            .fillna(0)
        )
        if not df_heat.empty:
            fig_heat = px.imshow(
                df_heat,
                color_continuous_scale=["#0a0a0a", "#2d0f0f", "#C0392B", "#E67E22", "#F9CA24"],
                title=f"Casos por municipio/semana — {anio_sel}",
                labels=dict(x="Semana", y="Municipio", color="Casos"),
            )
            apply_volcano_theme(fig_heat, height=370)
            st.plotly_chart(fig_heat, use_container_width=True)

    section_header("Tabla de detalle")
    df_tabla = (
        df.groupby(["municipio", "evento_estandar"])
        .agg(casos=("casos", "sum"),
             tasa_prom=("tasa_x_100k", "mean"),
             tasa_max=("tasa_x_100k", "max"))
        .reset_index()
        .rename(columns={"tasa_prom": "Tasa prom", "tasa_max": "Tasa máx"})
    )
    df_tabla["Tasa prom"] = df_tabla["Tasa prom"].round(2)
    df_tabla["Tasa máx"]  = df_tabla["Tasa máx"].round(2)
    st.dataframe(df_tabla, use_container_width=True, hide_index=True)

# ════════ TAB 4 — GALERAS SO₂ ════════════════════════════════════════════════
with tab4:
    section_header("🌋 Volcán Galeras — Emisiones SO₂ vs Casos de enfermedad")

    # Foto del volcán con glassmorphism
    st.markdown(f"""
    <div style="
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        margin: 0 0 24px 0;
        border: 1px solid rgba(255,255,255,0.08);
        height: 220px;
    ">
        <img
            src="{GALERAS_IMG}"
            onerror="this.style.display='none'"
            style="width:100%;height:100%;object-fit:cover;filter:saturate(0.65) brightness(0.45);display:block"
        />
        <div style="
            position:absolute;inset:0;
            background:linear-gradient(to top, rgba(15,15,15,0.95) 0%, rgba(15,15,15,0.2) 55%, transparent 100%);
            display:flex;align-items:flex-end;padding:20px 24px;
        ">
            <div>
                <div style="color:rgba(255,255,255,0.45);font-size:10px;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px">
                    🌋 Volcán Galeras · 4.276 msnm · Urcunina - Montaña de Fuego
                </div>
                <div style="color:white;font-size:19px;font-weight:800;letter-spacing:-0.01em">
                    El volcán más activo de Colombia
                </div>
                <div style="color:rgba(255,255,255,0.45);font-size:12px;margin-top:3px">
                    A 9 km de Pasto · Departamento de Nariño
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "so2_flux_ton_dia" not in df.columns or df["so2_flux_ton_dia"].isna().all():
        st.info("Datos SO₂ no disponibles aún. Ejecuta el pipeline completo con fuentes geofísicas.")
    else:
        df_g = (
            df.groupby("semana_epidemiologica")
            .agg(casos=("casos", "sum"), so2=("so2_flux_ton_dia", "mean"))
            .reset_index()
        )

        fig_doble = make_subplots(specs=[[{"secondary_y": True}]])
        fig_doble.add_trace(
            go.Bar(x=df_g["semana_epidemiologica"], y=df_g["casos"],
                   name="Casos IRA/EDA", marker_color="rgba(52,152,219,0.7)"),
            secondary_y=False,
        )
        fig_doble.add_trace(
            go.Scatter(x=df_g["semana_epidemiologica"], y=df_g["so2"],
                       name="SO₂ (t/día)",
                       line=dict(color="#E67E22", width=2.5),
                       mode="lines+markers",
                       marker=dict(size=5, color="#E67E22")),
            secondary_y=True,
        )
        fig_doble.add_hline(y=400, line_dash="dot", line_color="#E67E22",
                            annotation_text="Umbral amarillo (400 t/día)",
                            annotation_font_color="rgba(255,255,255,0.5)",
                            secondary_y=True)
        fig_doble.add_hline(y=700, line_dash="dot", line_color="#C0392B",
                            annotation_text="Umbral naranja (700 t/día)",
                            annotation_font_color="rgba(255,255,255,0.5)",
                            secondary_y=True)
        fig_doble.update_yaxes(title_text="Casos",
                               gridcolor="rgba(255,255,255,0.05)",
                               color="rgba(255,255,255,0.6)",
                               secondary_y=False)
        fig_doble.update_yaxes(title_text="SO₂ (ton/día)",
                               gridcolor="rgba(255,255,255,0.03)",
                               color="rgba(255,255,255,0.6)",
                               secondary_y=True)
        apply_volcano_theme(fig_doble, height=430,
                            title=f"Correlación SO₂ Galeras vs Casos ({anio_sel})",
                            hovermode="x unified",
                            legend=dict(orientation="h", y=-0.15,
                                        bgcolor="rgba(255,255,255,0.04)",
                                        bordercolor="rgba(255,255,255,0.08)"))
        st.plotly_chart(fig_doble, use_container_width=True)

        corr = df_g["casos"].corr(df_g["so2"])
        if abs(corr) > 0.3:
            st.info(f"📊 Correlación Pearson SO₂-Casos: **{corr:.3f}** — relación "
                    f"{'positiva' if corr > 0 else 'negativa'} "
                    f"{'moderada' if abs(corr) < 0.6 else 'fuerte'}")

        df_so2_anual = (
            df_full.groupby(["anio", "semana_epidemiologica"])["so2_flux_ton_dia"]
            .mean().reset_index()
        )
        fig_so2 = px.line(
            df_so2_anual, x="semana_epidemiologica", y="so2_flux_ton_dia",
            color="anio",
            color_discrete_sequence=["#E67E22", "#C0392B", "#9B59B6", "#3498DB", "#2ECC71"],
            title="Emisiones SO₂ históricas por semana",
            labels={"so2_flux_ton_dia": "SO₂ (t/día)", "semana_epidemiologica": "Semana"},
        )
        apply_volcano_theme(fig_so2, height=300)
        st.plotly_chart(fig_so2, use_container_width=True)

# ════════ TAB 5 — PREDICCIONES ═══════════════════════════════════════════════
with tab5:
    section_header("🤖 Predicciones XGBoost — Nivel de riesgo epidemiológico")

    df_pred = cargar_predicciones()

    if df_pred.empty:
        st.warning("⚠ Sin predicciones. El modelo no ha sido entrenado aún.")
        if st.button("🚀 Entrenar modelo ahora", type="primary"):
            with st.spinner("Entrenando XGBoost..."):
                r = subprocess.run(
                    ["python", "src/models/modelo_xgboost.py"],
                    cwd=str(BASE_DIR), capture_output=True, text=True
                )
            if r.returncode == 0:
                st.cache_data.clear()
                st.success("✅ Modelo entrenado")
                st.rerun()
            else:
                st.error(r.stderr[-600:])
    else:
        df_p = df_pred.copy()
        if municipio_sel != "Todos":
            df_p = df_p[df_p["municipio"] == municipio_sel]
        df_p = df_p[df_p["anio"] == anio_sel]
        if evento_sel != "IRA + EDA":
            df_p = df_p[df_p["evento_estandar"] == evento_sel]
        df_p = df_p[df_p["semana_epidemiologica"].between(semanas_rango[0], semanas_rango[1])]

        total_pred  = len(df_p)
        alto_pct    = (df_p["nivel_riesgo_predicho"] == "alto").mean() * 100 if total_pred else 0
        proba_prom  = df_p["probabilidad"].mean() if total_pred else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicciones", f"{total_pred:,}")
        col_b.metric("% Riesgo alto", f"{alto_pct:.1f}%")
        col_c.metric("Confianza promedio", f"{proba_prom:.1%}")

        if not df_p.empty:
            conteo = df_p["nivel_riesgo_predicho"].value_counts().reset_index()
            conteo.columns = ["nivel", "count"]

            fig_pie = px.pie(
                conteo, values="count", names="nivel",
                color="nivel", color_discrete_map=COLORES,
                title="Distribución de riesgo predicho",
                hole=0.48,
            )
            apply_volcano_theme(fig_pie, height=290)

            fig_prob = px.histogram(
                df_p, x="probabilidad", color="nivel_riesgo_predicho",
                color_discrete_map=COLORES, nbins=20,
                title="Distribución de probabilidad",
                labels={"probabilidad": "Probabilidad", "count": "Frecuencia"},
            )
            apply_volcano_theme(fig_prob, height=290)

            col_pie, col_prob = st.columns(2)
            with col_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_prob:
                st.plotly_chart(fig_prob, use_container_width=True)

        section_header("Detalle de predicciones")
        if not df_p.empty:
            df_show = df_p.copy()
            df_show["Riesgo"]    = df_show["nivel_riesgo_predicho"].str.upper()
            df_show["Confianza"] = df_show["probabilidad"].apply(lambda p: f"{p:.1%}")
            st.dataframe(
                df_show[["municipio", "semana_epidemiologica", "anio", "evento_estandar", "Riesgo", "Confianza"]]
                .sort_values(["semana_epidemiologica", "municipio"]),
                use_container_width=True, hide_index=True, height=300,
            )

        if not df_p.empty:
            df_pred_mapa = (
                df_p.loc[df_p.groupby("municipio")["probabilidad"].idxmax()]
                .reset_index(drop=True)
            )
            df_pred_mapa["municipio_upper"] = (
                df_pred_mapa["municipio"].str.upper().str.strip()
                .str.replace("Á", "A").str.replace("É", "E").str.replace("Í", "I")
                .str.replace("Ó", "O").str.replace("Ú", "U").str.replace("Ñ", "N")
            )
            df_pred_mapa["lat"]      = df_pred_mapa["municipio_upper"].map(lambda m: COORDS.get(m, (1.25, -77.35))[0])
            df_pred_mapa["lon"]      = df_pred_mapa["municipio_upper"].map(lambda m: COORDS.get(m, (1.25, -77.35))[1])
            df_pred_mapa["prob_size"] = (df_pred_mapa["probabilidad"] * 100).clip(lower=5)

            fig_mp = px.scatter_mapbox(
                df_pred_mapa, lat="lat", lon="lon",
                size="prob_size", color="nivel_riesgo_predicho",
                color_discrete_map=COLORES,
                hover_name="municipio",
                hover_data={"nivel_riesgo_predicho": True, "probabilidad": ":.1%",
                            "lat": False, "lon": False, "prob_size": False},
                size_max=45, zoom=9.5,
                mapbox_style="carto-darkmatter",
                title="Mapa de riesgo predicho",
            )
            fig_mp.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    bgcolor="rgba(255,255,255,0.05)",
                    bordercolor="rgba(255,255,255,0.1)",
                    font=dict(color="rgba(255,255,255,0.7)"),
                ),
            )
            st.plotly_chart(fig_mp, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 40px;
    padding: 24px 0 32px;
    text-align: center;
">
    <div style="
        display:inline-flex;align-items:center;gap:10px;
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:999px;
        padding:8px 24px;
    ">
        <span style="font-size:16px">🌋</span>
        <span style="color:rgba(255,255,255,0.35);font-size:12px;letter-spacing:0.04em">
            SentinelaIA Nariño
            &nbsp;·&nbsp;
            Concurso <em>Datos al Ecosistema 2026</em> — MinTIC
            &nbsp;·&nbsp;
            Universidad Cooperativa de Colombia
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
