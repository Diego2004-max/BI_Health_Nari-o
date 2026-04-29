"""
SentinelaIA Nariño — Dashboard Streamlit
Ejecutar desde BI_Health_Nari-o/:
    streamlit run src/dashboard/app.py
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FINAL = BASE_DIR / "data" / "final" / "dataset_final_eda_municipio_semana.csv"
PRED_PATH = BASE_DIR / "data" / "final" / "predicciones_riesgo.csv"
MODEL_PATH = BASE_DIR / "src" / "models" / "modelo_riesgo_xgboost.pkl"

st.set_page_config(
    page_title="SentinelaIA Nariño",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #F4F6F9; }
    [data-testid="stSidebar"] { background-color: #1a3a5c; }
    [data-testid="stSidebar"] * { color: #E8EFF8 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #B8D4E8 !important; }
    div[data-testid="metric-container"] {
        background: white; border-radius: 10px; padding: 14px 18px;
        border-left: 5px solid #1D9E75;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    h1, h2, h3 { color: #1a3a5c; }
    .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

COLORES = {"alto": "#E24B4A", "medio": "#EF9F27", "bajo": "#1D9E75"}
EMOJI_RIESGO = {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}

COORDS = {
    "PASTO":       (1.2136, -77.2811),
    "SANDONA":     (1.2889, -77.4656),
    "CONSACA":     (1.2333, -77.4667),
    "LA FLORIDA":  (1.3000, -77.4000),
    "YACUANQUER":  (1.1667, -77.3833),
    "NARINO":      (1.3500, -77.2000),
}

# ─── CARGA DE DATOS ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def cargar_datos():
    if DATA_FINAL.exists():
        df = pd.read_csv(DATA_FINAL)
        df["municipio_upper"] = df["municipio"].str.upper().str.strip()
        df["municipio_upper"] = (
            df["municipio_upper"]
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

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌋 SentinelaIA")
    st.markdown("*Área de influencia Volcán Galeras*")
    st.markdown("---")

    df_full = cargar_datos()

    if df_full.empty:
        st.error("Sin datos. Ejecuta el pipeline.")
        st.stop()

    municipios_disp = ["Todos"] + sorted(df_full["municipio"].unique().tolist())
    municipio_sel = st.selectbox("🏙️ Municipio", municipios_disp)

    anios_disp = sorted(df_full["anio"].dropna().unique().astype(int).tolist(), reverse=True)
    anio_sel = st.selectbox("📅 Año", anios_disp)

    evento_sel = st.selectbox("🦠 Enfermedad", ["IRA + EDA", "IRA", "EDA"])

    semanas_rango = st.slider("📊 Semanas", 1, 52, (1, 52))

    st.markdown("---")
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

    st.markdown("---")
    st.caption(f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("Concurso Datos al Ecosistema 2026 — MinTIC")

# ─── HEADER ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(90deg,#1a3a5c,#2E86AB);
            padding:18px 24px;border-radius:10px;margin-bottom:18px">
  <h2 style="color:white;margin:0;font-size:22px">
    🌋 SentinelaIA Nariño — Vigilancia Epidemiológica
  </h2>
  <p style="color:#B8D4E8;margin:4px 0 0;font-size:13px">
    IRA · EDA · Volcán Galeras · 6 municipios · 2020–2024
  </p>
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

total_casos = int(df["casos"].sum())
tasa_prom = df["tasa_x_100k"].mean() if not df.empty else 0
semanas_con_datos = df[df["casos"] > 0]["semana_epidemiologica"].nunique()
so2_prom = df["so2_flux_ton_dia"].mean() if "so2_flux_ton_dia" in df.columns else 0

# Comparar vs año anterior
anio_ant = anio_sel - 1
df_ant = df_full[df_full["anio"] == anio_ant]
if evento_sel != "IRA + EDA":
    df_ant = df_ant[df_ant["evento_estandar"] == evento_sel]
if municipio_sel != "Todos":
    df_ant = df_ant[df_ant["municipio"] == municipio_sel]
df_ant = df_ant[df_ant["semana_epidemiologica"].between(semanas_rango[0], semanas_rango[1])]

casos_ant = int(df_ant["casos"].sum()) if not df_ant.empty else None
delta_casos = int(total_casos - casos_ant) if casos_ant else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🦠 Total casos", f"{total_casos:,}",
              delta=delta_casos, delta_color="inverse")
with col2:
    st.metric("📊 Tasa x 100k hab.", f"{tasa_prom:.1f}")
with col3:
    st.metric("📅 Semanas con casos", semanas_con_datos)
with col4:
    so2_txt = f"{so2_prom:.0f} t/día" if so2_prom > 0 else "Sin dato"
    nivel_so2 = "🔴 Alto" if so2_prom > 700 else ("🟡 Medio" if so2_prom > 400 else "🟢 Normal")
    st.metric("🌋 SO₂ Galeras", so2_txt, delta=nivel_so2, delta_color="off")

st.markdown("---")

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
        st.subheader("Distribución de riesgo — Área Galeras")

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
            df_mapa["lat"] = df_mapa["municipio_upper"].map(lambda m: COORDS.get(m, (1.2, -77.3))[0])
            df_mapa["lon"] = df_mapa["municipio_upper"].map(lambda m: COORDS.get(m, (1.2, -77.3))[1])
            df_mapa["size_plot"] = df_mapa["casos"].clip(lower=1)

            fig_mapa = px.scatter_mapbox(
                df_mapa, lat="lat", lon="lon",
                size="size_plot", color="nivel_riesgo",
                color_discrete_map=COLORES,
                hover_name="municipio_upper",
                hover_data={"casos": True, "tasa": ":.1f", "lat": False, "lon": False, "size_plot": False},
                size_max=50, zoom=9.5,
                mapbox_style="carto-positron",
            )
            # Añadir marcador Galeras
            fig_mapa.add_trace(go.Scattermapbox(
                lat=[1.2217], lon=[-77.3597],
                mode="markers+text",
                marker=dict(size=14, color="#FF6B35", symbol="triangle"),
                text=["🌋 Galeras"], textposition="top right",
                name="Volcán Galeras",
                hovertext="Volcán Galeras (4.276 m)"
            ))
            fig_mapa.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                                   legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(fig_mapa, use_container_width=True)

    with col_alertas:
        st.subheader("Panel alertas")
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
            for _, row in df_al.iterrows():
                color = COLORES[row["nivel"]]
                em = EMOJI_RIESGO[row["nivel"]]
                st.markdown(f"""
                <div style="border-left:5px solid {color};padding:8px 12px;margin:5px 0;
                            background:white;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.07)">
                  <b>{em} {row['municipio']}</b><br>
                  <span style="color:{color};font-size:12px">{row['nivel'].upper()}</span>
                  <span style="float:right;font-size:12px;color:#666">{row['tasa']:.1f}/100k</span>
                </div>
                """, unsafe_allow_html=True)

# ════════ TAB 2 — CANAL ENDÉMICO ═════════════════════════════════════════════
with tab2:
    st.subheader("Canal endémico — Semanas epidemiológicas")

    df_hist = df_full.copy()
    if evento_sel != "IRA + EDA":
        df_hist = df_hist[df_hist["evento_estandar"] == evento_sel]
    if municipio_sel != "Todos":
        df_hist = df_hist[df_hist["municipio"] == municipio_sel]

    # Canal: mediana y percentiles históricos
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
        fill="toself", fillcolor="rgba(46,134,171,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Corredor histórico (P25-P75)"
    ))
    fig_canal.add_trace(go.Scatter(
        x=canal["semana_epidemiologica"], y=canal["mediana"],
        mode="lines", line=dict(color="#2E86AB", dash="dash", width=1.5),
        name="Mediana histórica"
    ))
    if not df_actual.empty:
        fig_canal.add_trace(go.Scatter(
            x=df_actual["semana_epidemiologica"], y=df_actual["casos"],
            mode="lines+markers", line=dict(color="#E84855", width=2.5),
            marker=dict(size=5), name=f"Casos {anio_sel}"
        ))
        # Marcar semanas sobre el corredor
        alerta = df_actual.merge(canal, on="semana_epidemiologica")
        alerta_alta = alerta[alerta["casos"] > alerta["p75"]]
        if not alerta_alta.empty:
            fig_canal.add_trace(go.Scatter(
                x=alerta_alta["semana_epidemiologica"], y=alerta_alta["casos"],
                mode="markers", marker=dict(size=10, color="#E84855", symbol="circle-open"),
                name="⚠ Sobre corredor"
            ))

    fig_canal.update_layout(
        xaxis_title="Semana epidemiológica",
        yaxis_title="Casos",
        height=400, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_canal, use_container_width=True)

    semanas_alerta = len(alerta_alta) if not df_actual.empty else 0
    if semanas_alerta > 0:
        st.warning(f"⚠ {semanas_alerta} semana(s) superaron el corredor histórico en {anio_sel}")

# ════════ TAB 3 — COMPARATIVO MUNICIPAL ══════════════════════════════════════
with tab3:
    st.subheader("Comparativo por municipio")

    col_bar, col_heat = st.columns([1, 1])

    with col_bar:
        df_muni = (
            df.groupby(["municipio", "evento_estandar"])
            .agg(casos=("casos", "sum"), tasa=("tasa_x_100k", "mean"))
            .reset_index()
            .sort_values("tasa", ascending=True)
        )
        fig_bar = px.bar(
            df_muni, x="tasa", y="municipio", color="evento_estandar",
            color_discrete_map={"IRA": "#2E86AB", "EDA": "#E84855"},
            orientation="h", barmode="group",
            text="tasa",
            title=f"Tasa x 100k hab — {anio_sel}",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(height=360, xaxis_title="Tasa x 100k")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_heat:
        df_heat = (
            df_full[
                (df_full["anio"] == anio_sel) &
                (df_full["evento_estandar"] == ("EDA" if evento_sel == "IRA + EDA" else evento_sel))
            ]
            .pivot_table(index="municipio", columns="semana_epidemiologica",
                         values="casos", aggfunc="sum")
            .fillna(0)
        )
        if not df_heat.empty:
            fig_heat = px.imshow(
                df_heat, color_continuous_scale="YlOrRd",
                title=f"Casos por municipio/semana — {anio_sel}",
                labels=dict(x="Semana", y="Municipio", color="Casos"),
            )
            fig_heat.update_layout(height=360)
            st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Tabla de detalle")
    df_tabla = (
        df.groupby(["municipio", "evento_estandar"])
        .agg(casos=("casos", "sum"), tasa_prom=("tasa_x_100k", "mean"),
             tasa_max=("tasa_x_100k", "max"))
        .reset_index()
        .rename(columns={"tasa_prom": "Tasa prom", "tasa_max": "Tasa máx"})
    )
    df_tabla["Tasa prom"] = df_tabla["Tasa prom"].round(2)
    df_tabla["Tasa máx"] = df_tabla["Tasa máx"].round(2)
    st.dataframe(df_tabla, use_container_width=True, hide_index=True)

# ════════ TAB 4 — GALERAS SO₂ ════════════════════════════════════════════════
with tab4:
    st.subheader("🌋 Emisiones SO₂ Volcán Galeras vs Casos de enfermedad")

    if "so2_flux_ton_dia" not in df.columns or df["so2_flux_ton_dia"].isna().all():
        st.info("Datos SO₂ no disponibles aún. Ejecuta el pipeline completo.")
    else:
        df_g = (
            df.groupby("semana_epidemiologica")
            .agg(casos=("casos", "sum"), so2=("so2_flux_ton_dia", "mean"))
            .reset_index()
        )

        fig_doble = make_subplots(specs=[[{"secondary_y": True}]])
        fig_doble.add_trace(
            go.Bar(x=df_g["semana_epidemiologica"], y=df_g["casos"],
                   name="Casos IRA/EDA", marker_color="#2E86AB", opacity=0.7),
            secondary_y=False
        )
        fig_doble.add_trace(
            go.Scatter(x=df_g["semana_epidemiologica"], y=df_g["so2"],
                       name="SO₂ (t/día)", line=dict(color="#FF6B35", width=2.5),
                       mode="lines+markers", marker=dict(size=4)),
            secondary_y=True
        )
        # Línea de alerta SO₂
        fig_doble.add_hline(y=400, line_dash="dot", line_color="#EF9F27",
                            annotation_text="Umbral amarillo (400 t/día)",
                            secondary_y=True)
        fig_doble.add_hline(y=700, line_dash="dot", line_color="#E24B4A",
                            annotation_text="Umbral naranja (700 t/día)",
                            secondary_y=True)
        fig_doble.update_yaxes(title_text="Casos", secondary_y=False)
        fig_doble.update_yaxes(title_text="SO₂ (ton/día)", secondary_y=True)
        fig_doble.update_layout(
            title=f"Correlación SO₂ Galeras vs Casos ({anio_sel})",
            hovermode="x unified", height=420,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_doble, use_container_width=True)

        # Correlación estadística
        corr = df_g["casos"].corr(df_g["so2"])
        if abs(corr) > 0.3:
            st.info(f"📊 Correlación Pearson SO₂-Casos: **{corr:.3f}** — relación {'positiva' if corr>0 else 'negativa'} {'moderada' if abs(corr)<0.6 else 'fuerte'}")

        # Serie temporal anual de SO₂
        df_so2_anual = (
            df_full.groupby(["anio", "semana_epidemiologica"])["so2_flux_ton_dia"]
            .mean().reset_index()
        )
        fig_so2 = px.line(
            df_so2_anual, x="semana_epidemiologica", y="so2_flux_ton_dia",
            color="anio", color_discrete_sequence=px.colors.sequential.Plasma_r,
            title="Emisiones SO₂ históricas por semana",
            labels={"so2_flux_ton_dia": "SO₂ (t/día)", "semana_epidemiologica": "Semana"},
        )
        fig_so2.update_layout(height=300)
        st.plotly_chart(fig_so2, use_container_width=True)

# ════════ TAB 5 — PREDICCIONES ═══════════════════════════════════════════════
with tab5:
    st.subheader("🤖 Predicciones XGBoost — Nivel de riesgo")

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
        # Filtrar predicciones
        df_p = df_pred.copy()
        if municipio_sel != "Todos":
            df_p = df_p[df_p["municipio"] == municipio_sel]
        df_p = df_p[df_p["anio"] == anio_sel]
        if evento_sel != "IRA + EDA":
            df_p = df_p[df_p["evento_estandar"] == evento_sel]
        df_p = df_p[df_p["semana_epidemiologica"].between(semanas_rango[0], semanas_rango[1])]

        # KPIs predicciones
        total_pred = len(df_p)
        alto_pct = (df_p["nivel_riesgo_predicho"] == "alto").mean() * 100 if total_pred else 0
        proba_prom = df_p["probabilidad"].mean() if total_pred else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicciones", f"{total_pred:,}")
        col_b.metric("% Riesgo alto", f"{alto_pct:.1f}%")
        col_c.metric("Confianza promedio", f"{proba_prom:.1%}")

        # Distribución
        if not df_p.empty:
            conteo = df_p["nivel_riesgo_predicho"].value_counts().reset_index()
            conteo.columns = ["nivel", "count"]
            fig_pie = px.pie(
                conteo, values="count", names="nivel",
                color="nivel", color_discrete_map=COLORES,
                title="Distribución de riesgo predicho",
                hole=0.45,
            )
            fig_pie.update_layout(height=280)

            fig_prob = px.histogram(
                df_p, x="probabilidad", color="nivel_riesgo_predicho",
                color_discrete_map=COLORES, nbins=20,
                title="Distribución de probabilidad",
                labels={"probabilidad": "Probabilidad", "count": "Frecuencia"},
            )
            fig_prob.update_layout(height=280)

            col_pie, col_prob = st.columns(2)
            with col_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_prob:
                st.plotly_chart(fig_prob, use_container_width=True)

        # Tabla detallada
        st.markdown("#### Detalle de predicciones")
        if not df_p.empty:
            df_show = df_p.copy()
            df_show["Riesgo"] = df_show["nivel_riesgo_predicho"].str.upper()
            df_show["Confianza"] = df_show["probabilidad"].apply(lambda p: f"{p:.1%}")
            st.dataframe(
                df_show[["municipio", "semana_epidemiologica", "anio", "evento_estandar", "Riesgo", "Confianza"]]
                .sort_values(["semana_epidemiologica", "municipio"]),
                use_container_width=True, hide_index=True, height=300,
            )

        # Mapa de predicciones
        if not df_p.empty:
            df_pred_mapa = (
                df_p.groupby("municipio")
                .apply(lambda g: g.loc[g["probabilidad"].idxmax()])
                .reset_index(drop=True)
            )
            df_pred_mapa["municipio_upper"] = (
                df_pred_mapa["municipio"].str.upper().str.strip()
                .str.replace("Á","A").str.replace("É","E").str.replace("Í","I")
                .str.replace("Ó","O").str.replace("Ú","U").str.replace("Ñ","N")
            )
            df_pred_mapa["lat"] = df_pred_mapa["municipio_upper"].map(
                lambda m: COORDS.get(m, (1.25, -77.35))[0])
            df_pred_mapa["lon"] = df_pred_mapa["municipio_upper"].map(
                lambda m: COORDS.get(m, (1.25, -77.35))[1])
            df_pred_mapa["prob_size"] = (df_pred_mapa["probabilidad"] * 100).clip(lower=5)

            fig_mp = px.scatter_mapbox(
                df_pred_mapa, lat="lat", lon="lon",
                size="prob_size", color="nivel_riesgo_predicho",
                color_discrete_map=COLORES,
                hover_name="municipio",
                hover_data={"nivel_riesgo_predicho": True, "probabilidad": ":.1%",
                            "lat": False, "lon": False, "prob_size": False},
                size_max=45, zoom=9.5, mapbox_style="carto-positron",
                title="Mapa de riesgo predicho",
            )
            fig_mp.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_mp, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px'>"
    "SentinelaIA Nariño &nbsp;·&nbsp; Concurso <i>Datos al Ecosistema 2026</i> — MinTIC &nbsp;·&nbsp;"
    " Universidad Cooperativa de Colombia, Pasto"
    "</div>",
    unsafe_allow_html=True,
)
