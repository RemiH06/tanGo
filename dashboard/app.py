import streamlit as st
import folium
from streamlit_folium import st_folium

from utils import (
    get_intersections,
    get_metrics,
    filter_by_phase,
    get_color
)

st.set_page_config(
    page_title="tanGo Dashboard",
    layout="wide"
)

st.title("🚦 Dashboard de Semáforos Inteligentes — tanGo")
st.write("Visualización de intersecciones, fases y presión vial usando datos simulados desde la API.")

# ── Cargar datos ───────────────────────────────
df = get_intersections()
metrics = get_metrics()

if df.empty:
    st.error("No se pudo cargar la información. Revisa que la API esté corriendo con uvicorn.")
    st.stop()

# ── Métricas principales ───────────────────────
col1, col2, col3 = st.columns(3)

col1.metric("Intersecciones", metrics["total_intersections"])
col2.metric("Registros", metrics["total_records"])
col3.metric("Ticks", metrics["total_ticks"])

st.divider()

# ── Filtros ────────────────────────────────────
st.sidebar.header("Filtros")

phase_filter = st.sidebar.selectbox(
    "Filtrar por fase del semáforo",
    ["Todas"] + sorted(df["phase"].unique().tolist())
)

df_filtered = filter_by_phase(df, phase_filter)

# ── Tabla ──────────────────────────────────────
st.subheader("📊 Intersecciones")

st.dataframe(
    df_filtered,
    use_container_width=True
)

# ── Mapa ───────────────────────────────────────
st.subheader("🗺️ Mapa de presión vial")

center_lat = df_filtered["latitude"].mean()
center_lon = df_filtered["longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13
)

for _, row in df_filtered.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=10,
        color=get_color(row["pressure"]),
        fill=True,
        fill_color=get_color(row["pressure"]),
        fill_opacity=0.75,
        popup=f"""
        <b>{row['name']}</b><br>
        ID: {row['node_id']}<br>
        Fase: {row['phase']}<br>
        Presión: {row['pressure']:.2f}<br>
        Vecinos: {', '.join(row['neighbors'])}
        """,
        tooltip=f"{row['name']} | Presión: {row['pressure']:.2f}"
    ).add_to(m)

st_folium(
    m,
    width=1100,
    height=550
)

# ── Interpretación simple ──────────────────────
st.subheader("🧠 Lectura rápida")

high_pressure = df_filtered[df_filtered["pressure"] >= 0.7]

if len(high_pressure) > 0:
    st.warning(f"Hay {len(high_pressure)} intersección(es) con presión alta.")
    st.dataframe(high_pressure[["node_id", "name", "phase", "pressure"]], use_container_width=True)
else:
    st.success("No hay intersecciones con presión alta en este filtro.")