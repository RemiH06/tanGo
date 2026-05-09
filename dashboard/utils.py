import os
import requests
import pandas as pd

# En Docker: API_URL=http://tango-api:8000 (inyectado por docker-compose)
# En local:  API_URL no definida → usa 127.0.0.1:8000
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


# ── INTERSECTIONS ───────────────────────
def get_intersections():
    try:
        res = requests.get(f"{API_URL}/intersections")
        res.raise_for_status()
        data = res.json()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error obteniendo intersecciones: {e}")
        return pd.DataFrame()


# ── METRICS ────────────────────────────
def get_metrics():
    try:
        res = requests.get(f"{API_URL}/metrics")
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Error obteniendo métricas: {e}")
        return {
            "total_intersections": 0,
            "total_records": 0,
            "total_ticks": 0
        }


# ── PRESSURE MAP ───────────────────────
def get_pressure_map():
    try:
        res = requests.get(f"{API_URL}/pressure-map")
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Error obteniendo pressure map: {e}")
        return {}


# ── FILTROS ────────────────────────────
def filter_by_phase(df, phase):
    if phase == "Todas":
        return df
    return df[df["phase"] == phase]


# ── COLORES PARA MAPA ──────────────────
def get_color(pressure):
    if pressure < 0.3:
        return "green"
    elif pressure < 0.7:
        return "orange"
    else:
        return "red"