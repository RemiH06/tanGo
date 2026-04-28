import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"


# ── INTERSECTIONS ───────────────────────
def get_intersections():
    """
    Obtiene todas las intersecciones desde la API
    y las convierte en DataFrame
    """
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
    """
    Obtiene métricas generales del sistema
    """
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
    """
    Obtiene el mapa de presión (dict)
    """
    try:
        res = requests.get(f"{API_URL}/pressure-map")
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"Error obteniendo pressure map: {e}")
        return {}


# ── FILTROS ────────────────────────────
def filter_by_phase(df, phase):
    """
    Filtra el dataframe por fase
    """
    if phase == "Todas":
        return df
    return df[df["phase"] == phase]


# ── COLORES PARA MAPA ──────────────────
def get_color(pressure):
    """
    Devuelve color según presión
    """
    if pressure < 0.3:
        return "green"
    elif pressure < 0.7:
        return "orange"
    else:
        return "red"