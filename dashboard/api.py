"""
dashboard/api.py
-----------------
API FastAPI para el dashboard de tanGo — KAN-11.

Lee tango_state.json (generado por el DAG de Airflow cada hora)
y expone los endpoints que consume el dashboard de Streamlit.

Endpoints:
    GET /                  → health check
    GET /intersections     → lista de intersecciones con fase y presión
    GET /metrics           → métricas generales del sistema
    GET /pressure-map      → mapa de presión por node_id

Correr:
    uvicorn dashboard.api:app --reload --port 8000

O desde la raíz del proyecto:
    uvicorn dashboard.api:app --reload
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Rutas ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
STATE_JSON = ROOT / "graph" / "tango_state.json"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "tanGo API",
    description = "API de datos para el dashboard de semáforos inteligentes",
    version     = "1.0.0",
)

# CORS — permite que Streamlit (mismo host, distinto puerto) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET"],
    allow_headers  = ["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Carga tango_state.json. Lanza 503 si no existe."""
    if not STATE_JSON.exists():
        raise HTTPException(
            status_code = 503,
            detail      = (
                "tango_state.json no encontrado. "
                "Corre el DAG de Airflow primero: "
                "airflow dags trigger tango_traffic_graph_pipeline"
            ),
        )
    with open(STATE_JSON, encoding="utf-8") as f:
        return json.load(f)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health() -> dict:
    """Health check — también muestra cuándo se actualizó el estado."""
    if STATE_JSON.exists():
        state = _load_state()
        return {
            "status":     "ok",
            "updated_at": state.get("updated_at", "desconocido"),
            "n_nodes":    state.get("n_nodes", 0),
            "n_signaled": state.get("n_signaled", 0),
        }
    return {"status": "ok", "warning": "tango_state.json no existe aún"}


@app.get("/intersections")
def get_intersections() -> list[dict]:
    """
    Lista de intersecciones con fase, presión, coordenadas y vecinos.
    Es exactamente lo que espera utils.py → get_intersections().
    """
    state = _load_state()
    return state.get("intersections", [])


@app.get("/metrics")
def get_metrics() -> dict:
    """
    Métricas generales del sistema.
    Es exactamente lo que espera utils.py → get_metrics().
    """
    state = _load_state()
    metrics = state.get("metrics", {})

    # Enriquecer con contexto de clima y tráfico
    ctx = state.get("context", {})
    metrics["weather_source"]  = ctx.get("weather_source",  "default")
    metrics["temperature_c"]   = ctx.get("temperature_c",   22.0)
    metrics["is_raining"]      = ctx.get("is_raining",      False)
    metrics["traffic_factor"]  = ctx.get("traffic_factor",  1.0)
    metrics["updated_at"]      = state.get("updated_at",    "")

    return metrics


@app.get("/pressure-map")
def get_pressure_map() -> dict[str, float]:
    """
    Mapa de presión por node_id.
    Retorna {node_id: pressure} para todos los nodos.
    """
    state = _load_state()
    return {
        inter["node_id"]: inter.get("pressure", 0.0)
        for inter in state.get("intersections", [])
    }


@app.get("/intersections/{node_id}")
def get_intersection(node_id: str) -> dict:
    """Detalle de una intersección específica por node_id."""
    state = _load_state()
    for inter in state.get("intersections", []):
        if inter["node_id"] == node_id:
            return inter
    raise HTTPException(status_code=404, detail=f"Nodo {node_id} no encontrado")


@app.get("/status")
def get_status() -> dict:
    """Estado completo del sistema — útil para debugging."""
    state = _load_state()
    intersections = state.get("intersections", [])

    phase_counts: dict[str, int] = {}
    for inter in intersections:
        phase = inter.get("phase", "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    return {
        "updated_at":    state.get("updated_at"),
        "n_nodes":       state.get("n_nodes"),
        "n_signaled":    state.get("n_signaled"),
        "phase_counts":  phase_counts,
        "context":       state.get("context", {}),
        "weight_stats":  state.get("weight_stats", {}),
    }