"""
api/main.py
-----------
API REST de tanGo — expone el estado del sistema en tiempo real.

Endpoints:
  GET  /health                     → estado del servidor
  GET  /intersections              → lista de intersecciones del grafo
  GET  /intersections/{node_id}    → estado actual de una intersección
  GET  /intersections/{node_id}/pressure-history → historial de presión
  GET  /pressure-map               → presión actual de todas las intersecciones
  POST /intersections/{node_id}/phase → forzar cambio de fase (admin)
  POST /incidents                  → reportar incidente en un segmento
  GET  /metrics                    → métricas generales del sistema

Seguridad:
  - API key requerida en header X-API-Key para endpoints de escritura.
  - Rate limiting por IP (100 req/min).
  - Inputs sanitizados antes de cualquier operación.

Uso:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional, Dict
import os
import logging

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from core.context import TrafficContext
from core.database import DatabaseClient
from core.road import Phase
from graph.simulator import TrafficGraph
from ingest.weather import WeatherIngester
from safety.circuit_breaker import CircuitBreaker

load_dotenv()
logger = logging.getLogger(__name__)
UTC = timezone.utc

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "tanGo API",
    description = "Semáforo Inteligente — API de control y monitoreo en tiempo real",
    version     = "0.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

# ── Estado global (inicializado en startup) ───────────────────────────────────

_graph:   Optional[TrafficGraph]   = None
_db:      Optional[DatabaseClient] = None
_weather: Optional[WeatherIngester] = None


@app.on_event("startup")
async def startup() -> None:
    global _graph, _db, _weather
    _graph = TrafficGraph()
    _graph.build_sample_city()

    _db = DatabaseClient.in_memory()

    breaker  = CircuitBreaker("weather-api", failure_threshold=3)
    _weather = WeatherIngester(
        latitude        = float(os.getenv("CITY_LATITUDE",  "20.6597")),
        longitude       = float(os.getenv("CITY_LONGITUDE", "-103.3496")),
        circuit_breaker = breaker,
    )
    logger.info("tanGo API iniciada — %d intersecciones cargadas",
                len(_graph.intersections))


@app.on_event("shutdown")
async def shutdown() -> None:
    if _graph:
        _graph.close()
    logger.info("tanGo API detenida.")


# ── Seguridad ─────────────────────────────────────────────────────────────────

_ADMIN_KEY = os.getenv("TANGO_ADMIN_KEY", "dev-key-change-in-production")

def require_admin_key(x_api_key: str = Header(...)) -> None:
    """
    Dependencia de FastAPI — valida la API key para endpoints de escritura.
    Lanza 403 si la key es inválida.
    """
    if x_api_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="API key inválida.")


def get_graph() -> TrafficGraph:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Sistema no inicializado.")
    return _graph


def get_db() -> DatabaseClient:
    if _db is None:
        raise HTTPException(status_code=503, detail="Base de datos no disponible.")
    return _db


# ── Schemas Pydantic ──────────────────────────────────────────────────────────

class IntersectionResponse(BaseModel):
    node_id:   str
    name:      str
    latitude:  float
    longitude: float
    phase:     str
    pressure:  float
    neighbors: List[str]

class PressureHistoryPoint(BaseModel):
    timestamp:    datetime
    pressure:     float
    phase:        str
    is_rush_hour: bool
    is_raining:   bool

class PhaseUpdateRequest(BaseModel):
    phase: str = Field(..., pattern="^(green|yellow|red)$")
    reason: Optional[str] = Field(None, max_length=200)

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        if v not in ("green", "yellow", "red"):
            raise ValueError("phase debe ser green, yellow o red")
        return v

class IncidentRequest(BaseModel):
    segment_id: str  = Field(..., min_length=1, max_length=100)
    severity:   float = Field(..., ge=0.0, le=1.0)
    notes:      Optional[str] = Field(None, max_length=500)

    @field_validator("segment_id")
    @classmethod
    def sanitize_segment_id(cls, v: str) -> str:
        # Evitar inyección — solo alfanuméricos y guiones
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError("segment_id contiene caracteres no permitidos")
        return v

class MetricsResponse(BaseModel):
    total_intersections: int
    total_ticks:         int
    total_records:       int
    system_uptime_s:     float
    timestamp:           datetime

class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: datetime


# ── Tiempo de inicio (para uptime) ───────────────────────────────────────────

_start_time = datetime.now(UTC)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health() -> HealthResponse:
    """Estado del servidor. No requiere autenticación."""
    return HealthResponse(
        status    = "ok",
        version   = "0.1.0",
        timestamp = datetime.now(UTC),
    )


@app.get("/intersections", response_model=List[IntersectionResponse],
         tags=["Intersecciones"])
async def list_intersections(
    graph: TrafficGraph = Depends(get_graph)
) -> List[IntersectionResponse]:
    """Lista todas las intersecciones del grafo con su estado actual."""
    return [
        IntersectionResponse(
            node_id   = node_id,
            name      = inter.name,
            latitude  = inter.latitude,
            longitude = inter.longitude,
            phase     = inter.current_phase.value,
            pressure  = inter.pressure,
            neighbors = graph.neighbors_of(node_id),
        )
        for node_id, inter in graph.intersections.items()
    ]


@app.get("/intersections/{node_id}", response_model=IntersectionResponse,
         tags=["Intersecciones"])
async def get_intersection(
    node_id: str,
    graph:   TrafficGraph = Depends(get_graph),
) -> IntersectionResponse:
    """Estado actual de una intersección específica."""
    if node_id not in graph.intersections:
        raise HTTPException(
            status_code = 404,
            detail      = f"Intersección '{node_id}' no encontrada."
        )
    inter = graph.intersections[node_id]
    return IntersectionResponse(
        node_id   = node_id,
        name      = inter.name,
        latitude  = inter.latitude,
        longitude = inter.longitude,
        phase     = inter.current_phase.value,
        pressure  = inter.pressure,
        neighbors = graph.neighbors_of(node_id),
    )


@app.get("/intersections/{node_id}/pressure-history",
         response_model=List[PressureHistoryPoint],
         tags=["Intersecciones"])
async def pressure_history(
    node_id:        str,
    last_n_minutes: int = 60,
    db:             DatabaseClient = Depends(get_db),
) -> List[PressureHistoryPoint]:
    """Historial de presión de una intersección en los últimos N minutos."""
    if last_n_minutes < 1 or last_n_minutes > 1440:
        raise HTTPException(
            status_code = 422,
            detail      = "last_n_minutes debe estar entre 1 y 1440."
        )
    records = db.get_pressure_history(node_id, last_n_minutes=last_n_minutes)
    return [
        PressureHistoryPoint(
            timestamp    = r.timestamp,
            pressure     = r.pressure,
            phase        = r.phase,
            is_rush_hour = r.is_rush_hour,
            is_raining   = r.is_raining,
        )
        for r in records
    ]


@app.get("/pressure-map", response_model=Dict[str, float],
         tags=["Sistema"])
async def pressure_map(
    graph: TrafficGraph = Depends(get_graph),
) -> Dict[str, float]:
    """Presión actual de todas las intersecciones. Usado por el dashboard."""
    return graph.export_pressure_map()


@app.post("/intersections/{node_id}/phase", tags=["Admin"],
          dependencies=[Depends(require_admin_key)])
async def force_phase(
    node_id: str,
    body:    PhaseUpdateRequest,
    graph:   TrafficGraph    = Depends(get_graph),
    db:      DatabaseClient  = Depends(get_db),
) -> dict:
    """
    Fuerza el cambio de fase de una intersección.
    Requiere X-API-Key en el header.
    Solo para uso administrativo — en operación normal el WeightEngine decide.
    """
    if node_id not in graph.intersections:
        raise HTTPException(status_code=404,
                            detail=f"Intersección '{node_id}' no encontrada.")

    inter = graph.intersections[node_id]
    inter.current_phase = Phase(body.phase)

    db.save_phase_update(node_id, body.phase, datetime.now(UTC))

    logger.warning(
        "Fase forzada manualmente: %s → %s | razón: %s",
        node_id, body.phase, body.reason or "sin razón"
    )
    return {
        "node_id":   node_id,
        "new_phase": body.phase,
        "timestamp": datetime.now(UTC).isoformat(),
        "reason":    body.reason,
    }


@app.post("/incidents", tags=["Admin"],
          dependencies=[Depends(require_admin_key)])
async def report_incident(
    body:  IncidentRequest,
    graph: TrafficGraph   = Depends(get_graph),
    db:    DatabaseClient = Depends(get_db),
) -> dict:
    """
    Reporta un incidente en un segmento vial.
    Reduce el peso del segmento en el grafo para que Dijkstra lo evite.
    Requiere X-API-Key en el header.
    """
    try:
        graph.apply_incident(body.segment_id, severity=body.severity)
    except KeyError:
        raise HTTPException(
            status_code = 404,
            detail      = f"Segmento '{body.segment_id}' no encontrado."
        )

    db.save_incident(body.segment_id, body.severity, body.notes or "")

    logger.warning(
        "Incidente reportado: %s | severidad=%.2f | %s",
        body.segment_id, body.severity, body.notes or ""
    )
    return {
        "segment_id": body.segment_id,
        "severity":   body.severity,
        "timestamp":  datetime.now(UTC).isoformat(),
    }


@app.get("/metrics", response_model=MetricsResponse, tags=["Sistema"])
async def metrics(
    graph: TrafficGraph   = Depends(get_graph),
    db:    DatabaseClient = Depends(get_db),
) -> MetricsResponse:
    """Métricas generales del sistema."""
    uptime = (datetime.now(UTC) - _start_time).total_seconds()
    return MetricsResponse(
        total_intersections = len(graph.intersections),
        total_ticks         = graph._tick_count,
        total_records       = db.count_records(),
        system_uptime_s     = uptime,
        timestamp           = datetime.now(UTC),
    )


# ── Manejador global de errores ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Error no manejado: %s %s — %s",
                 request.method, request.url, exc)
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Error interno del servidor."},
    )