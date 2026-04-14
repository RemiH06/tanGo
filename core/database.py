"""
core/database.py
----------------
Capa de almacenamiento — modelos ORM y cliente de base de datos.

Tecnología:
  - TimescaleDB en producción (PostgreSQL + extensión de series de tiempo).
  - SQLite en memoria para tests — misma interfaz, sin Docker.

Por qué TimescaleDB:
  tanGo genera datos de series de tiempo — cada intersección produce
  un registro cada 5 minutos con su fase, presión y entidades.
  TimescaleDB optimiza exactamente este patrón: inserciones rápidas,
  queries por rango de tiempo, y compresión automática de datos históricos.

Modelos:
  IntersectionRecord  → fase y presión por intersección por timestamp
  EntityRecord        → entidades detectadas en cada ciclo
  IncidentRecord      → incidentes aplicados al grafo

Uso:
    # Producción
    db = DatabaseClient.from_env()

    # Testing
    db = DatabaseClient.in_memory()

    # Escribir
    db.save_snapshot(snapshot, pressure_map)

    # Consultar
    records = db.get_pressure_history("A1", last_n_minutes=60)
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import os
import logging

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Float,
    Integer, Boolean, DateTime, Text,
    Index, event
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from sqlalchemy.engine import Engine

from graph.simulator import SimulationSnapshot

load_dotenv()
logger = logging.getLogger(__name__)

Base = declarative_base()


# ── Modelos ORM ───────────────────────────────────────────────────────────────

class IntersectionRecord(Base):
    """
    Registro de estado de una intersección en un instante.
    Una fila por intersección por ciclo del pipeline.

    En TimescaleDB esta tabla es un hypertable particionado por timestamp.
    """
    __tablename__ = "intersection_records"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, nullable=False, index=True)
    node_id       = Column(String(64), nullable=False, index=True)
    node_name     = Column(String(255), nullable=False)
    phase         = Column(String(16), nullable=False)   # green/yellow/red
    pressure      = Column(Float, nullable=False, default=0.0)
    tick_number   = Column(Integer, nullable=False)
    is_rush_hour  = Column(Boolean, nullable=False, default=False)
    is_weekend    = Column(Boolean, nullable=False, default=False)
    is_late_night = Column(Boolean, nullable=False, default=False)
    is_raining    = Column(Boolean, nullable=False, default=False)
    temperature_c = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_intersection_ts_node", "timestamp", "node_id"),
    )

    def __repr__(self) -> str:
        return (
            f"IntersectionRecord(node={self.node_id}, "
            f"phase={self.phase}, pressure={self.pressure:.1f}, "
            f"ts={self.timestamp})"
        )


class EntityRecord(Base):
    """
    Registro de entidades detectadas en una intersección en un ciclo.
    Agregado por tipo — no una fila por entidad individual.
    """
    __tablename__ = "entity_records"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime, nullable=False, index=True)
    node_id          = Column(String(64), nullable=False, index=True)
    tick_number      = Column(Integer, nullable=False)
    n_cars           = Column(Integer, nullable=False, default=0)
    n_buses          = Column(Integer, nullable=False, default=0)
    n_trucks         = Column(Integer, nullable=False, default=0)
    n_motorcycles    = Column(Integer, nullable=False, default=0)
    n_bicycles       = Column(Integer, nullable=False, default=0)
    n_pedestrians    = Column(Integer, nullable=False, default=0)
    n_wheelchairs    = Column(Integer, nullable=False, default=0)
    n_emergency      = Column(Integer, nullable=False, default=0)
    total_entities   = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_entity_ts_node", "timestamp", "node_id"),
    )


class IncidentRecord(Base):
    """
    Registro de incidentes aplicados a segmentos viales.
    """
    __tablename__ = "incident_records"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, nullable=False, default=datetime.utcnow)
    segment_id = Column(String(64), nullable=False, index=True)
    severity   = Column(Float, nullable=False, default=0.1)
    resolved   = Column(Boolean, nullable=False, default=False)
    notes      = Column(Text, nullable=True)


# ── Cliente de base de datos ──────────────────────────────────────────────────

class DatabaseClient:
    """
    Cliente de base de datos para tanGo.
    Abstrae la conexión a TimescaleDB (producción) o SQLite (tests).

    No instanciar directamente — usar los factory methods:
        DatabaseClient.from_env()    → producción con TimescaleDB
        DatabaseClient.in_memory()  → tests con SQLite en memoria
    """

    def __init__(self, engine: Engine) -> None:
        self._engine        = engine
        self._SessionFactory = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        logger.info("DatabaseClient listo: %s", engine.url)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "DatabaseClient":
        """
        Crea un cliente conectado a TimescaleDB usando variables de entorno.

        Variables requeridas en .env:
            TIMESCALE_URI=postgresql://user:password@localhost:5432/tango

        Returns
        -------
        DatabaseClient conectado a TimescaleDB.

        Raises
        ------
        EnvironmentError si TIMESCALE_URI no está configurada.
        """
        uri = os.getenv("TIMESCALE_URI", "").strip()
        if not uri:
            raise EnvironmentError(
                "TIMESCALE_URI no encontrada en .env. "
                "Formato: postgresql://user:password@localhost:5432/tango"
            )
        engine = create_engine(uri, pool_pre_ping=True, echo=False)
        client = cls(engine)

        # Crear hypertable en TimescaleDB si no existe
        client._setup_timescaledb()
        return client

    @classmethod
    def in_memory(cls) -> "DatabaseClient":
        """
        Crea un cliente con SQLite en memoria para tests.
        No requiere ninguna configuración ni Docker.

        Returns
        -------
        DatabaseClient con SQLite en memoria.
        """
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        return cls(engine)

    # ── Setup TimescaleDB ─────────────────────────────────────────────────────

    def _setup_timescaledb(self) -> None:
        """
        Convierte intersection_records y entity_records en hypertables
        de TimescaleDB para optimizar queries de series de tiempo.
        Solo se ejecuta si la extensión timescaledb está disponible.
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(
                    "SELECT create_hypertable("
                    "'intersection_records', 'timestamp', "
                    "if_not_exists => TRUE)"
                )
                conn.execute(
                    "SELECT create_hypertable("
                    "'entity_records', 'timestamp', "
                    "if_not_exists => TRUE)"
                )
                conn.commit()
            logger.info("TimescaleDB hypertables configurados.")
        except Exception as e:
            logger.warning(
                "TimescaleDB no disponible (%s) — usando PostgreSQL estándar.", e
            )

    # ── Escritura ─────────────────────────────────────────────────────────────

    def save_snapshot(self, snapshot: SimulationSnapshot,
                      pressure_map: Dict[str, float],
                      intersection_names: Dict[str, str],
                      ctx_kwargs: Optional[dict] = None) -> None:
        """
        Persiste un SimulationSnapshot completo en la base de datos.
        Una llamada por ciclo del pipeline.

        Parameters
        ----------
        snapshot            : Snapshot del tick actual.
        pressure_map        : Dict node_id → presión calculada por WeightEngine.
        intersection_names  : Dict node_id → nombre legible.
        ctx_kwargs          : Dict con campos del TrafficContext (opcional).
        """
        ctx = ctx_kwargs or {}

        with Session(self._engine) as session:
            for node_id, entities in snapshot.entities.items():
                # ── IntersectionRecord ──
                session.add(IntersectionRecord(
                    timestamp     = snapshot.timestamp,
                    node_id       = node_id,
                    node_name     = intersection_names.get(node_id, node_id),
                    phase         = "red",   # será actualizado por adjust_phase
                    pressure      = pressure_map.get(node_id, 0.0),
                    tick_number   = snapshot.tick_number,
                    is_rush_hour  = ctx.get("is_rush_hour",  False),
                    is_weekend    = ctx.get("is_weekend",    False),
                    is_late_night = ctx.get("is_late_night", False),
                    is_raining    = ctx.get("is_raining",    False),
                    temperature_c = ctx.get("temperature_c", None),
                ))

                # ── EntityRecord — contar por tipo ──
                from core.entities import Vehicle, Pedestrian, VehicleType
                counts = {
                    "n_cars":        0,
                    "n_buses":       0,
                    "n_trucks":      0,
                    "n_motorcycles": 0,
                    "n_bicycles":    0,
                    "n_pedestrians": 0,
                    "n_wheelchairs": 0,
                    "n_emergency":   0,
                }
                for e in entities:
                    if isinstance(e, Vehicle):
                        mapping = {
                            VehicleType.CAR:        "n_cars",
                            VehicleType.BUS:        "n_buses",
                            VehicleType.TRUCK:      "n_trucks",
                            VehicleType.MOTORCYCLE: "n_motorcycles",
                            VehicleType.BICYCLE:    "n_bicycles",
                            VehicleType.EMERGENCY:  "n_emergency",
                        }
                        key = mapping.get(e.vehicle_type)
                        if key:
                            counts[key] += 1
                    elif isinstance(e, Pedestrian):
                        counts["n_pedestrians"] += 1
                        if e.is_wheelchair:
                            counts["n_wheelchairs"] += 1

                session.add(EntityRecord(
                    timestamp      = snapshot.timestamp,
                    node_id        = node_id,
                    tick_number    = snapshot.tick_number,
                    total_entities = len(entities),
                    **counts,
                ))

            session.commit()
            logger.debug(
                "Snapshot tick #%d guardado — %d intersecciones",
                snapshot.tick_number, len(snapshot.entities)
            )

    def save_phase_update(self, node_id: str, phase: str,
                          timestamp: datetime) -> None:
        """
        Actualiza la fase del registro más reciente de una intersección.
        Llamar después de adjust_phase() para reflejar el cambio real.

        Parameters
        ----------
        node_id   : ID de la intersección.
        phase     : Nueva fase ("green", "yellow", "red").
        timestamp : Timestamp del ciclo actual.
        """
        with Session(self._engine) as session:
            record = (
                session.query(IntersectionRecord)
                .filter(
                    IntersectionRecord.node_id == node_id,
                    IntersectionRecord.timestamp == timestamp,
                )
                .first()
            )
            if record:
                record.phase = phase
                session.commit()

    def save_incident(self, segment_id: str, severity: float,
                      notes: str = "") -> None:
        """
        Registra un incidente en la base de datos.

        Parameters
        ----------
        segment_id : ID del segmento afectado.
        severity   : Factor de severidad (0.0–1.0).
        notes      : Descripción opcional del incidente.
        """
        with Session(self._engine) as session:
            session.add(IncidentRecord(
                segment_id = segment_id,
                severity   = severity,
                notes      = notes,
            ))
            session.commit()

    # ── Consultas ─────────────────────────────────────────────────────────────

    def get_pressure_history(self, node_id: str,
                             last_n_minutes: int = 60) -> List[IntersectionRecord]:
        """
        Historial de presión de una intersección en los últimos N minutos.
        Usado por el dashboard para graficar la evolución de la presión.

        Parameters
        ----------
        node_id         : ID de la intersección.
        last_n_minutes  : Ventana de tiempo a consultar.

        Returns
        -------
        Lista de IntersectionRecord ordenados por timestamp ascendente.
        """
        since = datetime.utcnow() - timedelta(minutes=last_n_minutes)
        with Session(self._engine) as session:
            return (
                session.query(IntersectionRecord)
                .filter(
                    IntersectionRecord.node_id  == node_id,
                    IntersectionRecord.timestamp >= since,
                )
                .order_by(IntersectionRecord.timestamp.asc())
                .all()
            )

    def get_latest_pressure_map(self) -> Dict[str, float]:
        """
        Presión más reciente de cada intersección.
        Usado por el dashboard para colorear el mapa en tiempo real.

        Returns
        -------
        Dict node_id → presión del último registro.
        """
        with Session(self._engine) as session:
            # Subconsulta: máximo timestamp por node_id
            from sqlalchemy import func
            subq = (
                session.query(
                    IntersectionRecord.node_id,
                    func.max(IntersectionRecord.timestamp).label("max_ts"),
                )
                .group_by(IntersectionRecord.node_id)
                .subquery()
            )
            records = (
                session.query(IntersectionRecord)
                .join(
                    subq,
                    (IntersectionRecord.node_id  == subq.c.node_id) &
                    (IntersectionRecord.timestamp == subq.c.max_ts),
                )
                .all()
            )
            return {r.node_id: r.pressure for r in records}

    def get_phase_distribution(self, node_id: str,
                               last_n_minutes: int = 60) -> Dict[str, int]:
        """
        Distribución de fases de una intersección en los últimos N minutos.
        Útil para el reporte: cuánto tiempo estuvo en verde vs rojo.

        Returns
        -------
        Dict phase → conteo de registros en esa fase.
        """
        since = datetime.utcnow() - timedelta(minutes=last_n_minutes)
        with Session(self._engine) as session:
            from sqlalchemy import func
            rows = (
                session.query(
                    IntersectionRecord.phase,
                    func.count(IntersectionRecord.id).label("count"),
                )
                .filter(
                    IntersectionRecord.node_id  == node_id,
                    IntersectionRecord.timestamp >= since,
                )
                .group_by(IntersectionRecord.phase)
                .all()
            )
            return {row.phase: row.count for row in rows}

    def count_records(self) -> int:
        """Total de registros en intersection_records. Útil para tests."""
        with Session(self._engine) as session:
            return session.query(IntersectionRecord).count()