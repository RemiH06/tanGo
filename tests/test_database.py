"""
tests/test_database.py
-----------------------
Tests de la capa de base de datos.

Todos usan DatabaseClient.in_memory() — sin TimescaleDB, sin Docker.
La interfaz es idéntica a producción.

Ejecutar:
    pytest tests/test_database.py -v
"""

import pytest
from datetime import datetime

from core.database import DatabaseClient, IntersectionRecord, EntityRecord
from core.context import TrafficContext
from core.entities import Vehicle, Pedestrian, VehicleType, Direction
from graph.simulator import TrafficGraph, SimulationSnapshot


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db() -> DatabaseClient:
    return DatabaseClient.in_memory()

@pytest.fixture
def graph() -> TrafficGraph:
    g = TrafficGraph()
    g.build_sample_city()
    return g

@pytest.fixture
def ctx() -> TrafficContext:
    return TrafficContext.build(
        timestamp      = datetime(2024, 3, 4, 8, 0),
        temperature_c  = 22.0,
        is_raining     = False,
        wind_speed_kmh = 10.0,
        visibility_m   = 10000.0,
    )

@pytest.fixture
def snapshot(graph, ctx) -> SimulationSnapshot:
    return graph.tick(ctx)

@pytest.fixture
def pressure_map(graph) -> dict:
    return {node_id: float(i * 10)
            for i, node_id in enumerate(graph.intersections)}

@pytest.fixture
def intersection_names(graph) -> dict:
    return {node_id: inter.name
            for node_id, inter in graph.intersections.items()}

@pytest.fixture
def ctx_kwargs(ctx) -> dict:
    return {
        "is_rush_hour":  ctx.is_rush_hour,
        "is_weekend":    ctx.is_weekend,
        "is_late_night": ctx.is_late_night,
        "is_raining":    ctx.is_raining,
        "temperature_c": ctx.temperature_c,
    }


# ── Tests: inicialización ─────────────────────────────────────────────────────

class TestInit:

    def test_in_memory_creates_tables(self, db):
        """Las tablas deben existir después de crear el cliente."""
        assert db.count_records() == 0

    def test_from_env_raises_without_uri(self, monkeypatch):
        monkeypatch.delenv("TIMESCALE_URI", raising=False)
        with pytest.raises(EnvironmentError, match="TIMESCALE_URI"):
            DatabaseClient.from_env()


# ── Tests: save_snapshot ──────────────────────────────────────────────────────

class TestSaveSnapshot:

    def test_saves_correct_number_of_records(self, db, snapshot,
                                              pressure_map,
                                              intersection_names,
                                              ctx_kwargs):
        db.save_snapshot(snapshot, pressure_map,
                         intersection_names, ctx_kwargs)
        # 5 intersecciones → 5 IntersectionRecords
        assert db.count_records() == 5

    def test_pressure_saved_correctly(self, db, snapshot,
                                       pressure_map, intersection_names,
                                       ctx_kwargs):
        db.save_snapshot(snapshot, pressure_map,
                         intersection_names, ctx_kwargs)
        history = db.get_pressure_history("A1", last_n_minutes=60)
        assert len(history) == 1
        assert history[0].pressure == pressure_map["A1"]

    def test_context_flags_saved(self, db, snapshot,
                                  pressure_map, intersection_names,
                                  ctx_kwargs):
        db.save_snapshot(snapshot, pressure_map,
                         intersection_names, ctx_kwargs)
        history = db.get_pressure_history("A1", last_n_minutes=60)
        record = history[0]
        assert record.is_rush_hour  == ctx_kwargs["is_rush_hour"]
        assert record.is_weekend    == ctx_kwargs["is_weekend"]
        assert record.is_raining    == ctx_kwargs["is_raining"]
        assert record.temperature_c == ctx_kwargs["temperature_c"]

    def test_multiple_ticks_accumulate(self, db, graph, ctx,
                                        intersection_names):
        for _ in range(3):
            snap = graph.tick(ctx)
            pm   = graph.export_pressure_map()
            db.save_snapshot(snap, pm, intersection_names)
        # 3 ticks × 5 intersecciones = 15 registros
        assert db.count_records() == 15

    def test_entity_records_saved(self, db, snapshot,
                                   pressure_map, intersection_names,
                                   ctx_kwargs):
        db.save_snapshot(snapshot, pressure_map,
                         intersection_names, ctx_kwargs)
        from sqlalchemy.orm import Session
        with Session(db._engine) as session:
            count = session.query(EntityRecord).count()
        assert count == 5   # una fila por intersección


# ── Tests: save_phase_update ──────────────────────────────────────────────────

class TestSavePhaseUpdate:

    def test_updates_phase(self, db, snapshot, pressure_map,
                            intersection_names, ctx_kwargs):
        db.save_snapshot(snapshot, pressure_map,
                         intersection_names, ctx_kwargs)
        db.save_phase_update("A1", "green", snapshot.timestamp)

        history = db.get_pressure_history("A1", last_n_minutes=60)
        assert history[0].phase == "green"

    def test_update_nonexistent_does_not_crash(self, db):
        """Si no hay registro para ese timestamp, no debe explotar."""
        db.save_phase_update("FAKE", "green", datetime.utcnow())


# ── Tests: save_incident ──────────────────────────────────────────────────────

class TestSaveIncident:

    def test_saves_incident(self, db):
        db.save_incident("seg-A1-A2", severity=0.1, notes="Accidente")
        from sqlalchemy.orm import Session
        from core.database import IncidentRecord
        with Session(db._engine) as session:
            count = session.query(IncidentRecord).count()
        assert count == 1


# ── Tests: get_latest_pressure_map ───────────────────────────────────────────

class TestLatestPressureMap:

    def test_returns_all_nodes(self, db, graph, ctx, intersection_names):
        snap = graph.tick(ctx)
        pm   = graph.export_pressure_map()
        db.save_snapshot(snap, pm, intersection_names)

        latest = db.get_latest_pressure_map()
        assert set(latest.keys()) == set(graph.intersections.keys())

    def test_returns_latest_values(self, db, graph, ctx, intersection_names):
        """El segundo tick debe sobreescribir los valores del primero."""
        for _ in range(2):
            snap = graph.tick(ctx)
            pm   = {nid: float(snap.tick_number * 10)
                    for nid in graph.intersections}
            db.save_snapshot(snap, pm, intersection_names)

        latest = db.get_latest_pressure_map()
        # El tick_number del último snapshot fue 2 → presión = 20.0
        assert all(v == 20.0 for v in latest.values())


# ── Tests: get_phase_distribution ────────────────────────────────────────────

class TestPhaseDistribution:

    def test_distribution_after_saves(self, db, snapshot,
                                       pressure_map, intersection_names):
        db.save_snapshot(snapshot, pressure_map, intersection_names)
        db.save_phase_update("A1", "green", snapshot.timestamp)

        dist = db.get_phase_distribution("A1", last_n_minutes=60)
        assert "green" in dist
        assert dist["green"] == 1

    def test_empty_distribution_for_new_node(self, db):
        dist = db.get_phase_distribution("NODO_NUEVO", last_n_minutes=60)
        assert dist == {}