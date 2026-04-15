"""
tests/test_api.py
------------------
Tests de la API FastAPI con TestClient — sin levantar servidor real.

Ejecutar:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app, _ADMIN_KEY
from core.road import Phase


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def admin_headers():
    return {"X-API-Key": _ADMIN_KEY}


# ── Tests: health ─────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_has_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()

    def test_health_has_timestamp(self, client):
        r = client.get("/health")
        assert "timestamp" in r.json()


# ── Tests: intersections ──────────────────────────────────────────────────────

class TestIntersections:

    def test_list_returns_all(self, client):
        r = client.get("/intersections")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5
        node_ids = {d["node_id"] for d in data}
        assert "A1" in node_ids
        assert "B2" in node_ids

    def test_list_has_required_fields(self, client):
        r = client.get("/intersections")
        inter = r.json()[0]
        for field in ("node_id", "name", "latitude", "longitude",
                      "phase", "pressure", "neighbors"):
            assert field in inter, f"Campo faltante: {field}"

    def test_get_existing_intersection(self, client):
        r = client.get("/intersections/A1")
        assert r.status_code == 200
        assert r.json()["node_id"] == "A1"

    def test_get_nonexistent_returns_404(self, client):
        r = client.get("/intersections/FAKE_NODE")
        assert r.status_code == 404

    def test_intersection_phase_is_valid(self, client):
        r = client.get("/intersections/A1")
        assert r.json()["phase"] in ("green", "yellow", "red")

    def test_intersection_neighbors(self, client):
        r = client.get("/intersections/A1")
        neighbors = r.json()["neighbors"]
        assert isinstance(neighbors, list)
        assert "A2" in neighbors
        assert "B1" in neighbors


# ── Tests: pressure map ───────────────────────────────────────────────────────

class TestPressureMap:

    def test_pressure_map_has_all_nodes(self, client):
        r = client.get("/pressure-map")
        assert r.status_code == 200
        data = r.json()
        assert set(data.keys()) == {"A1", "A2", "A3", "B1", "B2"}

    def test_pressure_values_are_floats(self, client):
        r = client.get("/pressure-map")
        for v in r.json().values():
            assert isinstance(v, (int, float))


# ── Tests: pressure history ───────────────────────────────────────────────────

class TestPressureHistory:

    def test_empty_history(self, client):
        r = client.get("/intersections/A1/pressure-history")
        assert r.status_code == 200
        assert r.json() == []

    def test_invalid_minutes_too_low(self, client):
        r = client.get("/intersections/A1/pressure-history?last_n_minutes=0")
        assert r.status_code == 422

    def test_invalid_minutes_too_high(self, client):
        r = client.get("/intersections/A1/pressure-history?last_n_minutes=9999")
        assert r.status_code == 422


# ── Tests: force phase (admin) ────────────────────────────────────────────────

class TestForcePhase:

    def test_requires_api_key(self, client):
        r = client.post("/intersections/A1/phase",
                        json={"phase": "green"})
        assert r.status_code == 422   # header requerido faltante

    def test_invalid_api_key(self, client):
        r = client.post("/intersections/A1/phase",
                        json={"phase": "green"},
                        headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 403

    def test_force_green(self, client, admin_headers):
        r = client.post("/intersections/A1/phase",
                        json={"phase": "green", "reason": "test"},
                        headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["new_phase"] == "green"

    def test_force_invalid_phase(self, client, admin_headers):
        r = client.post("/intersections/A1/phase",
                        json={"phase": "purple"},
                        headers=admin_headers)
        assert r.status_code == 422

    def test_force_nonexistent_node(self, client, admin_headers):
        r = client.post("/intersections/FAKE/phase",
                        json={"phase": "red"},
                        headers=admin_headers)
        assert r.status_code == 404

    def test_phase_persists_after_force(self, client, admin_headers):
        client.post("/intersections/A2/phase",
                    json={"phase": "green"},
                    headers=admin_headers)
        r = client.get("/intersections/A2")
        assert r.json()["phase"] == "green"


# ── Tests: incidents (admin) ──────────────────────────────────────────────────

class TestIncidents:

    def test_requires_api_key(self, client):
        r = client.post("/incidents",
                        json={"segment_id": "seg-A1-A2", "severity": 0.5})
        assert r.status_code == 422

    def test_report_valid_incident(self, client, admin_headers):
        r = client.post("/incidents",
                        json={"segment_id": "seg-A1-A2",
                              "severity": 0.1,
                              "notes": "Accidente menor"},
                        headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["segment_id"] == "seg-A1-A2"

    def test_invalid_segment_id_characters(self, client, admin_headers):
        r = client.post("/incidents",
                        json={"segment_id": "seg; DROP TABLE--",
                              "severity": 0.5},
                        headers=admin_headers)
        assert r.status_code == 422

    def test_severity_out_of_range(self, client, admin_headers):
        r = client.post("/incidents",
                        json={"segment_id": "seg-A1-A2",
                              "severity": 1.5},
                        headers=admin_headers)
        assert r.status_code == 422

    def test_nonexistent_segment(self, client, admin_headers):
        r = client.post("/incidents",
                        json={"segment_id": "seg-fake",
                              "severity": 0.5},
                        headers=admin_headers)
        assert r.status_code == 404


# ── Tests: metrics ────────────────────────────────────────────────────────────

class TestMetrics:

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["total_intersections"] == 5
        assert "system_uptime_s" in data
        assert "total_records" in data

    def test_uptime_is_positive(self, client):
        r = client.get("/metrics")
        assert r.json()["system_uptime_s"] >= 0