"""
tests/test_tomtom_ingester.py
------------------------------
Tests del TomTomIngester.

Todos los tests mockean la llamada HTTP — nunca se llama a TomTom real.
Esto garantiza que el CI funciona sin API key y sin red.

Ejecutar:
    pytest tests/test_tomtom_ingester.py -v
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from ingest.tomtom import TomTomIngester
from ingest.base import TrafficSnapshot
from safety.circuit_breaker import CircuitBreaker
from core.context import TrafficContext


# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_KEY = "a" * 32   # key falsa de 32 chars para tests

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Inyecta una API key falsa para todos los tests."""
    monkeypatch.setenv("TOMTOM_API_KEY", FAKE_KEY)

@pytest.fixture
def breaker() -> CircuitBreaker:
    return CircuitBreaker(name="tomtom-test", failure_threshold=3)

@pytest.fixture
def ingester(breaker) -> TomTomIngester:
    return TomTomIngester(circuit_breaker=breaker)

@pytest.fixture
def ctx() -> TrafficContext:
    return TrafficContext(
        timestamp      = datetime(2024, 3, 4, 8, 0),
        temperature_c  = 22.0,
        is_raining     = False,
        wind_speed_kmh = 10.0,
        visibility_m   = 10000.0,
        is_weekend     = False,
        is_rush_hour   = True,
        is_late_night  = False,
    )

# Respuesta simulada de TomTom
TOMTOM_RESPONSE = {
    "flowSegmentData": {
        "currentSpeed":  30.0,
        "freeFlowSpeed": 60.0,
        "confidence":    0.95,
        "coordinates": {
            "coordinate": [
                {"latitude": 20.6597, "longitude": -103.3496}
            ]
        }
    }
}


# ── Tests: inicialización ─────────────────────────────────────────────────────

class TestInit:

    def test_raises_without_api_key(self, monkeypatch, breaker):
        monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="TOMTOM_API_KEY"):
            TomTomIngester(circuit_breaker=breaker)

    def test_raises_with_wrong_key_length(self, monkeypatch, breaker):
        monkeypatch.setenv("TOMTOM_API_KEY", "short")
        with pytest.raises(ValueError, match="32 caracteres"):
            TomTomIngester(circuit_breaker=breaker)

    def test_creates_successfully_with_valid_key(self, ingester):
        assert ingester is not None


# ── Tests: parse() ────────────────────────────────────────────────────────────

class TestParse:

    def test_parse_returns_list(self, ingester):
        result = ingester.parse(TOMTOM_RESPONSE)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_parse_congestion_calculation(self, ingester):
        """congestion = 1 - (30 / 60) = 0.5"""
        result = ingester.parse(TOMTOM_RESPONSE)
        assert result[0].congestion_index == pytest.approx(0.5)

    def test_parse_speed_fields(self, ingester):
        result = ingester.parse(TOMTOM_RESPONSE)
        snap = result[0]
        assert snap.current_speed_kmh == 30.0
        assert snap.free_flow_kmh     == 60.0
        assert snap.confidence        == 0.95

    def test_parse_free_flow_zero_does_not_crash(self, ingester):
        """Si freeFlowSpeed es 0, no debe dividir por cero."""
        raw = {
            "flowSegmentData": {
                "currentSpeed":  0.0,
                "freeFlowSpeed": 0.0,
                "confidence":    0.5,
                "coordinates":   {"coordinate": []}
            }
        }
        result = ingester.parse(raw)
        assert 0.0 <= result[0].congestion_index <= 1.0

    def test_parse_congestion_clamped_to_one(self, ingester):
        """Si currentSpeed > freeFlowSpeed (dato raro), congestión = 0."""
        raw = {
            "flowSegmentData": {
                "currentSpeed":  100.0,
                "freeFlowSpeed": 60.0,
                "confidence":    0.8,
                "coordinates":   {"coordinate": []}
            }
        }
        result = ingester.parse(raw)
        assert result[0].congestion_index == 0.0

    def test_parse_raises_on_missing_key(self, ingester):
        with pytest.raises((KeyError, ValueError)):
            ingester.parse({"unexpected": "format"})


# ── Tests: fetch() con mock ───────────────────────────────────────────────────

class TestFetch:

    def test_fetch_calls_tomtom_endpoint(self, ingester, ctx):
        mock_response = MagicMock()
        mock_response.json.return_value = TOMTOM_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(ingester.client, "get",
                          new_callable=AsyncMock,
                          return_value=mock_response) as mock_get:
            result = asyncio.get_event_loop().run_until_complete(
                ingester.fetch(ctx)
            )
            assert mock_get.called
            # Verificar que la API key NO está en la URL sino en params
            call_kwargs = mock_get.call_args
            params = call_kwargs.kwargs.get("params", {})
            assert "key" in params
            assert params["key"] == FAKE_KEY

    def test_fetch_raises_on_http_error(self, ingester, ctx):
        import httpx
        with patch.object(ingester.client, "get",
                          new_callable=AsyncMock,
                          side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(httpx.TimeoutException):
                asyncio.get_event_loop().run_until_complete(
                    ingester.fetch(ctx)
                )


# ── Tests: fallback() ─────────────────────────────────────────────────────────

class TestFallback:

    def test_fallback_returns_list(self, ingester):
        result = ingester.fallback()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_fallback_confidence_is_zero(self, ingester):
        """confidence=0 indica al sistema que son datos de fallback."""
        result = ingester.fallback()
        assert result[0].confidence == 0.0

    def test_fallback_congestion_is_neutral(self, ingester):
        """El fallback no debe sesgar el sistema hacia ningún extremo."""
        result = ingester.fallback()
        assert result[0].congestion_index == pytest.approx(0.5)


# ── Tests: coordenadas ────────────────────────────────────────────────────────

class TestCoordinates:

    def test_invalid_latitude_raises(self, ctx):
        bad_ctx = MagicMock()
        bad_ctx.latitude  = 999.0
        bad_ctx.longitude = -103.3496
        with pytest.raises(ValueError, match="Latitud"):
            TomTomIngester._sanitize_coordinates(bad_ctx)

    def test_invalid_longitude_raises(self, ctx):
        bad_ctx = MagicMock()
        bad_ctx.latitude  = 20.6597
        bad_ctx.longitude = 999.0
        with pytest.raises(ValueError, match="Longitud"):
            TomTomIngester._sanitize_coordinates(bad_ctx)