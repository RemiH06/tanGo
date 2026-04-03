"""
tests/test_circuit_breaker.py
------------------------------
Tests del CircuitBreaker y DataIngester.fetch_safe().

Ejecutar:
    pytest tests/test_circuit_breaker.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from safety.circuit_breaker import CircuitBreaker, CircuitState
from ingest.base import DataIngester, TrafficSnapshot, WeatherSnapshot
from core.context import TrafficContext


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def breaker() -> CircuitBreaker:
    return CircuitBreaker(name="test-service", failure_threshold=3, recovery_timeout=30)


@pytest.fixture
def ctx() -> TrafficContext:
    return TrafficContext(
        timestamp       = datetime(2024, 3, 4, 8, 0),
        temperature_c   = 22.0,
        is_raining      = False,
        wind_speed_kmh  = 10.0,
        visibility_m    = 10000.0,
        is_weekend      = False,
        is_rush_hour    = True,
        is_late_night   = False,
    )


# ── Tests: CircuitBreaker ─────────────────────────────────────────────────────

class TestCircuitBreaker:

    def test_initial_state_is_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED

    def test_successful_call_stays_closed(self, breaker):
        result = breaker.call(func=lambda: "ok", fallback=lambda: "fallback")
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    def test_single_failure_stays_closed(self, breaker):
        def bad(): raise ConnectionError("timeout")
        result = breaker.call(func=bad, fallback=lambda: "fallback")
        assert result == "fallback"
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 1

    def test_opens_after_threshold_failures(self, breaker):
        def bad(): raise ConnectionError("timeout")
        for _ in range(3):
            breaker.call(func=bad, fallback=lambda: "fallback")
        assert breaker.state == CircuitState.OPEN

    def test_open_circuit_uses_fallback_without_calling_func(self, breaker):
        """Con el circuito abierto, func() nunca debe llamarse."""
        def bad(): raise ConnectionError("no debería llamarse")
        # Abrir el circuito
        for _ in range(3):
            breaker.call(func=bad, fallback=lambda: "fallback")

        called = []
        def should_not_be_called():
            called.append(True)
            return "real"

        result = breaker.call(func=should_not_be_called, fallback=lambda: "fallback")
        assert result == "fallback"
        assert not called, "func() no debería haberse llamado con el circuito abierto"

    def test_success_resets_failure_count(self, breaker):
        def bad(): raise ConnectionError()
        breaker.call(func=bad, fallback=lambda: "x")
        breaker.call(func=bad, fallback=lambda: "x")
        assert breaker._failure_count == 2

        # Una llamada exitosa resetea el contador
        breaker.call(func=lambda: "ok", fallback=lambda: "x")
        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self, breaker):
        """Simula que pasó el tiempo de recuperación."""
        def bad(): raise ConnectionError()
        for _ in range(3):
            breaker.call(func=bad, fallback=lambda: "x")
        assert breaker.state == CircuitState.OPEN

        # Simular que pasó el tiempo
        breaker._last_failure = datetime.now() - timedelta(seconds=31)
        result = breaker.call(func=lambda: "recovered", fallback=lambda: "x")
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED


# ── Tests: TrafficSnapshot validaciones ──────────────────────────────────────

class TestTrafficSnapshot:

    def test_valid_snapshot(self):
        s = TrafficSnapshot("seg-1", 0.5, 30.0, 60.0, 0.9)
        assert s.congestion_index == 0.5

    def test_invalid_congestion_raises(self):
        with pytest.raises(ValueError, match="congestion_index"):
            TrafficSnapshot("seg-1", 1.5, 30.0, 60.0, 0.9)

    def test_negative_speed_raises(self):
        with pytest.raises(ValueError, match="current_speed_kmh"):
            TrafficSnapshot("seg-1", 0.5, -10.0, 60.0, 0.9)

    def test_zero_free_flow_raises(self):
        with pytest.raises(ValueError, match="free_flow_kmh"):
            TrafficSnapshot("seg-1", 0.5, 30.0, 0.0, 0.9)


# ── Tests: DataIngester.fetch_safe() ─────────────────────────────────────────

class TestFetchSafe:
    """
    Prueba fetch_safe() con un ingester concreto mínimo.
    """

    def _make_ingester(self, breaker: CircuitBreaker,
                       should_fail: bool = False) -> DataIngester:
        """Crea un ingester concreto de prueba."""

        class DummyIngester(DataIngester):
            async def fetch(self, ctx):
                if should_fail:
                    raise ConnectionError("API no disponible")
                return {"value": 42}

            def parse(self, raw):
                return raw["value"]

            def fallback(self):
                return -1

        return DummyIngester(breaker)

    def test_fetch_safe_returns_parsed_data(self, breaker, ctx):
        ingester = self._make_ingester(breaker, should_fail=False)
        result = asyncio.get_event_loop().run_until_complete(
            ingester.fetch_safe(ctx)
        )
        assert result == 42

    def test_fetch_safe_returns_fallback_on_error(self, breaker, ctx):
        ingester = self._make_ingester(breaker, should_fail=True)
        result = asyncio.get_event_loop().run_until_complete(
            ingester.fetch_safe(ctx)
        )
        assert result == -1

    def test_fetch_safe_opens_circuit_after_repeated_failures(self, breaker, ctx):
        ingester = self._make_ingester(breaker, should_fail=True)
        for _ in range(3):
            asyncio.get_event_loop().run_until_complete(ingester.fetch_safe(ctx))
        assert breaker.state == CircuitState.OPEN