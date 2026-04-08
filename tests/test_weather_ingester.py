"""
tests/test_weather_ingester.py
--------------------------------
Tests del WeatherIngester — unitarios con mocks y prueba en vivo.

Ejecutar solo unitarios:
    pytest tests/test_weather_ingester.py -v

Ejecutar prueba en vivo:
    python tests/test_weather_ingester.py
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from ingest.weather import WeatherIngester
from ingest.base import WeatherSnapshot
from safety.circuit_breaker import CircuitBreaker
from core.context import TrafficContext


# ── Fixtures ──────────────────────────────────────────────────────────────────

GDL_LAT =  20.6597
GDL_LON = -103.3496

@pytest.fixture
def breaker() -> CircuitBreaker:
    return CircuitBreaker(name="weather-test", failure_threshold=3)

@pytest.fixture
def ingester(breaker) -> WeatherIngester:
    return WeatherIngester(GDL_LAT, GDL_LON, circuit_breaker=breaker)

@pytest.fixture
def ctx() -> TrafficContext:
    return TrafficContext.build(
        timestamp      = datetime(2024, 3, 4, 8, 0),
        temperature_c  = 20.0,
        is_raining     = False,
        wind_speed_kmh = 10.0,
        visibility_m   = 10000.0,
    )

# Respuesta simulada de Open-Meteo
OPEN_METEO_RESPONSE = {
    "current": {
        "temperature_2m": 18.5,
        "precipitation":  2.4,
        "windspeed_10m":  15.0,
        "visibility":     4000.0,
    }
}

OPEN_METEO_CLEAR = {
    "current": {
        "temperature_2m": 24.0,
        "precipitation":  0.0,
        "windspeed_10m":  8.0,
        "visibility":     10000.0,
    }
}


# ── Tests: inicialización ─────────────────────────────────────────────────────

class TestInit:

    def test_invalid_latitude_raises(self, breaker):
        with pytest.raises(ValueError, match="Latitud"):
            WeatherIngester(999.0, GDL_LON, circuit_breaker=breaker)

    def test_invalid_longitude_raises(self, breaker):
        with pytest.raises(ValueError, match="Longitud"):
            WeatherIngester(GDL_LAT, 999.0, circuit_breaker=breaker)

    def test_creates_successfully(self, ingester):
        assert ingester.latitude  == GDL_LAT
        assert ingester.longitude == GDL_LON


# ── Tests: parse() ────────────────────────────────────────────────────────────

class TestParse:

    def test_parse_lluvia(self, ingester):
        """precipitation > 0 → is_raining = True"""
        snap = ingester.parse(OPEN_METEO_RESPONSE)
        assert snap.is_raining is True

    def test_parse_sin_lluvia(self, ingester):
        snap = ingester.parse(OPEN_METEO_CLEAR)
        assert snap.is_raining is False

    def test_parse_temperatura(self, ingester):
        snap = ingester.parse(OPEN_METEO_RESPONSE)
        assert snap.temperature_c == pytest.approx(18.5)

    def test_parse_viento(self, ingester):
        snap = ingester.parse(OPEN_METEO_RESPONSE)
        assert snap.wind_speed_kmh == pytest.approx(15.0)

    def test_parse_visibilidad(self, ingester):
        snap = ingester.parse(OPEN_METEO_RESPONSE)
        assert snap.visibility_m == pytest.approx(4000.0)

    def test_parse_visibilidad_negativa_se_clampea(self, ingester):
        raw = {"current": {
            "temperature_2m": 20.0,
            "precipitation":  0.0,
            "windspeed_10m":  5.0,
            "visibility":     -100.0,
        }}
        snap = ingester.parse(raw)
        assert snap.visibility_m == 0.0

    def test_parse_campos_faltantes_usa_defaults(self, ingester):
        """Si Open-Meteo no devuelve un campo, no debe explotar."""
        snap = ingester.parse({"current": {}})
        assert isinstance(snap, WeatherSnapshot)
        assert snap.is_raining is False


# ── Tests: fallback() ─────────────────────────────────────────────────────────

class TestFallback:

    def test_fallback_retorna_weather_snapshot(self, ingester):
        snap = ingester.fallback()
        assert isinstance(snap, WeatherSnapshot)

    def test_fallback_no_llueve(self, ingester):
        assert ingester.fallback().is_raining is False

    def test_fallback_buena_visibilidad(self, ingester):
        assert ingester.fallback().visibility_m == 10000.0


# ── Tests: to_context_kwargs() ────────────────────────────────────────────────

class TestToContextKwargs:

    def test_retorna_campos_correctos(self, ingester):
        snap   = ingester.parse(OPEN_METEO_RESPONSE)
        kwargs = ingester.to_context_kwargs(snap)
        assert set(kwargs.keys()) == {
            "temperature_c", "is_raining", "wind_speed_kmh", "visibility_m"
        }

    def test_kwargs_construyen_contexto_valido(self, ingester):
        """to_context_kwargs() debe poder pasarse directo a TrafficContext.build()"""
        snap   = ingester.parse(OPEN_METEO_RESPONSE)
        kwargs = ingester.to_context_kwargs(snap)
        ctx    = TrafficContext.build(timestamp=datetime.now(), **kwargs)
        assert ctx.is_raining is True
        assert ctx.temperature_c == pytest.approx(18.5)


# ── Tests: fetch() con mock ───────────────────────────────────────────────────

class TestFetch:

    def test_fetch_llama_open_meteo(self, ingester, ctx):
        mock_response = MagicMock()
        mock_response.json.return_value = OPEN_METEO_RESPONSE
        mock_response.raise_for_status   = MagicMock()

        with patch.object(ingester.client, "get",
                          new_callable=AsyncMock,
                          return_value=mock_response):
            raw = asyncio.run(ingester.fetch(ctx))
            assert "current" in raw

    def test_fetch_falla_usa_fallback_via_fetch_safe(self, ingester, ctx):
        import httpx
        with patch.object(ingester.client, "get",
                          new_callable=AsyncMock,
                          side_effect=httpx.TimeoutException("timeout")):
            snap = asyncio.run(ingester.fetch_safe(ctx))
            assert isinstance(snap, WeatherSnapshot)
            assert snap.is_raining is False   # fallback neutro


# ── Prueba en vivo (no corre con pytest) ──────────────────────────────────────

async def live_test():
    import traceback
    print("=" * 55)
    print("  tanGo — prueba en vivo Open-Meteo")
    print("=" * 55)

    breaker  = CircuitBreaker(name="weather-live", failure_threshold=3)
    ingester = WeatherIngester(GDL_LAT, GDL_LON, circuit_breaker=breaker)

    ctx = TrafficContext.build(
        timestamp      = datetime.now(),
        temperature_c  = 20.0,
        is_raining     = False,
        wind_speed_kmh = 0.0,
        visibility_m   = 10000.0,
    )

    print(f"\n  Consultando clima en Guadalajara...")
    print(f"  Hora local : {ctx.timestamp.strftime('%H:%M:%S')}")

    try:
        raw  = await ingester.fetch(ctx)
        snap = ingester.parse(raw)

        # Construir nuevo contexto con datos reales
        ctx_real = TrafficContext.build(
            timestamp = datetime.now(),
            **ingester.to_context_kwargs(snap)
        )

        print(f"\n  Temperatura  : {snap.temperature_c:.1f}°C")
        print(f"  Lloviendo    : {'Sí' if snap.is_raining else 'No'}")
        print(f"  Viento       : {snap.wind_speed_kmh:.1f} km/h")
        print(f"  Visibilidad  : {snap.visibility_m:.0f} m")
        print(f"\n  Contexto resultante:")
        print(f"  Hora pico    : {ctx_real.is_rush_hour}")
        print(f"  Fin de semana: {ctx_real.is_weekend}")
        print(f"  Madrugada    : {ctx_real.is_late_night}")
        print(f"\n  CircuitBreaker: {breaker.state.name}")
        print("  Prueba completada exitosamente.")

    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()

    finally:
        await ingester.close()

    print("=" * 55)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    asyncio.run(live_test())