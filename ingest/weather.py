"""
ingest/weather.py
-----------------
Implementación de DataIngester para Open-Meteo API.

Endpoint:
    GET https://api.open-meteo.com/v1/forecast
    ?latitude={lat}&longitude={lon}
    &current=temperature_2m,precipitation,windspeed_10m,visibility
    &timezone=auto

Por qué Open-Meteo:
    - Gratuito, sin API key, sin límite de llamadas razonable.
    - Datos en tiempo real con actualización cada 15 minutos.
    - Cobertura global incluyendo México.

Campos usados de la respuesta:
    current.temperature_2m   → temperature_c
    current.precipitation    → is_raining (> 0.0 mm)
    current.windspeed_10m    → wind_speed_kmh
    current.visibility       → visibility_m
"""

from __future__ import annotations
import logging
from typing import Any, Dict

import httpx

from core.context import TrafficContext
from ingest.base import DataIngester, WeatherSnapshot
from safety.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Visibilidad mínima considerada segura en metros
_MIN_SAFE_VISIBILITY: float = 100.0


class WeatherIngester(DataIngester[WeatherSnapshot]):
    """
    Ingester para Open-Meteo — clima en tiempo real, sin API key.

    Attributes
    ----------
    latitude  : Latitud de la ciudad monitoreada.
    longitude : Longitud de la ciudad monitoreada.
    client    : Cliente HTTP async reutilizable.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Campos que pedimos a Open-Meteo
    _CURRENT_FIELDS = [
        "temperature_2m",
        "precipitation",
        "windspeed_10m",
        "visibility",
    ]

    def __init__(self, latitude: float, longitude: float,
                 circuit_breaker: CircuitBreaker) -> None:
        super().__init__(circuit_breaker)

        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitud fuera de rango: {latitude}")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Longitud fuera de rango: {longitude}")

        self.latitude  = round(latitude, 6)
        self.longitude = round(longitude, 6)
        self.client    = httpx.AsyncClient(timeout=5.0)

    async def fetch(self, ctx: TrafficContext) -> Dict[str, Any]:
        """
        Llama a Open-Meteo para obtener condiciones climáticas actuales.

        Parameters
        ----------
        ctx : Contexto del ciclo actual (se usa para logging).

        Returns
        -------
        JSON crudo de Open-Meteo como dict.

        Raises
        ------
        httpx.HTTPStatusError  : Si Open-Meteo responde con error.
        httpx.TimeoutException : Si la llamada supera los 5 segundos.
        ValueError             : Si la respuesta no tiene el formato esperado.
        """
        params = {
            "latitude":  self.latitude,
            "longitude": self.longitude,
            "current":   ",".join(self._CURRENT_FIELDS),
            "timezone":  "auto",
        }

        logger.debug(
            "Open-Meteo fetch → lat=%.4f lon=%.4f hora=%s",
            self.latitude, self.longitude,
            ctx.timestamp.strftime("%H:%M")
        )

        response = await self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if "current" not in data:
            raise ValueError(
                f"Respuesta de Open-Meteo inesperada — falta 'current': {data}"
            )

        return data

    def parse(self, raw: Dict[str, Any]) -> WeatherSnapshot:
        """
        Transforma la respuesta de Open-Meteo a WeatherSnapshot.

        Parameters
        ----------
        raw : JSON de Open-Meteo como dict.

        Returns
        -------
        WeatherSnapshot con los valores actuales.
        """
        current = raw["current"]

        temperature_c  = float(current.get("temperature_2m", 20.0))
        precipitation  = float(current.get("precipitation",  0.0))
        wind_speed_kmh = float(current.get("windspeed_10m",  0.0))

        # Open-Meteo devuelve visibilidad en metros
        # Si no está disponible, asumir buena visibilidad
        visibility_m = float(current.get("visibility", 10000.0))
        visibility_m = max(visibility_m, 0.0)

        is_raining = precipitation > 0.0

        snapshot = WeatherSnapshot(
            temperature_c  = temperature_c,
            is_raining     = is_raining,
            wind_speed_kmh = wind_speed_kmh,
            visibility_m   = visibility_m,
        )

        logger.debug(
            "Open-Meteo parse → temp=%.1f°C lluvia=%s viento=%.1f km/h visibilidad=%.0fm",
            temperature_c, is_raining, wind_speed_kmh, visibility_m
        )

        return snapshot

    def fallback(self) -> WeatherSnapshot:
        """
        Clima de respaldo cuando Open-Meteo no está disponible.
        Retorna condiciones neutras y seguras — no altera los pesos.

        Returns
        -------
        WeatherSnapshot con valores conservadores por defecto.
        """
        logger.warning("WeatherIngester usando fallback — API no disponible")
        return WeatherSnapshot(
            temperature_c  = 20.0,
            is_raining     = False,
            wind_speed_kmh = 0.0,
            visibility_m   = 10000.0,
        )

    def to_context_kwargs(self, snapshot: WeatherSnapshot) -> dict:
        """
        Convierte un WeatherSnapshot a los kwargs que necesita
        TrafficContext.build() — evita repetir la lógica de mapeo
        en cada parte del pipeline.

        Uso:
            snapshot = await weather_ingester.fetch_safe(ctx)
            ctx = TrafficContext.build(
                timestamp = datetime.now(),
                **weather_ingester.to_context_kwargs(snapshot)
            )

        Returns
        -------
        Dict con temperature_c, is_raining, wind_speed_kmh, visibility_m.
        """
        return {
            "temperature_c":  snapshot.temperature_c,
            "is_raining":     snapshot.is_raining,
            "wind_speed_kmh": snapshot.wind_speed_kmh,
            "visibility_m":   snapshot.visibility_m,
        }

    async def close(self) -> None:
        """Cierra el cliente HTTP. Llamar al apagar el pipeline."""
        await self.client.aclose()