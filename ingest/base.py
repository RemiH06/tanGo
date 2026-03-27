"""
ingest/base.py  |  ingest/tomtom.py  |  ingest/weather.py
-----------------------------------------------------------
Jerarquía de ingesters. Todos heredan de DataIngester (abstracta).

La interfaz común garantiza que CitySimulator y los ingesters reales
sean intercambiables — el pipeline no sabe si está hablando con TomTom
o con la simulación.

Nota sobre TomTom Traffic API:
  - Endpoint: /traffic/services/4/flowSegmentData/absolute/10/json
  - Devuelve por segmento: currentSpeed, freeFlowSpeed, confidence,
    currentTravelTime, freeFlowTravelTime, coordinates.
  - No cuenta vehículos individuales — infiere densidad desde velocidad.
  - Para tanGo usamos: currentSpeed / freeFlowSpeed → índice de congestión.
    Congestión 0.0 = libre, 1.0 = completamente congestionado.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

from core.context import TrafficContext
from safety.circuit_breaker import CircuitBreaker


# ── Tipos de retorno ──────────────────────────────────────────────────────────

@dataclass
class TrafficSnapshot:
    """
    Datos de tráfico de un segmento en un instante.

    Attributes
    ----------
    segment_id       : ID del segmento (debe coincidir con RoadSegment).
    congestion_index : 0.0 (libre) a 1.0 (congestionado).
    current_speed_kmh: Velocidad actual medida.
    free_flow_kmh    : Velocidad en condiciones libres (referencia).
    confidence       : Confianza del dato (0.0 – 1.0).
    """
    segment_id:        str
    congestion_index:  float
    current_speed_kmh: float
    free_flow_kmh:     float
    confidence:        float


@dataclass
class WeatherSnapshot:
    """
    Datos climáticos en un instante.

    Attributes
    ----------
    temperature_c   : Temperatura en Celsius.
    is_raining      : True si hay precipitación.
    wind_speed_kmh  : Velocidad del viento.
    visibility_m    : Visibilidad en metros.
    """
    temperature_c:  float
    is_raining:     bool
    wind_speed_kmh: float
    visibility_m:   float


# ── Clase base ────────────────────────────────────────────────────────────────

class DataIngester(ABC):
    """
    Interfaz común para todas las fuentes de datos.
    Permite intercambiar TomTom por el simulador sin cambiar el pipeline.

    Attributes
    ----------
    circuit_breaker : Protección ante fallos de la fuente externa.
    """

    def __init__(self, circuit_breaker: CircuitBreaker) -> None:
        self.circuit_breaker = circuit_breaker

    @abstractmethod
    async def fetch(self, ctx: TrafficContext) -> Any:
        """
        Obtiene datos de la fuente externa para el contexto dado.
        Debe ser async — las llamadas HTTP no deben bloquear el pipeline.

        Parameters
        ----------
        ctx : Contexto del ciclo actual (incluye timestamp y ubicación).

        Returns
        -------
        Datos crudos de la fuente — el tipo concreto depende del ingester.
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: Any) -> Any:
        """
        Transforma la respuesta cruda al tipo interno de tanGo.
        Separado de fetch() para que sea testeable sin red.

        Parameters
        ----------
        raw : Respuesta cruda de la API.

        Returns
        -------
        Datos parseados al tipo interno.
        """
        raise NotImplementedError

    async def fetch_safe(self, ctx: TrafficContext) -> Any:
        """
        Versión segura de fetch() que pasa por el CircuitBreaker.
        El pipeline siempre debe llamar fetch_safe(), nunca fetch() directo.

        Parameters
        ----------
        ctx : Contexto del ciclo actual.

        Returns
        -------
        Datos o fallback si el circuito está abierto.
        """
        # TODO: implementar usando self.circuit_breaker.call()
        raise NotImplementedError


# ── TomTom ────────────────────────────────────────────────────────────────────

class TomTomIngester(DataIngester):
    """
    Ingester para TomTom Traffic Flow API.

    Endpoint usado:
        GET /traffic/services/4/flowSegmentData/absolute/10/json
        ?key={api_key}&point={lat},{lon}&unit=KMPH

    Autenticación: API key en query param (nunca hardcoded — usar .env).

    Attributes
    ----------
    api_key     : Clave de API de TomTom (desde variable de entorno).
    base_url    : URL base de la API.
    client      : Cliente HTTP async (httpx.AsyncClient).
    """

    BASE_URL = "https://api.tomtom.com/traffic/services/4"

    def __init__(self, api_key: str,
                 circuit_breaker: CircuitBreaker) -> None:
        super().__init__(circuit_breaker)
        self.api_key = api_key
        self.client  = httpx.AsyncClient(timeout=5.0)

    async def fetch(self, ctx: TrafficContext) -> Dict:
        """
        Llama a TomTom para obtener datos de flujo.

        Parameters
        ----------
        ctx : Contexto del ciclo (se usa el timestamp para logging).

        Returns
        -------
        JSON crudo de TomTom como dict.
        """
        # TODO: implementar llamada HTTP con api_key
        # Seguridad: validar que api_key no esté vacío antes de llamar
        # Seguridad: sanitizar coordenadas antes de incluirlas en la URL
        raise NotImplementedError

    def parse(self, raw: Dict) -> List[TrafficSnapshot]:
        """
        Transforma la respuesta JSON de TomTom a List[TrafficSnapshot].

        Campos relevantes de TomTom:
          flowSegmentData.currentSpeed     → current_speed_kmh
          flowSegmentData.freeFlowSpeed    → free_flow_kmh
          flowSegmentData.confidence       → confidence
          congestión = 1 - (currentSpeed / freeFlowSpeed)

        Parameters
        ----------
        raw : JSON de TomTom como dict.

        Returns
        -------
        Lista de TrafficSnapshot parseados.
        """
        # TODO: implementar parsing
        raise NotImplementedError

    def fallback(self) -> List[TrafficSnapshot]:
        """
        Datos de respaldo cuando TomTom no está disponible.
        Retorna congestión neutral (0.5) para no alterar los pesos.

        Returns
        -------
        Lista vacía o snapshots con valores neutros.
        """
        # TODO: implementar fallback con valores seguros
        raise NotImplementedError


# ── Open-Meteo ────────────────────────────────────────────────────────────────

class WeatherIngester(DataIngester):
    """
    Ingester para Open-Meteo (clima en tiempo real, sin API key).

    Endpoint:
        GET https://api.open-meteo.com/v1/forecast
        ?latitude={lat}&longitude={lon}
        &current=temperature_2m,precipitation,windspeed_10m,visibility

    Attributes
    ----------
    latitude  : Latitud de la ciudad monitoreada.
    longitude : Longitud de la ciudad monitoreada.
    client    : Cliente HTTP async.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float, longitude: float,
                 circuit_breaker: CircuitBreaker) -> None:
        super().__init__(circuit_breaker)
        self.latitude  = latitude
        self.longitude = longitude
        self.client    = httpx.AsyncClient(timeout=5.0)

    async def fetch(self, ctx: TrafficContext) -> Dict:
        """
        Llama a Open-Meteo para obtener condiciones climáticas actuales.

        Returns
        -------
        JSON crudo de Open-Meteo como dict.
        """
        # TODO: implementar llamada HTTP
        raise NotImplementedError

    def parse(self, raw: Dict) -> WeatherSnapshot:
        """
        Transforma la respuesta de Open-Meteo a WeatherSnapshot.

        Campos relevantes:
          current.temperature_2m  → temperature_c
          current.precipitation   → is_raining (> 0)
          current.windspeed_10m   → wind_speed_kmh
          current.visibility      → visibility_m

        Returns
        -------
        WeatherSnapshot parseado.
        """
        # TODO: implementar parsing
        raise NotImplementedError

    def fallback(self) -> WeatherSnapshot:
        """
        Clima de respaldo: condiciones neutras (templado, sin lluvia).

        Returns
        -------
        WeatherSnapshot con valores seguros por defecto.
        """
        # TODO: implementar fallback
        raise NotImplementedError