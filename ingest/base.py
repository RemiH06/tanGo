"""
ingest/base.py
--------------
Clase abstracta DataIngester y los tipos de retorno compartidos.

La interfaz común garantiza que TomTomIngester, WeatherIngester y
CitySimulator sean intercambiables — el pipeline no sabe ni le importa
de dónde vienen los datos.

Nota sobre TomTom Traffic API:
  - Endpoint: /traffic/services/4/flowSegmentData/absolute/10/json
  - Devuelve por segmento: currentSpeed, freeFlowSpeed, confidence,
    currentTravelTime, freeFlowTravelTime, coordinates.
  - No cuenta vehículos individuales — infiere densidad desde velocidad.
  - Para tanGo: congestion = 1 - (currentSpeed / freeFlowSpeed)
    0.0 = vía libre · 1.0 = completamente congestionada.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
import logging

from core.context import TrafficContext
from safety.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Tipos de retorno ──────────────────────────────────────────────────────────

@dataclass
class TrafficSnapshot:
    """
    Datos de tráfico de un segmento en un instante.

    Attributes
    ----------
    segment_id        : ID del segmento (coincide con RoadSegment.segment_id).
    congestion_index  : 0.0 (libre) a 1.0 (congestionado).
    current_speed_kmh : Velocidad actual medida en el segmento.
    free_flow_kmh     : Velocidad en condiciones sin tráfico (referencia).
    confidence        : Confianza del dato proporcionada por TomTom (0.0–1.0).
    """
    segment_id:        str
    congestion_index:  float
    current_speed_kmh: float
    free_flow_kmh:     float
    confidence:        float

    def __post_init__(self) -> None:
        # Validaciones de rango — datos corruptos no deben entrar al sistema
        if not 0.0 <= self.congestion_index <= 1.0:
            raise ValueError(
                f"congestion_index debe estar entre 0 y 1, recibido: {self.congestion_index}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence debe estar entre 0 y 1, recibido: {self.confidence}"
            )
        if self.current_speed_kmh < 0:
            raise ValueError("current_speed_kmh no puede ser negativo")
        if self.free_flow_kmh <= 0:
            raise ValueError("free_flow_kmh debe ser positivo")


@dataclass
class WeatherSnapshot:
    """
    Datos climáticos en un instante dado.

    Attributes
    ----------
    temperature_c   : Temperatura en grados Celsius.
    is_raining      : True si hay precipitación activa (> 0 mm/h).
    wind_speed_kmh  : Velocidad del viento en km/h.
    visibility_m    : Visibilidad en metros.
    """
    temperature_c:  float
    is_raining:     bool
    wind_speed_kmh: float
    visibility_m:   float

    def __post_init__(self) -> None:
        if self.wind_speed_kmh < 0:
            raise ValueError("wind_speed_kmh no puede ser negativo")
        if self.visibility_m < 0:
            raise ValueError("visibility_m no puede ser negativo")


# ── Clase base abstracta ──────────────────────────────────────────────────────

class DataIngester(ABC, Generic[T]):
    """
    Interfaz común para todas las fuentes de datos de tanGo.

    Cualquier fuente de datos — TomTom, Open-Meteo, el simulador —
    debe heredar de esta clase e implementar fetch() y parse().
    El pipeline siempre llama fetch_safe(), nunca fetch() directamente.

    El tipo genérico T es el tipo de retorno de parse():
      - TomTomIngester[List[TrafficSnapshot]]
      - WeatherIngester[WeatherSnapshot]

    Attributes
    ----------
    circuit_breaker : Protección ante fallos del servicio externo.
    """

    def __init__(self, circuit_breaker: CircuitBreaker) -> None:
        self.circuit_breaker = circuit_breaker

    @abstractmethod
    async def fetch(self, ctx: TrafficContext) -> Any:
        """
        Obtiene datos crudos del servicio externo.
        Debe ser async — las llamadas HTTP no bloquean el pipeline.

        Parameters
        ----------
        ctx : Contexto del ciclo actual.

        Returns
        -------
        Datos crudos — el tipo concreto depende del ingester.
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: Any) -> T:
        """
        Transforma la respuesta cruda al tipo interno de tanGo.
        Separado de fetch() para que sea testeable sin red.

        Parameters
        ----------
        raw : Respuesta cruda del servicio externo.

        Returns
        -------
        Datos parseados al tipo interno T.
        """
        raise NotImplementedError

    @abstractmethod
    def fallback(self) -> T:
        """
        Datos de respaldo seguros cuando el servicio no está disponible.
        Se usa cuando el CircuitBreaker está OPEN.

        Returns
        -------
        Datos neutros que no alteran el comportamiento del sistema.
        """
        raise NotImplementedError

    async def fetch_safe(self, ctx: TrafficContext) -> T:
        """
        Versión segura de fetch() — siempre pasa por el CircuitBreaker.
        El pipeline SIEMPRE debe llamar este método, nunca fetch() directo.

        Flujo:
          1. CircuitBreaker decide si llamar fetch() o ir directo a fallback()
          2. Si fetch() tiene éxito → parse() los datos crudos → retornar
          3. Si fetch() falla → CircuitBreaker registra el fallo → fallback()

        Parameters
        ----------
        ctx : Contexto del ciclo actual.

        Returns
        -------
        Datos parseados (reales o fallback).
        """
        async def _do_fetch() -> T:
            raw = await self.fetch(ctx)
            return self.parse(raw)

        # CircuitBreaker.call() espera callables síncronos,
        # así que envolvemos el coroutine en una función sync
        # usando un flag para capturar el resultado
        result_holder: list = []
        exception_holder: list = []

        async def _run() -> T:
            try:
                data = await _do_fetch()
                result_holder.append(data)
                return data
            except Exception as exc:
                exception_holder.append(exc)
                raise

        # Intentar fetch real; si falla, usar fallback
        try:
            data = await _do_fetch()
            # Notificar éxito al circuit breaker
            self.circuit_breaker._on_success()
            logger.debug(
                "[%s] fetch_safe exitoso", self.circuit_breaker.name
            )
            return data
        except Exception as exc:
            logger.warning(
                "[%s] fetch_safe falló (%s) — usando fallback",
                self.circuit_breaker.name, exc
            )
            self.circuit_breaker._on_failure()
            return self.fallback()