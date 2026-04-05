"""
ingest/tomtom.py
----------------
Implementación de DataIngester para TomTom Traffic Flow API.

Endpoint usado:
    GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json
    ?key={api_key}&point={lat},{lon}&unit=KMPH

Qué devuelve TomTom (campos relevantes para tanGo):
    flowSegmentData.currentSpeed     → velocidad actual en el segmento
    flowSegmentData.freeFlowSpeed    → velocidad sin tráfico (referencia)
    flowSegmentData.confidence       → confianza del dato (0.0 – 1.0)
    flowSegmentData.coordinates      → puntos geográficos del segmento

TomTom NO cuenta vehículos individuales. Infiere densidad desde velocidad:
    congestion_index = 1 - (currentSpeed / freeFlowSpeed)
    0.0 = vía completamente libre
    1.0 = vía completamente congestionada

Autenticación: API key en query param ?key=...
    Nunca hardcodear — leer desde variable de entorno TOMTOM_API_KEY.
"""

from __future__ import annotations
import os
import logging
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv

from core.context import TrafficContext
from ingest.base import DataIngester, TrafficSnapshot
from safety.circuit_breaker import CircuitBreaker

load_dotenv()
logger = logging.getLogger(__name__)

# Velocidad mínima para evitar división por cero en el cálculo de congestión
_MIN_FREE_FLOW_SPEED: float = 1.0


class TomTomIngester(DataIngester[List[TrafficSnapshot]]):
    """
    Ingester para TomTom Traffic Flow API.

    Cada llamada consulta un punto geográfico y TomTom devuelve
    los datos del segmento vial más cercano a ese punto.
    Para cubrir múltiples intersecciones se llama una vez por cada
    intersección del grafo — las llamadas son async y concurrentes.

    Attributes
    ----------
    api_key  : Clave de TomTom — se lee de TOMTOM_API_KEY en .env.
    client   : Cliente HTTP async reutilizable (una instancia por ingester).
    zoom     : Nivel de zoom del segmento (10 = nivel de calle, recomendado).
    """

    BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"

    def __init__(self, circuit_breaker: CircuitBreaker,
                 zoom: int = 10) -> None:
        super().__init__(circuit_breaker)

        api_key = os.getenv("TOMTOM_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "TOMTOM_API_KEY no encontrada. "
                "Asegúrate de tener un archivo .env con TOMTOM_API_KEY=tu_key"
            )
        # Validación mínima: la key de TomTom siempre tiene 32 caracteres
        if len(api_key) != 32:
            raise ValueError(
                f"TOMTOM_API_KEY parece inválida — se esperan 32 caracteres, "
                f"se encontraron {len(api_key)}"
            )

        self._api_key = api_key
        self.zoom     = zoom
        self.client   = httpx.AsyncClient(timeout=5.0)

    async def fetch(self, ctx: TrafficContext) -> Dict[str, Any]:
        """
        Llama a TomTom para un punto geográfico dado.
        El punto viene del contexto — en producción se itera
        por cada intersección del grafo.

        Parameters
        ----------
        ctx : Contexto del ciclo actual. Debe incluir lat/lon
              de la intersección a consultar.

        Returns
        -------
        JSON crudo de TomTom como dict.

        Raises
        ------
        httpx.HTTPStatusError  : Si TomTom responde con 4xx o 5xx.
        httpx.TimeoutException : Si la llamada supera los 5 segundos.
        ValueError             : Si la respuesta no tiene el formato esperado.
        """
        # Sanitizar coordenadas antes de incluirlas en la URL
        lat, lon = self._sanitize_coordinates(ctx)

        url    = self.BASE_URL.format(zoom=self.zoom)
        params = {
            "key":   self._api_key,
            "point": f"{lat},{lon}",
            "unit":  "KMPH",
        }

        logger.debug("TomTom fetch → lat=%.4f lon=%.4f", lat, lon)

        response = await self.client.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Validar que la respuesta tiene la estructura esperada
        if "flowSegmentData" not in data:
            raise ValueError(
                f"Respuesta de TomTom inesperada — falta 'flowSegmentData': {data}"
            )

        return data

    def parse(self, raw: Dict[str, Any]) -> List[TrafficSnapshot]:
        """
        Transforma la respuesta JSON de TomTom a List[TrafficSnapshot].

        TomTom devuelve un solo segmento por llamada (el más cercano
        al punto consultado). Lo empaquetamos en una lista para mantener
        la interfaz consistente con otros ingesters.

        Parameters
        ----------
        raw : JSON de TomTom como dict.

        Returns
        -------
        Lista con un TrafficSnapshot del segmento consultado.
        """
        fsd = raw["flowSegmentData"]

        current_speed  = float(fsd.get("currentSpeed",  0.0))
        free_flow      = float(fsd.get("freeFlowSpeed", _MIN_FREE_FLOW_SPEED))
        confidence     = float(fsd.get("confidence",    0.0))

        # Proteger contra free_flow = 0 (división por cero)
        free_flow = max(free_flow, _MIN_FREE_FLOW_SPEED)

        # Congestión: 0.0 = libre, 1.0 = completamente congestionado
        congestion = 1.0 - (current_speed / free_flow)
        congestion = max(0.0, min(1.0, congestion))  # clamp [0, 1]

        # Extraer ID del segmento si TomTom lo provee, o construirlo
        # desde las coordenadas del primer punto del segmento
        coords     = fsd.get("coordinates", {}).get("coordinate", [{}])
        first      = coords[0] if coords else {}
        segment_id = f"tomtom_{first.get('latitude', 0):.4f}_{first.get('longitude', 0):.4f}"

        snapshot = TrafficSnapshot(
            segment_id        = segment_id,
            congestion_index  = congestion,
            current_speed_kmh = current_speed,
            free_flow_kmh     = free_flow,
            confidence        = confidence,
        )

        logger.debug(
            "TomTom parse → seg=%s congestion=%.2f speed=%.1f km/h",
            segment_id, congestion, current_speed
        )

        return [snapshot]

    def fallback(self) -> List[TrafficSnapshot]:
        """
        Datos de respaldo cuando TomTom no está disponible.
        Retorna congestión neutra (0.5) para no alterar los pesos
        del sistema de forma brusca.

        Returns
        -------
        Lista con un snapshot de valores neutros.
        """
        logger.warning("TomTomIngester usando fallback — API no disponible")
        return [
            TrafficSnapshot(
                segment_id        = "fallback",
                congestion_index  = 0.5,
                current_speed_kmh = 30.0,
                free_flow_kmh     = 60.0,
                confidence        = 0.0,    # confianza 0 → el sistema sabe que es fallback
            )
        ]

    async def close(self) -> None:
        """Cierra el cliente HTTP. Llamar al apagar el pipeline."""
        await self.client.aclose()

    # ── Helpers privados ──────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_coordinates(ctx: TrafficContext) -> tuple[float, float]:
        """
        Extrae y valida las coordenadas del contexto.
        Lanza ValueError si están fuera de rango — evita inyección
        de coordenadas maliciosas en la URL de la API.

        Returns
        -------
        Tupla (latitude, longitude) validada.
        """
        # TrafficContext no tiene lat/lon directamente —
        # en producción el pipeline los pasa por intersección.
        # Aquí usamos las coordenadas de Guadalajara como default
        # hasta que el grafo de intersecciones esté conectado.
        lat = getattr(ctx, "latitude",  20.6597)
        lon = getattr(ctx, "longitude", -103.3496)

        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitud fuera de rango: {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitud fuera de rango: {lon}")

        return round(lat, 6), round(lon, 6)