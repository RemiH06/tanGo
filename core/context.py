"""
context.py
----------
Contexto ambiental inmutable que se crea en cada ciclo del pipeline.
Al ser frozen=True, ninguna función puede mutarlo — garantiza que
WeightEngine sea siempre pura y testeable.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TrafficContext:
    """
    Snapshot del entorno en un instante dado.
    Se pasa por referencia a WeightEngine y SafetyGuard
    sin que nadie pueda modificarlo.

    Attributes
    ----------
    timestamp       : Momento exacto de la medición.
    temperature_c   : Temperatura en Celsius.
    is_raining      : True si hay precipitación activa.
    wind_speed_kmh  : Velocidad del viento en km/h.
    visibility_m    : Visibilidad en metros (niebla, lluvia intensa).
    is_weekend      : True si es sábado o domingo.
    is_rush_hour    : True si es hora pico (07-09h, 17-20h aprox.).
    is_late_night   : True si es entre 00:00 y 05:00.
    """

    timestamp: datetime
    temperature_c: float
    is_raining: bool
    wind_speed_kmh: float
    visibility_m: float
    is_weekend: bool
    is_rush_hour: bool
    is_late_night: bool

    @property
    def hour(self) -> int:
        """Hora del día (0-23) derivada del timestamp."""
        return self.timestamp.hour

    @classmethod
    def build(cls, timestamp: datetime, temperature_c: float,
              is_raining: bool, wind_speed_kmh: float,
              visibility_m: float) -> "TrafficContext":
        """
        Factory que infiere is_weekend, is_rush_hour e is_late_night
        automáticamente a partir del timestamp.

        Parameters
        ----------
        timestamp       : Momento de la medición.
        temperature_c   : Temperatura en Celsius.
        is_raining      : Precipitación activa.
        wind_speed_kmh  : Velocidad del viento.
        visibility_m    : Visibilidad en metros.

        Returns
        -------
        TrafficContext inmutable con todos los campos calculados.
        """
        # TODO: implementar lógica de is_weekend, is_rush_hour, is_late_night
        raise NotImplementedError