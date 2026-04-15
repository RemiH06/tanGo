"""
entities.py
-----------
Jerarquía de entidades físicas que participan en el tráfico.
Todas heredan de TrafficEntity (abstracta).

Jerarquía:
    TrafficEntity (ABC)
    ├── Vehicle
    └── Pedestrian
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field

from core.context import TrafficContext


# ── Enums ─────────────────────────────────────────────────────────────────────

class VehicleType(Enum):
    CAR        = auto()
    BUS        = auto()
    TRUCK      = auto()
    MOTORCYCLE = auto()
    BICYCLE    = auto()
    EMERGENCY  = auto()


class Direction(Enum):
    NORTH = "N"
    SOUTH = "S"
    EAST  = "E"
    WEST  = "W"


# ── Constantes de velocidad peatonal ─────────────────────────────────────────

_PEDESTRIAN_SPEED_MS:  float = 1.2   # m/s velocidad normal
_WHEELCHAIR_SPEED_MS:  float = 0.8   # m/s velocidad en silla de ruedas
_CROSSING_BUFFER_S:    int   = 3     # segundos de buffer de seguridad


# ── Clase base ────────────────────────────────────────────────────────────────

class TrafficEntity(ABC):
    """
    Clase base abstracta para cualquier participante del tráfico.

    Attributes
    ----------
    entity_id    : Identificador único (UUID).
    base_weight  : Peso estático antes de aplicar modificadores.
    is_vulnerable: True si requiere protección adicional.
    """

    def __init__(self, entity_id: str, base_weight: float,
                 is_vulnerable: bool = False) -> None:
        self.entity_id    = entity_id
        self.base_weight  = base_weight
        self.is_vulnerable = is_vulnerable

    @abstractmethod
    def compute_weight(self, ctx: TrafficContext) -> float:
        """
        Retorna el peso efectivo dado el contexto ambiental.

        Parameters
        ----------
        ctx : Contexto ambiental inmutable del ciclo actual.

        Returns
        -------
        Peso efectivo como float positivo.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(id={self.entity_id[:8]}…, base_weight={self.base_weight})"
        )


# ── Vehicle ───────────────────────────────────────────────────────────────────

# Pesos base por tipo de vehículo
_VEHICLE_BASE_WEIGHTS: dict = {
    VehicleType.CAR:        5.0,
    VehicleType.BUS:        8.0,
    VehicleType.TRUCK:      6.0,
    VehicleType.MOTORCYCLE: 3.0,
    VehicleType.BICYCLE:    2.0,
    VehicleType.EMERGENCY: 999.0,
}


@dataclass
class Vehicle(TrafficEntity):
    """
    Vehículo motorizado o de propulsión humana.

    Modificadores de peso:
      - EMERGENCY    → base_weight sin modificar (SafetyGuard maneja override)
      - Madrugada    → × 1.5  (más prioridad a autos de noche)
      - Lluvia + BICYCLE → × 1.3 (ciclista más vulnerable con lluvia)
      - Hora pico    → × 1.1  (refuerza la prioridad vehicular)

    Attributes
    ----------
    entity_id    : UUID único.
    vehicle_type : Tipo de vehículo.
    direction    : Dirección de circulación.
    """

    entity_id:    str
    vehicle_type: VehicleType
    direction:    Direction
    base_weight:  float = field(init=False)
    is_vulnerable: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.base_weight   = _VEHICLE_BASE_WEIGHTS[self.vehicle_type]
        self.is_vulnerable = self.vehicle_type == VehicleType.BICYCLE

    def compute_weight(self, ctx: TrafficContext) -> float:
        """
        Calcula el peso efectivo del vehículo según el contexto.

        Los vehículos de emergencia retornan su peso base sin modificar
        — SafetyGuard los detecta por el peso 999 y hace override inmediato.

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo.

        Returns
        -------
        Peso efectivo del vehículo.
        """
        # Emergencia: no se modifica, SafetyGuard lo toma directo
        if self.vehicle_type == VehicleType.EMERGENCY:
            return self.base_weight

        weight = self.base_weight

        # Madrugada → más prioridad a vehículos (menos peatones, más autos)
        if ctx.is_late_night:
            weight *= 1.5

        # Hora pico → refuerzo vehicular
        elif ctx.is_rush_hour:
            weight *= 1.1

        # Lluvia + bicicleta → más vulnerabilidad
        if ctx.is_raining and self.vehicle_type == VehicleType.BICYCLE:
            weight *= 1.3

        return weight


# ── Pedestrian ────────────────────────────────────────────────────────────────

@dataclass
class Pedestrian(TrafficEntity):
    """
    Peatón en una intersección.

    Peso base: 10.0 — mayor que un auto porque son más vulnerables.

    Modificadores de peso:
      - Silla de ruedas          → × 1.5  (siempre, mínimo garantizado)
      - Lluvia                   → × 1.3
      - Temperatura < 5°C        → × 1.3  (frío extremo)
      - Temperatura > 35°C       → × 1.3  (calor extremo)
      - Madrugada (00-05h)       → × 0.8  (menos peatones, menos prioridad)

    Los modificadores se aplican en orden y se acumulan.
    Ejemplo: silla de ruedas + lluvia → × 1.5 × 1.3 = × 1.95

    Attributes
    ----------
    entity_id        : UUID único.
    is_wheelchair    : True si usa silla de ruedas u otro dispositivo.
    crossing_width_m : Ancho del cruce en metros (para calcular tiempo verde).
    """

    entity_id:        str
    is_wheelchair:    bool  = False
    crossing_width_m: float = 10.0
    base_weight:      float = field(init=False, default=10.0)
    is_vulnerable:    bool  = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.base_weight   = 10.0
        self.is_vulnerable = True

    def compute_weight(self, ctx: TrafficContext) -> float:
        """
        Calcula el peso efectivo del peatón según el contexto.
        Los modificadores se acumulan multiplicativamente.

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo.

        Returns
        -------
        Peso efectivo del peatón.
        """
        weight = self.base_weight

        # Silla de ruedas — modificador base siempre activo
        if self.is_wheelchair:
            weight *= 1.5

        # Lluvia — más peligroso cruzar
        if ctx.is_raining:
            weight *= 1.3

        # Temperatura extrema
        if ctx.temperature_c < 5.0 or ctx.temperature_c > 35.0:
            weight *= 1.3

        # Madrugada — menos peatones, menor prioridad relativa
        if ctx.is_late_night:
            weight *= 0.8

        return weight

    def required_green_seconds(self) -> float:
        """
        Tiempo mínimo de verde para cruzar con seguridad.

        Fórmula:
            t = crossing_width_m / velocidad + buffer

        Velocidades:
            Peatón normal   → 1.2 m/s
            Silla de ruedas → 0.8 m/s
            Buffer          → 3 s

        Returns
        -------
        Segundos mínimos de verde.
        """
        speed = _WHEELCHAIR_SPEED_MS if self.is_wheelchair else _PEDESTRIAN_SPEED_MS
        return (self.crossing_width_m / speed) + _CROSSING_BUFFER_S