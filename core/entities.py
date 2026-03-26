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
from typing import Optional

from core.context import TrafficContext


# ── Enums ────────────────────────────────────────────────────────────────────

class VehicleType(Enum):
    CAR = auto()
    BUS = auto()
    TRUCK = auto()
    MOTORCYCLE = auto()
    BICYCLE = auto()
    EMERGENCY = auto()       # ambulancia, bomberos — override especial


class Direction(Enum):
    NORTH = "N"
    SOUTH = "S"
    EAST  = "E"
    WEST  = "W"


# ── Clase base ────────────────────────────────────────────────────────────────

class TrafficEntity(ABC):
    """
    Clase base abstracta para cualquier participante del tráfico.
    Define la interfaz común: cada entidad sabe calcular su propio
    peso dado un contexto ambiental.

    Attributes
    ----------
    entity_id   : Identificador único (UUID recomendado).
    base_weight : Peso estático antes de aplicar modificadores.
    is_vulnerable: True si la entidad requiere protección adicional
                   (peatones, sillas de ruedas, ciclistas).
    """

    def __init__(self, entity_id: str, base_weight: float,
                 is_vulnerable: bool = False) -> None:
        self.entity_id   = entity_id
        self.base_weight = base_weight
        self.is_vulnerable = is_vulnerable

    @abstractmethod
    def compute_weight(self, ctx: TrafficContext) -> float:
        """
        Retorna el peso efectivo de esta entidad dado el contexto.
        Las subclases aplican sus propios modificadores.

        Parameters
        ----------
        ctx : Contexto ambiental inmutable del ciclo actual.

        Returns
        -------
        Peso efectivo como float positivo.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.entity_id}, base_weight={self.base_weight})"


# ── Vehicle ──────────────────────────────────────────────────────────────────

@dataclass
class Vehicle(TrafficEntity):
    """
    Vehículo motorizado o de propulsión humana (bicicleta).

    Peso base por tipo:
        CAR        → 5.0
        BUS        → 8.0   (transporta más personas)
        TRUCK      → 6.0
        MOTORCYCLE → 3.0
        BICYCLE    → 2.0
        EMERGENCY  → 999.0 (override inmediato, manejado por SafetyGuard)

    Attributes
    ----------
    vehicle_type : Tipo de vehículo (VehicleType).
    direction    : Dirección de circulación actual.
    """

    entity_id:    str
    vehicle_type: VehicleType
    direction:    Direction
    base_weight:  float = field(init=False)
    is_vulnerable: bool = field(init=False, default=False)

    # Pesos base por tipo — ajustables vía WeightEngine según contexto
    _BASE_WEIGHTS: dict = field(default_factory=lambda: {
        VehicleType.CAR:        5.0,
        VehicleType.BUS:        8.0,
        VehicleType.TRUCK:      6.0,
        VehicleType.MOTORCYCLE: 3.0,
        VehicleType.BICYCLE:    2.0,
        VehicleType.EMERGENCY: 999.0,
    }, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_weight  = self._BASE_WEIGHTS[self.vehicle_type]
        self.is_vulnerable = self.vehicle_type == VehicleType.BICYCLE

    def compute_weight(self, ctx: TrafficContext) -> float:
        """
        Calcula el peso del vehículo según contexto.
        Ejemplos de reglas a implementar:
          - Si is_late_night: peso × 1.5 (más prioridad de madrugada)
          - Si is_raining y BICYCLE: peso × 1.3 (más vulnerable)
          - Si EMERGENCY: retorna base_weight sin modificar (SafetyGuard lo maneja)

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo actual.

        Returns
        -------
        Peso efectivo del vehículo.
        """
        # TODO: implementar modificadores según ctx
        raise NotImplementedError


# ── Pedestrian ────────────────────────────────────────────────────────────────

@dataclass
class Pedestrian(TrafficEntity):
    """
    Peatón en una intersección.

    Peso base: 10.0 (mayor que un auto individual — son más vulnerables).

    Casos esquina:
      - is_wheelchair=True → tiempo mínimo de verde extendido
        (calculado por SafetyGuard según ancho de cruce / velocidad estándar)
      - Temperatura extrema (< 5°C o > 35°C) → peso × 1.3
      - Lluvia → peso × 1.3
      - Madrugada (00-05h) → peso × 0.8 (menos peatones, menos prioridad)

    Attributes
    ----------
    entity_id    : Identificador único.
    is_wheelchair: True si usa silla de ruedas u otro dispositivo de movilidad.
    crossing_width_m: Ancho del cruce en metros (para calcular tiempo verde).
    """

    entity_id:        str
    is_wheelchair:    bool  = False
    crossing_width_m: float = 10.0   # ancho estándar de cruce; ajustar por intersección
    base_weight:      float = field(init=False, default=10.0)
    is_vulnerable:    bool  = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.base_weight  = 10.0
        self.is_vulnerable = True

    def compute_weight(self, ctx: TrafficContext) -> float:
        """
        Calcula el peso del peatón según contexto.

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo actual.

        Returns
        -------
        Peso efectivo del peatón.
        """
        # TODO: implementar modificadores según ctx
        raise NotImplementedError

    def required_green_seconds(self) -> float:
        """
        Tiempo mínimo de verde requerido para cruzar con seguridad.
        Fórmula: ancho_cruce / velocidad_peatón + buffer de seguridad.

        Velocidades estándar (m/s):
            Peatón normal   → 1.2 m/s
            Silla de ruedas → 0.8 m/s
            Buffer          → 3 s adicionales

        Returns
        -------
        Segundos mínimos de verde como float.
        """
        # TODO: implementar cálculo
        raise NotImplementedError