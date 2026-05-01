"""
core/entities.py
----------------
Jerarquía de entidades físicas que participan en el tráfico.

Jerarquía:
    TrafficEntity (ABC)
    ├── Vehicle
    └── Pedestrian

Unidad de tiempo: 1 tick = 30 segundos simulados (TICK_DURATION_S en algorithm.py)

Velocidades:
    Cada entidad tiene una velocidad en km/h que determina cuántos ticks
    tarda en recorrer un segmento. Esto habilita el lifetime entre ticks —
    las entidades persisten en el grafo mientras no llegan a su destino.

    La velocidad de los automóviles varía según el tipo de vía y contexto:
        Autopista libre:    60-90 km/h
        Avenida principal:  40-70 km/h  (con variabilidad aleatoria)
        Calle residencial:  20-40 km/h
        Hora pico:          × 0.6  (congestión)
        Lluvia:             × 0.8  (precaución)
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

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


# ── Velocidades base por tipo de entidad (km/h) ───────────────────────────────
# Estas son velocidades libres — sin congestión ni modificadores de contexto.
# Los autos tienen rango (min, max) para variabilidad realista.

VEHICLE_SPEED_KMH: dict = {
    # (min, max) — se samplea uniformemente por instancia
    VehicleType.CAR:        (40.0, 70.0),   # alta variabilidad — ver docstring
    VehicleType.BUS:        (35.0, 50.0),   # rutas fijas, velocidad moderada
    VehicleType.TRUCK:      (30.0, 50.0),   # pesado, más lento
    VehicleType.MOTORCYCLE: (45.0, 75.0),   # más ágil que el auto
    VehicleType.BICYCLE:    (12.0, 20.0),   # propulsión humana
    VehicleType.EMERGENCY:  (80.0, 100.0),  # máxima prioridad
}

# Modificadores de velocidad por categoría de vía
# El auto en autopista puede ir más rápido que en calle residencial
ROAD_SPEED_FACTOR: dict = {
    "HIGHWAY":          1.3,   # autopista — más rápido que la velocidad base
    "MAIN_AVENUE":      1.0,   # avenida principal — velocidad de referencia
    "SECONDARY_AVENUE": 0.85,  # avenida secundaria
    "STREET":           0.65,  # calle residencial
    "ALLEY":            0.45,  # callejón
}

# Modificadores de velocidad por contexto
CONTEXT_SPEED_FACTOR: dict = {
    "rush_hour":  0.60,  # congestión — reducción significativa
    "late_night": 1.15,  # vías libres — ligeramente más rápido
    "rain":       0.80,  # precaución por lluvia
    "cold":       0.90,  # frío extremo — más precaución
}

# Velocidades peatonales (m/s)
_PEDESTRIAN_SPEED_MS:  float = 1.2
_WHEELCHAIR_SPEED_MS:  float = 0.8
_CROSSING_BUFFER_S:    int   = 3


# ── Clase base ────────────────────────────────────────────────────────────────

class TrafficEntity(ABC):
    """
    Clase base abstracta para cualquier participante del tráfico.

    Attributes
    ----------
    entity_id      : Identificador único (UUID).
    base_weight    : Peso estático antes de aplicar modificadores.
    is_vulnerable  : True si requiere protección adicional.
    speed_kmh      : Velocidad actual en km/h (con variabilidad).
    origin_node    : Nodo donde apareció la entidad.
    destination_node: Nodo destino (None = sin destino definido).
    ticks_alive    : Cuántos ticks lleva esta entidad en el grafo.
    current_node   : Nodo actual.
    ticks_to_next  : Ticks restantes para llegar al siguiente nodo.
    """

    def __init__(self, entity_id: str, base_weight: float,
                 is_vulnerable: bool = False,
                 speed_kmh: float = 30.0,
                 origin_node: str = "",
                 destination_node: Optional[str] = None) -> None:
        self.entity_id         = entity_id
        self.base_weight       = base_weight
        self.is_vulnerable     = is_vulnerable
        self.speed_kmh         = speed_kmh
        self.origin_node       = origin_node
        self.destination_node  = destination_node
        self.current_node      = origin_node
        self.ticks_alive       = 0
        self.ticks_to_next     = 0   # ticks hasta llegar al siguiente nodo

    def travel_time_ticks(self, distance_m: float,
                          road_category: str = "MAIN_AVENUE",
                          ctx: Optional[TrafficContext] = None) -> float:
        """
        Calcula cuántos ticks tarda esta entidad en recorrer distance_m
        sobre una vía de la categoría dada, considerando el contexto.

        Fórmula:
            ticks = distancia_m / (velocidad_efectiva_ms × TICK_DURATION_S)

        Parameters
        ----------
        distance_m    : Longitud del segmento en metros.
        road_category : Categoría de la vía (afecta la velocidad).
        ctx           : Contexto ambiental (afecta velocidad por congestión).

        Returns
        -------
        Número de ticks (puede ser fraccionario — redondear al usar).
        """
        from core.algorithm import TICK_DURATION_S

        speed = self.speed_kmh

        # Modificador por tipo de vía
        road_factor = ROAD_SPEED_FACTOR.get(road_category, 1.0)
        speed *= road_factor

        # Modificadores de contexto
        if ctx:
            if ctx.is_rush_hour:
                speed *= CONTEXT_SPEED_FACTOR["rush_hour"]
            elif ctx.is_late_night:
                speed *= CONTEXT_SPEED_FACTOR["late_night"]
            if ctx.is_raining:
                speed *= CONTEXT_SPEED_FACTOR["rain"]
            if ctx.temperature_c < 5.0:
                speed *= CONTEXT_SPEED_FACTOR["cold"]

        # Velocidad mínima razonable — no puede quedar parado
        speed = max(speed, 5.0)

        speed_ms = speed / 3.6
        return distance_m / (speed_ms * TICK_DURATION_S)

    def tick(self) -> None:
        """Avanza un tick — incrementa el contador de vida."""
        self.ticks_alive += 1
        if self.ticks_to_next > 0:
            self.ticks_to_next -= 1

    @property
    def has_arrived(self) -> bool:
        """True si llegó a su destino o no tiene destino y completó el cruce."""
        if self.destination_node is None:
            return self.ticks_to_next <= 0 and self.ticks_alive > 0
        return self.current_node == self.destination_node

    @abstractmethod
    def compute_weight(self, ctx: TrafficContext) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"(id={self.entity_id[:8]}…, "
                f"speed={self.speed_kmh:.0f}km/h, "
                f"alive={self.ticks_alive})")


# ── Vehicle ───────────────────────────────────────────────────────────────────

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

    La velocidad se samplea aleatoriamente dentro del rango del tipo
    al momento de la creación — cada instancia tiene su propia velocidad.
    Esto modela la variabilidad real del tráfico: no todos los autos
    van a la misma velocidad aunque sean del mismo tipo.

    Attributes
    ----------
    entity_id    : UUID único.
    vehicle_type : Tipo de vehículo.
    direction    : Dirección de circulación.
    speed_kmh    : Velocidad individual (sampleada del rango del tipo).
    """

    entity_id:    str
    vehicle_type: VehicleType
    direction:    Direction
    base_weight:  float = field(init=False)
    is_vulnerable: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.base_weight   = _VEHICLE_BASE_WEIGHTS[self.vehicle_type]
        self.is_vulnerable = self.vehicle_type == VehicleType.BICYCLE

        # Samplear velocidad individual del rango del tipo
        v_min, v_max = VEHICLE_SPEED_KMH.get(
            self.vehicle_type, (30.0, 50.0)
        )
        self.speed_kmh = round(random.uniform(v_min, v_max), 1)

        # Inicializar campos de TrafficEntity
        self.origin_node       = ""
        self.destination_node  = None
        self.current_node      = ""
        self.ticks_alive       = 0
        self.ticks_to_next     = 0

    def compute_weight(self, ctx: TrafficContext) -> float:
        """
        Peso efectivo del vehículo según el contexto.

        Emergencias retornan 999 sin modificar — SafetyGuard los detecta.
        La velocidad alta en madrugada refleja que los autos van más rápido
        (menos tráfico) pero también pesan más para el algoritmo
        porque representan más riesgo en intersecciones vacías.
        """
        if self.vehicle_type == VehicleType.EMERGENCY:
            return self.base_weight

        weight = self.base_weight

        if ctx.is_late_night:
            weight *= 1.5
        elif ctx.is_rush_hour:
            weight *= 1.1

        if ctx.is_raining and self.vehicle_type == VehicleType.BICYCLE:
            weight *= 1.3

        return weight

    @property
    def speed_category(self) -> str:
        """Categoría de velocidad para logging y visualización."""
        if self.speed_kmh >= 70:
            return "rápido"
        if self.speed_kmh >= 45:
            return "normal"
        return "lento"


# ── Pedestrian ────────────────────────────────────────────────────────────────

@dataclass
class Pedestrian(TrafficEntity):
    """
    Peatón en una intersección.

    Velocidad fija por tipo (normal vs silla de ruedas) — la variabilidad
    peatonal es menor que la vehicular y más importante modelarla por
    capacidad física que por elección.
    """

    entity_id:        str
    is_wheelchair:    bool  = False
    crossing_width_m: float = 10.0
    base_weight:      float = field(init=False, default=10.0)
    is_vulnerable:    bool  = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.base_weight   = 10.0
        self.is_vulnerable = True
        self.origin_node       = ""
        self.destination_node  = None
        self.current_node      = ""
        self.ticks_alive       = 0
        self.ticks_to_next     = 0

        # Velocidad en km/h (convertida de m/s)
        speed_ms   = _WHEELCHAIR_SPEED_MS if self.is_wheelchair else _PEDESTRIAN_SPEED_MS
        self.speed_kmh = round(speed_ms * 3.6, 2)   # 4.32 o 2.88 km/h

    def compute_weight(self, ctx: TrafficContext) -> float:
        weight = self.base_weight
        if self.is_wheelchair:
            weight *= 1.5
        if ctx.is_raining:
            weight *= 1.3
        if ctx.temperature_c < 5.0 or ctx.temperature_c > 35.0:
            weight *= 1.3
        if ctx.is_late_night:
            weight *= 0.8
        return weight

    def required_green_seconds(self) -> float:
        """Segundos mínimos de verde para cruzar con seguridad."""
        speed = _WHEELCHAIR_SPEED_MS if self.is_wheelchair else _PEDESTRIAN_SPEED_MS
        return (self.crossing_width_m / speed) + _CROSSING_BUFFER_S

    def required_green_ticks(self) -> int:
        """Ticks mínimos de verde (usando TICK_DURATION_S = 30s)."""
        from core.algorithm import TICK_DURATION_S
        return max(1, round(self.required_green_seconds() / TICK_DURATION_S))