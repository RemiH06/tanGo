"""
road.py
-------
Modela la red vial como grafos:
  - RoadSegment : arista dirigida entre dos intersecciones.
  - Intersection: nodo del grafo con lógica de semáforo.

Los pesos son dinámicos — se recalculan en cada ciclo del pipeline
vía WeightEngine, nunca se guardan como estado permanente en estas clases.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional, TYPE_CHECKING
import logging

from core.context import TrafficContext
from core.entities import TrafficEntity

if TYPE_CHECKING:
    from core.weight_engine import WeightEngine

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class Phase(Enum):
    """Estado actual del semáforo."""
    GREEN  = "green"
    YELLOW = "yellow"
    RED    = "red"


class Turn(Enum):
    """Giros posibles al final de un segmento."""
    LEFT     = auto()
    RIGHT    = auto()
    STRAIGHT = auto()
    U_TURN   = auto()


class RoadCategory(Enum):
    """
    Categoría de la vía — determina el peso base.
    WeightEngine ajusta estos valores dinámicamente según el contexto.
    """
    HIGHWAY          = 100
    MAIN_AVENUE      = 80
    SECONDARY_AVENUE = 50
    STREET           = 20
    ALLEY            = 5


# ── RoadSegment ───────────────────────────────────────────────────────────────

@dataclass
class RoadSegment:
    """
    Arista dirigida del grafo vial.
    Representa un tramo de calle entre dos intersecciones.

    En Neo4j es una relación (:Intersection)-[:ROAD]->(:Intersection)
    con propiedades equivalentes a los atributos de esta clase.

    Attributes
    ----------
    segment_id       : ID único (coincide con el ID en Neo4j).
    from_node_id     : ID de la intersección de origen.
    to_node_id       : ID de la intersección de destino.
    category         : Categoría de la vía.
    length_m         : Longitud en metros.
    speed_limit_kmh  : Velocidad máxima permitida.
    allowed_turns    : Si no está vacío, SOLO estos giros son válidos.
    forbidden_turns  : Giros explícitamente prohibidos (caso esquina).
    has_bike_lane    : True si tiene carril exclusivo para bicicletas.
    has_sidewalk     : True si tiene banqueta / acera.
    """

    segment_id:      str
    from_node_id:    str
    to_node_id:      str
    category:        RoadCategory
    length_m:        float
    speed_limit_kmh: float
    allowed_turns:   List[Turn] = field(default_factory=list)
    forbidden_turns: List[Turn] = field(default_factory=list)
    has_bike_lane:   bool       = False
    has_sidewalk:    bool       = True

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise ValueError(f"length_m debe ser positivo, recibido: {self.length_m}")
        if self.speed_limit_kmh <= 0:
            raise ValueError(f"speed_limit_kmh debe ser positivo, recibido: {self.speed_limit_kmh}")
        # Un giro no puede estar permitido y prohibido al mismo tiempo
        conflict = set(self.allowed_turns) & set(self.forbidden_turns)
        if conflict:
            raise ValueError(f"Giros en conflicto (permitido y prohibido): {conflict}")

    @property
    def base_weight(self) -> float:
        """Peso base estático derivado de la categoría de la vía."""
        return float(self.category.value)

    @property
    def travel_time_seconds(self) -> float:
        """
        Tiempo de viaje en condiciones libres (sin tráfico).
        Útil para calcular el offset de la ola verde.
        """
        speed_ms = self.speed_limit_kmh / 3.6
        return self.length_m / speed_ms

    def is_turn_allowed(self, turn: Turn) -> bool:
        """
        Valida si un giro está permitido en este segmento.

        Lógica:
          1. Si el giro está en forbidden_turns → siempre False.
          2. Si allowed_turns no está vacío → solo esos son válidos.
          3. Si allowed_turns está vacío → todos permitidos (excepto los prohibidos).

        Parameters
        ----------
        turn : Giro a validar.

        Returns
        -------
        True si el giro está permitido.
        """
        if turn in self.forbidden_turns:
            return False
        if self.allowed_turns:
            return turn in self.allowed_turns
        return True

    def is_accessible_for_pedestrians(self) -> bool:
        """
        True si los peatones pueden usar este segmento con seguridad.
        Requiere banqueta — las autopistas no son accesibles a pie.
        """
        return self.has_sidewalk and self.category != RoadCategory.HIGHWAY

    def to_neo4j_props(self) -> dict:
        """
        Serializa el segmento a un dict compatible con propiedades de Neo4j.
        Usado por CitySimulator.write_to_neo4j().

        Returns
        -------
        Dict con todas las propiedades del segmento.
        """
        return {
            "segment_id":      self.segment_id,
            "from_node_id":    self.from_node_id,
            "to_node_id":      self.to_node_id,
            "category":        self.category.name,
            "base_weight":     self.base_weight,
            "length_m":        self.length_m,
            "speed_limit_kmh": self.speed_limit_kmh,
            "has_bike_lane":   self.has_bike_lane,
            "has_sidewalk":    self.has_sidewalk,
            "allowed_turns":   [t.name for t in self.allowed_turns],
            "forbidden_turns": [t.name for t in self.forbidden_turns],
        }

    @classmethod
    def from_neo4j_props(cls, props: dict) -> "RoadSegment":
        """
        Reconstruye un RoadSegment desde propiedades de Neo4j.
        Usado por CitySimulator.load_from_neo4j().

        Parameters
        ----------
        props : Dict de propiedades de la relación :ROAD en Neo4j.

        Returns
        -------
        RoadSegment reconstruido.
        """
        return cls(
            segment_id      = props["segment_id"],
            from_node_id    = props["from_node_id"],
            to_node_id      = props["to_node_id"],
            category        = RoadCategory[props["category"]],
            length_m        = float(props["length_m"]),
            speed_limit_kmh = float(props["speed_limit_kmh"]),
            has_bike_lane   = bool(props.get("has_bike_lane", False)),
            has_sidewalk    = bool(props.get("has_sidewalk", True)),
            allowed_turns   = [Turn[t] for t in props.get("allowed_turns", [])],
            forbidden_turns = [Turn[t] for t in props.get("forbidden_turns", [])],
        )


# ── Intersection ──────────────────────────────────────────────────────────────

# Duración de fase amarilla — fija por seguridad vial
_YELLOW_DURATION_S: int = 3

# Rangos de duración de ciclo verde según condición
_GREEN_RUSH_HOUR_S:  int = 45
_GREEN_NORMAL_S:     int = 30
_GREEN_LATE_NIGHT_S: int = 20
_GREEN_MIN_S:        int = 7    # nunca menos de esto por seguridad


@dataclass
class Intersection:
    """
    Nodo del grafo vial. Representa una intersección física con semáforo.

    En Neo4j es un nodo (:Intersection) con propiedades equivalentes.

    Ciclo de vida en cada tick del pipeline:
      1. CitySimulator.tick()         → entidades presentes en esta intersección
      2. WeightEngine.aggregate_pressure() → presión total
      3. SafetyGuard.validate()       → aprueba o modifica el cambio de fase
      4. adjust_phase()               → actualiza current_phase

    Attributes
    ----------
    node_id           : ID único (coincide con el nodo en Neo4j).
    name              : Nombre descriptivo ("Av. Patria y Periférico").
    latitude          : Latitud geográfica.
    longitude         : Longitud geográfica.
    incoming_segments : Segmentos que llegan a esta intersección.
    current_phase     : Fase actual del semáforo.
    pressure          : Presión calculada en el último ciclo.
    min_green_seconds : Tiempo mínimo de verde — SafetyGuard puede aumentarlo.
    _phase_started_at : Timestamp en que comenzó la fase actual (interno).
    """

    node_id:           str
    name:              str
    latitude:          float
    longitude:         float
    incoming_segments: List[RoadSegment] = field(default_factory=list)
    current_phase:     Phase             = Phase.RED
    pressure:          float             = 0.0
    min_green_seconds: int               = _GREEN_MIN_S
    _phase_started_at: datetime          = field(
        default_factory=datetime.now, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Latitud fuera de rango: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Longitud fuera de rango: {self.longitude}")

    # ── Lógica de fase ────────────────────────────────────────────────────────

    def adjust_phase(self, engine: "WeightEngine", ctx: TrafficContext,
                     entities: List[TrafficEntity]) -> Phase:
        """
        Recalcula la fase del semáforo para este ciclo.

        Flujo:
          1. Calcular presión con WeightEngine
          2. Decidir si cambiar de fase
          3. Aplicar transición correcta (RED→GREEN, GREEN→YELLOW, YELLOW→RED)
          4. Actualizar pressure y _phase_started_at si hubo cambio

        Parameters
        ----------
        engine   : Motor de pesos (funciones puras).
        ctx      : Contexto ambiental del ciclo.
        entities : Entidades presentes en esta intersección ahora.

        Returns
        -------
        La nueva fase (o la misma si no hubo cambio).
        """
        # Calcular presión actual
        self.pressure = engine.aggregate_pressure(entities, self, ctx)

        should_change = engine.should_change_phase(self.pressure)
        seconds_in_phase = (datetime.now() - self._phase_started_at).total_seconds()

        new_phase = self._next_phase(
            should_change     = should_change,
            seconds_in_phase  = seconds_in_phase,
            ctx               = ctx,
        )

        if new_phase != self.current_phase:
            logger.info(
                "[%s] Fase: %s → %s | presión=%.1f | tiempo_en_fase=%.0fs",
                self.name, self.current_phase.value,
                new_phase.value, self.pressure, seconds_in_phase
            )
            self.current_phase    = new_phase
            self._phase_started_at = datetime.now()

        return self.current_phase

    def _next_phase(self, should_change: bool,
                    seconds_in_phase: float,
                    ctx: TrafficContext) -> Phase:
        """
        Determina la siguiente fase según la fase actual y las condiciones.

        Máquina de estados:
          RED    → GREEN  : si should_change y tiempo mínimo cumplido
          GREEN  → YELLOW : si NOT should_change y tiempo mínimo cumplido
          YELLOW → RED    : siempre después de _YELLOW_DURATION_S segundos

        Parameters
        ----------
        should_change    : Si la presión supera el umbral.
        seconds_in_phase : Segundos transcurridos en la fase actual.
        ctx              : Contexto para calcular duración de ciclo.

        Returns
        -------
        La siguiente fase.
        """
        green_duration = self.get_cycle_duration(ctx)

        if self.current_phase == Phase.RED:
            if should_change and seconds_in_phase >= self.min_green_seconds:
                return Phase.GREEN

        elif self.current_phase == Phase.GREEN:
            if not should_change and seconds_in_phase >= green_duration:
                return Phase.YELLOW
            # También terminar verde si el tiempo máximo se cumplió
            if seconds_in_phase >= green_duration * 1.5:
                return Phase.YELLOW

        elif self.current_phase == Phase.YELLOW:
            if seconds_in_phase >= _YELLOW_DURATION_S:
                return Phase.RED

        return self.current_phase

    def get_cycle_duration(self, ctx: TrafficContext) -> int:
        """
        Duración del verde en segundos según el contexto.

        Reglas:
          - Hora pico    → verde más largo (más flujo vehicular)
          - Madrugada    → verde más corto (menos tráfico)
          - Normal       → duración estándar

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo.

        Returns
        -------
        Duración del verde en segundos.
        """
        if ctx.is_late_night:
            return _GREEN_LATE_NIGHT_S
        if ctx.is_rush_hour:
            return _GREEN_RUSH_HOUR_S
        return _GREEN_NORMAL_S

    def seconds_in_current_phase(self) -> float:
        """Segundos transcurridos en la fase actual."""
        return (datetime.now() - self._phase_started_at).total_seconds()

    # ── Grafo ─────────────────────────────────────────────────────────────────

    def neighbors(self) -> List[str]:
        """
        IDs de las intersecciones alcanzables desde esta.
        Usado por TrafficGraph para propagar la ola verde.

        Returns
        -------
        Lista de node_id de intersecciones vecinas.
        """
        return [seg.to_node_id for seg in self.incoming_segments]

    def main_segment(self) -> Optional[RoadSegment]:
        """
        Segmento de mayor peso entre los que llegan a esta intersección.
        Usado por WeightEngine como referencia para normalizar la presión.

        Returns
        -------
        RoadSegment de mayor categoría, o None si no hay segmentos.
        """
        if not self.incoming_segments:
            return None
        return max(self.incoming_segments, key=lambda s: s.base_weight)

    # ── Neo4j ─────────────────────────────────────────────────────────────────

    def to_neo4j_props(self) -> dict:
        """
        Serializa la intersección a un dict compatible con Neo4j.

        Returns
        -------
        Dict con las propiedades del nodo :Intersection.
        """
        return {
            "node_id":   self.node_id,
            "name":      self.name,
            "latitude":  self.latitude,
            "longitude": self.longitude,
            "phase":     self.current_phase.value,
            "pressure":  self.pressure,
        }

    @classmethod
    def from_neo4j_props(cls, props: dict) -> "Intersection":
        """
        Reconstruye una Intersection desde propiedades de Neo4j.

        Parameters
        ----------
        props : Dict de propiedades del nodo :Intersection.

        Returns
        -------
        Intersection reconstruida.
        """
        return cls(
            node_id   = props["node_id"],
            name      = props["name"],
            latitude  = float(props["latitude"]),
            longitude = float(props["longitude"]),
        )