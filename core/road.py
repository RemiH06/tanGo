"""
road.py
-------
Modela la red vial como grafos:
  - RoadSegment: arista dirigida entre dos intersecciones.
  - Intersection: nodo del grafo con lógica de semáforo.

Los pesos son dinámicos — se recalculan en cada ciclo del pipeline
vía WeightEngine, no se guardan como estado permanente.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, TYPE_CHECKING

from core.context import TrafficContext
from core.entities import TrafficEntity

if TYPE_CHECKING:
    from core.weight_engine import WeightEngine


# ── Enums ────────────────────────────────────────────────────────────────────

class Phase(Enum):
    """Estado actual del semáforo."""
    GREEN  = "green"
    YELLOW = "yellow"
    RED    = "red"


class Turn(Enum):
    """Giros posibles en una intersección."""
    LEFT    = auto()
    RIGHT   = auto()
    STRAIGHT = auto()
    U_TURN  = auto()


class RoadCategory(Enum):
    """
    Categoría de la vía — determina el peso base.
    Los pesos son orientativos; WeightEngine los ajusta dinámicamente.
    """
    HIGHWAY          = 100   # autopista / periférico
    MAIN_AVENUE      = 80    # avenida principal
    SECONDARY_AVENUE = 50    # avenida secundaria
    STREET           = 20    # calle residencial
    ALLEY            = 5     # callejón / acceso


# ── RoadSegment ───────────────────────────────────────────────────────────────

@dataclass
class RoadSegment:
    """
    Arista dirigida del grafo vial.
    Representa un tramo de calle entre dos intersecciones.

    En Neo4j esto es una relación (:Intersection)-[:ROAD]->(:Intersection)
    con propiedades equivalentes a los atributos de esta clase.

    Attributes
    ----------
    segment_id       : ID único del segmento (coincide con el ID en Neo4j).
    from_node_id     : ID de la intersección de origen.
    to_node_id       : ID de la intersección de destino.
    category         : Categoría de la vía (determina peso base).
    length_m         : Longitud del segmento en metros.
    speed_limit_kmh  : Velocidad máxima permitida.
    allowed_turns    : Giros permitidos al final del segmento.
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
    allowed_turns:   List[Turn]  = field(default_factory=list)
    forbidden_turns: List[Turn]  = field(default_factory=list)
    has_bike_lane:   bool        = False
    has_sidewalk:    bool        = True

    @property
    def base_weight(self) -> float:
        """Peso base estático derivado de la categoría de la vía."""
        return float(self.category.value)

    def current_weight(self, ctx: TrafficContext) -> float:
        """
        Peso efectivo del segmento dado el contexto actual.
        Delega el cálculo a WeightEngine para mantener esta clase
        como modelo de datos puro (sin lógica de negocio).

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo actual.

        Returns
        -------
        Peso efectivo como float.

        Note
        ----
        La implementación real llama a WeightEngine.compute_road_weight(self, ctx).
        Se deja aquí como stub para que el grafo pueda consultarlo
        sin importar WeightEngine directamente (evita import circular).
        """
        # TODO: implementar — llamar WeightEngine.compute_road_weight(self, ctx)
        raise NotImplementedError

    def is_turn_allowed(self, turn: Turn) -> bool:
        """
        Valida si un giro está permitido en este segmento.
        Caso esquina crítico: giros prohibidos deben bloquear
        el cambio de fase en SafetyGuard.

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


# ── Intersection ──────────────────────────────────────────────────────────────

@dataclass
class Intersection:
    """
    Nodo del grafo vial. Representa una intersección física con semáforo.

    En Neo4j esto es un nodo (:Intersection) con propiedades equivalentes.

    Cada ciclo del pipeline:
      1. WeightEngine agrega la presión de todas las entidades presentes.
      2. SafetyGuard valida que el cambio de fase sea seguro.
      3. adjust_phase() actualiza current_phase si se supera el umbral.

    Attributes
    ----------
    node_id          : ID único (coincide con el nodo en Neo4j).
    name             : Nombre descriptivo (ej: "Av. Patria y Periférico").
    latitude         : Latitud geográfica.
    longitude        : Longitud geográfica.
    incoming_segments: Segmentos que llegan a esta intersección.
    current_phase    : Fase actual del semáforo.
    pressure         : Presión acumulada en el ciclo actual.
    min_green_seconds: Tiempo mínimo de verde (sobreescrito por SafetyGuard
                       si hay peatones vulnerables).
    """

    node_id:           str
    name:              str
    latitude:          float
    longitude:         float
    incoming_segments: List[RoadSegment] = field(default_factory=list)
    current_phase:     Phase             = Phase.RED
    pressure:          float             = 0.0
    min_green_seconds: int               = 15

    def adjust_phase(self, engine: "WeightEngine", ctx: TrafficContext,
                     entities: List[TrafficEntity]) -> Phase:
        """
        Recalcula la fase del semáforo para este ciclo.
        Flujo:
          1. engine.aggregate_pressure(entities, self) → presión total
          2. engine.should_change_phase(presión, umbral) → bool
          3. Si debe cambiar: actualiza current_phase

        Parameters
        ----------
        engine   : Instancia de WeightEngine (funciones puras).
        ctx      : Contexto ambiental del ciclo.
        entities : Entidades presentes en esta intersección ahora.

        Returns
        -------
        La nueva fase (o la misma si no hubo cambio).
        """
        # TODO: implementar lógica de cambio de fase
        raise NotImplementedError

    def get_cycle_duration(self, ctx: TrafficContext) -> int:
        """
        Duración total del ciclo en segundos (verde + amarillo + rojo).
        Varía según hora, clima y presión acumulada.

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo.

        Returns
        -------
        Duración del ciclo en segundos.
        """
        # TODO: implementar — considerar ctx.is_rush_hour, ctx.is_late_night
        raise NotImplementedError

    def neighbors(self) -> List[str]:
        """
        IDs de las intersecciones alcanzables desde esta.
        Usado por TrafficGraph para propagar la ola verde.

        Returns
        -------
        Lista de node_id de intersecciones vecinas.
        """
        return [seg.to_node_id for seg in self.incoming_segments]