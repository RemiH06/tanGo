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
    """
    Estado actual del semáforo.

    BLINK  : Amarillo intermitente — modo espera activa.
             Se activa cuando no hay entidades en la intersección
             ni en sus vecinos. Indica "sin tráfico, precaución".
             En semáforos reales es el modo nocturno o de baja demanda.
             No requiere exclusión mutua entre ejes — todos parpadean.
    """
    GREEN  = "green"
    YELLOW = "yellow"
    RED    = "red"
    BLINK  = "blink"   # amarillo intermitente — sin tráfico


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



class IntersectionType(Enum):
    """
    Tipo de intersección — determina si tiene semáforo y cuánto peso tiene.
    Esta clasificación es ORTOGONAL a IntersectionGeometry: una glorieta
    puede ser MASTER o NORMAL; una T puede ser NORMAL o BLIND.

    MASTER : Cruce de avenidas principales. Semáforo siempre presente.
             Multiplicador de peso × 1.5. Umbral de presión 120.
    NORMAL : Cruce mixto (avenida + calle). Semáforo presente.
             Multiplicador × 1.0. Umbral 100.
    BLIND  : Cruce ciego sin semáforo — calles internas de colonia.
             Solo se mapea para el grafo de rutas. Multiplicador × 0.3.
    """
    MASTER = "master"
    NORMAL = "normal"
    BLIND  = "blind"


class IntersectionGeometry(Enum):
    """
    Geometría física de la intersección — cómo se conectan las vías.
    Afecta el peso base, los giros permitidos y la lógica de fases.

    CROSS      : Cruce en +. Cuatro ramas. El más común en ciudad.
                 Tiene dos ejes (NS y EW) que alternan verde/rojo.
                 Peso base × 1.0.

    T          : Cruce en T. Tres ramas. Una calle termina en otra.
                 Solo un eje principal; el ramal tiene fase propia corta.
                 Peso base × 0.8. No hay giro prohibido en el extremo.

    Y          : Cruce en Y / bifurcación. Tres ramas en ángulo oblicuo.
                 Común en zonas de trazo irregular o histórico.
                 Peso base × 0.8. Fases ajustadas por ángulo.

    ROUNDABOUT : Glorieta. N ramas (≥3). El tráfico circula en un sentido.
                 No tiene semáforo propio — el flujo se regula por prioridad.
                 Peso base × 1.2 (absorbe mucho flujo sin semáforo).
                 IntersectionType siempre BLIND o NORMAL.

    PEDESTRIAN : Cruce peatonal en una sola calle (no es cruce de dos calles).
                 Un eje vehicular + fase peatonal.
                 Peso peatón aumentado × 1.4. Sin eje EW vehicular.
                 Puede tener botón de solicitud de verde.

    MULTIWAY   : Cruce de 5 o más ramas. Poco común, alto flujo.
                 Peso base × 1.4. Fases múltiples necesarias.
                 Ejemplo: La Minerva en Guadalajara (glorieta con 8 ramas).

    MERGE      : Incorporación / carril de aceleración. Una vía se une
                 a otra sin cruce perpendicular. Sin semáforo.
                 Solo mapeo para el grafo. Peso base × 0.5.
    """
    CROSS      = "cross"       # + estándar
    T          = "t"           # T (tres ramas)
    Y          = "y"           # Y (bifurcación oblicua)
    ROUNDABOUT = "roundabout"  # glorieta
    PEDESTRIAN = "pedestrian"  # cruce peatonal
    MULTIWAY   = "multiway"    # 5+ ramas
    MERGE      = "merge"       # incorporación sin cruce


# Multiplicadores de peso base por geometría
GEOMETRY_WEIGHT_MULTIPLIER: dict[IntersectionGeometry, float] = {
    IntersectionGeometry.CROSS:      1.0,
    IntersectionGeometry.T:          0.8,
    IntersectionGeometry.Y:          0.8,
    IntersectionGeometry.ROUNDABOUT: 1.2,
    IntersectionGeometry.PEDESTRIAN: 0.6,
    IntersectionGeometry.MULTIWAY:   1.4,
    IntersectionGeometry.MERGE:      0.5,
}

# Geometrías que tienen semáforo por defecto
GEOMETRY_HAS_LIGHT: dict[IntersectionGeometry, bool] = {
    IntersectionGeometry.CROSS:      True,
    IntersectionGeometry.T:          True,
    IntersectionGeometry.Y:          False,
    IntersectionGeometry.ROUNDABOUT: False,
    IntersectionGeometry.PEDESTRIAN: True,
    IntersectionGeometry.MULTIWAY:   True,
    IntersectionGeometry.MERGE:      False,
}

# Semáforos presentes por geometría (qué direcciones tienen semáforo físico)
# CROSS:      N, S, E, O — 4 semáforos, 2 ejes
# T:          N, S, E    — 3 semáforos (falta la rama ciega)
# PEDESTRIAN: solo E y O (vía principal) — el cruce peatonal es perpendicular
# MULTIWAY:   los 4 cardinales más señalización extra (modelamos con 4)
# Dirs por geometría — strings para evitar forward reference con CardinalDirection
_GEO_DIRS = {
    "cross":      ["N","S","E","W"],
    "t":          ["N","S","E"],
    "y":          ["N","E","W"],
    "roundabout": [],
    "pedestrian": ["E","W"],
    "multiway":   ["N","S","E","W"],
    "merge":      [],
}



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

class TrafficAxis(Enum):
    """
    Eje de circulación — agrupa las direcciones que son mutuamente compatibles.

    GARANTÍA DE EXCLUSIÓN MUTUA:
      Solo UN eje puede estar en verde en un momento dado.
      Si NS=GREEN entonces EW=RED, y viceversa.
      NUNCA pueden estar ambos en verde simultáneamente.

    Esto refleja la realidad física: en un cruce en + los carros
    que van N↔S y los que van E↔O se cruzarían, por eso nunca
    pueden tener luz verde al mismo tiempo.
    """
    NS = "ns"   # Norte y Sur — eje vertical
    EW = "ew"   # Este y Oeste — eje horizontal


# Semáforos presentes y su eje según geometría de la intersección.
# Clave: (geometría) → lista de (dirección_label, eje)
# Esto define exactamente qué semáforos físicos existen en cada tipo de cruce.
GEOMETRY_SIGNALS: dict = {
    # Cruce en + : 4 semáforos, 2 ejes
    IntersectionGeometry.CROSS:      [("N", TrafficAxis.NS), ("S", TrafficAxis.NS),
                                       ("E", TrafficAxis.EW), ("W", TrafficAxis.EW)],
    # Cruce en T : 3 semáforos — la rama que termina + los dos lados del eje principal
    IntersectionGeometry.T:          [("N", TrafficAxis.NS), ("S", TrafficAxis.NS),
                                       ("E", TrafficAxis.EW)],
    # Bifurcación Y : 3 semáforos, misma lógica que T pero ángulo oblicuo
    IntersectionGeometry.Y:          [("N", TrafficAxis.NS), ("E", TrafficAxis.EW),
                                       ("W", TrafficAxis.EW)],
    # Glorieta: sin semáforos — flujo continuo por prioridad al circulante
    IntersectionGeometry.ROUNDABOUT: [],
    # Cruce peatonal: 1 semáforo vehicular (NS) + fase peatonal implícita
    IntersectionGeometry.PEDESTRIAN: [("N", TrafficAxis.NS)],
    # Multiway (5+ ramas): 4 semáforos como CROSS — simplificación válida
    IntersectionGeometry.MULTIWAY:   [("N", TrafficAxis.NS), ("S", TrafficAxis.NS),
                                       ("E", TrafficAxis.EW), ("W", TrafficAxis.EW)],
    # Incorporación: sin semáforos
    IntersectionGeometry.MERGE:      [],
}


# Duración de fase amarilla — fija por seguridad vial
_YELLOW_DURATION_S: int = 3

# Rangos de duración de ciclo verde según condición
_GREEN_RUSH_HOUR_S:  int = 45
_GREEN_NORMAL_S:     int = 30
_GREEN_LATE_NIGHT_S: int = 20
_GREEN_MIN_S:        int = 7    # nunca menos de esto por seguridad

# ── Timeout de semáforo ───────────────────────────────────────────────────────
# Si la presión no alcanza el umbral en este tiempo, el semáforo cambia de
# todas formas para garantizar que nadie espere indefinidamente.
#
# El timeout es INVERSAMENTE proporcional al umbral de la intersección:
#   umbral alto (MASTER=120) → timeout más corto (la gente espera menos en
#     intersecciones importantes porque hay más tráfico rotando)
#   umbral bajo (NORMAL=100) → timeout estándar
#
# Fórmula: timeout_ticks = BASE_TIMEOUT / (threshold / 100)
# Con BASE_TIMEOUT=8: MASTER→6.6 ticks, NORMAL→8 ticks
# En ticks de simulación (cada tick ≈ 1 ciclo de semáforo real ~30-60s)
_RED_TIMEOUT_BASE_TICKS: int = 8    # ticks máximos en rojo sin cambiar


def _resolve_signals(geometry: "IntersectionGeometry") -> list:
    """
    Retorna las CardinalDirection que tienen semaforo fisico para una geometria.
    Usa strings internamente para evitar forward reference.
    """
    _m = {"N": CardinalDirection.NORTH, "S": CardinalDirection.SOUTH,
          "E": CardinalDirection.EAST,  "W": CardinalDirection.WEST}
    return [_m[d] for d in _GEO_DIRS.get(geometry.value, ["N","S","E","W"])]


@dataclass
class Intersection:
    """
    Nodo del grafo vial.

    Tres tipos (IntersectionType):
      MASTER → cruce de avenidas, siempre tiene semáforo, umbral de presión
               más alto porque absorbe más tráfico.
      NORMAL → cruce mixto, tiene semáforo, umbral estándar.
      BLIND  → cruce ciego sin semáforo — el tráfico fluye por probabilidad
               hacia la salida más cercana a una avenida.
               adjust_phase() es no-op en intersecciones BLIND.

    En Neo4j es un nodo (:Intersection) con propiedades equivalentes.

    Attributes
    ----------
    node_id           : ID único (coincide con el nodo en Neo4j).
    name              : Nombre descriptivo.
    latitude          : Latitud geográfica.
    longitude         : Longitud geográfica.
    intersection_type : Tipo de intersección (MASTER/NORMAL/BLIND).
    incoming_segments : Segmentos que llegan a esta intersección.
    current_phase     : Fase actual del semáforo (N/A para BLIND).
    pressure          : Presión calculada en el último ciclo.
    min_green_seconds : Tiempo mínimo de verde.
    _phase_started_at : Timestamp de inicio de fase actual (interno).
    """

    node_id:           str
    name:              str
    latitude:          float
    longitude:         float
    intersection_type: IntersectionType     = IntersectionType.NORMAL
    geometry:          IntersectionGeometry = IntersectionGeometry.CROSS
    incoming_segments: List[RoadSegment]    = field(default_factory=list)
    current_phase:     Phase                = Phase.RED
    pressure:          float                = 0.0
    min_green_seconds: int                  = _GREEN_MIN_S
    _phase_started_at: datetime          = field(
        default_factory=datetime.now, init=False, repr=False
    )
    _ticks_in_phase:    int   = field(default=0,   init=False, repr=False)
    _timeout_triggered: bool  = field(default=False, init=False, repr=False)
    _active_axis: "TrafficAxis" = field(default=None, init=False, repr=False)
    # Presión por eje — determina cuál gana el verde en exclusión mutua
    _pressure_ns: float = field(default=0.0, init=False, repr=False)
    _pressure_ew: float = field(default=0.0, init=False, repr=False)
    # Ticks consecutivos sin entidades → activa BLINK
    _ticks_empty: int   = field(default=0,   init=False, repr=False)

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Latitud fuera de rango: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Longitud fuera de rango: {self.longitude}")
        self._active_axis = TrafficAxis.NS

    @property
    def signals(self) -> list[tuple[str, TrafficAxis]]:
        """
        Semáforos físicos de esta intersección como lista de (dirección, eje).
        Derivado de la geometría — glorietas y merges retornan lista vacía.
        """
        return GEOMETRY_SIGNALS.get(self.geometry, [])

    @property
    def signal_summary(self) -> dict[str, str]:
        """
        Fase por dirección cardinal con EXCLUSIÓN MUTUA GARANTIZADA.

        Solo el eje activo puede estar en verde. El contrario está
        siempre en rojo. Nunca pueden coincidir N=GREEN y E=GREEN.

        Ejemplo CROSS, eje NS activo, fase GREEN:
            {"N": "green", "S": "green", "E": "red", "W": "red"}

        Ejemplo CROSS, fase YELLOW (transición):
            {"N": "yellow", "S": "yellow", "E": "yellow", "W": "yellow"}

        Returns
        -------
        Dict dirección → fase como string.
        """
        if not self.has_traffic_light:
            return {d: "none" for d, _ in self.signals}
        result = {}
        for sig_dir, sig_axis in self.signals:
            if self.current_phase == Phase.BLINK:
                # BLINK: todos parpadean en amarillo — no hay exclusión mutua
                # porque no hay tráfico que proteger
                result[sig_dir] = "blink"
            elif self.current_phase == Phase.YELLOW:
                result[sig_dir] = "yellow"
            elif (self.current_phase == Phase.GREEN
                  and sig_axis == self._active_axis):
                # Solo el eje activo está en verde — exclusión mutua garantizada
                result[sig_dir] = "green"
            else:
                # Eje contrario SIEMPRE en rojo cuando hay tráfico
                result[sig_dir] = "red"
        return result

    @property
    def phase_ns(self) -> Phase:
        """
        Fase del eje Norte-Sur.
        Si EW está en verde, este es RED (exclusión mutua).
        Si BLINK, retorna BLINK — todos los ejes parpadean.
        """
        if not self.has_traffic_light: return Phase.RED
        if self.current_phase == Phase.BLINK:  return Phase.BLINK
        if self.current_phase == Phase.YELLOW: return Phase.YELLOW
        return (self.current_phase if self._active_axis == TrafficAxis.NS
                else Phase.RED)

    @property
    def phase_ew(self) -> Phase:
        """
        Fase del eje Este-Oeste.
        Si NS está en verde, este es RED (exclusión mutua).
        Si BLINK, retorna BLINK — todos los ejes parpadean.
        """
        if not self.has_traffic_light: return Phase.RED
        if self.current_phase == Phase.BLINK:  return Phase.BLINK
        if self.current_phase == Phase.YELLOW: return Phase.YELLOW
        return (self.current_phase if self._active_axis == TrafficAxis.EW
                else Phase.RED)

    @property
    def has_traffic_light(self) -> bool:
        """
        True si esta intersección tiene semáforo físico.
        Depende tanto del tipo como de la geometría:
          - BLIND nunca tiene semáforo.
          - Glorietas, incorporaciones y bifurcaciones Y tampoco.
          - El resto sí, salvo que el tipo lo fuerce a BLIND.
        """
        if self.intersection_type == IntersectionType.BLIND:
            return False
        return GEOMETRY_HAS_LIGHT.get(self.geometry, True)

    @property
    def pressure_threshold(self) -> float:
        """
        Umbral de presión para cambiar de fase.
        Las intersecciones maestras necesitan más demanda para cambiar
        porque absorben más tráfico naturalmente.
        """
        return {
            IntersectionType.MASTER: 120.0,
            IntersectionType.NORMAL: 100.0,
            IntersectionType.BLIND:  999.0,  # nunca cambia sola
        }[self.intersection_type]

    @property
    def weight_multiplier(self) -> float:
        """
        Multiplicador de peso base combinando tipo y geometría.
        Ambos factores se multiplican entre sí.

        Ejemplos:
          MASTER + CROSS      → 1.5 × 1.0 = 1.5
          MASTER + MULTIWAY   → 1.5 × 1.4 = 2.1  (La Minerva)
          NORMAL + ROUNDABOUT → 1.0 × 1.2 = 1.2
          NORMAL + T          → 1.0 × 0.8 = 0.8
          BLIND  + cualquiera → 0.3 × geo  (siempre bajo)
        """
        type_mult = {
            IntersectionType.MASTER: 1.5,
            IntersectionType.NORMAL: 1.0,
            IntersectionType.BLIND:  0.3,
        }[self.intersection_type]
        geo_mult = GEOMETRY_WEIGHT_MULTIPLIER.get(self.geometry, 1.0)
        return type_mult * geo_mult

    @property
    def geometry_label(self) -> str:
        """Etiqueta corta de la geometría para mostrar en el dashboard."""
        return {
            IntersectionGeometry.CROSS:      "+",
            IntersectionGeometry.T:          "T",
            IntersectionGeometry.Y:          "Y",
            IntersectionGeometry.ROUNDABOUT: "O",
            IntersectionGeometry.PEDESTRIAN: "P",
            IntersectionGeometry.MULTIWAY:   "*",
            IntersectionGeometry.MERGE:      ">",
        }.get(self.geometry, "?")

    @property
    def red_timeout_ticks(self) -> int:
        """
        Número máximo de ticks que puede estar en ROJO sin cambiar.
        Inversamente proporcional al umbral — intersecciones más importantes
        tienen timeout más corto para garantizar rotación del tráfico.

        Fórmula: BASE / (threshold / 100)
          MASTER (120) → 8 / 1.2 ≈ 6 ticks
          NORMAL (100) → 8 / 1.0 = 8 ticks

        Esto garantiza que ningún conductor ni peatón espere
        indefinidamente aunque la presión no alcance el umbral.
        """
        return max(3, int(_RED_TIMEOUT_BASE_TICKS / (self.pressure_threshold / 100)))

    # ── Lógica de fase ────────────────────────────────────────────────────────

    def adjust_phase(self, engine: "WeightEngine", ctx: TrafficContext,
                     entities: List[TrafficEntity],
                     entities_ns: List[TrafficEntity] | None = None,
                     entities_ew: List[TrafficEntity] | None = None) -> Phase:
        """
        Recalcula la fase del semáforo para este ciclo.

        Lógica de exclusión mutua real:
          1. Sin entidades → BLINK (amarillo intermitente)
          2. Con entidades → calcular presión por eje (NS vs EW)
          3. El eje con más presión gana el verde
          4. El eje contrario está en ROJO — siempre, sin excepción
          5. Timeout: si ningún eje supera el umbral en N ticks, forzar
             el verde al eje de mayor presión por equidad

        Parameters
        ----------
        engine      : Motor de pesos.
        ctx         : Contexto ambiental.
        entities    : Todas las entidades (para presión global).
        entities_ns : Entidades en el eje N-S (opcional — se infiere si None).
        entities_ew : Entidades en el eje E-O (opcional — se infiere si None).
        """
        # ── BLIND: sin semáforo, no-op ────────────────────────────────────
        if self.intersection_type == IntersectionType.BLIND:
            self.pressure = engine.aggregate_pressure(entities, self, ctx)
            return self.current_phase

        # ── Sin entidades → BLINK ─────────────────────────────────────────
        if not entities:
            self._ticks_empty += 1
            if self._ticks_empty >= 3:
                if self.current_phase != Phase.BLINK:
                    logger.debug("[%s] Sin trafico → BLINK", self.name)
                self.current_phase = Phase.BLINK
                self.pressure      = 0.0
                self._pressure_ns  = 0.0
                self._pressure_ew  = 0.0
            return self.current_phase
        else:
            # Hay entidades → salir de BLINK si estaba activo
            if self.current_phase == Phase.BLINK:
                logger.info("[%s] Trafico detectado — saliendo de BLINK", self.name)
                self.current_phase = Phase.RED
                self._ticks_in_phase = 0
            self._ticks_empty = 0

        # ── Presión global ────────────────────────────────────────────────
        self.pressure = engine.aggregate_pressure(entities, self, ctx)

        # ── Presión por eje (exclusión mutua basada en demanda real) ──────
        # Si no se proporcionan entidades por eje, se divide por dirección
        # de los vehículos como aproximación: N/S → NS, E/W → EW.
        # En producción (KAN-10) esto vendría de detectores por carril.
        from core.entities import Vehicle
        from core.entities import Direction as Dir

        ns_dirs = {Dir.NORTH, Dir.SOUTH}
        ew_dirs = {Dir.EAST,  Dir.WEST}

        if entities_ns is None:
            entities_ns = [e for e in entities
                           if isinstance(e, Vehicle) and e.direction in ns_dirs]
        if entities_ew is None:
            entities_ew = [e for e in entities
                           if isinstance(e, Vehicle) and e.direction in ew_dirs]

        # Peatones se asignan al eje más ocupado (van a cruzar donde hay más espacio)
        from core.entities import Pedestrian
        peds  = [e for e in entities if isinstance(e, Pedestrian)]
        if len(entities_ns) >= len(entities_ew):
            entities_ns = entities_ns + peds
        else:
            entities_ew = entities_ew + peds

        # Calcular presión de cada eje usando WeightEngine real
        # Si no hay señales EW (geometría T, cruce peatonal), toda la presión es NS
        has_ew = any(axis == TrafficAxis.EW for _, axis in self.signals)
        has_ns = any(axis == TrafficAxis.NS for _, axis in self.signals)

        self._pressure_ns = (
            engine.aggregate_pressure(entities_ns, self, ctx)
            if has_ns and entities_ns else 0.0
        )
        self._pressure_ew = (
            engine.aggregate_pressure(entities_ew, self, ctx)
            if has_ew and entities_ew else 0.0
        )

        # ── Decidir qué eje tiene mayor demanda ──────────────────────────
        # El eje ganador es el de mayor presión.
        # Si las presiones son iguales, mantener el eje actual para estabilidad.
        if self._pressure_ns > self._pressure_ew:
            winner_axis = TrafficAxis.NS
            winner_pressure = self._pressure_ns
        elif self._pressure_ew > self._pressure_ns:
            winner_axis = TrafficAxis.EW
            winner_pressure = self._pressure_ew
        else:
            winner_axis = self._active_axis   # empate → mantener
            winner_pressure = self.pressure

        # ── Timeout de rojo ───────────────────────────────────────────────
        self._ticks_in_phase += 1
        timeout_forced = (
            self.current_phase == Phase.RED
            and self._ticks_in_phase >= self.red_timeout_ticks
            and not self._timeout_triggered
        )
        if timeout_forced:
            self._timeout_triggered = True
            logger.info(
                "[%s][%s] TIMEOUT → forzando verde al eje %s "
                "(NS=%.1f, EW=%.1f)",
                self.name, self.intersection_type.value,
                winner_axis.value, self._pressure_ns, self._pressure_ew,
            )

        should_change = (
            timeout_forced
            or engine.should_change_phase(winner_pressure,
                                          threshold=self.pressure_threshold)
        )
        seconds_in_phase = (datetime.now() - self._phase_started_at).total_seconds()

        # ── Si el eje ganador cambió mientras estábamos en verde, ─────────
        # pasar a amarillo para ceder el paso al nuevo ganador.
        # Esto implementa el cambio dinámico basado en presión real.
        axis_switch = (
            self.current_phase == Phase.GREEN
            and winner_axis != self._active_axis
            and abs(self._pressure_ns - self._pressure_ew) > 20.0
        )

        new_phase = self._next_phase(
            should_change    = should_change,
            seconds_in_phase = seconds_in_phase,
            ctx              = ctx,
            force_yellow     = axis_switch,
        )

        if new_phase != self.current_phase:
            reason = ("timeout"       if timeout_forced else
                      "cambio de eje" if axis_switch    else
                      f"presión={winner_pressure:.1f}")
            logger.info(
                "[%s][%s] %s → %s | eje=%s | NS=%.1f EW=%.1f | umbral=%.0f",
                self.name, self.intersection_type.value,
                self.current_phase.value, new_phase.value,
                winner_axis.value,
                self._pressure_ns, self._pressure_ew,
                self.pressure_threshold,
            )
            self.current_phase      = new_phase
            self._phase_started_at  = datetime.now()
            self._ticks_in_phase    = 0
            self._timeout_triggered = False

            # Al pasar a verde, activar el eje ganador (exclusión mutua)
            if new_phase == Phase.GREEN:
                self._active_axis = winner_axis

            # Al pasar a rojo, preparar el eje opuesto para el próximo ciclo
            elif new_phase == Phase.RED:
                self._active_axis = (
                    TrafficAxis.EW if self._active_axis == TrafficAxis.NS
                    else TrafficAxis.NS
                )

        return self.current_phase

    def receive_neighbor_signal(self, neighbor_pressure: float,
                                 distance_m: float,
                                 speed_kmh: float) -> float:
        """
        Recibe la señal de presión de un nodo vecino y calcula
        cuánto influye en la decisión de fase propia.

        Implementa la lógica de "mente colmena": si un vecino tiene
        alta presión y los vehículos llegarán pronto (offset pequeño),
        esta intersección debe prepararse para recibirlos.

        Fórmula:
            influencia = presión_vecino × factor_proximidad
            factor_proximidad = 1 / (1 + tiempo_llegada_s / 30)

        Un vecino a 10 segundos tiene influencia ≈ 0.75.
        Un vecino a 60 segundos tiene influencia ≈ 0.33.
        Un vecino a 120 segundos tiene influencia ≈ 0.20.

        Parameters
        ----------
        neighbor_pressure : Presión actual del vecino.
        distance_m        : Distancia al vecino en metros.
        speed_kmh         : Velocidad límite del segmento entre ambos.

        Returns
        -------
        Presión de influencia (0.0 si el vecino no tiene efecto relevante).
        """
        if speed_kmh <= 0 or distance_m <= 0:
            return 0.0
        travel_time_s  = distance_m / (speed_kmh / 3.6)
        proximity_factor = 1.0 / (1.0 + travel_time_s / 30.0)
        influence = neighbor_pressure * proximity_factor
        logger.debug(
            "[%s] señal vecinal: presión=%.1f dist=%.0fm t=%.0fs influencia=%.1f",
            self.name, neighbor_pressure, distance_m, travel_time_s, influence
        )
        return influence

    def _next_phase(self, should_change: bool,
                    seconds_in_phase: float,
                    ctx: TrafficContext,
                    force_yellow: bool = False) -> Phase:
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
            if should_change:
                return Phase.GREEN

        elif self.current_phase == Phase.GREEN:
            # Ceder el paso al eje de mayor presión
            if force_yellow:
                return Phase.YELLOW
            if not should_change and seconds_in_phase >= green_duration:
                return Phase.YELLOW
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