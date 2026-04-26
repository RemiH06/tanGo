"""
tests/sim0/timer_algorithm.py
------------------------------
Algoritmo de semáforos con timers fijos — sim0.

Modela cómo funcionan la mayoría de semáforos actualmente:
  - Cada semáforo tiene un ciclo fijo independiente de la demanda.
  - El ciclo no varía por contexto (hora, lluvia, etc.).
  - No hay coordinación entre semáforos vecinos.
  - No hay green wave.
  - No hay prioridad para peatones, emergencias o vulnerables.

Se usa como baseline comparativo contra sim1 (tanGo).

Parámetros típicos en Guadalajara (fuente: SIOP Jalisco):
  - Verde: 30s en calles, 45s en avenidas
  - Amarillo: 3s
  - Rojo: duración del verde del eje contrario + amarillo

En ticks (1 tick ≈ 30s):
  - Verde: 1 tick calles, 1-2 ticks avenidas
  - Amarillo: 1 tick
  - Rojo: espera hasta que le toca
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from collections import defaultdict

from core.road       import Phase, IntersectionType
from core.entities   import Vehicle, Pedestrian
from graph.simulator import TrafficGraph

logger = logging.getLogger(__name__)

# ── Duración de fases en ticks (timers fijos) ─────────────────────────────────
TIMER_GREEN = {
    "master": 2,   # avenidas principales: ~60s (2 ticks)
    "normal": 1,   # calles secundarias:   ~30s (1 tick)
    "blind":  0,   # sin semáforo
}
TIMER_YELLOW = 1   # amarillo siempre 1 tick
TIMER_RED    = 2   # rojo fijo — no depende de demanda


@dataclass
class TimerState:
    """Estado de un semáforo de timer fijo."""
    node_id:       str
    itype:         str
    phase:         str = "red"
    ticks_in:      int = 0
    # Desfase inicial — en timers reales se configura manualmente
    # para intentar evitar que todos cambien al mismo tiempo
    phase_offset:  int = 0

    def tick(self) -> None:
        """Avanza un tick con timer fijo — sin considerar demanda."""
        if self.itype == "blind":
            return   # sin semáforo, sin cambio

        self.ticks_in += 1
        green_dur = TIMER_GREEN[self.itype]

        if self.phase == "red":
            if self.ticks_in >= TIMER_RED:
                self.phase    = "green"
                self.ticks_in = 0

        elif self.phase == "green":
            if self.ticks_in >= green_dur:
                self.phase    = "yellow"
                self.ticks_in = 0

        elif self.phase == "yellow":
            if self.ticks_in >= TIMER_YELLOW:
                self.phase    = "red"
                self.ticks_in = 0


@dataclass
class TimerTickResult:
    """Resultado de un tick del algoritmo de timers."""
    tick_number:    int
    nodes:          dict[str, dict]
    flows:          list[dict]
    total_entities: int
    green_count:    int
    yellow_count:   int
    red_count:      int
    blink_count:    int = 0   # no existe en timers fijos


class TimerAlgorithm:
    """
    Algoritmo de semáforos con timers fijos.
    Baseline para comparar contra TrafficAlgorithm (tanGo).

    No importa cuántas entidades haya — el semáforo sigue su ciclo.
    No hay coordinación entre nodos.
    No hay green wave.
    No hay BLINK — en timers fijos todo sigue aunque no haya nadie.
    """

    def __init__(self, graph: TrafficGraph) -> None:
        self.graph  = graph
        self._tick  = 0
        self._states: dict[str, TimerState] = {}
        self._init_states()

    def _init_states(self) -> None:
        """
        Inicializa los timers con desfases manuales.
        En la realidad estos desfases los configura un técnico de tránsito
        una vez al año — no cambian con el tráfico.
        """
        offsets = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # desfases rotativos
        for i, (node_id, inter) in enumerate(self.graph.intersections.items()):
            itype = inter.intersection_type.value
            state = TimerState(
                node_id      = node_id,
                itype        = itype,
                phase        = "red",
                ticks_in     = offsets[i % len(offsets)],
                phase_offset = offsets[i % len(offsets)],
            )
            self._states[node_id] = state

    def reset(self) -> None:
        self._tick = 0
        self._init_states()

    def run_tick(self, entities_by_node: dict, ctx=None) -> TimerTickResult:
        """
        Avanza un tick — ignora entidades y contexto.
        El ciclo es fijo independientemente de la demanda.
        """
        self._tick += 1

        green_count = yellow_count = red_count = 0
        total_entities = 0
        nodes: dict[str, dict] = {}

        for node_id, state in self._states.items():
            state.tick()
            inter = self.graph.intersections[node_id]
            ents  = entities_by_node.get(node_id, [])
            total_entities += len(ents)

            # Contar fases
            if   state.phase == "green":  green_count  += 1
            elif state.phase == "yellow": yellow_count += 1
            elif state.phase == "red":    red_count    += 1

            # Contar entidades para el panel
            counts: dict[str, int] = defaultdict(int)
            for e in ents:
                if isinstance(e, Vehicle):
                    counts[e.vehicle_type.name] += 1
                elif isinstance(e, Pedestrian):
                    counts["PEDESTRIAN"] += 1
                    if e.is_wheelchair:
                        counts["WHEELCHAIR"] += 1

            nodes[node_id] = {
                "phase":     state.phase,
                "phase_ns":  state.phase,    # sin distinción de eje
                "phase_ew":  "red" if state.phase == "green" else state.phase,
                "active_axis": "ns",
                "signals":   {"N": state.phase, "S": state.phase,
                              "E": "red", "W": "red"},
                "pressure":      0.0,   # timers no calculan presión
                "pressure_own":  0.0,
                "pressure_ns":   0.0,
                "pressure_ew":   0.0,
                "wave_offset_s": 0.0,   # sin green wave
                "threshold":     100.0,
                "ticks_in":      state.ticks_in,
                "timeout":       TIMER_GREEN[state.itype] + TIMER_YELLOW + TIMER_RED,
                "ticks_red":     state.ticks_in if state.phase == "red" else 0,
                "has_light":     state.itype != "blind",
                "itype":         inter.intersection_type,
                "geometry":      inter.geometry,
                "geo_label":     inter.geometry_label,
                "name":          inter.name,
                "lat":           inter.latitude,
                "lon":           inter.longitude,
                "counts":        dict(counts),
                "cluster_id":    None,
            }

        # Flujos (igual que en sim1 — solo para visualización)
        flows = []
        drawn = set()
        for from_id, to_id, _ in self.graph.graph.edges(data=True):
            pair = tuple(sorted([from_id, to_id]))
            if pair in drawn: continue
            drawn.add(pair)
            n_veh = sum(1 for e in entities_by_node.get(from_id, [])
                        if isinstance(e, Vehicle))
            flows.append({"from": pair[0], "to": pair[1],
                          "fwd": n_veh, "bwd": 0})

        return TimerTickResult(
            tick_number    = self._tick,
            nodes          = nodes,
            flows          = flows,
            total_entities = total_entities,
            green_count    = green_count,
            yellow_count   = yellow_count,
            red_count      = red_count,
        )