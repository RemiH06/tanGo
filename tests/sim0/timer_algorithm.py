"""
tests/sim0/timer_algorithm.py
------------------------------
Algoritmo de semáforos con timers fijos — sim0.

Modela cómo funcionan los semáforos actualmente en Guadalajara:
  - Ciclo fijo predefinido por tipo de intersección y hora del día.
  - Sin coordinación entre semáforos vecinos.
  - Sin green wave.
  - Sin prioridad para peatones, emergencias o vulnerabilidades.

Calibrado según HCM 7th Edition + estimados SEMOVI GDL:
  - MASTER hora pico:  verde 60s, amarillo 30s, rojo 60s  (ciclo 90s)
  - MASTER normal:     verde 60s, amarillo 30s, rojo 90s  (ciclo 120s)
  - NORMAL hora pico:  verde 30s, amarillo 30s, rojo 60s  (ciclo 60s)
  - NORMAL normal:     verde 30s, amarillo 30s, rojo 90s  (ciclo 75s)

En ticks (1 tick = 30s simulados).

Los parámetros son editables en timer_config.json sin tocar este archivo.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from core.road       import Phase, IntersectionType
from core.entities   import Vehicle, Pedestrian
from graph.simulator import TrafficGraph
from core.context    import TrafficContext

# ── Cargar timer_config.json ──────────────────────────────────────────────────

_CFG_PATH = Path(__file__).parent / "timer_config.json"

def _load_timer_cfg() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH, encoding="utf-8") as f:
            return json.load(f)
    # Defaults si no existe el JSON
    return {
        "MASTER": {
            "rush_hour":  {"green_ticks": 2, "yellow_ticks": 1, "red_ticks": 2},
            "normal":     {"green_ticks": 2, "yellow_ticks": 1, "red_ticks": 3},
            "late_night": {"green_ticks": 1, "yellow_ticks": 1, "red_ticks": 2},
        },
        "NORMAL": {
            "rush_hour":  {"green_ticks": 1, "yellow_ticks": 1, "red_ticks": 2},
            "normal":     {"green_ticks": 1, "yellow_ticks": 1, "red_ticks": 3},
            "late_night": {"green_ticks": 1, "yellow_ticks": 1, "red_ticks": 4},
        },
        "phase_offsets": {"pattern": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]},
    }

_CFG = _load_timer_cfg()


def _get_timing(itype_str: str, ctx: TrafficContext) -> dict:
    """Retorna los tiempos de fase según tipo e intersección y contexto."""
    itype_cfg = _CFG.get(itype_str, _CFG.get("NORMAL", {}))
    if ctx.is_rush_hour:
        return itype_cfg.get("rush_hour",  {"green_ticks": 1, "yellow_ticks": 1, "red_ticks": 2})
    if ctx.is_late_night:
        return itype_cfg.get("late_night", {"green_ticks": 1, "yellow_ticks": 1, "red_ticks": 4})
    return itype_cfg.get("normal",         {"green_ticks": 1, "yellow_ticks": 1, "red_ticks": 3})


# ── Estado de semáforo ────────────────────────────────────────────────────────

@dataclass
class TimerState:
    """Estado de un semáforo de timer fijo."""
    node_id:      str
    itype:        str          # "master" | "normal" | "blind"
    phase:        str = "red"
    ticks_in:     int = 0
    phase_offset: int = 0      # desfase inicial en ticks

    # Tiempos actuales (se actualizan según contexto)
    green_ticks:  int = 2
    yellow_ticks: int = 1
    red_ticks:    int = 3

    def update_timing(self, ctx: TrafficContext) -> None:
        """Actualiza los tiempos según el contexto actual."""
        itype_key = self.itype.upper()
        timing = _get_timing(itype_key, ctx)
        self.green_ticks  = timing["green_ticks"]
        self.yellow_ticks = timing["yellow_ticks"]
        self.red_ticks    = timing["red_ticks"]

    @property
    def cycle_ticks(self) -> int:
        return self.green_ticks + self.yellow_ticks + self.red_ticks

    def tick(self, ctx: TrafficContext | None = None) -> None:
        """Avanza un tick con timer fijo — sin considerar demanda."""
        if self.itype == "blind":
            self.phase = "blink"
            return

        if ctx:
            self.update_timing(ctx)

        self.ticks_in += 1

        if self.phase == "red":
            if self.ticks_in >= self.red_ticks:
                self.phase   = "green"
                self.ticks_in = 0

        elif self.phase == "green":
            if self.ticks_in >= self.green_ticks:
                self.phase   = "yellow"
                self.ticks_in = 0

        elif self.phase == "yellow":
            if self.ticks_in >= self.yellow_ticks:
                self.phase   = "red"
                self.ticks_in = 0

        elif self.phase == "blink":
            # BLINK solo en nodos BLIND — no cambia
            pass


@dataclass
class TimerTickResult:
    """Resultado de un tick del algoritmo de timers."""
    tick_number:    int
    nodes:          dict
    flows:          list
    total_entities: int
    green_count:    int
    yellow_count:   int
    red_count:      int
    blink_count:    int = 0


# ── Algoritmo ─────────────────────────────────────────────────────────────────

class TimerAlgorithm:
    """
    Algoritmo de semáforos con timers fijos calibrados según HCM.

    Los tiempos varían por:
      - Tipo de intersección (MASTER / NORMAL / BLIND)
      - Contexto (hora pico / normal / madrugada)

    Lo que NO varía:
      - La demanda actual de vehículos — el timer la ignora completamente.
      - La coordinación con vecinos — cada semáforo es independiente.
      - Las emergencias — no tienen prioridad.
    """

    def __init__(self, graph: TrafficGraph) -> None:
        self.graph  = graph
        self._tick  = 0
        self._states: dict[str, TimerState] = {}
        self._init_states()

    def _init_states(self) -> None:
        offsets = _CFG.get("phase_offsets", {}).get(
            "pattern", [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        )
        for i, (node_id, inter) in enumerate(self.graph.intersections.items()):
            itype = inter.intersection_type.value   # "master"|"normal"|"blind"
            offset = offsets[i % len(offsets)]
            state = TimerState(
                node_id      = node_id,
                itype        = itype,
                phase        = "red",
                ticks_in     = offset,
                phase_offset = offset,
            )
            # Aplicar timing inicial (sin contexto — usará "normal")
            state.green_ticks  = _get_timing(itype.upper(), _mock_normal_ctx()).get("green_ticks",  2)
            state.yellow_ticks = _get_timing(itype.upper(), _mock_normal_ctx()).get("yellow_ticks", 1)
            state.red_ticks    = _get_timing(itype.upper(), _mock_normal_ctx()).get("red_ticks",    3)
            self._states[node_id] = state

    def reset(self) -> None:
        self._tick = 0
        self._init_states()

    def run_tick(self, entities_by_node: dict,
                 ctx: TrafficContext | None = None) -> TimerTickResult:
        """
        Avanza un tick — ignora entidades, respeta el contexto solo
        para ajustar la duración de las fases.

        Después de avanzar todos los timers, aplica las garantías
        mínimas de verdes y rojos definidas en timer_config.json:
          - Si hay menos de min_green_count verdes → fuerza verde
            en los nodos que llevan más tiempo en rojo.
          - Si hay menos de min_red_count rojos → fuerza rojo
            en los nodos que llevan más tiempo en verde.
        """
        coord = _CFG.get("coordination", {})
        min_green = int(coord.get("min_green_count", 2))
        min_red   = int(coord.get("min_red_count",   2))

        self._tick += 1
        green_count = yellow_count = red_count = blink_count = 0
        total_entities = 0
        nodes: dict = {}

        # Paso 1: avanzar todos los timers normalmente
        for node_id, state in self._states.items():
            state.tick(ctx)

        # Paso 2: contar fases actuales (solo semaforizados)
        signaled = {nid: s for nid, s in self._states.items()
                    if s.itype != "blind"}
        cur_greens = [nid for nid, s in signaled.items() if s.phase == "green"]
        cur_reds   = [nid for nid, s in signaled.items() if s.phase == "red"]

        # Paso 3: aplicar mínimo de verdes
        # Si faltan verdes, forzar verde en los nodos con más ticks en rojo
        if len(cur_greens) < min_green:
            needed = min_green - len(cur_greens)
            # Ordenar rojos por ticks_in descendente (más tiempo esperando → prioridad)
            candidates = sorted(
                [(nid, s) for nid, s in signaled.items()
                 if s.phase == "red"],
                key=lambda x: -x[1].ticks_in
            )
            for nid, state in candidates[:needed]:
                state.phase    = "green"
                state.ticks_in = 0

        # Paso 4: aplicar mínimo de rojos
        # Si faltan rojos, forzar rojo en los nodos con más ticks en verde
        cur_greens = [nid for nid, s in signaled.items() if s.phase == "green"]
        if len(cur_greens) > len(signaled) - min_red:
            # Cuántos verdes hay de más
            excess = len(cur_greens) - (len(signaled) - min_red)
            candidates = sorted(
                [(nid, s) for nid, s in signaled.items()
                 if s.phase == "green"],
                key=lambda x: -x[1].ticks_in   # más tiempo en verde → primero en ceder
            )
            for nid, state in candidates[:excess]:
                state.phase    = "yellow"
                state.ticks_in = 0

        # Paso 5: construir resultado
        for node_id, state in self._states.items():
            inter = self.graph.intersections[node_id]
            ents  = entities_by_node.get(node_id, [])
            total_entities += len(ents)

            if   state.phase == "green":  green_count  += 1
            elif state.phase == "yellow": yellow_count += 1
            elif state.phase == "red":    red_count    += 1
            else:                         blink_count  += 1

            counts: dict = defaultdict(int)
            for e in ents:
                if isinstance(e, Vehicle):
                    counts[e.vehicle_type.name] += 1
                elif isinstance(e, Pedestrian):
                    counts["PEDESTRIAN"] += 1
                    if e.is_wheelchair:
                        counts["WHEELCHAIR"] += 1

            # Timing actual para mostrar en el panel
            timing = _get_timing(state.itype.upper(), ctx) if ctx else {}

            nodes[node_id] = {
                "phase":     state.phase,
                "phase_ns":  state.phase,
                "phase_ew":  "red" if state.phase == "green" else state.phase,
                "active_axis": "ns",
                "signals":   {"N": state.phase, "S": state.phase,
                              "E": "red" if state.phase == "green" else state.phase,
                              "W": "red" if state.phase == "green" else state.phase},
                "pressure":      0.0,
                "pressure_own":  0.0,
                "pressure_ns":   0.0,
                "pressure_ew":   0.0,
                "wave_offset_s": 0.0,
                "threshold":     100.0,
                "has_light":     state.itype != "blind",
                "itype":         inter.intersection_type,
                "geometry":      inter.geometry,
                "geo_label":     inter.geometry_label,
                "name":          inter.name,
                "lat":           inter.latitude,
                "lon":           inter.longitude,
                "counts":        dict(counts),
                "cluster_id":    None,
                # Info del timer para el panel
                "ticks_red":     state.ticks_in if state.phase == "red"    else 0,
                "ticks_in":      state.ticks_in,
                "timeout":       state.cycle_ticks,
                "green_ticks":   state.green_ticks,
                "yellow_ticks":  state.yellow_ticks,
                "red_ticks":     state.red_ticks,
                "cycle_s":       state.cycle_ticks * 30,
            }

        return TimerTickResult(
            tick_number    = self._tick,
            nodes          = nodes,
            flows          = [],
            total_entities = total_entities,
            green_count    = green_count,
            yellow_count   = yellow_count,
            red_count      = red_count,
            blink_count    = blink_count,
        )


def _mock_normal_ctx():
    """Contexto 'normal' para inicialización sin contexto real."""
    from datetime import datetime
    from core.context import TrafficContext
    return TrafficContext.build(
        timestamp      = datetime(2024, 3, 6, 14, 0),
        temperature_c  = 22.0,
        is_raining     = False,
        wind_speed_kmh = 10.0,
        visibility_m   = 10000.0,
    )