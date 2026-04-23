"""
core/algorithm.py
-----------------
Motor central de tanGo — algoritmo de coordinación de semáforos.

Este módulo es completamente independiente de cualquier simulación o
visualización. Contiene únicamente la lógica que en producción correría
dentro del DAG de Airflow (KAN-10) cada N segundos.

Cualquier simulación (sim1, sim2, ...) o el servidor real simplemente
importa TrafficAlgorithm y lo llama con su propio grafo y contexto.

Arquitectura (inspirada en SCOOT + coordinación vecinal):

    Tick N — tres pasos secuenciales:

    Paso 1 — Presión propia
        WeightEngine.aggregate_pressure() por cada intersección.
        Función pura: mismo input → mismo output.

    Paso 2 — Mente colmena (coordinación vecinal + green wave)
        Cada nodo recibe señales de sus vecinos upstream.
        Si un vecino está en verde, se calcula cuándo llegará su flujo
        usando WeightEngine.compute_green_wave_offset() y se aplica
        un boost de presión proporcional a la urgencia.
        Nodos MASTER propagan señal 1.3× más fuerte.

    Paso 3 — Ajuste de fases con coordinación de cluster
        Nodos del mismo cluster de intersección son interdependientes:
        el de mayor presión gana el verde y los demás ceden.
        Intersection.adjust_phase() aplica la máquina de estados con
        exclusión mutua NS/EW y BLINK cuando no hay tráfico.

Notas de diseño:
    - Todo el estado de fase vive en los objetos Intersection (road.py).
    - Este módulo no guarda estado propio — es stateless entre llamadas.
    - Para producción: llamar run_tick() desde el DAG de Airflow.
    - Para simulación: llamar run_tick() en un loop con entidades generadas.

TODO (sim2 / KAN-14b):
    - Calibrar NEIGHBOR_WEIGHT y WAVE_BOOST_WEIGHT con datos reales de aforo.
    - Agregar capa de RL para ajuste dinámico de pressure_threshold por hora.
    - Transformación senoidal de hora: sin(h/24*2π) / cos(h/24*2π) como
      features para el modelo de ML.
    - Implementar split dinámico NS/EW por intersección (actualmente fijo).

Referencia:
    - SCOOT: Robertson & Bretherton (1991), Transportation Research Record
    - FUSION: TRL / TfL London (2019), multi-source data fusion
    - Green wave: compute_green_wave_offset() en core/weight_engine.py
"""

from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from core.context       import TrafficContext
from core.entities      import TrafficEntity, Vehicle, Pedestrian
from core.road          import Phase
from core.weight_engine import WeightEngine
from graph.simulator    import TrafficGraph

logger = logging.getLogger(__name__)

# ── Parámetros del algoritmo ──────────────────────────────────────────────────
# Estos son los knobs que la capa de RL ajustaría en sim2.
# Por ahora son constantes calibradas manualmente.

NEIGHBOR_WEIGHT:   float = 0.25   # peso de la señal vecinal sobre presión propia
WAVE_BOOST_WEIGHT: float = 0.15   # peso del boost de green wave
WAVE_URGENCY_S:    float = 20.0   # segundos de referencia para urgencia (offset < esto → alto boost)
MASTER_AMPLIFIER:  float = 1.3    # nodos MASTER amplifican su señal × este factor
CLUSTER_YIELD:     float = 0.3    # el nodo perdedor del cluster reduce su presión a este factor


# ── TickResult ────────────────────────────────────────────────────────────────

@dataclass
class NodeState:
    """
    Estado de una intersección al final de un tick.
    Inmutable — snapshot para logging, visualización o base de datos.
    """
    node_id:        str
    phase:          str            # "green" | "yellow" | "red" | "blink"
    phase_ns:       str
    phase_ew:       str
    active_axis:    str            # "ns" | "ew"
    signals:        dict[str, str] # {"N":"green","S":"green","E":"red","W":"red"}
    pressure:       float          # presión combinada (propia + vecinal)
    pressure_own:   float          # presión solo de entidades locales
    pressure_ns:    float          # presión del eje N-S
    pressure_ew:    float          # presión del eje E-O
    wave_offset_s:  float          # segundos hasta que llegue la ola del vecino verde
    threshold:      float          # umbral de presión para cambiar de fase
    ticks_in_phase: int
    timeout_ticks:  int
    cluster_id:     str | None
    entity_counts:  dict[str, int] # conteo por tipo de entidad


@dataclass
class TickResult:
    """
    Resultado completo de un tick del algoritmo.
    Contiene el estado de todos los nodos y los flujos entre ellos.
    """
    tick_number:    int
    nodes:          dict[str, NodeState]
    flows:          list[dict]
    total_entities: int
    green_count:    int
    yellow_count:   int
    red_count:      int
    blink_count:    int


# ── TrafficAlgorithm ──────────────────────────────────────────────────────────

class TrafficAlgorithm:
    """
    Motor de coordinación de semáforos de tanGo.

    Instanciar una vez por ejecución (o por DAG run) y reutilizar.
    No guarda estado entre ticks — todo el estado vive en graph.intersections.

    Uso básico:
        algo = TrafficAlgorithm(graph)
        result = algo.run_tick(entities_by_node, ctx)

    Uso en simulación:
        algo = TrafficAlgorithm(graph)
        for tick in range(n_ticks):
            entities = spawner.generate(graph, ctx)
            result   = algo.run_tick(entities, ctx)
            visualizer.update(result)
    """

    def __init__(self, graph: TrafficGraph) -> None:
        self.graph  = graph
        self.engine = WeightEngine()
        self._tick  = 0

        # Cargar clusters del grafo si existen
        self._clusters       = getattr(graph, "intersection_clusters", {})
        self._node_to_cluster = getattr(graph, "node_to_cluster", {})

        if self._clusters:
            logger.info(
                "TrafficAlgorithm iniciado: %d nodos, %d clusters",
                graph.graph.number_of_nodes(), len(self._clusters)
            )
        else:
            logger.info(
                "TrafficAlgorithm iniciado: %d nodos (sin clusters)",
                graph.graph.number_of_nodes()
            )

    def reset(self) -> None:
        """
        Reinicia el estado de todas las intersecciones al inicio del ciclo.
        Llamar antes de iniciar una nueva simulación con el mismo grafo.
        """
        self._tick = 0
        for inter in self.graph.intersections.values():
            inter.current_phase      = Phase.RED
            inter._ticks_in_phase    = 0
            inter._timeout_triggered = False
            inter._ticks_empty       = 0
            inter._pressure_ns       = 0.0
            inter._pressure_ew       = 0.0
            inter.pressure           = 0.0
        logger.debug("TrafficAlgorithm reiniciado")

    # ── API principal ─────────────────────────────────────────────────────────

    def run_tick(self,
                 entities_by_node: dict[str, list[TrafficEntity]],
                 ctx: TrafficContext) -> TickResult:
        """
        Ejecuta un tick completo del algoritmo de coordinación.

        Este es el único método que necesitas llamar desde la simulación
        o desde el DAG de Airflow. Internamente ejecuta los tres pasos.

        Parameters
        ----------
        entities_by_node : Dict node_id → lista de entidades presentes.
                           Puede venir del simulador o de VisionIngester.
        ctx              : Contexto ambiental del ciclo (hora, clima, etc.)

        Returns
        -------
        TickResult con el estado de todas las intersecciones y flujos.
        """
        self._tick += 1

        # Paso 1: presiones propias
        pressures_own = self._compute_own_pressures(entities_by_node, ctx)

        # Paso 2: coordinación vecinal + green wave
        pressures_combined, wave_offsets = self._propagate_neighbor_signals(
            pressures_own, ctx
        )

        # Paso 3: ajuste de fases con coordinación de cluster
        self._adjust_phases(entities_by_node, pressures_combined, pressures_own, ctx)

        # Construir resultado
        return self._build_result(
            entities_by_node, pressures_own,
            pressures_combined, wave_offsets
        )

    # ── Paso 1: presiones propias ─────────────────────────────────────────────

    def _compute_own_pressures(
            self,
            entities_by_node: dict[str, list[TrafficEntity]],
            ctx: TrafficContext) -> dict[str, float]:
        """
        Calcula la presión de cada intersección basada únicamente en
        las entidades presentes en ella — sin influencia de vecinos.
        Usa WeightEngine.aggregate_pressure() que es función pura.
        """
        pressures = {}
        for node_id, inter in self.graph.intersections.items():
            ents = entities_by_node.get(node_id, [])
            p = self.engine.aggregate_pressure(ents, inter, ctx)
            pressures[node_id] = p
            inter.pressure = p
        return pressures

    # ── Paso 2: mente colmena ─────────────────────────────────────────────────

    def _propagate_neighbor_signals(
            self,
            pressures_own: dict[str, float],
            ctx: TrafficContext
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Implementa la coordinación tipo mente colmena:

        Cada nodo suma a su presión propia:
          a) Influencia de vecinos upstream ponderada por proximidad.
             Intersection.receive_neighbor_signal() calcula el factor
             de decaimiento por distancia y tiempo de viaje.
          b) Boost de green wave: si el vecino está en verde, el flujo
             se aproxima — preparar el verde anticipadamente.

        Los nodos MASTER amplifican su señal MASTER_AMPLIFIER veces
        porque representan corredores de mayor capacidad.

        Returns
        -------
        (pressures_combined, wave_offsets)
          pressures_combined : presión total por nodo (propia + vecinal + wave)
          wave_offsets       : segundos hasta la ola verde por nodo
        """
        pressures_combined: dict[str, float] = {}
        wave_offsets:       dict[str, float] = {}

        for node_id, inter in self.graph.intersections.items():
            combined   = pressures_own[node_id]
            min_offset = float("inf")

            for from_id, to_id, edge_data in self.graph.graph.edges(data=True):
                if to_id != node_id:
                    continue
                neighbor_id    = from_id
                neighbor_inter = self.graph.intersections.get(neighbor_id)
                if not neighbor_inter or neighbor_id not in pressures_own:
                    continue

                seg = edge_data["segment"]

                # a) Señal vecinal ponderada por proximidad
                influence = inter.receive_neighbor_signal(
                    neighbor_pressure = pressures_own[neighbor_id],
                    distance_m        = seg.length_m,
                    speed_kmh         = seg.speed_limit_kmh,
                )
                if neighbor_inter.intersection_type.value == "master":
                    influence *= MASTER_AMPLIFIER

                combined += influence * NEIGHBOR_WEIGHT

                # b) Green wave boost
                if neighbor_inter.current_phase.value == "green":
                    try:
                        offset_s = self.engine.compute_green_wave_offset(
                            distance_m      = seg.length_m,
                            speed_limit_kmh = seg.speed_limit_kmh,
                        )
                        urgency    = 1.0 / (1.0 + offset_s / WAVE_URGENCY_S)
                        wave_boost = pressures_own[neighbor_id] * urgency
                        combined  += wave_boost * WAVE_BOOST_WEIGHT
                        min_offset = min(min_offset, offset_s)
                    except (ValueError, ZeroDivisionError):
                        pass

            pressures_combined[node_id] = combined
            wave_offsets[node_id] = (
                min_offset if min_offset < float("inf") else 0.0
            )

        return pressures_combined, wave_offsets

    # ── Paso 3: ajuste de fases ───────────────────────────────────────────────

    def _adjust_phases(
            self,
            entities_by_node: dict[str, list[TrafficEntity]],
            pressures_combined: dict[str, float],
            pressures_own: dict[str, float],
            ctx: TrafficContext) -> None:
        """
        Aplica la presión combinada a cada intersección y determina la fase.

        Coordinación de cluster:
          Nodos del mismo cluster son interdependientes. El nodo de mayor
          presión gana el verde; los demás reducen su presión efectiva
          a CLUSTER_YIELD × presión si el ganador ya está en verde.
          Esto evita que dos intersecciones adyacentes del mismo cruce
          complejo estén verdes al mismo tiempo.

        La máquina de estados final vive en Intersection.adjust_phase()
        que garantiza la exclusión mutua NS/EW y activa BLINK sin tráfico.
        """
        # Determinar ganador por cluster
        cluster_winner: dict[str, str] = {}
        for cid, members in self._clusters.items():
            valid = [n for n in members if n in pressures_combined]
            if valid:
                cluster_winner[cid] = max(
                    valid, key=lambda n: pressures_combined[n]
                )

        for node_id, inter in self.graph.intersections.items():
            ents = entities_by_node.get(node_id, [])
            inter.pressure = pressures_combined[node_id]

            # Ceder el verde al ganador del cluster
            cid = self._node_to_cluster.get(node_id)
            if cid and cluster_winner.get(cid) != node_id:
                winner_id    = cluster_winner[cid]
                winner_phase = self.graph.intersections[winner_id].current_phase
                if winner_phase == Phase.GREEN:
                    inter.pressure *= CLUSTER_YIELD
                    logger.debug(
                        "[%s] cluster %s — cediendo verde a %s",
                        node_id, cid, winner_id
                    )

            inter.adjust_phase(self.engine, ctx, ents)

    # ── Construcción del resultado ────────────────────────────────────────────

    def _build_result(
            self,
            entities_by_node: dict[str, list[TrafficEntity]],
            pressures_own: dict[str, float],
            pressures_combined: dict[str, float],
            wave_offsets: dict[str, float]) -> TickResult:
        """
        Construye el TickResult con el estado actual de todas las
        intersecciones. Inmutable — puede guardarse en BD o enviarse
        a la API sin modificar el estado del algoritmo.
        """
        nodes: dict[str, NodeState] = {}
        total_entities = 0
        green_count    = 0
        yellow_count   = 0
        red_count      = 0
        blink_count    = 0

        for node_id, inter in self.graph.intersections.items():
            ents   = entities_by_node.get(node_id, [])
            counts = _count_entities(ents)
            total_entities += len(ents)

            phase = inter.current_phase.value
            if phase == "green":  green_count  += 1
            elif phase == "yellow": yellow_count += 1
            elif phase == "red":  red_count    += 1
            elif phase == "blink": blink_count  += 1

            nodes[node_id] = NodeState(
                node_id        = node_id,
                phase          = phase,
                phase_ns       = inter.phase_ns.value,
                phase_ew       = inter.phase_ew.value,
                active_axis    = getattr(inter._active_axis, "value", "ns"),
                signals        = inter.signal_summary,
                pressure       = round(pressures_combined.get(node_id, 0.0), 1),
                pressure_own   = round(pressures_own.get(node_id, 0.0), 1),
                pressure_ns    = round(inter._pressure_ns, 1),
                pressure_ew    = round(inter._pressure_ew, 1),
                wave_offset_s  = round(wave_offsets.get(node_id, 0.0), 1),
                threshold      = inter.pressure_threshold,
                ticks_in_phase = inter._ticks_in_phase,
                timeout_ticks  = inter.red_timeout_ticks,
                cluster_id     = self._node_to_cluster.get(node_id),
                entity_counts  = counts,
            )

        # Flujos bidireccionales entre nodos
        flows = _compute_flows(self.graph, entities_by_node)

        return TickResult(
            tick_number    = self._tick,
            nodes          = nodes,
            flows          = flows,
            total_entities = total_entities,
            green_count    = green_count,
            yellow_count   = yellow_count,
            red_count      = red_count,
            blink_count    = blink_count,
        )


# ── Helpers privados ──────────────────────────────────────────────────────────

def _count_entities(entities: list[TrafficEntity]) -> dict[str, int]:
    """Cuenta entidades por tipo de vehículo y peatón."""
    from core.entities import VehicleType
    counts: dict[str, int] = defaultdict(int)
    for e in entities:
        if isinstance(e, Vehicle):
            counts[e.vehicle_type.name] += 1
        elif isinstance(e, Pedestrian):
            counts["PEDESTRIAN"] += 1
            if e.is_wheelchair:
                counts["WHEELCHAIR"] += 1
    return dict(counts)


def _compute_flows(graph: TrafficGraph,
                   entities_by_node: dict[str, list[TrafficEntity]]
                   ) -> list[dict]:
    """
    Calcula flujos bidireccionales entre pares de nodos.
    El flujo de A→B es el número de vehículos en A que se dirigen a B.
    Como aproximación, contamos todos los vehículos del nodo origen.
    """
    flow_map: dict[tuple, dict] = defaultdict(lambda: {"fwd": 0, "bwd": 0})

    for from_id, to_id, _ in graph.graph.edges(data=True):
        n_veh = sum(
            1 for e in entities_by_node.get(from_id, [])
            if isinstance(e, Vehicle)
        )
        key = tuple(sorted([from_id, to_id]))
        if from_id <= to_id:
            flow_map[key]["fwd"] += n_veh
        else:
            flow_map[key]["bwd"] += n_veh

    return [
        {"from": k[0], "to": k[1], "fwd": v["fwd"], "bwd": v["bwd"]}
        for k, v in flow_map.items()
    ]