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
WAVE_URGENCY_S:    float = 20.0   # segundos de referencia para urgencia
MASTER_AMPLIFIER:  float = 1.3    # nodos MASTER amplifican señal × este factor
CLUSTER_YIELD:     float = 0.3    # nodo perdedor de cluster reduce presión a este factor

# Duración de un tick en segundos simulados (documentado en sim_params.json)
# Usado para convertir offsets de segundos a ticks en la coordinación temporal
TICK_DURATION_S:   float = 30.0   # 1 tick ≈ 30 segundos simulados

# ── Parámetros de coordinación temporal (offset real) ────────────────────────
# 1 tick ≈ TICK_DURATION_S segundos de tiempo simulado.
# Documentado en sim_params.json["simulation"]["tick_duration_seconds"].
# Este valor convierte offsets en segundos a ticks para la coordinación.
TICK_DURATION_S:   float = 30.0   # segundos reales que representa 1 tick

# Si el flujo del vecino ya debería haber llegado (offset cumplido),
# forzar verde en este nodo aunque la presión no alcance el umbral.
# Esto implementa la ola verde garantizada independiente de la presión.
GREEN_WAVE_FORCE:  bool  = True


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
    wave_forced:    bool           # True si este tick fue forzado a verde por la ola
    has_light:      bool           # True si tiene semáforo físico
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
            inter.current_phase        = Phase.RED
            inter._ticks_in_phase      = 0
            inter._timeout_triggered   = False
            inter._ticks_empty         = 0
            inter._pressure_ns         = 0.0
            inter._pressure_ew         = 0.0
            inter.pressure             = 0.0
            inter._green_started_tick  = -1
            inter._current_tick        = 0
        logger.debug("TrafficAlgorithm reiniciado")

    # ── API principal ─────────────────────────────────────────────────────────

    def run_tick(self,
                 entities_by_node: dict[str, list[TrafficEntity]],
                 ctx: TrafficContext) -> TickResult:
        """
        Ejecuta un tick completo del algoritmo de coordinación.

        Cuatro pasos secuenciales:

        Paso 1 — Presión propia
            WeightEngine.aggregate_pressure() por nodo. Función pura.

        Paso 2 — Mente colmena (boost de presión vecinal)
            Influencia de vecinos upstream ponderada por proximidad.
            Boost de urgencia si el vecino está en verde.

        Paso 2b — Offset temporal real (green wave garantizada)
            Si el flujo de un vecino upstream en verde ya debería haber
            llegado (offset_ticks cumplidos desde que el vecino se puso
            en verde), marcar este nodo como "wave_forced" para que
            Paso 3 lo fuerce a verde independientemente de la presión.
            Esto es lo que diferencia tanGo de un sistema de presión puro
            y lo acerca a la coordinación temporal de SCOOT.

        Paso 3 — Ajuste de fases con coordinación de cluster
            Aplica fases con exclusión mutua NS/EW, BLINK sin tráfico,
            timeout de equidad, y coordinación de cluster.
        """
        self._tick += 1

        # Informar el tick actual a cada intersección (para _green_started_tick)
        for inter in self.graph.intersections.values():
            inter._current_tick = self._tick

        # Paso 1: presiones propias
        pressures_own = self._compute_own_pressures(entities_by_node, ctx)

        # Paso 2: coordinación vecinal + green wave (boost anticipatorio + offset temporal)
        # _propagate_neighbor_signals marca inter._wave_forced=True cuando el offset se cumple
        pressures_combined, wave_offsets = self._propagate_neighbor_signals(
            pressures_own, ctx
        )

        # Paso 3: ajuste de fases — consume _wave_forced y registra _green_started_tick
        self._adjust_phases(
            entities_by_node, pressures_combined, pressures_own, ctx
        )

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

    # ── Paso 2b: offset temporal real ────────────────────────────────────────

    def _apply_green_wave_offset(
            self,
            wave_offsets: dict[str, float]) -> set[str]:
        """
        Implementa la coordinación temporal real de la ola verde.

        Para cada nodo que tiene un vecino upstream en verde, calcula
        cuántos ticks han pasado desde que el vecino se puso en verde.
        Si ese número supera el offset_ticks calculado, el flujo ya
        debería haber llegado — forzar verde en este nodo.

        Esto garantiza la ola verde independientemente de la presión,
        cerrando la diferencia con SCOOT en el componente de offset.

        Parameters
        ----------
        wave_offsets : Dict node_id → offset en segundos desde el vecino
                       upstream más cercano en verde (calculado en Paso 2).

        Returns
        -------
        Set de node_ids que deben ser forzados a verde por la ola.
        """
        if not GREEN_WAVE_FORCE:
            return set()

        wave_forced: set[str] = set()

        for node_id, inter in self.graph.intersections.items():
            # Solo aplicar a nodos que actualmente están en RED
            if inter.current_phase.value not in ("red", "blink"):
                continue

            for from_id, to_id, edge_data in self.graph.graph.edges(data=True):
                if to_id != node_id:
                    continue

                upstream = self.graph.intersections.get(from_id)
                if not upstream:
                    continue

                # El upstream debe estar en verde
                if upstream.current_phase.value != "green":
                    continue

                # ¿Cuándo se puso en verde el upstream?
                green_tick = getattr(upstream, "_green_started_tick", -1)
                if green_tick < 0:
                    continue

                seg = edge_data["segment"]
                try:
                    offset_s    = self.engine.compute_green_wave_offset(
                        distance_m      = seg.length_m,
                        speed_limit_kmh = seg.speed_limit_kmh,
                    )
                    offset_ticks = max(1, round(offset_s / TICK_DURATION_S))
                except (ValueError, ZeroDivisionError):
                    continue

                ticks_since_green = self._tick - green_tick

                if ticks_since_green >= offset_ticks:
                    wave_forced.add(node_id)
                    logger.info(
                        "[%s] GREEN WAVE — flujo de [%s] llegó "
                        "(offset=%d ticks, transcurridos=%d)",
                        node_id, from_id, offset_ticks, ticks_since_green
                    )
                    break   # un upstream es suficiente para forzar verde

        return wave_forced

    # ── Paso 2: mente colmena ─────────────────────────────────────────────────

    def _propagate_neighbor_signals(
            self,
            pressures_own: dict[str, float],
            ctx: TrafficContext
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Coordinación tipo SCOOT: señal vecinal + offset temporal real.

        Para cada nodo B con vecino A upstream, calcula:

          a) Señal de presión vecinal — cuánto tráfico viene de A.
             Decae con la distancia (Intersection.receive_neighbor_signal).

          b) Offset temporal — si A cambió a verde hace exactamente
             offset_ticks ticks, el flujo de A ya llegó a B.
             En ese caso, B DEBE estar en verde — se marca _wave_forced=True.
             Esto implementa la coordinación temporal de SCOOT.

          c) Boost anticipatorio — si A está en verde pero el flujo
             aún no llega (offset no cumplido), aumentar presión de B
             para prepararlo. Factor inversamente proporcional al tiempo
             de llegada restante.

        _wave_forced es consumido por _adjust_phases para forzar verde
        independientemente de la presión local.

        Returns
        -------
        (pressures_combined, wave_offsets)
        """
        pressures_combined: dict[str, float] = {}
        wave_offsets:       dict[str, float] = {}

        # Primero limpiar flags de ola forzada del tick anterior
        for inter in self.graph.intersections.values():
            inter._wave_forced = False

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

                # a) Señal vecinal de presión
                influence = inter.receive_neighbor_signal(
                    neighbor_pressure = pressures_own[neighbor_id],
                    distance_m        = seg.length_m,
                    speed_kmh         = seg.speed_limit_kmh,
                )
                if neighbor_inter.intersection_type.value == "master":
                    influence *= MASTER_AMPLIFIER
                combined += influence * NEIGHBOR_WEIGHT

                # b + c) Green wave: offset temporal + boost anticipatorio
                if neighbor_inter.current_phase.value == "green":
                    try:
                        offset_s     = self.engine.compute_green_wave_offset(
                            distance_m      = seg.length_m,
                            speed_limit_kmh = seg.speed_limit_kmh,
                        )
                        offset_ticks = max(1, round(offset_s / TICK_DURATION_S))
                        min_offset   = min(min_offset, offset_s)

                        # Ticks desde que el vecino cambió a verde
                        ticks_since_green = (
                            self._tick - neighbor_inter._green_started_tick
                            if neighbor_inter._green_started_tick >= 0
                            else float("inf")
                        )

                        if ticks_since_green >= offset_ticks:
                            # ── Offset cumplido: el flujo ya llegó ──────────
                            # Forzar verde en B — coordinación temporal SCOOT
                            if inter.has_traffic_light:
                                inter._wave_forced = True
                                logger.info(
                                    "[%s] GREEN WAVE forzada desde [%s] "
                                    "(offset=%.0fs=%d ticks, transcurridos=%d)",
                                    node_id, neighbor_id,
                                    offset_s, offset_ticks,
                                    int(min(ticks_since_green, 9999))
                                )
                        else:
                            # ── Offset pendiente: boost anticipatorio ────────
                            ticks_remaining = offset_ticks - ticks_since_green
                            urgency    = 1.0 / (1.0 + ticks_remaining * TICK_DURATION_S
                                                / WAVE_URGENCY_S)
                            wave_boost = pressures_own[neighbor_id] * urgency
                            combined  += wave_boost * WAVE_BOOST_WEIGHT

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
            ctx: TrafficContext,
            ) -> None:
        """
        Aplica la presión combinada a cada intersección y determina la fase.

        Si inter._wave_forced=True (puesto por _propagate_neighbor_signals),
        se fuerza el verde independientemente de la presión local —
        la ola verde llegó temporalmente (SCOOT offset cumplido).
        """
        # Determinar ganador por cluster — _wave_forced también puede ganar
        cluster_winner: dict[str, str] = {}
        for cid, members in self._clusters.items():
            # Filtrar solo los nodos que existen en el grafo actual
            existing = [n for n in members if n in self.graph.intersections]
            if not existing:
                continue
            wave_in_cluster = [n for n in existing
                               if self.graph.intersections[n]._wave_forced]
            if wave_in_cluster:
                cluster_winner[cid] = wave_in_cluster[0]
            else:
                valid = [n for n in existing if n in pressures_combined]
                if valid:
                    cluster_winner[cid] = max(
                        valid, key=lambda n: pressures_combined[n]
                    )

        # Construir mapa de nodos con entidades para verificar adyacencia
        nodes_with_entities = {
            nid for nid, ents in entities_by_node.items() if ents
        }

        # Boost para nodos donde hay entidades esperando en rojo.
        # Solo se activa si el nodo lleva al menos 2 ticks en rojo
        # con entidades — así el algoritmo de presión natural tiene
        # oportunidad de resolver conflictos antes de intervenir.
        for nid, inter in self.graph.intersections.items():
            if not inter.has_traffic_light:
                continue
            if inter.current_phase not in (Phase.RED, Phase.BLINK):
                continue
            ents_here = entities_by_node.get(nid, [])
            if not ents_here:
                continue
            # Esperar al menos 1 tick antes de boostar — da tiempo al algoritmo
            # natural de resolver si hay conflicto con otro carro adyacente
            if inter._ticks_in_phase < 1:
                continue
            waiting_boost = inter.pressure_threshold * 1.1 * len(ents_here)
            pressures_combined[nid] = pressures_combined.get(nid, 0) + waiting_boost
            inter.pressure     = pressures_combined[nid]
            inter._pressure_ns = max(inter._pressure_ns, inter.pressure_threshold * 1.1)
            inter._pressure_ew = max(inter._pressure_ew, inter.pressure_threshold * 1.1)
            if inter.current_phase == Phase.BLINK:
                inter.current_phase   = Phase.RED
                inter._ticks_in_phase = 0
                inter._ticks_empty    = 0

        for node_id, inter in self.graph.intersections.items():
            ents = entities_by_node.get(node_id, [])
            inter.pressure = pressures_combined[node_id]

            # Ceder el verde al ganador del cluster
            cid = self._node_to_cluster.get(node_id)
            if cid and cluster_winner.get(cid) != node_id:
                winner_inter = self.graph.intersections.get(cluster_winner[cid])
                if winner_inter and winner_inter.current_phase == Phase.GREEN:
                    inter.pressure *= CLUSTER_YIELD

            prev_phase = inter.current_phase

            # Ola verde temporal (offset SCOOT cumplido) — forzar presión
            if inter._wave_forced and inter.has_traffic_light:
                inter.pressure = max(inter.pressure,
                                     inter.pressure_threshold + 50.0)

            # Verificación de adyacencia: si hay entidades en nodos
            # inmediatamente upstream (a 1 hop) esperando llegar aquí,
            # aumentar la presión para facilitar el verde anticipado.
            # El carro está en upstream con ticks_to_next==0 — quiere cruzar
            # a este nodo pero necesita que esté en verde primero.
            if inter.has_traffic_light:
                upstream_nodes = list(self.graph.graph.predecessors(node_id))
                for upstream_id in upstream_nodes:
                    if upstream_id not in nodes_with_entities:
                        continue
                    upstream_ents = entities_by_node.get(upstream_id, [])
                    if not upstream_ents:
                        continue
                    n_waiting = len(upstream_ents)
                    # Boost proporcional al threshold: con 1 entidad upstream
                    # ya se supera el umbral del nodo destino.
                    upstream_boost = inter.pressure_threshold * 1.1 * n_waiting
                    inter.pressure += upstream_boost
                    # Si el nodo está en BLINK (sin tráfico propio pero con
                    # carro esperando upstream) — sacar de BLINK para que
                    # el boost pueda activar el verde
                    if inter.current_phase == Phase.BLINK:
                        inter.current_phase  = Phase.RED
                        inter._ticks_in_phase = 0
                        inter._ticks_empty    = 0
                        logger.debug("[%s] saliendo de BLINK por carro en upstream [%s]",
                                     node_id, upstream_id)
                    logger.debug(
                        "[%s] boost upstream [%s]: %d ents → +%.1f (total=%.1f, thr=%.1f)",
                        node_id, upstream_id, n_waiting,
                        upstream_boost, inter.pressure, inter.pressure_threshold
                    )

            inter.adjust_phase(self.engine, ctx, ents)

            # Registrar tick de inicio de verde (para propagar la ola al siguiente)
            if inter.current_phase == Phase.GREEN and prev_phase != Phase.GREEN:
                inter._green_started_tick = self._tick

    # ── Construcción del resultado ────────────────────────────────────────────

    def _build_result(
            self,
            entities_by_node: dict[str, list[TrafficEntity]],
            pressures_own: dict[str, float],
            pressures_combined: dict[str, float],
            wave_offsets: dict[str, float],
            wave_forced: set[str] | None = None) -> TickResult:
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
                wave_forced    = node_id in (wave_forced or set()),
                has_light      = inter.has_traffic_light,
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