"""
core/movement.py
----------------
Motor de movimiento de entidades en el grafo vial.

En sim1 las entidades aparecen y desaparecen en cada tick — no hay
persistencia. En sim2 las entidades tienen origen, destino y velocidad,
y se mueven nodo a nodo usando Dijkstra para encontrar la ruta óptima.

Esto modela la realidad más fielmente:
  - Un auto que sale de B3 hacia M2 tarda varios ticks en llegar.
  - Durante esos ticks pasa por N1, presionando ese semáforo.
  - El algoritmo de semáforos lo detecta en cada nodo que atraviesa.

En producción (KAN-10 + KAN-16), el origen/destino vendría de:
  - VisionIngester: detección por cámara de dónde viene y va el vehículo
  - Datos históricos de TomTom: patrones de origen-destino por hora

Relación con el algoritmo:
  MovementEngine.tick() → genera entities_by_node (entidades en tránsito)
  TrafficAlgorithm.run_tick(entities_by_node, ctx) → decide fases

Los pesos de las aristas para Dijkstra combinan:
  - Distancia física (length_m)
  - Velocidad límite de la vía
  - Estado actual del semáforo (rojo penaliza — evitar rutas congestionadas)
  - node_weight del destino (nodos importantes atraen más tráfico)
"""

from __future__ import annotations
import random
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from core.context   import TrafficContext
from core.entities  import (Vehicle, Pedestrian, VehicleType, Direction,
                             ROAD_SPEED_FACTOR, CONTEXT_SPEED_FACTOR)
from core.road      import Phase
from graph.simulator import TrafficGraph

logger = logging.getLogger(__name__)

# Penalización por semáforo en rojo (en segundos de espera equivalente)
# Un semáforo rojo hace que Dijkstra prefiera rutas alternativas
RED_LIGHT_PENALTY_S: float = 60.0   # 1 tick × 30s + margen

# Proporción de entidades con destino explícito vs sin destino
# (sin destino = aparecen, presionan el semáforo, desaparecen al cruzar)
DIRECTED_RATIO: float = 0.6   # 60% tienen destino definido


@dataclass
class MovingEntity:
    """
    Entidad en tránsito entre nodos del grafo.

    Attributes
    ----------
    entity      : La entidad física (Vehicle o Pedestrian).
    route       : Lista de node_ids desde origen hasta destino.
    route_idx   : Índice actual en la ruta (0 = en el primer nodo).
    ticks_to_next: Ticks restantes hasta llegar al siguiente nodo.
    total_ticks : Ticks totales de viaje (para estadísticas).
    """
    entity:       Vehicle | Pedestrian
    route:        list[str]
    route_idx:    int   = 0
    ticks_to_next: float = 0.0
    total_ticks:  int   = 0

    @property
    def current_node(self) -> str:
        if self.route_idx < len(self.route):
            return self.route[self.route_idx]
        return self.route[-1]

    @property
    def next_node(self) -> Optional[str]:
        if self.route_idx + 1 < len(self.route):
            return self.route[self.route_idx + 1]
        return None

    @property
    def has_arrived(self) -> bool:
        return self.route_idx >= len(self.route) - 1

    @property
    def origin(self) -> str:
        return self.route[0]

    @property
    def destination(self) -> str:
        return self.route[-1]

    @property
    def progress_pct(self) -> float:
        """Porcentaje de ruta completado."""
        if len(self.route) <= 1:
            return 100.0
        return self.route_idx / (len(self.route) - 1) * 100


@dataclass
class MovementStats:
    """
    Estadísticas de movimiento acumuladas por tick.
    Útil para el heatmap y las métricas de sim2.
    """
    tick:              int
    active_entities:   int             # entidades en tránsito
    arrived:           int             # llegaron a destino este tick
    spawned:           int             # nuevas entidades este tick
    flow_by_edge:      dict[str, int]  # "from-to" → entidades que pasaron
    avg_travel_ticks:  float           # tiempo promedio de viaje
    heatmap:           dict[str, float] # node_id → calor acumulado


class MovementEngine:
    """
    Motor de movimiento de entidades en el grafo vial.

    Responsabilidades:
      1. Generar entidades con origen-destino usando Dijkstra.
      2. Moverlas tick a tick según su velocidad y el estado del semáforo.
      3. Exponer entities_by_node para que TrafficAlgorithm las procese.
      4. Acumular estadísticas para el heatmap.

    Uso en sim2:
        engine = MovementEngine(graph)
        for tick in range(n_ticks):
            entities_by_node = engine.tick(ctx, algo.get_phases())
            result = algo.run_tick(entities_by_node, ctx)
    """

    def __init__(self, graph: TrafficGraph,
                 spawn_rate: int = 3,
                 max_entities: int = 50) -> None:
        """
        Parameters
        ----------
        graph        : Grafo vial.
        spawn_rate   : Entidades nuevas por tick.
        max_entities : Máximo de entidades simultáneas (rendimiento).
        """
        self.graph        = graph
        self.spawn_rate   = spawn_rate
        self.max_entities = max_entities

        self._active:   list[MovingEntity] = []
        self._arrived:  list[MovingEntity] = []   # completaron viaje
        self._tick      = 0
        self._heatmap:  dict[str, float]  = {
            nid: 0.0 for nid in graph.intersections
        }
        self._flow_history: list[dict[str, int]] = []

        # Nodos con semáforo — son los preferidos como destino
        self._signaled_nodes = [
            nid for nid, inter in graph.intersections.items()
            if inter.has_traffic_light
        ]

        logger.info(
            "MovementEngine iniciado: %d nodos, %d semaforizados",
            graph.graph.number_of_nodes(), len(self._signaled_nodes)
        )

    # ── API principal ─────────────────────────────────────────────────────────

    def tick(self, ctx: TrafficContext,
             current_phases: dict[str, str] | None = None
             ) -> dict[str, list]:
        """
        Avanza un tick del motor de movimiento.

        1. Generar entidades nuevas con origen-destino.
        2. Mover entidades existentes (avanzar ticks_to_next).
        3. Si el semáforo está en verde, cruzar al siguiente nodo.
        4. Eliminar las que llegaron.
        5. Retornar entities_by_node para TrafficAlgorithm.

        Parameters
        ----------
        ctx            : Contexto ambiental del tick.
        current_phases : Dict node_id → "green"|"yellow"|"red"|"blink".
                         Si None, ignora el estado del semáforo para moverse.

        Returns
        -------
        Dict node_id → lista de entidades presentes en ese nodo.
        """
        self._tick += 1
        phases = current_phases or {}

        # 1. Spawn de entidades nuevas
        self._spawn(ctx)

        # 2. Mover entidades
        flow_this_tick: dict[str, int] = {}
        newly_arrived  = []

        for me in self._active:
            if me.has_arrived:
                newly_arrived.append(me)
                continue

            me.total_ticks += 1
            me.entity.ticks_alive += 1

            # Decrementar ticks hasta el siguiente nodo
            if me.ticks_to_next > 0:
                me.ticks_to_next -= 1
                continue

            # Ticks_to_next llegó a 0 — intentar cruzar al siguiente nodo
            next_nid = me.next_node
            if next_nid is None:
                newly_arrived.append(me)
                continue

            cur_phase = phases.get(me.current_node, "green")

            # Si el semáforo está en rojo, la entidad espera en el nodo actual
            if cur_phase == "red" and self.graph.intersections[me.current_node].has_traffic_light:
                # Esperar — ticks_to_next permanece en 0 hasta que cambie a verde
                continue

            # Verde o sin semáforo — cruzar al siguiente nodo
            edge_key = f"{me.current_node}-{next_nid}"
            flow_this_tick[edge_key] = flow_this_tick.get(edge_key, 0) + 1

            # Calcular tiempo de viaje al siguiente segmento
            seg_data = self._get_segment(me.current_node, next_nid)
            if seg_data:
                seg       = seg_data["segment"]
                travel_t  = me.entity.travel_time_ticks(
                    distance_m    = seg.length_m,
                    road_category = seg.category.name,
                    ctx           = ctx,
                )
                me.ticks_to_next = max(1, round(travel_t))
            else:
                me.ticks_to_next = 1

            # Avanzar al siguiente nodo
            me.route_idx        += 1
            me.entity.current_node = me.current_node

            # Actualizar heatmap
            self._heatmap[me.current_node] = (
                self._heatmap.get(me.current_node, 0.0) + 1.0
            )

            if me.has_arrived:
                newly_arrived.append(me)

        # 3. Retirar entidades que llegaron
        arrived_count = len(newly_arrived)
        for me in newly_arrived:
            if me in self._active:
                self._active.remove(me)
                self._arrived.append(me)

        self._flow_history.append(flow_this_tick)

        # 4. Construir entities_by_node
        return self._build_entities_by_node()

    def get_stats(self) -> MovementStats:
        """Estadísticas del tick actual para el dashboard."""
        arrived_this_tick = [
            me for me in self._arrived
            if me.total_ticks > 0
        ]
        avg_travel = (
            sum(me.total_ticks for me in arrived_this_tick) / len(arrived_this_tick)
            if arrived_this_tick else 0.0
        )
        flow = self._flow_history[-1] if self._flow_history else {}

        return MovementStats(
            tick             = self._tick,
            active_entities  = len(self._active),
            arrived          = len([me for me in self._arrived
                                    if me.total_ticks == self._tick]),
            spawned          = 0,   # se actualiza en _spawn
            flow_by_edge     = flow,
            avg_travel_ticks = round(avg_travel, 1),
            heatmap          = dict(self._heatmap),
        )

    def get_heatmap(self) -> dict[str, float]:
        """
        Calor acumulado por nodo — normalizado a [0, 1].
        0 = nunca visitado, 1 = el más visitado de la red.
        """
        if not self._heatmap:
            return {}
        max_heat = max(self._heatmap.values()) or 1.0
        return {nid: round(v / max_heat, 3)
                for nid, v in self._heatmap.items()}

    def get_particles(self) -> list[dict]:
        """
        Posiciones interpoladas de entidades para animar partículas en el mapa.
        Retorna lat/lon interpolado entre nodo actual y siguiente según progreso.
        """
        particles = []
        for me in self._active:
            cur_inter = self.graph.intersections.get(me.current_node)
            if not cur_inter:
                continue

            lat, lon = cur_inter.latitude, cur_inter.longitude

            # Interpolar si tiene nodo siguiente y está en tránsito
            if me.next_node and me.ticks_to_next > 0:
                nxt_inter = self.graph.intersections.get(me.next_node)
                if nxt_inter:
                    # Progreso entre 0 (salió) y 1 (llegó)
                    # ticks_to_next va bajando — cuando llega a 0 cruza
                    seg_data = self._get_segment(me.current_node, me.next_node)
                    seg_ticks = 1
                    if seg_data:
                        seg_ticks = max(1, round(me.entity.travel_time_ticks(
                            seg_data["segment"].length_m,
                            seg_data["segment"].category.name,
                        )))
                    progress = 1.0 - (me.ticks_to_next / seg_ticks)
                    progress = max(0.0, min(1.0, progress))

                    lat = cur_inter.latitude  + progress * (nxt_inter.latitude  - cur_inter.latitude)
                    lon = cur_inter.longitude + progress * (nxt_inter.longitude - cur_inter.longitude)

            entity = me.entity
            ptype  = "vehicle" if isinstance(entity, Vehicle) else "pedestrian"
            vtype  = entity.vehicle_type.name if isinstance(entity, Vehicle) else "PEDESTRIAN"

            particles.append({
                "id":          entity.entity_id[:8],
                "lat":         round(lat, 7),
                "lon":         round(lon, 7),
                "type":        ptype,
                "vtype":       vtype,
                "speed_kmh":   round(entity.speed_kmh, 1),
                "origin":      me.origin,
                "destination": me.destination,
                "progress":    round(me.progress_pct, 1),
                "ticks_alive": entity.ticks_alive,
            })
        return particles

    # ── Internos ──────────────────────────────────────────────────────────────

    def _spawn(self, ctx: TrafficContext) -> None:
        """Genera entidades nuevas con origen-destino por Dijkstra."""
        if len(self._active) >= self.max_entities:
            return

        n_to_spawn = min(
            self.spawn_rate,
            self.max_entities - len(self._active)
        )
        # Más entidades en hora pico
        if ctx.is_rush_hour:
            n_to_spawn = min(n_to_spawn * 2, self.max_entities - len(self._active))
        elif ctx.is_late_night:
            n_to_spawn = max(1, n_to_spawn // 2)

        nodes = list(self.graph.intersections.keys())

        for _ in range(n_to_spawn):
            origin = random.choice(nodes)

            # 60% con destino explícito (nodo semaforizado preferido)
            if random.random() < DIRECTED_RATIO and self._signaled_nodes:
                destination = random.choice(self._signaled_nodes)
                if destination == origin:
                    destination = random.choice(nodes)
            else:
                destination = None

            route = self._find_route(origin, destination)
            if len(route) < 2:
                continue

            # Crear entidad según contexto
            entity = self._create_entity(ctx)
            entity.origin_node      = origin
            entity.destination_node = destination
            entity.current_node     = origin

            # Calcular tiempo inicial al primer segmento
            seg_data = self._get_segment(route[0], route[1])
            if seg_data:
                seg = seg_data["segment"]
                ticks = max(1, round(entity.travel_time_ticks(
                    seg.length_m, seg.category.name, ctx
                )))
            else:
                ticks = 1

            self._active.append(MovingEntity(
                entity        = entity,
                route         = route,
                route_idx     = 0,
                ticks_to_next = ticks,
            ))

    def _find_route(self, origin: str,
                    destination: Optional[str] = None,
                    phases: dict[str, str] | None = None) -> list[str]:
        """
        Encuentra ruta óptima con Dijkstra ponderado.

        El peso de cada arista combina:
          - Tiempo de viaje real (distancia / velocidad_límite)
          - Penalización si el semáforo destino está en rojo
          - Factor inverso de node_weight del destino
            (nodos importantes atraen más tráfico)
        """
        if destination is None:
            # Sin destino — ruta corta aleatoria (1-3 saltos)
            neighbors = list(self.graph.graph.successors(origin))
            if not neighbors:
                return [origin]
            dest = random.choice(neighbors)
            try:
                return nx.shortest_path(
                    self.graph.graph, origin, dest, weight="travel_time"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return [origin, dest]

        try:
            # Construir pesos dinámicos para Dijkstra
            def edge_weight(u, v, data):
                seg = data.get("segment")
                if not seg:
                    return 1000.0
                # Tiempo base de viaje en segundos
                speed_ms = (seg.speed_limit_kmh / 3.6) or 1.0
                base_t   = seg.length_m / speed_ms

                # Penalización por semáforo rojo
                v_phase = (phases or {}).get(v, "green")
                penalty = RED_LIGHT_PENALTY_S if v_phase == "red" else 0.0

                # Factor de atracción por node_weight del destino
                # Nodos más importantes (alta centralidad) atraen más
                v_inter  = self.graph.intersections.get(v)
                nw       = getattr(v_inter, "node_weight", 1.0) if v_inter else 1.0
                attract  = 1.0 / max(nw, 0.1)   # más peso → más atractivo

                return (base_t + penalty) * attract

            return nx.shortest_path(
                self.graph.graph, origin, destination,
                weight=edge_weight,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [origin]

    def _create_entity(self, ctx: TrafficContext) -> Vehicle | Pedestrian:
        """Crea una entidad según distribución realista del contexto."""
        # Distribución vehicular — misma que el simulador
        pool = (
            [VehicleType.CAR] * 60 +
            [VehicleType.MOTORCYCLE] * 15 +
            [VehicleType.BUS] * 10 +
            [VehicleType.TRUCK] * 8 +
            [VehicleType.BICYCLE] * (2 if ctx.is_raining else 5) +
            [VehicleType.EMERGENCY] * 2
        )

        # 15% de probabilidad de ser peatón
        if random.random() < 0.15:
            wc = random.random() < 0.08
            return Pedestrian(str(uuid.uuid4()), is_wheelchair=wc)

        vtype = random.choice(pool)
        return Vehicle(
            str(uuid.uuid4()), vtype,
            random.choice(list(Direction))
        )

    def _get_segment(self, from_id: str, to_id: str) -> Optional[dict]:
        """Retorna los datos del segmento entre dos nodos si existe."""
        if self.graph.graph.has_edge(from_id, to_id):
            return self.graph.graph[from_id][to_id]
        return None

    def _build_entities_by_node(self) -> dict[str, list]:
        """Agrupa entidades activas por nodo actual."""
        by_node: dict[str, list] = {
            nid: [] for nid in self.graph.intersections
        }
        for me in self._active:
            nid = me.current_node
            if nid in by_node:
                by_node[nid].append(me.entity)
        return by_node