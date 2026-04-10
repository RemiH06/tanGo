"""
graph/simulator.py
------------------
TrafficGraph — grafo vial con NetworkX + Neo4j.

Responsabilidades:
  1. Construir y mantener el grafo de la ciudad (nodos = intersecciones,
     aristas = segmentos viales con pesos dinámicos).
  2. Cargar/persistir el grafo desde/hacia Neo4j.
  3. Simular entidades (vehículos, peatones) para desarrollo y testing.
  4. Propagar la ola verde entre intersecciones vecinas.
  5. Exportar el mapa de presión para el dashboard.

Arquitectura dual:
  Neo4j  → persistencia, visualización con Bloom, análisis con GDS.
  NetworkX → algoritmos en memoria por ciclo (Dijkstra, vecinos, pesos).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import random
import uuid

import networkx as nx
from neo4j import GraphDatabase, Driver

from core.context import TrafficContext
from core.entities import (
    TrafficEntity, Vehicle, Pedestrian,
    VehicleType, Direction
)
from core.road import RoadSegment, Intersection, RoadCategory, Phase, Turn

logger = logging.getLogger(__name__)


# ── Snapshot de simulación ────────────────────────────────────────────────────

@dataclass
class SimulationSnapshot:
    """
    Estado completo de la simulación en un tick.

    Attributes
    ----------
    timestamp     : Momento del tick.
    entities      : Dict node_id → lista de entidades en esa intersección.
    tick_number   : Contador de ticks desde el inicio.
    """
    timestamp:   datetime
    entities:    Dict[str, List[TrafficEntity]]
    tick_number: int


# ── Probabilidades de spawn por contexto ─────────────────────────────────────

def _vehicle_count(ctx: TrafficContext) -> int:
    """Número de vehículos a generar por intersección según el contexto."""
    if ctx.is_rush_hour:
        return random.randint(4, 10)
    if ctx.is_late_night:
        return random.randint(0, 2)
    return random.randint(1, 5)


def _pedestrian_count(ctx: TrafficContext) -> int:
    """Número de peatones a generar por intersección según el contexto."""
    if ctx.is_rush_hour:
        return random.randint(2, 8)
    if ctx.is_late_night:
        return random.randint(0, 1)
    if ctx.is_raining:
        return random.randint(1, 4)
    return random.randint(0, 4)


def _wheelchair_probability(ctx: TrafficContext) -> float:
    """Probabilidad de que un peatón use silla de ruedas."""
    # Aumenta si es de día (más movilidad urbana accesible)
    if ctx.is_late_night:
        return 0.02
    return 0.08


def _vehicle_type_weights(ctx: TrafficContext) -> List[Tuple[VehicleType, float]]:
    """Distribución ponderada de tipos de vehículo según contexto."""
    weights = [
        (VehicleType.CAR,        0.70),
        (VehicleType.MOTORCYCLE, 0.10),
        (VehicleType.BUS,        0.08),
        (VehicleType.TRUCK,      0.05),
        (VehicleType.BICYCLE,    0.05 if not ctx.is_raining else 0.01),
        (VehicleType.EMERGENCY,  0.02),
    ]
    # Normalizar si la lluvia cambió los pesos
    total = sum(w for _, w in weights)
    return [(vt, w / total) for vt, w in weights]


# ── TrafficGraph ──────────────────────────────────────────────────────────────

class TrafficGraph:
    """
    Grafo vial de la ciudad — combina Neo4j y NetworkX.

    Uso en desarrollo (sin Neo4j):
        graph = TrafficGraph()
        graph.build_sample_city()
        snapshot = graph.tick(ctx)

    Uso en producción (con Neo4j):
        graph = TrafficGraph(uri, user, password)
        graph.load_from_neo4j()
        snapshot = graph.tick(ctx)

    Attributes
    ----------
    graph         : DiGraph de NetworkX — nodos son node_id (str),
                    aristas tienen atributos 'segment' y 'weight'.
    intersections : Dict node_id → Intersection.
    _tick_count   : Contador de ticks.
    _driver       : Driver de Neo4j (None en modo offline).
    """

    def __init__(self, neo4j_uri:      Optional[str] = None,
                       neo4j_user:     Optional[str] = None,
                       neo4j_password: Optional[str] = None) -> None:
        self.graph:         nx.DiGraph               = nx.DiGraph()
        self.intersections: Dict[str, Intersection]  = {}
        self._tick_count:   int                      = 0
        self._driver:       Optional[Driver]         = None

        if neo4j_uri and neo4j_user and neo4j_password:
            self._driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            logger.info("Neo4j conectado: %s", neo4j_uri)

    # ── Construcción del grafo ────────────────────────────────────────────────

    def add_intersection(self, intersection: Intersection) -> None:
        """
        Agrega una intersección al grafo como nodo.

        Parameters
        ----------
        intersection : Intersección a agregar.
        """
        self.intersections[intersection.node_id] = intersection
        self.graph.add_node(
            intersection.node_id,
            name      = intersection.name,
            latitude  = intersection.latitude,
            longitude = intersection.longitude,
        )
        logger.debug("Nodo agregado: %s (%s)", intersection.node_id, intersection.name)

    def add_segment(self, segment: RoadSegment) -> None:
        """
        Agrega un segmento vial al grafo como arista dirigida.
        También actualiza la lista de incoming_segments de la intersección destino.

        Parameters
        ----------
        segment : Segmento a agregar.

        Raises
        ------
        KeyError si from_node_id o to_node_id no existen en el grafo.
        """
        if segment.from_node_id not in self.intersections:
            raise KeyError(f"Nodo origen no existe: {segment.from_node_id}")
        if segment.to_node_id not in self.intersections:
            raise KeyError(f"Nodo destino no existe: {segment.to_node_id}")

        self.graph.add_edge(
            segment.from_node_id,
            segment.to_node_id,
            segment = segment,
            weight  = segment.base_weight,
        )
        # Registrar el segmento en la intersección destino
        dest = self.intersections[segment.to_node_id]
        if segment not in dest.incoming_segments:
            dest.incoming_segments.append(segment)

        logger.debug(
            "Arista agregada: %s → %s (w=%.0f)",
            segment.from_node_id, segment.to_node_id, segment.base_weight
        )

    def build_sample_city(self) -> None:
        """
        Construye una ciudad de ejemplo en memoria sin Neo4j.
        Útil para tests, demos y desarrollo local.

        Topología (Guadalajara simplificado):

            [A1]──Av.Principal──>[A2]──Av.Principal──>[A3]
             │                    │
           Calle               Calle
             │                    │
            [B1]──────Calle──────>[B2]

        A1, A2, A3 → avenida principal (w=80)
        B1, B2     → calles secundarias (w=20)
        """
        # Intersecciones
        intersections = [
            Intersection("A1", "Av. Vallarta y López Mateos", 20.6756, -103.3910),
            Intersection("A2", "Av. Vallarta y Av. México",   20.6756, -103.3700),
            Intersection("A3", "Av. Vallarta y Chapultepec",  20.6756, -103.3500),
            Intersection("B1", "Calle Juárez y López Mateos", 20.6600, -103.3910),
            Intersection("B2", "Calle Juárez y Av. México",   20.6600, -103.3700),
        ]
        for inter in intersections:
            self.add_intersection(inter)

        # Segmentos
        segments = [
            RoadSegment("seg-A1-A2", "A1", "A2",
                        RoadCategory.MAIN_AVENUE, 500.0, 60.0),
            RoadSegment("seg-A2-A3", "A2", "A3",
                        RoadCategory.MAIN_AVENUE, 500.0, 60.0),
            RoadSegment("seg-A1-B1", "A1", "B1",
                        RoadCategory.STREET, 200.0, 30.0),
            RoadSegment("seg-A2-B2", "A2", "B2",
                        RoadCategory.STREET, 200.0, 30.0),
            RoadSegment("seg-B1-B2", "B1", "B2",
                        RoadCategory.STREET, 500.0, 30.0,
                        forbidden_turns=[Turn.U_TURN]),
        ]
        for seg in segments:
            self.add_segment(seg)

        logger.info(
            "Ciudad de ejemplo construida: %d nodos, %d aristas",
            self.graph.number_of_nodes(), self.graph.number_of_edges()
        )

    # ── Neo4j ─────────────────────────────────────────────────────────────────

    def load_from_neo4j(self) -> None:
        """
        Carga el grafo desde Neo4j a NetworkX.

        Query CQL:
            MATCH (a:Intersection)-[r:ROAD]->(b:Intersection)
            RETURN a, r, b

        Raises
        ------
        RuntimeError si no hay driver configurado.
        """
        if not self._driver:
            raise RuntimeError(
                "Neo4j driver no configurado. "
                "Usa build_sample_city() para modo offline."
            )

        with self._driver.session() as session:
            result = session.run(
                "MATCH (a:Intersection)-[r:ROAD]->(b:Intersection) "
                "RETURN a, r, b"
            )
            for record in result:
                a_props = dict(record["a"])
                b_props = dict(record["b"])
                r_props = dict(record["r"])

                # Reconstruir intersecciones si no existen
                for props in (a_props, b_props):
                    if props["node_id"] not in self.intersections:
                        self.add_intersection(
                            Intersection.from_neo4j_props(props)
                        )

                # Reconstruir segmento
                r_props["from_node_id"] = a_props["node_id"]
                r_props["to_node_id"]   = b_props["node_id"]
                segment = RoadSegment.from_neo4j_props(r_props)
                self.add_segment(segment)

        logger.info(
            "Grafo cargado desde Neo4j: %d nodos, %d aristas",
            self.graph.number_of_nodes(), self.graph.number_of_edges()
        )

    def write_to_neo4j(self) -> None:
        """
        Persiste el grafo en Neo4j (MERGE — no duplica si ya existe).
        Útil para visualizar con Bloom y analizar con GDS.

        Raises
        ------
        RuntimeError si no hay driver configurado.
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver no configurado.")

        with self._driver.session() as session:
            # Nodos
            for node_id, inter in self.intersections.items():
                session.run(
                    "MERGE (i:Intersection {node_id: $node_id}) "
                    "SET i.name = $name, i.latitude = $lat, i.longitude = $lon",
                    node_id = node_id,
                    name    = inter.name,
                    lat     = inter.latitude,
                    lon     = inter.longitude,
                )

            # Aristas
            for from_id, to_id, data in self.graph.edges(data=True):
                seg: RoadSegment = data["segment"]
                props = seg.to_neo4j_props()
                session.run(
                    "MATCH (a:Intersection {node_id: $from_id}) "
                    "MATCH (b:Intersection {node_id: $to_id}) "
                    "MERGE (a)-[r:ROAD {segment_id: $segment_id}]->(b) "
                    "SET r += $props",
                    from_id    = from_id,
                    to_id      = to_id,
                    segment_id = props["segment_id"],
                    props      = props,
                )

        logger.info("Grafo persistido en Neo4j.")

    def update_weights_in_graph(self, ctx: TrafficContext) -> None:
        """
        Actualiza los pesos de todas las aristas en NetworkX
        según el contexto actual. Llamar al inicio de cada ciclo
        para que Dijkstra use pesos dinámicos.

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo.
        """
        for _, _, data in self.graph.edges(data=True):
            seg: RoadSegment = data["segment"]
            # Modificadores básicos sobre el peso base
            weight = seg.base_weight
            if ctx.is_weekend:
                weight *= 0.7
            if ctx.is_rush_hour:
                weight *= 1.2
            if ctx.is_late_night:
                weight *= 0.5
            data["weight"] = weight

    # ── Algoritmos de grafo ───────────────────────────────────────────────────

    def shortest_path(self, from_id: str, to_id: str) -> List[str]:
        """
        Ruta más corta entre dos intersecciones usando Dijkstra.
        Usa los pesos actuales de las aristas.

        Parameters
        ----------
        from_id : ID de la intersección de origen.
        to_id   : ID de la intersección de destino.

        Returns
        -------
        Lista de node_id en orden desde origen a destino.

        Raises
        ------
        nx.NetworkXNoPath si no existe ruta.
        KeyError si alguno de los IDs no existe.
        """
        return nx.shortest_path(
            self.graph, from_id, to_id, weight="weight"
        )

    def green_wave_offsets(self, start_id: str,
                           ctx: TrafficContext) -> Dict[str, float]:
        """
        Calcula el offset en segundos para cada intersección vecina
        de start_id, para sincronizar la ola verde.

        Fórmula: offset = length_m / (speed_limit_kmh / 3.6)

        Parameters
        ----------
        start_id : Intersección que acaba de ponerse en verde.
        ctx      : Contexto del ciclo (no usado directamente aquí,
                   disponible para extensiones futuras).

        Returns
        -------
        Dict node_id → offset en segundos.
        """
        offsets: Dict[str, float] = {}
        for neighbor_id in self.graph.successors(start_id):
            data    = self.graph[start_id][neighbor_id]
            seg: RoadSegment = data["segment"]
            offsets[neighbor_id] = seg.travel_time_seconds
        return offsets

    def neighbors_of(self, node_id: str) -> List[str]:
        """
        IDs de intersecciones directamente alcanzables desde node_id.

        Parameters
        ----------
        node_id : ID de la intersección.

        Returns
        -------
        Lista de node_id vecinos.
        """
        return list(self.graph.successors(node_id))

    # ── Simulación de entidades ───────────────────────────────────────────────

    def spawn_vehicle(self, node_id: str,
                      vehicle_type: VehicleType = VehicleType.CAR,
                      direction:    Direction    = Direction.NORTH) -> Vehicle:
        """
        Crea un vehículo en una intersección.

        Parameters
        ----------
        node_id      : Intersección destino.
        vehicle_type : Tipo de vehículo.
        direction    : Dirección de circulación.

        Returns
        -------
        Vehicle con entity_id único (UUID).
        """
        if node_id not in self.intersections:
            raise KeyError(f"Intersección no existe: {node_id}")
        return Vehicle(
            entity_id    = str(uuid.uuid4()),
            vehicle_type = vehicle_type,
            direction    = direction,
        )

    def spawn_pedestrian(self, node_id: str,
                         is_wheelchair:    bool  = False,
                         crossing_width_m: float = 10.0) -> Pedestrian:
        """
        Crea un peatón en una intersección.

        Parameters
        ----------
        node_id          : Intersección destino.
        is_wheelchair    : True si usa silla de ruedas.
        crossing_width_m : Ancho del cruce en metros.

        Returns
        -------
        Pedestrian con entity_id único (UUID).
        """
        if node_id not in self.intersections:
            raise KeyError(f"Intersección no existe: {node_id}")
        return Pedestrian(
            entity_id        = str(uuid.uuid4()),
            is_wheelchair    = is_wheelchair,
            crossing_width_m = crossing_width_m,
        )

    # ── Tick del pipeline ─────────────────────────────────────────────────────

    def tick(self, ctx: TrafficContext) -> SimulationSnapshot:
        """
        Avanza la simulación un ciclo.
        Genera entidades probabilísticamente según el contexto
        y las asigna a cada intersección.

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo.

        Returns
        -------
        SimulationSnapshot con entidades por intersección.
        """
        self._tick_count += 1
        self.update_weights_in_graph(ctx)

        entities: Dict[str, List[TrafficEntity]] = {}
        vehicle_type_pool = _vehicle_type_weights(ctx)
        types   = [vt for vt, _ in vehicle_type_pool]
        weights = [w  for _, w  in vehicle_type_pool]

        for node_id in self.intersections:
            node_entities: List[TrafficEntity] = []

            # Vehículos
            n_vehicles = _vehicle_count(ctx)
            for _ in range(n_vehicles):
                vtype = random.choices(types, weights=weights, k=1)[0]
                direction = random.choice(list(Direction))
                node_entities.append(
                    self.spawn_vehicle(node_id, vtype, direction)
                )

            # Peatones
            n_pedestrians = _pedestrian_count(ctx)
            wheelchair_prob = _wheelchair_probability(ctx)
            for _ in range(n_pedestrians):
                is_wc = random.random() < wheelchair_prob
                node_entities.append(
                    self.spawn_pedestrian(node_id, is_wheelchair=is_wc)
                )

            entities[node_id] = node_entities

        logger.debug(
            "Tick %d — %d intersecciones, contexto: rush=%s night=%s rain=%s",
            self._tick_count, len(entities),
            ctx.is_rush_hour, ctx.is_late_night, ctx.is_raining
        )

        return SimulationSnapshot(
            timestamp   = ctx.timestamp,
            entities    = entities,
            tick_number = self._tick_count,
        )

    # ── Incidentes y análisis ─────────────────────────────────────────────────

    def apply_incident(self, segment_id: str,
                       severity: float = 0.1) -> None:
        """
        Simula un incidente en un segmento (accidente, obra).
        Reduce el peso de la arista para que Dijkstra evite esa ruta.

        Parameters
        ----------
        segment_id : ID del segmento afectado.
        severity   : Factor multiplicador del peso (0.1 = muy severo).
        """
        for _, _, data in self.graph.edges(data=True):
            seg: RoadSegment = data["segment"]
            if seg.segment_id == segment_id:
                data["weight"] = data["weight"] * severity
                logger.warning(
                    "Incidente en %s — peso reducido a %.1f",
                    segment_id, data["weight"]
                )
                return
        raise KeyError(f"Segmento no encontrado: {segment_id}")

    def export_pressure_map(self) -> Dict[str, float]:
        """
        Exporta la presión actual de cada intersección.
        Usado por el dashboard de Streamlit/Folium para colorear el mapa.

        Returns
        -------
        Dict node_id → presión (0.0 – ∞).
        """
        return {
            node_id: inter.pressure
            for node_id, inter in self.intersections.items()
        }

    def summary(self) -> str:
        """Resumen legible del estado actual del grafo."""
        lines = [
            f"TrafficGraph — {self.graph.number_of_nodes()} nodos, "
            f"{self.graph.number_of_edges()} aristas, "
            f"tick #{self._tick_count}",
        ]
        for node_id, inter in self.intersections.items():
            lines.append(
                f"  {node_id} | {inter.name} | "
                f"fase={inter.current_phase.value} | "
                f"presión={inter.pressure:.1f}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        """Cierra la conexión a Neo4j si existe."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j desconectado.")