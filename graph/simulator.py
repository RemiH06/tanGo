"""
graph/simulator.py
------------------
Simulador de ciudad para desarrollo y testing.

Por qué un simulador:
  - En desarrollo no siempre tenemos acceso a TomTom o datos reales.
  - El simulador permite probar el WeightEngine y SafetyGuard con
    escenarios controlados (hora pico, lluvia, emergencia, etc.).
  - En producción se reemplaza por TomTomIngester — misma interfaz.

Arquitectura dual Neo4j + NetworkX:
  - Neo4j: persiste el grafo de la ciudad (nodos, relaciones, propiedades).
    Permite queries CQL para análisis, Bloom para visualización, GDS para
    algoritmos de grafos avanzados.
  - NetworkX: carga el grafo en memoria para calcular rutas y propagar
    señales en cada ciclo del pipeline (más rápido que roundtrips a Neo4j).

Flujo de datos:
  CitySimulator.load_from_neo4j() → grafo en NetworkX
  CitySimulator.tick(ctx)         → snapshot de entidades por intersección
  WeightEngine.aggregate_pressure → decisión de fase
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import random

import networkx as nx
from neo4j import GraphDatabase, Driver

from core.context import TrafficContext
from core.entities import TrafficEntity, Vehicle, Pedestrian, VehicleType, Direction
from core.road import RoadSegment, Intersection, RoadCategory, Phase


# ── Snapshot de simulación ────────────────────────────────────────────────────

@dataclass
class SimulationSnapshot:
    """
    Estado de la simulación en un tick.

    Attributes
    ----------
    timestamp     : Momento del tick.
    intersections : Dict node_id → Intersection con entidades asignadas.
    tick_number   : Número de tick desde el inicio de la simulación.
    """
    timestamp:     datetime
    intersections: Dict[str, List[TrafficEntity]]
    tick_number:   int


# ── CitySimulator ─────────────────────────────────────────────────────────────

class CitySimulator:
    """
    Simula el tráfico de una ciudad para desarrollo y testing.

    Attributes
    ----------
    neo4j_driver : Conexión a Neo4j (puede ser None en modo offline).
    graph        : Grafo de NetworkX cargado desde Neo4j o construido manualmente.
    intersections: Dict de Intersection indexado por node_id.
    _tick_count  : Contador interno de ticks.
    """

    def __init__(self, neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None) -> None:
        self.graph:         nx.DiGraph             = nx.DiGraph()
        self.intersections: Dict[str, Intersection] = {}
        self._tick_count:   int                    = 0

        # Conexión a Neo4j (opcional — el simulador funciona sin ella)
        self._driver: Optional[Driver] = None
        if neo4j_uri and neo4j_user and neo4j_password:
            self._driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )

    # ── Construcción del grafo ─────────────────────────────────────────────

    def load_from_neo4j(self) -> None:
        """
        Carga el grafo de la ciudad desde Neo4j a NetworkX.

        Query CQL esperada:
            MATCH (a:Intersection)-[r:ROAD]->(b:Intersection)
            RETURN a, r, b

        Cada nodo :Intersection debe tener: node_id, name, latitude, longitude.
        Cada relación :ROAD debe tener: segment_id, category, length_m,
            speed_limit_kmh, allowed_turns, forbidden_turns.

        Raises
        ------
        RuntimeError si no hay driver de Neo4j configurado.
        """
        if not self._driver:
            raise RuntimeError(
                "Neo4j driver no configurado. "
                "Usa build_sample_city() para modo offline."
            )
        # TODO: implementar query CQL y construcción del grafo NetworkX
        raise NotImplementedError

    def build_sample_city(self) -> None:
        """
        Construye una ciudad de ejemplo en memoria (sin Neo4j).
        Útil para tests unitarios y demos.

        Topología de ejemplo:
            [A1] --Av.Principal(w=80)--> [A2] --Av.Principal(w=80)--> [A3]
             |                            |
          Calle(w=20)                  Calle(w=20)
             |                            |
            [B1] --Calle(w=20)---------->[B2]

        A1 y A2 están en la avenida principal.
        B1 y B2 son calles secundarias.
        """
        # TODO: implementar ciudad de ejemplo con intersecciones y segmentos
        raise NotImplementedError

    def write_to_neo4j(self) -> None:
        """
        Persiste el grafo actual (construido manualmente o simulado)
        en Neo4j para visualización con Bloom y análisis con GDS.

        Query CQL para crear nodo:
            MERGE (i:Intersection {node_id: $node_id})
            SET i.name = $name, i.latitude = $lat, i.longitude = $lon

        Query CQL para crear relación:
            MATCH (a:Intersection {node_id: $from_id})
            MATCH (b:Intersection {node_id: $to_id})
            MERGE (a)-[r:ROAD {segment_id: $seg_id}]->(b)
            SET r.category = $category, r.length_m = $length

        Raises
        ------
        RuntimeError si no hay driver de Neo4j configurado.
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver no configurado.")
        # TODO: implementar escritura a Neo4j
        raise NotImplementedError

    # ── Simulación de entidades ────────────────────────────────────────────

    def spawn_vehicle(self, node_id: str,
                      vehicle_type: VehicleType = VehicleType.CAR,
                      direction: Direction = Direction.NORTH) -> Vehicle:
        """
        Crea un vehículo en una intersección específica.

        Parameters
        ----------
        node_id      : Intersección donde aparece el vehículo.
        vehicle_type : Tipo de vehículo.
        direction    : Dirección de circulación.

        Returns
        -------
        Vehicle creado.
        """
        # TODO: implementar con UUID como entity_id
        raise NotImplementedError

    def spawn_pedestrian(self, node_id: str,
                         is_wheelchair: bool = False,
                         crossing_width_m: float = 10.0) -> Pedestrian:
        """
        Crea un peatón en una intersección específica.

        Parameters
        ----------
        node_id          : Intersección donde aparece el peatón.
        is_wheelchair    : True si usa silla de ruedas.
        crossing_width_m : Ancho del cruce en esa intersección.

        Returns
        -------
        Pedestrian creado.
        """
        # TODO: implementar con UUID como entity_id
        raise NotImplementedError

    # ── Tick del pipeline ──────────────────────────────────────────────────

    def tick(self, ctx: TrafficContext) -> SimulationSnapshot:
        """
        Avanza la simulación un ciclo.
        Genera entidades según las probabilidades del contexto:
          - Hora pico → más vehículos
          - Madrugada → pocos peatones, pocos vehículos
          - Lluvia    → menos ciclistas, más peatones vulnerables

        Parameters
        ----------
        ctx : Contexto ambiental del ciclo actual.

        Returns
        -------
        SimulationSnapshot con las entidades por intersección.
        """
        self._tick_count += 1
        # TODO: implementar generación probabilística de entidades
        raise NotImplementedError

    def apply_incident(self, segment_id: str) -> None:
        """
        Simula un incidente (accidente, obra) en un segmento.
        Reduce drásticamente el peso del segmento en el grafo.

        Parameters
        ----------
        segment_id : ID del segmento afectado.
        """
        # TODO: implementar modificación del peso en el grafo NetworkX
        raise NotImplementedError

    def export_pressure_map(self) -> Dict[str, float]:
        """
        Exporta la presión actual de cada intersección.
        Útil para el dashboard de Streamlit / Folium.

        Returns
        -------
        Dict node_id → presión (0.0 – ∞).
        """
        # TODO: implementar
        raise NotImplementedError

    def close(self) -> None:
        """Cierra la conexión a Neo4j si existe."""
        if self._driver:
            self._driver.close()