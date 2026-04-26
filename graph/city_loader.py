"""
graph/city_loader.py
---------------------
Descarga el grafo vial real de una ciudad desde OpenStreetMap
vía Overpass API y lo guarda como JSON para uso del simulador.

Por qué Overpass API:
  - Gratuita, sin API key, sin límite razonable para queries académicos.
  - Datos reales de OpenStreetMap — intersecciones con coordenadas precisas.
  - Permite filtrar por tipo de vía (autopista, avenida, calle, etc.).

Por qué JSON:
  - Evita hardcodear coordenadas en el código.
  - Permite regenerar el grafo sin tocar el simulador.
  - Fácil de inspeccionar y editar manualmente si se necesita.
  - Sirve como caché — no se llama a la API en cada ejecución.

Esquema del JSON generado:
  {
    "metadata": { ciudad, fecha, bbox, n_nodes, n_edges },
    "nodes": [
      {
        "node_id": "osm_123456",
        "name": "Av. Vallarta y Av. López Mateos",
        "latitude": 20.6757,
        "longitude": -103.4093,
        "intersection_type": "MASTER",    ← inferido del tipo de vías que cruzan
        "osm_id": 123456,
        "street_count": 4                 ← cuántas calles confluyen aquí
      }, ...
    ],
    "edges": [
      {
        "segment_id": "osm_123456_789012",
        "from_node_id": "osm_123456",
        "to_node_id": "osm_789012",
        "category": "MAIN_AVENUE",
        "length_m": 350.4,
        "speed_limit_kmh": 60.0,
        "name": "Av. Vallarta",
        "oneway": false,
        "lanes": 2
      }, ...
    ]
  }

Uso:
  # Descargar grafo de Guadalajara (área de ~4km²)
  python graph/city_loader.py --city guadalajara --output graph/city_graph.json

  # Área personalizada por bounding box
  python graph/city_loader.py --bbox 20.64,20.70,-103.45,-103.34

  # Solo verificar el JSON existente
  python graph/city_loader.py --verify graph/city_graph.json
"""

from __future__ import annotations
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── Cargar city_config.json ───────────────────────────────────────────────────

_CONFIG_FILE = Path(__file__).parent / "city_config.json"

def _load_config() -> dict:
    """Lee city_config.json si existe, retorna defaults si no."""
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info("Configuracion cargada desde %s", _CONFIG_FILE.name)
        return cfg
    logger.warning("city_config.json no encontrado — usando defaults")
    return {}

_CFG = _load_config()

# ── Constantes desde config (con fallback hardcodeado) ───────────────────────

OVERPASS_URL = _CFG.get("overpass", {}).get(
    "url", "https://overpass-api.de/api/interpreter")

OSM_HIGHWAY_MAP: dict = _CFG.get("category_map", {
    "motorway":       "HIGHWAY",
    "trunk":          "HIGHWAY",
    "primary":        "MAIN_AVENUE",
    "secondary":      "SECONDARY_AVENUE",
    "tertiary":       "STREET",
    "residential":    "STREET",
    "living_street":  "ALLEY",
    "service":        "ALLEY",
    "unclassified":   "STREET",
})

DEFAULT_SPEEDS: dict = _CFG.get("default_speeds_kmh", {
    "HIGHWAY":          90.0,
    "MAIN_AVENUE":      60.0,
    "SECONDARY_AVENUE": 50.0,
    "STREET":           30.0,
    "ALLEY":            15.0,
})

_itype_rules = _CFG.get("intersection_type_rules", {})
MASTER_HIGHWAY_TYPES: set = set(
    _itype_rules.get("master_highway_types", ["primary", "trunk", "motorway"])
)
MASTER_MIN_STREETS: int = int(
    _itype_rules.get("master_min_street_count", 3)
)
BLIND_MAX_STREETS: int = int(
    _itype_rules.get("blind_max_street_count", 1)
)
# Combinaciones de tipos de vía que generan BLIND aunque haya 2+ calles
# (cruces entre calles residenciales sin semáforo)
BLIND_WAY_COMBOS: list = [
    frozenset(pair)
    for pair in _itype_rules.get("blind_way_combinations", [
        ["residential", "residential"],
        ["residential", "unclassified"],
        ["unclassified", "unclassified"],
        ["living_street", "living_street"],
        ["living_street", "residential"],
        ["service", "service"],
        ["service", "residential"],
    ])
]

_ov_cfg = _CFG.get("overpass", {})
OVERPASS_TIMEOUT:  int = int(_ov_cfg.get("timeout_s",       60))
OVERPASS_RETRY:    int = int(_ov_cfg.get("retry_attempts",   3))
OVERPASS_WAIT:     int = int(_ov_cfg.get("retry_wait_s",     3))
OVERPASS_UA:       str = _ov_cfg.get(
    "user_agent", "tanGo-academic-project/0.1")

BIDIR_ENABLED:     bool = _CFG.get("bidirectional", {}).get("enabled", True)
BIDIR_RESPECT_OW:  bool = _CFG.get("bidirectional", {}).get(
    "respect_oneway_tag", True)

_geo_rules = _CFG.get("geometry_rules", {})
GEO_MULTIWAY_MIN:  int = int(_geo_rules.get("multiway_min_streets", 5))
GEO_T_COUNT:       int = int(_geo_rules.get("t_street_count",       3))
GEO_DEFAULT:       str = _geo_rules.get("default", "cross")
GEO_PEDESTRIAN_TAGS: list = _geo_rules.get(
    "pedestrian_tags", ["pedestrian", "footway", "crossing"])
GEO_MERGE_TAGS:    list = _geo_rules.get(
    "merge_tags", ["motorway_link","trunk_link","primary_link","service"])

_city = _CFG.get("city", {})
CFG_CENTER_LAT = _city.get("center_lat", None)
CFG_CENTER_LON = _city.get("center_lon", None)
CFG_RADIUS_M   = int(_city.get("radius_m",  800))
CFG_MAX_NODES  = int(_city.get("max_nodes",  80))
CFG_OUTPUT     = _city.get("output", "graph/city_graph.json")

_cl = _CFG.get("intersection_clustering", {})
MERGE_RADIUS_M   = float(_cl.get("merge_radius_m",   15))
CLUSTER_RADIUS_M = float(_cl.get("cluster_radius_m",  60))
CLUSTER_COORD    = bool(_cl.get("cluster_coordination", True))


# ── Bounding boxes de ciudades predefinidas ───────────────────────────────────

CITY_BBOXES = {
    "guadalajara": {
        "south": 20.640, "north": 20.710,
        "west": -103.420, "east": -103.340,
        "center_lat": 20.675, "center_lon": -103.380,
    },
    "zapopan": {
        "south": 20.680, "north": 20.750,
        "west": -103.450, "east": -103.370,
        "center_lat": 20.715, "center_lon": -103.410,
    },
    "zmg_centro": {
        "south": 20.650, "north": 20.700,
        "west": -103.410, "east": -103.350,
        "center_lat": 20.675, "center_lon": -103.380,
    },
}

# Puntos de interés predefinidos para el modo radio
CITY_CENTERS = {
    "vallarta_lopez":    (20.6757, -103.4093),  # Av. Vallarta y López Mateos
    "vallarta_patria":   (20.6757, -103.3800),  # Av. Vallarta y Av. Patria
    "americas_mexico":   (20.6650, -103.3800),  # Av. Américas y Av. México
    "centro_gdl":        (20.6736, -103.3447),  # Centro histórico GDL
    "zapopan_centro":    (20.7209, -103.3893),  # Centro Zapopan
    "tlaquepaque":       (20.6452, -103.3105),  # Tlaquepaque centro
}


def radius_to_bbox(lat: float, lon: float, radius_m: float) -> dict:
    """
    Convierte un centro + radio en metros a un bounding box.
    Aproximación válida para radios < 50km en latitudes medias.

    Parameters
    ----------
    lat      : Latitud del centro.
    lon      : Longitud del centro.
    radius_m : Radio en metros.

    Returns
    -------
    Dict con south, north, west, east.
    """
    import math
    # 1 grado de latitud ≈ 111,320 m
    # 1 grado de longitud ≈ 111,320 * cos(lat) m
    delta_lat = radius_m / 111_320
    delta_lon = radius_m / (111_320 * math.cos(math.radians(lat)))
    return {
        "south": round(lat - delta_lat, 6),
        "north": round(lat + delta_lat, 6),
        "west":  round(lon - delta_lon, 6),
        "east":  round(lon + delta_lon, 6),
    }


# ── Descarga desde Overpass ───────────────────────────────────────────────────

def download_graph(bbox: dict,
                   max_nodes: int = 200,
                   retry: int = 3) -> dict:
    """
    Descarga el grafo vial de una zona desde Overpass API.

    Query QL:
      - Nodos de intersección (donde confluyen ≥2 vías)
      - Vías vehiculares (excluye senderos, ciclovías exclusivas, etc.)

    Parameters
    ----------
    bbox      : Dict con south, north, west, east en grados decimales.
    max_nodes : Límite de nodos para no saturar la API en áreas grandes.
    retry     : Intentos ante error de red.

    Returns
    -------
    Dict con la respuesta JSON de Overpass.
    """
    south = bbox["south"]
    north = bbox["north"]
    west  = bbox["west"]
    east  = bbox["east"]

    # Query Overpass QL — obtiene vías vehiculares y sus nodos
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|living_street|unclassified)$"]
         ({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    logger.info("Descargando grafo de Overpass API "
                f"bbox=[{south:.3f},{west:.3f},{north:.3f},{east:.3f}]...")

    for attempt in range(1, retry + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=OVERPASS_TIMEOUT,
                headers={"User-Agent": OVERPASS_UA},
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"  Descargados {len(data.get('elements',[]))} elementos")
            return data
        except requests.RequestException as e:
            logger.warning(f"  Intento {attempt}/{retry} falló: {e}")
            if attempt < retry:
                time.sleep(OVERPASS_WAIT * attempt)
            else:
                raise RuntimeError(
                    f"No se pudo contactar Overpass API tras {retry} intentos. "
                    "Verifica tu conexión a internet."
                ) from e


# ── Procesamiento del grafo ───────────────────────────────────────────────────

def process_graph(raw: dict, max_nodes: int = 200) -> dict:
    """
    Convierte la respuesta de Overpass al esquema de tanGo.

    Proceso:
      1. Indexar todos los nodos (lat/lon) por osm_id
      2. Para cada vía: extraer segmentos entre nodos consecutivos
      3. Identificar intersecciones (nodos compartidos por ≥2 vías)
      4. Inferir IntersectionType según el tipo de vías que confluyen
      5. Generar segmentos bidireccionales si la vía no es oneway

    Parameters
    ----------
    raw      : JSON crudo de Overpass.
    max_nodes: Límite de nodos de intersección a incluir.

    Returns
    -------
    Dict con el esquema de tanGo (metadata, nodes, edges).
    """
    elements = raw.get("elements", [])

    # Indexar nodos OSM por id
    osm_nodes: dict[int, dict] = {}
    for el in elements:
        if el["type"] == "node":
            osm_nodes[el["id"]] = {
                "lat": el["lat"],
                "lon": el["lon"],
            }

    # Indexar vías
    osm_ways: list[dict] = [el for el in elements if el["type"] == "way"]
    logger.info(f"  Nodos OSM: {len(osm_nodes)} | Vías: {len(osm_ways)}")

    # Contar en cuántas vías aparece cada nodo → identifica intersecciones
    node_way_count: dict[int, int]       = {}
    node_way_types: dict[int, set[str]]  = {}

    for way in osm_ways:
        hw_type = way.get("tags", {}).get("highway", "unclassified")
        for node_id in way.get("nodes", []):
            node_way_count[node_id] = node_way_count.get(node_id, 0) + 1
            node_way_types.setdefault(node_id, set()).add(hw_type)

    # Nodos de intersección: aparecen en ≥2 vías
    intersection_osm_ids = {
        nid for nid, count in node_way_count.items()
        if count >= 2 and nid in osm_nodes
    }

    logger.info(f"  Intersecciones encontradas: {len(intersection_osm_ids)}")

    # Limitar a max_nodes tomando los más conectados
    if len(intersection_osm_ids) > max_nodes:
        logger.info(f"  Limitando a {max_nodes} nodos más conectados...")
        intersection_osm_ids = set(
            sorted(intersection_osm_ids,
                   key=lambda n: node_way_count.get(n, 0),
                   reverse=True)[:max_nodes]
        )

    # Construir nodos tanGo
    tango_nodes = []
    for osm_id in intersection_osm_ids:
        node_data  = osm_nodes[osm_id]
        way_types  = node_way_types.get(osm_id, set())
        street_cnt = node_way_count.get(osm_id, 1)

        # ── Inferir IntersectionType con lógica granular ─────────────────
        # Regla 1: MASTER — cruza al menos una vía principal con 3+ ramas
        is_master = (
            any(t in MASTER_HIGHWAY_TYPES for t in way_types)
            and street_cnt >= MASTER_MIN_STREETS
        )

        # Regla 2: BLIND — cruce entre solo calles residenciales/servicio
        # (sin vías importantes), aunque tenga 2+ ramas
        way_types_list = sorted(way_types)
        is_blind_combo = any(
            frozenset([a, b]) in BLIND_WAY_COMBOS
            for i, a in enumerate(way_types_list)
            for b in way_types_list[i:]
        )
        is_blind = (
            street_cnt <= BLIND_MAX_STREETS
            or (is_blind_combo and not any(
                t in MASTER_HIGHWAY_TYPES
                or t in ("secondary", "tertiary")
                for t in way_types
            ))
        )

        if is_master:
            itype = "MASTER"
        elif is_blind:
            itype = "BLIND"
        else:
            itype = "NORMAL"

        # ── Peso por degree (conectividad del nodo) ───────────────────────
        # Más conexiones = más peso = más prioridad en el algoritmo.
        # Se normaliza sobre el máximo observado en el grafo.
        # Se guarda como degree_weight para usarlo en WeightEngine.
        degree_weight = round(1.0 + (street_cnt - 1) * 0.15, 2)

        # Inferir geometría desde OSM
        geometry = _infer_geometry(osm_id, way_types, street_cnt, osm_ways)

        tango_nodes.append({
            "node_id":           f"osm_{osm_id}",
            "name":              _guess_name(osm_id, way_types, osm_ways),
            "latitude":          node_data["lat"],
            "longitude":         node_data["lon"],
            "intersection_type": itype,
            "geometry":          geometry,
            "osm_ids":           [osm_id],
            "osm_id":            osm_id,
            "street_count":      street_cnt,
            "degree_weight":     degree_weight,
            "way_types":         sorted(way_types),
        })

    logger.info(f"  Nodos tanGo: {len(tango_nodes)}")

    # Construir aristas tanGo
    node_id_set = {n["osm_id"] for n in tango_nodes}
    tango_edges = []
    seen_edges  = set()

    for way in osm_ways:
        hw_type  = way.get("tags", {}).get("highway", "unclassified")
        category = OSM_HIGHWAY_MAP.get(hw_type, "STREET")
        oneway   = way.get("tags", {}).get("oneway", "no") in ("yes", "1", "true")
        name     = way.get("tags", {}).get("name", "")
        maxspeed = way.get("tags", {}).get("maxspeed", "")
        lanes    = int(way.get("tags", {}).get("lanes", "1"))

        # Parsear velocidad máxima
        try:
            speed = float(str(maxspeed).replace(" mph", "").replace(" km/h", ""))
        except (ValueError, AttributeError):
            speed = DEFAULT_SPEEDS.get(category, 30.0)

        way_nodes = way.get("nodes", [])

        # Generar segmentos entre nodos de intersección consecutivos
        i = 0
        while i < len(way_nodes) - 1:
            from_osm = way_nodes[i]
            # Buscar el siguiente nodo de intersección
            j = i + 1
            while j < len(way_nodes) and way_nodes[j] not in node_id_set:
                j += 1
            if j >= len(way_nodes):
                break

            to_osm = way_nodes[j]
            if from_osm in node_id_set and to_osm in node_id_set:
                # Calcular longitud aproximada del segmento
                from_data = osm_nodes.get(from_osm, {})
                to_data   = osm_nodes.get(to_osm,   {})
                length = _haversine(
                    from_data.get("lat", 0), from_data.get("lon", 0),
                    to_data.get("lat",   0), to_data.get("lon",   0),
                )

                seg_id = f"osm_{from_osm}_{to_osm}"
                if seg_id not in seen_edges:
                    tango_edges.append({
                        "segment_id":    seg_id,
                        "from_node_id":  f"osm_{from_osm}",
                        "to_node_id":    f"osm_{to_osm}",
                        "category":      category,
                        "length_m":      round(length, 1),
                        "speed_limit_kmh": speed,
                        "name":          name,
                        "oneway":        oneway,
                        "lanes":         lanes,
                    })
                    seen_edges.add(seg_id)

                    # Bidireccional si no es oneway
                    if not oneway:
                        rev_id = f"osm_{to_osm}_{from_osm}"
                        if rev_id not in seen_edges:
                            tango_edges.append({
                                "segment_id":    rev_id,
                                "from_node_id":  f"osm_{to_osm}",
                                "to_node_id":    f"osm_{from_osm}",
                                "category":      category,
                                "length_m":      round(length, 1),
                                "speed_limit_kmh": speed,
                                "name":          name,
                                "oneway":        False,
                                "lanes":         lanes,
                            })
                            seen_edges.add(rev_id)
            i = j

    logger.info(f"  Aristas tanGo: {len(tango_edges)}")

    # ── Eliminar nodos aislados ───────────────────────────────────────────────
    # Un nodo aislado es aquel que no aparece en ninguna arista.
    # Puede ocurrir cuando el segmento que lo conectaba se filtró porque
    # el nodo vecino no era intersección (tenía street_count < 2).
    nodes_in_edges = set()
    for e in tango_edges:
        nodes_in_edges.add(e["from_node_id"])
        nodes_in_edges.add(e["to_node_id"])

    isolated = [n for n in tango_nodes if n["node_id"] not in nodes_in_edges]
    if isolated:
        logger.warning(
            "  Eliminando %d nodos aislados (sin aristas): %s",
            len(isolated),
            [n["node_id"] for n in isolated[:5]]
        )
        tango_nodes = [n for n in tango_nodes if n["node_id"] in nodes_in_edges]

    logger.info(
        "  Nodos finales: %d (%d aislados eliminados)",
        len(tango_nodes), len(isolated)
    )

    # ── Calcular pesos estáticos (centralidad + degree + road_quality) ──
    tango_nodes = compute_static_weights(tango_nodes, tango_edges)

    # ── Fusionar nodos duplicados del mismo cruce físico ──────────────────
    tango_nodes = merge_nearby_nodes(tango_nodes, MERGE_RADIUS_M)

    # ── Identificar clusters de coordinación ─────────────────────────────
    clusters = {}
    if CLUSTER_COORD:
        clusters = find_intersection_clusters(tango_nodes, CLUSTER_RADIUS_M)
        node_to_cluster = {
            nid: cid
            for cid, members in clusters.items()
            for nid in members
        }
        for node in tango_nodes:
            node["cluster_id"] = node_to_cluster.get(node["node_id"])

    return {
        "metadata": {
            "ciudad":           "Guadalajara ZMG",
            "fuente":           "OpenStreetMap via Overpass API",
            "fecha":            datetime.now().isoformat(),
            "bbox":             raw.get("_bbox", {}),
            "n_nodes":          len(tango_nodes),
            "n_edges":          len(tango_edges),
            "merge_radius_m":   MERGE_RADIUS_M,
            "cluster_radius_m": CLUSTER_RADIUS_M,
            "n_clusters":       len(clusters),
        },
        "nodes":    tango_nodes,
        "edges":    tango_edges,
        "clusters": clusters,
    }


def _infer_geometry(osm_id: int, way_types: set,
                    street_count: int, ways: list) -> str:
    """
    Infiere la geometría de la intersección desde los datos de OSM.
    Las reglas se leen desde city_config.json["geometry_rules"].
    """
    for way in ways:
        if osm_id not in way.get("nodes", []):
            continue
        tags = way.get("tags", {})
        if tags.get("junction") == "roundabout":
            return "roundabout"
        hw = tags.get("highway", "")
        if hw in GEO_PEDESTRIAN_TAGS:
            return "pedestrian"
        if hw in GEO_MERGE_TAGS and street_count <= 2:
            return "merge"

    if street_count >= GEO_MULTIWAY_MIN:
        return "multiway"
    if street_count == GEO_T_COUNT:
        return "t"
    return GEO_DEFAULT


def _guess_name(osm_id: int, way_types: set, ways: list) -> str:
    """Intenta construir el nombre de la intersección desde las vías adyacentes."""
    names = []
    for way in ways:
        if osm_id in way.get("nodes", []):
            name = way.get("tags", {}).get("name", "")
            if name and name not in names:
                names.append(name)
            if len(names) >= 2:
                break
    if len(names) >= 2:
        return f"{names[0]} y {names[1]}"
    if names:
        return names[0]
    return f"Intersección OSM {osm_id}"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos coordenadas geográficas."""
    import math
    R   = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def compute_static_weights(tango_nodes: list[dict],
                            tango_edges: list[dict]) -> list[dict]:
    """
    Calcula los pesos estáticos de cada nodo usando tres dimensiones:

    1. degree_weight   : ya existe — 1 + (street_count - 1) × 0.15
    2. centrality      : betweenness centrality normalizada [0,1]
                         Mide qué tan central es el nodo para el flujo
                         de la red. Un nodo por el que pasan muchas rutas
                         cortas tiene alta centralidad.
    3. road_quality    : calidad promedio de las vías que llegan al nodo
                         (HIGHWAY=1.0, MAIN_AVENUE=0.8, SECONDARY=0.5,
                          STREET=0.2, ALLEY=0.1)

    El peso combinado es la media geométrica de los tres:
        combined = (degree × centrality_norm × road_quality) ^ (1/3)

    Donde centrality_norm se escala para que el nodo de mayor centralidad
    tenga centrality_weight = 1.5 y el menor 0.5.

    Parameters
    ----------
    tango_nodes : Lista de nodos del grafo tanGo.
    tango_edges : Lista de aristas del grafo tanGo.

    Returns
    -------
    Lista de nodos con campo "static_weight" agregado.
    """
    import math
    import networkx as nx

    # Construir grafo NetworkX temporal para calcular centralidad
    G = nx.DiGraph()
    node_ids = {n["node_id"] for n in tango_nodes}

    for n in tango_nodes:
        G.add_node(n["node_id"])

    for e in tango_edges:
        if e["from_node_id"] in node_ids and e["to_node_id"] in node_ids:
            G.add_edge(e["from_node_id"], e["to_node_id"],
                       weight=float(e.get("length_m", 300)))

    # Betweenness centrality — normalizada automáticamente por NetworkX
    logger.info("Calculando betweenness centrality...")
    if G.number_of_nodes() > 1:
        centrality = nx.betweenness_centrality(G, normalized=True, weight="weight")
    else:
        centrality = {n["node_id"]: 0.5 for n in tango_nodes}

    # Escalar centralidad a [0.5, 1.5]
    c_values = list(centrality.values())
    c_min, c_max = min(c_values), max(c_values)
    c_range = c_max - c_min if c_max > c_min else 1.0

    def scale_centrality(c: float) -> float:
        return 0.5 + (c - c_min) / c_range

    # Calidad de vía por categoría
    ROAD_QUALITY = {
        "HIGHWAY":          1.0,
        "MAIN_AVENUE":      0.8,
        "SECONDARY_AVENUE": 0.5,
        "STREET":           0.2,
        "ALLEY":            0.1,
    }

    # Calcular calidad de vía para cada nodo
    # (promedio de las categorías de sus aristas entrantes)
    node_road_quality: dict[str, float] = {}
    for n in tango_nodes:
        incoming = [e for e in tango_edges if e["to_node_id"] == n["node_id"]]
        if incoming:
            avg_q = sum(ROAD_QUALITY.get(e.get("category","STREET"), 0.2)
                        for e in incoming) / len(incoming)
        else:
            avg_q = 0.2
        node_road_quality[n["node_id"]] = avg_q

    # Añadir pesos estáticos a cada nodo
    for n in tango_nodes:
        nid          = n["node_id"]
        degree_w     = float(n.get("degree_weight", 1.0))
        centrality_w = scale_centrality(centrality.get(nid, 0.0))
        road_q       = node_road_quality.get(nid, 0.2)

        # Media geométrica de las tres dimensiones
        combined = (degree_w * centrality_w * road_q) ** (1/3)
        combined = round(combined, 3)

        n["static_weight"] = {
            "degree":      round(degree_w, 3),
            "centrality":  round(centrality_w, 3),
            "road_quality": round(road_q, 3),
            "combined":    combined,
        }
        n["node_weight"] = combined   # campo plano para acceso rápido

        logger.debug(
            "%s → degree=%.2f centrality=%.2f road_q=%.2f combined=%.3f",
            nid, degree_w, centrality_w, road_q, combined
        )

    logger.info(
        "Pesos estáticos calculados: %d nodos | "
        "combined min=%.3f max=%.3f",
        len(tango_nodes),
        min(n["node_weight"] for n in tango_nodes),
        max(n["node_weight"] for n in tango_nodes),
    )
    return tango_nodes


def merge_nearby_nodes(nodes: list[dict],
                       merge_radius_m: float = MERGE_RADIUS_M) -> list[dict]:
    """
    Fusiona nodos OSM que están a menos de merge_radius_m entre sí.

    Esto corrige el caso donde OSM genera múltiples nodos para lo que
    físicamente es un solo cruce — común en glorietas y cruces complejos.

    Algoritmo:
      - Para cada par de nodos, si la distancia es < merge_radius_m,
        se fusionan tomando el centroide geográfico como posición final.
      - El tipo resultante es el de mayor jerarquía (MASTER > NORMAL > BLIND).
      - Los osm_ids se acumulan en una lista para trazabilidad.
      - El street_count es la suma de ambos (más conectividad).

    Parameters
    ----------
    nodes         : Lista de nodos del grafo (output de process_graph).
    merge_radius_m: Radio de fusión en metros (default de city_config.json).

    Returns
    -------
    Lista de nodos fusionados — puede ser más corta que la entrada.
    """
    if not nodes:
        return nodes

    TYPE_RANK = {"MASTER": 3, "NORMAL": 2, "BLIND": 1}
    merged    = []
    used      = set()

    for i, a in enumerate(nodes):
        if i in used:
            continue
        group = [a]
        used.add(i)

        for j, b in enumerate(nodes):
            if j in used or j == i:
                continue
            dist = _haversine(a["latitude"], a["longitude"],
                              b["latitude"], b["longitude"])
            if dist <= merge_radius_m:
                group.append(b)
                used.add(j)

        if len(group) == 1:
            merged.append(a)
            continue

        # Centroide del grupo
        center_lat = sum(n["latitude"]  for n in group) / len(group)
        center_lon = sum(n["longitude"] for n in group) / len(group)

        # Tipo de mayor jerarquía
        best_type = max(group,
                        key=lambda n: TYPE_RANK.get(n["intersection_type"], 0))

        # Nombre del más conectado
        best_name = max(group, key=lambda n: n.get("street_count", 0))

        # OSM IDs fusionados (para trazabilidad)
        all_osm_ids = []
        for n in group:
            ids = n.get("osm_ids", [n.get("osm_id", 0)])
            all_osm_ids.extend(ids if isinstance(ids, list) else [ids])

        merged_node = {
            "node_id":           group[0]["node_id"],  # usar el ID del primero
            "name":              best_name["name"],
            "latitude":          round(center_lat, 7),
            "longitude":         round(center_lon, 7),
            "intersection_type": best_type["intersection_type"],
            "geometry":          best_type.get("geometry", "cross"),
            "osm_ids":           all_osm_ids,
            "osm_id":            group[0].get("osm_id", 0),
            "street_count":      sum(n.get("street_count", 1) for n in group),
            "merged_count":      len(group),
        }
        merged.append(merged_node)
        logger.info(
            "Fusionados %d nodos en radio %.0fm → %s",
            len(group), merge_radius_m, merged_node["name"]
        )

    logger.info("Fusión: %d nodos → %d (%.0f%% reducción)",
                len(nodes), len(merged),
                (1 - len(merged)/len(nodes)) * 100)
    return merged


def find_intersection_clusters(nodes: list[dict],
                                cluster_radius_m: float = CLUSTER_RADIUS_M
                                ) -> dict[str, list[str]]:
    """
    Identifica grupos de intersecciones cercanas que deben coordinarse.

    Un cluster es un grupo de nodos (post-fusión) que están a menos de
    cluster_radius_m entre sí. Estos nodos representan intersecciones
    distintas pero tan próximas que sus semáforos deben sincronizarse
    (ejemplo: avenida con camellón donde hay dos cruces separados 30m).

    La coordinación significa: si un nodo del cluster está en VERDE,
    los demás deben esperar o estar en ROJO — se trata como un semáforo
    complejo de múltiples puntos.

    Parameters
    ----------
    nodes           : Nodos post-fusión.
    cluster_radius_m: Radio de clustering en metros.

    Returns
    -------
    Dict cluster_id → lista de node_ids en ese cluster.
    Solo incluye clusters con 2+ nodos.
    """
    clusters: dict[str, list[str]] = {}
    assigned: dict[str, str]       = {}  # node_id → cluster_id
    cluster_idx = 0

    for i, a in enumerate(nodes):
        a_id = a["node_id"]
        if a_id in assigned:
            c_id = assigned[a_id]
        else:
            c_id = f"cluster_{cluster_idx}"
            cluster_idx += 1
            clusters[c_id] = [a_id]
            assigned[a_id] = c_id

        for j, b in enumerate(nodes):
            if i == j:
                continue
            b_id = b["node_id"]
            dist = _haversine(a["latitude"],  a["longitude"],
                              b["latitude"],  b["longitude"])
            if dist <= cluster_radius_m and b_id not in assigned:
                clusters[c_id].append(b_id)
                assigned[b_id] = c_id

    # Filtrar solo clusters con 2+ nodos
    multi = {k: v for k, v in clusters.items() if len(v) >= 2}
    if multi:
        logger.info(
            "Clusters de coordinacion: %d grupos con 2+ nodos",
            len(multi)
        )
        for cid, members in multi.items():
            logger.info("  %s: %s", cid, ", ".join(members))
    return multi


# ── Carga del JSON en el simulador ────────────────────────────────────────────

def load_graph_from_json(path: str | Path) -> dict:
    """
    Carga el grafo desde el JSON generado por city_loader.
    Retorna el dict tal cual — el simulador lo convierte a objetos.

    Parameters
    ----------
    path : Ruta al archivo city_graph.json.

    Returns
    -------
    Dict con metadata, nodes y edges.

    Raises
    ------
    FileNotFoundError si el archivo no existe.
    ValueError si el JSON no tiene el esquema esperado.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. "
            "Ejecuta primero: python graph/city_loader.py"
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for key in ("metadata", "nodes", "edges"):
        if key not in data:
            raise ValueError(f"JSON inválido — falta la clave '{key}'")

    logger.info(
        "Grafo cargado desde JSON: %d nodos, %d aristas | "
        "generado: %s",
        data["metadata"]["n_nodes"],
        data["metadata"]["n_edges"],
        data["metadata"]["fecha"][:10],
    )
    return data


def json_to_traffic_graph(path: str | Path) -> "TrafficGraph":
    """
    Carga el JSON y construye un TrafficGraph listo para usar.
    Reemplaza a build_sample_city() cuando se tienen datos reales.

    Parameters
    ----------
    path : Ruta al city_graph.json.

    Returns
    -------
    TrafficGraph con las intersecciones y segmentos del JSON.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from graph.simulator import TrafficGraph
    from core.road import (Intersection, IntersectionType, IntersectionGeometry,
                           RoadSegment, RoadCategory, Turn)

    data  = load_graph_from_json(path)
    graph = TrafficGraph()

    # Mapas de conversión
    itype_map = {
        "MASTER": IntersectionType.MASTER,
        "NORMAL": IntersectionType.NORMAL,
        "BLIND":  IntersectionType.BLIND,
    }
    geo_map = {
        "cross":      IntersectionGeometry.CROSS,
        "t":          IntersectionGeometry.T,
        "y":          IntersectionGeometry.Y,
        "roundabout": IntersectionGeometry.ROUNDABOUT,
        "pedestrian": IntersectionGeometry.PEDESTRIAN,
        "multiway":   IntersectionGeometry.MULTIWAY,
        "merge":      IntersectionGeometry.MERGE,
    }
    cat_map = {
        "HIGHWAY":          RoadCategory.HIGHWAY,
        "MAIN_AVENUE":      RoadCategory.MAIN_AVENUE,
        "SECONDARY_AVENUE": RoadCategory.SECONDARY_AVENUE,
        "STREET":           RoadCategory.STREET,
        "ALLEY":            RoadCategory.ALLEY,
    }

    # Nodos
    for n in data["nodes"]:
        graph.add_intersection(Intersection(
            node_id           = n["node_id"],
            name              = n["name"],
            latitude          = n["latitude"],
            longitude         = n["longitude"],
            intersection_type = itype_map.get(n.get("intersection_type","NORMAL"),
                                              IntersectionType.NORMAL),
            geometry          = geo_map.get(n.get("geometry", "cross"),
                                           IntersectionGeometry.CROSS),
            degree_weight     = float(n.get("degree_weight", 1.0)),
            node_weight       = float(n.get("node_weight", 1.0)),
        ))

    # Registrar clusters de coordinación en el grafo
    # Filtrar miembros del cluster contra nodos realmente cargados
    raw_clusters = data.get("clusters", {})
    clusters = {
        cid: [n for n in members if n in graph.intersections]
        for cid, members in raw_clusters.items()
    }
    clusters = {cid: mems for cid, mems in clusters.items() if len(mems) >= 2}

    if clusters:
        graph.intersection_clusters = clusters
        graph.node_to_cluster = {
            nid: cid
            for cid, members in clusters.items()
            for nid in members
        }
        logger.info(
            "Clusters de coordinacion registrados: %d grupos", len(clusters)
        )
    else:
        graph.intersection_clusters = {}
        graph.node_to_cluster = {}

    # Aristas (solo las que tienen ambos extremos en el grafo)
    skipped = 0
    for e in data["edges"]:
        if (e["from_node_id"] not in graph.intersections or
                e["to_node_id"] not in graph.intersections):
            skipped += 1
            continue
        try:
            graph.add_segment(RoadSegment(
                segment_id      = e["segment_id"],
                from_node_id    = e["from_node_id"],
                to_node_id      = e["to_node_id"],
                category        = cat_map.get(e["category"], RoadCategory.STREET),
                length_m        = float(e["length_m"]),
                speed_limit_kmh = float(e["speed_limit_kmh"]),
            ))
        except Exception as exc:
            skipped += 1
            logger.debug("Segmento omitido %s: %s", e["segment_id"], exc)

    if skipped:
        logger.warning("Segmentos omitidos: %d", skipped)

    # Verificar y eliminar nodos aislados en el grafo cargado
    isolated_ids = [nid for nid in graph.intersections
                    if graph.graph.degree(nid) == 0]
    if isolated_ids:
        logger.warning(
            "Nodos aislados en el grafo: %d — eliminando", len(isolated_ids)
        )
        for nid in isolated_ids:
            graph.graph.remove_node(nid)
            del graph.intersections[nid]

    logger.info(
        "TrafficGraph listo: %d nodos, %d aristas",
        graph.graph.number_of_nodes(),
        graph.graph.number_of_edges(),
    )
    return graph


# ── Verificación ──────────────────────────────────────────────────────────────

def verify_json(path: str | Path) -> None:
    """Imprime un resumen del JSON sin levantar el grafo completo."""
    data = load_graph_from_json(path)
    m    = data["metadata"]
    nodes = data["nodes"]
    edges = data["edges"]

    masters = sum(1 for n in nodes if n["intersection_type"] == "MASTER")
    normals = sum(1 for n in nodes if n["intersection_type"] == "NORMAL")
    blinds  = sum(1 for n in nodes if n["intersection_type"] == "BLIND")

    print(f"\n{'='*50}")
    print(f"  tanGo City Graph — {m['ciudad']}")
    print(f"{'='*50}")
    print(f"  Generado   : {m['fecha'][:19]}")
    print(f"  Fuente     : {m['fuente']}")
    print(f"  Nodos      : {m['n_nodes']} total")
    print(f"               {masters} MASTER · {normals} NORMAL · {blinds} BLIND")
    print(f"  Aristas    : {m['n_edges']}")
    print(f"\n  Primeras 5 intersecciones:")
    for n in nodes[:5]:
        print(f"    {n['node_id']} | {n['name'][:40]} "
              f"| {n['intersection_type']} "
              f"| ({n['latitude']:.4f}, {n['longitude']:.4f})")
    print(f"{'='*50}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga el grafo vial de la ZMG desde OpenStreetMap"
    )
    parser.add_argument(
        "--city", choices=list(CITY_BBOXES.keys()),
        default=None,
        help="Ciudad predefinida por bounding box",
    )
    parser.add_argument(
        "--center", choices=list(CITY_CENTERS.keys()),
        default=None,
        help="Centro predefinido + radio (usar con --radius)",
    )
    parser.add_argument(
        "--radius", type=float, default=800,
        help="Radio en metros desde --center (default: 800m ≈ ~10 cuadras)",
    )
    parser.add_argument(
        "--lat", type=float,
        help="Latitud del centro (alternativa a --center)",
    )
    parser.add_argument(
        "--lon", type=float,
        help="Longitud del centro (alternativa a --center)",
    )
    parser.add_argument(
        "--bbox",
        help="Bounding box manual: south,north,west,east",
    )
    parser.add_argument(
        "--output", default=CFG_OUTPUT,
        help=f"Ruta de salida (config: {CFG_OUTPUT})",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=CFG_MAX_NODES,
        help=f"Maximo de intersecciones (config: {CFG_MAX_NODES})",
    )
    parser.add_argument(
        "--verify",
        help="Solo verificar un JSON existente sin descargar",
    )
    args = parser.parse_args()

    if args.verify:
        verify_json(args.verify)
        raise SystemExit(0)

    # ── Determinar bbox ───────────────────────────────────────────────────────
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        bbox  = dict(zip(["south","north","west","east"], parts))
        print(f"  Modo: bbox manual {bbox}")

    elif args.lat and args.lon:
        bbox = radius_to_bbox(args.lat, args.lon, args.radius)
        print(f"  Modo: radio {args.radius:.0f}m desde ({args.lat:.4f}, {args.lon:.4f})")

    elif args.center:
        lat, lon = CITY_CENTERS[args.center]
        bbox = radius_to_bbox(lat, lon, args.radius)
        print(f"  Modo: radio {args.radius:.0f}m desde '{args.center}' "
              f"({lat:.4f}, {lon:.4f})")

    elif args.city:
        bbox = CITY_BBOXES[args.city]
        print(f"  Modo: ciudad '{args.city}'")

    else:
        # Default desde city_config.json
        if CFG_CENTER_LAT and CFG_CENTER_LON:
            r    = CFG_RADIUS_M
            bbox = radius_to_bbox(CFG_CENTER_LAT, CFG_CENTER_LON, r)
            print(f"  Modo: city_config — {r:.0f}m desde "
                  f"({CFG_CENTER_LAT:.4f}, {CFG_CENTER_LON:.4f})")
        else:
            lat, lon = CITY_CENTERS["vallarta_lopez"]
            bbox = radius_to_bbox(lat, lon, 800)
            print(f"  Modo: default — 800m desde Vallarta y Lopez Mateos")

    print(f"  Bbox: {bbox}")
    print(f"  Máx nodos: {args.max_nodes}")


    # Descargar y procesar
    raw   = download_graph(bbox, max_nodes=args.max_nodes)
    graph = process_graph(raw, max_nodes=args.max_nodes)

    # Guardar JSON
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"\n✓ JSON guardado en {output}")
    verify_json(output)