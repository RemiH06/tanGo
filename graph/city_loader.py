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

# ── Overpass API ──────────────────────────────────────────────────────────────

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Tipos de vía de OSM → RoadCategory de tanGo
OSM_HIGHWAY_MAP = {
    "motorway":       "HIGHWAY",
    "trunk":          "HIGHWAY",
    "primary":        "MAIN_AVENUE",
    "secondary":      "SECONDARY_AVENUE",
    "tertiary":       "STREET",
    "residential":    "STREET",
    "living_street":  "ALLEY",
    "service":        "ALLEY",
    "unclassified":   "STREET",
}

# Velocidades por defecto (km/h) si OSM no las especifica
DEFAULT_SPEEDS = {
    "HIGHWAY":          90.0,
    "MAIN_AVENUE":      60.0,
    "SECONDARY_AVENUE": 50.0,
    "STREET":           30.0,
    "ALLEY":            15.0,
}

# Umbral de street_count para determinar IntersectionType
# OSM cuenta cuántas vías confluyen en cada nodo
MASTER_HIGHWAY_TYPES = {"primary", "trunk", "motorway"}


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
                timeout=60,
                headers={"User-Agent": "tanGo-academic-project/0.1"},
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"  Descargados {len(data.get('elements',[]))} elementos")
            return data
        except requests.RequestException as e:
            logger.warning(f"  Intento {attempt}/{retry} falló: {e}")
            if attempt < retry:
                time.sleep(3 * attempt)
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

        # Inferir IntersectionType
        if any(t in MASTER_HIGHWAY_TYPES for t in way_types) and street_cnt >= 3:
            itype = "MASTER"
        elif street_cnt >= 2:
            itype = "NORMAL"
        else:
            itype = "BLIND"

        tango_nodes.append({
            "node_id":           f"osm_{osm_id}",
            "name":              _guess_name(osm_id, way_types, osm_ways),
            "latitude":          node_data["lat"],
            "longitude":         node_data["lon"],
            "intersection_type": itype,
            "osm_id":            osm_id,
            "street_count":      street_cnt,
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

    return {
        "metadata": {
            "ciudad":    "Guadalajara ZMG",
            "fuente":    "OpenStreetMap via Overpass API",
            "fecha":     datetime.now().isoformat(),
            "bbox":      raw.get("_bbox", {}),
            "n_nodes":   len(tango_nodes),
            "n_edges":   len(tango_edges),
        },
        "nodes": tango_nodes,
        "edges": tango_edges,
    }


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
    from core.road import (Intersection, IntersectionType,
                           RoadSegment, RoadCategory, Turn)

    data  = load_graph_from_json(path)
    graph = TrafficGraph()

    # Mapas de conversión
    itype_map = {
        "MASTER": IntersectionType.MASTER,
        "NORMAL": IntersectionType.NORMAL,
        "BLIND":  IntersectionType.BLIND,
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
            intersection_type = itype_map.get(n["intersection_type"],
                                              IntersectionType.NORMAL),
        ))

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
        "--output", default="graph/city_graph.json",
        help="Ruta de salida (default: graph/city_graph.json)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=80,
        help="Máximo de intersecciones (default: 80 — rápido en simulación)",
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
        # Default: 800m alrededor de Vallarta y López Mateos
        lat, lon = CITY_CENTERS["vallarta_lopez"]
        bbox = radius_to_bbox(lat, lon, 800)
        print(f"  Modo: default — 800m desde Vallarta y López Mateos")

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