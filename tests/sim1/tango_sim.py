"""
tests/sim1/tango_sim.py
------------------------
Simulación visual de tanGo sobre mapa real de la ZMG.

Importa directamente del core:
  TrafficGraph, WeightEngine, TrafficContext,
  Intersection, IntersectionType, Vehicle, Pedestrian

Genera dos vistas:
  1. tango_sim.html   — grafo animado con Plotly sobre coordenadas reales
  2. tango_map.html   — mapa estático Folium con estado final

Ejecutar:
    python tests/sim1/tango_sim.py
"""

from __future__ import annotations
import sys, random, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster

from core.context       import TrafficContext
from core.algorithm     import TrafficAlgorithm, TickResult, NodeState
from core.road          import (Intersection, IntersectionType,
                                IntersectionGeometry, RoadSegment,
                                RoadCategory, Phase, Turn, TrafficAxis)
from core.entities      import Vehicle, Pedestrian, VehicleType, Direction
from graph.simulator    import TrafficGraph
from graph.city_loader  import json_to_traffic_graph, load_graph_from_json

import json as _json

OUTPUT_PLOTLY  = Path(__file__).parent / "tango_sim.html"
OUTPUT_FOLIUM  = Path(__file__).parent / "tango_map.html"
OUTPUT_VIS     = Path(__file__).parent / "tango_vis.html"
PARAMS_FILE    = Path(__file__).parent / "sim_params.json"

# Cargar parámetros desde JSON — edita sim_params.json para cambiar
# el comportamiento sin tocar este archivo
def load_params() -> dict:
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE, encoding="utf-8") as f:
            return _json.load(f)
    # Defaults si no existe el JSON
    return {
        "n_ticks": 40,
        "scenarios": [],
        "spawn": {"vehicle_multiplier": 1.0, "pedestrian_multiplier": 1.0,
                  "emergency_probability": 0.02, "wheelchair_probability": 0.08},
        "node_overrides": {},
        "experiment_presets": {},
    }

PARAMS = load_params()
N_TICKS = PARAMS.get("n_ticks", 40)


def compute_center(graph: TrafficGraph) -> tuple[float, float]:
    """Calcula el centroide geográfico del grafo actual."""
    nodes = list(graph.intersections.values())
    if not nodes:
        return 20.6656, -103.3863
    lat = sum(n.latitude  for n in nodes) / len(nodes)
    lon = sum(n.longitude for n in nodes) / len(nodes)
    return lat, lon

# JSON del grafo real — generado por graph/city_loader.py
CITY_JSON = ROOT / "graph" / "city_graph.json"

# ─────────────────────────────────────────────────────────────────────────────
#  RED VIAL REAL DE LA ZMG — 16 intersecciones
#  Coordenadas reales de Guadalajara / Zapopan
# ─────────────────────────────────────────────────────────────────────────────

ZMG_INTERSECTIONS = [
    # ── Avenidas principales — MASTER ────────────────────────────────────────
    ("M1", "Av. Vallarta y Av. López Mateos",      20.6757, -103.4093, IntersectionType.MASTER),
    ("M2", "Av. Vallarta y Av. Patria",            20.6757, -103.3800, IntersectionType.MASTER),
    ("M3", "Av. Vallarta y Av. Chapultepec",       20.6756, -103.3575, IntersectionType.MASTER),
    ("M4", "Periférico y Av. López Mateos",        20.6540, -103.4093, IntersectionType.MASTER),
    ("M5", "Periférico y Av. Patria",              20.6540, -103.3800, IntersectionType.MASTER),
    # ── Avenidas secundarias — NORMAL ────────────────────────────────────────
    ("N1", "Av. Américas y López Mateos",          20.6650, -103.4093, IntersectionType.NORMAL),
    ("N2", "Av. Américas y Av. Patria",            20.6650, -103.3800, IntersectionType.NORMAL),
    ("N3", "Av. Américas y Chapultepec",           20.6650, -103.3575, IntersectionType.NORMAL),
    ("N4", "Av. Tepeyac y López Mateos",           20.6757, -103.4300, IntersectionType.NORMAL),
    ("N5", "Av. Tepeyac y Av. Patria",             20.6540, -103.3575, IntersectionType.NORMAL),
    # ── Calles internas (colonias) — BLIND ───────────────────────────────────
    ("B1", "C. Lerdo de Tejada y Colonias",        20.6700, -103.3950, IntersectionType.BLIND),
    ("B2", "C. Madrid y Col. Moderna",             20.6700, -103.3700, IntersectionType.BLIND),
    ("B3", "C. Guadalupe y Col. Chapalita",        20.6600, -103.4000, IntersectionType.BLIND),
    ("B4", "C. Enrique Díaz y Col. Ladrón",        20.6600, -103.3700, IntersectionType.BLIND),
    ("B5", "C. Independencia y Col. Providencia",  20.6820, -103.3850, IntersectionType.BLIND),
    ("B6", "C. Manuel Acuña y Col. Del Fresno",    20.6480, -103.3900, IntersectionType.BLIND),
]

ZMG_SEGMENTS = [
    # Av. Vallarta (MAIN_AVENUE) — bidireccional
    ("seg-M1-M2", "M1","M2", RoadCategory.MAIN_AVENUE,  1800.0, 60.0),
    ("seg-M2-M3", "M2","M3", RoadCategory.MAIN_AVENUE,  1200.0, 60.0),
    ("seg-M1-N4", "M1","N4", RoadCategory.MAIN_AVENUE,  1500.0, 60.0),
    # Periférico (MAIN_AVENUE) — bidireccional
    ("seg-M4-M5", "M4","M5", RoadCategory.MAIN_AVENUE,  1800.0, 80.0),
    ("seg-M5-N5", "M5","N5", RoadCategory.MAIN_AVENUE,  1200.0, 80.0),
    # Av. Américas (SECONDARY_AVENUE) — bidireccional
    ("seg-N1-N2", "N1","N2", RoadCategory.SECONDARY_AVENUE, 1800.0, 50.0),
    ("seg-N2-N3", "N2","N3", RoadCategory.SECONDARY_AVENUE, 1200.0, 50.0),
    # Transversales principales — bidireccional
    ("seg-M1-N1", "M1","N1", RoadCategory.SECONDARY_AVENUE, 600.0, 50.0),
    ("seg-N1-M4", "N1","M4", RoadCategory.SECONDARY_AVENUE, 600.0, 50.0),
    ("seg-M2-N2", "M2","N2", RoadCategory.SECONDARY_AVENUE, 600.0, 50.0),
    ("seg-N2-M5", "N2","M5", RoadCategory.SECONDARY_AVENUE, 600.0, 50.0),
    ("seg-M3-N3", "M3","N3", RoadCategory.SECONDARY_AVENUE, 600.0, 50.0),
    ("seg-N3-N5", "N3","N5", RoadCategory.SECONDARY_AVENUE, 600.0, 50.0),
    # Calles internas (STREET) — bidireccional, conectan BLIND a NORMAL
    ("seg-N1-B1", "N1","B1", RoadCategory.STREET, 400.0, 30.0),
    ("seg-B1-N2", "B1","N2", RoadCategory.STREET, 400.0, 30.0),
    ("seg-N2-B2", "N2","B2", RoadCategory.STREET, 300.0, 30.0),
    ("seg-B2-N3", "B2","N3", RoadCategory.STREET, 300.0, 30.0),
    ("seg-N1-B3", "N1","B3", RoadCategory.STREET, 350.0, 30.0),
    ("seg-B3-M4", "B3","M4", RoadCategory.STREET, 350.0, 30.0),
    ("seg-N2-B4", "N2","B4", RoadCategory.STREET, 300.0, 30.0),
    ("seg-B4-M5", "B4","M5", RoadCategory.STREET, 300.0, 30.0),
    ("seg-M2-B5", "M2","B5", RoadCategory.STREET, 500.0, 30.0),
    ("seg-B5-N2", "B5","N2", RoadCategory.STREET, 500.0, 30.0),
    ("seg-M5-B6", "M5","B6", RoadCategory.STREET, 400.0, 30.0),
    ("seg-B6-N5", "B6","N5", RoadCategory.STREET, 400.0, 30.0),
]

SCENARIOS = [
    dict(label="Hora pico — lun 8am",
         timestamp=datetime(2024,3,4,8,0),
         temperature_c=22.0, is_raining=False,
         wind_speed_kmh=10.0, visibility_m=10000.0),
    dict(label="Madrugada — mié 2am",
         timestamp=datetime(2024,3,6,2,0),
         temperature_c=18.0, is_raining=False,
         wind_speed_kmh=5.0,  visibility_m=10000.0),
    dict(label="Lluvia — sáb 3pm",
         timestamp=datetime(2024,3,9,15,0),
         temperature_c=16.0, is_raining=True,
         wind_speed_kmh=20.0, visibility_m=3000.0),
]

PHASE_COLOR = {"green":"#22c55e", "yellow":"#eab308", "red":"#ef4444", "blink":"#f59e0b"}
TYPE_SYMBOL = {IntersectionType.MASTER:"star", IntersectionType.NORMAL:"circle",
               IntersectionType.BLIND:"diamond"}
TYPE_RING   = {IntersectionType.MASTER:"#f59e0b", IntersectionType.NORMAL:"#3b82f6",
               IntersectionType.BLIND:"#64748b"}


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTRUCCIÓN DEL GRAFO ZMG
# ─────────────────────────────────────────────────────────────────────────────

def build_zmg_graph() -> TrafficGraph:
    """Construye el grafo real de la ZMG con 16 intersecciones."""
    g = TrafficGraph()

    for node_id, name, lat, lon, itype in ZMG_INTERSECTIONS:
        g.add_intersection(Intersection(node_id, name, lat, lon,
                                        intersection_type=itype))

    for seg_id, frm, to, cat, length, speed in ZMG_SEGMENTS:
        # Bidireccional
        g.add_segment(RoadSegment(f"{seg_id}-fwd", frm, to, cat, length, speed))
        g.add_segment(RoadSegment(f"{seg_id}-bwd", to, frm, cat, length, speed))

    print(f"  Grafo ZMG: {g.graph.number_of_nodes()} nodos, "
          f"{g.graph.number_of_edges()} aristas")
    return g


# ─────────────────────────────────────────────────────────────────────────────
#  SPAWN DE ENTIDADES — más denso y realista
# ─────────────────────────────────────────────────────────────────────────────

V_TYPES_POOL = (
    [VehicleType.CAR] * 60 +
    [VehicleType.MOTORCYCLE] * 15 +
    [VehicleType.BUS] * 10 +
    [VehicleType.TRUCK] * 8 +
    [VehicleType.BICYCLE] * 5 +
    [VehicleType.EMERGENCY] * 2
)

def spawn_for_node(node_id: str, itype: IntersectionType,
                   ctx: TrafficContext,
                   spawn_params: dict | None = None) -> list:
    """
    Genera entidades para un nodo según el tipo de intersección,
    el contexto ambiental y los multiplicadores de sim_params.json.

    spawn_params puede venir de:
      - PARAMS["spawn"]           (base)
      - PARAMS["experiment_presets"][key]  (preset activo)
      - PARAMS["node_overrides"][node_id]  (override fijo por nodo)
    """
    import uuid

    sp  = spawn_params or PARAMS.get("spawn", {})
    vm  = float(sp.get("vehicle_multiplier",  1.0))
    pm  = float(sp.get("pedestrian_multiplier", 1.0))
    ep  = float(sp.get("emergency_probability", 0.02))
    wcp = float(sp.get("wheelchair_probability", 0.08))
    brf = float(sp.get("bicycle_rain_factor", 0.2))

    # Override fijo por nodo (suma fija en TODOS los ticks)
    node_ov = PARAMS.get("node_overrides", {}).get(node_id, {})

    # Volumen base × multiplicador
    if itype == IntersectionType.MASTER:
        nv_base  = random.randint(8,20)  if ctx.is_rush_hour else                    random.randint(1,4)   if ctx.is_late_night else random.randint(5,14)
        np_base  = random.randint(5,15)  if ctx.is_rush_hour else                    random.randint(0,2)   if ctx.is_late_night else random.randint(2,8)
    elif itype == IntersectionType.NORMAL:
        nv_base  = random.randint(5,12)  if ctx.is_rush_hour else                    random.randint(0,3)   if ctx.is_late_night else random.randint(3,8)
        np_base  = random.randint(3,10)  if ctx.is_rush_hour else                    random.randint(0,1)   if ctx.is_late_night else random.randint(1,6)
    else:
        nv_base  = random.randint(2,7)   if ctx.is_rush_hour else                    random.randint(0,2)   if ctx.is_late_night else random.randint(1,5)
        np_base  = random.randint(0,3)

    nv  = max(0, int(nv_base  * vm))
    np_ = max(0, int(np_base  * pm))

    # Pool de vehículos — ajustar bicicletas con lluvia
    if ctx.is_raining:
        pool = ([v for v in V_TYPES_POOL if v != VehicleType.BICYCLE]
                + [VehicleType.CAR] * 5)
    else:
        pool = V_TYPES_POOL

    entities = []

    # Emergencias según probabilidad del preset
    only_master = sp.get("only_master_nodes", False)
    should_spawn_emerg = (not only_master or itype == IntersectionType.MASTER)
    if should_spawn_emerg and random.random() < ep:
        entities.append(Vehicle(str(uuid.uuid4()), VehicleType.EMERGENCY,
                                random.choice(list(Direction))))

    for _ in range(nv):
        vtype = random.choice(pool)
        if vtype == VehicleType.EMERGENCY:
            vtype = VehicleType.CAR   # emergencias solo via probabilidad
        entities.append(Vehicle(str(uuid.uuid4()), vtype,
                                random.choice(list(Direction))))

    for _ in range(np_):
        wc = random.random() < wcp
        entities.append(Pedestrian(str(uuid.uuid4()), is_wheelchair=wc))

    # Aplicar overrides fijos del nodo (suma adicional)
    for _ in range(int(node_ov.get("cars", 0))):
        entities.append(Vehicle(str(uuid.uuid4()), VehicleType.CAR,
                                random.choice(list(Direction))))
    for _ in range(int(node_ov.get("buses", 0))):
        entities.append(Vehicle(str(uuid.uuid4()), VehicleType.BUS,
                                random.choice(list(Direction))))
    for _ in range(int(node_ov.get("pedestrians", 0))):
        entities.append(Pedestrian(str(uuid.uuid4())))
    for _ in range(int(node_ov.get("wheelchairs", 0))):
        entities.append(Pedestrian(str(uuid.uuid4()), is_wheelchair=True))
    for _ in range(int(node_ov.get("emergency", 0))):
        entities.append(Vehicle(str(uuid.uuid4()), VehicleType.EMERGENCY,
                                random.choice(list(Direction))))

    return entities


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def simulate(scenario: dict, graph: TrafficGraph, n_ticks: int,
             spawn_params: dict | None = None) -> list[dict]:
    """
    Corre n_ticks del algoritmo para un escenario dado.
    Usa TrafficAlgorithm del core — sin lógica duplicada aquí.

    Parameters
    ----------
    scenario     : Dict con timestamp, temperatura, lluvia, etc.
    graph        : Grafo vial ya construido.
    n_ticks      : Número de ticks a simular.
    spawn_params : Multiplicadores de spawn del sim_params.json.

    Returns
    -------
    Lista de frames serializables para la visualización.
    """
    # Parsear timestamp si viene como string
    sc_clean = {}
    for k, v in scenario.items():
        if k == "label": continue
        if k == "timestamp" and isinstance(v, str):
            sc_clean[k] = datetime.fromisoformat(v)
        else:
            sc_clean[k] = v

    ctx  = TrafficContext.build(**sc_clean)
    algo = TrafficAlgorithm(graph)
    algo.reset()

    history = []

    for _ in range(n_ticks):
        # Generar entidades (única responsabilidad que queda en sim)
        entities_by_node = {
            node_id: spawn_for_node(
                node_id, inter.intersection_type, ctx,
                spawn_params=spawn_params
            )
            for node_id, inter in graph.intersections.items()
        }

        # Ejecutar el algoritmo — todo lo complejo está en TrafficAlgorithm
        result: TickResult = algo.run_tick(entities_by_node, ctx)

        # Convertir TickResult al formato que espera la visualización
        frame = _tick_result_to_frame(result, graph)
        history.append(frame)

    return history


def _tick_result_to_frame(result: TickResult, graph: TrafficGraph) -> dict:
    """
    Convierte un TickResult en el dict que usa la visualización.
    Separa el formato de visualización del resultado del algoritmo.
    """
    nodes_frame = {}
    for node_id, ns in result.nodes.items():
        inter = graph.intersections[node_id]
        nodes_frame[node_id] = {
            "phase":        ns.phase,
            "phase_ns":     ns.phase_ns,
            "phase_ew":     ns.phase_ew,
            "active_axis":  ns.active_axis,
            "signals":      ns.signals,
            "pressure":     ns.pressure,
            "pressure_own": ns.pressure_own,
            "pressure_ns":  ns.pressure_ns,
            "pressure_ew":  ns.pressure_ew,
            "wave_offset_s": ns.wave_offset_s,
            "itype":        inter.intersection_type,
            "geometry":     inter.geometry,
            "geo_label":    inter.geometry_label,
            "has_light":    inter.has_traffic_light,
            "threshold":    ns.threshold,
            "timeout":      ns.timeout_ticks,
            "ticks_red":    ns.ticks_in_phase,
            "name":         inter.name,
            "lat":          inter.latitude,
            "lon":          inter.longitude,
            "counts":       ns.entity_counts,
            "cluster_id":   ns.cluster_id,
        }

    return {
        "tick":    result.tick_number,
        "nodes":   nodes_frame,
        "flows":   result.flows,
        "total":   result.total_entities,
        "greens":  result.green_count,
        "yellows": result.yellow_count,
        "reds":    result.red_count,
        "blinks":  result.blink_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY — animación sobre coordenadas reales
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly(graph: TrafficGraph, all_histories: list[tuple]) -> go.Figure:
    from core.road import RoadCategory

    # Aristas estáticas
    drawn = set()
    edge_traces = []
    for from_id, to_id, data in graph.graph.edges(data=True):
        pair = tuple(sorted([from_id,to_id]))
        if pair in drawn: continue
        drawn.add(pair)
        seg = data["segment"]
        n_from = graph.intersections[from_id]
        n_to   = graph.intersections[to_id]
        color = {RoadCategory.MAIN_AVENUE:"#1d4ed8",
                 RoadCategory.SECONDARY_AVENUE:"#6d28d9",
                 RoadCategory.STREET:"#1e293b"}.get(seg.category,"#1e293b")
        width = {RoadCategory.MAIN_AVENUE:4,
                 RoadCategory.SECONDARY_AVENUE:2.5,
                 RoadCategory.STREET:1.2}.get(seg.category,1.2)
        edge_traces.append(go.Scattergeo(
            lon=[n_from.longitude, n_to.longitude, None],
            lat=[n_from.latitude,  n_to.latitude,  None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip", showlegend=False,
        ))

    # Leyenda
    legend = [
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=12,color="#22c55e",symbol="circle"),
                      name="Verde",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=12,color="#eab308",symbol="circle"),
                      name="Amarillo",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=12,color="#ef4444",symbol="circle"),
                      name="Rojo",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=11,color="#1e293b",symbol="star",
                                  line=dict(color="#f59e0b",width=2)),
                      name="★ MASTER",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=11,color="#1e293b",symbol="circle",
                                  line=dict(color="#3b82f6",width=2)),
                      name="● NORMAL",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=11,color="#1e293b",symbol="diamond",
                                  line=dict(color="#64748b",width=2)),
                      name="◆ BLIND",showlegend=True),
        # Geometrías
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=10,color="#475569",symbol="circle"),
                      name="＋ CROSS · ⊤ T · Ψ Y",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="markers",
                      marker=dict(size=10,color="#475569",symbol="circle"),
                      name="⊙ Glorieta · ✳ Multiway · ♿ Peatonal",showlegend=True),
        # Flujo
        go.Scattergeo(lon=[None],lat=[None],mode="lines",
                      line=dict(width=4,color="#22c55e"),
                      name="Flujo bajo (<8 veh)",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="lines",
                      line=dict(width=4,color="#f59e0b"),
                      name="Flujo medio (8-14)",showlegend=True),
        go.Scattergeo(lon=[None],lat=[None],mode="lines",
                      line=dict(width=4,color="#ef4444"),
                      name="Flujo alto (≥15 veh)",showlegend=True),
    ]

    all_frames   = []
    slider_steps = []

    for scenario_label, history in all_histories:
        for snap in history:
            node_ids = list(snap["nodes"].keys())
            lons, lats, colors, sizes, symbols, borders, hovers, texts = \
                [],[],[],[],[],[],[],[]

            for n in node_ids:
                nd = snap["nodes"][n]
                lons.append(nd["lon"])
                lats.append(nd["lat"])
                # Color principal = eje activo en verde
                # Si ambos están en rojo (transición) → rojo
                # Si hay amarillo → amarillo
                if nd["phase_ns"] == "yellow" or nd["phase_ew"] == "yellow":
                    main_color = PHASE_COLOR["yellow"]
                elif nd["phase_ns"] == "green" or nd["phase_ew"] == "green":
                    main_color = PHASE_COLOR["green"]
                else:
                    main_color = PHASE_COLOR["red"]
                colors.append(main_color)

                # Símbolo: flecha según eje activo para indicar qué dirección fluye
                axis = nd.get("active_axis", "ns")
                if not nd["has_light"]:
                    sym = TYPE_SYMBOL[nd["itype"]]  # diamante para BLIND
                elif nd["phase"] == "red":
                    sym = "circle-x"  # X cuando ambos ejes en rojo
                elif axis == "ns":
                    sym = "arrow-up"  # flecha arriba = NS en verde
                else:
                    sym = "arrow-right"  # flecha derecha = EW en verde
                symbols.append(sym)
                borders.append(TYPE_RING[nd["itype"]])

                base = {"master":22,"normal":17,"blind":12}[nd["itype"].value]
                sizes.append(min(42, base + nd["pressure"]*0.08))

                pct = min(100, nd["pressure"]/nd["threshold"]*100)
                c   = nd["counts"]
                timeout_info = (
                    f"⏱ timeout en {max(0, nd['timeout'] - nd['ticks_red'])} ticks"
                    if nd["phase"]=="red" and nd["has_light"] else ""
                )
                ns_col = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(nd["phase_ns"],"⚫")
                ew_col = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(nd["phase_ew"],"⚫")
                geo = nd.get("geo_label","?")
                geo_name = nd.get("geometry", IntersectionGeometry.CROSS)
                geo_str = geo_name.value if hasattr(geo_name,"value") else str(geo_name)
                hovers.append(
                    f"<b>{n}</b> {nd['name']}<br>"
                    f"Tipo: {nd['itype'].value.upper()} | "
                    f"Geometría: {geo} {geo_str.upper()}<br>"
                    f"{'🚦 Con semáforo' if nd['has_light'] else '⛔ Sin semáforo'}<br>"
                    f"─────────────────<br>"
                    f"{ns_col} Eje N-S: <b>{nd['phase_ns'].upper()}</b><br>"
                    f"{ew_col} Eje E-O: <b>{nd['phase_ew'].upper()}</b><br>"
                    f"Eje activo: {nd['active_axis'].upper()} {timeout_info}<br>"
                    f"─────────────────<br>"
                    f"Presión: {nd['pressure']:.1f}/{nd['threshold']:.0f} ({pct:.0f}%)<br>"
                    f"🚗{c.get('CAR',0)} 🏍{c.get('MOTORCYCLE',0)} "
                    f"🚌{c.get('BUS',0)} 🚛{c.get('TRUCK',0)} "
                    f"🚲{c.get('BICYCLE',0)} 🚑{c.get('EMERGENCY',0)}<br>"
                    f"🚶{c.get('PEDESTRIAN',0)} ♿{c.get('WHEELCHAIR',0)}"
                )
                texts.append(n)

            # Etiqueta de geometría bajo el ID del nodo
            geo_labels = []
            for n in node_ids:
                nd = snap["nodes"][n]
                geo_labels.append(nd.get("geo_label","?"))

            node_trace = go.Scattergeo(
                lon=lons, lat=lats,
                mode="markers+text",
                marker=dict(size=sizes, color=colors, symbol=symbols,
                            line=dict(width=2,color=borders), opacity=0.9),
                text=[f"{nid}<br><sup>{gl}</sup>"
                      for nid, gl in zip(node_ids, geo_labels)],
                textposition="middle center",
                textfont=dict(size=9,color="#f8fafc",family="monospace"),
                hovertext=hovers, hoverinfo="text",
                showlegend=False,
            )

            # ── Flechas de flujo animadas por arista ──────────────────────
            # Una traza por par de nodos — grosor proporcional al volumen
            flow_traces = []
            for fl in snap["flows"]:
                if fl["from"] not in graph.intersections: continue
                if fl["to"]   not in graph.intersections: continue
                total = fl["fwd"] + fl["bwd"]
                if total == 0: continue

                n_a = graph.intersections[fl["from"]]
                n_b = graph.intersections[fl["to"]]

                # Color según intensidad de flujo
                if total >= 15:
                    flow_color = "#ef4444"   # rojo — flujo alto
                elif total >= 8:
                    flow_color = "#f59e0b"   # ámbar — flujo medio
                else:
                    flow_color = "#22c55e"   # verde — flujo bajo

                flow_width = min(8, 1.5 + total * 0.3)

                # Dirección dominante — flecha apunta al destino mayor
                if fl["fwd"] >= fl["bwd"]:
                    lon_seq = [n_a.longitude, n_b.longitude, None]
                    lat_seq = [n_a.latitude,  n_b.latitude,  None]
                else:
                    lon_seq = [n_b.longitude, n_a.longitude, None]
                    lat_seq = [n_b.latitude,  n_a.latitude,  None]

                flow_traces.append(go.Scattergeo(
                    lon=lon_seq, lat=lat_seq,
                    mode="lines",
                    line=dict(width=flow_width, color=flow_color),
                    opacity=0.55,
                    hoverinfo="skip",
                    showlegend=False,
                ))

                # Texto de volumen en el centro de la arista
                flow_traces.append(go.Scattergeo(
                    lon=[(n_a.longitude+n_b.longitude)/2],
                    lat=[(n_a.latitude +n_b.latitude )/2],
                    mode="text",
                    text=[f"↑{fl['fwd']} ↓{fl['bwd']}"],
                    textfont=dict(size=7, color="#94a3b8"),
                    hoverinfo="skip",
                    showlegend=False,
                ))

            flow_annotations = []

            title_str = (
                f"{scenario_label} | tick #{snap['tick']} | "
                f"entidades: {snap['total']} | "
                f"en verde: {snap['greens']}/{len(snap['nodes'])}"
            )
            frame_name = f"{scenario_label[:8]}|{snap['tick']}"

            all_frames.append(go.Frame(
                data=flow_traces + [node_trace],
                name=frame_name,
                layout=go.Layout(
                    title=dict(text=title_str,
                               font=dict(size=12,color="#e2e8f0")),
                    geo=dict(
                        showcountries=True,
                        countrycolor="#1e293b",
                        showland=True,
                        landcolor="#0f1117",
                        showocean=False,
                        showrivers=False,
                        showlakes=False,
                        bgcolor="#0f1117",
                        projection_type="mercator",
                        center=dict(lon=-103.39, lat=20.666),
                        lonaxis_range=[-103.45, -103.34],
                        lataxis_range=[20.64, 20.70],
                    ),
                )
            ))
            slider_steps.append(dict(
                args=[[frame_name],
                      {"frame":{"duration":500,"redraw":True},
                       "mode":"immediate",
                       "transition":{"duration":250}}],
                label=f"{scenario_label[:6]} t{snap['tick']}",
                method="animate",
            ))

    # Nodos iniciales
    init_lons = [d[2] for d in ZMG_INTERSECTIONS]
    init_lats = [d[3] for d in ZMG_INTERSECTIONS]
    init_node = go.Scattergeo(
        lon=init_lons, lat=init_lats,
        mode="markers+text",
        marker=dict(size=14,color="#ef4444",
                    line=dict(width=2,color="#f59e0b")),
        text=[d[0] for d in ZMG_INTERSECTIONS],
        textposition="middle center",
        textfont=dict(size=9,color="#f8fafc"),
        hoverinfo="skip", showlegend=False,
    )

    fig = go.Figure(
        data=edge_traces + [init_node] + legend,
        frames=all_frames,
    )

    fig.update_layout(
        title=dict(
            text="tanGo — ZMG Guadalajara · Semáforo inteligente",
            font=dict(size=14,color="#e2e8f0",family="monospace"), x=0.5,
        ),
        paper_bgcolor="#0f1117",
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="#1a1d2e",bordercolor="#2a2d3e",
                    borderwidth=1,font=dict(size=10),x=1.01,y=1),
        geo=dict(
            showcountries=True, countrycolor="#1e293b",
            showland=True, landcolor="#0f1117",
            showocean=False, showrivers=False, showlakes=False,
            bgcolor="#0f1117",
            projection_type="mercator",
            center=dict(lon=-103.39, lat=20.666),
            lonaxis_range=[-103.45, -103.34],
            lataxis_range=[20.64, 20.70],
        ),
        margin=dict(l=0,r=160,t=50,b=120),
        height=700,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.08, x=0.5, xanchor="center",
            direction="left",
            buttons=[
                dict(label="▶ Reproducir", method="animate",
                     args=[None,{"frame":{"duration":600,"redraw":True},
                                 "fromcurrent":True,
                                 "transition":{"duration":300}}]),
                dict(label="⏸ Pausar", method="animate",
                     args=[[None],{"frame":{"duration":0,"redraw":False},
                                   "mode":"immediate"}]),
            ],
            bgcolor="#1a1d2e",bordercolor="#2a2d3e",
            font=dict(color="#e2e8f0"),
        )],
        sliders=[dict(
            active=0, steps=slider_steps,
            currentvalue=dict(prefix="",font=dict(color="#e2e8f0",size=10)),
            pad=dict(t=55,b=5),
            bgcolor="#1a1d2e",bordercolor="#2a2d3e",
            tickcolor="#64748b",font=dict(color="#64748b",size=7),
            len=0.95,x=0.025,
        )],
        annotations=[
            dict(x=0.5,y=-0.16,xref="paper",yref="paper",
                 text="★ MASTER (umbral 120) · ● NORMAL (umbral 100) · "
                      "◆ BLIND sin semáforo · Timeout: rojo forzado a verde "
                      "si no hay presión en N ticks proporcionales al umbral",
                 showarrow=False,font=dict(size=10,color="#475569")),
            dict(x=0.5,y=-0.20,xref="paper",yref="paper",
                 text="Hover sobre nodo → detalle de entidades · "
                      "→N ←M = vehículos por sentido · "
                      "Tamaño ∝ presión",
                 showarrow=False,font=dict(size=10,color="#475569")),
        ],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  FOLIUM — mapa estático con estado final
# ─────────────────────────────────────────────────────────────────────────────

def build_folium_map(graph: TrafficGraph, final_snap: dict) -> folium.Map:
    m = folium.Map(
        location=[20.666, -103.39],
        zoom_start=14,
        tiles="CartoDB dark_matter",
    )

    # Aristas
    from core.road import RoadCategory
    for from_id, to_id, data in graph.graph.edges(data=True):
        n_a = graph.intersections[from_id]
        n_b = graph.intersections[to_id]
        seg = data["segment"]
        color = {
            RoadCategory.MAIN_AVENUE:      "#1d4ed8",
            RoadCategory.SECONDARY_AVENUE: "#7c3aed",
            RoadCategory.STREET:           "#334155",
        }.get(seg.category,"#334155")
        weight = {
            RoadCategory.MAIN_AVENUE:      5,
            RoadCategory.SECONDARY_AVENUE: 3,
            RoadCategory.STREET:           1.5,
        }.get(seg.category,1.5)
        folium.PolyLine(
            [(n_a.latitude,n_a.longitude),(n_b.latitude,n_b.longitude)],
            color=color, weight=weight, opacity=0.7,
            tooltip=f"{seg.category.name} w={seg.base_weight:.0f}",
        ).add_to(m)

    # Nodos
    phase_folium = {"green":"#22c55e","yellow":"#eab308","red":"#ef4444"}
    type_icon    = {
        IntersectionType.MASTER:"star",
        IntersectionType.NORMAL:"circle",
        IntersectionType.BLIND:"times",
    }

    for node_id, nd in final_snap["nodes"].items():
        color = phase_folium[nd["phase"]]
        icon  = type_icon[nd["itype"]]
        c     = nd["counts"]

        popup_html = f"""
        <div style='font-family:monospace;min-width:200px'>
        <b>{node_id}</b> — {nd['name']}<br>
        <span style='color:{color}'>● {nd['phase'].upper()}</span>
        | {nd['itype'].value.upper()}<br>
        Presión: {nd['pressure']:.1f} / {nd['threshold']:.0f}<br>
        {'🚦 Con semáforo' if nd['has_light'] else '⛔ Sin semáforo'}<br>
        <hr>
        🚗{c.get('CAR',0)} 🏍{c.get('MOTORCYCLE',0)} 🚌{c.get('BUS',0)}<br>
        🚛{c.get('TRUCK',0)} 🚲{c.get('BICYCLE',0)} 🚑{c.get('EMERGENCY',0)}<br>
        🚶{c.get('PEDESTRIAN',0)} ♿{c.get('WHEELCHAIR',0)}
        </div>
        """

        folium.Marker(
            location=[nd["lat"], nd["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{node_id} | {nd['phase'].upper()} | p={nd['pressure']:.0f}",
            icon=folium.Icon(color="green" if nd["phase"]=="green"
                             else "orange" if nd["phase"]=="yellow"
                             else "red",
                             icon=icon, prefix="fa"),
        ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

#  tango_vis — mapa interactivo + animación fusionados
# ─────────────────────────────────────────────────────────────────────────────



def build_vis(graph: TrafficGraph, all_histories: list[tuple]) -> str:
    """
    HTML único que combina mapa Leaflet interactivo + animación de frames.

    - Mapa Leaflet sobre CartoDB dark, centrado en el centroide del grafo.
    - Nodos coloreados por fase (verde/amarillo/rojo) actualizados cada tick.
    - Flechas de flujo sobre aristas con grosor proporcional al volumen.
    - Click en nodo/arista → panel de información detallado.
    - Controles play/pause/step/slider para navegar frames.
    - Botones de experimento generados desde sim_params.json["experiment_presets"].
      Cada preset corresponde a un escenario ya simulado — el botón salta al
      primer frame de ese escenario en el slider.
    """
    import json as _json

    center_lat, center_lon = compute_center(graph)

    # ── Serializar snapshots ──────────────────────────────────────────────────
    all_snaps_js = []
    scenario_index: dict[str, int] = {}   # label → primer índice de frame

    for sc_label, history in all_histories:
        scenario_index[sc_label] = len(all_snaps_js)
        for snap in history:
            nodes_js = {}
            for nid, nd in snap["nodes"].items():
                nodes_js[nid] = {
                    "phase":         nd["phase"],
                    "phase_ns":      nd["phase_ns"],
                    "phase_ew":      nd["phase_ew"],
                    "active_axis":   nd["active_axis"],
                    "signals":       nd.get("signals", {}),
                    "pressure":      round(nd["pressure"], 1),
                    "pressure_own":  round(nd.get("pressure_own", nd["pressure"]), 1),
                    "wave_offset_s": round(nd.get("wave_offset_s", 0.0), 1),
                    "threshold":     nd["threshold"],
                    "itype":         nd["itype"].value,
                    "geo_label":     nd.get("geo_label", "+"),
                    "has_light":     nd["has_light"],
                    "ticks_red":     nd["ticks_red"],
                    "timeout":       nd["timeout"],
                    "name":          nd["name"],
                    "lat":           nd["lat"],
                    "lon":           nd["lon"],
                    "counts":    nd["counts"],
                    "cluster_id": nd.get("cluster_id"),
                }
            flows_js = [
                {"from": fl["from"], "to": fl["to"],
                 "fwd": fl["fwd"], "bwd": fl["bwd"]}
                for fl in snap["flows"]
            ]
            all_snaps_js.append({
                "scenario": sc_label,
                "tick":     snap["tick"],
                "total":    snap["total"],
                "greens":   snap["greens"],
                "nodes":    nodes_js,
                "flows":    flows_js,
            })

    # ── Serializar aristas estáticas ──────────────────────────────────────────
    from core.road import RoadCategory
    edges_js = []
    drawn = set()
    for from_id, to_id, data in graph.graph.edges(data=True):
        pair = tuple(sorted([from_id, to_id]))
        if pair in drawn: continue
        drawn.add(pair)
        seg = data["segment"]
        n_a = graph.intersections[from_id]
        n_b = graph.intersections[to_id]
        edges_js.append({
            "from": from_id, "to": to_id,
            "lat_a": n_a.latitude,  "lon_a": n_a.longitude,
            "lat_b": n_b.latitude,  "lon_b": n_b.longitude,
            "category": seg.category.name,
            "weight":   seg.base_weight,
            "length_m": seg.length_m,
            "speed_kmh": seg.speed_limit_kmh,
            "name": getattr(seg, "name", ""),
        })

    # ── Serializar nodos estáticos ────────────────────────────────────────────
    nodes_static_js = {}
    for nid, inter in graph.intersections.items():
        nodes_static_js[nid] = {
            "lat": inter.latitude, "lon": inter.longitude,
            "name": inter.name,
            "itype": inter.intersection_type.value,
            "geometry": inter.geometry.value,
            "geo_label": inter.geometry_label,
            "has_light": inter.has_traffic_light,
            "threshold": inter.pressure_threshold,
        }

    # ── Botones de experimento desde sim_params.json ──────────────────────────
    presets = {k: v for k, v in PARAMS.get("experiment_presets", {}).items()
               if not k.startswith("_")}
    # Generar JS del mapa label→frame_index
    scenario_index_js = {
        label: idx for label, idx in scenario_index.items()
    }

    snaps_json      = _json.dumps(all_snaps_js)
    edges_json      = _json.dumps(edges_js)
    nodes_s_json    = _json.dumps(nodes_static_js)
    sc_index_json   = _json.dumps(scenario_index_js)

    # Generar HTML de botones de preset
    preset_buttons_html = ""
    for key, preset in presets.items():
        label = preset.get("label", key)
        desc  = preset.get("description", "")
        preset_buttons_html += (
            f'<button class="btn btn-exp" '
            f'title="{desc}" '
            f'onclick="jumpToScenario({_json.dumps(label)})">'
            f'{label}</button>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tanGo — Visualizacion Interactiva</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{
  --bg:#0f1117;--surface:#1a1d2e;--border:#2a2d3e;
  --text:#e2e8f0;--muted:#64748b;
  --green:#22c55e;--yellow:#eab308;--red:#ef4444;
  --teal:#14b8a6;--blue:#3b82f6;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);
      font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;
      height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{padding:8px 16px;background:var(--surface);
        border-bottom:1px solid var(--border);
        display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:15px;font-weight:600;letter-spacing:.04em}}
header h1 span{{color:var(--teal)}}
.badges{{display:flex;gap:8px;align-items:center}}
.badge{{font-size:10px;padding:2px 8px;border-radius:999px;
        background:var(--border);color:var(--muted)}}
.badge.run{{background:#166534;color:var(--green)}}
.layout{{display:flex;flex:1;overflow:hidden}}
#map{{flex:1;padding-bottom:48px}}
aside{{width:296px;background:var(--surface);
       border-left:1px solid var(--border);
       display:flex;flex-direction:column;overflow:hidden}}
.sec{{padding:10px 12px;border-bottom:1px solid var(--border)}}
.sec h3{{font-size:10px;font-weight:600;text-transform:uppercase;
          letter-spacing:.08em;color:var(--muted);margin-bottom:8px}}
.row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
label{{font-size:11px;color:var(--muted);min-width:68px}}
input[type=range]{{flex:1;accent-color:var(--teal)}}
.val{{font-size:11px;min-width:38px;text-align:right}}
.btn{{padding:5px 10px;border:none;border-radius:6px;
      font-size:11px;font-weight:500;cursor:pointer;transition:opacity .15s}}
.btn:hover{{opacity:.82}}
.btn-p{{background:var(--teal);color:#000}}
.btn-s{{background:var(--border);color:var(--text)}}
.btn-exp{{background:#1e3a5f;color:#93c5fd;margin:2px;font-size:10px;
          padding:4px 8px;border-radius:5px;border:1px solid #1d4ed8}}
.btn-exp:hover{{background:#1d4ed8;color:#fff}}
.btn-row{{display:flex;gap:5px;flex-wrap:wrap}}
.stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.stat{{background:var(--bg);border-radius:6px;padding:7px;
       border:1px solid var(--border)}}
.stat-val{{font-size:17px;font-weight:700;color:var(--teal)}}
.stat-lbl{{font-size:9px;color:var(--muted)}}
#node-info{{flex:1;overflow-y:auto;padding:10px}}
.placeholder{{color:var(--muted);font-size:11px;text-align:center;margin-top:24px}}
.ic{{background:var(--bg);border:1px solid var(--border);
     border-radius:7px;padding:10px;margin-bottom:7px}}
.ic h4{{font-size:12px;font-weight:600;margin-bottom:5px}}
.ir{{display:flex;justify-content:space-between;
     font-size:11px;color:var(--muted);margin:2px 0}}
.ir span{{color:var(--text)}}
.pill{{display:inline-block;padding:2px 7px;border-radius:999px;
       font-size:10px;font-weight:700}}
.pg{{background:#166534;color:var(--green)}}
.py{{background:#713f12;color:var(--yellow)}}
.pr{{background:#7f1d1d;color:var(--red)}}
.pbar-bg{{background:var(--border);border-radius:3px;height:5px;margin:5px 0}}
.pbar{{height:100%;border-radius:3px;transition:width .3s}}
#log{{max-height:110px;overflow-y:auto;padding:7px 10px;
      font-size:10px;font-family:monospace;color:var(--muted);
      border-top:1px solid var(--border)}}
.lok{{color:var(--green)}}.lwarn{{color:var(--yellow)}}.lerr{{color:var(--red)}}
/* toolbar bottom */
#toolbar{{position:fixed;bottom:0;left:0;right:296px;
          background:var(--surface);border-top:1px solid var(--border);
          padding:7px 14px;display:flex;gap:6px;align-items:center;
          flex-wrap:wrap;z-index:9999}}
#toolbar .lbl{{font-size:10px;color:var(--muted);font-weight:600;
               text-transform:uppercase;letter-spacing:.06em}}
.leaflet-tile{{filter:brightness(.7) saturate(.6)}}
.leaflet-container{{background:var(--bg)}}
</style>
</head>
<body>
<header>
  <h1>tan<span>Go</span> &mdash; visualizacion interactiva ZMG</h1>
  <div class="badges">
    <span class="badge" id="b-tick">tick #0</span>
    <span class="badge" id="b-sc">—</span>
    <span class="badge" id="b-status">detenido</span>
  </div>
</header>
<div class="layout">
  <div id="map"></div>
  <aside>
    <!-- Controles -->
    <div class="sec">
      <h3>Reproduccion</h3>
      <div class="btn-row" style="margin-bottom:8px">
        <button class="btn btn-p" id="btn-play">&#9654; Iniciar</button>
        <button class="btn btn-s" id="btn-step">&#9197; Paso</button>
        <button class="btn btn-s" id="btn-reset">&#8635; Reset</button>
      </div>
      <div class="row">
        <label>Velocidad</label>
        <input type="range" id="speed" min="200" max="3000" value="700" step="100">
        <span class="val" id="v-speed">0.7s</span>
      </div>
      <div class="row">
        <label>Frame</label>
        <input type="range" id="frame-sl" min="0" max="0" value="0">
        <span class="val" id="v-frame">0/{len(all_snaps_js)-1}</span>
      </div>
    </div>
    <!-- Stats -->
    <div class="sec">
      <h3>Estadisticas</h3>
      <div class="stat-grid">
        <div class="stat"><div class="stat-val" id="s-tick">0</div><div class="stat-lbl">Tick</div></div>
        <div class="stat"><div class="stat-val" id="s-total">0</div><div class="stat-lbl">Entidades</div></div>
        <div class="stat"><div class="stat-val" id="s-green" style="color:var(--green)">0</div><div class="stat-lbl">Verde</div></div>
        <div class="stat"><div class="stat-val" id="s-yellow" style="color:var(--yellow)">0</div><div class="stat-lbl">Amarillo</div></div>
        <div class="stat"><div class="stat-val" id="s-red" style="color:var(--red)">0</div><div class="stat-lbl">Rojo</div></div>
        <div class="stat"><div class="stat-val" id="s-blink" style="color:#f59e0b">0</div><div class="stat-lbl">Blink</div></div>
        <div class="stat"><div class="stat-val" id="s-blind" style="color:var(--muted)">0</div><div class="stat-lbl">Sin semaf.</div></div>
        <div class="stat"><div class="stat-val" id="s-nodes">{len(nodes_static_js)}</div><div class="stat-lbl">Nodos</div></div>
      </div>
    </div>
    <!-- Info panel -->
    <div id="node-info">
      <div class="placeholder">Clic en nodo o arista<br>para ver informacion</div>
    </div>
    <!-- Log -->
    <div id="log"></div>
  </aside>
</div>

<!-- Toolbar de experimentos/presets -->
<div id="toolbar">
  <span class="lbl">Escenarios:</span>
  {preset_buttons_html}
  <span style="color:var(--border);margin:0 4px">|</span>
  <span class="lbl">Info:</span>
  <button class="btn btn-s" style="font-size:10px" onclick="toggleLifetime()">
    Lifetime: <span id="lt-lbl">OFF</span>
  </button>
  <button class="btn btn-s" style="font-size:10px" onclick="log('Todos los semaforos tienen timeout proporcional al umbral: MASTER=6 ticks, NORMAL=8 ticks','lok')">
    ? Timeout
  </button>
</div>

<script>
const ALL_SNAPS    = {snaps_json};
const EDGES        = {edges_json};
const NODES_STATIC = {nodes_s_json};
const SC_INDEX     = {sc_index_json};
const N_SNAPS      = ALL_SNAPS.length;

// ── Mapa Leaflet ──────────────────────────────────────────────────────────
const map = L.map('map').setView([{center_lat},{center_lon}], 14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'CartoDB',maxZoom:19
}}).addTo(map);

const PHASE_C = {{green:'#22c55e',yellow:'#eab308',red:'#ef4444',blink:'#f59e0b'}};
    let _blinkOn = true;
    setInterval(()=>{{ _blinkOn=!_blinkOn; }}, 600);
const ITYPE_R = {{master:'#f59e0b',normal:'#3b82f6',blind:'#64748b'}};
const CAT_C   = {{MAIN_AVENUE:'#1d4ed8',SECONDARY_AVENUE:'#6d28d9',
                  STREET:'#1e293b',HIGHWAY:'#0f172a',ALLEY:'#111827'}};
const CAT_W   = {{MAIN_AVENUE:5,SECONDARY_AVENUE:3,STREET:1.5,HIGHWAY:6,ALLEY:1}};

// Aristas estáticas
EDGES.forEach(e=>{{
  const ln = L.polyline([[e.lat_a,e.lon_a],[e.lat_b,e.lon_b]],{{
    color:CAT_C[e.category]||'#1e293b',
    weight:CAT_W[e.category]||1.5,opacity:.75
  }}).addTo(map);
  ln.on('click',()=>showEdgeInfo(e,null));
}});

// Nodos
const NM = {{}};
Object.entries(NODES_STATIC).forEach(([nid,nd])=>{{
  const r = nd.itype==='master'?13:nd.itype==='normal'?10:7;
  const m = L.circleMarker([nd.lat,nd.lon],{{
    radius:r,color:ITYPE_R[nd.itype]||'#64748b',
    fillColor:'#ef4444',fillOpacity:.88,weight:2.5
  }}).addTo(map);
  m.bindTooltip(`<b>${{nid}}</b> ${{nd.geo_label}}<br>${{nd.name}}`,
    {{permanent:false,direction:'top'}});
  m.on('click',()=>showNodeInfo(nid,ALL_SNAPS[frameIdx]));
  NM[nid]=m;
}});

// Capas de flujo (dinámicas)
const FL = {{}};  // key → [polyline, label_marker]

function clearFlows(){{
  Object.values(FL).forEach(([ln,lm])=>{{
    map.removeLayer(ln);
    if(lm) map.removeLayer(lm);
  }});
  Object.keys(FL).forEach(k=>delete FL[k]);
}}

function applySnap(snap){{
  // Nodos
  Object.entries(snap.nodes).forEach(([nid,nd])=>{{
    const m=NM[nid]; if(!m) return;
    const st=NODES_STATIC[nid];
    const baseR = st.itype==='master'?13:st.itype==='normal'?10:7;
    m.setStyle({{
      fillColor: nd.phase==='blink' ? (_blinkOn?'#f59e0b':'#1e293b') : (PHASE_C[nd.phase]||'#ef4444'),
      radius: Math.min(22,baseR+nd.pressure*.06),
    }});
    const ns=nd.has_light?`NS:${{nd.phase_ns.toUpperCase()}} EW:${{nd.phase_ew.toUpperCase()}}`:'sin semaforo';
    m.setTooltipContent(`<b>${{nid}}</b> ${{st.geo_label}}<br>${{ns}}<br>P=${{nd.pressure}}/${{nd.threshold}}`);
    m.off('click');
    m.on('click',()=>showNodeInfo(nid,snap));
  }});

  // Flujos
  clearFlows();
  // Construir mapa de flujo para lookup rápido
  const flowLookup={{}};
  snap.flows.forEach(fl=>{{
    const key=fl.from+'-'+fl.to;
    flowLookup[key]=fl;
  }});

  // Dibujar TODAS las aristas — con flujo o vacías (gris con ceros)
  EDGES.forEach(e=>{{
    const na=NODES_STATIC[e.from],nb=NODES_STATIC[e.to];
    if(!na||!nb) return;

    // Buscar flujo en ambas direcciones
    const fl  = flowLookup[e.from+'-'+e.to]
             || flowLookup[e.to+'-'+e.from]
             || {{from:e.from,to:e.to,fwd:0,bwd:0}};
    const total = fl.fwd+fl.bwd;

    const fc = total===0 ? '#334155'               // gris — sin vehículos
             : total>=15 ? '#ef4444'               // rojo — alto
             : total>=8  ? '#f59e0b'               // ámbar — medio
                         : '#22c55e';              // verde — bajo
    const w  = total===0 ? 1.2 : Math.min(7,1.2+total*.28);
    const op = total===0 ? 0.35 : 0.65;

    const coords = fl.fwd>=fl.bwd
      ? [[na.lat,na.lon],[nb.lat,nb.lon]]
      : [[nb.lat,nb.lon],[na.lat,na.lon]];

    const ln = L.polyline(coords,{{color:fc,weight:w,opacity:op}}).addTo(map);
    ln.on('click',()=>showEdgeInfo(e, total>0?fl:null));

    // Label en el centro — siempre visible ("+0 -0" para vacías)
    const mx=(na.lat+nb.lat)/2, my=(na.lon+nb.lon)/2;
    const labelColor = total===0 ? '#475569' : fc;
    const lm = L.marker([mx,my],{{
      icon:L.divIcon({{
        html:`<div style="color:${{labelColor}};font-size:8px;font-weight:700;
                          text-shadow:0 0 3px #000;white-space:nowrap;
                          opacity:${{total===0?0.5:0.9}}">
              +${{fl.fwd}} -${{fl.bwd}}</div>`,
        className:'',iconAnchor:[16,5]
      }}),interactive:false
    }}).addTo(map);
    FL[`${{e.from}}-${{e.to}}`]=[ln,lm];
  }});

  // Stats
  // Contar fases
  const phaseCounts = {{green:0,yellow:0,red:0,blink:0,blind:0}};
  Object.values(snap.nodes).forEach(nd=>{{
    if (!nd.has_light) phaseCounts.blind++;
    else if (phaseCounts[nd.phase] !== undefined) phaseCounts[nd.phase]++;
  }});

  document.getElementById('s-tick').textContent   = snap.tick;
  document.getElementById('s-total').textContent  = snap.total;
  document.getElementById('s-green').textContent  = phaseCounts.green;
  document.getElementById('s-yellow').textContent = phaseCounts.yellow;
  document.getElementById('s-red').textContent    = phaseCounts.red;
  document.getElementById('s-blink').textContent  = phaseCounts.blink;
  document.getElementById('s-blind').textContent  = phaseCounts.blind;
  document.getElementById('b-tick').textContent   = `tick #${{snap.tick}}`;
  document.getElementById('b-sc').textContent     = snap.scenario.substring(0,20);

  if(selectedNode && snap.nodes[selectedNode])
    renderNodePanel(selectedNode,snap.nodes[selectedNode],NODES_STATIC[selectedNode]);
}}

// ── Paneles de información ────────────────────────────────────────────────
let selectedNode = null;

function pillCls(p){{ return {{green:'pg',yellow:'py',red:'pr'}}[p]||'pr'; }}

function renderNodePanel(nid,nd,st){{
  const pct=Math.min(100,(nd.pressure/nd.threshold*100)).toFixed(0);
  const c=nd.counts||{{}};
  const barColor=nd.pressure>=nd.threshold?'var(--red)':'var(--teal)';
  const toInfo = nd.has_light&&nd.phase==='red'
    ? `<div class="ir">Timeout en <span>${{Math.max(0,nd.timeout-nd.ticks_red)}} ticks</span></div>`:'';
  document.getElementById('node-info').innerHTML=`
  <div class="ic">
    <h4>${{nid}} &mdash; ${{st?st.geo_label:''}} ${{nd.name}}</h4>
    <div class="ir">Tipo<span>${{(nd.itype||'').toUpperCase()}}</span></div>
    <div class="ir">Geometria<span>${{st?st.geometry:''}}</span></div>
    <div class="ir">Semaforo<span>${{nd.has_light?'Si':'No'}}</span></div>
    ${{nd.cluster_id?`<div class="ir">Cluster<span style="color:#f59e0b">${{nd.cluster_id}}</span></div>`:''}}
  </div>
  ${{nd.has_light?`
  <div class="ic">
    <h4>Semaforos por direccion</h4>
    <div style="font-size:10px;color:var(--muted);margin-bottom:6px">
      Eje activo: <b>${{nd.active_axis.toUpperCase()}}</b> — exclusion mutua garantizada
    </div>
    ${{Object.entries(nd.signals||{{}}).map(([dir,ph])=>`
      <div class="ir">
        <span style="font-family:monospace;font-weight:700">${{dir}}</span>
        <span>
          <span class="pill ${{pillCls(ph)}}">${{ph.toUpperCase()}}</span>
        </span>
      </div>
    `).join('')}}
    <div style="font-size:9px;color:var(--muted);margin-top:4px;padding-top:4px;
                border-top:1px solid var(--border)">
      Eje NS (N+S) y Eje EW (E+O) son mutuamente excluyentes.<br>
      Si NS esta en verde, EW esta en rojo — siempre.
    </div>
    ${{toInfo}}
  </div>`:'<div class="ic" style="color:var(--muted);font-size:11px">Sin semaforo fisico (glorieta, incorporacion o calle interna)</div>'}}
  <div class="ic">
    <h4>Presion</h4>
    <div class="ir">Propia
      <span style="color:var(--blue)">${{nd.pressure_own||nd.pressure}} / ${{nd.threshold}}</span>
    </div>
    <div class="ir">+ Vecinal
      <span style="color:${{nd.pressure>=nd.threshold?'var(--red)':'var(--teal)'}}">${{nd.pressure}}</span>
    </div>
    <div class="ir" style="font-size:10px">Eje N-S
      <span style="color:#a78bfa">${{nd.pressure_ns||0}}</span>
    </div>
    <div class="ir" style="font-size:10px">Eje E-O
      <span style="color:#60a5fa">${{nd.pressure_ew||0}}</span>
    </div>
    ${{nd.wave_offset_s>0?`
    <div class="ir">Ola verde en
      <span style="color:#f59e0b">${{nd.wave_offset_s}}s</span>
    </div>`:''}}
    <div class="pbar-bg">
      <div class="pbar" style="width:${{Math.min(100,pct)}}%;background:${{barColor}}"></div>
    </div>
    <div style="font-size:10px;color:var(--muted)">${{pct}}% del umbral</div>
    <div style="font-size:9px;color:var(--muted);margin-top:3px">
      La presion vecinal incluye la influencia de nodos adyacentes.<br>
      Nodos MASTER propagan señal 1.3x mas fuerte.
    </div>
  </div>
  <div class="ic">
    <h4>Entidades</h4>
    <div class="ir">Autos<span>${{c.CAR||0}}</span></div>
    <div class="ir">Motos<span>${{c.MOTORCYCLE||0}}</span></div>
    <div class="ir">Buses<span>${{c.BUS||0}}</span></div>
    <div class="ir">Camiones<span>${{c.TRUCK||0}}</span></div>
    <div class="ir">Bicicletas<span>${{c.BICYCLE||0}}</span></div>
    <div class="ir">Peatones<span>${{c.PEDESTRIAN||0}}</span></div>
    <div class="ir">Sillas<span>${{c.WHEELCHAIR||0}}</span></div>
    <div class="ir" style="color:var(--red)">Emergencias<span style="color:var(--red)">${{c.EMERGENCY||0}}</span></div>
  </div>`;
}}

function showNodeInfo(nid,snap){{
  selectedNode=nid;
  const st=NODES_STATIC[nid];
  const nd=snap?snap.nodes[nid]:null;
  if(!nd){{
    document.getElementById('node-info').innerHTML=
      `<div class="ic"><h4>${{nid}}</h4>
       <div class="ir">Nombre<span>${{st.name}}</span></div>
       <div class="ir">Tipo<span>${{st.itype.toUpperCase()}}</span></div>
       <div class="ir">Geometria<span>${{st.geometry}}</span></div>
       <div class="ir">Semaforo<span>${{st.has_light?'Si':'No'}}</span></div>
       <div style="color:var(--muted);font-size:10px;margin-top:6px">Inicia la simulacion para datos en tiempo real</div></div>`;
    return;
  }}
  renderNodePanel(nid,nd,st);
}}

function showEdgeInfo(e,fl){{
  selectedNode=null;
  document.getElementById('node-info').innerHTML=`
  <div class="ic">
    <h4>Segmento vial</h4>
    <div class="ir">Nombre<span>${{e.name||'Sin nombre'}}</span></div>
    <div class="ir">Categoria<span>${{e.category}}</span></div>
    <div class="ir">Peso base<span>${{e.weight}}</span></div>
    <div class="ir">Longitud<span>${{e.length_m}} m</span></div>
    <div class="ir">Velocidad max<span>${{e.speed_kmh}} km/h</span></div>
    <div class="ir">De → A<span>${{e.from}} → ${{e.to}}</span></div>
  </div>
  ${{fl?`<div class="ic"><h4>Flujo actual</h4>
    <div class="ir">Sentido +<span>${{fl.fwd}} vehiculos</span></div>
    <div class="ir">Sentido -<span>${{fl.bwd}} vehiculos</span></div>
    <div class="ir">Total<span style="color:var(--teal)">${{fl.fwd+fl.bwd}}</span></div>
  </div>`:'<div class="ic" style="color:var(--muted);font-size:10px">Inicia la simulacion para datos de flujo</div>'}}`;
}}

// ── Animación ─────────────────────────────────────────────────────────────
let frameIdx=0, running=false, timer=null;
const slider=document.getElementById('frame-sl');
slider.max=N_SNAPS-1;

function goToFrame(idx){{
  if(idx<0||idx>=N_SNAPS) return;
  frameIdx=idx; slider.value=idx;
  document.getElementById('v-frame').textContent=`${{idx}}/${{N_SNAPS-1}}`;
  applySnap(ALL_SNAPS[idx]);
}}

function scheduleNext(){{
  if(!running) return;
  goToFrame((frameIdx+1)%N_SNAPS);
  timer=setTimeout(scheduleNext,parseInt(document.getElementById('speed').value));
}}

document.getElementById('btn-play').addEventListener('click',()=>{{
  running=!running;
  const btn=document.getElementById('btn-play');
  const bs=document.getElementById('b-status');
  if(running){{
    btn.innerHTML='&#9646;&#9646; Pausar';
    bs.textContent='corriendo'; bs.className='badge run';
    scheduleNext();
  }}else{{
    btn.innerHTML='&#9654; Iniciar';
    bs.textContent='detenido'; bs.className='badge';
    clearTimeout(timer);
  }}
}});

document.getElementById('btn-step').addEventListener('click',()=>{{
  clearTimeout(timer); running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  goToFrame((frameIdx+1)%N_SNAPS);
}});

document.getElementById('btn-reset').addEventListener('click',()=>{{
  clearTimeout(timer); running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  selectedNode=null;
  document.getElementById('node-info').innerHTML=
    '<div class="placeholder">Clic en nodo o arista<br>para ver informacion</div>';
  goToFrame(0);
}});

document.getElementById('speed').addEventListener('input',function(){{
  document.getElementById('v-speed').textContent=(this.value/1000).toFixed(1)+'s';
}});

slider.addEventListener('input',function(){{
  clearTimeout(timer); running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  goToFrame(parseInt(this.value));
}});

// ── Botones de escenario/preset ───────────────────────────────────────────
function jumpToScenario(label){{
  const idx = SC_INDEX[label];
  if(idx===undefined){{ log('Escenario no encontrado: '+label,'lerr'); return; }}
  clearTimeout(timer); running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  goToFrame(idx);
  log('Saltando a: '+label+' (frame '+idx+')','lok');
}}

// ── Lifetime toggle (informativo) ─────────────────────────────────────────
let lifetimeOn=false;
function toggleLifetime(){{
  lifetimeOn=!lifetimeOn;
  document.getElementById('lt-lbl').textContent=lifetimeOn?'ON':'OFF';
  if(lifetimeOn)
    log('Lifetime ON: en esta sim cada tick genera entidades nuevas (sin persistencia entre ticks). Para lifetime real se necesita el servidor Python.','lwarn');
  else
    log('Lifetime OFF','lok');
}}

// ── Log ───────────────────────────────────────────────────────────────────
function log(msg,cls=''){{
  const el=document.getElementById('log');
  const d=document.createElement('div');
  d.className=cls;
  d.textContent=`[${{new Date().toLocaleTimeString('es',{{hour12:false}})}}] ${{msg}}`;
  el.prepend(d);
  while(el.children.length>25) el.removeChild(el.lastChild);
}}

// Init
goToFrame(0);
log('tanGo listo — '+Object.keys(NODES_STATIC).length+' nodos · '+N_SNAPS+' frames','lok');
log('Usa los botones de escenario abajo para saltar entre experimentos');
</script>
</body>
</html>"""
    return html

if __name__ == "__main__":
    import time as _time
    _t0       = _time.perf_counter()
    _ts_start = datetime.now()
    print("tanGo — simulación ZMG Guadalajara")
    print(f"  Inicio: {_ts_start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Cargar grafo: JSON real si existe, hardcodeado como fallback
    if CITY_JSON.exists():
        print(f"  Usando grafo desde JSON: {CITY_JSON.name}")
        graph = json_to_traffic_graph(CITY_JSON)
        print(f"  {graph.graph.number_of_nodes()} intersecciones reales · "
              f"{graph.graph.number_of_edges()} aristas")
    else:
        print(f"  JSON no encontrado — usando grafo de ejemplo.")
        print(f"  Para datos reales ejecuta:")
        print(f"    python graph/city_loader.py --city zmg_centro\n")
        graph = build_zmg_graph()

    # Escenarios: desde JSON si existen, sino los hardcodeados
    json_scenarios = [s for s in PARAMS.get("scenarios", [])
                      if not str(s.get("label","")).startswith("_")]
    active_scenarios = json_scenarios if json_scenarios else SCENARIOS

    # Presets del JSON como escenarios adicionales
    presets = {k: v for k, v in PARAMS.get("experiment_presets", {}).items()
               if not k.startswith("_")}

    print(f"  {len(active_scenarios)} escenarios base + "
          f"{len(presets)} presets × {N_TICKS} ticks\n")

    all_histories = []
    final_snap    = None

    # Escenarios base
    for sc in active_scenarios:
        print(f"  Simulando: {sc['label']}...")
        history = simulate(sc, graph, N_TICKS,
                           spawn_params=PARAMS.get("spawn"))
        all_histories.append((sc["label"], history))
        final_snap = history[-1]

    # Presets como escenarios extra
    base_sc = dict(active_scenarios[0]) if active_scenarios else dict(SCENARIOS[0])
    for preset_key, preset in presets.items():
        label = preset.get("label", preset_key)
        print(f"  Simulando preset: {label}...")
        sc_merged = dict(base_sc)
        sc_merged["label"] = label
        if preset.get("force_rain"):    sc_merged["is_raining"]     = True
        if preset.get("force_weekend"): sc_merged["timestamp"]      = "2024-03-09T15:00:00"
        history = simulate(sc_merged, graph, N_TICKS, spawn_params=preset)
        all_histories.append((label, history))
        final_snap = history[-1]

    print("\n  Generando animación Plotly...")
    fig = build_plotly(graph, all_histories)
    fig.write_html(str(OUTPUT_PLOTLY), include_plotlyjs="cdn",
                   full_html=True,
                   config={"responsive":True,"displayModeBar":True})
    print(f"  ✓ {OUTPUT_PLOTLY}")

    print("  Generando mapa Folium (estado final)...")
    fmap = build_folium_map(graph, final_snap)
    fmap.save(str(OUTPUT_FOLIUM))
    print(f"  ✓ {OUTPUT_FOLIUM}")

    print("  Generando visualizacion interactiva fusionada...")
    vis_html = build_vis(graph, all_histories)
    OUTPUT_VIS.write_text(vis_html, encoding="utf-8")
    print(f"  ✓ {OUTPUT_VIS}")

    graph.close()

    _t1       = _time.perf_counter()
    _ts_end   = datetime.now()
    _elapsed  = _t1 - _t0
    _mins     = int(_elapsed // 60)
    _secs     = _elapsed % 60

    print("\n" + "─" * 52)
    print(f"  Inicio  : {_ts_start.strftime('%H:%M:%S')}")
    print(f"  Fin     : {_ts_end.strftime('%H:%M:%S')}")
    print(f"  Duracion: {_mins}m {_secs:.1f}s")
    print(f"  Frames  : {sum(len(h) for _,h in all_histories)}")
    print(f"  Nodos   : {graph.graph.number_of_nodes()}")
    print("─" * 52)
    print(f"\n✓ Listo.")
    print(f"  Recomendado: abre tango_vis.html (mapa interactivo + animacion)")
    print(f"  Alternativo: tango_sim.html (Plotly) · tango_map.html (Folium estatico)")


# ─────────────────────────────────────────────────────────────────────────────