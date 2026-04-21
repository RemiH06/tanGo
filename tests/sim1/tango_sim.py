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
import sys, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster

from core.context       import TrafficContext
from core.weight_engine import WeightEngine
from core.road          import (Intersection, IntersectionType,
                                IntersectionGeometry, RoadSegment,
                                RoadCategory, Phase, Turn, TrafficAxis)
from core.entities      import Vehicle, Pedestrian, VehicleType, Direction
from graph.simulator    import TrafficGraph
from graph.city_loader  import json_to_traffic_graph, load_graph_from_json

OUTPUT_PLOTLY = Path(__file__).parent / "tango_sim.html"
OUTPUT_FOLIUM = Path(__file__).parent / "tango_map.html"
OUTPUT_VIS    = Path(__file__).parent / "tango_vis.html"
N_TICKS = 40


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

PHASE_COLOR = {"green":"#22c55e", "yellow":"#eab308", "red":"#ef4444"}
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
                   ctx: TrafficContext) -> list:
    """Genera entidades realistas según tipo de intersección y contexto."""
    import uuid

    # Volumen base por tipo e intersección
    if itype == IntersectionType.MASTER:
        nv = random.randint(8,20) if ctx.is_rush_hour else \
             random.randint(1,4)  if ctx.is_late_night else random.randint(5,14)
        np_ = random.randint(5,15) if ctx.is_rush_hour else \
              random.randint(0,2)  if ctx.is_late_night else random.randint(2,8)
    elif itype == IntersectionType.NORMAL:
        nv = random.randint(5,12) if ctx.is_rush_hour else \
             random.randint(0,3)  if ctx.is_late_night else random.randint(3,8)
        np_ = random.randint(3,10) if ctx.is_rush_hour else \
              random.randint(0,1)  if ctx.is_late_night else random.randint(1,6)
    else:  # BLIND — tráfico de colonia
        nv = random.randint(2,7) if ctx.is_rush_hour else \
             random.randint(0,2) if ctx.is_late_night else random.randint(1,5)
        np_ = random.randint(0,3)

    # Reducir ciclistas con lluvia
    pool = V_TYPES_POOL if not ctx.is_raining else \
           [v for v in V_TYPES_POOL if v != VehicleType.BICYCLE] + \
           [VehicleType.CAR] * 5

    entities = []
    for _ in range(nv):
        vtype = random.choice(pool)
        entities.append(Vehicle(str(uuid.uuid4()), vtype,
                                random.choice(list(Direction))))
    for _ in range(np_):
        wc = random.random() < (0.08 if not ctx.is_late_night else 0.02)
        entities.append(Pedestrian(str(uuid.uuid4()), is_wheelchair=wc))

    return entities


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def simulate(scenario: dict, graph: TrafficGraph, n_ticks: int) -> list[dict]:
    engine = WeightEngine()
    ctx    = TrafficContext.build(**{k:v for k,v in scenario.items() if k!="label"})
    history = []

    # Reset fases
    for inter in graph.intersections.values():
        inter.current_phase    = Phase.RED
        inter._ticks_in_phase  = 0
        inter._timeout_triggered = False
        inter.pressure         = 0.0

    for _ in range(n_ticks):
        frame = {"tick":0, "nodes":{}, "flows":[], "total":0, "greens":0}

        # Generar entidades por nodo
        all_entities = {}
        for node_id, inter in graph.intersections.items():
            ents = spawn_for_node(node_id, inter.intersection_type, ctx)
            all_entities[node_id] = ents

        # Calcular flujos bidireccionales
        flow_map: dict = defaultdict(lambda: {"fwd":0,"bwd":0})
        for from_id, to_id, data in graph.graph.edges(data=True):
            n_veh = sum(1 for e in all_entities.get(from_id,[])
                        if isinstance(e, Vehicle))
            key = tuple(sorted([from_id,to_id]))
            if from_id <= to_id:
                flow_map[key]["fwd"] += n_veh
            else:
                flow_map[key]["bwd"] += n_veh

        frame["flows"] = [
            {"from":k[0],"to":k[1],"fwd":v["fwd"],"bwd":v["bwd"]}
            for k,v in flow_map.items()
        ]

        # Ajustar fases con WeightEngine real + timeout
        for node_id, inter in graph.intersections.items():
            ents = all_entities[node_id]
            pressure = engine.aggregate_pressure(ents, inter, ctx)
            inter.pressure = pressure
            inter.adjust_phase(engine, ctx, ents)

            counts = defaultdict(int)
            for e in ents:
                if isinstance(e, Vehicle):
                    counts[e.vehicle_type.name] += 1
                elif isinstance(e, Pedestrian):
                    counts["PEDESTRIAN"] += 1
                    if e.is_wheelchair: counts["WHEELCHAIR"] += 1

            frame["nodes"][node_id] = {
                "phase":      inter.current_phase.value,
                "phase_ns":   inter.phase_ns.value,
                "phase_ew":   inter.phase_ew.value,
                "active_axis": getattr(inter._active_axis, "value", "ns"),
                "pressure":   inter.pressure,
                "itype":      inter.intersection_type,
                "geometry":   inter.geometry,
                "geo_label":  inter.geometry_label,
                "has_light":  inter.has_traffic_light,
                "threshold":  inter.pressure_threshold,
                "timeout":    inter.red_timeout_ticks,
                "ticks_red":  inter._ticks_in_phase,
                "name":       inter.name,
                "lat":        inter.latitude,
                "lon":        inter.longitude,
                "counts":     dict(counts),
            }
            frame["total"] += len(ents)
            if inter.current_phase == Phase.GREEN:
                frame["greens"] += 1

        # Usar coordenadas reales para Plotly (lon→x, lat→y)
        tick_val = list(graph.intersections.values())[0]._ticks_in_phase
        frame["tick"] = _ + 1
        history.append(frame)

    return history


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
    Genera un HTML único que combina:
      - Mapa Folium interactivo (movible, zoom, click para info)
      - Panel lateral con estado actual de cada nodo
      - Controles de animación (play/pause/step/velocidad)
      - Flechas de flujo sobre el mapa actualizadas por tick

    No usa iframes ni archivos externos — todo embebido en un solo HTML.

    Returns
    -------
    String con el HTML completo.
    """
    from core.road import RoadCategory

    center_lat, center_lon = compute_center(graph)

    # Serializar todos los snapshots de todos los escenarios como JSON
    # para que el JS del navegador los anime sin servidor
    all_snaps_js = []
    for sc_label, history in all_histories:
        for snap in history:
            # Serializar nodos
            nodes_js = {}
            for nid, nd in snap["nodes"].items():
                nodes_js[nid] = {
                    "phase":      nd["phase"],
                    "phase_ns":   nd["phase_ns"],
                    "phase_ew":   nd["phase_ew"],
                    "active_axis": nd["active_axis"],
                    "pressure":   round(nd["pressure"], 1),
                    "threshold":  nd["threshold"],
                    "itype":      nd["itype"].value,
                    "geo_label":  nd.get("geo_label", "+"),
                    "has_light":  nd["has_light"],
                    "ticks_red":  nd["ticks_red"],
                    "timeout":    nd["timeout"],
                    "name":       nd["name"],
                    "lat":        nd["lat"],
                    "lon":        nd["lon"],
                    "counts":     nd["counts"],
                }
            # Serializar flujos
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

    # Serializar aristas estáticas del grafo
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
            "weight": seg.base_weight,
            "length_m": seg.length_m,
            "speed_kmh": seg.speed_limit_kmh,
            "name": getattr(seg, "name", ""),
        })

    # Serializar nodos estáticos
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

    import json as _json
    snaps_json   = _json.dumps(all_snaps_js)
    edges_json   = _json.dumps(edges_js)
    nodes_s_json = _json.dumps(nodes_static_js)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tanGo — Visualización Interactiva</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d2e; --border: #2a2d3e;
    --text: #e2e8f0; --muted: #64748b;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444;
    --blue: #3b82f6; --purple: #7c3aed; --teal: #14b8a6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
          font-family: 'Segoe UI', system-ui, sans-serif;
          font-size: 13px; height: 100vh; overflow: hidden;
          display: flex; flex-direction: column; }}

  header {{ padding: 8px 16px; background: var(--surface);
             border-bottom: 1px solid var(--border);
             display: flex; align-items: center; justify-content: space-between; }}
  header h1 {{ font-size: 15px; font-weight: 600; letter-spacing:.04em; }}
  header h1 span {{ color: var(--teal); }}
  .badges {{ display: flex; gap: 8px; align-items: center; }}
  .badge {{ font-size: 10px; padding: 2px 8px; border-radius: 999px;
             background: var(--border); color: var(--muted); }}
  .badge.running {{ background:#166534; color:var(--green); }}

  .layout {{ display: flex; flex: 1; overflow: hidden; }}
  #map {{ flex: 1; }}

  aside {{ width: 300px; background: var(--surface);
            border-left: 1px solid var(--border);
            display: flex; flex-direction: column; overflow: hidden; }}

  .section {{ padding: 12px; border-bottom: 1px solid var(--border); }}
  .section h3 {{ font-size: 10px; font-weight: 600; text-transform: uppercase;
                  letter-spacing:.08em; color: var(--muted); margin-bottom: 10px; }}

  .row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  label {{ font-size: 11px; color: var(--muted); min-width: 70px; }}
  input[type=range] {{ flex: 1; accent-color: var(--teal); }}
  .val {{ font-size: 11px; min-width: 40px; text-align: right; }}

  .btn-row {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .btn {{ padding: 5px 12px; border: none; border-radius: 6px;
           font-size: 11px; font-weight: 500; cursor: pointer; }}
  .btn-primary   {{ background: var(--teal);   color: #000; }}
  .btn-secondary {{ background: var(--border); color: var(--text); }}

  .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
  .stat {{ background: var(--bg); border-radius: 6px; padding: 8px;
            border: 1px solid var(--border); }}
  .stat-val {{ font-size: 18px; font-weight: 700; color: var(--teal); }}
  .stat-lbl {{ font-size: 9px; color: var(--muted); }}

  #node-info {{ flex: 1; overflow-y: auto; padding: 12px; }}
  #node-info .placeholder {{ color: var(--muted); font-size: 11px;
                               text-align: center; margin-top: 30px; }}
  .info-card {{ background: var(--bg); border: 1px solid var(--border);
                 border-radius: 8px; padding: 12px; margin-bottom: 8px; }}
  .info-card h4 {{ font-size: 12px; font-weight: 600; margin-bottom: 6px; }}
  .info-row {{ display: flex; justify-content: space-between;
                font-size: 11px; color: var(--muted); margin: 2px 0; }}
  .info-row span {{ color: var(--text); }}
  .phase-pill {{ display:inline-block; padding:2px 8px; border-radius:999px;
                  font-size:10px; font-weight:700; }}
  .pill-green  {{ background:#166534; color:var(--green); }}
  .pill-yellow {{ background:#713f12; color:var(--yellow); }}
  .pill-red    {{ background:#7f1d1d; color:var(--red); }}

  #log {{ max-height: 120px; overflow-y: auto; padding: 8px 12px;
           font-size: 10px; font-family: monospace; color: var(--muted);
           border-top: 1px solid var(--border); }}
  .log-ok   {{ color: var(--green); }}
  .log-warn {{ color: var(--yellow); }}
  .log-err  {{ color: var(--red); }}

  /* Leaflet dark override */
  .leaflet-tile {{ filter: brightness(0.7) saturate(0.6); }}
  .leaflet-container {{ background: var(--bg); }}
</style>
</head>
<body>

<header>
  <h1>tan<span>Go</span> &mdash; visualización interactiva ZMG</h1>
  <div class="badges">
    <span class="badge" id="badge-tick">tick #0</span>
    <span class="badge" id="badge-scenario">—</span>
    <span class="badge" id="badge-status">detenido</span>
  </div>
</header>

<div class="layout">
  <div id="map"></div>

  <aside>
    <!-- Controles -->
    <div class="section">
      <h3>Reproducción</h3>
      <div class="btn-row" style="margin-bottom:10px">
        <button class="btn btn-primary"  id="btn-play">&#9654; Iniciar</button>
        <button class="btn btn-secondary" id="btn-step">&#9197; Paso</button>
        <button class="btn btn-secondary" id="btn-reset">&#8635; Reset</button>
      </div>
      <div class="row">
        <label>Velocidad</label>
        <input type="range" id="speed" min="300" max="3000" value="800" step="100">
        <span class="val" id="speed-val">0.8s</span>
      </div>
      <div class="row">
        <label>Frame</label>
        <input type="range" id="frame-slider" min="0" max="0" value="0" style="flex:1">
        <span class="val" id="frame-val">0</span>
      </div>
    </div>

    <!-- Stats -->
    <div class="section">
      <h3>Estadísticas del tick</h3>
      <div class="stat-grid">
        <div class="stat">
          <div class="stat-val" id="s-tick">0</div>
          <div class="stat-lbl">Tick</div>
        </div>
        <div class="stat">
          <div class="stat-val" id="s-total">0</div>
          <div class="stat-lbl">Entidades</div>
        </div>
        <div class="stat">
          <div class="stat-val" id="s-green">0</div>
          <div class="stat-lbl">En verde</div>
        </div>
        <div class="stat">
          <div class="stat-val" id="s-nodes">0</div>
          <div class="stat-lbl">Nodos</div>
        </div>
      </div>
    </div>

    <!-- Info de nodo seleccionado -->
    <div id="node-info">
      <div class="placeholder">Haz clic sobre una<br>intersección o arista<br>para ver su información</div>
    </div>

    <!-- Log -->
    <div id="log"></div>
  </aside>
</div>

<script>
// ═══════════════════════════════════════════════════════
//  DATOS (inyectados desde Python)
// ═══════════════════════════════════════════════════════
const ALL_SNAPS   = {snaps_json};
const EDGES       = {edges_json};
const NODES_STATIC = {nodes_s_json};

// ═══════════════════════════════════════════════════════
//  MAPA LEAFLET
// ═══════════════════════════════════════════════════════
const map = L.map('map', {{ zoomControl: true }}).setView(
  [{center_lat}, {center_lon}], 14
);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CartoDB',
  maxZoom: 19,
}}).addTo(map);

// ═══════════════════════════════════════════════════════
//  CAPAS DINÁMICAS
// ═══════════════════════════════════════════════════════
const nodeMarkers = {{}};   // nid → L.circleMarker
const edgeLines   = {{}};   // "from-to" → L.polyline (flujo)
const edgeStatic  = [];     // L.polyline estáticas

const PHASE_COLOR  = {{ green:'#22c55e', yellow:'#eab308', red:'#ef4444' }};
const ITYPE_RING   = {{ master:'#f59e0b', normal:'#3b82f6', blind:'#64748b' }};
const CAT_COLOR    = {{
  MAIN_AVENUE:'#1d4ed8', SECONDARY_AVENUE:'#6d28d9',
  STREET:'#1e293b', HIGHWAY:'#0f172a', ALLEY:'#0f172a'
}};
const CAT_WEIGHT   = {{
  MAIN_AVENUE:5, SECONDARY_AVENUE:3, STREET:1.5,
  HIGHWAY:6, ALLEY:1
}};

// Aristas estáticas
EDGES.forEach(e => {{
  const line = L.polyline(
    [[e.lat_a, e.lon_a],[e.lat_b, e.lon_b]],
    {{ color: CAT_COLOR[e.category] || '#1e293b',
       weight: CAT_WEIGHT[e.category] || 1.5,
       opacity: 0.7 }}
  ).addTo(map);
  line.on('click', () => showEdgeInfo(e));
  edgeStatic.push(line);
}});

// Nodos iniciales
Object.entries(NODES_STATIC).forEach(([nid, nd]) => {{
  const r = nd.itype === 'master' ? 12 : nd.itype === 'normal' ? 9 : 7;
  const m = L.circleMarker([nd.lat, nd.lon], {{
    radius: r,
    color:  ITYPE_RING[nd.itype] || '#64748b',
    fillColor: '#ef4444',
    fillOpacity: 0.85,
    weight: 2,
  }}).addTo(map);
  m.bindTooltip(`${{nid}} ${{nd.geo_label}}`, {{
    permanent: false, direction: 'top', className: 'leaflet-dark-tip'
  }});
  m.on('click', () => showNodeInfo(nid, null));
  nodeMarkers[nid] = m;
}});

// ═══════════════════════════════════════════════════════
//  ACTUALIZAR MAPA EN CADA TICK
// ═══════════════════════════════════════════════════════
let selectedNode = null;

function applySnap(snap) {{
  // Actualizar nodos
  Object.entries(snap.nodes).forEach(([nid, nd]) => {{
    const m = nodeMarkers[nid];
    if (!m) return;
    const st = NODES_STATIC[nid];
    const phase = nd.phase;
    const baseR = st.itype === 'master' ? 12 : st.itype === 'normal' ? 9 : 7;
    const r = Math.min(20, baseR + nd.pressure * 0.05);
    m.setStyle({{
      fillColor: PHASE_COLOR[phase] || '#ef4444',
      radius: r,
      color: ITYPE_RING[st.itype] || '#64748b',
    }});
    // Actualizar tooltip con info resumida
    const ns = nd.has_light ? `NS:${{nd.phase_ns.toUpperCase()}} EW:${{nd.phase_ew.toUpperCase()}}` : 'sin semáforo';
    m.setTooltipContent(`<b>${{nid}}</b> ${{st.geo_label}}<br>${{ns}}<br>P=${{nd.pressure}}`);
    m.on('click', () => showNodeInfo(nid, snap));
  }});

  // Actualizar flujos
  // Limpiar flujos anteriores
  Object.values(edgeLines).forEach(l => map.removeLayer(l));
  Object.keys(edgeLines).forEach(k => delete edgeLines[k]);

  snap.flows.forEach(fl => {{
    const total = fl.fwd + fl.bwd;
    if (total === 0) return;
    const na = NODES_STATIC[fl.from];
    const nb = NODES_STATIC[fl.to];
    if (!na || !nb) return;

    const flowColor = total >= 15 ? '#ef4444' : total >= 8 ? '#f59e0b' : '#22c55e';
    const w = Math.min(7, 1 + total * 0.25);

    // Dirección dominante
    const coords = fl.fwd >= fl.bwd
      ? [[na.lat, na.lon],[nb.lat, nb.lon]]
      : [[nb.lat, nb.lon],[na.lat, na.lon]];

    const line = L.polyline(coords, {{
      color: flowColor, weight: w, opacity: 0.6,
      // Flecha en el extremo
    }}).addTo(map);

    // Decorator-like: pequeño marcador en el punto medio indicando dirección
    const mx = (na.lat + nb.lat) / 2;
    const my = (na.lon + nb.lon) / 2;
    const lbl = L.marker([mx, my], {{
      icon: L.divIcon({{
        html: `<div style="color:${{flowColor}};font-size:9px;font-weight:bold;
                            white-space:nowrap;text-shadow:0 0 3px #000">
               +${{fl.fwd}} -${{fl.bwd}}</div>`,
        className: '',
        iconAnchor: [20, 6],
      }}),
      interactive: false,
    }}).addTo(map);

    const key = `${{fl.from}}-${{fl.to}}`;
    edgeLines[key] = line;
    // Guardar label para limpiar
    edgeLines[key+'_lbl'] = lbl;

    line.on('click', () => {{
      const e = EDGES.find(e => e.from === fl.from && e.to === fl.to
                              || e.from === fl.to && e.to === fl.from);
      if (e) showEdgeInfo(e, fl);
    }});
  }});

  // Si hay nodo seleccionado, actualizar su panel
  if (selectedNode && snap.nodes[selectedNode]) {{
    renderNodePanel(selectedNode, snap.nodes[selectedNode],
                    NODES_STATIC[selectedNode]);
  }}

  // Stats
  document.getElementById('s-tick').textContent    = snap.tick;
  document.getElementById('s-total').textContent   = snap.total;
  document.getElementById('s-green').textContent   = snap.greens;
  document.getElementById('s-nodes').textContent   = Object.keys(snap.nodes).length;
  document.getElementById('badge-tick').textContent = `tick #${{snap.tick}}`;
  document.getElementById('badge-scenario').textContent = snap.scenario;
}}

// ═══════════════════════════════════════════════════════
//  PANELES DE INFORMACIÓN (click)
// ═══════════════════════════════════════════════════════

function pillClass(phase) {{
  return {{ green:'pill-green', yellow:'pill-yellow', red:'pill-red' }}[phase] || 'pill-red';
}}

function renderNodePanel(nid, nd, st) {{
  const pct = Math.min(100, (nd.pressure / nd.threshold * 100)).toFixed(0);
  const timeoutLeft = nd.has_light && nd.phase === 'red'
    ? `<div class="info-row">Timeout en <span>${{Math.max(0, nd.timeout - nd.ticks_red)}} ticks</span></div>`
    : '';
  const c = nd.counts || {{}};

  document.getElementById('node-info').innerHTML = `
    <div class="info-card">
      <h4>${{nid}} &mdash; ${{st ? st.geo_label : ''}} ${{nd.name}}</h4>
      <div class="info-row">Tipo <span>${{(nd.itype||'').toUpperCase()}}</span></div>
      <div class="info-row">Geometría <span>${{st ? st.geometry : ''}}</span></div>
      <div class="info-row">Semáforo <span>${{nd.has_light ? 'Si' : 'No'}}</span></div>
    </div>
    <div class="info-card">
      <h4>Fases</h4>
      ${{nd.has_light ? `
        <div class="info-row">Eje N-S
          <span><span class="phase-pill ${{pillClass(nd.phase_ns)}}">${{nd.phase_ns.toUpperCase()}}</span></span>
        </div>
        <div class="info-row">Eje E-O
          <span><span class="phase-pill ${{pillClass(nd.phase_ew)}}">${{nd.phase_ew.toUpperCase()}}</span></span>
        </div>
        <div class="info-row">Eje activo <span>${{nd.active_axis.toUpperCase()}}</span></div>
        ${{timeoutLeft}}
      ` : '<div class="info-row" style="color:var(--muted)">Sin semáforo fisico</div>'}}
    </div>
    <div class="info-card">
      <h4>Presion</h4>
      <div class="info-row">Valor <span style="color:${{nd.pressure>=nd.threshold?'var(--red)':'var(--teal)'}}">${{nd.pressure}} / ${{nd.threshold}}</span></div>
      <div class="info-row">Porcentaje <span>${{pct}}%</span></div>
      <div style="background:var(--border);border-radius:4px;height:6px;margin:6px 0">
        <div style="background:${{nd.pressure>=nd.threshold?'var(--red)':'var(--teal)'}};
                    width:${{Math.min(100,pct)}}%;height:100%;border-radius:4px"></div>
      </div>
    </div>
    <div class="info-card">
      <h4>Entidades</h4>
      <div class="info-row">Autos <span>${{c.CAR||0}}</span></div>
      <div class="info-row">Motos <span>${{c.MOTORCYCLE||0}}</span></div>
      <div class="info-row">Buses <span>${{c.BUS||0}}</span></div>
      <div class="info-row">Camiones <span>${{c.TRUCK||0}}</span></div>
      <div class="info-row">Bicicletas <span>${{c.BICYCLE||0}}</span></div>
      <div class="info-row">Emergencias <span style="color:var(--red)">${{c.EMERGENCY||0}}</span></div>
      <div class="info-row">Peatones <span>${{c.PEDESTRIAN||0}}</span></div>
      <div class="info-row">Sillas de ruedas <span>${{c.WHEELCHAIR||0}}</span></div>
    </div>
  `;
}}

function showNodeInfo(nid, snap) {{
  selectedNode = nid;
  const st = NODES_STATIC[nid];
  const nd = snap ? snap.nodes[nid] : null;
  if (!nd) {{
    document.getElementById('node-info').innerHTML =
      `<div class="info-card"><h4>${{nid}}</h4>
       <div class="info-row">Nombre <span>${{st.name}}</span></div>
       <div class="info-row">Tipo <span>${{st.itype.toUpperCase()}}</span></div>
       <div class="info-row">Geometria <span>${{st.geometry}}</span></div>
       <div class="info-row">Semaforo <span>${{st.has_light ? 'Si' : 'No'}}</span></div>
       <div style="color:var(--muted);font-size:11px;margin-top:8px">Inicia la simulacion para ver datos en tiempo real</div>
       </div>`;
    return;
  }}
  renderNodePanel(nid, nd, st);
}}

function showEdgeInfo(e, fl) {{
  selectedNode = null;
  document.getElementById('node-info').innerHTML = `
    <div class="info-card">
      <h4>Segmento vial</h4>
      <div class="info-row">Nombre <span>${{e.name || 'Sin nombre'}}</span></div>
      <div class="info-row">Categoria <span>${{e.category}}</span></div>
      <div class="info-row">Peso base <span>${{e.weight}}</span></div>
      <div class="info-row">Longitud <span>${{e.length_m}} m</span></div>
      <div class="info-row">Velocidad max <span>${{e.speed_kmh}} km/h</span></div>
      <div class="info-row">De <span>${{e.from}}</span></div>
      <div class="info-row">A <span>${{e.to}}</span></div>
    </div>
    ${{fl ? `
    <div class="info-card">
      <h4>Flujo actual</h4>
      <div class="info-row">Sentido + <span>${{fl.fwd}} vehiculos</span></div>
      <div class="info-row">Sentido - <span>${{fl.bwd}} vehiculos</span></div>
      <div class="info-row">Total <span>${{fl.fwd + fl.bwd}} vehiculos</span></div>
    </div>` : ''}}
  `;
}}

// ═══════════════════════════════════════════════════════
//  ANIMACIÓN
// ═══════════════════════════════════════════════════════

let frameIdx = 0;
let running  = false;
let timer    = null;

const slider = document.getElementById('frame-slider');
slider.max   = ALL_SNAPS.length - 1;

function goToFrame(idx) {{
  if (idx < 0 || idx >= ALL_SNAPS.length) return;
  frameIdx = idx;
  slider.value = idx;
  document.getElementById('frame-val').textContent = idx;
  applySnap(ALL_SNAPS[idx]);
}}

function scheduleNext() {{
  if (!running) return;
  const nextIdx = (frameIdx + 1) % ALL_SNAPS.length;
  goToFrame(nextIdx);
  const delay = parseInt(document.getElementById('speed').value);
  timer = setTimeout(scheduleNext, delay);
}}

document.getElementById('btn-play').addEventListener('click', () => {{
  running = !running;
  const btn = document.getElementById('btn-play');
  const badge = document.getElementById('badge-status');
  if (running) {{
    btn.innerHTML = '&#9646;&#9646; Pausar';
    badge.textContent = 'corriendo';
    badge.className = 'badge running';
    scheduleNext();
  }} else {{
    btn.innerHTML = '&#9654; Iniciar';
    badge.textContent = 'detenido';
    badge.className = 'badge';
    clearTimeout(timer);
  }}
}});

document.getElementById('btn-step').addEventListener('click', () => {{
  clearTimeout(timer);
  running = false;
  document.getElementById('btn-play').innerHTML = '&#9654; Iniciar';
  document.getElementById('badge-status').className = 'badge';
  document.getElementById('badge-status').textContent = 'detenido';
  goToFrame((frameIdx + 1) % ALL_SNAPS.length);
}});

document.getElementById('btn-reset').addEventListener('click', () => {{
  clearTimeout(timer);
  running = false;
  document.getElementById('btn-play').innerHTML = '&#9654; Iniciar';
  document.getElementById('badge-status').className = 'badge';
  document.getElementById('badge-status').textContent = 'detenido';
  selectedNode = null;
  document.getElementById('node-info').innerHTML =
    '<div class="placeholder">Haz clic sobre una<br>interseccion o arista<br>para ver su informacion</div>';
  goToFrame(0);
}});

document.getElementById('speed').addEventListener('input', function() {{
  document.getElementById('speed-val').textContent = (this.value/1000).toFixed(1) + 's';
}});

slider.addEventListener('input', function() {{
  clearTimeout(timer);
  running = false;
  document.getElementById('btn-play').innerHTML = '&#9654; Iniciar';
  document.getElementById('badge-status').className = 'badge';
  document.getElementById('badge-status').textContent = 'detenido';
  goToFrame(parseInt(this.value));
}});

function log(msg, cls='') {{
  const el = document.getElementById('log');
  const d  = document.createElement('div');
  d.className = cls;
  d.textContent = `[${{new Date().toLocaleTimeString('es',{{hour12:false}})}}] ${{msg}}`;
  el.prepend(d);
  while (el.children.length > 20) el.removeChild(el.lastChild);
}}

// Init
document.getElementById('s-nodes').textContent = Object.keys(NODES_STATIC).length;
goToFrame(0);
log('tanGo iniciado — ' + Object.keys(NODES_STATIC).length + ' intersecciones cargadas', 'log-ok');
log('Haz clic en un nodo o arista para ver detalles');
</script>
</body>
</html>"""
    return html



if __name__ == "__main__":
    print("tanGo — simulación ZMG Guadalajara")

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

    print(f"  {len(SCENARIOS)} escenarios × {N_TICKS} ticks\n")

    all_histories = []
    final_snap    = None

    for sc in SCENARIOS:
        print(f"  Simulando: {sc['label']}...")
        history = simulate(sc, graph, N_TICKS)
        all_histories.append((sc["label"], history))
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
    print("\n✓ Listo.")
    print(f"  Recomendado: abre tango_vis.html (mapa interactivo + animacion)")
    print(f"  Alternativo: tango_sim.html (Plotly) · tango_map.html (Folium estatico)")


# ─────────────────────────────────────────────────────────────────────────────