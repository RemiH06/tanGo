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
                                RoadSegment, RoadCategory, Phase, Turn,
                                TrafficAxis)
from core.entities      import Vehicle, Pedestrian, VehicleType, Direction
from graph.simulator    import TrafficGraph
from graph.city_loader  import json_to_traffic_graph, load_graph_from_json

OUTPUT_PLOTLY = Path(__file__).parent / "tango_sim.html"
OUTPUT_FOLIUM = Path(__file__).parent / "tango_map.html"
N_TICKS = 40

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
                "phase":     inter.current_phase.value,
                "phase_ns":  inter.phase_ns.value,
                "phase_ew":  inter.phase_ew.value,
                "active_axis": getattr(inter._active_axis, "value", "ns"),
                "pressure":  inter.pressure,
                "itype":     inter.intersection_type,
                "has_light": inter.has_traffic_light,
                "threshold": inter.pressure_threshold,
                "timeout":   inter.red_timeout_ticks,
                "ticks_red": inter._ticks_in_phase,
                "name":      inter.name,
                "lat":       inter.latitude,
                "lon":       inter.longitude,
                "counts":    dict(counts),
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
                hovers.append(
                    f"<b>{n}</b> {nd['name']}<br>"
                    f"Tipo: {nd['itype'].value.upper()} "
                    f"{'🚦' if nd['has_light'] else '⛔ sin semáforo'}<br>"
                    f"─────────────<br>"
                    f"{ns_col} Eje N-S: <b>{nd['phase_ns'].upper()}</b><br>"
                    f"{ew_col} Eje E-O: <b>{nd['phase_ew'].upper()}</b><br>"
                    f"Eje activo: {nd['active_axis'].upper()} {timeout_info}<br>"
                    f"─────────────<br>"
                    f"Presión: {nd['pressure']:.1f}/{nd['threshold']:.0f} ({pct:.0f}%)<br>"
                    f"🚗{c.get('CAR',0)} 🏍{c.get('MOTORCYCLE',0)} "
                    f"🚌{c.get('BUS',0)} 🚛{c.get('TRUCK',0)} "
                    f"🚲{c.get('BICYCLE',0)} 🚑{c.get('EMERGENCY',0)}<br>"
                    f"🚶{c.get('PEDESTRIAN',0)} ♿{c.get('WHEELCHAIR',0)}"
                )
                texts.append(n)

            node_trace = go.Scattergeo(
                lon=lons, lat=lats,
                mode="markers+text",
                marker=dict(size=sizes, color=colors, symbol=symbols,
                            line=dict(width=2,color=borders), opacity=0.9),
                text=texts,
                textposition="middle center",
                textfont=dict(size=9,color="#f8fafc",family="monospace"),
                hovertext=hovers, hoverinfo="text",
                showlegend=False,
            )

            # Flujos como texto sobre aristas
            flow_annotations = []
            for fl in snap["flows"]:
                if fl["from"] not in graph.intersections: continue
                if fl["to"]   not in graph.intersections: continue
                if fl["fwd"]+fl["bwd"] == 0: continue
                n_a = graph.intersections[fl["from"]]
                n_b = graph.intersections[fl["to"]]
                flow_annotations.append(dict(
                    lon=(n_a.longitude+n_b.longitude)/2,
                    lat=(n_a.latitude +n_b.latitude)/2,
                    text=f"→{fl['fwd']}←{fl['bwd']}",
                    showarrow=False,
                    font=dict(size=7,color="#64748b"),
                ))

            title_str = (
                f"{scenario_label} | tick #{snap['tick']} | "
                f"entidades: {snap['total']} | "
                f"en verde: {snap['greens']}/{len(snap['nodes'])}"
            )
            frame_name = f"{scenario_label[:8]}|{snap['tick']}"

            all_frames.append(go.Frame(
                data=[node_trace],
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

    graph.close()
    print("\n✓ Listo. Abre los HTML en tu navegador.")