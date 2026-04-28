"""
tests/sim2/casos_exp/casos_exp.py
----------------------------------
Script unificado para todos los casos experimentales de sim2.

Cada caso define un número de carros en rutas diversas que se cruzan.
El algoritmo tanGo resuelve los conflictos en tiempo real.

Casos disponibles (configurables en casos_config.json):
  246 — 2 carros
  369 — 3 carros
  48C — 4 carros
  5AF — 5 carros

Uso:
    python tests/sim2/casos_exp/casos_exp.py           # corre todos
    python tests/sim2/casos_exp/casos_exp.py --caso 369
    python tests/sim2/casos_exp/casos_exp.py --caso 48C --caso 5AF

El script reutiliza la lógica de caso246 pero es completamente
paramétrico — añadir un nuevo caso solo requiere editar casos_config.json.
"""

from __future__ import annotations
import sys, argparse, logging, json as _json, uuid, time as _time
from pathlib import Path
from datetime import datetime
from collections import Counter

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import networkx as nx

from core.context    import TrafficContext
from core.algorithm  import TrafficAlgorithm, TICK_DURATION_S
from core.movement   import MovementEngine, MovingEntity
from core.road       import Phase, IntersectionType
from core.entities   import Vehicle, VehicleType, Direction
from graph.simulator import TrafficGraph
from graph.city_loader import json_to_traffic_graph

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")

CASOS_CFG  = Path(__file__).parent / "casos_config.json"
CITY_JSON  = ROOT / "graph" / "city_graph.json"
OUTPUT_DIR = Path(__file__).parent

CTX_DICT = dict(
    timestamp      = datetime(2024, 3, 4, 8, 0),
    temperature_c  = 22.0,
    is_raining     = False,
    wind_speed_kmh = 10.0,
    visibility_m   = 10000.0,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_center(graph: TrafficGraph) -> tuple[float, float]:
    nodes = list(graph.intersections.values())
    return (sum(n.latitude  for n in nodes) / len(nodes),
            sum(n.longitude for n in nodes) / len(nodes))



def find_diverse_routes(graph: TrafficGraph,
                        n_cars: int) -> list[tuple[str, str, list[str]]]:
    """
    Encuentra n_cars rutas con spawn en nodos de mayor weight.

    Estrategia:
      1. Ordenar nodos semaforizados por node_weight descendente.
      2. Orígenes: top N nodos (mayor centralidad → más tráfico real).
      3. Destinos: bottom N nodos en orden inverso (lados opuestos).
         Carro 1: nodo más central → nodo menos central
         Carro 2: 2do más central → 2do menos central
         Esto garantiza que los carros vengan de extremos distintos
         y crucen por el centro de la red.
      4. Rutas calculadas con Dijkstra sobre el grafo real.
    """
    signaled = sorted(
        [nid for nid, inter in graph.intersections.items()
         if inter.has_traffic_light],
        key=lambda nid: -graph.intersections[nid].node_weight
    )
    if not signaled:
        signaled = list(graph.intersections.keys())
    if len(signaled) < 2:
        return []

    n = len(signaled)
    # Orígenes: nodos con más peso (más centrales, más tráfico)
    origins = signaled[:min(n_cars, n // 2 + 1)]
    # Destinos: nodos con menos peso, en orden inverso
    # (el de menor peso es el destino del carro que sale del de mayor peso)
    dests   = list(reversed(signaled[-(min(n_cars, n // 2 + 1)):]))

    selected: list[tuple[str, str, list[str]]] = []
    used_pairs: set[tuple] = set()

    for i in range(n_cars):
        origin = origins[i % len(origins)]
        dest   = dests[i % len(dests)]

        if origin == dest:
            # Rotar destino si coincide con origen
            dest = dests[(i + 1) % len(dests)]

        if (origin, dest) in used_pairs:
            # Buscar par alternativo
            found = False
            for alt_dest in reversed(signaled):
                if alt_dest == origin or (origin, alt_dest) in used_pairs:
                    continue
                try:
                    path = nx.shortest_path(graph.graph, origin, alt_dest)
                    selected.append((origin, alt_dest, path))
                    used_pairs.add((origin, alt_dest))
                    found = True
                    break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            if not found:
                continue
            continue

        try:
            path = nx.shortest_path(graph.graph, origin, dest)
            selected.append((origin, dest, path))
            used_pairs.add((origin, dest))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback: cualquier ruta desde este origen
            for candidate in reversed(signaled):
                if candidate == origin or (origin, candidate) in used_pairs:
                    continue
                try:
                    path = nx.shortest_path(graph.graph, origin, candidate)
                    selected.append((origin, candidate, path))
                    used_pairs.add((origin, candidate))
                    break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

    return selected[:n_cars]



# ── Simulación ────────────────────────────────────────────────────────────────

def run_caso(graph: TrafficGraph, n_cars: int,
             n_ticks: int) -> tuple[list[dict], list[tuple]]:
    """
    Simula n_cars carros en rutas diversas con TrafficAlgorithm.
    Lógica idéntica a caso246 pero parametrizada.
    """
    ctx     = TrafficContext.build(**CTX_DICT)
    routes  = find_diverse_routes(graph, n_cars)

    if not routes:
        print("  ⚠ No se encontraron rutas — grafo demasiado pequeño")
        return [], []

    print(f"  Rutas seleccionadas ({len(routes)} carros):")
    for i, (o, d, path) in enumerate(routes):
        sig = sum(1 for n in path if graph.intersections[n].has_traffic_light)
        print(f"    Carro {i+1}: {o} → {d}  "
              f"({len(path)} nodos, {sig} semáforos)  "
              f"{' → '.join(path)}")

    # Identificar nodos de cruce (nodos compartidos por 2+ rutas)
    node_appearances = Counter(n for _, _, path in routes for n in path)
    crossing_nodes = {n for n, c in node_appearances.items() if c >= 2
                      and graph.intersections[n].has_traffic_light}
    print(f"  Nodos de cruce esperados: {crossing_nodes or 'ninguno detectado'}\n")

    algo     = TrafficAlgorithm(graph)
    algo.reset()
    movement = MovementEngine(graph, spawn_rate=0,
                              max_entities=n_cars + 5)

    # Inyectar carros
    mov_cfg = _json.load(open(CASOS_CFG)).get("movement", {})
    max_ttn = int(mov_cfg.get("ticks_to_next_max", 2))

    for origin, dest, path in routes:
        car = Vehicle(str(uuid.uuid4()), VehicleType.CAR, Direction.EAST)
        car.origin_node = car.current_node = origin
        car.destination_node = dest

        ticks = 1
        if len(path) > 1:
            seg = movement._get_segment(path[0], path[1])
            if seg:
                ticks = max(1, min(max_ttn, round(
                    car.travel_time_ticks(seg["segment"].length_m,
                                          seg["segment"].category.name, ctx)
                )))
        movement._active.append(MovingEntity(
            entity=car, route=path, route_idx=0, ticks_to_next=ticks
        ))

    history = []
    for tick_n in range(n_ticks):
        phases = {nid: inter.current_phase.value
                  for nid, inter in graph.intersections.items()}

        entities = movement.tick(ctx, phases)
        result   = algo.run_tick(entities, ctx)

        # Override por distancia en ruta:
        # Un carro solo fuerza verde cuando NO hay otro carro
        # a ≤1 nodo de distancia EN SU PROPIA RUTA.
        # Si hay alguien cerca en la ruta → el semáforo arbitra por presión.
        # Si nadie en la ruta está cerca → es rojo innecesario, forzar.
        # Anti-deadlock: si llevan 5+ ticks detenidos, el de mayor node_weight gana.

        waiting = sorted(
            [me for me in movement._active
             if (me.ticks_to_next == 0
                 and graph.intersections[me.current_node].has_traffic_light
                 and graph.intersections[me.current_node].current_phase == Phase.RED)],
            key=lambda me: -graph.intersections[me.current_node].node_weight
        )

        for me in waiting:
            inter     = graph.intersections[me.current_node]
            ticks_red = inter._ticks_in_phase

            # Nodos en la ruta del carro a 1 salto adelante y atrás
            route_neighbors = set()
            idx = me.route_idx
            if idx > 0:
                route_neighbors.add(me.route[idx - 1])
            if idx + 1 < len(me.route):
                route_neighbors.add(me.route[idx + 1])

            # Carros de otros en esos nodos de ruta
            other_on_route = any(
                other.current_node in route_neighbors
                for other in movement._active
                if other is not me
            )

            if other_on_route:
                # Hay alguien en la ruta cerca → anti-deadlock solo si llevan mucho
                if ticks_red >= 5:
                    inter.current_phase       = Phase.GREEN
                    inter._ticks_in_phase     = 0
                    inter._timeout_triggered  = False
                    inter._green_started_tick = algo._tick
            else:
                # Nadie cerca en la ruta → rojo innecesario, forzar rápido
                if ticks_red >= 2:
                    inter.current_phase       = Phase.GREEN
                    inter._ticks_in_phase     = 0
                    inter._timeout_triggered  = False
                    inter._green_started_tick = algo._tick

        stats   = movement.get_stats(phases)
        heatmap = movement.get_heatmap()

        # Posiciones de carros
        car_positions = []
        for i, (origin, dest, path) in enumerate(routes):
            me_found = next(
                (me for me in movement._active
                 if me.route == path and me.origin == origin), None
            )
            if me_found:
                inter = graph.intersections.get(me_found.current_node)
                car_positions.append({
                    "idx": i, "node": me_found.current_node,
                    "node_name": inter.name if inter else me_found.current_node,
                    "lat": inter.latitude  if inter else 0,
                    "lon": inter.longitude if inter else 0,
                    "progress":      round(me_found.progress_pct, 0),
                    "ticks_to_next": me_found.ticks_to_next,
                    "phase": phases.get(me_found.current_node, "?"),
                    "arrived": False,
                    "is_crossing": me_found.current_node in crossing_nodes,
                })
            else:
                arrived_me = next(
                    (me for me in movement._arrived
                     if me.route == path and me.origin == origin), None
                )
                inter = graph.intersections.get(dest)
                car_positions.append({
                    "idx": i, "node": dest,
                    "node_name": inter.name if inter else dest,
                    "lat": inter.latitude  if inter else 0,
                    "lon": inter.longitude if inter else 0,
                    "progress": 100.0, "ticks_to_next": 0,
                    "phase": "arrived", "arrived": True,
                    "is_crossing": False,
                })

        # Frame
        nodes_frame = {}
        for nid, ns in result.nodes.items():
            inter = graph.intersections[nid]
            nodes_frame[nid] = {
                "phase": ns.phase, "phase_ns": ns.phase_ns,
                "phase_ew": ns.phase_ew, "active_axis": ns.active_axis,
                "signals": ns.signals,
                "pressure": ns.pressure, "pressure_own": ns.pressure_own,
                "wave_offset_s": ns.wave_offset_s,
                "has_light": ns.has_light, "threshold": ns.threshold,
                "ticks_red": ns.ticks_in_phase, "timeout": ns.timeout_ticks,
                "itype": inter.intersection_type,
                "geo_label": inter.geometry_label,
                "name": inter.name, "lat": inter.latitude, "lon": inter.longitude,
                "counts": ns.entity_counts, "cluster_id": ns.cluster_id,
                "node_weight": round(inter.node_weight, 3),
                "heat": round(heatmap.get(nid, 0.0), 3),
                "is_route": nid in {n for _, _, p in routes for n in p},
                "is_crossing": nid in crossing_nodes,
            }

        history.append({
            "tick": result.tick_number, "nodes": nodes_frame,
            "flows": result.flows, "particles": movement.get_particles(),
            "heatmap": heatmap, "total": result.total_entities,
            "greens": result.green_count, "reds": result.red_count,
            "blinks": result.blink_count,
            "active_moving": stats.active_entities,
            "stopped": stats.stopped, "moving": stats.moving,
            "arrived": len(movement._arrived),
            "car_positions": car_positions,
        })

        status = (f"Tick {result.tick_number:2d}: "
                  f"{stats.active_entities} activos "
                  f"({stats.moving} mov, {stats.stopped} det) "
                  f"| {len(movement._arrived)}/{len(routes)} llegaron")
        if stats.stopped > 0:
            status += f" ← CONFLICTO"
        print(f"  {status}")

        if stats.active_entities == 0:
            print(f"\n  ✓ Todos llegaron en tick {result.tick_number}")
            break

    return history, routes


# ── Visualización ─────────────────────────────────────────────────────────────

def build_vis(graph: TrafficGraph, history: list[dict],
              routes: list[tuple], caso_cfg: dict) -> str:
    """HTML interactivo — igual que caso246 pero con config dinámica."""
    clat, clon  = compute_center(graph)
    colors      = caso_cfg.get("color_set",
                               ["#f97316","#3b82f6","#22c55e","#a855f7","#ef4444"])
    nombre      = caso_cfg.get("nombre", "Caso experimental")
    descripcion = caso_cfg.get("descripcion", "")
    n_cars      = len(routes)

    # Nodos de cruce
    node_appearances = Counter(n for _, _, path in routes for n in path)
    crossing_nodes   = {n for n, c in node_appearances.items() if c >= 2
                        and graph.intersections[n].has_traffic_light}

    routes_js = [
        {
            "id":          i,
            "color":       colors[i % len(colors)],
            "origin":      o,
            "destination": d,
            "path":        path,
            "sig_count":   sum(1 for n in path
                               if graph.intersections[n].has_traffic_light),
        }
        for i, (o, d, path) in enumerate(routes)
    ]

    route_nodes = {n for _, _, path in routes for n in path}

    # Serializar snapshots
    snaps_js = []
    for snap in history:
        njs = {}
        for nid, nd in snap["nodes"].items():
            inter = graph.intersections[nid]
            njs[nid] = {
                "phase": nd["phase"], "phase_ns": nd["phase_ns"],
                "phase_ew": nd["phase_ew"], "active_axis": nd["active_axis"],
                "signals": nd.get("signals", {}),
                "pressure": round(nd["pressure"], 1),
                "wave_offset_s": round(nd.get("wave_offset_s", 0), 1),
                "has_light": nd["has_light"], "threshold": nd["threshold"],
                "ticks_red": nd.get("ticks_red", 0),
                "timeout": nd.get("timeout", 8),
                "itype": inter.intersection_type.value,
                "geo_label": inter.geometry_label,
                "name": nd["name"], "lat": nd["lat"], "lon": nd["lon"],
                "counts": nd.get("counts", {}),
                "node_weight": nd.get("node_weight", 1.0),
                "heat": nd.get("heat", 0.0),
                "is_route": nid in route_nodes,
                "is_crossing": nid in crossing_nodes,
            }
        snaps_js.append({
            "tick": snap["tick"], "nodes": njs,
            "flows": snap.get("flows", []),
            "particles": snap.get("particles", []),
            "heatmap": snap.get("heatmap", {}),
            "total": snap["total"], "greens": snap["greens"],
            "reds": snap.get("reds", 0), "blinks": snap.get("blinks", 0),
            "active_moving": snap.get("active_moving", 0),
            "stopped": snap.get("stopped", 0),
            "moving": snap.get("moving", 0),
            "arrived": snap.get("arrived", 0),
            "car_positions": snap.get("car_positions", []),
        })

    # Aristas y nodos estáticos
    from core.road import RoadCategory
    edges_js, drawn = [], set()
    for a, b, data in graph.graph.edges(data=True):
        pair = tuple(sorted([a, b]))
        if pair in drawn: continue
        drawn.add(pair)
        seg = data["segment"]
        na, nb = graph.intersections[a], graph.intersections[b]
        edges_js.append({
            "from": a, "to": b,
            "lat_a": na.latitude, "lon_a": na.longitude,
            "lat_b": nb.latitude, "lon_b": nb.longitude,
            "category": seg.category.name,
        })

    ns_js = {}
    for nid, inter in graph.intersections.items():
        ns_js[nid] = {
            "lat": inter.latitude, "lon": inter.longitude,
            "name": inter.name, "itype": inter.intersection_type.value,
            "geo_label": inter.geometry_label,
            "has_light": inter.has_traffic_light,
            "node_weight": round(inter.node_weight, 3),
            "is_route": nid in route_nodes,
            "is_crossing": nid in crossing_nodes,
        }

    sj  = _json.dumps(snaps_js)
    ej  = _json.dumps(edges_js)
    nj  = _json.dumps(ns_js)
    rj  = _json.dumps(routes_js)
    cxj = _json.dumps(list(crossing_nodes))
    ns  = len(snaps_js)

    # Leyenda inicial de rutas
    legend_init = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
        f'border-bottom:1px solid var(--brd)">'
        f'<div style="background:{r["color"]};border-radius:50%;width:18px;height:18px;'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:10px;font-weight:900;color:#000;flex-shrink:0">{r["id"]+1}</div>'
        f'<div style="flex:1;font-size:10px">'
        f'<div style="color:{r["color"]};font-weight:600">{r["origin"]}→{r["destination"]}</div>'
        f'<div style="color:var(--mut);font-size:9px">{r["sig_count"]} semáforos</div>'
        f'</div></div>'
        for r in routes_js
    )

    crossing_badge = (
        f'<div style="background:#7f1d1d;color:#fca5a5;padding:4px 8px;'
        f'border-radius:5px;font-size:10px;margin-bottom:6px">'
        f'⬡ Nodos de cruce: {", ".join(sorted(crossing_nodes)) or "ninguno"}</div>'
        if crossing_nodes else ""
    )

    return f"""<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tanGo — {nombre}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{--bg:#0f1117;--sur:#1a1d2e;--brd:#2a2d3e;--txt:#e2e8f0;--mut:#64748b;
      --grn:#22c55e;--yel:#eab308;--red:#ef4444;--tel:#14b8a6;--blue:#3b82f6}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;
      font-size:13px;height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{padding:8px 16px;background:#0c1a0c;border-bottom:2px solid #166534;
        display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:15px;font-weight:600;color:var(--grn)}}
.cbadge{{background:#166534;color:var(--grn);padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700}}
.badge{{font-size:10px;padding:2px 8px;border-radius:999px;background:var(--brd);color:var(--mut)}}
.layout{{display:flex;flex:1;overflow:hidden}}
#map{{flex:1}}
aside{{width:296px;background:var(--sur);border-left:1px solid var(--brd);
       display:flex;flex-direction:column;overflow:hidden}}
.sec{{padding:10px 12px;border-bottom:1px solid var(--brd)}}
.sec h3{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:6px}}
.row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
label{{font-size:11px;color:var(--mut);min-width:68px}}
input[type=range]{{flex:1;accent-color:var(--grn)}}
.val{{font-size:11px;min-width:36px;text-align:right}}
.btn{{padding:5px 10px;border:none;border-radius:6px;font-size:11px;font-weight:500;cursor:pointer}}
.bp{{background:var(--grn);color:#000}}.bs{{background:var(--brd);color:var(--txt)}}
.btn-row{{display:flex;gap:5px;flex-wrap:wrap}}
.sg{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.st{{background:var(--bg);border-radius:6px;padding:7px;border:1px solid var(--brd)}}
.sv{{font-size:17px;font-weight:700;color:var(--grn)}}.sl{{font-size:9px;color:var(--mut)}}
#ni{{flex:1;overflow-y:auto;padding:10px}}
.ic{{background:var(--bg);border:1px solid var(--brd);border-radius:7px;padding:10px;margin-bottom:7px}}
.ic h4{{font-size:12px;font-weight:600;margin-bottom:5px}}
.ir{{display:flex;justify-content:space-between;font-size:11px;color:var(--mut);margin:2px 0}}
.ir span{{color:var(--txt)}}
.pill{{display:inline-block;padding:2px 7px;border-radius:999px;font-size:10px;font-weight:700}}
.pg{{background:#166534;color:var(--grn)}}.py{{background:#713f12;color:var(--yel)}}
.pr{{background:#7f1d1d;color:var(--red)}}.pb{{background:#1e293b;color:#f59e0b}}
#log{{max-height:90px;overflow-y:auto;padding:6px 10px;font-size:10px;
      font-family:monospace;color:var(--mut);border-top:1px solid var(--brd)}}
.lok{{color:var(--grn)}}.lw{{color:var(--yel)}}.le{{color:var(--red)}}
.crossing-node{{box-shadow:0 0 12px #ef4444 !important}}
.leaflet-tile{{filter:brightness(.7) saturate(.6)}}
.leaflet-container{{background:var(--bg)}}
</style></head><body>
<header>
  <h1>tanGo — {nombre}</h1>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="cbadge">{nombre.upper()}</span>
    <span class="badge" id="bt">tick #0</span>
    <span class="badge" id="bc">{n_cars} carros</span>
    <span class="badge" id="bst">0 detenidos</span>
  </div>
</header>
<div class="layout">
<div id="map"></div>
<aside>
  <div class="sec">
    <h3>{nombre} — {descripcion}</h3>
    {crossing_badge}
    <div class="btn-row" style="margin-bottom:8px">
      <button class="btn bp" id="pp">&#9654; Iniciar</button>
      <button class="btn bs" id="st2">&#9197; Paso</button>
      <button class="btn bs" id="rs">&#8635; Reset</button>
    </div>
    <div class="row"><label>Velocidad</label>
      <input type="range" id="spd" min="300" max="3000" value="900" step="100">
      <span class="val" id="vs">0.9s</span></div>
    <div class="row"><label>Frame</label>
      <input type="range" id="fsl" min="0" max="{ns-1}" value="0">
      <span class="val" id="vf">0/{ns-1}</span></div>
    <div class="row" style="margin-top:4px">
      <label style="font-size:10px">Heatmap</label>
      <input type="checkbox" id="ht" checked>
      <label style="font-size:10px;min-width:auto">Rutas</label>
      <input type="checkbox" id="rt" checked>
    </div>
  </div>
  <div class="sec"><h3>Estadisticas</h3>
    <div class="sg">
      <div class="st"><div class="sv" id="sk">0</div><div class="sl">Tick</div></div>
      <div class="st"><div class="sv" id="sc2" style="color:var(--grn)">{n_cars}</div><div class="sl">Total carros</div></div>
      <div class="st"><div class="sv" id="smoving" style="color:var(--tel)">0</div><div class="sl">Moviendose</div></div>
      <div class="st"><div class="sv" id="sstopped" style="color:var(--red)">0</div><div class="sl">Detenidos</div></div>
      <div class="st"><div class="sv" id="sa" style="color:#a78bfa">0</div><div class="sl">Llegaron</div></div>
      <div class="st"><div class="sv" id="sg2" style="color:var(--grn)">0</div><div class="sl">Verdes</div></div>
      <div class="st"><div class="sv" id="sr" style="color:var(--red)">0</div><div class="sl">Rojos</div></div>
      <div class="st"><div class="sv" id="sn">{len(ns_js)}</div><div class="sl">Nodos</div></div>
    </div>
  </div>
  <div class="sec">
    <h3>Posicion de carros</h3>
    <div style="font-size:9px;color:var(--mut);margin-bottom:4px">
      🔴 borde rojo = detenido &nbsp;⬡ = nodo de cruce &nbsp;🏁 = llegó
    </div>
    <div id="car-legend">{legend_init}</div>
  </div>
  <div id="ni"><div style="color:var(--mut);font-size:11px;text-align:center;margin-top:12px">Clic en nodo</div></div>
  <div id="log"></div>
</aside>
</div>
<script>
const S={sj};
const E={ej};
const N={nj};
const R={rj};
const CX=new Set({cxj});
const NS={ns};
const PC={{green:'#22c55e',yellow:'#eab308',red:'#ef4444',blink:'#f59e0b'}};
const IR={{master:'#f59e0b',normal:'#3b82f6',blind:'#64748b'}};
const CC={{MAIN_AVENUE:'#1d4ed8',SECONDARY_AVENUE:'#6d28d9',STREET:'#1e293b',HIGHWAY:'#0f172a',ALLEY:'#111827'}};
const CW={{MAIN_AVENUE:5,SECONDARY_AVENUE:3,STREET:1.5,HIGHWAY:6,ALLEY:1}};
let _blinkOn=true; setInterval(()=>{{_blinkOn=!_blinkOn;}},600);

const map=L.map('map').setView([{clat},{clon}],14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'CartoDB',maxZoom:19}}).addTo(map);

E.forEach(e=>L.polyline([[e.lat_a,e.lon_a],[e.lat_b,e.lon_b]],
  {{color:CC[e.category]||'#1e293b',weight:CW[e.category]||1.5,opacity:.5}}).addTo(map));

// Rutas coloreadas
const routeLayers=[];
R.forEach(r=>{{
  const coords=r.path.map(id=>N[id]?[N[id].lat,N[id].lon]:null).filter(Boolean);
  if(coords.length>1){{
    const ln=L.polyline(coords,{{color:r.color,weight:3.5,opacity:.75,dashArray:'8,3'}}).addTo(map);
    routeLayers.push(ln);
    L.circleMarker(coords[0],{{radius:9,color:r.color,fillColor:r.color,fillOpacity:1,weight:2}})
     .addTo(map).bindTooltip('Origen '+r.origin,{{permanent:false}});
    L.circleMarker(coords[coords.length-1],{{radius:9,color:r.color,fillColor:'#000',fillOpacity:.8,weight:3}})
     .addTo(map).bindTooltip('Destino '+r.destination,{{permanent:false}});
  }}
}});
document.getElementById('rt').addEventListener('change',function(){{
  routeLayers.forEach(ln=>this.checked?map.addLayer(ln):map.removeLayer(ln));
}});

// Nodos — los de cruce tienen borde especial
const NM={{}};
Object.entries(N).forEach(([id,nd])=>{{
  const r=nd.itype==='master'?13:nd.itype==='normal'?10:7;
  const isCrossing=CX.has(id);
  const ringColor=isCrossing?'#ef4444':(IR[nd.itype]||'#64748b');
  const ringW=isCrossing?4:nd.is_route?2.5:1.5;
  const m=L.circleMarker([nd.lat,nd.lon],{{
    radius:isCrossing?r+3:r,
    color:ringColor,fillColor:'#f59e0b',fillOpacity:.88,weight:ringW,
  }}).addTo(map);
  m.bindTooltip(
    `<b>${{id}}</b> ${{nd.geo_label}}${{isCrossing?' ⬡ CRUCE':''}}`,
    {{permanent:false,direction:'top'}}
  );
  m.on('click',()=>showNode(id,S[fi]));NM[id]=m;
}});

// Marcadores de carros
const PM={{}};const HM={{}};
function clearP(){{Object.values(PM).forEach(m=>map.removeLayer(m));Object.keys(PM).forEach(k=>delete PM[k]);}}
function clearH(){{Object.values(HM).forEach(m=>map.removeLayer(m));Object.keys(HM).forEach(k=>delete HM[k]);}}

function apply(snap){{
  const showHeat=document.getElementById('ht').checked;

  Object.entries(snap.nodes).forEach(([id,nd])=>{{
    const m=NM[id];if(!m)return;
    const st=N[id];
    const baseR=st.itype==='master'?13:st.itype==='normal'?10:7;
    const isCX=CX.has(id);
    m.setStyle({{
      fillColor:!nd.has_light?'#374151':nd.phase==='blink'?(_blinkOn?'#f59e0b':'#1e293b'):(PC[nd.phase]||'#ef4444'),
      fillOpacity:!nd.has_light?.3:.9,
      radius:isCX?(baseR+3+(showHeat?nd.heat*8:0)):(baseR+(showHeat?nd.heat*6:0)),
      color:isCX?'#ef4444':(IR[nd.itype]||'#64748b'),
    }});
    m.off('click');m.on('click',()=>showNode(id,snap));
  }});

  // Heatmap
  clearH();
  if(showHeat){{
    Object.entries(snap.heatmap||{{}}).forEach(([nid,heat])=>{{
      if(heat<0.05)return;
      const nd=N[nid];if(!nd)return;
      L.circleMarker([nd.lat,nd.lon],{{
        radius:6+heat*20,color:'transparent',
        fillColor:'#f59e0b',fillOpacity:heat*.35,interactive:false,
      }}).addTo(map);
    }});
  }}

  // Marcadores numerados de carros
  clearP();
  (snap.car_positions||[]).forEach(cp=>{{
    if(!cp.lat&&!cp.lon)return;
    const r=R[cp.idx]||{{}};
    const col=r.color||'#e2e8f0';
    const isStopped=cp.phase==='red'&&!cp.arrived;
    const isArrived=cp.arrived;
    const isCrossing=cp.is_crossing;
    const icon=L.divIcon({{
      html:`<div style="
        background:${{col}};
        border:3px solid ${{isStopped?'#ef4444':isArrived?'#22c55e':'#fff'}};
        border-radius:50%;width:30px;height:30px;
        display:flex;align-items:center;justify-content:center;
        font-size:14px;font-weight:900;color:#000;
        box-shadow:0 0 ${{isStopped?'14px #ef4444':isCrossing?'14px '+col:'8px '+col}};
        opacity:${{isArrived?0.5:1}};
      ">${{cp.idx+1}}</div>`,
      className:'',iconAnchor:[15,15]
    }});
    const phLabel={{green:'🟢 Verde',red:'🔴 Rojo — detenido',
                    yellow:'🟡 Amarillo',blink:'🟠 Blink',arrived:'🏁 Llegó','?':'❓'}};
    const pm=L.marker([cp.lat,cp.lon],{{icon,zIndexOffset:2000+cp.idx}}).addTo(map);
    pm.bindTooltip(
      `<b>Carro ${{cp.idx+1}}</b> (${{r.origin||'?'}}→${{r.destination||'?'}})<br>`+
      `Nodo: <b>${{cp.node}}</b>${{cp.is_crossing?' ⬡':''}}<br>`+
      `${{cp.node_name}}<br>`+
      `${{phLabel[cp.phase]||cp.phase}} · ${{cp.progress}}%`,
      {{permanent:false,direction:'top'}}
    );
    PM[`car_${{cp.idx}}`]=pm;
  }});

  // Leyenda dinámica
  const posHtml=(snap.car_positions||[]).map(cp=>{{
    const r=R[cp.idx]||{{}};
    const col=r.color||'#e2e8f0';
    const ph={{green:'🟢',red:'🔴',yellow:'🟡',blink:'🟠',arrived:'🏁','?':'❓'}}[cp.phase]||'❓';
    const cx=cp.is_crossing?'⬡':'';
    return `<div style="display:flex;align-items:center;gap:6px;padding:3px 0;border-bottom:1px solid var(--brd)">
      <div style="background:${{col}};border:2px solid ${{cp.phase==='red'&&!cp.arrived?'#ef4444':'transparent'}};
                  border-radius:50%;width:20px;height:20px;
                  display:flex;align-items:center;justify-content:center;
                  font-size:11px;font-weight:900;color:#000;flex-shrink:0">${{cp.idx+1}}</div>
      <div style="flex:1;font-size:10px;min-width:0">
        <div style="color:${{col}};font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
          ${{cp.node}} ${{ph}} ${{cx}}</div>
        <div style="color:var(--mut);font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
          ${{cp.node_name}}</div>
      </div>
      <div style="font-size:10px;color:var(--mut);flex-shrink:0">${{cp.progress}}%</div>
    </div>`;
  }}).join('');
  document.getElementById('car-legend').innerHTML=posHtml||
    '<div style="color:var(--mut);font-size:10px">Sin carros activos</div>';

  // Stats
  const pc={{green:0,red:0,blink:0,blind:0}};
  Object.values(snap.nodes).forEach(nd=>{{
    if(!nd.has_light)pc.blind++;
    else if(nd.phase==='blink')pc.blink++;
    else if(nd.phase==='green')pc.green++;
    else if(nd.phase==='red')pc.red++;
  }});
  document.getElementById('sk').textContent=snap.tick;
  document.getElementById('smoving').textContent=snap.moving||0;
  document.getElementById('sstopped').textContent=snap.stopped||0;
  document.getElementById('sa').textContent=snap.arrived||0;
  document.getElementById('sg2').textContent=pc.green;
  document.getElementById('sr').textContent=pc.red;
  document.getElementById('bt').textContent='tick #'+snap.tick;
  document.getElementById('bc').textContent=(snap.moving||0)+' en mov.';
  document.getElementById('bst').textContent=(snap.stopped||0)+' det.';
  document.getElementById('bst').style.background=(snap.stopped||0)>0?'#7f1d1d':'var(--brd)';
  document.getElementById('bst').style.color=(snap.stopped||0)>0?'#fca5a5':'var(--mut)';

  if((snap.stopped||0)>0)
    log((snap.stopped)+' carro(s) en nodo de cruce — resolviendo conflicto','lw');
  if(snap.arrived>=snap.active_moving+snap.arrived&&snap.arrived>0)
    log('✓ Todos los carros llegaron al destino','lok');
}}

function showNode(id,snap){{
  const nd=snap?snap.nodes[id]:null;
  const st=N[id];const isCX=CX.has(id);
  const pC=p=>({{'green':'pg','yellow':'py','red':'pr','blink':'pb'}}[p]||'pr');
  const carHere=(snap?.car_positions||[]).filter(cp=>cp.node===id&&!cp.arrived);
  document.getElementById('ni').innerHTML=nd?`
  <div class="ic">
    <h4>${{id}} ${{isCX?'<span style="color:#ef4444">⬡ CRUCE</span>':''}}
       ${{nd.is_route?'<span style="color:var(--grn);font-size:9px">● ruta</span>':''}}</h4>
    <div class="ir">Tipo<span>${{(nd.itype||'').toUpperCase()}}</span></div>
    <div class="ir">node_weight<span style="color:var(--blue)">${{nd.node_weight}}</span></div>
    ${{carHere.length?`<div class="ir" style="color:#fca5a5">Carros aqui<span>${{carHere.map(c=>c.idx+1).join(', ')}}</span></div>`:''}}
  </div>
  <div class="ic"><h4>Semaforo</h4>
    <div class="ir">Estado<span><span class="pill ${{pC(nd.phase)}}">${{nd.phase.toUpperCase()}}</span></span></div>
    <div class="ir">Presion<span style="color:${{nd.pressure>=nd.threshold?'var(--red)':'var(--grn)'}}">${{nd.pressure}}/${{nd.threshold}}</span></div>
    ${{nd.wave_offset_s>0?`<div class="ir">Ola verde<span style="color:#f59e0b">${{nd.wave_offset_s}}s</span></div>`:''}}
  </div>`:'<div class="ic" style="color:var(--mut)">Sin datos</div>';
}}

let fi=0,run=false,tmr=null;
const fsl=document.getElementById('fsl');
function goTo(idx){{if(idx<0||idx>=NS)return;fi=idx;fsl.value=idx;
  document.getElementById('vf').textContent=idx+'/'+(NS-1);apply(S[idx]);}}
function next(){{if(!run)return;goTo((fi+1)%NS);
  tmr=setTimeout(next,parseInt(document.getElementById('spd').value));}}
document.getElementById('pp').addEventListener('click',()=>{{
  run=!run;const b=document.getElementById('pp');
  if(run){{b.innerHTML='&#9646;&#9646; Pausar';next();}}
  else{{b.innerHTML='&#9654; Iniciar';clearTimeout(tmr);}}}});
document.getElementById('st2').addEventListener('click',()=>{{
  clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';
  goTo((fi+1)%NS);}});
document.getElementById('rs').addEventListener('click',()=>{{
  clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';
  goTo(0);}});
document.getElementById('spd').addEventListener('input',function(){{
  document.getElementById('vs').textContent=(this.value/1000).toFixed(1)+'s';}});
fsl.addEventListener('input',function(){{
  clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';
  goTo(parseInt(this.value));}});
document.getElementById('ht').addEventListener('change',()=>apply(S[fi]));

function log(m,c=''){{const el=document.getElementById('log');
  const d=document.createElement('div');d.className=c;
  d.textContent='[t'+S[fi].tick+'] '+m;
  el.prepend(d);while(el.children.length>20)el.removeChild(el.lastChild);}}

goTo(0);
log('{nombre} iniciado — {n_cars} carros en rutas cruzadas','lok');
log('Los nodos ⬡ en rojo son los cruces esperados','lw');
</script></body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Casos experimentales tanGo sim2")
    parser.add_argument("--caso", action="append", default=[],
                        help="Caso a ejecutar (246, 369, 48C, 5AF). Repetible.")
    args = parser.parse_args()

    cfg = _json.load(open(CASOS_CFG))
    casos_disponibles = cfg["casos"]

    # Si no se especifica, correr todos
    casos_a_correr = args.caso if args.caso else list(casos_disponibles.keys())

    # Cargar grafo una sola vez
    if CITY_JSON.exists():
        graph = json_to_traffic_graph(CITY_JSON)
        signaled = sum(1 for i in graph.intersections.values() if i.has_traffic_light)
        print(f"\ntanGo — Casos experimentales sim2")
        print(f"  Grafo: {graph.graph.number_of_nodes()} nodos "
              f"({signaled} semaforizados)\n")
    else:
        from graph.simulator import TrafficGraph as _TG
        graph = _TG(); graph.build_sample_city()
        print("  Usando grafo de ejemplo (9 nodos)\n")

    resultados = []

    for caso_id in casos_a_correr:
        if caso_id not in casos_disponibles:
            print(f"⚠ Caso '{caso_id}' no encontrado en casos_config.json")
            continue

        caso_cfg = casos_disponibles[caso_id]
        n_cars   = caso_cfg["n_cars"]
        n_ticks  = caso_cfg["n_ticks"]
        nombre   = caso_cfg["nombre"]

        print(f"{'='*56}")
        print(f"  {nombre} — {caso_cfg['descripcion']}")
        print(f"  {n_cars} carros | {n_ticks} ticks max")
        print(f"{'='*56}")

        t0 = _time.perf_counter()
        history, routes = run_caso(graph, n_cars, n_ticks)

        if not history:
            print(f"  ⚠ Sin resultados para {nombre}\n")
            continue

        output = OUTPUT_DIR / f"caso{caso_id}_vis.html"
        vis    = build_vis(graph, history, routes, caso_cfg)
        output.write_text(vis, encoding="utf-8")

        elapsed = _time.perf_counter() - t0
        ticks_detenidos = sum(1 for f in history if (f.get("stopped") or 0) > 0)
        llegaron_todos  = history[-1].get("arrived", 0) == n_cars

        print(f"\n  ✓ {output.name}")
        print(f"  Duracion:        {int(elapsed//60)}m {elapsed%60:.1f}s")
        print(f"  Ticks totales:   {len(history)}")
        print(f"  Ticks detenidos: {ticks_detenidos} ({ticks_detenidos/len(history)*100:.0f}% del tiempo)")
        print(f"  Todos llegaron:  {'✓ Sí' if llegaron_todos else '✗ No (aumentar n_ticks)'}\n")

        resultados.append({
            "caso": caso_id, "nombre": nombre, "n_cars": n_cars,
            "ticks": len(history), "ticks_detenidos": ticks_detenidos,
            "llegaron": llegaron_todos,
        })

    # Resumen comparativo
    if len(resultados) > 1:
        print(f"{'='*56}")
        print("  RESUMEN COMPARATIVO")
        print(f"{'─'*56}")
        print(f"  {'Caso':<8} {'Carros':<8} {'Ticks':<8} {'Detenidos':<12} {'% det':<8} {'Llegaron'}")
        print(f"  {'─'*50}")
        for r in resultados:
            pct = r['ticks_detenidos']/r['ticks']*100 if r['ticks'] else 0
            llegaron = '✓' if r['llegaron'] else '✗'
            print(f"  {r['caso']:<8} {r['n_cars']:<8} {r['ticks']:<8} "
                  f"{r['ticks_detenidos']:<12} {pct:<8.0f} {llegaron}")
        print(f"{'='*56}\n")


if __name__ == "__main__":
    main()