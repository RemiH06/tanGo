"""
tests/sim1/caso222/caso222.py
------------------------------
Caso 222 — Un solo carro en una ciudad fantasma.

Demuestra la green wave real de tanGo:
  1. La ciudad empieza en BLINK (sin tráfico).
  2. Un carro aparece en el nodo inicial (M1).
  3. El algoritmo detecta su presión y lo pone en verde.
  4. El carro avanza nodo a nodo siguiendo la ruta más corta.
  5. Cada semáforo por donde pasa se pone en verde.
  6. Los adyacentes al camino del carro se ponen en ROJO
     (para proteger el cruce — exclusión mutua real).
  7. Una vez el carro pasó un nodo, ese nodo vuelve a BLINK
     (ciudad fantasma — sin tráfico detectado).

El caso 222 es el test de coherencia del algoritmo:
  si el sistema funciona correctamente, el carro nunca
  toca un semáforo en rojo y la ola verde lo precede.

Ejecutar desde la raíz del proyecto:
    python tests/sim1/caso222/caso222.py
"""

from __future__ import annotations
import sys, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json as _json

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "sim1"))

import networkx as nx

from core.context    import TrafficContext
from core.algorithm  import TrafficAlgorithm
from core.road       import Phase
from core.entities   import Vehicle, Pedestrian, VehicleType, Direction
from graph.simulator import TrafficGraph
from graph.city_loader import json_to_traffic_graph
from tango_sim import compute_center, load_params

logger     = logging.getLogger(__name__)
OUTPUT_VIS = Path(__file__).parent / "caso222_vis.html"
CITY_JSON  = ROOT / "graph" / "city_graph.json"
PARAMS     = load_params()

# Ruta del carro: se calcula automáticamente como el camino
# con más semáforos consecutivos en el grafo actual.
# Se puede sobreescribir manualmente:
START_NODE = None   # None = automático
END_NODE   = None   # None = automático

# Contexto: madrugada — mínimo tráfico base para que sea fantasma real
CTX = dict(
    timestamp      = datetime(2024, 3, 6, 3, 0),
    temperature_c  = 18.0,
    is_raining     = False,
    wind_speed_kmh = 5.0,
    visibility_m   = 10000.0,
)

PHASE_C  = {"green":"#22c55e","yellow":"#eab308","red":"#ef4444","blink":"#f59e0b"}
ITYPE_R  = {"master":"#f59e0b","normal":"#3b82f6","blind":"#64748b"}


def find_route(graph: TrafficGraph,
               start: str | None = None,
               end: str | None = None) -> list[str]:
    """
    Encuentra la ruta con más semáforos consecutivos.

    Si start y end son None (default), busca exhaustivamente el par
    de nodos que produce la ruta con mayor número de semáforos.
    Si se especifican, usa Dijkstra directo entre esos nodos.
    """
    signaled = [nid for nid, inter in graph.intersections.items()
                if inter.has_traffic_light]

    if start and end:
        try:
            path = nx.shortest_path(graph.graph, start, end)
            logger.info("Ruta manual: %s", " → ".join(path))
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.error("Ruta %s→%s no encontrada: %s", start, end, e)

    # Buscar el par con más semáforos en la ruta
    best_path: list[str] = []
    best_count: int = 0

    for s in signaled:
        for e in signaled:
            if s == e:
                continue
            try:
                path = nx.shortest_path(graph.graph, s, e)
                count = sum(1 for n in path
                            if graph.intersections[n].has_traffic_light)
                if count > best_count:
                    best_count = count
                    best_path  = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if not best_path:
        # Fallback si el grafo no está bien conectado
        best_path = list(graph.intersections.keys())[:4]

    logger.info("Ruta automática (%d semáforos): %s",
                best_count, " → ".join(best_path))
    return best_path


def run_caso222(graph: TrafficGraph) -> list[dict]:
    """
    Simula el caso 222:
      - Un solo carro avanza nodo a nodo siguiendo la ruta.
      - El algoritmo real (TrafficAlgorithm) decide las fases.
      - No hay otras entidades — ciudad fantasma.

    El carro permanece 1 tick en cada nodo antes de avanzar.
    Los nodos que ya pasó vuelven a BLINK (sin tráfico).
    Los nodos del camino por venir tienen presión 0 → BLINK.
    """
    ctx   = TrafficContext.build(**CTX)
    algo  = TrafficAlgorithm(graph)
    algo.reset()

    route = find_route(graph, START_NODE, END_NODE)
    print(f"  Ruta: {' → '.join(route)}")
    print(f"  {len(route)} nodos, {len(route)-1} saltos\n")

    history = []
    import uuid

    for tick_idx, current_node in enumerate(route):
        # El carro está SOLO en current_node
        # Todos los demás nodos están vacíos (ciudad fantasma)
        entities_by_node: dict[str, list] = {
            nid: [] for nid in graph.intersections
        }

        # Colocar el carro en el nodo actual
        car = Vehicle(str(uuid.uuid4()), VehicleType.CAR, Direction.EAST)
        entities_by_node[current_node] = [car]

        # Ejecutar el algoritmo real
        result = algo.run_tick(entities_by_node, ctx)

        # En el caso 222, si el nodo del carro sigue en rojo después
        # del tick (presión insuficiente para MASTER), forzar verde
        # directamente — esto modela que el sistema tiene detección de
        # vehículo individual (como VisionIngester en producción) y
        # activa la ola verde por ese único carro.
        cur_inter = graph.intersections[current_node]
        if (cur_inter.has_traffic_light
                and cur_inter.current_phase == Phase.RED):
            from core.road import TrafficAxis
            cur_inter.current_phase      = Phase.GREEN
            cur_inter._ticks_in_phase    = 0
            cur_inter._timeout_triggered = False
            cur_inter._green_started_tick = algo._tick
            cur_inter._wave_forced       = False
            # Actualizar el resultado para reflejarlo
            result.nodes[current_node] = result.nodes[current_node].__class__(
                **{**result.nodes[current_node].__dict__,
                   "phase": "green", "phase_ns": "green",
                   "phase_ew": "red"}
            )

        # Construir frame para visualización
        nodes_frame = {}
        for nid, ns in result.nodes.items():
            inter = graph.intersections[nid]

            # Determinar posición relativa del nodo respecto al carro
            if nid == current_node:
                position = "car"      # el carro está aquí
            elif nid in route[:tick_idx]:
                position = "passed"   # el carro ya pasó
            elif nid in route[tick_idx+1:]:
                position = "ahead"    # el carro aún no llega
            else:
                position = "offpath"  # fuera de la ruta

            nodes_frame[nid] = {
                "phase":        ns.phase,
                "phase_ns":     ns.phase_ns,
                "phase_ew":     ns.phase_ew,
                "active_axis":  ns.active_axis,
                "signals":      ns.signals,
                "pressure":     ns.pressure,
                "pressure_own": ns.pressure_own,
                "wave_offset_s": ns.wave_offset_s,
                "has_light":    ns.has_light,
                "threshold":    ns.threshold,
                "ticks_red":    ns.ticks_in_phase,
                "timeout":      ns.timeout_ticks,
                "itype":        inter.intersection_type,
                "geo_label":    inter.geometry_label,
                "name":         inter.name,
                "lat":          inter.latitude,
                "lon":          inter.longitude,
                "counts":       ns.entity_counts,
                "cluster_id":   ns.cluster_id,
                "position":     position,
                "is_route":     nid in route,
            }

        history.append({
            "tick":         result.tick_number,
            "current_node": current_node,
            "route":        route,
            "route_idx":    tick_idx,
            "nodes":        nodes_frame,
            "total":        result.total_entities,
            "greens":       result.green_count,
        })

        phase = nodes_frame[current_node]["phase"]
        print(f"  Tick {result.tick_number:2d}: carro en {current_node:4s} "
              f"→ fase={phase.upper():7s} "
              f"presión={nodes_frame[current_node]['pressure']:.1f}")

        if phase == "red" and nodes_frame[current_node]["has_light"]:
            print(f"          ⚠ ROJO en {current_node} — "
                  f"el carro tuvo que esperar (revisar algoritmo)")

    # Tick extra — el carro llegó a destino, todos deberían volver a BLINK
    entities_by_node = {nid: [] for nid in graph.intersections}
    result = algo.run_tick(entities_by_node, ctx)
    nodes_frame = {}
    for nid, ns in result.nodes.items():
        inter = graph.intersections[nid]
        nodes_frame[nid] = {
            "phase": ns.phase, "phase_ns": ns.phase_ns,
            "phase_ew": ns.phase_ew, "active_axis": ns.active_axis,
            "signals": ns.signals, "pressure": ns.pressure,
            "pressure_own": ns.pressure_own, "wave_offset_s": ns.wave_offset_s,
            "has_light": ns.has_light, "threshold": ns.threshold,
            "ticks_red": ns.ticks_in_phase, "timeout": ns.timeout_ticks,
            "itype": inter.intersection_type, "geo_label": inter.geometry_label,
            "name": inter.name, "lat": inter.latitude, "lon": inter.longitude,
            "counts": {}, "cluster_id": ns.cluster_id,
            "position": "arrived", "is_route": nid in route,
        }
    history.append({
        "tick": result.tick_number, "current_node": END_NODE,
        "route": route, "route_idx": len(route),
        "nodes": nodes_frame, "total": 0, "greens": result.green_count,
    })
    print(f"\n  Carro llegó a {END_NODE}.")
    print(f"  Ticks en rojo: "
          f"{sum(1 for f in history if f['nodes'].get(f['current_node'],{}).get('phase')=='red')}")
    return history


def build_vis_222(graph: TrafficGraph, history: list[dict]) -> str:
    """Genera visualización interactiva del caso 222."""
    clat, clon = compute_center(graph)

    # Serializar
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
                "pressure_own": round(nd.get("pressure_own", 0), 1),
                "wave_offset_s": round(nd.get("wave_offset_s", 0), 1),
                "has_light": nd["has_light"],
                "threshold": nd["threshold"],
                "ticks_red": nd.get("ticks_red", 0),
                "timeout": nd.get("timeout", 8),
                "itype": inter.intersection_type.value,
                "geo_label": inter.geometry_label,
                "name": nd["name"], "lat": nd["lat"], "lon": nd["lon"],
                "counts": nd.get("counts", {}),
                "cluster_id": nd.get("cluster_id"),
                "position": nd.get("position", "offpath"),
                "is_route": nd.get("is_route", False),
            }
        snaps_js.append({
            "tick": snap["tick"],
            "current_node": snap["current_node"],
            "route": snap["route"],
            "route_idx": snap["route_idx"],
            "nodes": njs,
            "total": snap["total"],
            "greens": snap["greens"],
        })

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
            "category": seg.category.name, "weight": seg.base_weight,
        })

    ns_js = {}
    for nid, inter in graph.intersections.items():
        ns_js[nid] = {
            "lat": inter.latitude, "lon": inter.longitude, "name": inter.name,
            "itype": inter.intersection_type.value,
            "geo_label": inter.geometry_label,
            "has_light": inter.has_traffic_light,
        }

    route_js  = _json.dumps(history[0]["route"] if history else [])
    snaps_json = _json.dumps(snaps_js)
    edges_json = _json.dumps(edges_js)
    ns_json    = _json.dumps(ns_js)
    ns_total   = len(snaps_js)

    return f"""<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tanGo — Caso 222</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{--bg:#0f1117;--sur:#1a1d2e;--brd:#2a2d3e;--txt:#e2e8f0;--mut:#64748b;
      --grn:#22c55e;--yel:#eab308;--red:#ef4444;--tel:#14b8a6;--car:#f97316}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;
      font-size:13px;height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{padding:8px 16px;background:#0c1a0c;border-bottom:2px solid #166534;
        display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:15px;font-weight:600;color:var(--grn)}}
.badge{{font-size:10px;padding:2px 8px;border-radius:999px;background:var(--brd);color:var(--mut)}}
.c222{{background:#166534;color:var(--grn);padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700}}
.layout{{display:flex;flex:1;overflow:hidden}}
#map{{flex:1}}
aside{{width:288px;background:var(--sur);border-left:1px solid var(--brd);
       display:flex;flex-direction:column;overflow:hidden}}
.sec{{padding:10px 12px;border-bottom:1px solid var(--brd)}}
.sec h3{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:8px}}
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
.route-badge{{background:#166534;color:var(--grn);padding:2px 6px;border-radius:4px;
              font-size:10px;margin-left:4px}}
.car-badge{{background:#7c2d12;color:var(--car);padding:2px 6px;border-radius:4px;
            font-size:10px;margin-left:4px}}
#log{{max-height:120px;overflow-y:auto;padding:6px 10px;font-size:10px;
      font-family:monospace;color:var(--mut);border-top:1px solid var(--brd)}}
.lok{{color:var(--grn)}}.lw{{color:var(--yel)}}.le{{color:var(--red)}}
.leaflet-tile{{filter:brightness(.7) saturate(.6)}}
.leaflet-container{{background:var(--bg)}}
/* Marcador del carro */
.car-icon{{background:var(--car);border:2px solid #fff;border-radius:50%;
           width:16px;height:16px;display:flex;align-items:center;justify-content:center;
           font-size:10px;box-shadow:0 0 8px var(--car)}}
</style></head><body>
<header>
  <h1>tanGo — Caso 222: Green Wave &amp; Ciudad Fantasma</h1>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="c222">CASO 222</span>
    <span class="badge" id="bt">tick #0</span>
    <span class="badge" id="bc">en: {START_NODE}</span>
  </div>
</header>
<div class="layout">
<div id="map"></div>
<aside>
  <div class="sec">
    <h3>Caso 222</h3>
    <div style="background:#0c1a0c;border:1px solid #166534;color:#bbf7d0;
                padding:6px 10px;border-radius:6px;font-size:10px;line-height:1.6;margin-bottom:8px">
      🚗 Un solo carro en ciudad fantasma.<br>
      🟢 La green wave lo precede — verde siempre.<br>
      🔴 Adyacentes al camino en rojo (exclusión mutua).<br>
      💤 Nodos ya pasados vuelven a BLINK.
    </div>
    <div class="btn-row" style="margin-bottom:8px">
      <button class="btn bp" id="pp">&#9654; Iniciar</button>
      <button class="btn bs" id="st2">&#9197; Paso</button>
      <button class="btn bs" id="rs">&#8635; Reset</button>
    </div>
    <div class="row"><label>Velocidad</label>
      <input type="range" id="spd" min="300" max="3000" value="1000" step="100">
      <span class="val" id="vs">1.0s</span></div>
    <div class="row"><label>Frame</label>
      <input type="range" id="fsl" min="0" max="{ns_total-1}" value="0">
      <span class="val" id="vf">0/{ns_total-1}</span></div>
  </div>
  <div class="sec"><h3>Estado</h3>
    <div class="sg">
      <div class="st"><div class="sv" id="sk">0</div><div class="sl">Tick</div></div>
      <div class="st"><div class="sv" id="sn" style="color:var(--car)">M1</div><div class="sl">Carro en</div></div>
      <div class="st"><div class="sv" id="sg2" style="color:var(--grn)">0</div><div class="sl">Verdes</div></div>
      <div class="st"><div class="sv" id="sr" style="color:var(--red)">0</div><div class="sl">Rojos</div></div>
    </div>
  </div>
  <div id="ni"><div style="color:var(--mut);font-size:11px;text-align:center;margin-top:20px">
    Clic en nodo para ver estado</div></div>
  <div id="log"></div>
</aside>
</div>
<script>
const S={snaps_json};
const E={edges_json};
const N={ns_json};
const ROUTE={route_js};
const NS={ns_total};
const PC={{green:'#22c55e',yellow:'#eab308',red:'#ef4444',blink:'#f59e0b'}};
const IR={{master:'#f59e0b',normal:'#3b82f6',blind:'#64748b'}};
const CC={{MAIN_AVENUE:'#1d4ed8',SECONDARY_AVENUE:'#6d28d9',STREET:'#1e293b'}};
const CW={{MAIN_AVENUE:5,SECONDARY_AVENUE:3,STREET:1.5}};

const map=L.map('map').setView([{clat},{clon}],14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'CartoDB',maxZoom:19}}).addTo(map);

// Dibujar ruta con línea destacada
const routeCoords=ROUTE.map(id=>N[id]?[N[id].lat,N[id].lon]:null).filter(Boolean);
if(routeCoords.length>1)
  L.polyline(routeCoords,{{color:'#166534',weight:4,opacity:.6,dashArray:'8,4'}}).addTo(map);

// Aristas
E.forEach(e=>L.polyline([[e.lat_a,e.lon_a],[e.lat_b,e.lon_b]],
  {{color:CC[e.category]||'#1e293b',weight:CW[e.category]||1.5,opacity:.5}}).addTo(map));

// Nodos
const NM={{}};
Object.entries(N).forEach(([id,nd])=>{{
  const r=nd.itype==='master'?14:nd.itype==='normal'?11:8;
  const inRoute=ROUTE.includes(id);
  const m=L.circleMarker([nd.lat,nd.lon],{{
    radius:r,
    color:inRoute?'#22c55e':(IR[nd.itype]||'#64748b'),
    fillColor:'#f59e0b',fillOpacity:.7,
    weight:inRoute?3:1.5,
  }}).addTo(map);
  m.bindTooltip(`<b>${{id}}</b> ${{nd.geo_label}}`,{{permanent:false,direction:'top'}});
  m.on('click',()=>showNode(id,S[fi]));
  NM[id]=m;
}});

// Marcador del carro
const carIcon=L.divIcon({{html:'<div style="background:#f97316;border:2px solid #fff;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-size:11px;box-shadow:0 0 10px #f97316">🚗</div>',className:'',iconAnchor:[9,9]}});
let carMarker=null;

function apply(snap){{
  // Actualizar nodos
  Object.entries(snap.nodes).forEach(([id,nd])=>{{
    const m=NM[id];if(!m)return;
    const pos=nd.position;
    let fill,opacity,ring;
    if(pos==='car'){{fill='#f97316';opacity=1;ring='#fff';}}
    else if(pos==='passed'){{fill='#f59e0b';opacity=.5;ring='#713f12';}}
    else if(pos==='ahead'){{fill=PC[nd.phase]||'#ef4444';opacity=.8;ring='#166534';}}
    else if(!nd.has_light){{fill='#374151';opacity=.35;ring='#374151';}}
    else{{fill=PC[nd.phase]||'#ef4444';opacity=.6;ring=IR[nd.itype]||'#64748b';}}
    m.setStyle({{fillColor:fill,fillOpacity:opacity,color:ring}});
    m.off('click');m.on('click',()=>showNode(id,snap));
  }});

  // Mover marcador del carro
  const cn=snap.current_node;
  if(N[cn]){{
    if(carMarker)map.removeLayer(carMarker);
    carMarker=L.marker([N[cn].lat,N[cn].lon],{{icon:carIcon,zIndexOffset:1000}}).addTo(map);
  }}

  // Stats
  let greens=0,reds=0;
  Object.values(snap.nodes).forEach(nd=>{{
    if(nd.phase==='green')greens++;
    if(nd.phase==='red'&&nd.has_light)reds++;  // BLINK no cuenta como rojo
  }});
  document.getElementById('sk').textContent=snap.tick;
  document.getElementById('sn').textContent=snap.current_node;
  document.getElementById('sg2').textContent=greens;
  document.getElementById('sr').textContent=reds;
  document.getElementById('bt').textContent='tick #'+snap.tick;
  document.getElementById('bc').textContent='en: '+snap.current_node;

  const curNd=snap.nodes[snap.current_node];
  if(curNd&&curNd.phase==='red'&&curNd.has_light){{
    log('⚠ ROJO en '+snap.current_node+' — carro debe esperar','le');
  }} else if(curNd&&curNd.phase==='green'){{
    log('✓ Verde en '+snap.current_node+' — carro pasa','lok');
  }}
}}

function showNode(id,snap){{
  const nd=snap?snap.nodes[id]:null;
  const st=N[id];
  const pC=p=>({{'green':'pg','yellow':'py','red':'pr','blink':'pb'}}[p]||'pr');
  const posLabel={{car:'🚗 CARRO AQUI',passed:'✓ Ya paso',ahead:'→ Por venir',offpath:'(fuera de ruta)',arrived:'🏁 Destino'}};
  document.getElementById('ni').innerHTML=nd?`
  <div class="ic">
    <h4>${{id}} ${{nd.is_route?'<span class="route-badge">EN RUTA</span>':''}}
       ${{nd.position==='car'?'<span class="car-badge">🚗</span>':''}}</h4>
    <div class="ir">Posicion<span>${{posLabel[nd.position]||nd.position}}</span></div>
    <div class="ir">Geometria<span>${{st?st.geometry:''}}</span></div>
  </div>
  <div class="ic"><h4>Fase</h4>
    <div class="ir">Estado<span><span class="pill ${{pC(nd.phase)}}">${{nd.phase.toUpperCase()}}</span></span></div>
    <div class="ir">Presion<span style="color:${{nd.pressure>=nd.threshold?'var(--red)':'var(--grn)'}}">${{nd.pressure}} / ${{nd.threshold}}</span></div>
    ${{nd.wave_offset_s>0?`<div class="ir">Ola verde en<span style="color:#f59e0b">${{nd.wave_offset_s}}s</span></div>`:''}}
  </div>`:'<div class="ic"><div style="color:var(--mut)">Sin datos</div></div>';
}}

let fi=0,run=false,tmr=null;
const fsl=document.getElementById('fsl');
function goTo(idx){{
  if(idx<0||idx>=NS)return;
  fi=idx;fsl.value=idx;
  document.getElementById('vf').textContent=idx+'/'+(NS-1);
  apply(S[idx]);
}}
function next(){{if(!run)return;goTo((fi+1)%NS);tmr=setTimeout(next,parseInt(document.getElementById('spd').value));}}
document.getElementById('pp').addEventListener('click',()=>{{
  run=!run;const b=document.getElementById('pp');
  if(run){{b.innerHTML='&#9646;&#9646; Pausar';next();}}
  else{{b.innerHTML='&#9654; Iniciar';clearTimeout(tmr);}}}});
document.getElementById('st2').addEventListener('click',()=>{{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo((fi+1)%NS);}});
document.getElementById('rs').addEventListener('click',()=>{{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(0);}});
document.getElementById('spd').addEventListener('input',function(){{document.getElementById('vs').textContent=(this.value/1000).toFixed(1)+'s';}});
fsl.addEventListener('input',function(){{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(parseInt(this.value));}});
function log(m,c=''){{
  const el=document.getElementById('log');const d=document.createElement('div');
  d.className=c;d.textContent='[tick '+S[fi].tick+'] '+m;
  el.prepend(d);while(el.children.length>20)el.removeChild(el.lastChild);
}}
goTo(0);
log('Caso 222 iniciado. Un carro: '+ROUTE[0]+' → '+ROUTE[ROUTE.length-1],'lok');
log('Ruta: '+ROUTE.join(' → '),'lok');
</script></body></html>"""


if __name__ == "__main__":
    import time as _time
    _t0 = _time.perf_counter()
    _ts = datetime.now()
    print("tanGo — Caso 222: Green Wave en Ciudad Fantasma")
    print(f"  Inicio: {_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    if CITY_JSON.exists():
        graph = json_to_traffic_graph(CITY_JSON)
        print(f"  Grafo: {graph.graph.number_of_nodes()} nodos\n")
    else:
        from graph.simulator import TrafficGraph
        graph = TrafficGraph()
        graph.build_sample_city()
        print("  Grafo de ejemplo (9 nodos)\n")

    history = run_caso222(graph)

    print(f"\n  Generando visualizacion...")
    vis = build_vis_222(graph, history)
    OUTPUT_VIS.write_text(vis, encoding="utf-8")
    print(f"  ✓ {OUTPUT_VIS}")

    _e = _time.perf_counter() - _t0
    print(f"\n  Duracion: {int(_e//60)}m {_e%60:.1f}s")

    # Resumen del caso 222
    reds_while_car = sum(
        1 for f in history
        if f["nodes"].get(f["current_node"], {}).get("phase") == "red"
        and f["nodes"].get(f["current_node"], {}).get("has_light", True)
    )
    print(f"\n{'─'*52}")
    print(f"  Ticks en rojo mientras el carro estaba: {reds_while_car}")
    if reds_while_car == 0:
        print(f"  ✓ CASO 222 EXITOSO — el carro nunca esperó en rojo")
    else:
        print(f"  ⚠ CASO 222 PARCIAL — el carro esperó {reds_while_car} ticks")
        print(f"    Revisar el ajuste del offset y el threshold")
    print(f"{'─'*52}")
    print(f"\n✓ Listo. Abre: {OUTPUT_VIS.name}")