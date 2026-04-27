"""
tests/sim2/tango_sim2.py
-------------------------
Simulación 2 — pesos estáticos por nodo + movimiento real de entidades.

Novedades respecto a sim1:
  1. Entidades con origen-destino real (Dijkstra ponderado).
  2. Movimiento tick a tick con velocidad individual por tipo y vía.
  3. Lifetime genuino — entidades persisten mientras viajan.
  4. node_weight combinado (degree + betweenness + pagerank + road_quality).
  5. Partículas animadas en el mapa mostrando el movimiento.
  6. Heatmap de flujo acumulado por nodo.

El algoritmo de semáforos es el mismo TrafficAlgorithm de sim1 —
la diferencia está en cómo se generan y mueven las entidades.

Ejecutar:
    python tests/sim2/tango_sim2.py
"""

from __future__ import annotations
import sys, logging, json as _json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time as _time

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import folium

from core.context    import TrafficContext
from core.algorithm  import TrafficAlgorithm, TICK_DURATION_S
from core.movement   import MovementEngine
from core.road       import Phase, IntersectionType
from core.entities   import Vehicle, Pedestrian
from graph.simulator import TrafficGraph
from graph.city_loader import json_to_traffic_graph

logger     = logging.getLogger(__name__)
OUTPUT_VIS = Path(__file__).parent / "tango_vis_sim2.html"
CITY_JSON  = ROOT / "graph" / "city_graph.json"
PARAMS_FILE = Path(__file__).parent / "sim_params.json"

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")


def load_params() -> dict:
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE, encoding="utf-8") as f:
            return _json.load(f)
    return {"n_ticks": 60, "scenarios": [], "movement": {}}


def compute_center(graph: TrafficGraph) -> tuple[float, float]:
    nodes = list(graph.intersections.values())
    if not nodes:
        return 20.6656, -103.3863
    return (sum(n.latitude  for n in nodes) / len(nodes),
            sum(n.longitude for n in nodes) / len(nodes))


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def simulate(scenario: dict, graph: TrafficGraph,
             n_ticks: int, params: dict) -> list[dict]:
    """
    Simula n_ticks con:
      - TrafficAlgorithm (mismo que sim1)
      - MovementEngine con Dijkstra y velocidades individuales
    """
    sc = {k: v for k, v in scenario.items() if k != "label"}
    if "timestamp" in sc and isinstance(sc["timestamp"], str):
        sc["timestamp"] = datetime.fromisoformat(sc["timestamp"])
    ctx = TrafficContext.build(**sc)

    mov_params = params.get("movement", {})
    movement = MovementEngine(
        graph,
        spawn_rate   = mov_params.get("spawn_rate",   15),
        max_entities = mov_params.get("max_entities", 300),
    )

    algo = TrafficAlgorithm(graph)
    algo.reset()

    history = []
    arrived_total = 0
    travel_times  = []

    for tick_n in range(n_ticks):
        # Obtener fases actuales para que Dijkstra las evite
        current_phases = {
            nid: inter.current_phase.value
            for nid, inter in graph.intersections.items()
        }

        # Mover entidades — genera entities_by_node
        entities_by_node = movement.tick(ctx, current_phases)

        # Ejecutar algoritmo de semáforos
        result = algo.run_tick(entities_by_node, ctx)

        # Estadísticas de movimiento
        stats  = movement.get_stats()
        particles = movement.get_particles()
        heatmap   = movement.get_heatmap()

        # Contar llegadas para estadísticas
        arrived_total += stats.arrived
        if stats.avg_travel_ticks > 0:
            travel_times.append(stats.avg_travel_ticks)

        # Construir frame
        nodes_frame = {}
        for nid, ns in result.nodes.items():
            inter = graph.intersections[nid]
            nodes_frame[nid] = {
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
                # Nuevo en sim2:
                "node_weight":  round(inter.node_weight, 3),
                "heat":         round(heatmap.get(nid, 0.0), 3),
                "static_weight": getattr(inter, "static_weight", {}),
            }

        history.append({
            "tick":          result.tick_number,
            "nodes":         nodes_frame,
            "flows":         result.flows,
            "particles":     particles,
            "heatmap":       heatmap,
            "total":         result.total_entities,
            "greens":        result.green_count,
            "yellows":       result.yellow_count,
            "reds":          result.red_count,
            "blinks":        result.blink_count,
            "active_moving": stats.active_entities,
            "arrived":       arrived_total,
            "cluster_sizes": {cid: len(mems)
                              for cid, mems in getattr(graph, "intersection_clusters", {}).items()},
        })

    avg_travel = (sum(travel_times) / len(travel_times)
                  if travel_times else 0.0)
    print(f"    Entidades llegaron: {arrived_total} | "
          f"Tiempo promedio: {avg_travel:.1f} ticks "
          f"({avg_travel * TICK_DURATION_S / 60:.1f} min)")

    return history


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

PHASE_C  = {"green":"#22c55e","yellow":"#eab308","red":"#ef4444","blink":"#f59e0b"}
ITYPE_R  = {"master":"#f59e0b","normal":"#3b82f6","blind":"#64748b"}


def build_vis(graph: TrafficGraph, all_histories: list[tuple],
              params: dict) -> str:
    """Genera tango_vis_sim2.html — partículas + heatmap + pesos estáticos."""
    clat, clon = compute_center(graph)
    vis_params = params.get("visualization", {})
    particle_colors = vis_params.get("particle_color", {})

    # Serializar snapshots
    snaps_js, sc_idx = [], {}
    for sc_label, history in all_histories:
        sc_idx[sc_label] = len(snaps_js)
        for snap in history:
            njs = {}
            for nid, nd in snap["nodes"].items():
                inter = graph.intersections[nid]
                sw = nd.get("static_weight", {})
                njs[nid] = {
                    "phase":        nd["phase"],
                    "phase_ns":     nd["phase_ns"],
                    "phase_ew":     nd["phase_ew"],
                    "active_axis":  nd["active_axis"],
                    "signals":      nd.get("signals", {}),
                    "pressure":     round(nd["pressure"], 1),
                    "pressure_own": round(nd.get("pressure_own", 0), 1),
                    "wave_offset_s": round(nd.get("wave_offset_s", 0), 1),
                    "has_light":    nd["has_light"],
                    "threshold":    nd["threshold"],
                    "ticks_red":    nd.get("ticks_red", 0),
                    "timeout":      nd.get("timeout", 8),
                    "itype":        inter.intersection_type.value,
                    "geo_label":    inter.geometry_label,
                    "name":         nd["name"],
                    "lat":          nd["lat"],
                    "lon":          nd["lon"],
                    "counts":       nd.get("counts", {}),
                    "cluster_id":   nd.get("cluster_id"),
                    "node_weight":  nd.get("node_weight", 1.0),
                    "heat":         nd.get("heat", 0.0),
                    "btw":          round(sw.get("betweenness", 1.0), 3),
                    "pr":           round(sw.get("pagerank", 1.0), 3),
                    "road_q":       round(sw.get("road_quality", 0.5), 3),
                }
            snaps_js.append({
                "scenario":      sc_label,
                "tick":          snap["tick"],
                "total":         snap["total"],
                "greens":        snap["greens"],
                "yellows":       snap.get("yellows", 0),
                "reds":          snap.get("reds", 0),
                "blinks":        snap.get("blinks", 0),
                "active_moving": snap.get("active_moving", 0),
                "arrived":       snap.get("arrived", 0),
                "cluster_sizes": snap.get("cluster_sizes", {}),
                "particles":     snap.get("particles", []),
                "heatmap":       snap.get("heatmap", {}),
                "nodes":         njs,
                "flows":         snap.get("flows", []),
            })

    # Aristas estáticas
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
            "length_m": seg.length_m, "speed_kmh": seg.speed_limit_kmh,
        })

    ns_js = {}
    for nid, inter in graph.intersections.items():
        ns_js[nid] = {
            "lat": inter.latitude, "lon": inter.longitude,
            "name": inter.name, "itype": inter.intersection_type.value,
            "geometry": inter.geometry.value, "geo_label": inter.geometry_label,
            "has_light": inter.has_traffic_light,
            "threshold": inter.pressure_threshold,
            "node_weight": round(inter.node_weight, 3),
        }

    sj  = _json.dumps(snaps_js)
    ej  = _json.dumps(edges_js)
    nj  = _json.dumps(ns_js)
    scj = _json.dumps(sc_idx)
    pcj = _json.dumps(particle_colors)
    ns  = len(snaps_js)
    btns = "".join(
        f'<button class="btn btn-exp" onclick="jumpTo({_json.dumps(l)})">{l[:28]}</button>\n'
        for l in sc_idx
    )

    return f"""<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tanGo sim2 — Pesos estaticos + Movimiento</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{--bg:#0f1117;--sur:#1a1d2e;--brd:#2a2d3e;--txt:#e2e8f0;--mut:#64748b;
      --grn:#22c55e;--yel:#eab308;--red:#ef4444;--tel:#14b8a6;--blue:#3b82f6}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;
      font-size:13px;height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{padding:8px 16px;background:#0c1117;border-bottom:2px solid var(--blue);
        display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:15px;font-weight:600}}
header h1 span{{color:var(--blue)}}
.sim2{{background:#1e3a5f;color:#93c5fd;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700}}
.badge{{font-size:10px;padding:2px 8px;border-radius:999px;background:var(--brd);color:var(--mut)}}
.layout{{display:flex;flex:1;overflow:hidden}}
#map{{flex:1;padding-bottom:46px}}
aside{{width:296px;background:var(--sur);border-left:1px solid var(--brd);
       display:flex;flex-direction:column;overflow:hidden}}
.sec{{padding:10px 12px;border-bottom:1px solid var(--brd)}}
.sec h3{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:8px}}
.row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
label{{font-size:11px;color:var(--mut);min-width:68px}}
input[type=range]{{flex:1;accent-color:var(--blue)}}
.val{{font-size:11px;min-width:36px;text-align:right}}
.btn{{padding:5px 10px;border:none;border-radius:6px;font-size:11px;font-weight:500;cursor:pointer}}
.bp{{background:var(--blue);color:#fff}}.bs{{background:var(--brd);color:var(--txt)}}
.btn-exp{{background:#1e3a5f;color:#93c5fd;margin:2px;font-size:10px;padding:4px 8px;
          border-radius:5px;border:1px solid var(--blue)}}
.btn-row{{display:flex;gap:5px;flex-wrap:wrap}}
.sg{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.st{{background:var(--bg);border-radius:6px;padding:7px;border:1px solid var(--brd)}}
.sv{{font-size:17px;font-weight:700;color:var(--blue)}}.sl{{font-size:9px;color:var(--mut)}}
#ni{{flex:1;overflow-y:auto;padding:10px}}
.ic{{background:var(--bg);border:1px solid var(--brd);border-radius:7px;padding:10px;margin-bottom:7px}}
.ic h4{{font-size:12px;font-weight:600;margin-bottom:5px}}
.ir{{display:flex;justify-content:space-between;font-size:11px;color:var(--mut);margin:2px 0}}
.ir span{{color:var(--txt)}}
.pill{{display:inline-block;padding:2px 7px;border-radius:999px;font-size:10px;font-weight:700}}
.pg{{background:#166534;color:var(--grn)}}.py{{background:#713f12;color:var(--yel)}}
.pr{{background:#7f1d1d;color:var(--red)}}.pb{{background:#1e293b;color:#f59e0b}}
.wbar{{height:6px;border-radius:3px;background:var(--brd);margin:4px 0}}
.wfill{{height:100%;border-radius:3px;background:var(--blue)}}
#log{{max-height:100px;overflow-y:auto;padding:6px 10px;font-size:10px;
      font-family:monospace;color:var(--mut);border-top:1px solid var(--brd)}}
.lok{{color:var(--grn)}}.lw{{color:var(--yel)}}.le{{color:var(--red)}}
#tb{{position:fixed;bottom:0;left:0;right:296px;background:var(--sur);
     border-top:1px solid var(--brd);padding:7px 14px;display:flex;gap:6px;
     align-items:center;flex-wrap:wrap;z-index:9999}}
#tb .lbl{{font-size:10px;color:var(--mut);font-weight:700;text-transform:uppercase;letter-spacing:.06em}}
.leaflet-tile{{filter:brightness(.7) saturate(.6)}}
.leaflet-container{{background:var(--bg)}}
</style></head><body>
<header>
  <h1>tan<span>Go</span> — <span style="color:var(--blue)">sim2</span> Pesos estaticos + Movimiento</h1>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="sim2">SIM2</span>
    <span class="badge" id="bt">tick #0</span>
    <span class="badge" id="bsc">—</span>
    <span class="badge" id="bm">0 en movimiento</span>
  </div>
</header>
<div class="layout">
<div id="map"></div>
<aside>
  <div class="sec">
    <h3>Reproduccion</h3>
    <div class="btn-row" style="margin-bottom:8px">
      <button class="btn bp" id="pp">&#9654; Iniciar</button>
      <button class="btn bs" id="st2">&#9197; Paso</button>
      <button class="btn bs" id="rs">&#8635; Reset</button>
    </div>
    <div class="row"><label>Velocidad</label>
      <input type="range" id="spd" min="200" max="3000" value="700" step="100">
      <span class="val" id="vs">0.7s</span></div>
    <div class="row"><label>Frame</label>
      <input type="range" id="fsl" min="0" max="{ns-1}" value="0">
      <span class="val" id="vf">0/{ns-1}</span></div>
    <div class="row" style="margin-top:6px">
      <label style="font-size:10px">Heatmap</label>
      <input type="checkbox" id="heat-toggle" checked>
      <label style="font-size:10px;min-width:auto">Particulas</label>
      <input type="checkbox" id="part-toggle" checked>
    </div>
  </div>
  <div class="sec"><h3>Estadisticas</h3>
    <div class="sg">
      <div class="st"><div class="sv" id="sk">0</div><div class="sl">Tick</div></div>
      <div class="st"><div class="sv" id="se">0</div><div class="sl">Entidades</div></div>
      <div class="st"><div class="sv" id="sg2" style="color:var(--grn)">0</div><div class="sl">Verde</div></div>
      <div class="st"><div class="sv" id="sy" style="color:var(--yel)">0</div><div class="sl">Amarillo</div></div>
      <div class="st"><div class="sv" id="sr" style="color:var(--red)">0</div><div class="sl">Rojo</div></div>
      <div class="st"><div class="sv" id="sblink" style="color:#f59e0b">0</div><div class="sl">Blink</div></div>
      <div class="st"><div class="sv" id="sm" style="color:var(--blue)">0</div><div class="sl">En ruta</div></div>
      <div class="st"><div class="sv" id="sa" style="color:var(--tel)">0</div><div class="sl">Llegaron</div></div>
      <div class="st"><div class="sv" id="sn">{len(ns_js)}</div><div class="sl">Nodos</div></div>
    </div>
  </div>
  <div id="ni"><div style="color:var(--mut);font-size:11px;text-align:center;margin-top:20px">
    Clic en nodo para ver pesos estaticos</div></div>
  <div id="log"></div>
</aside>
</div>
<div id="tb">
  <span class="lbl">Escenarios:</span>
  {btns}
  <span style="color:var(--brd);margin:0 4px">|</span>
  <span class="lbl" style="color:var(--blue)">sim2: Dijkstra + node_weight</span>
</div>
<script>
const S={sj},E={ej},N={nj},SI={scj},PC={pcj},NS={ns};
const PHASE_C={{green:'#22c55e',yellow:'#eab308',red:'#ef4444',blink:'#f59e0b'}};
const ITYPE_R={{master:'#f59e0b',normal:'#3b82f6',blind:'#64748b'}};
const CAT_C={{MAIN_AVENUE:'#1d4ed8',SECONDARY_AVENUE:'#6d28d9',STREET:'#1e293b',HIGHWAY:'#0f172a',ALLEY:'#111827'}};
const CAT_W={{MAIN_AVENUE:5,SECONDARY_AVENUE:3,STREET:1.5,HIGHWAY:6,ALLEY:1}};
let _blinkOn=true; setInterval(()=>{{_blinkOn=!_blinkOn;}},600);

const map=L.map('map').setView([{clat},{clon}],14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{attribution:'CartoDB',maxZoom:19}}).addTo(map);

// Aristas estaticas
E.forEach(e=>L.polyline([[e.lat_a,e.lon_a],[e.lat_b,e.lon_b]],
  {{color:CAT_C[e.category]||'#1e293b',weight:CAT_W[e.category]||1.5,opacity:.6}}).addTo(map));

// Nodos
const NM={{}};
Object.entries(N).forEach(([id,nd])=>{{
  const r=nd.itype==='master'?13:nd.itype==='normal'?10:7;
  const ringW=1+nd.node_weight*2;   // borde más grueso = mayor peso
  const m=L.circleMarker([nd.lat,nd.lon],{{radius:r,color:ITYPE_R[nd.itype]||'#64748b',
    fillColor:'#ef4444',fillOpacity:.88,weight:ringW}}).addTo(map);
  m.bindTooltip(`<b>${{id}}</b> ${{nd.geo_label}}<br>w=${{nd.node_weight}}`,{{permanent:false,direction:'top'}});
  m.on('click',()=>showNode(id,S[fi]));NM[id]=m;
}});

// Capas dinamicas
const FL={{}};  // flujos
const PM={{}};  // particulas — id → L.circleMarker
const HM={{}};  // heatmap  — nid → L.circleMarker

function clearFlows(){{Object.values(FL).forEach(([l,m])=>{{map.removeLayer(l);if(m)map.removeLayer(m);}});Object.keys(FL).forEach(k=>delete FL[k]);}}
function clearParticles(){{Object.values(PM).forEach(m=>map.removeLayer(m));Object.keys(PM).forEach(k=>delete PM[k]);}}
function clearHeatmap(){{Object.values(HM).forEach(m=>map.removeLayer(m));Object.keys(HM).forEach(k=>delete HM[k]);}}

function apply(snap){{
  const showHeat=document.getElementById('heat-toggle').checked;
  const showPart=document.getElementById('part-toggle').checked;

  // Actualizar nodos
  Object.entries(snap.nodes).forEach(([id,nd])=>{{
    const m=NM[id];if(!m)return;
    const st=N[id];
    const baseR=st.itype==='master'?13:st.itype==='normal'?10:7;
    const heatR=baseR+nd.heat*8;   // radio crece con el calor acumulado
    const fillCol=!nd.has_light?'#374151':nd.phase==='blink'?(_blinkOn?'#f59e0b':'#1e293b'):(PHASE_C[nd.phase]||'#ef4444');
    m.setStyle({{
      fillColor:fillCol,fillOpacity:!nd.has_light?.35:.88,
      radius:showHeat?heatR:baseR,
      weight:1+nd.node_weight*2,
    }});
    m.off('click');m.on('click',()=>showNode(id,snap));
  }});

  // Heatmap — círculos semi-transparentes proporcionales al calor
  clearHeatmap();
  if(showHeat){{
    Object.entries(snap.heatmap||{{}}).forEach(([nid,heat])=>{{
      if(heat<0.05)return;
      const nd=N[nid];if(!nd)return;
      const hm=L.circleMarker([nd.lat,nd.lon],{{
        radius:8+heat*20,color:'transparent',
        fillColor:'#f59e0b',fillOpacity:heat*0.35,
        interactive:false,
      }}).addTo(map);
      HM[nid]=hm;
    }});
  }}

  // Flujos
  clearFlows();
  snap.flows.forEach(fl=>{{
    const n=fl.fwd+fl.bwd;if(!n)return;
    const na=N[fl.from],nb=N[fl.to];if(!na||!nb)return;
    const fc=n>=15?'#ef4444':n>=8?'#f59e0b':'#22c55e';
    const ln=L.polyline([[na.lat,na.lon],[nb.lat,nb.lon]],{{color:fc,weight:Math.min(7,1.2+n*.28),opacity:.6}}).addTo(map);
    const mx=(na.lat+nb.lat)/2,my=(na.lon+nb.lon)/2;
    const lm=L.marker([mx,my],{{icon:L.divIcon({{html:`<div style="color:${{fc}};font-size:8px;font-weight:700;text-shadow:0 0 3px #000;white-space:nowrap">+${{fl.fwd}} -${{fl.bwd}}</div>`,className:'',iconAnchor:[16,5]}}),interactive:false}}).addTo(map);
    FL[fl.from+'-'+fl.to]=[ln,lm];
  }});

  // Partículas animadas
  clearParticles();
  if(showPart){{
    (snap.particles||[]).forEach(p=>{{
      const col=PC[p.vtype]||'#e2e8f0';
      const r=p.type==='pedestrian'?3:5;
      const pm=L.circleMarker([p.lat,p.lon],{{
        radius:r,color:'rgba(0,0,0,0.3)',
        fillColor:col,fillOpacity:.9,weight:1,
      }}).addTo(map);
      pm.bindTooltip(
        `${{p.vtype}} ${{p.speed_kmh}}km/h<br>${{p.origin}}→${{p.destination}}<br>${{p.progress.toFixed(0)}}%`,
        {{permanent:false,direction:'top'}}
      );
      PM[p.id]=pm;
    }});
  }}

  // Stats — blink separado de rojo
  const pc={{green:0,yellow:0,red:0,blink:0}};
  Object.values(snap.nodes).forEach(nd=>{{
    if(!nd.has_light||nd.phase==='blink') pc.blink++;
    else if(pc[nd.phase]!==undefined) pc[nd.phase]++;
  }});
  document.getElementById('sk').textContent=snap.tick;
  document.getElementById('se').textContent=snap.total;
  document.getElementById('sg2').textContent=pc.green;
  document.getElementById('sy').textContent=pc.yellow;
  document.getElementById('sr').textContent=pc.red;
  document.getElementById('sblink').textContent=pc.blink;
  document.getElementById('sm').textContent=snap.active_moving||0;
  document.getElementById('sa').textContent=snap.arrived||0;
  document.getElementById('bt').textContent='tick #'+snap.tick;
  document.getElementById('bsc').textContent=(snap.scenario||'').substring(0,18);
  document.getElementById('bm').textContent=(snap.active_moving||0)+' en movimiento';
}}

function showNode(id,snap){{
  const nd=snap?snap.nodes[id]:null;
  const st=N[id];
  const pC=p=>({{'green':'pg','yellow':'py','red':'pr','blink':'pb'}}[p]||'pr');
  const c=nd?nd.counts||{{}}:{{}};
  const nw=nd?nd.node_weight:st.node_weight;
  const nwPct=Math.round(nw*100);
  document.getElementById('ni').innerHTML=nd?`
  <div class="ic">
    <h4>${{id}} — ${{st.geo_label}} ${{nd.name}}</h4>
    <div class="ir">Tipo<span>${{(nd.itype||'').toUpperCase()}}</span></div>
    <div class="ir">Geometria<span>${{st.geometry}}</span></div>
  </div>
  <div class="ic">
    <h4>Pesos estaticos (sim2)</h4>
    <div class="ir">node_weight<span style="color:var(--blue);font-weight:700">${{nw}}</span></div>
    <div class="wbar"><div class="wfill" style="width:${{Math.min(100,nwPct)}}%"></div></div>
    <div class="ir" style="font-size:10px">Betweenness<span>${{nd.btw||'?'}}</span></div>
    <div class="ir" style="font-size:10px">PageRank<span>${{nd.pr||'?'}}</span></div>
    <div class="ir" style="font-size:10px">Road quality<span>${{nd.road_q||'?'}}</span></div>
    <div class="ir" style="font-size:10px">Calor acum.<span>${{nd.heat||0}}</span></div>
  </div>
  <div class="ic">
    <h4>Semaforo</h4>
    <div class="ir">Estado<span><span class="pill ${{pC(nd.phase)}}">${{nd.phase.toUpperCase()}}</span></span></div>
    <div class="ir">Eje N-S<span><span class="pill ${{pC(nd.phase_ns)}}">${{nd.phase_ns.toUpperCase()}}</span></span></div>
    <div class="ir">Eje E-O<span><span class="pill ${{pC(nd.phase_ew)}}">${{nd.phase_ew.toUpperCase()}}</span></span></div>
    <div class="ir">Presion<span style="color:${{nd.pressure>=nd.threshold?'var(--red)':'var(--tel)'}}">${{nd.pressure}}/${{nd.threshold}}</span></div>
    ${{nd.wave_offset_s>0?`<div class="ir">Ola verde en<span style="color:#f59e0b">${{nd.wave_offset_s}}s</span></div>`:''}}
  </div>
  <div class="ic">
    <h4>Entidades</h4>
    <div class="ir">Autos<span>${{c.CAR||0}}</span></div>
    <div class="ir">Motos<span>${{c.MOTORCYCLE||0}}</span></div>
    <div class="ir">Buses<span>${{c.BUS||0}}</span></div>
    <div class="ir">Peatones<span>${{c.PEDESTRIAN||0}}</span></div>
    <div class="ir" style="color:var(--red)">Emergencias<span>${{c.EMERGENCY||0}}</span></div>
  </div>`:'<div class="ic" style="color:var(--mut)">Sin datos — inicia la simulacion</div>';
}}

let fi=0,run=false,tmr=null;
const fsl=document.getElementById('fsl');
function goTo(idx){{
  if(idx<0||idx>=NS)return;fi=idx;fsl.value=idx;
  document.getElementById('vf').textContent=idx+'/'+(NS-1);
  apply(S[idx]);
}}
function next(){{if(!run)return;goTo((fi+1)%NS);tmr=setTimeout(next,parseInt(document.getElementById('spd').value));}}
document.getElementById('pp').addEventListener('click',()=>{{
  run=!run;const b=document.getElementById('pp');
  if(run){{b.innerHTML='&#9646;&#9646; Pausar';next();}}else{{b.innerHTML='&#9654; Iniciar';clearTimeout(tmr);}}}});
document.getElementById('st2').addEventListener('click',()=>{{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo((fi+1)%NS);}});
document.getElementById('rs').addEventListener('click',()=>{{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(0);}});
document.getElementById('spd').addEventListener('input',function(){{document.getElementById('vs').textContent=(this.value/1000).toFixed(1)+'s';}});
fsl.addEventListener('input',function(){{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(parseInt(this.value));}});
document.getElementById('heat-toggle').addEventListener('change',()=>apply(S[fi]));
document.getElementById('part-toggle').addEventListener('change',()=>apply(S[fi]));
function jumpTo(l){{const i=SI[l];if(i===undefined)return;clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(i);}}
function log(m,c=''){{
  const el=document.getElementById('log');const d=document.createElement('div');
  d.className=c;d.textContent='[t'+S[fi].tick+'] '+m;
  el.prepend(d);while(el.children.length>20)el.removeChild(el.lastChild);
}}
goTo(0);
log('sim2 iniciado. '+Object.keys(N).length+' nodos | node_weight calculado.','lok');
log('Heatmap y particulas activos. Clic en nodo para ver pesos.','lok');
</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = _time.perf_counter()
    ts = datetime.now()
    print("tanGo sim2 — Pesos estaticos + Movimiento real")
    print(f"  Inicio: {ts.strftime('%Y-%m-%d %H:%M:%S')}\n")

    params  = load_params()
    n_ticks = params.get("n_ticks", 60)

    scenarios = [s for s in params.get("scenarios", [])
                 if not str(s.get("label","")).startswith("_")]
    if not scenarios:
        scenarios = [
            dict(label="Hora pico — lun 8am",
                 timestamp=datetime(2024,3,4,8,0), temperature_c=22.0,
                 is_raining=False, wind_speed_kmh=10.0, visibility_m=10000.0),
        ]

    if CITY_JSON.exists():
        graph = json_to_traffic_graph(CITY_JSON)
        print(f"  Grafo: {graph.graph.number_of_nodes()} nodos, "
              f"{graph.graph.number_of_edges()} aristas")
    else:
        from graph.simulator import TrafficGraph
        graph = TrafficGraph()
        graph.build_sample_city()
        print("  Grafo de ejemplo (9 nodos)")

    # Verificar que node_weight está calculado
    sample = list(graph.intersections.values())[0]
    if sample.node_weight == 1.0:
        print("  ⚠ node_weight=1.0 en todos los nodos.")
        print("  Ejecuta: python graph/city_loader.py para recalcular.\n")
    else:
        nw_vals = [inter.node_weight for inter in graph.intersections.values()]
        print(f"  node_weight: min={min(nw_vals):.3f} max={max(nw_vals):.3f}\n")

    # Presets del JSON como escenarios adicionales (igual que sim1)
    presets = {k: v for k, v in params.get("experiment_presets", {}).items()
               if not k.startswith("_")}

    print(f"  {len(scenarios)} escenarios base + {len(presets)} presets × {n_ticks} ticks\n")

    all_histories = []
    for sc in scenarios:
        print(f"  Simulando: {sc['label']}...")
        hist = simulate(sc, graph, n_ticks, params)
        all_histories.append((sc["label"], hist))

    # Presets como escenarios extra
    base_sc = dict(scenarios[0]) if scenarios else {}
    for preset_key, preset in presets.items():
        label = preset.get("label", preset_key)
        print(f"  Simulando preset: {label}...")
        sc_merged = dict(base_sc)
        sc_merged["label"] = label
        if preset.get("force_rain"):    sc_merged["is_raining"] = True
        if preset.get("force_weekend"): sc_merged["timestamp"]  = "2024-03-09T15:00:00"
        # Para ghost_city: forzar spawn_rate=0
        preset_params = dict(params)
        preset_params["movement"] = dict(params.get("movement", {}))
        if preset.get("vehicle_multiplier", 1.0) == 0.0:
            preset_params["movement"]["spawn_rate"] = 0
        else:
            vm = preset.get("vehicle_multiplier", 1.0)
            preset_params["movement"]["spawn_rate"] = max(1, int(
                params.get("movement", {}).get("spawn_rate", 8) * vm
            ))
        hist = simulate(sc_merged, graph, n_ticks, preset_params)
        all_histories.append((label, hist))

    print(f"\n  Generando visualizacion...")
    vis = build_vis(graph, all_histories, params)
    OUTPUT_VIS.write_text(vis, encoding="utf-8")
    print(f"  ✓ {OUTPUT_VIS}")

    elapsed = _time.perf_counter() - t0
    print(f"\n{'─'*52}")
    print(f"  Inicio  : {ts.strftime('%H:%M:%S')}")
    print(f"  Fin     : {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Duracion: {int(elapsed//60)}m {elapsed%60:.1f}s")
    print(f"  Frames  : {sum(len(h) for _,h in all_histories)}")
    print(f"{'─'*52}")
    print(f"\n✓ Listo. Abre: {OUTPUT_VIS.name}")