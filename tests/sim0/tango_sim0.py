"""
tests/sim0/tango_sim0.py
-------------------------
Simulación 0 — timers fijos (baseline comparativo).

Usa el mismo visor que sim1 pero con TimerAlgorithm.
Las diferencias clave que el visor muestra:
  - La presión siempre es 0 (el timer no la calcula)
  - No hay green wave (wave_offset_s = 0)
  - Los semáforos cambian en ciclo fijo sin importar el tráfico
  - Aparece el aviso "TIMER FIJO" en el header

Ejecutar desde la raíz:
    python tests/sim0/tango_sim0.py
"""
from __future__ import annotations
import sys, logging, time as _time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json as _json

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import robusto de timer_algorithm (mismo directorio)
import importlib.util as _ilu, os as _os, sys as _sys
_ta_path = str(Path(__file__).parent / "timer_algorithm.py")
_ta_spec = _ilu.spec_from_file_location("timer_algorithm_sim0", _ta_path)
_ta_mod  = _ilu.module_from_spec(_ta_spec)
_sys.modules["timer_algorithm_sim0"] = _ta_mod
_ta_spec.loader.exec_module(_ta_mod)
TimerAlgorithm = _ta_mod.TimerAlgorithm

from core.context    import TrafficContext
from core.road       import Phase, IntersectionType
from core.entities   import Vehicle, Pedestrian
from graph.simulator import TrafficGraph
from graph.city_loader import json_to_traffic_graph

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_VIS  = Path(__file__).parent / "tango_vis_sim0.html"
CITY_JSON   = ROOT / "graph" / "city_graph.json"
PARAMS_FILE = ROOT / "tests" / "sim1" / "sim_params.json"


def load_params() -> dict:
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE, encoding="utf-8") as f:
            return _json.load(f)
    return {"n_ticks": 40, "scenarios": []}


def compute_center(graph: TrafficGraph) -> tuple[float, float]:
    nodes = list(graph.intersections.values())
    if not nodes:
        return 20.6656, -103.3863
    return (sum(n.latitude  for n in nodes) / len(nodes),
            sum(n.longitude for n in nodes) / len(nodes))


PARAMS  = load_params()
N_TICKS = PARAMS.get("n_ticks", 40)

SCENARIOS = [
    dict(label="Timer — hora pico lun 8am",
         timestamp=datetime(2024, 3, 4, 8, 0), temperature_c=22.0,
         is_raining=False, wind_speed_kmh=10.0, visibility_m=10000.0),
    dict(label="Timer — madrugada mie 2am",
         timestamp=datetime(2024, 3, 6, 2, 0), temperature_c=18.0,
         is_raining=False, wind_speed_kmh=5.0, visibility_m=10000.0),
    dict(label="Timer — lluvia sab 3pm",
         timestamp=datetime(2024, 3, 9, 15, 0), temperature_c=16.0,
         is_raining=True, wind_speed_kmh=20.0, visibility_m=3000.0),
]


# ── Spawn de entidades (igual que sim1) ───────────────────────────────────────

V_TYPES_POOL = (
    [None] * 60 +   # autos (None → se elige el tipo)
    ["MOTORCYCLE"] * 15 +
    ["BUS"] * 10 +
    ["TRUCK"] * 8 +
    ["BICYCLE"] * 5 +
    ["EMERGENCY"] * 2
)

def spawn_for_node(node_id: str, itype: IntersectionType,
                   ctx: TrafficContext,
                   spawn_params: dict | None = None) -> list:
    """Igual que sim1 — mismo spawn para que la comparativa sea justa."""
    import random, uuid
    from core.entities import VehicleType, Direction

    sp  = spawn_params or PARAMS.get("spawn", {})
    vm  = float(sp.get("vehicle_multiplier",  1.0))
    pm  = float(sp.get("pedestrian_multiplier", 1.0))
    ep  = float(sp.get("emergency_probability", 0.02))
    wcp = float(sp.get("wheelchair_probability", 0.08))

    if itype == IntersectionType.MASTER:
        nv = random.randint(8,20) if ctx.is_rush_hour else \
             random.randint(1,4)  if ctx.is_late_night else random.randint(5,14)
        np_= random.randint(5,15) if ctx.is_rush_hour else \
             random.randint(0,2)  if ctx.is_late_night else random.randint(2,8)
    elif itype == IntersectionType.NORMAL:
        nv = random.randint(5,12) if ctx.is_rush_hour else \
             random.randint(0,3)  if ctx.is_late_night else random.randint(3,8)
        np_= random.randint(3,10) if ctx.is_rush_hour else \
             random.randint(0,1)  if ctx.is_late_night else random.randint(1,6)
    else:
        nv  = random.randint(2,7) if ctx.is_rush_hour else \
              random.randint(0,2) if ctx.is_late_night else random.randint(1,5)
        np_ = random.randint(0,3)

    nv  = max(0, int(nv  * vm))
    np_ = max(0, int(np_ * pm))

    vtype_pool = [VehicleType.CAR] * 60 + [VehicleType.MOTORCYCLE] * 15 + \
                 [VehicleType.BUS] * 10 + [VehicleType.TRUCK] * 8 + \
                 [VehicleType.BICYCLE] * (2 if ctx.is_raining else 5) + \
                 [VehicleType.EMERGENCY] * 2

    entities = []
    if random.random() < ep:
        entities.append(Vehicle(str(uuid.uuid4()), VehicleType.EMERGENCY,
                                random.choice(list(Direction))))
    for _ in range(nv):
        vtype = random.choice(vtype_pool)
        if vtype == VehicleType.EMERGENCY:
            vtype = VehicleType.CAR
        entities.append(Vehicle(str(uuid.uuid4()), vtype,
                                random.choice(list(Direction))))
    for _ in range(np_):
        entities.append(Pedestrian(str(uuid.uuid4()),
                                   is_wheelchair=random.random() < wcp))
    return entities


# ── Simulación ────────────────────────────────────────────────────────────────

def simulate(scenario: dict, graph: TrafficGraph, n_ticks: int) -> list[dict]:
    sc = {k: v for k, v in scenario.items() if k != "label"}
    if "timestamp" in sc and isinstance(sc["timestamp"], str):
        sc["timestamp"] = datetime.fromisoformat(sc["timestamp"])
    ctx  = TrafficContext.build(**sc)
    algo = TimerAlgorithm(graph)
    algo.reset()

    history = []
    for _ in range(n_ticks):
        entities_by_node = {
            nid: spawn_for_node(nid, inter.intersection_type, ctx,
                                spawn_params=PARAMS.get("spawn"))
            for nid, inter in graph.intersections.items()
        }
        result = algo.run_tick(entities_by_node, ctx)

        # TimerTickResult tiene .nodes como dict[str, dict]
        nodes_frame = {}
        result_nodes = (result.nodes if hasattr(result, "nodes")
                        else result.get("nodes", {}))
        for nid, nd in result_nodes.items():
            inter = graph.intersections[nid]
            nodes_frame[nid] = {
                "phase":        nd["phase"],
                "phase_ns":     nd["phase_ns"],
                "phase_ew":     nd["phase_ew"],
                "active_axis":  "ns",
                "signals":      nd.get("signals", {}),
                "pressure":     0.0,      # timer no calcula presión
                "pressure_own": 0.0,
                "pressure_ns":  0.0,
                "pressure_ew":  0.0,
                "wave_offset_s": 0.0,     # sin green wave
                "has_light":    nd["has_light"],
                "threshold":    100.0,
                "ticks_red":    nd.get("ticks_red", 0),
                "timeout":      nd.get("timeout", 4),
                "itype":        inter.intersection_type,
                "geometry":     inter.geometry,
                "geo_label":    inter.geometry_label,
                "name":         nd["name"],
                "lat":          nd["lat"],
                "lon":          nd["lon"],
                "counts":       nd.get("counts", {}),
                "cluster_id":   None,
            }

        # Conteo de fases
        green = sum(1 for nd in nodes_frame.values() if nd["phase"] == "green")
        yellow= sum(1 for nd in nodes_frame.values() if nd["phase"] == "yellow")
        red   = sum(1 for nd in nodes_frame.values()
                    if nd["phase"] == "red" and nd["has_light"])
        blink = sum(1 for nd in nodes_frame.values()
                    if nd["phase"] == "blink" or not nd["has_light"])

        tick_num = (result.tick_number
                    if hasattr(result, "tick_number") else _ + 1)
        history.append({
            "tick":    tick_num,
            "nodes":   nodes_frame,
            "flows":   [],
            "total":   sum(len(e) for e in entities_by_node.values()),
            "greens":  green,
            "yellows": yellow,
            "reds":    red,
            "blinks":  blink,
            "cluster_sizes": {},
        })
    return history


# ── Visualización — mismo build_vis que sim1 con badge TIMER FIJO ─────────────

def build_vis(graph: TrafficGraph, all_histories: list[tuple]) -> str:
    """Mismo visor que sim1 — badge amarillo indica TIMER FIJO."""
    clat, clon = compute_center(graph)

    all_snaps_js = []
    sc_index: dict[str, int] = {}

    for sc_label, history in all_histories:
        sc_index[sc_label] = len(all_snaps_js)
        for snap in history:
            nodes_js = {}
            for nid, nd in snap["nodes"].items():
                inter = graph.intersections[nid]
                nodes_js[nid] = {
                    "phase":         nd["phase"],
                    "phase_ns":      nd["phase_ns"],
                    "phase_ew":      nd["phase_ew"],
                    "active_axis":   nd.get("active_axis", "ns"),
                    "signals":       nd.get("signals", {}),
                    "pressure":      0.0,
                    "pressure_own":  0.0,
                    "pressure_ns":   0.0,
                    "pressure_ew":   0.0,
                    "wave_offset_s": 0.0,
                    "threshold":     100.0,
                    "ticks_red":     nd.get("ticks_red", 0),
                    "timeout":       nd.get("timeout", 4),
                    "has_light":     nd["has_light"],
                    "itype":         inter.intersection_type.value,
                    "geo_label":     inter.geometry_label,
                    "name":          nd["name"],
                    "lat":           nd["lat"],
                    "lon":           nd["lon"],
                    "counts":        nd.get("counts", {}),
                    "cluster_id":    None,
                }
            all_snaps_js.append({
                "scenario":     sc_label,
                "tick":         snap["tick"],
                "total":        snap["total"],
                "greens":       snap.get("greens",  0),
                "yellows":      snap.get("yellows", 0),
                "reds":         snap.get("reds",    0),
                "blinks":       snap.get("blinks",  0),
                "cluster_sizes":{},
                "nodes":        nodes_js,
                "flows":        [],
            })

    # Aristas estáticas
    edges_js, drawn = [], set()
    for a, b, data in graph.graph.edges(data=True):
        pair = tuple(sorted([a, b]))
        if pair in drawn: continue
        drawn.add(pair)
        seg = data["segment"]
        na, nb = graph.intersections[a], graph.intersections[b]
        edges_js.append({
            "from": a, "to": b,
            "lat_a": na.latitude,  "lon_a": na.longitude,
            "lat_b": nb.latitude,  "lon_b": nb.longitude,
            "category": seg.category.name, "weight": seg.base_weight,
            "length_m": seg.length_m, "speed_kmh": seg.speed_limit_kmh,
            "name": "",
        })

    nodes_static_js = {}
    for nid, inter in graph.intersections.items():
        nodes_static_js[nid] = {
            "lat": inter.latitude, "lon": inter.longitude,
            "name": inter.name, "itype": inter.intersection_type.value,
            "geometry": inter.geometry.value, "geo_label": inter.geometry_label,
            "has_light": inter.has_traffic_light,
            "threshold": inter.pressure_threshold,
        }

    snaps_json    = _json.dumps(all_snaps_js)
    edges_json    = _json.dumps(edges_js)
    nodes_s_json  = _json.dumps(nodes_static_js)
    sc_index_json = _json.dumps(sc_index)
    n_snaps       = len(all_snaps_js)
    n_nodes       = len(nodes_static_js)

    preset_buttons_html = ""
    for label in sc_index:
        preset_buttons_html += (
            f'<button class="btn btn-exp" '
            f'onclick="jumpToScenario({_json.dumps(label)})">'
            f'{label[:30]}</button>\n'
        )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tanGo sim0 — Timer fijo</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{--bg:#0f1117;--surface:#1a1d2e;--border:#2a2d3e;--text:#e2e8f0;--muted:#64748b;
      --green:#22c55e;--yellow:#eab308;--red:#ef4444;--teal:#14b8a6;--blue:#3b82f6}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
      font-size:13px;height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{padding:8px 16px;background:#1c1a0e;border-bottom:2px solid #713f12;
        display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:15px;font-weight:600}}header h1 span{{color:var(--yellow)}}
.badges{{display:flex;gap:8px;align-items:center}}
.badge{{font-size:10px;padding:2px 8px;border-radius:999px;background:var(--border);color:var(--muted)}}
.badge.run{{background:#166534;color:var(--green)}}
.sim0-badge{{background:#713f12;color:var(--yellow);padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700}}
.layout{{display:flex;flex:1;overflow:hidden}}
#map{{flex:1;padding-bottom:48px}}
aside{{width:296px;background:var(--surface);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}}
.sec{{padding:10px 12px;border-bottom:1px solid var(--border)}}
.sec h3{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}}
.row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
label{{font-size:11px;color:var(--muted);min-width:68px}}
input[type=range]{{flex:1;accent-color:var(--yellow)}}
.val{{font-size:11px;min-width:38px;text-align:right}}
.btn{{padding:5px 10px;border:none;border-radius:6px;font-size:11px;font-weight:500;cursor:pointer;transition:opacity .15s}}
.btn:hover{{opacity:.82}}
.btn-p{{background:var(--yellow);color:#000}}
.btn-s{{background:var(--border);color:var(--text)}}
.btn-exp{{background:#1c1a0e;color:#fde68a;margin:2px;font-size:10px;padding:4px 8px;border-radius:5px;border:1px solid #713f12}}
.btn-row{{display:flex;gap:5px;flex-wrap:wrap}}
.stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.stat{{background:var(--bg);border-radius:6px;padding:7px;border:1px solid var(--border)}}
.stat-val{{font-size:17px;font-weight:700;color:var(--yellow)}}.stat-lbl{{font-size:9px;color:var(--muted)}}
#node-info{{flex:1;overflow-y:auto;padding:10px}}
.placeholder{{color:var(--muted);font-size:11px;text-align:center;margin-top:24px}}
.ic{{background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:10px;margin-bottom:7px}}
.ic h4{{font-size:12px;font-weight:600;margin-bottom:5px}}
.ir{{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin:2px 0}}
.ir span{{color:var(--text)}}
.pill{{display:inline-block;padding:2px 7px;border-radius:999px;font-size:10px;font-weight:700}}
.pg{{background:#166534;color:var(--green)}}.py{{background:#713f12;color:var(--yellow)}}
.pr{{background:#7f1d1d;color:var(--red)}}
.notice{{background:#1c1a0e;border:1px solid #713f12;color:#fde68a;padding:6px 10px;border-radius:6px;font-size:10px;margin-bottom:8px;line-height:1.5}}
#log{{max-height:110px;overflow-y:auto;padding:7px 10px;font-size:10px;font-family:monospace;color:var(--muted);border-top:1px solid var(--border)}}
.lok{{color:var(--green)}}.lwarn{{color:var(--yellow)}}.lerr{{color:var(--red)}}
#toolbar{{position:fixed;bottom:0;left:0;right:296px;background:#1c1a0e;
          border-top:2px solid #713f12;padding:7px 14px;display:flex;gap:6px;
          align-items:center;flex-wrap:wrap;z-index:9999}}
#toolbar .lbl{{font-size:10px;color:#fde68a;font-weight:700;text-transform:uppercase;letter-spacing:.06em}}
.leaflet-tile{{filter:brightness(.7) saturate(.6)}}
.leaflet-container{{background:var(--bg)}}
</style>
</head>
<body>
<header>
  <h1>tan<span>Go</span> &mdash; <span>sim0</span> Timer fijo (sin inteligencia)</h1>
  <div class="badges">
    <span class="sim0-badge">BASELINE</span>
    <span class="badge" id="b-tick">tick #0</span>
    <span class="badge" id="b-sc">—</span>
    <span class="badge" id="b-status">detenido</span>
  </div>
</header>
<div class="layout">
  <div id="map"></div>
  <aside>
    <div class="sec">
      <div class="notice">⚠ Timers fijos — sin deteccion de trafico.<br>
      El ciclo no cambia con la demanda. Sin green wave.</div>
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
        <input type="range" id="frame-sl" min="0" max="{n_snaps-1}" value="0">
        <span class="val" id="v-frame">0/{n_snaps-1}</span>
      </div>
    </div>
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
        <div class="stat"><div class="stat-val" id="s-nodes">{n_nodes}</div><div class="stat-lbl">Nodos</div></div>
      </div>
    </div>
    <div id="node-info">
      <div class="placeholder">Clic en nodo o arista<br>para ver informacion</div>
    </div>
    <div id="log"></div>
  </aside>
</div>
<div id="toolbar">
  <span class="lbl">Escenarios:</span>
  {preset_buttons_html}
</div>
<script>
const ALL_SNAPS    = {snaps_json};
const EDGES        = {edges_json};
const NODES_STATIC = {nodes_s_json};
const SC_INDEX     = {sc_index_json};
const N_SNAPS      = {n_snaps};

const PHASE_C = {{green:'#22c55e',yellow:'#eab308',red:'#ef4444',blink:'#f59e0b'}};
const ITYPE_R = {{master:'#f59e0b',normal:'#3b82f6',blind:'#64748b'}};
const CAT_C   = {{MAIN_AVENUE:'#1d4ed8',SECONDARY_AVENUE:'#6d28d9',
                  STREET:'#1e293b',HIGHWAY:'#0f172a',ALLEY:'#111827'}};
const CAT_W   = {{MAIN_AVENUE:5,SECONDARY_AVENUE:3,STREET:1.5,HIGHWAY:6,ALLEY:1}};
let _blinkOn = true;
setInterval(()=>{{ _blinkOn=!_blinkOn; }}, 600);

const map = L.map('map').setView([{clat},{clon}], 14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'CartoDB',maxZoom:19}}).addTo(map);

EDGES.forEach(e=>{{
  L.polyline([[e.lat_a,e.lon_a],[e.lat_b,e.lon_b]],
    {{color:CAT_C[e.category]||'#1e293b',weight:CAT_W[e.category]||1.5,opacity:.75}})
   .addTo(map).on('click',()=>showEdgeInfo(e,null));
}});

const NM={{}};
Object.entries(NODES_STATIC).forEach(([nid,nd])=>{{
  const r=nd.itype==='master'?13:nd.itype==='normal'?10:7;
  const m=L.circleMarker([nd.lat,nd.lon],{{radius:r,color:ITYPE_R[nd.itype]||'#64748b',
    fillColor:'#ef4444',fillOpacity:.88,weight:2.5}}).addTo(map);
  m.bindTooltip(`<b>${{nid}}</b> ${{nd.geo_label}}<br>${{nd.name}}`,{{permanent:false,direction:'top'}});
  m.on('click',()=>showNodeInfo(nid,ALL_SNAPS[frameIdx]));
  NM[nid]=m;
}});

const FL={{}};
function clearFlows(){{
  Object.values(FL).forEach(([ln,lm])=>{{map.removeLayer(ln);if(lm)map.removeLayer(lm);}});
  Object.keys(FL).forEach(k=>delete FL[k]);
}}

function applySnap(snap){{
  Object.entries(snap.nodes).forEach(([nid,nd])=>{{
    const m=NM[nid]; if(!m) return;
    const st=NODES_STATIC[nid];
    const baseR=st.itype==='master'?13:st.itype==='normal'?10:7;
    const isBlink=nd.phase==='blink';
    m.setStyle({{
      fillColor:!nd.has_light?'#374151':isBlink?(_blinkOn?'#f59e0b':'#1e293b'):(PHASE_C[nd.phase]||'#ef4444'),
      fillOpacity:!nd.has_light?0.45:0.88,
      radius:baseR,
    }});
    m.setTooltipContent(`<b>${{nid}}</b> ${{st.geo_label}}<br>Timer: ${{nd.phase.toUpperCase()}}`);
    m.off('click');m.on('click',()=>showNodeInfo(nid,snap));
  }});

  clearFlows();

  // Conteo de fases
  const pc={{green:0,yellow:0,red:0,blink:0,blind:0}};
  Object.values(snap.nodes).forEach(nd=>{{
    if(!nd.has_light) pc.blind++;
    else if(nd.phase==='blink') pc.blink++;
    else if(pc[nd.phase]!==undefined) pc[nd.phase]++;
  }});

  document.getElementById('s-tick').textContent   = snap.tick;
  document.getElementById('s-total').textContent  = snap.total;
  document.getElementById('s-green').textContent  = pc.green;
  document.getElementById('s-yellow').textContent = pc.yellow;
  document.getElementById('s-red').textContent    = pc.red;
  document.getElementById('s-blink').textContent  = pc.blink;
  document.getElementById('s-blind').textContent  = pc.blind;
  document.getElementById('b-tick').textContent   = `tick #${{snap.tick}}`;
  document.getElementById('b-sc').textContent     = snap.scenario.substring(0,20);
}}

let selectedNode = null;
function pillCls(p){{ return {{green:'pg',yellow:'py',red:'pr'}}[p]||'pr'; }}

function showNodeInfo(nid, snap){{
  selectedNode = nid;
  const st = NODES_STATIC[nid];
  const nd = snap ? snap.nodes[nid] : null;
  if(!nd){{
    document.getElementById('node-info').innerHTML=
      `<div class="ic"><h4>${{nid}}</h4>
       <div class="ir">Tipo<span>${{st.itype.toUpperCase()}}</span></div></div>`;
    return;
  }}
  const c = nd.counts || {{}};
  document.getElementById('node-info').innerHTML=`
  <div class="ic">
    <h4>${{nid}} — ${{nd.name}}</h4>
    <div class="ir">Tipo<span>${{(nd.itype||'').toUpperCase()}}</span></div>
    <div class="ir">Semaforo<span>${{nd.has_light?'Timer fijo':'Sin semaforo'}}</span></div>
    <div style="background:#1c1a0e;border:1px solid #713f12;color:#fde68a;
                padding:4px 6px;border-radius:4px;font-size:9px;margin-top:4px">
      ⚠ Sin deteccion — ciclo fijo. No responde al trafico.
    </div>
  </div>
  ${{nd.has_light?`
  <div class="ic">
    <h4>Fase (timer fijo)</h4>
    <div class="ir">Estado<span><span class="pill ${{pillCls(nd.phase)}}">${{nd.phase.toUpperCase()}}</span></span></div>
    <div class="ir">Ticks en fase<span>${{nd.ticks_in||0}} / ${{nd.phase==='green'?nd.green_ticks:nd.phase==='yellow'?nd.yellow_ticks:nd.red_ticks}}</span></div>
    <div class="ir">Ciclo total<span>${{nd.cycle_s||'?'}}s (${{nd.timeout||'?'}} ticks)</span></div>
  </div>
  <div class="ic">
    <h4>Programa del timer</h4>
    <div class="ir" style="color:var(--green)">Verde<span>${{(nd.green_ticks||0)*30}}s (${{nd.green_ticks}} ticks)</span></div>
    <div class="ir" style="color:var(--yellow)">Amarillo<span>${{(nd.yellow_ticks||0)*30}}s (${{nd.yellow_ticks}} ticks)</span></div>
    <div class="ir" style="color:var(--red)">Rojo<span>${{(nd.red_ticks||0)*30}}s (${{nd.red_ticks}} ticks)</span></div>
    <div class="ir">Presion<span style="color:var(--muted)">N/A — timer ignora trafico</span></div>
    <div class="ir">Green wave<span style="color:var(--muted)">N/A — sin coordinacion</span></div>
  </div>`:'<div class="ic" style="color:var(--muted);font-size:11px">Sin semaforo — ceda el paso</div>'}}
  <div class="ic">
    <h4>Entidades (ignoradas por timer)</h4>
    <div class="ir">Autos<span>${{c.CAR||0}}</span></div>
    <div class="ir">Buses<span>${{c.BUS||0}}</span></div>
    <div class="ir">Peatones<span>${{c.PEDESTRIAN||0}}</span></div>
    <div style="font-size:9px;color:#fde68a;margin-top:4px">
      El timer cambia de fase aunque no haya nadie aqui.
    </div>
  </div>`;
}}

function showEdgeInfo(e, fl){{
  document.getElementById('node-info').innerHTML=`
  <div class="ic"><h4>Segmento vial</h4>
    <div class="ir">Categoria<span>${{e.category}}</span></div>
    <div class="ir">Longitud<span>${{e.length_m}} m</span></div>
    <div class="ir">Velocidad max<span>${{e.speed_kmh}} km/h</span></div>
    <div class="ir">De → A<span>${{e.from}} → ${{e.to}}</span></div>
  </div>`;
}}

let frameIdx=0, running=false, timer=null;
const slider = document.getElementById('frame-sl');
slider.max = N_SNAPS-1;

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
  if(running){{btn.innerHTML='&#9646;&#9646; Pausar';bs.textContent='corriendo';bs.className='badge run';scheduleNext();}}
  else{{btn.innerHTML='&#9654; Iniciar';bs.textContent='detenido';bs.className='badge';clearTimeout(timer);}}
}});
document.getElementById('btn-step').addEventListener('click',()=>{{
  clearTimeout(timer);running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  goToFrame((frameIdx+1)%N_SNAPS);
}});
document.getElementById('btn-reset').addEventListener('click',()=>{{
  clearTimeout(timer);running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  selectedNode=null;
  document.getElementById('node-info').innerHTML='<div class="placeholder">Clic en nodo o arista<br>para ver informacion</div>';
  goToFrame(0);
}});
document.getElementById('speed').addEventListener('input',function(){{
  document.getElementById('v-speed').textContent=(this.value/1000).toFixed(1)+'s';
}});
slider.addEventListener('input',function(){{
  clearTimeout(timer);running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  goToFrame(parseInt(this.value));
}});
function jumpToScenario(label){{
  const idx=SC_INDEX[label];if(idx===undefined)return;
  clearTimeout(timer);running=false;
  document.getElementById('btn-play').innerHTML='&#9654; Iniciar';
  document.getElementById('b-status').className='badge';
  document.getElementById('b-status').textContent='detenido';
  goToFrame(idx);
  log('Saltando a: '+label,'lok');
}}
function log(msg,cls=''){{
  const el=document.getElementById('log');
  const d=document.createElement('div');d.className=cls;
  d.textContent=`[${{new Date().toLocaleTimeString('es',{{hour12:false}})}}] ${{msg}}`;
  el.prepend(d);while(el.children.length>25)el.removeChild(el.lastChild);
}}
goToFrame(0);
log('sim0 — Timer fijo. Sin green wave. Sin coordinacion.','lwarn');
log('Compara con sim1 para ver la diferencia del algoritmo tanGo.');
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = _time.perf_counter()
    ts = datetime.now()
    print("tanGo sim0 — Timer fijo (comparativa)")
    print(f"  Inicio: {ts.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if CITY_JSON.exists():
        graph = json_to_traffic_graph(CITY_JSON)
        print(f"  {graph.graph.number_of_nodes()} nodos desde JSON")
    else:
        from graph.simulator import TrafficGraph as _TG
        graph = _TG()
        graph.build_sample_city()
        print("  Grafo de ejemplo (9 nodos)")

    # Leer escenarios del JSON si existen
    json_sc = [s for s in PARAMS.get("scenarios", [])
               if not str(s.get("label","")).startswith("_")]
    active_scenarios = json_sc if json_sc else SCENARIOS

    print(f"  {len(active_scenarios)} escenarios × {N_TICKS} ticks\n")

    all_histories = []
    for sc in active_scenarios:
        label = sc.get("label", str(sc))
        print(f"  Simulando: {label}...")
        hist = simulate(sc, graph, N_TICKS)
        all_histories.append((label, hist))

    print("\n  Generando visualizacion...")
    vis = build_vis(graph, all_histories)
    OUTPUT_VIS.write_text(vis, encoding="utf-8")
    print(f"  ✓ {OUTPUT_VIS}")

    elapsed = _time.perf_counter() - t0
    frames  = sum(len(h) for _, h in all_histories)
    print(f"\n{'─'*52}")
    print(f"  Inicio  : {ts.strftime('%H:%M:%S')}")
    print(f"  Fin     : {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Duracion: {int(elapsed//60)}m {elapsed%60:.1f}s")
    print(f"  Frames  : {frames}")
    print(f"{'─'*52}")
    print(f"\n✓ Listo. Abre: {OUTPUT_VIS.name}")