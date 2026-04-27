"""
tests/sim0/tango_sim0.py
Simulación comparativa — timers fijos.
Ejecutar: python tests/sim0/tango_sim0.py
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json as _json

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "sim1"))

import folium
from core.context    import TrafficContext
from core.road       import Phase
from core.entities   import Vehicle, Pedestrian
from graph.simulator import TrafficGraph
from graph.city_loader import json_to_traffic_graph
# Import robusto — agrega el directorio de sim0 al path y luego importa normal
import sys as _sys, os as _os
_sim0_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sim0_dir not in _sys.path:
    _sys.path.insert(0, _sim0_dir)
from timer_algorithm import TimerAlgorithm
# Import robusto de tango_sim — funciona desde cualquier directorio
import importlib.util as _ilu2, sys as _sys2
_ts_path = str(Path(__file__).parent.parent / "sim1" / "tango_sim.py")
_ts_spec = _ilu2.spec_from_file_location("tango_sim_mod", _ts_path)
_ts_mod  = _ilu2.module_from_spec(_ts_spec)
_sys2.modules["tango_sim_mod"] = _ts_mod
_ts_spec.loader.exec_module(_ts_mod)
spawn_for_node   = _ts_mod.spawn_for_node
load_params      = _ts_mod.load_params
compute_center   = _ts_mod.compute_center
build_folium_map = _ts_mod.build_folium_map

logger     = logging.getLogger(__name__)
OUTPUT_VIS = Path(__file__).parent / "tango_vis_sim0.html"
OUTPUT_MAP = Path(__file__).parent / "tango_map_sim0.html"
CITY_JSON  = ROOT / "graph" / "city_graph.json"
PARAMS     = load_params()
N_TICKS    = PARAMS.get("n_ticks", 40)

SCENARIOS = [
    dict(label="Timer — hora pico lun 8am",
         timestamp=datetime(2024,3,4,8,0), temperature_c=22.0,
         is_raining=False, wind_speed_kmh=10.0, visibility_m=10000.0),
    dict(label="Timer — madrugada mie 2am",
         timestamp=datetime(2024,3,6,2,0), temperature_c=18.0,
         is_raining=False, wind_speed_kmh=5.0, visibility_m=10000.0),
]
PHASE_C = {"green":"#22c55e","yellow":"#eab308","red":"#ef4444","blink":"#f59e0b"}


def simulate_timers(scenario, graph, n_ticks):
    sc = {k:v for k,v in scenario.items() if k!="label"}
    if "timestamp" in sc and isinstance(sc["timestamp"], str):
        sc["timestamp"] = datetime.fromisoformat(sc["timestamp"])
    ctx  = TrafficContext.build(**sc)
    algo = TimerAlgorithm(graph)
    algo.reset()
    history = []
    for _ in range(n_ticks):
        ents = {nid: spawn_for_node(nid, inter.intersection_type, ctx,
                                    spawn_params=PARAMS.get("spawn"))
                for nid, inter in graph.intersections.items()}
        result = algo.run_tick(ents, ctx)
        frame  = {
            "tick": result.tick_number, "total": result.total_entities,
            "greens": result.green_count, "yellows": result.yellow_count,
            "reds": result.red_count, "blinks": 0, "cluster_sizes": {},
            "nodes": {nid: nd for nid, nd in result.nodes.items()},
            "flows": result.flows,
        }
        history.append(frame)
    return history


def build_vis(graph, all_histories):
    clat, clon = compute_center(graph)
    snaps_js, sc_idx = [], {}
    for label, hist in all_histories:
        sc_idx[label] = len(snaps_js)
        for snap in hist:
            njs = {}
            for nid, nd in snap["nodes"].items():
                inter = graph.intersections[nid]
                njs[nid] = {
                    "phase": nd["phase"], "phase_ns": nd["phase_ns"],
                    "phase_ew": nd["phase_ew"], "active_axis": "ns",
                    "signals": nd.get("signals",{}),
                    "pressure":0.0,"pressure_own":0.0,"pressure_ns":0.0,
                    "pressure_ew":0.0,"wave_offset_s":0.0,"threshold":100.0,
                    "ticks_red": nd.get("ticks_red",0), "timeout": nd.get("timeout",4),
                    "has_light": nd["has_light"], "itype": inter.intersection_type.value,
                    "geo_label": inter.geometry_label, "name": nd["name"],
                    "lat": nd["lat"], "lon": nd["lon"], "counts": nd["counts"],
                    "cluster_id": None,
                }
            snaps_js.append({
                "scenario": label, "tick": snap["tick"],
                "total": snap["total"], "greens": snap["greens"],
                "yellows": snap.get("yellows",0), "reds": snap.get("reds",0),
                "blinks": 0, "cluster_sizes": {},
                "nodes": njs, "flows": snap["flows"],
            })

    from core.road import RoadCategory
    edges_js, drawn = [], set()
    for a, b, data in graph.graph.edges(data=True):
        pair = tuple(sorted([a,b]))
        if pair in drawn: continue
        drawn.add(pair)
        seg = data["segment"]
        na, nb = graph.intersections[a], graph.intersections[b]
        edges_js.append({"from":a,"to":b,"lat_a":na.latitude,"lon_a":na.longitude,
                          "lat_b":nb.latitude,"lon_b":nb.longitude,
                          "category":seg.category.name,"weight":seg.base_weight,
                          "length_m":seg.length_m,"speed_kmh":seg.speed_limit_kmh,"name":""})

    ns_js = {}
    for nid, inter in graph.intersections.items():
        ns_js[nid] = {"lat":inter.latitude,"lon":inter.longitude,"name":inter.name,
                      "itype":inter.intersection_type.value,"geometry":inter.geometry.value,
                      "geo_label":inter.geometry_label,"has_light":inter.has_traffic_light,
                      "threshold":inter.pressure_threshold}

    sj  = _json.dumps(snaps_js)
    ej  = _json.dumps(edges_js)
    nj  = _json.dumps(ns_js)
    scj = _json.dumps(sc_idx)
    ns  = len(snaps_js)
    btns = "".join(f'<button class="btn btn-exp" onclick="jumpTo({_json.dumps(l)})">{l[:28]}</button>\n'
                   for l in sc_idx)

    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tanGo sim0 — Timer fijo</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{{--bg:#0f1117;--sur:#1a1d2e;--brd:#2a2d3e;--txt:#e2e8f0;--mut:#64748b;
      --grn:#22c55e;--yel:#eab308;--red:#ef4444;--tel:#14b8a6}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;
      font-size:13px;height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{padding:8px 16px;background:#1c1a0e;border-bottom:2px solid #713f12;
        display:flex;align-items:center;justify-content:space-between}}
header h1{{font-size:15px;font-weight:600;color:var(--yel)}}
.badge{{font-size:10px;padding:2px 8px;border-radius:999px;background:var(--brd);color:var(--mut)}}
.sim0{{background:#713f12;color:var(--yel);padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700}}
.layout{{display:flex;flex:1;overflow:hidden}}
#map{{flex:1;padding-bottom:46px}}
aside{{width:288px;background:var(--sur);border-left:1px solid var(--brd);
       display:flex;flex-direction:column;overflow:hidden}}
.sec{{padding:10px 12px;border-bottom:1px solid var(--brd)}}
.sec h3{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:8px}}
.row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
label{{font-size:11px;color:var(--mut);min-width:68px}}
input[type=range]{{flex:1;accent-color:var(--yel)}}
.val{{font-size:11px;min-width:36px;text-align:right}}
.btn{{padding:5px 10px;border:none;border-radius:6px;font-size:11px;font-weight:500;cursor:pointer}}
.bp{{background:var(--yel);color:#000}}.bs{{background:var(--brd);color:var(--txt)}}
.btn-exp{{background:#1c1a0e;color:#fde68a;margin:2px;font-size:10px;padding:4px 8px;
          border-radius:5px;border:1px solid #713f12}}
.btn-row{{display:flex;gap:5px;flex-wrap:wrap}}
.sg{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.st{{background:var(--bg);border-radius:6px;padding:7px;border:1px solid var(--brd)}}
.sv{{font-size:17px;font-weight:700;color:var(--yel)}}.sl{{font-size:9px;color:var(--mut)}}
#ni{{flex:1;overflow-y:auto;padding:10px}}
.ph{{color:var(--mut);font-size:11px;text-align:center;margin-top:24px}}
.ic{{background:var(--bg);border:1px solid var(--brd);border-radius:7px;padding:10px;margin-bottom:7px}}
.ic h4{{font-size:12px;font-weight:600;margin-bottom:5px}}
.ir{{display:flex;justify-content:space-between;font-size:11px;color:var(--mut);margin:2px 0}}
.ir span{{color:var(--txt)}}
.pill{{display:inline-block;padding:2px 7px;border-radius:999px;font-size:10px;font-weight:700}}
.pg{{background:#166534;color:var(--grn)}}.py{{background:#713f12;color:var(--yel)}}.pr{{background:#7f1d1d;color:var(--red)}}
#log{{max-height:100px;overflow-y:auto;padding:6px 10px;font-size:10px;font-family:monospace;color:var(--mut);border-top:1px solid var(--brd)}}
.lok{{color:var(--grn)}}.lw{{color:var(--yel)}}
#tb{{position:fixed;bottom:0;left:0;right:288px;background:#1c1a0e;
     border-top:2px solid #713f12;padding:7px 14px;display:flex;gap:6px;
     align-items:center;flex-wrap:wrap;z-index:9999}}
#tb .lbl{{font-size:10px;color:var(--yel);font-weight:700;text-transform:uppercase;letter-spacing:.06em}}
.notice{{background:#1c1a0e;border:1px solid #713f12;color:#fde68a;padding:6px 10px;
         border-radius:6px;font-size:10px;margin-bottom:8px;line-height:1.5}}
.leaflet-tile{{filter:brightness(.7) saturate(.6)}}
.leaflet-container{{background:var(--bg)}}
</style></head><body>
<header>
  <h1>tanGo sim0 — Timer fijo (sin inteligencia)</h1>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="sim0">BASELINE</span>
    <span class="badge" id="bt">tick #0</span>
    <span class="badge" id="bs2">—</span>
  </div>
</header>
<div class="layout">
<div id="map"></div>
<aside>
  <div class="sec">
    <div class="notice">⚠ Timers fijos — sin deteccion de trafico.<br>
    El ciclo es fijo y no cambia con la demanda.<br>
    Sin green wave. Sin coordinacion vecinal.</div>
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
  </div>
  <div class="sec"><h3>Estadisticas</h3>
    <div class="sg">
      <div class="st"><div class="sv" id="sk">0</div><div class="sl">Tick</div></div>
      <div class="st"><div class="sv" id="se">0</div><div class="sl">Entidades</div></div>
      <div class="st"><div class="sv" id="sg2" style="color:var(--grn)">0</div><div class="sl">Verde</div></div>
      <div class="st"><div class="sv" id="sy" style="color:var(--yel)">0</div><div class="sl">Amarillo</div></div>
      <div class="st"><div class="sv" id="sr" style="color:var(--red)">0</div><div class="sl">Rojo</div></div>
      <div class="st"><div class="sv" id="sn">{len(ns_js)}</div><div class="sl">Nodos</div></div>
    </div>
  </div>
  <div id="ni"><div class="ph">Clic en nodo para ver informacion</div></div>
  <div id="log"></div>
</aside>
</div>
<div id="tb">
  <span class="lbl">Escenarios:</span>
  {btns}
</div>
<script>
const S={sj};
const E={ej};
const N={nj};
const SI={scj};
const NS={ns};
const PC={{green:'#22c55e',yellow:'#eab308',red:'#ef4444'}};
const IR={{master:'#f59e0b',normal:'#eab308',blind:'#64748b'}};
const CC={{MAIN_AVENUE:'#1d4ed8',SECONDARY_AVENUE:'#6d28d9',STREET:'#1e293b',HIGHWAY:'#0f172a',ALLEY:'#111827'}};
const CW={{MAIN_AVENUE:5,SECONDARY_AVENUE:3,STREET:1.5,HIGHWAY:6,ALLEY:1}};
const map=L.map('map').setView([{clat},{clon}],14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{attribution:'CartoDB',maxZoom:19}}).addTo(map);
E.forEach(e=>L.polyline([[e.lat_a,e.lon_a],[e.lat_b,e.lon_b]],{{color:CC[e.category]||'#1e293b',weight:CW[e.category]||1.5,opacity:.75}}).addTo(map).on('click',()=>showEdge(e)));
const NM={{}};
Object.entries(N).forEach(([id,nd])=>{{
  const r=nd.itype==='master'?13:nd.itype==='normal'?10:7;
  const m=L.circleMarker([nd.lat,nd.lon],{{radius:r,color:IR[nd.itype]||'#64748b',fillColor:'#ef4444',fillOpacity:.88,weight:2.5}}).addTo(map);
  m.bindTooltip(`<b>${{id}}</b> Timer`,{{permanent:false,direction:'top'}});
  m.on('click',()=>showNode(id,S[fi]));NM[id]=m;
}});
const FL={{}};
function cF(){{Object.values(FL).forEach(([l,m])=>{{map.removeLayer(l);if(m)map.removeLayer(m);}});Object.keys(FL).forEach(k=>delete FL[k]);}}
function apply(snap){{
  Object.entries(snap.nodes).forEach(([id,nd])=>{{
    const m=NM[id];if(!m)return;
    const st=N[id],r=st.itype==='master'?13:st.itype==='normal'?10:7;
    m.setStyle({{fillColor:!nd.has_light?'#374151':(PC[nd.phase]||'#ef4444'),fillOpacity:!nd.has_light?.45:.88,radius:r}});
    m.off('click');m.on('click',()=>showNode(id,snap));
  }});
  cF();
  snap.flows.forEach(fl=>{{
    const n=fl.fwd+fl.bwd;if(!n)return;
    const na=N[fl.from],nb=N[fl.to];if(!na||!nb)return;
    const fc=n>=15?'#ef4444':n>=8?'#f59e0b':'#22c55e';
    const ln=L.polyline([[na.lat,na.lon],[nb.lat,nb.lon]],{{color:fc,weight:Math.min(7,1.2+n*.28),opacity:.6}}).addTo(map);
    const mx=(na.lat+nb.lat)/2,my=(na.lon+nb.lon)/2;
    const lm=L.marker([mx,my],{{icon:L.divIcon({{html:`<div style="color:${{fc}};font-size:8px;font-weight:700;text-shadow:0 0 3px #000">+${{fl.fwd}}</div>`,className:'',iconAnchor:[12,5]}}),interactive:false}}).addTo(map);
    FL[fl.from+fl.to]=[ln,lm];
  }});
  const pc={{green:0,yellow:0,red:0}};
  Object.values(snap.nodes).forEach(nd=>{{if(pc[nd.phase]!==undefined)pc[nd.phase]++;}});
  document.getElementById('sk').textContent=snap.tick;
  document.getElementById('se').textContent=snap.total;
  document.getElementById('sg2').textContent=pc.green;
  document.getElementById('sy').textContent=pc.yellow;
  document.getElementById('sr').textContent=pc.red;
  document.getElementById('bt').textContent='tick #'+snap.tick;
  document.getElementById('bs2').textContent=snap.scenario.substring(0,18);
}}
function showNode(id,snap){{
  const st=N[id],nd=snap?snap.nodes[id]:null;
  const pC=p=>({{'green':'pg','yellow':'py','red':'pr'}}[p]||'pr');
  const c=nd?nd.counts||{{}}:{{}};
  document.getElementById('ni').innerHTML=`
  <div class="ic"><h4>${{id}} — ${{st.name}}</h4>
    <div class="ir">Tipo<span>${{st.itype.toUpperCase()}}</span></div>
    <div class="ir">Semaforo<span>${{st.has_light?'Si (timer fijo)':'No'}}</span></div>
    <div style="color:#fde68a;font-size:10px;margin-top:6px;background:#1c1a0e;padding:4px 6px;border-radius:4px">
      ⚠ Sin deteccion. El ciclo no depende del trafico.
    </div></div>
  ${{nd?`<div class="ic"><h4>Fase</h4>
    <div class="ir">Estado<span><span class="pill ${{pC(nd.phase)}}">${{nd.phase.toUpperCase()}}</span></span></div>
    <div class="ir">Ticks en fase<span>${{nd.ticks_in||0}}</span></div></div>
  <div class="ic"><h4>Entidades (ignoradas)</h4>
    <div class="ir">Autos<span>${{c.CAR||0}}</span></div>
    <div class="ir">Peatones<span>${{c.PEDESTRIAN||0}}</span></div></div>`:''}};
}}
function showEdge(e){{
  document.getElementById('ni').innerHTML=`<div class="ic"><h4>Segmento</h4>
    <div class="ir">Categoria<span>${{e.category}}</span></div>
    <div class="ir">Velocidad<span>${{e.speed_kmh}} km/h</span></div></div>`;
}}
let fi=0,run=false,tmr=null;
const fsl=document.getElementById('fsl');
function goTo(idx){{if(idx<0||idx>=NS)return;fi=idx;fsl.value=idx;document.getElementById('vf').textContent=idx+'/'+(NS-1);apply(S[idx]);}}
function next(){{if(!run)return;goTo((fi+1)%NS);tmr=setTimeout(next,parseInt(document.getElementById('spd').value));}}
document.getElementById('pp').addEventListener('click',()=>{{run=!run;const b=document.getElementById('pp');if(run){{b.innerHTML='&#9646;&#9646; Pausar';next();}}else{{b.innerHTML='&#9654; Iniciar';clearTimeout(tmr);}}}});
document.getElementById('st2').addEventListener('click',()=>{{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo((fi+1)%NS);}});
document.getElementById('rs').addEventListener('click',()=>{{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(0);}});
document.getElementById('spd').addEventListener('input',function(){{document.getElementById('vs').textContent=(this.value/1000).toFixed(1)+'s';}});
fsl.addEventListener('input',function(){{clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(parseInt(this.value));}});
function jumpTo(l){{const i=SI[l];if(i===undefined)return;clearTimeout(tmr);run=false;document.getElementById('pp').innerHTML='&#9654; Iniciar';goTo(i);}}
function log(m,c=''){{const el=document.getElementById('log');const d=document.createElement('div');d.className=c;d.textContent='['+new Date().toLocaleTimeString('es',{{hour12:false}})+']] '+m;el.prepend(d);while(el.children.length>15)el.removeChild(el.lastChild);}}
goTo(0);
log('sim0 — Timer fijo. Sin green wave. Sin coordinacion.','lw');
</script></body></html>"""


if __name__ == "__main__":
    import time as _time
    _t0 = _time.perf_counter()
    _ts = datetime.now()
    print("tanGo sim0 — Timer fijo (comparativa)")
    print(f"  Inicio: {_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if CITY_JSON.exists():
        graph = json_to_traffic_graph(CITY_JSON)
        print(f"  {graph.graph.number_of_nodes()} nodos desde JSON")
    else:
        from graph.simulator import TrafficGraph
        graph = TrafficGraph()
        graph.build_sample_city()
        print("  Grafo de ejemplo (9 nodos)")

    all_histories = []
    for sc in SCENARIOS:
        print(f"  Simulando: {sc['label']}...")
        history = simulate_timers(sc, graph, N_TICKS)
        all_histories.append((sc["label"], history))

    print("  Generando visualizacion...")
    vis = build_vis(graph, all_histories)
    OUTPUT_VIS.write_text(vis, encoding="utf-8")
    print(f"  ✓ {OUTPUT_VIS}")

    _e = _time.perf_counter() - _t0
    print(f"\n  Duracion: {int(_e//60)}m {_e%60:.1f}s")
    print(f"✓ Listo. Abre: {OUTPUT_VIS.name}")