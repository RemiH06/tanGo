"""
tests/sim3/tango_sim3.py
-------------------------
sim3 — Visualización del agente PPO entrenado.

Carga el checkpoint PPO, corre un episodio completo tick a tick,
y genera tango_vis_sim3.html con:

  · Mapa Leaflet (dark) igual que sim2
  · Partículas + heatmap (heredados de sim2)
  · Pulso visual en semáforos que el agente cambia ese tick
  · Reward curve animada tick a tick
  · Panel de decisiones: cuántos semáforos mantuvo vs cambió
  · Comparativa inline sim0 / sim1 / sim3 en sparklines
  · Botón para cargar comparison.json si existe

Uso:
    cd /mnt/c/Users/hecto/.../36.TanGo
    source ~/tango_env/bin/activate
    python tests/sim3/tango_sim3.py
    python tests/sim3/tango_sim3.py --checkpoint tests/sim3/checkpoints/checkpoint_00550
    python tests/sim3/tango_sim3.py --ticks 120
"""

from __future__ import annotations
import sys, json as _json, random, uuid, math, time as _time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CFG_FILE   = Path(__file__).parent / "sim3_config.json"
CITY_JSON  = ROOT / "graph" / "city_graph.json"
COMP_FILE  = Path(__file__).parent / "results" / "comparison.json"
OUTPUT_VIS = Path(__file__).parent / "tango_vis_sim3.html"


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg() -> dict:
    with open(CFG_FILE, encoding="utf-8") as f:
        return json.load(f) if False else _json.load(f)


def spawn_entities(itype_str: str, is_rush: bool):
    from core.entities import Vehicle, Pedestrian, VehicleType, Direction
    from core.road import IntersectionType
    imap = {"master": IntersectionType.MASTER,
            "normal": IntersectionType.NORMAL,
            "blind":  IntersectionType.BLIND}
    itype = imap.get(itype_str, IntersectionType.NORMAL)
    if itype == IntersectionType.MASTER:
        nv = random.randint(5, 14) if is_rush else random.randint(2, 8)
    elif itype == IntersectionType.NORMAL:
        nv = random.randint(2, 8)  if is_rush else random.randint(1, 5)
    else:
        nv = random.randint(0, 3)
    pool = (
        [VehicleType.CAR]        * 60 +
        [VehicleType.MOTORCYCLE] * 15 +
        [VehicleType.BUS]        * 10 +
        [VehicleType.TRUCK]      * 8  +
        [VehicleType.BICYCLE]    * 5  +
        [VehicleType.EMERGENCY]  * 2
    )
    ents = [Vehicle(str(uuid.uuid4()), random.choice(pool),
                    random.choice(list(__import__(
                        'core.entities', fromlist=['Direction']).Direction)))
            for _ in range(nv)]
    if random.random() < 0.15:
        ents.append(Pedestrian(str(uuid.uuid4())))
    return ents


def compute_center(graph) -> tuple[float, float]:
    nodes = list(graph.intersections.values())
    if not nodes:
        return 20.6656, -103.3863
    return (sum(n.latitude  for n in nodes) / len(nodes),
            sum(n.longitude for n in nodes) / len(nodes))


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULACIÓN PPO
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ppo(checkpoint_path: str, graph, n_ticks: int, cfg: dict) -> list[dict]:
    """
    Corre n_ticks con el agente PPO cargado desde checkpoint.
    Devuelve historia de frames con datos extra de RL.
    """
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from tests.sim3.tango_env import TanGoEnv
    from core.movement import MovementEngine

    ray.init(ignore_reinit_error=True, num_cpus=2)
    register_env("TanGoEnv", lambda c: TanGoEnv(c))

    env_cfg = cfg["environment"]
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner       = False,
            enable_env_runner_and_connector_v2 = False,
        )
        .environment("TanGoEnv", env_config={
            "spawn_rate":   env_cfg["spawn_rate"],
            "max_entities": env_cfg["max_entities"],
        })
        .framework("torch")
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
    )
    config.model = {
        "fcnet_hiddens":    [256, 256, 128],
        "fcnet_activation": "relu",
        "max_seq_len":      20,
    }

    algo = config.build_algo()
    algo.restore(checkpoint_path)

    env = TanGoEnv({
        "env_config": {
            "spawn_rate":   env_cfg["spawn_rate"],
            "max_entities": env_cfg["max_entities"],
        }
    })

    obs, _ = env.reset()

    history      = []
    reward_curve = []
    arrived_total = 0

    print(f"  Corriendo {n_ticks} ticks con agente PPO...")

    for tick_n in range(n_ticks):
        # El agente decide
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, _, info = env.step(action)

        reward_curve.append(round(float(reward), 4))
        arrived_total = info.get("arrived", 0)

        # Semáforos que el agente cambió este tick
        signaled    = env.signaled
        changed_ids = [nid for i, nid in enumerate(signaled) if int(action[i]) == 1]

        # Fases del grafo INTERNO del env (el que el agente realmente modifica)
        from core.road import Phase
        env_graph = env.graph   # ← grafo vivo, no el externo
        nodes_frame = {}
        for nid, inter in env_graph.intersections.items():
            nodes_frame[nid] = {
                "phase":       inter.current_phase.value,
                "has_light":   inter.has_traffic_light,
                "pressure":    round(inter.pressure, 1),
                "threshold":   inter.pressure_threshold,
                "ticks_red":   inter._ticks_in_phase,
                "name":        inter.name,
                "lat":         inter.latitude,
                "lon":         inter.longitude,
                "itype":       inter.intersection_type.value,
                "geo_label":   inter.geometry_label,
                "node_weight": round(inter.node_weight, 3),
                "changed":     nid in changed_ids,
            }

        # Partículas y heatmap del MovementEngine del env
        mv = env._movement
        particles = mv.get_particles() if mv else []
        heatmap   = mv.get_heatmap()   if mv else {}

        green_count = sum(1 for i in env_graph.intersections.values()
                          if i.current_phase == Phase.GREEN)
        red_count   = sum(1 for i in env_graph.intersections.values()
                          if i.current_phase == Phase.RED)
        blink_count = sum(1 for i in env_graph.intersections.values()
                          if i.current_phase == Phase.BLINK)

        history.append({
            "tick":          tick_n + 1,
            "nodes":         nodes_frame,
            "particles":     particles,
            "heatmap":       heatmap,
            "reward":        round(float(reward), 4),
            "reward_curve":  list(reward_curve),           # acumulado hasta este tick
            "reward_sum":    round(sum(reward_curve), 4),
            "action":        [int(a) for a in action],
            "changed_ids":   changed_ids,
            "n_changed":     len(changed_ids),
            "n_kept":        len(signaled) - len(changed_ids),
            "green_count":   green_count,
            "red_count":     red_count,
            "blink_count":   blink_count,
            "total":         info.get("moving", 0) + info.get("stopped", 0),
            "moving":        info.get("moving", 0),
            "stopped":       info.get("stopped", 0),
            "arrived":       arrived_total,
            "flows":         [],  # env no expone flows directamente
        })

        if done:
            break

    algo.stop()
    ray.shutdown()
    print(f"  ✓ {len(history)} ticks | reward_sum={sum(reward_curve):.2f} | arrived={arrived_total}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def build_vis(graph, history: list[dict], comp: dict | None) -> str:
    clat, clon = compute_center(graph)
    ns = len(history)

    # Nodos estáticos
    ns_js = {}
    for nid, inter in graph.intersections.items():
        ns_js[nid] = {
            "lat":         inter.latitude,
            "lon":         inter.longitude,
            "name":        inter.name,
            "itype":       inter.intersection_type.value,
            "geo_label":   inter.geometry_label,
            "has_light":   inter.has_traffic_light,
            "threshold":   inter.pressure_threshold,
            "node_weight": round(inter.node_weight, 3),
        }

    # Aristas estáticas
    edges_js, drawn = [], set()
    for a, b, data in graph.graph.edges(data=True):
        pair = tuple(sorted([a, b]))
        if pair in drawn:
            continue
        drawn.add(pair)
        seg = data["segment"]
        na, nb = graph.intersections[a], graph.intersections[b]
        edges_js.append({
            "from": a, "to": b,
            "lat_a": na.latitude,  "lon_a": na.longitude,
            "lat_b": nb.latitude,  "lon_b": nb.longitude,
            "category": seg.category.name,
        })

    # Serializar history (snaps)
    snaps_js = []
    for snap in history:
        njs = {}
        for nid, nd in snap["nodes"].items():
            njs[nid] = {
                "phase":       nd["phase"],
                "has_light":   nd["has_light"],
                "pressure":    nd["pressure"],
                "threshold":   nd["threshold"],
                "ticks_red":   nd["ticks_red"],
                "name":        nd["name"],
                "lat":         nd["lat"],
                "lon":         nd["lon"],
                "itype":       nd["itype"],
                "geo_label":   nd["geo_label"],
                "node_weight": nd["node_weight"],
                "changed":     nd["changed"],
            }
        snaps_js.append({
            "tick":         snap["tick"],
            "nodes":        njs,
            "particles":    snap.get("particles", []),
            "heatmap":      snap.get("heatmap", {}),
            "reward":       snap["reward"],
            "reward_curve": snap["reward_curve"],
            "reward_sum":   snap["reward_sum"],
            "n_changed":    snap["n_changed"],
            "n_kept":       snap["n_kept"],
            "changed_ids":  snap["changed_ids"],
            "green_count":  snap["green_count"],
            "red_count":    snap["red_count"],
            "blink_count":  snap.get("blink_count", 0),
            "total":        snap["total"],
            "moving":       snap["moving"],
            "stopped":      snap["stopped"],
            "arrived":      snap["arrived"],
            "flows":        snap.get("flows", []),
        })

    sj   = _json.dumps(snaps_js,  ensure_ascii=False)
    ej   = _json.dumps(edges_js,  ensure_ascii=False)
    nj   = _json.dumps(ns_js,     ensure_ascii=False)
    compj = _json.dumps(comp or {}, ensure_ascii=False)
    total_nodes = len(ns_js)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tanGo sim3 — Agente PPO</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
/* ── tokens — dashboard dark mode ── */
:root{{
  --bg:#0a0c10;--sur:#0f1219;--sur2:#161b26;--brd:#1e2535;
  --txt:#e8eaf0;--mut:#7a8099;
  --grn:#00e5a0;--yel:#f5a623;--red:#ff4560;--blu:#457BFF;
  --tel:#00e5a0;--pur:#457BFF;--ora:#f5a623;
  --rl:#00e5a0;   /* acento RL = accent del dashboard */
}}

/* ── reset ── */
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:var(--bg);color:var(--txt);
  font-family:'Courier New',monospace;   /* monospace da feeling técnico */
  font-size:12px;height:100vh;overflow:hidden;
  display:flex;flex-direction:column;
}}

/* ── header ── */
header{{
  padding:7px 16px;
  background:linear-gradient(90deg,#080c12 0%,#0a111e 60%,#060e1a 100%);
  border-bottom:1px solid var(--rl);
  display:flex;align-items:center;justify-content:space-between;
  position:relative;
}}
header::after{{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--rl),transparent);
  animation:scanline 3s linear infinite;
}}
@keyframes scanline{{0%{{background-position:0% 0%}}100%{{background-position:200% 0%}}}}
.logo{{font-size:16px;font-weight:700;letter-spacing:.12em;
       color:#fff;font-family:'Courier New',monospace}}
.logo .tg{{color:var(--rl);text-shadow:0 0 12px var(--rl)}}
.logo .s3{{color:var(--rl);font-size:11px;margin-left:6px;
           background:rgba(0,229,255,.12);border:1px solid var(--rl);
           padding:1px 7px;border-radius:3px}}
.badges{{display:flex;gap:7px;align-items:center}}
.badge{{
  font-size:10px;padding:2px 9px;border-radius:2px;
  background:rgba(255,255,255,.05);border:1px solid var(--brd);
  color:var(--mut);font-family:'Courier New',monospace;
  transition:color .2s,border-color .2s;
}}
.badge.live{{border-color:var(--rl);color:var(--rl)}}
.badge.warn{{border-color:var(--yel);color:var(--yel)}}

/* ── layout ── */
.layout{{display:flex;flex:1;overflow:hidden}}
#map{{flex:1;padding-bottom:44px}}
aside{{
  width:300px;background:var(--sur);
  border-left:1px solid var(--brd);
  display:flex;flex-direction:column;overflow:hidden;
}}

/* ── secciones aside ── */
.sec{{padding:10px 12px;border-bottom:1px solid var(--brd)}}
.sec h3{{
  font-size:9px;font-weight:700;text-transform:uppercase;
  letter-spacing:.14em;color:var(--rl);margin-bottom:8px;
  display:flex;align-items:center;gap:6px;
}}
.sec h3::before{{content:'▸';opacity:.6}}

/* ── controles ── */
.btn-row{{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px}}
.btn{{
  padding:5px 11px;border:1px solid var(--brd);border-radius:2px;
  font-size:10px;font-weight:700;cursor:pointer;
  font-family:'Courier New',monospace;transition:all .15s;
  background:var(--sur2);color:var(--txt);
}}
.btn:hover{{border-color:var(--rl);color:var(--rl)}}
.btn.primary{{background:rgba(0,229,255,.15);border-color:var(--rl);color:var(--rl)}}
.row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
label.lbl{{font-size:10px;color:var(--mut);min-width:72px}}
input[type=range]{{flex:1;accent-color:var(--rl)}}
.val{{font-size:10px;min-width:34px;text-align:right;color:var(--txt)}}

/* ── stats grid ── */
.sg{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}
.st{{
  background:var(--bg);border:1px solid var(--brd);border-radius:3px;
  padding:6px 8px;transition:border-color .3s;
}}
.st:hover{{border-color:var(--rl)}}
.sv{{font-size:16px;font-weight:700;font-family:'Courier New',monospace}}
.sl{{font-size:9px;color:var(--mut);margin-top:1px}}

/* ── reward chart ── */
#reward-canvas{{
  width:100%;height:64px;display:block;
  border:1px solid var(--brd);border-radius:2px;
  background:var(--bg);
}}

/* ── decisión bar ── */
.dec-bar{{height:8px;border-radius:2px;background:var(--brd);margin:4px 0;overflow:hidden}}
.dec-fill-kept{{height:100%;background:var(--mut);display:inline-block;transition:width .3s}}
.dec-fill-chng{{height:100%;background:var(--rl);display:inline-block;transition:width .3s}}

/* ── comparativa sparklines ── */
.spark-row{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.spark-lbl{{font-size:9px;min-width:36px;color:var(--mut)}}
.spark-bar{{flex:1;height:12px;border-radius:1px;position:relative;overflow:hidden;
            background:var(--bg);border:1px solid var(--brd)}}
.spark-fill{{height:100%;transition:width .4s}}
.spark-val{{font-size:9px;min-width:38px;text-align:right}}

/* ── node info ── */
#ni{{flex:1;overflow-y:auto;padding:10px;scrollbar-width:thin;scrollbar-color:var(--brd) transparent}}
.ic{{
  background:var(--bg);border:1px solid var(--brd);border-radius:3px;
  padding:9px;margin-bottom:6px;
}}
.ic.rl-active{{border-color:var(--rl);box-shadow:0 0 10px rgba(0,229,255,.15)}}
.ic h4{{font-size:11px;font-weight:700;margin-bottom:6px;color:var(--txt)}}
.ir{{display:flex;justify-content:space-between;font-size:10px;color:var(--mut);margin:2px 0}}
.ir span{{color:var(--txt)}}
.pill{{display:inline-block;padding:1px 6px;border-radius:2px;font-size:9px;font-weight:700}}
.pg{{background:rgba(34,197,94,.15);color:var(--grn);border:1px solid rgba(34,197,94,.3)}}
.py{{background:rgba(234,179,8,.15);color:var(--yel);border:1px solid rgba(234,179,8,.3)}}
.pr{{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}}
.prl{{background:rgba(0,229,255,.15);color:var(--rl);border:1px solid var(--rl)}}

/* ── toolbar ── */
#tb{{
  position:fixed;bottom:0;left:0;right:300px;
  background:var(--sur);border-top:1px solid var(--rl);
  padding:6px 14px;display:flex;gap:8px;align-items:center;
  z-index:9999;
}}
#tb .lbl{{font-size:9px;color:var(--rl);font-weight:700;text-transform:uppercase;letter-spacing:.1em}}

/* ── mapa ── */
.leaflet-tile{{filter:brightness(.55) saturate(.4) hue-rotate(190deg)}}
.leaflet-container{{background:var(--bg)}}

/* ── pulso animado en semáforos cambiados ── */
@keyframes pulse-ring{{
  0%{{transform:scale(.8);opacity:1}}
  100%{{transform:scale(2.4);opacity:0}}
}}
.pulse-marker{{
  border-radius:50%;background:rgba(0,229,255,.6);
  animation:pulse-ring .9s ease-out forwards;
  pointer-events:none;
}}

/* ── scrollbar ── */
::-webkit-scrollbar{{width:4px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--brd);border-radius:2px}}
</style>
</head>
<body>

<header>
  <div class="logo">tan<span class="tg">Go</span><span class="s3">SIM3 · PPO</span></div>
  <div class="badges">
    <span class="badge live" id="bt">TICK 0/{ns}</span>
    <span class="badge" id="breward">R: —</span>
    <span class="badge" id="bchanged">∆ —</span>
    <span class="badge live">PPO · checkpoint_00550</span>
  </div>
</header>

<div class="layout">
  <div id="map"></div>

  <aside>
    <!-- Reproducción -->
    <div class="sec">
      <h3>Reproducción</h3>
      <div class="btn-row">
        <button class="btn primary" id="pp">▶ PLAY</button>
        <button class="btn" id="st2">▷ PASO</button>
        <button class="btn" id="rs">↺ RESET</button>
      </div>
      <div class="row">
        <label class="lbl">Velocidad</label>
        <input type="range" id="spd" min="150" max="2500" value="600" step="50">
        <span class="val" id="vs">0.6s</span>
      </div>
      <div class="row">
        <label class="lbl">Frame</label>
        <input type="range" id="fsl" min="0" max="{ns-1}" value="0">
        <span class="val" id="vf">0/{ns-1}</span>
      </div>
      <div class="row" style="margin-top:4px">
        <label class="lbl" style="font-size:10px">Heatmap</label>
        <input type="checkbox" id="heat-toggle" checked>
        <label class="lbl" style="font-size:10px;min-width:auto;margin-left:8px">Partículas</label>
        <input type="checkbox" id="part-toggle" checked>
        <label class="lbl" style="font-size:10px;min-width:auto;margin-left:8px">Pulsos</label>
        <input type="checkbox" id="pulse-toggle" checked>
      </div>
    </div>

    <!-- Reward curve -->
    <div class="sec">
      <h3>Reward del Agente</h3>
      <canvas id="reward-canvas"></canvas>
      <div style="display:flex;justify-content:space-between;margin-top:4px">
        <span style="font-size:9px;color:var(--mut)">Tick 1</span>
        <span class="val" id="srewardsum" style="color:var(--rl)">Σ 0.000</span>
        <span style="font-size:9px;color:var(--mut)">Tick {ns}</span>
      </div>
    </div>

    <!-- Decisiones del agente -->
    <div class="sec">
      <h3>Decisiones PPO</h3>
      <div class="ir">
        <span style="color:var(--rl)">▸ Cambió</span>
        <span id="snchanged" style="color:var(--rl);font-weight:700">0</span>
      </div>
      <div class="ir">
        <span style="color:var(--mut)">◈ Mantuvo</span>
        <span id="snkept" style="color:var(--mut)">0</span>
      </div>
      <div class="dec-bar" style="margin-top:6px">
        <span class="dec-fill-kept" id="dkept" style="width:50%"></span>
        <span class="dec-fill-chng" id="dchng" style="width:50%"></span>
      </div>
    </div>

    <!-- Stats -->
    <div class="sec">
      <h3>Estado de la Red</h3>
      <div class="sg">
        <div class="st"><div class="sv" id="sk" style="color:var(--rl)">0</div><div class="sl">Tick</div></div>
        <div class="st"><div class="sv" id="se">0</div><div class="sl">Entidades</div></div>
        <div class="st"><div class="sv" id="smoving" style="color:var(--tel)">0</div><div class="sl">Moviéndose</div></div>
        <div class="st"><div class="sv" id="sstopped" style="color:var(--red)">0</div><div class="sl">Detenidas</div></div>
        <div class="st"><div class="sv" id="sa" style="color:var(--pur)">0</div><div class="sl">Llegaron</div></div>
        <div class="st"><div class="sv" id="sg2" style="color:var(--grn)">0</div><div class="sl">Verde</div></div>
        <div class="st"><div class="sv" id="sr" style="color:var(--red)">0</div><div class="sl">Rojo</div></div>
        <div class="st"><div class="sv" id="sblink" style="color:#f59e0b">0</div><div class="sl">Blink</div></div>
        <div class="st"><div class="sv" id="sblind" style="color:var(--mut)">{sum(1 for n in __import__('json').loads(nj).values() if not n['has_light'])}</div><div class="sl">Sin semáf.</div></div>
      </div>
    </div>

    <!-- Comparativa -->
    <div class="sec" id="comp-sec" style="display:none">
      <h3>Comparativa Final</h3>
      <div id="comp-content"></div>
    </div>

    <!-- Info nodo -->
    <div id="ni">
      <div style="color:var(--mut);font-size:11px;text-align:center;margin-top:24px;line-height:1.8">
        <div style="font-size:24px;opacity:.3">◈</div>
        Clic en nodo para<br>ver decisiones del agente
      </div>
    </div>
  </aside>
</div>

<!-- toolbar -->
<div id="tb">
  <span class="lbl">sim3 · PPO</span>
  <span style="color:var(--brd)">|</span>
  <span style="color:var(--mut);font-size:9px">
    Cyan = agente cambió semáforo &nbsp;·&nbsp;
    Tamaño ∝ presión &nbsp;·&nbsp;
    Calor ∝ flujo acumulado
  </span>
</div>

<script>
// ── Datos ──────────────────────────────────────────────────────────────────
const S={sj};
const E={ej};
const N={nj};
const COMP={compj};
const NS={ns};

const PHASE_C={{green:'#00e5a0',yellow:'#f5a623',red:'#ff4560',blink:'#f5a623'}};
const ITYPE_R={{master:'#f5a623',normal:'#457BFF',blind:'#1e2535'}};
const CAT_C={{MAIN_AVENUE:'#1d4ed8',SECONDARY_AVENUE:'#4c1d95',
              STREET:'#1e293b',HIGHWAY:'#0f172a',ALLEY:'#0f172a'}};
const CAT_W={{MAIN_AVENUE:5,SECONDARY_AVENUE:3,STREET:1.5,HIGHWAY:6,ALLEY:1}};
const RL_COLOR='#00e5a0';

let _blinkOn=true;
setInterval(()=>{{_blinkOn=!_blinkOn;}}, 600);

// ── Mapa ───────────────────────────────────────────────────────────────────
const map=L.map('map').setView([{clat},{clon}],14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'CartoDB',maxZoom:19}}).addTo(map);

// Aristas estáticas
E.forEach(e=>L.polyline(
  [[e.lat_a,e.lon_a],[e.lat_b,e.lon_b]],
  {{color:CAT_C[e.category]||'#1e293b',weight:CAT_W[e.category]||1.5,opacity:.55}}
).addTo(map));

// Nodos
const NM={{}};
Object.entries(N).forEach(([id,nd])=>{{
  const r=nd.itype==='master'?13:nd.itype==='normal'?10:7;
  const m=L.circleMarker([nd.lat,nd.lon],{{
    radius:r,
    color:ITYPE_R[nd.itype]||'#1e2535',
    fillColor:PHASE_C.red,fillOpacity:.85,
    weight:1+nd.node_weight*2,
  }}).addTo(map);
  m.bindTooltip(`<b>${{id}}</b> ${{nd.geo_label}}<br>${{nd.name}}`,
    {{permanent:false,direction:'top'}});
  m.on('click',()=>{{
    map.setView([nd.lat,nd.lon],16,{{animate:true,duration:.4}});
    showNode(id,S[fi]);
  }});
  NM[id]=m;
}});

// Capas dinámicas
const PM={{}};  // partículas
const HM={{}};  // heatmap
const PL=[];    // pulsos (se autodestruyen)

function clearParticles(){{
  Object.values(PM).forEach(m=>map.removeLayer(m));
  Object.keys(PM).forEach(k=>delete PM[k]);
}}
function clearHeatmap(){{
  Object.values(HM).forEach(m=>map.removeLayer(m));
  Object.keys(HM).forEach(k=>delete HM[k]);
}}

// ── Reward canvas ──────────────────────────────────────────────────────────
const cvs=document.getElementById('reward-canvas');
const ctx2=cvs.getContext('2d');

function drawReward(curve){{
  const W=cvs.offsetWidth, H=cvs.offsetHeight;
  cvs.width=W; cvs.height=H;
  if(!curve||!curve.length)return;

  ctx2.clearRect(0,0,W,H);

  // Línea de zero
  const mn=Math.min(...curve), mx2=Math.max(...curve);
  const range=mx2-mn||1;
  const toY=v=>H-((v-mn)/range*(H-8)+4);
  const y0=toY(0);

  ctx2.strokeStyle='rgba(78,90,110,.4)';
  ctx2.lineWidth=1;
  ctx2.beginPath();ctx2.moveTo(0,y0);ctx2.lineTo(W,y0);ctx2.stroke();

  // Área bajo la curva
  const grad=ctx2.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'rgba(0,229,160,.25)');
  grad.addColorStop(1,'rgba(0,229,160,.02)');
  ctx2.fillStyle=grad;
  ctx2.beginPath();
  curve.forEach((v,i)=>{{
    const x=i/(NS-1||1)*W, y=toY(v);
    i===0?ctx2.moveTo(x,y):ctx2.lineTo(x,y);
  }});
  ctx2.lineTo(W,H);ctx2.lineTo(0,H);ctx2.closePath();ctx2.fill();

  // Línea
  ctx2.strokeStyle=RL_COLOR;
  ctx2.lineWidth=1.5;
  ctx2.shadowColor=RL_COLOR;
  ctx2.shadowBlur=4;
  ctx2.beginPath();
  curve.forEach((v,i)=>{{
    const x=i/(NS-1||1)*W, y=toY(v);
    i===0?ctx2.moveTo(x,y):ctx2.lineTo(x,y);
  }});
  ctx2.stroke();
  ctx2.shadowBlur=0;

  // Punto actual
  const li=curve.length-1;
  const lx=li/(NS-1||1)*W, ly=toY(curve[li]);
  ctx2.fillStyle=RL_COLOR;
  ctx2.beginPath();ctx2.arc(lx,ly,3,0,Math.PI*2);ctx2.fill();
}}

// ── Pulsos ─────────────────────────────────────────────────────────────────
function firePulses(changedIds){{
  if(!document.getElementById('pulse-toggle').checked) return;
  changedIds.forEach(nid=>{{
    const nd=N[nid];
    if(!nd)return;
    const r=nd.itype==='master'?22:nd.itype==='normal'?17:13;
    // Usar DivIcon con CSS animation
    const icon=L.divIcon({{
      html:`<div class="pulse-marker" style="width:${{r*2}}px;height:${{r*2}}px;
             margin-left:-${{r}}px;margin-top:-${{r}}px;
             background:rgba(0,229,160,.55)"></div>`,
      className:'',
      iconSize:[0,0],
    }});
    const m=L.marker([nd.lat,nd.lon],{{icon,interactive:false,zIndexOffset:1000}}).addTo(map);
    setTimeout(()=>map.removeLayer(m), 950);
  }});
}}

// ── Apply frame ────────────────────────────────────────────────────────────
function apply(snap){{
  const showHeat=document.getElementById('heat-toggle').checked;
  const showPart=document.getElementById('part-toggle').checked;

  // Nodos
  Object.entries(snap.nodes).forEach(([id,nd])=>{{
    const m=NM[id];if(!m)return;
    const st=N[id];
    const baseR=st.itype==='master'?13:st.itype==='normal'?10:7;
    // Radio crece con presión (como sim2 con heat)
    const pressNorm=Math.min(1,(nd.pressure||0)/Math.max(nd.threshold||1,1));
    const heatR=baseR+pressNorm*6;
    const fillCol=!nd.has_light
      ?'#0f1219'
      :nd.phase==='blink'?(_blinkOn?PHASE_C.blink:'#0f1219')
      :(PHASE_C[nd.phase]||PHASE_C.red);
    // Borde verde neón si el agente lo cambió este tick
    const strokeCol=nd.changed?RL_COLOR:(ITYPE_R[nd.itype]||'#1e2535');
    const strokeW=nd.changed?3:(1+st.node_weight*2);
    m.setStyle({{
      fillColor:fillCol,fillOpacity:!nd.has_light?.2:.88,
      radius:heatR,
      color:strokeCol,weight:strokeW,
    }});
    m.off('click');m.on('click',()=>{{
      map.setView([nd.lat,nd.lon],16,{{animate:true,duration:.4}});
      showNode(id,snap);
    }});
  }});

  // Heatmap
  clearHeatmap();
  if(showHeat){{
    Object.entries(snap.heatmap||{{}}).forEach(([nid,heat])=>{{
      if(heat<0.05)return;
      const nd=N[nid];if(!nd)return;
      const hm=L.circleMarker([nd.lat,nd.lon],{{
        radius:8+heat*20,color:'transparent',
        fillColor:'#f5a623',fillOpacity:heat*0.3,
        interactive:false,
      }}).addTo(map);
      HM[nid]=hm;
    }});
  }}

  // Partículas
  clearParticles();
  if(showPart){{
    (snap.particles||[]).forEach(p=>{{
      const col=p.vtype==='EMERGENCY'?'#ff4560':p.vtype==='BUS'?'#457BFF':'#e8eaf0';
      const r=p.type==='pedestrian'?3:5;
      const pm=L.circleMarker([p.lat,p.lon],{{
        radius:r,color:'rgba(0,0,0,.3)',
        fillColor:col,fillOpacity:.9,weight:1,
      }}).addTo(map);
      pm.bindTooltip(
        `${{p.vtype}} · ${{p.speed_kmh}}km/h<br>${{p.origin}}→${{p.destination}}<br>${{p.progress.toFixed(0)}}%`,
        {{permanent:false,direction:'top'}}
      );
      PM[p.id]=pm;
    }});
  }}

  // Reward chart
  drawReward(snap.reward_curve);

  // Decisiones
  const total=snap.n_changed+snap.n_kept||1;
  const pChng=snap.n_changed/total*100;
  const pKept=snap.n_kept/total*100;
  document.getElementById('dkept').style.width=pKept+'%';
  document.getElementById('dchng').style.width=pChng+'%';
  document.getElementById('snchanged').textContent=snap.n_changed;
  document.getElementById('snkept').textContent=snap.n_kept;

  // Stats
  // Contar fases separadas desde los nodos del snap
  let gc=0,rc=0,bc=0;
  Object.values(snap.nodes).forEach(nd=>{{
    if(!nd.has_light) return;
    if(nd.phase==='green')       gc++;
    else if(nd.phase==='red')    rc++;
    else if(nd.phase==='blink')  bc++;
  }});
  document.getElementById('sk').textContent=snap.tick;
  document.getElementById('se').textContent=snap.total||0;
  document.getElementById('smoving').textContent=snap.moving||0;
  document.getElementById('sstopped').textContent=snap.stopped||0;
  document.getElementById('sa').textContent=snap.arrived||0;
  document.getElementById('sg2').textContent=snap.green_count||0;
  document.getElementById('sr').textContent=snap.red_count||0;
  document.getElementById('sblink').textContent=snap.blink_count||0;
  document.getElementById('breward').textContent='R: '+(snap.reward>=0?'+':'')+snap.reward.toFixed(3);
  document.getElementById('breward').className='badge '+(snap.reward>=0?'live':'warn');
  document.getElementById('srewardsum').textContent='Σ '+(snap.reward_sum>=0?'+':'')+snap.reward_sum.toFixed(3);
  document.getElementById('bt').textContent='TICK '+snap.tick+'/'+NS;
  document.getElementById('bchanged').textContent='∆ '+snap.n_changed+' cambios';

  // Pulsos
  firePulses(snap.changed_ids||[]);
}}

// ── Comparativa ────────────────────────────────────────────────────────────
(function renderComp(){{
  if(!COMP||!Object.keys(COMP).length) return;
  const sec=document.getElementById('comp-sec');
  const cnt=document.getElementById('comp-content');
  sec.style.display='block';

  const rows=[
    ['Detenidos/tick','avg_stopped_per_tick',true],
    ['Llegaron','total_arrived',false],
    ['% verde','pct_green',false],
  ];
  const sims=['sim0','sim1','sim3'];
  const colors={{sim0:'#7a8099',sim1:'#457BFF',sim3:'#00e5a0'}};

  rows.forEach(([label,key,lowerBetter])=>{{
    const vals=sims.map(s=>COMP[s]?COMP[s][key]:null).filter(v=>v!==null);
    const maxV=Math.max(...vals)||1;
    const div=document.createElement('div');
    div.style.marginBottom='8px';
    div.innerHTML=`<div style="font-size:9px;color:var(--mut);margin-bottom:3px">${{label}}</div>`;
    sims.forEach(s=>{{
      if(!COMP[s]) return;
      const v=COMP[s][key];
      const pct=lowerBetter?(1-v/maxV)*100:v/maxV*100;
      div.innerHTML+=`
        <div class="spark-row">
          <span class="spark-lbl" style="color:${{colors[s]}}">${{s}}</span>
          <div class="spark-bar">
            <div class="spark-fill" style="width:${{Math.max(2,pct)}}%;background:${{colors[s]}}"></div>
          </div>
          <span class="spark-val" style="color:${{colors[s]}}">${{v}}</span>
        </div>`;
    }});
    cnt.appendChild(div);
  }});
}})();

// ── showNode ───────────────────────────────────────────────────────────────
function showNode(id,snap){{
  const nd=snap?snap.nodes[id]:null;
  const st=N[id];
  if(!nd)return;
  const pC=p=>({{'green':'pg','yellow':'py','red':'pr','blink':'py'}}[p]||'pr');
  const rlClass=nd.changed?'rl-active':'';
  document.getElementById('ni').innerHTML=`
  <div class="ic ${{rlClass}}">
    <h4>${{id}} — ${{nd.geo_label}}</h4>
    <div class="ir">Nombre<span>${{nd.name}}</span></div>
    <div class="ir">Tipo<span>${{nd.itype.toUpperCase()}}</span></div>
    ${{nd.changed
      ?`<div class="ir"><span class="pill prl">▸ AGENTE CAMBIÓ ESTE TICK</span></div>`
      :`<div class="ir" style="color:var(--mut)">◈ Agente mantuvo</div>`
    }}
  </div>
  <div class="ic">
    <h4>Semáforo</h4>
    <div class="ir">Estado<span><span class="pill ${{pC(nd.phase)}}">${{nd.phase.toUpperCase()}}</span></span></div>
    <div class="ir">Presión<span style="color:${{nd.pressure>=nd.threshold?'var(--red)':'var(--tel)'}}">${{nd.pressure}}/${{nd.threshold}}</span></div>
    <div class="ir">Ticks en fase<span>${{nd.ticks_red}}</span></div>
    <div class="ir">node_weight<span style="color:var(--blu)">${{nd.node_weight}}</span></div>
  </div>`;
}}

// ── Reproductor ────────────────────────────────────────────────────────────
let fi=0,run=false,tmr=null;
const fsl=document.getElementById('fsl');

function goTo(idx){{
  if(idx<0||idx>=NS) return;
  fi=idx; fsl.value=idx;
  document.getElementById('vf').textContent=idx+'/'+(NS-1);
  apply(S[idx]);
}}

function next(){{
  if(!run) return;
  goTo((fi+1)%NS);
  tmr=setTimeout(next, parseInt(document.getElementById('spd').value));
}}

document.getElementById('pp').addEventListener('click',()=>{{
  run=!run;
  const b=document.getElementById('pp');
  if(run){{b.textContent='⏸ PAUSA';next();}}
  else{{b.textContent='▶ PLAY';clearTimeout(tmr);}}
}});
document.getElementById('st2').addEventListener('click',()=>{{
  clearTimeout(tmr);run=false;
  document.getElementById('pp').textContent='▶ PLAY';
  goTo((fi+1)%NS);
}});
document.getElementById('rs').addEventListener('click',()=>{{
  clearTimeout(tmr);run=false;
  document.getElementById('pp').textContent='▶ PLAY';
  goTo(0);
}});
document.getElementById('spd').addEventListener('input',function(){{
  document.getElementById('vs').textContent=(this.value/1000).toFixed(1)+'s';
}});
fsl.addEventListener('input',function(){{
  clearTimeout(tmr);run=false;
  document.getElementById('pp').textContent='▶ PLAY';
  goTo(parseInt(this.value));
}});
document.getElementById('heat-toggle').addEventListener('change',()=>apply(S[fi]));
document.getElementById('part-toggle').addEventListener('change',()=>apply(S[fi]));

// Iniciar
goTo(0);
</script>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="tanGo sim3 — visualización PPO")
    parser.add_argument("--checkpoint", default=None,
                        help="Ruta al checkpoint (default: checkpoints/checkpoint_00550)")
    parser.add_argument("--ticks", type=int, default=None,
                        help="Número de ticks (default: n_ticks_per_episode del config)")
    args = parser.parse_args()

    t0 = _time.perf_counter()
    print("tanGo sim3 — Visualización del agente PPO")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cfg = load_cfg()

    # Checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        base = Path(__file__).parent / "checkpoints"
        for name in ["checkpoint_00550", "best", "final"]:
            p = base / name
            if p.exists():
                ckpt = str(p)
                break
    if ckpt is None:
        print("ERROR: no se encontró checkpoint. Usa --checkpoint <ruta>")
        sys.exit(1)
    print(f"  Checkpoint: {ckpt}")

    # Grafo
    from graph.city_loader import json_to_traffic_graph
    graph = json_to_traffic_graph(CITY_JSON)
    print(f"  Grafo: {graph.graph.number_of_nodes()} nodos, "
          f"{graph.graph.number_of_edges()} aristas\n")

    # Ticks
    n_ticks = args.ticks or cfg["environment"]["n_ticks_per_episode"]
    print(f"  Ticks: {n_ticks}\n")

    # Simular
    history = simulate_ppo(ckpt, graph, n_ticks, cfg)

    # Cargar comparativa si existe
    comp = None
    if COMP_FILE.exists():
        with open(COMP_FILE) as f:
            comp = _json.load(f)
        print(f"  Comparativa cargada: {COMP_FILE.name}")

    # Generar HTML
    print(f"\n  Generando visualización...")
    vis = build_vis(graph, history, comp)
    OUTPUT_VIS.write_text(vis, encoding="utf-8")
    print(f"  ✓ {OUTPUT_VIS}")

    elapsed = _time.perf_counter() - t0
    print(f"\n{'─'*52}")
    print(f"  Ticks generados : {len(history)}")
    print(f"  Duración        : {int(elapsed//60)}m {elapsed%60:.1f}s")
    print(f"{'─'*52}")
    print(f"\n✓ Listo. Abre: {OUTPUT_VIS.name}")