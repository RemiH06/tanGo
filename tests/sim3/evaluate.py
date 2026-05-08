"""
tests/sim3/evaluate.py
-----------------------
Evaluación del agente PPO entrenado y comparativa con sim0 y sim1.

Uso:
    # Usa el checkpoint 'best' automáticamente
    python tests/sim3/evaluate.py

    # Checkpoint específico
    python tests/sim3/evaluate.py --checkpoint tests/sim3/checkpoints/final

    # Solo comparar sim0 vs sim1 (sin PPO)
    python tests/sim3/evaluate.py --no-ppo
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CFG_FILE  = Path(__file__).parent / "sim3_config.json"
CITY_JSON = ROOT / "graph" / "city_graph.json"
COMP_FILE = Path(__file__).parent / "results" / "comparison.json"

def load_cfg():
    with open(CFG_FILE) as f:
        return json.load(f)


# ── Spawn sintético (igual que sim1 y el DAG) ─────────────────────────────────

def spawn_entities(itype_str, ctx_is_rush):
    from core.entities import Vehicle, Pedestrian, VehicleType, Direction
    from core.road import IntersectionType

    itype_map = {
        "master": IntersectionType.MASTER,
        "normal": IntersectionType.NORMAL,
        "blind":  IntersectionType.BLIND,
    }
    itype = itype_map.get(itype_str, IntersectionType.NORMAL)

    if itype == IntersectionType.MASTER:
        nv = random.randint(5, 14) if ctx_is_rush else random.randint(2, 8)
    elif itype == IntersectionType.NORMAL:
        nv = random.randint(2, 8) if ctx_is_rush else random.randint(1, 5)
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
                    random.choice(list(__import__('core.entities',
                    fromlist=['Direction']).Direction)))
            for _ in range(nv)]
    if random.random() < 0.15:
        ents.append(Pedestrian(str(uuid.uuid4())))
    return ents


# ── Evaluador genérico ────────────────────────────────────────────────────────

def run_episode(graph, ctx, algo, n_ticks, cfg, env=None):
    """
    Corre un episodio completo y retorna métricas.
    Si algo=None usa el algoritmo greedy (sim1).
    Si algo='timer' usa timers fijos (sim0).
    """
    from core.algorithm import TrafficAlgorithm
    from core.road import Phase

    algorithm = TrafficAlgorithm(graph)
    algorithm.reset()

    metrics = {
        "total_ticks":           0,
        "green_ticks_sum":       0,
        "stopped_sum":           0,
        "emergencies_stopped":   0,
        "arrived":               0,
        "total_entities_seen":   0,
        "reward_sum":            0.0,
    }

    obs = None
    if env is not None:
        obs, _ = env.reset()

    for tick in range(n_ticks):
        # Generar entidades
        is_rush = ctx.is_rush_hour if hasattr(ctx, 'is_rush_hour') else False
        entities_by_node = {
            nid: spawn_entities(inter.intersection_type.value, is_rush)
            for nid, inter in graph.intersections.items()
        }

        # Acción del agente (PPO) o algoritmo base
        if env is not None and algo is not None and obs is not None:
            action = algo.compute_single_action(obs, explore=False)
            # Aplicar acción al entorno
            obs, reward, done, _, info = env.step(action)
            metrics["reward_sum"] += reward
            if done:
                break
        else:
            # sim1: greedy SCOOT
            result = algorithm.run_tick(entities_by_node, ctx)

        # Métricas comunes
        phases = {nid: inter.current_phase.value
                  for nid, inter in graph.intersections.items()}

        green_count = sum(1 for p in phases.values() if p == 'green')
        stopped     = sum(
            len(ents) for nid, ents in entities_by_node.items()
            if phases.get(nid) == 'red'
        )
        emerg_stopped = sum(
            1 for nid, inter in graph.intersections.items()
            if inter.current_phase == Phase.RED
            for e in entities_by_node.get(nid, [])
            if hasattr(e, 'vehicle_type') and
               e.vehicle_type.value == 'EMERGENCY'
        )

        metrics["total_ticks"]         += 1
        metrics["green_ticks_sum"]     += green_count
        metrics["stopped_sum"]         += stopped
        metrics["emergencies_stopped"] += emerg_stopped
        metrics["total_entities_seen"] += sum(
            len(v) for v in entities_by_node.values())

    t = max(metrics["total_ticks"], 1)
    total_s = max(metrics["total_entities_seen"], 1)

    return {
        "pct_green":             round(metrics["green_ticks_sum"] / t * 100, 1),
        "avg_stopped_per_tick":  round(metrics["stopped_sum"] / t, 2),
        "emergencies_stopped":   metrics["emergencies_stopped"],
        "total_arrived":         metrics["arrived"],
        "reward_sum":            round(metrics["reward_sum"], 4),
        "ticks_evaluated":       t,
        "total_entities_seen":   metrics["total_entities_seen"],
    }


# ── sim0: timer fijo ──────────────────────────────────────────────────────────

def evaluate_sim0(graph, ctx, n_ticks):
    """Baseline: todos los semaforizados en verde fijo."""
    from core.road import Phase

    # Forzar todos en verde
    for inter in graph.intersections.values():
        if inter.has_traffic_light:
            inter.current_phase = Phase.GREEN

    is_rush = ctx.is_rush_hour if hasattr(ctx, 'is_rush_hour') else False
    green_total, stopped_total, ticks = 0, 0, 0
    emerg_stopped = 0

    for _ in range(n_ticks):
        entities_by_node = {
            nid: spawn_entities(inter.intersection_type.value, is_rush)
            for nid, inter in graph.intersections.items()
        }
        green_count  = sum(1 for i in graph.intersections.values()
                           if i.current_phase == Phase.GREEN)
        stopped      = sum(len(ents) for nid, ents in entities_by_node.items()
                           if graph.intersections[nid].current_phase == Phase.RED)
        green_total  += green_count
        stopped_total += stopped
        ticks         += 1

    return {
        "pct_green":            round(green_total / ticks * 100, 1),
        "avg_stopped_per_tick": round(stopped_total / ticks, 2),
        "emergencies_stopped":  0,
        "total_arrived":        0,
        "reward_sum":           0,
        "ticks_evaluated":      ticks,
    }


# ── sim1: greedy SCOOT ────────────────────────────────────────────────────────

def evaluate_sim1(graph, ctx, n_ticks, cfg):
    from core.algorithm import TrafficAlgorithm
    from core.road import Phase

    algo = TrafficAlgorithm(graph)
    algo.reset()
    is_rush = ctx.is_rush_hour if hasattr(ctx, 'is_rush_hour') else False

    green_total, stopped_total, ticks = 0, 0, 0
    emerg_stopped = 0

    for _ in range(n_ticks):
        entities_by_node = {
            nid: spawn_entities(inter.intersection_type.value, is_rush)
            for nid, inter in graph.intersections.items()
        }
        result = algo.run_tick(entities_by_node, ctx)
        stopped = sum(
            len(ents) for nid, ents in entities_by_node.items()
            if graph.intersections[nid].current_phase == Phase.RED
        )
        emerg_stopped += sum(
            1 for nid, inter in graph.intersections.items()
            if inter.current_phase == Phase.RED
            for e in entities_by_node.get(nid, [])
            if hasattr(e, 'vehicle_type') and e.vehicle_type.value == 'EMERGENCY'
        )
        green_total   += result.green_count
        stopped_total += stopped
        ticks         += 1

    return {
        "pct_green":            round(green_total / ticks * 100, 1),
        "avg_stopped_per_tick": round(stopped_total / ticks, 2),
        "emergencies_stopped":  emerg_stopped,
        "total_arrived":        0,
        "reward_sum":           0,
        "ticks_evaluated":      ticks,
    }


# ── sim3: PPO ────────────────────────────────────────────────────────────────

def evaluate_sim3(checkpoint_path, graph, ctx, n_ticks, cfg):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from tests.sim3.tango_env import TanGoEnv

    ray.init(ignore_reinit_error=True)
    register_env("TanGoEnv", lambda c: TanGoEnv(c))

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner       = False,
            enable_env_runner_and_connector_v2 = False,
        )
        .environment("TanGoEnv", env_config={
            "spawn_rate":   cfg["environment"]["spawn_rate"],
            "max_entities": cfg["environment"]["max_entities"],
        })
        .framework("torch")
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
    )
    config.model = {"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu", "max_seq_len": 20}

    algo = config.build_algo()
    algo.restore(checkpoint_path)

    env = TanGoEnv({"env_config": {
        "spawn_rate":   cfg["environment"]["spawn_rate"],
        "max_entities": cfg["environment"]["max_entities"],
    }})
    obs, _ = env.reset()

    reward_sum   = 0.0
    green_total  = 0
    stopped_total = 0
    emerg_stopped = 0
    ticks = 0

    from core.road import Phase
    for _ in range(n_ticks):
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, _, info = env.step(action)
        reward_sum    += reward
        green_total   += info.get("green_count", 0)
        stopped_total += info.get("stopped",     0)
        ticks         += 1
        if done:
            break

    algo.stop()
    ray.shutdown()

    return {
        "pct_green":            round(green_total / max(ticks,1) * 100, 1),
        "avg_stopped_per_tick": round(stopped_total / max(ticks,1), 2),
        "emergencies_stopped":  emerg_stopped,
        "total_arrived":        info.get("arrived", 0),
        "reward_sum":           round(reward_sum, 4),
        "ticks_evaluated":      ticks,
    }


# ── Comparativa ───────────────────────────────────────────────────────────────

def compare(checkpoint_path=None, run_ppo=True):
    cfg = load_cfg()
    n_ticks = cfg["environment"]["n_ticks_per_episode"]

    from core.context import TrafficContext
    from graph.city_loader import json_to_traffic_graph

    # Escenario hora pico
    sc = next(s for s in cfg["environment"]["scenarios"] if s["label"] == "rush_hour")
    ctx = TrafficContext.build(
        timestamp      = datetime.fromisoformat(sc["timestamp"]),
        temperature_c  = sc["temperature_c"],
        is_raining     = sc["is_raining"],
        wind_speed_kmh = sc["wind_speed_kmh"],
        visibility_m   = sc["visibility_m"],
    )

    print(f"\ntanGo sim3 — Evaluación comparativa")
    print(f"  Escenario: rush_hour | {n_ticks} ticks\n")

    results = {}

    print("  Evaluando sim0 (timers fijos)...")
    graph = json_to_traffic_graph(CITY_JSON)
    results["sim0"] = evaluate_sim0(graph, ctx, n_ticks)
    _print(results["sim0"], "sim0")

    print("  Evaluando sim1 (greedy SCOOT)...")
    graph = json_to_traffic_graph(CITY_JSON)
    results["sim1"] = evaluate_sim1(graph, ctx, n_ticks, cfg)
    _print(results["sim1"], "sim1")

    if run_ppo and checkpoint_path:
        print(f"  Evaluando sim3 (PPO) → {checkpoint_path}")
        graph = json_to_traffic_graph(CITY_JSON)
        results["sim3"] = evaluate_sim3(checkpoint_path, graph, ctx, n_ticks, cfg)
        _print(results["sim3"], "sim3")
    elif run_ppo:
        print("  sim3: no se encontró checkpoint — omitiendo PPO")

    # Guardar resultados
    COMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMP_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Tabla resumen
    sims = [k for k in ["sim0","sim1","sim3"] if k in results]
    print(f"\n{'='*65}")
    print(f"  {'Métrica':<32} " + "".join(f"{s:>10}" for s in sims))
    print(f"  {'─'*61}")
    for key, label in [
        ("pct_green",            "% ticks en verde"),
        ("avg_stopped_per_tick", "Detenidos prom/tick"),
        ("emergencies_stopped",  "Emergencias detenidas"),
        ("reward_sum",           "Recompensa total"),
        ("ticks_evaluated",      "Ticks evaluados"),
    ]:
        vals = [str(results[s].get(key, "—")) for s in sims]
        print(f"  {label:<32} " + "".join(f"{v:>10}" for v in vals))
    print(f"{'='*65}")
    print(f"\n  Resultados → {COMP_FILE}\n")


def _print(m, name):
    print(f"    {name}: verde={m['pct_green']}% "
          f"| det/tick={m['avg_stopped_per_tick']} "
          f"| emerg={m['emergencies_stopped']} "
          f"| reward={m['reward_sum']}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación tanGo sim3")
    parser.add_argument("--checkpoint", default=None,
                        help="Ruta al checkpoint PPO (default: checkpoints/best)")
    parser.add_argument("--no-ppo", action="store_true",
                        help="Solo comparar sim0 vs sim1")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not args.no_ppo and ckpt is None:
        best = Path(__file__).parent / "checkpoints" / "best"
        final = Path(__file__).parent / "checkpoints" / "final"
        if best.exists():
            ckpt = str(best)
        elif final.exists():
            ckpt = str(final)

    compare(checkpoint_path=ckpt, run_ppo=not args.no_ppo)