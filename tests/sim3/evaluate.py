"""
tests/sim3/evaluate.py
-----------------------
Evaluación del modelo entrenado y comparativa con sim0 y sim1.

Carga el checkpoint del agente PPO entrenado y lo ejecuta
sobre los mismos escenarios que sim0 y sim1 para comparar:

  sim0 (timer fijo)   → baseline sin inteligencia
  sim1 (greedy SCOOT) → algoritmo tanGo base
  sim3 (PPO)          → algoritmo tanGo con RL

Métricas comparadas:
  - Tiempo promedio de espera por entidad
  - Porcentaje de ticks con semáforo verde
  - Número de emergencias detenidas
  - Entidades que llegaron a destino

Genera:
  - tests/sim3/results/comparison.json
  - tests/sim3/tango_vis_sim3.html (visor igual que sim1/sim2)

Uso:
    python tests/sim3/evaluate.py
    python tests/sim3/evaluate.py --checkpoint tests/sim3/checkpoints/best
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from core.context    import TrafficContext
from core.algorithm  import TrafficAlgorithm, TICK_DURATION_S
from core.movement   import MovementEngine
from core.road       import Phase
from core.entities   import Vehicle, VehicleType
from graph.city_loader import json_to_traffic_graph
from tests.sim3.tango_env import TanGoEnv

CFG_FILE  = Path(__file__).parent / "sim3_config.json"
CITY_JSON = ROOT / "graph" / "city_graph.json"
OUTPUT_VIS = Path(__file__).parent / "tango_vis_sim3.html"
COMP_FILE  = Path(__file__).parent / "results" / "comparison.json"


def load_cfg() -> dict:
    with open(CFG_FILE, encoding="utf-8") as f:
        return json.load(f)


# ── Evaluación de un agente ───────────────────────────────────────────────────

def evaluate_ppo(checkpoint_path: str, graph, ctx: TrafficContext,
                 n_ticks: int, cfg: dict) -> dict:
    """Evalúa el agente PPO sobre un escenario."""
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    ray.init(ignore_reinit_error=True)
    register_env("TanGoEnv", lambda c: TanGoEnv(c))

    algo = PPOConfig().environment("TanGoEnv").framework("torch").build()
    algo.restore(checkpoint_path)

    env = TanGoEnv({"env_config": {
        "spawn_rate":   cfg["environment"]["spawn_rate"],
        "max_entities": cfg["environment"]["max_entities"],
    }})
    obs, _ = env.reset()

    metrics = _empty_metrics()
    for tick in range(n_ticks):
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, _, info = env.step(action)
        _accumulate(metrics, info, env)
        if done:
            break

    algo.stop()
    ray.shutdown()
    return _finalize(metrics, n_ticks)


def evaluate_sim1(graph, ctx: TrafficContext,
                  n_ticks: int, cfg: dict) -> dict:
    """Evalúa el algoritmo greedy de sim1 (TrafficAlgorithm)."""
    from tests.sim1.tango_sim import spawn_for_node

    algo     = TrafficAlgorithm(graph)
    algo.reset()
    movement = MovementEngine(
        graph,
        spawn_rate   = cfg["environment"]["spawn_rate"],
        max_entities = cfg["environment"]["max_entities"],
    )

    metrics = _empty_metrics()
    for tick in range(n_ticks):
        phases = {nid: inter.current_phase.value
                  for nid, inter in graph.intersections.items()}
        entities = movement.tick(ctx, phases)
        result   = algo.run_tick(entities, ctx)
        stats    = movement.get_stats(phases)

        info = {
            "tick":        tick + 1,
            "green_count": result.green_count,
            "stopped":     stats.stopped,
            "moving":      stats.moving,
            "arrived":     len(movement._arrived),
            "reward":      0,
        }
        _accumulate_simple(metrics, info, result, entities, graph)

    return _finalize(metrics, n_ticks)


def evaluate_sim0(graph, ctx: TrafficContext, n_ticks: int) -> dict:
    """Evalúa sim0 (timers fijos) para baseline."""
    import importlib.util
    ta_path = ROOT / "tests" / "sim0" / "timer_algorithm.py"
    spec = importlib.util.spec_from_file_location("timer_algorithm_sim0", ta_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TimerAlgorithm = mod.TimerAlgorithm

    from tests.sim1.tango_sim import spawn_for_node

    algo = TimerAlgorithm(graph)
    algo.reset()

    metrics = _empty_metrics()
    for tick in range(n_ticks):
        entities = {
            nid: spawn_for_node(nid, inter.intersection_type, ctx)
            for nid, inter in graph.intersections.items()
        }
        result = algo.run_tick(entities, ctx)
        info = {
            "tick":        tick + 1,
            "green_count": result.green_count,
            "stopped":     0,
            "moving":      result.total_entities,
            "arrived":     0,
            "reward":      0,
        }
        _accumulate_timer(metrics, info, result, entities, graph)

    return _finalize(metrics, n_ticks)


# ── Métricas ──────────────────────────────────────────────────────────────────

def _empty_metrics() -> dict:
    return {
        "total_ticks":       0,
        "green_ticks":       0,
        "stopped_entity_ticks": 0,
        "emergencies_stopped": 0,
        "arrived":           0,
        "total_entities":    0,
    }


def _accumulate(metrics: dict, info: dict, env: TanGoEnv) -> None:
    metrics["total_ticks"]    += 1
    metrics["green_ticks"]    += info["green_count"]
    metrics["stopped_entity_ticks"] += info["stopped"]
    metrics["arrived"]        += info.get("arrived", 0)


def _accumulate_simple(metrics: dict, info: dict, result,
                        entities: dict, graph) -> None:
    metrics["total_ticks"]   += 1
    metrics["green_ticks"]   += result.green_count
    metrics["stopped_entity_ticks"] += info["stopped"]
    metrics["arrived"]       += info.get("arrived", 0)
    metrics["total_entities"] = max(metrics["total_entities"],
                                    result.total_entities)
    # Emergencias detenidas
    for nid, inter in graph.intersections.items():
        if inter.current_phase == Phase.RED:
            for e in entities.get(nid, []):
                if isinstance(e, Vehicle) and e.vehicle_type == VehicleType.EMERGENCY:
                    metrics["emergencies_stopped"] += 1


def _accumulate_timer(metrics: dict, info: dict, result,
                       entities: dict, graph) -> None:
    metrics["total_ticks"]   += 1
    metrics["green_ticks"]   += result.green_count
    metrics["total_entities"] = max(metrics["total_entities"],
                                    result.total_entities)


def _finalize(metrics: dict, n_ticks: int) -> dict:
    ticks = max(metrics["total_ticks"], 1)
    return {
        "pct_green":          round(metrics["green_ticks"] / ticks * 100, 1),
        "avg_stopped_per_tick": round(metrics["stopped_entity_ticks"] / ticks, 2),
        "emergencies_stopped": metrics["emergencies_stopped"],
        "total_arrived":      metrics["arrived"],
        "ticks_evaluated":    ticks,
    }


# ── Comparativa ───────────────────────────────────────────────────────────────

def compare(checkpoint_path: str | None) -> None:
    cfg    = load_cfg()
    graph  = json_to_traffic_graph(CITY_JSON)
    n_ticks = cfg["environment"]["n_ticks_per_episode"]

    # Usar escenario de hora pico para la comparativa
    sc = next(s for s in cfg["environment"]["scenarios"]
              if s["label"] == "rush_hour")
    ctx = TrafficContext.build(
        timestamp      = datetime.fromisoformat(sc["timestamp"]),
        temperature_c  = sc["temperature_c"],
        is_raining     = sc["is_raining"],
        wind_speed_kmh = sc["wind_speed_kmh"],
        visibility_m   = sc["visibility_m"],
    )

    print("\ntanGo sim3 — Evaluación comparativa")
    print(f"  Escenario: {sc['label']} | {n_ticks} ticks\n")

    results = {}

    print("  Evaluando sim0 (timers fijos)...")
    results["sim0"] = evaluate_sim0(graph, ctx, n_ticks)
    _print_metrics("sim0", results["sim0"])

    print("  Evaluando sim1 (greedy SCOOT)...")
    results["sim1"] = evaluate_sim1(graph, ctx, n_ticks, cfg)
    _print_metrics("sim1", results["sim1"])

    if checkpoint_path:
        print(f"  Evaluando sim3 (PPO) desde {checkpoint_path}...")
        results["sim3"] = evaluate_ppo(checkpoint_path, graph, ctx, n_ticks, cfg)
        _print_metrics("sim3", results["sim3"])

    # Guardar comparativa
    COMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMP_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Resumen
    print(f"\n{'='*60}")
    print(f"  {'Métrica':<30} {'sim0':>8} {'sim1':>8} {'sim3':>8}")
    print(f"  {'─'*56}")
    for key, label in [
        ("pct_green",            "% ticks en verde"),
        ("avg_stopped_per_tick", "Detenidos prom/tick"),
        ("emergencies_stopped",  "Emergencias det."),
        ("total_arrived",        "Llegaron a destino"),
    ]:
        vals = [results.get(s, {}).get(key, "-") for s in ["sim0","sim1","sim3"]]
        print(f"  {label:<30} {str(vals[0]):>8} {str(vals[1]):>8} {str(vals[2]):>8}")
    print(f"{'='*60}")
    print(f"\n  Resultados → {COMP_FILE}")


def _print_metrics(name: str, m: dict) -> None:
    print(f"    {name}: verde={m['pct_green']}% "
          f"| det/tick={m['avg_stopped_per_tick']} "
          f"| emerg={m['emergencies_stopped']} "
          f"| llegaron={m['total_arrived']}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluacion tanGo sim3")
    parser.add_argument("--checkpoint", default=None,
                        help="Ruta al checkpoint del agente PPO")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        # Buscar el mejor checkpoint automáticamente
        best = ROOT / "tests" / "sim3" / "checkpoints" / "best"
        if best.exists():
            ckpt = str(best)
            print(f"  Usando checkpoint: {ckpt}")
        else:
            print("  No se encontró checkpoint de PPO — solo comparando sim0 vs sim1")

    compare(ckpt)