"""
tests/sim3/train.py  —  Ray 2.55 / PPO
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from tests.sim3.tango_env import TanGoEnv

CFG_FILE = Path(__file__).parent / "sim3_config.json"

def load_cfg():
    with open(CFG_FILE) as f:
        return json.load(f)

def save_metrics(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def train(resume=False):
    cfg = load_cfg()
    tr  = cfg["training"]
    env = cfg["environment"]
    paths = cfg["paths"]

    ckpt_dir    = ROOT / paths["checkpoint_dir"]
    metrics_path = ROOT / paths["results_dir"] / "training_metrics.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\ntanGo sim3 — PPO  (Ray 2.55)")
    print(f"  Episodios={tr['n_episodes']}  Workers={tr['n_workers']}  LR={tr['lr']}\n")

    ray.init(ignore_reinit_error=True,
             num_cpus=max(2, tr["n_workers"] + 1))
    register_env("TanGoEnv", lambda c: TanGoEnv(c))

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner       = False,
            enable_env_runner_and_connector_v2 = False,
        )
        .environment(
            env="TanGoEnv",
            env_config={
                "spawn_rate":   env["spawn_rate"],
                "max_entities": env["max_entities"],
            },
        )
        .framework("torch")
        .env_runners(
            num_env_runners         = tr["n_workers"],
            rollout_fragment_length = tr["rollout_fragment"],
        )
        # Solo parámetros PPO-específicos aquí
        .training(
            lambda_    = tr["lambda_gae"],
            clip_param = tr["clip_param"],
        )
        .resources(num_gpus=0)
    )

    # Parámetros base — se asignan al objeto directamente (Ray 2.55)
    config.lr                = tr["lr"]
    config.gamma             = tr["gamma"]
    config.train_batch_size  = tr["train_batch_size"]
    config.sgd_minibatch_size = tr["sgd_minibatch_size"]
    config.num_sgd_iter       = tr["num_sgd_iter"]
    config.model = {
        "fcnet_hiddens":    [256, 256, 128],
        "fcnet_activation": "relu",
        "max_seq_len": 20,
    }

    algo = config.build_algo()

    if resume:
        ckpts = sorted(ckpt_dir.glob("checkpoint_*"))
        if ckpts:
            algo.restore(str(ckpts[-1]))
            print(f"  Reanudando: {ckpts[-1]}")

    history, best = [], float("-inf")
    t0 = time.perf_counter()

    for ep in range(1, tr["n_episodes"] + 1):
        result  = algo.train()
        reward  = result.get("episode_reward_mean") or 0
        ep_len  = result.get("episode_len_mean")    or 0
        steps   = result.get("timesteps_total")     or 0

        history.append({"episode": ep, "reward_mean": round(reward,4),
                         "episode_len": round(ep_len,1), "timesteps_total": steps})

        if reward > best:
            best = reward
            algo.save(str(ckpt_dir / "best"))
            print(f"  ★  ep {ep:4d} | reward={reward:+.4f}  (nuevo mejor)")

        if ep % tr["checkpoint_freq"] == 0:
            algo.save(str(ckpt_dir / f"checkpoint_{ep:05d}"))
            print(f"  →  ep {ep:4d}/{tr['n_episodes']} | "
                  f"reward={reward:+.4f} | len={ep_len:.0f} | "
                  f"steps={steps:,} | {(time.perf_counter()-t0)/60:.1f}min")
            save_metrics(history, metrics_path)

    algo.save(str(ckpt_dir / "final"))
    save_metrics(history, metrics_path)
    print(f"\n  Listo — mejor reward: {best:.4f} | "
          f"{(time.perf_counter()-t0)/60:.1f} min")
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    train(resume=p.parse_args().resume)