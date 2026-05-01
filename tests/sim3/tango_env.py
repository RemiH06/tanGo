"""
tests/sim3/tango_env.py
------------------------
Entorno Gymnasium para el entrenamiento de RL de tanGo.

Envuelve TrafficAlgorithm + MovementEngine en la interfaz
estándar de gymnasium.Env para que RLlib pueda entrenarlo.

Arquitectura:
    Agente central — observa todo el grafo, decide todos los semáforos.
    Escalable a multi-agente (sim3b) sin cambiar este módulo.

Espacio de observación: vector de 10 features × N_semaforos
Espacio de acción: MultiDiscrete([2] * N_semaforos)
  0 = mantener fase actual
  1 = cambiar fase (rojo→verde o verde→amarillo→rojo)

Función de recompensa:
  + w_flow    × (entidades cruzando en verde / total)
  - w_wait    × (entidades detenidas × ticks esperando)
  - w_emerg   × emergencias detenidas
  + w_arrived × llegadas al destino este tick
"""

from __future__ import annotations
import math
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.context    import TrafficContext
from core.algorithm  import TrafficAlgorithm, TICK_DURATION_S
from core.movement   import MovementEngine
from core.road       import Phase, IntersectionType
from core.entities   import Vehicle, VehicleType
from graph.city_loader import json_to_traffic_graph
from graph.simulator   import TrafficGraph

CFG_FILE  = Path(__file__).parent / "sim3_config.json"
CITY_JSON = ROOT / "graph" / "city_graph.json"

N_FEATURES = 10   # features por nodo semaforizado


def _load_cfg() -> dict:
    with open(CFG_FILE, encoding="utf-8") as f:
        return json.load(f)


class TanGoEnv(gym.Env):
    """
    Entorno tanGo para RLlib / Gymnasium.

    Cada step() ejecuta 1 tick de la simulación con la acción del agente.
    El episodio termina cuando se alcanzan n_ticks_per_episode ticks.

    Parameters
    ----------
    config : dict con clave "env_config" (RLlib lo pasa automáticamente).
             Puede incluir "city_json" para sobreescribir la ciudad.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = _load_cfg()
        env_cfg = (config or {}).get("env_config", {})

        # Cargar grafo
        city_path = Path(env_cfg.get("city_json", str(CITY_JSON)))
        self.graph = json_to_traffic_graph(city_path)

        # Nodos semaforizados — orden fijo para que obs/action sean consistentes
        self.signaled: list[str] = sorted(
            [nid for nid, inter in self.graph.intersections.items()
             if inter.has_traffic_light]
        )
        self.n_signals = len(self.signaled)
        self._signal_idx = {nid: i for i, nid in enumerate(self.signaled)}

        # Parámetros del entorno
        self.n_ticks      = cfg["environment"]["n_ticks_per_episode"]
        self.spawn_rate   = env_cfg.get("spawn_rate",
                                         cfg["environment"]["spawn_rate"])
        self.max_entities = env_cfg.get("max_entities",
                                         cfg["environment"]["max_entities"])
        self.scenarios    = cfg["environment"]["scenarios"]
        self.reward_cfg   = cfg["reward"]

        # Espacios de observación y acción
        obs_dim = N_FEATURES * self.n_signals
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [2] * self.n_signals
        )

        # Estado interno
        self._tick     = 0
        self._ctx      = None
        self._algo     = None
        self._movement = None
        self._total_wait_ticks: dict[str, int] = {}  # entity_id → ticks esperando

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._tick = 0
        self._total_wait_ticks = {}

        # Seleccionar escenario aleatorio ponderado
        weights  = [s["weight"] for s in self.scenarios]
        scenario = random.choices(self.scenarios, weights=weights)[0]
        self._ctx = TrafficContext.build(
            timestamp      = datetime.fromisoformat(scenario["timestamp"]),
            temperature_c  = scenario["temperature_c"],
            is_raining     = scenario["is_raining"],
            wind_speed_kmh = scenario["wind_speed_kmh"],
            visibility_m   = scenario["visibility_m"],
        )

        # Reiniciar algoritmo y movimiento
        self._algo = TrafficAlgorithm(self.graph)
        self._algo.reset()
        self._movement = MovementEngine(
            self.graph,
            spawn_rate   = self.spawn_rate,
            max_entities = self.max_entities,
        )

        obs = self._get_obs({})
        return obs, {}

    def step(self, action: np.ndarray):
        self._tick += 1

        # 1. Aplicar acción: cambiar fases según lo que decide el agente
        self._apply_action(action)

        # 2. Mover entidades
        current_phases = {
            nid: inter.current_phase.value
            for nid, inter in self.graph.intersections.items()
        }
        entities_by_node = self._movement.tick(self._ctx, current_phases)

        # 3. Ejecutar el algoritmo (Paso 1 y 2 — presión y green wave)
        #    El Paso 3 (ajuste de fases) lo reemplaza la acción del agente
        result = self._algo.run_tick(entities_by_node, self._ctx)

        # 4. Calcular recompensa
        stats  = self._movement.get_stats(current_phases)
        reward = self._compute_reward(result, entities_by_node, stats)

        # 5. Actualizar tiempos de espera
        self._update_wait_ticks(entities_by_node, current_phases)

        # 6. Observación del nuevo estado
        obs = self._get_obs(entities_by_node)

        # 7. ¿Terminó el episodio?
        done      = self._tick >= self.n_ticks
        truncated = False

        info = {
            "tick":          self._tick,
            "green_count":   result.green_count,
            "stopped":       stats.stopped,
            "moving":        stats.moving,
            "arrived":       len(self._movement._arrived),
            "reward":        reward,
        }

        return obs, reward, done, truncated, info

    # ── Observación ───────────────────────────────────────────────────────────

    def _get_obs(self, entities_by_node: dict) -> np.ndarray:
        """
        Construye el vector de observación del estado actual.
        10 features por nodo semaforizado, concatenados en un vector plano.
        """
        obs = np.zeros(N_FEATURES * self.n_signals, dtype=np.float32)

        # Hora del día para features cíclicos
        hour = self._ctx.timestamp.hour + self._ctx.timestamp.minute / 60.0 \
               if self._ctx else 12.0
        sin_h = (math.sin(hour / 24.0 * 2 * math.pi) + 1) / 2   # → [0,1]
        cos_h = (math.cos(hour / 24.0 * 2 * math.pi) + 1) / 2   # → [0,1]

        # Presión promedio de vecinos para cada nodo
        neighbor_pressures = self._compute_neighbor_pressures()

        for i, nid in enumerate(self.signaled):
            inter = self.graph.intersections[nid]
            ents  = entities_by_node.get(nid, [])

            # Feature 0: presión normalizada
            p_norm = min(1.0, inter.pressure / max(inter.pressure_threshold, 1.0))

            # Feature 1: fase codificada
            phase_map = {Phase.RED: 0.0, Phase.GREEN: 1.0,
                         Phase.YELLOW: 0.5, Phase.BLINK: 0.25}
            phase_enc = phase_map.get(inter.current_phase, 0.0)

            # Feature 2: ticks en fase normalizado
            ticks_norm = min(1.0, inter._ticks_in_phase /
                             max(inter.red_timeout_ticks, 1))

            # Feature 3: número de entidades normalizado
            ents_norm = min(1.0, len(ents) / max(self.max_entities / 4, 1))

            # Feature 4: hay emergencia
            has_emerg = float(any(
                isinstance(e, Vehicle) and e.vehicle_type == VehicleType.EMERGENCY
                for e in ents
            ))

            # Feature 5: node_weight (ya está en [0.4, 1.0] aprox)
            nw = min(1.0, inter.node_weight)

            # Feature 6-7: hora cíclica
            # (sin_h, cos_h ya calculados)

            # Feature 8: presión promedio vecinos upstream
            nbr_p = neighbor_pressures.get(nid, 0.0)

            # Feature 9: wave_offset normalizado
            wave_norm = 0.0  # se actualiza en el próximo tick via algorithm

            base = i * N_FEATURES
            obs[base + 0] = p_norm
            obs[base + 1] = phase_enc
            obs[base + 2] = ticks_norm
            obs[base + 3] = ents_norm
            obs[base + 4] = has_emerg
            obs[base + 5] = nw
            obs[base + 6] = sin_h
            obs[base + 7] = cos_h
            obs[base + 8] = nbr_p
            obs[base + 9] = wave_norm

        return obs

    def _compute_neighbor_pressures(self) -> dict[str, float]:
        """Presión promedio de los vecinos upstream de cada nodo."""
        result = {}
        for nid in self.signaled:
            preds = list(self.graph.graph.predecessors(nid))
            if not preds:
                result[nid] = 0.0
                continue
            pressures = [
                self.graph.intersections[p].pressure
                for p in preds
                if p in self.graph.intersections
            ]
            if pressures:
                max_p = max(inter.pressure_threshold
                            for inter in self.graph.intersections.values())
                result[nid] = min(1.0, sum(pressures) / len(pressures) / max(max_p, 1))
            else:
                result[nid] = 0.0
        return result

    # ── Acción ────────────────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        """
        Aplica la acción del agente a los semáforos.

        action[i] = 0 → mantener fase actual del semáforo i
        action[i] = 1 → cambiar fase:
            RED    → GREEN
            GREEN  → YELLOW
            YELLOW → RED
            BLINK  → RED (hay tráfico)
        """
        for i, nid in enumerate(self.signaled):
            if int(action[i]) == 0:
                continue   # mantener

            inter = self.graph.intersections[nid]
            # Transición de fase
            if inter.current_phase == Phase.RED:
                inter.current_phase       = Phase.GREEN
                inter._ticks_in_phase     = 0
                inter._timeout_triggered  = False
                inter._green_started_tick = self._algo._tick if self._algo else 0
            elif inter.current_phase == Phase.GREEN:
                inter.current_phase   = Phase.YELLOW
                inter._ticks_in_phase = 0
            elif inter.current_phase in (Phase.YELLOW, Phase.BLINK):
                inter.current_phase   = Phase.RED
                inter._ticks_in_phase = 0
                inter._active_axis    = (
                    __import__('core.road', fromlist=['TrafficAxis'])
                    .TrafficAxis.EW
                    if inter._active_axis.value == 'ns'
                    else __import__('core.road', fromlist=['TrafficAxis'])
                    .TrafficAxis.NS
                )

    # ── Recompensa ────────────────────────────────────────────────────────────

    def _compute_reward(self, result, entities_by_node: dict,
                        stats) -> float:
        """
        Recompensa combinada:
          + w_flow    × flujo (entidades cruzando verde / total)
          - w_wait    × carga de espera acumulada
          - w_emerg   × emergencias detenidas
          + w_arrived × llegadas al destino
        """
        w_flow   = self.reward_cfg.get("w_flow",      1.0)
        w_wait   = self.reward_cfg.get("w_wait",      0.5)
        w_emerg  = self.reward_cfg.get("w_emergency", 5.0)
        w_arr    = self.reward_cfg.get("w_arrived",   2.0)

        total = max(result.total_entities, 1)

        # Flujo: proporción de entidades en nodos verdes
        entities_in_green = sum(
            len(entities_by_node.get(nid, []))
            for nid, inter in self.graph.intersections.items()
            if inter.current_phase == Phase.GREEN
        )
        flow_reward = w_flow * (entities_in_green / total)

        # Penalización por espera acumulada
        wait_penalty = w_wait * (
            sum(self._total_wait_ticks.values()) / max(total * self.n_ticks, 1)
        )

        # Penalización por emergencias detenidas
        emerg_stopped = sum(
            1 for nid, inter in self.graph.intersections.items()
            if inter.current_phase == Phase.RED
            for e in entities_by_node.get(nid, [])
            if isinstance(e, Vehicle) and e.vehicle_type == VehicleType.EMERGENCY
        )
        emerg_penalty = w_emerg * emerg_stopped

        # Bonificación por llegadas
        arrived_this_tick = len([
            me for me in self._movement._arrived
            if me.total_ticks == self._tick
        ])
        arrived_bonus = w_arr * (arrived_this_tick / max(total, 1))

        reward = flow_reward - wait_penalty - emerg_penalty + arrived_bonus
        return float(reward)

    def _update_wait_ticks(self, entities_by_node: dict,
                           phases: dict) -> None:
        """Incrementa el contador de espera para entidades en rojo."""
        for nid, ents in entities_by_node.items():
            if phases.get(nid) == "red":
                for e in ents:
                    self._total_wait_ticks[e.entity_id] = (
                        self._total_wait_ticks.get(e.entity_id, 0) + 1
                    )