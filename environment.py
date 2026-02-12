"""
environment.py — Gymnasium environment wrapping the multi-class ARZ solver.

3-intersection signalized corridor with side-street queue dynamics.
Action space: Discrete(8) — 3 binary signal phases encoded as one integer.
Observation: Box(0, 1, shape=(18,)) — per intersection: densities, speeds, queue, phase.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from params import (
    N_CELLS, N_SIGNALS, SIGNAL_CELLS, DX,
    N_CLASSES, MOTO, CAR, V_FREE, RHO_MAX_PCE, RHO_MAX_PHYS, PHI,
    DT_DECISION, N_STEPS_EPISODE, N_UPSTREAM_CELLS,
    SIDE_SERVICE_RATE, SIDE_MAX_QUEUE,
    REWARD_ALPHA, REWARD_KAPPA,
    SCENARIOS, SCENARIO_NAMES, EPS_RHO,
)
from solver import (
    cons_to_prim, total_pce_density, init_uniform,
    make_inflow_state, simulate,
)


class TrafficCorridorEnv(gym.Env):
    """
    Multi-class ARZ traffic corridor with RL signal control.

    Action encoding (Discrete(8)):
        action = b2*4 + b1*2 + b0
        where b_i ∈ {0, 1} is the phase for intersection i.
        0 = main-green (main road flows, side blocked)
        1 = side-green (main road blocked, side injects)

    Observation (Box, 18-dim):
        Per intersection (6 values × 3 intersections):
          [0] moto density upstream (normalized)
          [1] car density upstream (normalized)
          [2] moto velocity upstream (normalized)
          [3] car velocity upstream (normalized)
          [4] side-street queue (normalized)
          [5] current signal phase (0 or 1)

    Reward:
        R = -(α · mean_pce_density + κ · mean_normalized_queue)
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_name: str = None, render_mode=None):
        """
        Parameters
        ----------
        scenario_name : If None, randomly samples a scenario each reset.
                        If given, uses that specific scenario.
        """
        super().__init__()
        self.scenario_name = scenario_name
        self.render_mode = render_mode

        # Spaces
        self.action_space = spaces.Discrete(2 ** N_SIGNALS)  # 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(N_SIGNALS * 6,),  # 18
            dtype=np.float32,
        )

        # Internal state
        self.U = None
        self.side_queues = None
        self.signal_phases = None
        self.current_step = 0
        self.scenario = None
        self.inflow_state = None

    def _decode_action(self, action: int) -> np.ndarray:
        """Decode integer action → array of N_SIGNALS binary phases."""
        phases = np.zeros(N_SIGNALS, dtype=np.int32)
        for i in range(N_SIGNALS):
            phases[i] = (action >> i) & 1
        return phases

    def _get_upstream_obs(self, intersection_idx: int):
        """
        Get mean density and velocity in N_UPSTREAM_CELLS upstream
        of the given intersection.
        """
        cell_idx = SIGNAL_CELLS[intersection_idx]
        start = max(0, cell_idx - N_UPSTREAM_CELLS)
        end = cell_idx

        if start >= end:
            start = max(0, cell_idx - 1)
            end = cell_idx + 1

        rho_m, v_m, rho_c, v_c = cons_to_prim(self.U[:, start:end])

        return (
            np.mean(rho_m),
            np.mean(rho_c),
            np.mean(v_m),
            np.mean(v_c),
        )

    def _get_obs(self) -> np.ndarray:
        """Build the 18-dim normalized observation vector."""
        obs = np.zeros(N_SIGNALS * 6, dtype=np.float32)

        for i in range(N_SIGNALS):
            mean_rho_m, mean_rho_c, mean_v_m, mean_v_c = self._get_upstream_obs(i)

            offset = i * 6
            obs[offset + 0] = np.clip(mean_rho_m / RHO_MAX_PHYS[MOTO], 0, 1)
            obs[offset + 1] = np.clip(mean_rho_c / RHO_MAX_PHYS[CAR], 0, 1)
            obs[offset + 2] = np.clip(mean_v_m / V_FREE[MOTO], 0, 1)
            obs[offset + 3] = np.clip(mean_v_c / V_FREE[CAR], 0, 1)
            obs[offset + 4] = np.clip(self.side_queues[i] / SIDE_MAX_QUEUE, 0, 1)
            obs[offset + 5] = float(self.signal_phases[i])

        return obs

    def _compute_reward(self) -> float:
        """
        Reward = -(α · mean_pce_density_normalized + κ · mean_queue_normalized)

        Both terms are in [0, ~1], so reward is in [~-1.5, 0].
        """
        rho_m, _, rho_c, _ = cons_to_prim(self.U)
        rho_pce = total_pce_density(rho_m, rho_c)

        # Normalize by max possible PCE density (use car jam as reference)
        mean_pce_norm = np.mean(rho_pce) / RHO_MAX_PCE[CAR]

        # Mean normalized side queue
        mean_queue_norm = np.mean(self.side_queues / SIDE_MAX_QUEUE)

        reward = -(REWARD_ALPHA * mean_pce_norm + REWARD_KAPPA * mean_queue_norm)
        return float(reward)

    def _compute_side_injection_rates(self) -> dict:
        """
        Compute physical injection rates (veh/m/s) for side-street traffic
        entering at each signal cell when side is green.

        The side_demand (PCE/s) is split into moto and car fractions,
        converted to physical density rate by dividing by dx.
        """
        mf = self.scenario.moto_fraction
        sd = self.scenario.side_demand  # PCE/s

        # Split demand: moto PCE/s and car PCE/s
        moto_pce_rate = sd * mf              # PCE/s of motos
        car_pce_rate = sd * (1.0 - mf)       # PCE/s of cars

        # Convert to physical density injection rate (veh/m/s)
        # PCE/s → veh/s: divide by φ_k
        # veh/s → veh/m/s: divide by dx (injected into one cell of width dx)
        moto_rate = (moto_pce_rate / PHI[MOTO]) / DX
        car_rate = (car_pce_rate / PHI[CAR]) / DX

        return {
            'rho_moto_rate': np.full(N_SIGNALS, moto_rate),
            'rho_car_rate': np.full(N_SIGNALS, car_rate),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick scenario
        if self.scenario_name is not None:
            self.scenario = SCENARIOS[self.scenario_name]
        else:
            name = self.np_random.choice(SCENARIO_NAMES)
            self.scenario = SCENARIOS[name]

        # Initialize state
        self.U = init_uniform(self.scenario)
        self.side_queues = np.zeros(N_SIGNALS, dtype=np.float64)
        self.signal_phases = np.zeros(N_SIGNALS, dtype=np.int32)  # start main-green
        self.current_step = 0
        self.inflow_state = make_inflow_state(self.scenario)

        return self._get_obs(), {"scenario": self.scenario.name}

    def step(self, action: int):
        """
        Execute one RL decision step (15 seconds of simulation).

        1. Decode action → signal phases
        2. Update side queues based on phases
        3. Advance ARZ solver
        4. Compute observation and reward
        """
        self.signal_phases = self._decode_action(action)
        self.current_step += 1

        # ── Side-street queue dynamics ──────────────────────────────────
        sd = self.scenario.side_demand  # PCE/s
        for i in range(N_SIGNALS):
            if self.signal_phases[i] == 0:
                # Main-green: side queue accumulates
                self.side_queues[i] += sd * DT_DECISION
            else:
                # Side-green: side queue drains
                drain = SIDE_SERVICE_RATE * DT_DECISION
                self.side_queues[i] = max(0.0, self.side_queues[i] - drain)

            # Clamp
            self.side_queues[i] = min(self.side_queues[i], SIDE_MAX_QUEUE)

        # ── Side-street injection into main road ────────────────────────
        side_inj = self._compute_side_injection_rates()

        # ── Advance ARZ solver ──────────────────────────────────────────
        self.U = simulate(
            self.U,
            duration=DT_DECISION,
            signal_phases=self.signal_phases,
            signal_cells=SIGNAL_CELLS,
            inflow_state=self.inflow_state,
            side_injections=side_inj,
        )

        # ── Observation, reward, termination ────────────────────────────
        obs = self._get_obs()
        reward = self._compute_reward()
        truncated = self.current_step >= N_STEPS_EPISODE
        terminated = False

        info = {
            "scenario": self.scenario.name,
            "step": self.current_step,
            "side_queues": self.side_queues.copy(),
            "signal_phases": self.signal_phases.copy(),
        }

        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════
# FIXED-TIMING BASELINE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class FixedTimingController:
    """
    Fixed-timing controller with configurable main:side split.

    Default: 20 steps main / 10 steps side per 30-step cycle (67/33 split).
    All intersections switch simultaneously (no offset).
    """

    def __init__(self, main_steps: int = 20, side_steps: int = 10):
        self.main_steps = main_steps
        self.side_steps = side_steps
        self.cycle = main_steps + side_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0

    def predict(self, obs=None, deterministic=True):
        """Match SB3 predict interface: returns (action, None)."""
        pos_in_cycle = self.step_count % self.cycle
        if pos_in_cycle < self.main_steps:
            action = 0  # all main-green (000 in binary)
        else:
            action = 7  # all side-green (111 in binary)
        self.step_count += 1
        return action, None
