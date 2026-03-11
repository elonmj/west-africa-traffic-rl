"""
network_env.py — Gymnasium environment for the 2×2 grid network.

Action space:  Discrete(16) — 4 binary signal phases (one per junction).
Observation:   Box(0, 1, shape=(24,)) — per junction: densities, queue, phase.
Reward:        -(α · mean_pce_density + κ · mean_queue)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from params import MOTO, CAR, V_FREE, RHO_MAX_PCE, RHO_MAX_PHYS, PHI, EPS_RHO
from solver import cons_to_prim, total_pce_density
from network_params import (
    LINKS, LINK_DX, JUNCTIONS, JUNCTION_IDS, N_JUNCTIONS,
    NETWORK_SCENARIOS, NETWORK_SCENARIO_NAMES,
    NET_DT_DECISION, NET_N_STEPS_EPISODE, NET_N_UPSTREAM_CELLS,
    NET_SIDE_MAX_QUEUE, NET_SIDE_SERVICE_RATE,
    NET_REWARD_ALPHA, NET_REWARD_KAPPA,
    NET_OBS_DIM, NET_N_ACTIONS,
    NET_BASELINE_MAIN_STEPS, NET_BASELINE_SIDE_STEPS,
)
from network_solver import (
    init_network_state, simulate_network, get_network_pce_density,
)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE: which links approach each junction, and from which direction
# ═══════════════════════════════════════════════════════════════════════════

_JUNCTION_INCOMING_LINKS = {}   # junc_id → list of (link_id, direction)
_JUNCTION_EXT_SOURCES = {}      # junc_id → list of (ext_id, direction, demand_rate_PCE_per_s)

for jid, jdef in JUNCTIONS.items():
    _JUNCTION_INCOMING_LINKS[jid] = []
    _JUNCTION_EXT_SOURCES[jid] = []
    for src_type, src_id, src_dir in jdef['incoming']:
        if src_type == 'link':
            _JUNCTION_INCOMING_LINKS[jid].append((src_id, src_dir))
        else:
            _JUNCTION_EXT_SOURCES[jid].append((src_id, src_dir))


class TrafficNetworkEnv(gym.Env):
    """
    Multi-class ARZ traffic network with RL signal control.

    Action encoding (Discrete(16)):
        action = b3*8 + b2*4 + b1*2 + b0
        where b_i ∈ {0,1} is the phase for junction i (I1..I4).
        0 = EW-green (E-W traffic flows, N-S blocked)
        1 = NS-green (N-S traffic flows, E-W blocked)

    Observation (Box, 24-dim):
        Per junction (6 values × 4 junctions):
          [0] moto density approaching from EW (normalized)
          [1] car density approaching from EW (normalized)
          [2] moto density approaching from NS (normalized)
          [3] car density approaching from NS (normalized)
          [4] side-street queue (normalized)
          [5] current signal phase (0 or 1)

    Reward:
        R = -(α · mean_pce_density + κ · mean_queue_normalized)
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_name=None, render_mode=None):
        super().__init__()
        self.scenario_name = scenario_name
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(NET_N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NET_OBS_DIM,), dtype=np.float32,
        )

        self.link_states = None
        self.junction_phases = None
        self.side_queues = None
        self.current_step = 0
        self.scenario = None

    # ─── Action decode ──────────────────────────────────────────────────

    def _decode_action(self, action):
        phases = {}
        for i, jid in enumerate(JUNCTION_IDS):
            phases[jid] = (action >> i) & 1
        return phases

    # ─── Observation ────────────────────────────────────────────────────

    def _get_approach_density(self, junc_id, direction):
        """Mean moto/car density approaching junction from given direction."""
        rm_sum, rc_sum, count = 0.0, 0.0, 0

        # Link-based approach
        for lid, ldir in _JUNCTION_INCOMING_LINKS[junc_id]:
            if ldir != direction:
                continue
            U = self.link_states[lid]
            n = min(NET_N_UPSTREAM_CELLS, U.shape[1])
            rm, _, rc, _ = cons_to_prim(U[:, -n:])
            rm_sum += np.mean(rm)
            rc_sum += np.mean(rc)
            count += 1

        # External approaches
        for ext_id, ext_dir in _JUNCTION_EXT_SOURCES[junc_id]:
            if ext_dir != direction:
                continue
            rm_ext, rc_ext = self.scenario.ext_inflows[ext_id]
            rm_sum += rm_ext
            rc_sum += rc_ext
            count += 1

        if count == 0:
            return 0.0, 0.0
        return rm_sum / count, rc_sum / count

    def _get_obs(self):
        obs = np.zeros(NET_OBS_DIM, dtype=np.float32)
        for i, jid in enumerate(JUNCTION_IDS):
            offset = i * 6

            rm_ew, rc_ew = self._get_approach_density(jid, 'EW')
            rm_ns, rc_ns = self._get_approach_density(jid, 'NS')

            obs[offset + 0] = np.clip(rm_ew / RHO_MAX_PHYS[MOTO], 0, 1)
            obs[offset + 1] = np.clip(rc_ew / RHO_MAX_PHYS[CAR], 0, 1)
            obs[offset + 2] = np.clip(rm_ns / RHO_MAX_PHYS[MOTO], 0, 1)
            obs[offset + 3] = np.clip(rc_ns / RHO_MAX_PHYS[CAR], 0, 1)
            obs[offset + 4] = np.clip(
                self.side_queues[jid] / NET_SIDE_MAX_QUEUE, 0, 1)
            obs[offset + 5] = float(self.junction_phases[jid])

        return obs

    # ─── Reward ─────────────────────────────────────────────────────────

    def _compute_reward(self):
        mean_pce = get_network_pce_density(self.link_states)
        mean_pce_norm = mean_pce / RHO_MAX_PCE[CAR]

        queue_vals = np.array([self.side_queues[jid] for jid in JUNCTION_IDS])
        mean_q_norm = np.mean(queue_vals / NET_SIDE_MAX_QUEUE)

        return float(-(NET_REWARD_ALPHA * mean_pce_norm +
                        NET_REWARD_KAPPA * mean_q_norm))

    # ─── Side-street queue dynamics ─────────────────────────────────────

    def _compute_side_demand(self, junc_id):
        """PCE/s demand from blocked external sources at this junction."""
        phase = self.junction_phases[junc_id]
        demand = 0.0
        for ext_id, ext_dir in _JUNCTION_EXT_SOURCES[junc_id]:
            is_blocked = (ext_dir == 'EW' and phase == 1) or \
                         (ext_dir == 'NS' and phase == 0)
            if is_blocked:
                rm, rc = self.scenario.ext_inflows[ext_id]
                pce = PHI[MOTO] * rm + PHI[CAR] * rc
                # Convert density (veh/m) to flow estimate (PCE/s)
                # Rough: demand ≈ pce_density × free_flow_speed_mix
                v_mix = 0.5 * (V_FREE[MOTO] + V_FREE[CAR])
                demand += pce * v_mix
        return demand

    def _update_side_queues(self):
        for jid in JUNCTION_IDS:
            blocked_demand = self._compute_side_demand(jid)
            phase = self.junction_phases[jid]

            # Accumulate if blocked
            self.side_queues[jid] += blocked_demand * NET_DT_DECISION

            # Drain if the blocked direction now has green (check each source)
            for ext_id, ext_dir in _JUNCTION_EXT_SOURCES[jid]:
                is_green = (ext_dir == 'EW' and phase == 0) or \
                           (ext_dir == 'NS' and phase == 1)
                if is_green:
                    drain = NET_SIDE_SERVICE_RATE * NET_DT_DECISION
                    self.side_queues[jid] = max(0.0,
                                                self.side_queues[jid] - drain)

            self.side_queues[jid] = min(self.side_queues[jid], NET_SIDE_MAX_QUEUE)

    # ─── Reset / Step ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.scenario_name is not None:
            self.scenario = NETWORK_SCENARIOS[self.scenario_name]
        else:
            name = self.np_random.choice(NETWORK_SCENARIO_NAMES)
            self.scenario = NETWORK_SCENARIOS[name]

        self.link_states = init_network_state(self.scenario)
        self.junction_phases = {jid: 0 for jid in JUNCTION_IDS}
        self.side_queues = {jid: 0.0 for jid in JUNCTION_IDS}
        self.current_step = 0

        return self._get_obs(), {"scenario": self.scenario.name}

    def step(self, action):
        self.junction_phases = self._decode_action(action)
        self.current_step += 1

        # Update side-street queues
        self._update_side_queues()

        # Advance network solver
        self.link_states = simulate_network(
            self.link_states,
            duration=NET_DT_DECISION,
            junction_phases=self.junction_phases,
            ext_inflows=self.scenario.ext_inflows,
        )

        obs = self._get_obs()
        reward = self._compute_reward()
        truncated = self.current_step >= NET_N_STEPS_EPISODE
        terminated = False

        info = {
            "scenario": self.scenario.name,
            "step": self.current_step,
            "side_queues": dict(self.side_queues),
            "phases": dict(self.junction_phases),
        }
        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════
# FIXED-TIMING BASELINE
# ═══════════════════════════════════════════════════════════════════════════

class NetworkFixedTimingController:
    """Fixed-timing with 2:1 E-W / N-S split, no coordination."""

    def __init__(self, main_steps=NET_BASELINE_MAIN_STEPS,
                 side_steps=NET_BASELINE_SIDE_STEPS):
        self.main_steps = main_steps
        self.side_steps = side_steps
        self.cycle = main_steps + side_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0

    def predict(self, obs=None, deterministic=True):
        pos = self.step_count % self.cycle
        if pos < self.main_steps:
            action = 0   # all EW-green (0000)
        else:
            action = 15  # all NS-green (1111)
        self.step_count += 1
        return action, None
