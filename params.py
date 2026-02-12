"""
params.py — Single source of truth for the multi-class ARZ traffic model.

Two vehicle classes (motorcycles, cars) on a 1500 m signalized corridor.
All physical parameters are literature-based (see JITS article Table I).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# VEHICLE CLASS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
# Indices: 0 = motorcycles, 1 = cars
N_CLASSES = 2
MOTO, CAR = 0, 1

# Free-flow speeds (m/s)  — converted from km/h
V_FREE = np.array([50.0 / 3.6, 60.0 / 3.6])  # [13.89, 16.67] m/s

# Jam densities in PCE/m (article: 220 and 150 PCE/km)
RHO_MAX_PCE = np.array([0.220, 0.150])  # PCE/m

# PCE equivalence factors
PHI = np.array([0.25, 1.0])

# Physical jam densities (veh/m) = PCE jam / phi
RHO_MAX_PHYS = RHO_MAX_PCE / PHI  # [0.88, 0.15] veh/m

# Greenshields exponents (curvature of fundamental diagram)
BETA = np.array([2.0, 1.2])

# ═══════════════════════════════════════════════════════════════════════════
# PRESSURE / INTERACTION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
# Interaction matrix alpha[k, j]: impact of class j on class k's anticipation.
# Motos weave around cars (low alpha_01), cars are disrupted by motos (high alpha_10).
ALPHA = np.array([
    [1.0, 0.3],   # moto row: self=1.0, cars→motos=0.3 (motos barely notice cars)
    [1.2, 1.0],   # car  row: motos→cars=1.2 (cars disturbed by unpredictable motos)
])

# Pressure exponents
GAMMA = np.array([1.0, 1.0])

# Reference velocities for pressure (m/s) — set to free-flow
V_REF = V_FREE.copy()

# Relaxation times (s) — motorcycles adapt faster
TAU = np.array([3.0, 8.0])

# ═══════════════════════════════════════════════════════════════════════════
# CORRIDOR GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════
L_CORRIDOR = 1500.0      # corridor length (m)
N_CELLS = 100            # number of finite-volume cells
DX = L_CORRIDOR / N_CELLS  # 15.0 m

# Signal positions (cell indices) — at 375m, 750m, 1125m
N_SIGNALS = 3
SIGNAL_CELLS = np.array([25, 50, 75])

# ═══════════════════════════════════════════════════════════════════════════
# NUMERICAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
CFL = 0.5
EPS_RHO = 1e-10          # floor for density (avoid division by zero)
V_MIN = 0.0              # minimum velocity (m/s)

# ═══════════════════════════════════════════════════════════════════════════
# RL ENVIRONMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
DT_DECISION = 15.0       # RL decision interval (s)
N_STEPS_EPISODE = 30     # steps per episode → 450 s total
N_UPSTREAM_CELLS = 5     # cells to average for state observation

# Side-street queue parameters
SIDE_SERVICE_RATE = 0.9  # PCE/s drained when side is green
SIDE_MAX_QUEUE = 100.0   # PCE — maximum queue capacity

# Reward weights
REWARD_ALPHA = 1.0       # weight on mean PCE density
REWARD_KAPPA = 0.5       # weight on mean normalized side queue

# ═══════════════════════════════════════════════════════════════════════════
# DEMAND SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════
# Each scenario defines initial densities (veh/m) and inflow for both classes,
# plus side-street demand (PCE/s) and motorcycle fraction.
# Side demand is inversely proportional to main density:
#   - light main traffic → heavy side demand (agent must serve side streets)
#   - saturated main traffic → light side demand (agent should prioritize main)


@dataclass
class Scenario:
    """Traffic demand scenario for one episode."""
    name: str
    rho_moto_init: float    # initial moto density (veh/m)
    rho_car_init: float     # initial car density (veh/m)
    inflow_moto: float      # inflow moto density at left boundary (veh/m)
    inflow_car: float       # inflow car density at left boundary (veh/m)
    side_demand: float      # side-street demand rate (PCE/s) per intersection
    moto_fraction: float    # fraction of side injection that is motorcycles


SCENARIOS: Dict[str, Scenario] = {
    "light": Scenario(
        name="light",
        rho_moto_init=0.020,   # ~20 veh/km (physical)
        rho_car_init=0.015,    # ~15 veh/km
        inflow_moto=0.025,
        inflow_car=0.018,
        side_demand=0.40,      # heavy side demand
        moto_fraction=0.60,
    ),
    "moderate": Scenario(
        name="moderate",
        rho_moto_init=0.040,   # ~40 veh/km
        rho_car_init=0.030,    # ~30 veh/km
        inflow_moto=0.045,
        inflow_car=0.035,
        side_demand=0.30,
        moto_fraction=0.65,
    ),
    "heavy": Scenario(
        name="heavy",
        rho_moto_init=0.070,   # ~70 veh/km
        rho_car_init=0.045,    # ~45 veh/km
        inflow_moto=0.080,
        inflow_car=0.050,
        side_demand=0.20,
        moto_fraction=0.70,
    ),
    "saturated": Scenario(
        name="saturated",
        rho_moto_init=0.120,   # ~120 veh/km
        rho_car_init=0.060,    # ~60 veh/km
        inflow_moto=0.140,
        inflow_car=0.070,
        side_demand=0.15,
        moto_fraction=0.70,
    ),
}

SCENARIO_NAMES = list(SCENARIOS.keys())

# ═══════════════════════════════════════════════════════════════════════════
# DQN HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
DQN_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-3,
    "buffer_size": 50_000,
    "learning_starts": 500,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.15,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [64, 64]},
    "seed": 42,
}

# Fixed-timing baseline: 2:1 main:side split
# With 30 steps and 15s each → 90s cycle = 60s main + 30s side
BASELINE_MAIN_STEPS = 20   # out of 30 → 67% main green
BASELINE_SIDE_STEPS = 10   # out of 30 → 33% side green
BASELINE_CYCLE = BASELINE_MAIN_STEPS + BASELINE_SIDE_STEPS  # 30 steps = 1 cycle

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING BUDGETS
# ═══════════════════════════════════════════════════════════════════════════
PROOF_TIMESTEPS = 10_000
FULL_TIMESTEPS = 60_000

PROOF_EVAL_EPISODES = 10
FULL_EVAL_EPISODES = 50
