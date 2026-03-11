"""
network_params.py — Topology and parameters for the Quartier Ganhi network case study.

A 2×2 grid network from the Quartier Ganhi commercial district of Cotonou (Benin),
extracted from OpenStreetMap via osmnx.  Real intersections and road lengths.

    I1 (Adjovi×Éboué) —— L1 (Rue F.Éboué, 300m) —— I2 (Nikoué×Éboué) → exit_E
           |                                               |
    L3 (Av. Adjovi                                  L4 (Av. Nikoué
         279m)                                           282m)
           |                                               |
    I3 (Adjovi×Santos) — L2 (Rue Santos, 299m) — I4 (Nikoué×Santos) → exit_E
    ext_W → ↑                                          ↓ → exit_S
         ext_W                                       ext_S

Real OSM node IDs:
  I1 = 2782446101  (Av. Capitaine Adjovi × Rue Gén. Félix Éboué)
  I2 = 2782446111  (Av. Augustin Nikoué  × Rue Gén. Félix Éboué)
  I3 = 2782446087  (Av. Capitaine Adjovi × Rue José Firmin Santos)
  I4 = 2782446098  (Av. Augustin Nikoué  × Rue José Firmin Santos)

4 signalized intersections, 4 directional links, real road lengths.
Source: OpenStreetMap / osmnx, Quartier Ganhi, Cotonou, Benin.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Reuse vehicle-class physics from the original model
from params import (
    N_CLASSES, MOTO, CAR, CFL, EPS_RHO, V_MIN,
    V_FREE, RHO_MAX_PCE, RHO_MAX_PHYS, PHI, BETA,
    ALPHA, GAMMA, V_REF, TAU,
)


# ═══════════════════════════════════════════════════════════════════════════
# NETWORK TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Link:
    """A directional road segment between two junctions."""
    id: str
    from_node: str
    to_node: str
    length: float       # meters
    n_cells: int         # computed from length / ~15m
    name: str
    direction: str       # 'EW' or 'NS' — determines signal phase interaction


# Link definitions — REAL road lengths from OpenStreetMap
# Using ~30m cells for computational efficiency
LINKS = {
    'L1': Link('L1', 'I1', 'I2', 299.8, 10, 'Rue Gén. Félix Éboué', 'EW'),
    'L2': Link('L2', 'I3', 'I4', 298.8, 10, 'Rue José Firmin Santos', 'EW'),
    'L3': Link('L3', 'I1', 'I3', 279.3,  9, 'Av. Capitaine Adjovi', 'NS'),
    'L4': Link('L4', 'I2', 'I4', 281.5,  9, 'Av. Augustin Nikoué', 'NS'),
}

# dx per link
LINK_DX = {lid: link.length / link.n_cells for lid, link in LINKS.items()}

TOTAL_CELLS = sum(link.n_cells for link in LINKS.values())  # 89

N_JUNCTIONS = 4
JUNCTION_IDS = ['I1', 'I2', 'I3', 'I4']


# ═══════════════════════════════════════════════════════════════════════════
# JUNCTION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
# Each junction has:
#   - incoming: list of (source_type, source_id, direction)
#   - outgoing: list of (dest_type, dest_id)
#   - turning_ratios: dict (src_id, dst_id) → fraction
#   - Phase 0 = EW green, Phase 1 = NS green

JUNCTIONS = {
    'I1': {
        'incoming': [
            ('external', 'ext_W_I1', 'EW'),     # Main inflow from West
            ('external', 'ext_N_I1', 'NS'),     # Side inflow from North
        ],
        'outgoing': [
            ('link', 'L1'),   # East → Bd Steinmetz
            ('link', 'L3'),   # South → Av. Clozel
        ],
        'turning_ratios': {
            ('ext_W_I1', 'L1'): 0.70,   # 70% goes straight east
            ('ext_W_I1', 'L3'): 0.30,   # 30% turns south
            ('ext_N_I1', 'L1'): 0.40,   # 40% turns east
            ('ext_N_I1', 'L3'): 0.60,   # 60% goes straight south
        },
    },
    'I2': {
        'incoming': [
            ('link', 'L1', 'EW'),               # From I1 via Bd Steinmetz
            ('external', 'ext_N_I2', 'NS'),     # Side inflow from North
        ],
        'outgoing': [
            ('link', 'L4'),      # South → Rue des Cheminots
            ('exit', 'exit_E'),  # Exit East (leaves network)
        ],
        'turning_ratios': {
            ('L1', 'L4'): 0.55,
            ('L1', 'exit_E'): 0.45,
            ('ext_N_I2', 'L4'): 0.30,
            ('ext_N_I2', 'exit_E'): 0.70,
        },
    },
    'I3': {
        'incoming': [
            ('link', 'L3', 'NS'),               # From I1 via Av. Clozel
            ('external', 'ext_W_I3', 'EW'),     # Side inflow from West
        ],
        'outgoing': [
            ('link', 'L2'),       # East → Rue du Marché
            ('exit', 'exit_S'),   # Exit South (leaves network)
        ],
        'turning_ratios': {
            ('L3', 'L2'): 0.65,
            ('L3', 'exit_S'): 0.35,
            ('ext_W_I3', 'L2'): 0.75,
            ('ext_W_I3', 'exit_S'): 0.25,
        },
    },
    'I4': {
        'incoming': [
            ('link', 'L2', 'EW'),   # From I3 via Rue du Marché
            ('link', 'L4', 'NS'),   # From I2 via Rue des Cheminots
        ],
        'outgoing': [
            ('exit', 'exit_E4'),    # Exit East
            ('exit', 'exit_S4'),    # Exit South
        ],
        'turning_ratios': {
            ('L2', 'exit_E4'): 0.60,
            ('L2', 'exit_S4'): 0.40,
            ('L4', 'exit_E4'): 0.45,
            ('L4', 'exit_S4'): 0.55,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DEMAND SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkScenario:
    """Traffic demand for the network with external inflow rates."""
    name: str
    # External inflow densities (veh/m) — used as ghost-cell densities
    ext_inflows: Dict[str, Tuple[float, float]]  # ext_id → (rho_moto, rho_car)
    moto_fraction: float  # for side-street queue injection


NETWORK_SCENARIOS: Dict[str, NetworkScenario] = {
    'light': NetworkScenario(
        name='light',
        ext_inflows={
            'ext_W_I1': (0.022, 0.016),   # Main boulevard entry
            'ext_N_I1': (0.012, 0.008),   # Side street
            'ext_N_I2': (0.010, 0.007),   # Side street
            'ext_W_I3': (0.015, 0.012),   # Secondary road entry
        },
        moto_fraction=0.60,
    ),
    'moderate': NetworkScenario(
        name='moderate',
        ext_inflows={
            'ext_W_I1': (0.042, 0.032),
            'ext_N_I1': (0.025, 0.018),
            'ext_N_I2': (0.020, 0.015),
            'ext_W_I3': (0.035, 0.025),
        },
        moto_fraction=0.65,
    ),
    'heavy': NetworkScenario(
        name='heavy',
        ext_inflows={
            'ext_W_I1': (0.072, 0.048),
            'ext_N_I1': (0.040, 0.025),
            'ext_N_I2': (0.035, 0.022),
            'ext_W_I3': (0.060, 0.038),
        },
        moto_fraction=0.70,
    ),
    'saturated': NetworkScenario(
        name='saturated',
        ext_inflows={
            'ext_W_I1': (0.120, 0.065),
            'ext_N_I1': (0.060, 0.035),
            'ext_N_I2': (0.050, 0.030),
            'ext_W_I3': (0.095, 0.055),
        },
        moto_fraction=0.70,
    ),
}

NETWORK_SCENARIO_NAMES = list(NETWORK_SCENARIOS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# RL ENVIRONMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

NET_DT_DECISION = 15.0        # seconds per RL step
NET_N_STEPS_EPISODE = 30      # steps → 450s total
NET_N_UPSTREAM_CELLS = 3      # cells to observe per approach

# Side-street queues at each junction (for blocked external inflows)
NET_SIDE_MAX_QUEUE = 80.0     # PCE
NET_SIDE_SERVICE_RATE = 0.8   # PCE/s drained when green

# Reward weights
NET_REWARD_ALPHA = 1.0
NET_REWARD_KAPPA = 0.4

# Observation: 6 features × 4 junctions = 24
NET_OBS_DIM = 6 * N_JUNCTIONS  # 24
NET_N_ACTIONS = 2 ** N_JUNCTIONS  # 16

# DQN hyperparameters (same as corridor but larger buffer for network)
NET_DQN_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-3,
    "buffer_size": 80_000,
    "learning_starts": 800,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.15,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [128, 64]},  # slightly larger for network
    "seed": 42,
}

# Training budgets
NET_PROOF_TIMESTEPS = 10_000
NET_FULL_TIMESTEPS = 60_000
NET_PROOF_EVAL_EPISODES = 10
NET_FULL_EVAL_EPISODES = 50

# Fixed-timing baseline
NET_BASELINE_MAIN_STEPS = 20
NET_BASELINE_SIDE_STEPS = 10
