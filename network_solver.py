"""
network_solver.py — Network-level ARZ solver for the 2×2 grid.

Reuses link-level physics from solver.py (pressure, equilibrium speed, LxF flux).
Adds junction coupling: at each time step, boundary conditions for each link
are computed from the junction signal phases and turning ratios.
"""

import numpy as np
from solver import (
    prim_to_cons, cons_to_prim, physical_flux, max_wavespeed,
    apply_relaxation, total_pce_density, equilibrium_speed,
    EPS_RHO, V_MIN,
)
from params import MOTO, CAR, CFL, V_FREE, RHO_MAX_PCE, PHI
from network_params import (
    LINKS, LINK_DX, JUNCTIONS, JUNCTION_IDS,
)


# ═══════════════════════════════════════════════════════════════════════════
# LINK-LEVEL LxF STEP (reuses solver.py physics, custom boundaries)
# ═══════════════════════════════════════════════════════════════════════════

def link_lxf_step(U, dt, dx, left_ghost, right_blocked=False, a_max_global=None):
    """
    One LxF step for a single link.

    Parameters
    ----------
    U             : (4, N) conservative state of the link
    dt            : time step
    dx            : cell size
    left_ghost    : (4,) state for the left ghost cell (from junction)
    right_blocked : if True, zero-flux at right boundary (downstream signal red)
    a_max_global  : precomputed global max wavespeed (avoids recomputation)
    """
    N = U.shape[1]
    U_ext = np.zeros((4, N + 2))
    U_ext[:, 1:N + 1] = U

    # Left boundary: from upstream junction
    U_ext[:, 0] = left_ghost

    # Right boundary: zero-gradient outflow
    U_ext[:, N + 1] = U[:, N - 1]

    # Fluxes
    F_ext = physical_flux(U_ext)
    a_max = a_max_global if a_max_global is not None else max_wavespeed(U_ext)

    U_L = U_ext[:, :-1]
    U_R = U_ext[:, 1:]
    F_L = F_ext[:, :-1]
    F_R = F_ext[:, 1:]

    F_num = 0.5 * (F_L + F_R) - 0.5 * a_max * (U_R - U_L)

    # Block right boundary if downstream signal is red for this link
    if right_blocked:
        F_num[:, N] = 0.0

    # Conservative update
    U_new = U - (dt / dx) * (F_num[:, 1:N + 1] - F_num[:, 0:N])
    U_new[0] = np.maximum(U_new[0], 0.0)
    U_new[2] = np.maximum(U_new[2], 0.0)
    return U_new


# ═══════════════════════════════════════════════════════════════════════════
# JUNCTION COUPLING
# ═══════════════════════════════════════════════════════════════════════════

def compute_junction_ghosts(link_states, junction_phases, ext_inflows):
    """
    Compute left ghost cells for each outgoing link at each junction,
    and determine which incoming links are blocked (right boundary).

    Parameters
    ----------
    link_states    : dict link_id → U (4, N)
    junction_phases: dict junction_id → int (0=EW green, 1=NS green)
    ext_inflows    : dict ext_id → (rho_moto, rho_car)

    Returns
    -------
    left_ghosts    : dict link_id → (4,) ghost cell state
    right_blocked  : dict link_id → bool (True if downstream junction blocks)
    """
    left_ghosts = {}
    right_blocked = {}

    # For each link, determine if its destination junction blocks it
    for lid, link in LINKS.items():
        dst_junc = link.to_node
        phase = junction_phases[dst_junc]
        # Phase 0 = EW green → EW links pass, NS blocked
        # Phase 1 = NS green → NS links pass, EW blocked
        if link.direction == 'EW':
            right_blocked[lid] = (phase == 1)  # blocked when NS green
        else:  # NS
            right_blocked[lid] = (phase == 0)  # blocked when EW green

    # For each link, compute the left ghost cell from its source junction
    for lid, link in LINKS.items():
        src_junc = link.from_node
        phase = junction_phases[src_junc]
        junc_def = JUNCTIONS[src_junc]

        # Collect allowed incoming flows at this junction
        rho_m_total = 0.0
        rho_c_total = 0.0

        for src_type, src_id, src_dir in junc_def['incoming']:
            # Check if this incoming movement is allowed by the signal
            allowed = (src_dir == 'EW' and phase == 0) or \
                      (src_dir == 'NS' and phase == 1)
            if not allowed:
                continue

            # Get turning ratio for this source → this link
            tr_key = (src_id, lid)
            tr = junc_def['turning_ratios'].get(tr_key, 0.0)
            if tr == 0.0:
                continue

            # Get source state
            if src_type == 'external':
                rm, rc = ext_inflows[src_id]
            else:  # 'link'
                src_U = link_states[src_id]
                rm_arr, _, rc_arr, _ = cons_to_prim(src_U[:, -3:])  # last 3 cells
                rm = float(np.mean(rm_arr))
                rc = float(np.mean(rc_arr))

            rho_m_total += tr * rm
            rho_c_total += tr * rc

        # Build ghost cell at equilibrium
        if rho_m_total < EPS_RHO and rho_c_total < EPS_RHO:
            # No inflow — use first cell of the link (zero-gradient)
            left_ghosts[lid] = link_states[lid][:, 0].copy()
        else:
            rm_arr = np.array([rho_m_total])
            rc_arr = np.array([rho_c_total])
            rho_tot = total_pce_density(rm_arr, rc_arr)
            vm = equilibrium_speed(rho_tot, MOTO)
            vc = equilibrium_speed(rho_tot, CAR)
            left_ghosts[lid] = prim_to_cons(rm_arr, vm, rc_arr, vc).flatten()

    return left_ghosts, right_blocked


# ═══════════════════════════════════════════════════════════════════════════
# NETWORK-LEVEL SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def init_network_state(scenario):
    """Initialize all links with uniform densities derived from scenario."""
    link_states = {}
    # Use average of Bd Steinmetz entry (main inflow) for initial condition
    main_ext = scenario.ext_inflows.get('ext_W_I1', (0.02, 0.015))
    rm_init, rc_init = main_ext[0] * 0.8, main_ext[1] * 0.8  # 80% of inflow density

    for lid, link in LINKS.items():
        N = link.n_cells
        rho_m = np.full(N, rm_init)
        rho_c = np.full(N, rc_init)
        rho_tot = total_pce_density(rho_m, rho_c)
        v_m = equilibrium_speed(rho_tot, MOTO)
        v_c = equilibrium_speed(rho_tot, CAR)
        link_states[lid] = prim_to_cons(rho_m, v_m, rho_c, v_c)

    return link_states


def network_max_wavespeed(link_states):
    """Global maximum wave speed across all links."""
    a_max = EPS_RHO
    for lid, U in link_states.items():
        a_max = max(a_max, max_wavespeed(U))
    return a_max


def simulate_network(link_states, duration, junction_phases, ext_inflows):
    """
    Advance the entire network by `duration` seconds.

    Parameters
    ----------
    link_states     : dict link_id → U (4, N_link)
    duration        : simulation time (s)
    junction_phases : dict junction_id → int (0 or 1)
    ext_inflows     : dict ext_id → (rho_moto, rho_car)

    Returns
    -------
    link_states : updated states
    """
    t = 0.0
    while t < duration:
        # Global CFL time step
        a_max = network_max_wavespeed(link_states)
        min_dx = min(LINK_DX.values())
        dt = CFL * min_dx / max(a_max, EPS_RHO)
        dt = min(dt, duration - t)

        # Junction coupling: compute boundary conditions
        left_ghosts, blocked = compute_junction_ghosts(
            link_states, junction_phases, ext_inflows)

        # Step each link independently (pass global a_max to avoid recomputation)
        new_states = {}
        for lid, link in LINKS.items():
            U = link_states[lid]
            dx = LINK_DX[lid]
            U = link_lxf_step(U, dt, dx, left_ghosts[lid], blocked[lid],
                              a_max_global=a_max)
            U = apply_relaxation(U, dt)
            new_states[lid] = U

        link_states = new_states
        t += dt

    return link_states


def get_network_pce_density(link_states):
    """Compute mean PCE density across the entire network."""
    total_pce = 0.0
    total_cells = 0
    for lid, U in link_states.items():
        rm, _, rc, _ = cons_to_prim(U)
        rho_pce = total_pce_density(rm, rc)
        total_pce += np.sum(rho_pce)
        total_cells += U.shape[1]
    return total_pce / total_cells
