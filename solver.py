"""
solver.py — Multi-class ARZ finite-volume solver (Local Lax-Friedrichs).

State vector per cell:  U = [ρ_m, ρ_m·w_m, ρ_c, ρ_c·w_c]
  where ρ_k = physical density (veh/m), w_k = v_k + p_k (generalized momentum).

Spatial: 1D finite-volume, ghost cells at boundaries.
Flux:    Local Lax-Friedrichs (simple, stable, first-order).
Time:    Forward Euler with CFL-adaptive Δt.
Source:  Relaxation toward equilibrium speed (operator splitting).
Signals: Zero-flux wall at red signal interfaces.
"""

import numpy as np
from params import (
    N_CLASSES, MOTO, CAR, N_CELLS, DX, CFL, EPS_RHO, V_MIN,
    V_FREE, RHO_MAX_PCE, RHO_MAX_PHYS, PHI, BETA,
    ALPHA, GAMMA, V_REF, TAU,
)


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS: Pressure, equilibrium speed, primitive variable recovery
# ═══════════════════════════════════════════════════════════════════════════

def total_pce_density(rho_m: np.ndarray, rho_c: np.ndarray) -> np.ndarray:
    """PCE-weighted total density: ρ_tot = φ_m·ρ_m + φ_c·ρ_c."""
    return PHI[MOTO] * rho_m + PHI[CAR] * rho_c


def equilibrium_speed(rho_tot_pce: np.ndarray, k: int) -> np.ndarray:
    """
    Generalized Greenshields:
        v_k^e(ρ_tot) = v_k^free × max(0, 1 - ρ_tot / ρ_k^max)^β_k

    Parameters
    ----------
    rho_tot_pce : PCE-weighted total density (PCE/m)
    k           : vehicle class index (MOTO=0, CAR=1)
    """
    ratio = np.clip(1.0 - rho_tot_pce / RHO_MAX_PCE[k], 0.0, 1.0)
    return V_FREE[k] * ratio ** BETA[k]


def pressure(rho_m: np.ndarray, rho_c: np.ndarray, k: int) -> np.ndarray:
    """
    Multi-class pressure:
        p_k = v_k^ref × (Σ_j α_{kj}·φ_j·ρ_j / ρ_k^max_pce)^γ_k

    Uses PCE-weighted densities in the numerator for dimensional consistency.
    """
    weighted_rho = ALPHA[k, MOTO] * PHI[MOTO] * rho_m + ALPHA[k, CAR] * PHI[CAR] * rho_c
    ratio = np.clip(weighted_rho / RHO_MAX_PCE[k], 0.0, 5.0)  # soft cap
    return V_REF[k] * ratio ** GAMMA[k]


def pressure_derivative(rho_m: np.ndarray, rho_c: np.ndarray, k: int) -> np.ndarray:
    """
    ∂p_k/∂ρ_k (partial w.r.t. own physical density).
    For γ_k = 1:  ∂p_k/∂ρ_k = v_k^ref × α_{kk}·φ_k / ρ_k^max_pce
    """
    weighted_rho = ALPHA[k, MOTO] * PHI[MOTO] * rho_m + ALPHA[k, CAR] * PHI[CAR] * rho_c
    ratio = np.clip(weighted_rho / RHO_MAX_PCE[k], 0.0, 5.0)

    if GAMMA[k] == 1.0:
        return V_REF[k] * ALPHA[k, k] * PHI[k] / RHO_MAX_PCE[k] * np.ones_like(rho_m)
    else:
        safe_ratio = np.maximum(ratio, EPS_RHO)
        return (V_REF[k] * GAMMA[k] * ALPHA[k, k] * PHI[k] / RHO_MAX_PCE[k]
                * safe_ratio ** (GAMMA[k] - 1.0))


def cons_to_prim(U: np.ndarray):
    """
    Convert conservative → primitive variables.

    U : shape (4, N)
        U[0] = ρ_m, U[1] = ρ_m·w_m, U[2] = ρ_c, U[3] = ρ_c·w_c

    Returns
    -------
    rho_m, v_m, rho_c, v_c : each shape (N,)
    """
    rho_m = np.maximum(U[0], EPS_RHO)
    rho_c = np.maximum(U[2], EPS_RHO)

    # w_k = m_k / ρ_k, then v_k = w_k - p_k
    w_m = U[1] / rho_m
    w_c = U[3] / rho_c

    p_m = pressure(rho_m, rho_c, MOTO)
    p_c = pressure(rho_m, rho_c, CAR)

    v_m = np.clip(w_m - p_m, V_MIN, V_FREE[MOTO])
    v_c = np.clip(w_c - p_c, V_MIN, V_FREE[CAR])

    return rho_m, v_m, rho_c, v_c


def prim_to_cons(rho_m, v_m, rho_c, v_c) -> np.ndarray:
    """Convert primitive → conservative variables."""
    p_m = pressure(rho_m, rho_c, MOTO)
    p_c = pressure(rho_m, rho_c, CAR)

    U = np.zeros((4, len(rho_m)))
    U[0] = rho_m
    U[1] = rho_m * (v_m + p_m)  # ρ_m · w_m
    U[2] = rho_c
    U[3] = rho_c * (v_c + p_c)  # ρ_c · w_c
    return U


# ═══════════════════════════════════════════════════════════════════════════
# FLUX AND WAVE SPEEDS
# ═══════════════════════════════════════════════════════════════════════════

def physical_flux(U: np.ndarray) -> np.ndarray:
    """
    Compute the physical flux F(U) for the 4-component system.

    F = [ρ_m·v_m, ρ_m·w_m·v_m, ρ_c·v_c, ρ_c·w_c·v_c]
    """
    rho_m, v_m, rho_c, v_c = cons_to_prim(U)
    F = np.zeros_like(U)
    F[0] = rho_m * v_m
    F[1] = U[1] * v_m    # (ρ_m·w_m) · v_m
    F[2] = rho_c * v_c
    F[3] = U[3] * v_c    # (ρ_c·w_c) · v_c
    return F


def max_wavespeed(U: np.ndarray) -> float:
    """
    Maximum absolute wave speed across all cells and classes.

    Characteristic speeds: λ_k^(1) = v_k,  λ_k^(2) = v_k - ρ_k · ∂p_k/∂ρ_k
    """
    rho_m, v_m, rho_c, v_c = cons_to_prim(U)

    dp_m = pressure_derivative(rho_m, rho_c, MOTO)
    dp_c = pressure_derivative(rho_m, rho_c, CAR)

    # All characteristic speeds
    speeds = np.array([
        np.abs(v_m),
        np.abs(v_m - rho_m * dp_m),
        np.abs(v_c),
        np.abs(v_c - rho_c * dp_c),
    ])
    return float(np.max(speeds)) + EPS_RHO


# ═══════════════════════════════════════════════════════════════════════════
# LAX-FRIEDRICHS FLUX AND TIME STEPPING
# ═══════════════════════════════════════════════════════════════════════════

def _apply_boundary_conditions(U: np.ndarray, inflow_state: np.ndarray) -> np.ndarray:
    """
    Create extended array with ghost cells at both ends.

    U : shape (4, N)
    Returns: U_ext shape (4, N+2) with ghost cells at [0] and [N+1].
    """
    N = U.shape[1]
    U_ext = np.zeros((4, N + 2))
    U_ext[:, 1:N + 1] = U

    # Left: Dirichlet inflow
    U_ext[:, 0] = inflow_state

    # Right: zero-gradient outflow
    U_ext[:, N + 1] = U[:, N - 1]

    return U_ext


def lax_friedrichs_step(U: np.ndarray, dt: float, dx: float,
                        signal_mask: np.ndarray = None,
                        inflow_state: np.ndarray = None) -> np.ndarray:
    """
    One forward-Euler time step with Local Lax-Friedrichs flux.

    Parameters
    ----------
    U            : (4, N) conservative state
    dt           : time step (s)
    dx           : cell size (m)
    signal_mask  : (N+1,) array of 0/1 — flux multiplier at each interface.
                   0 = blocked (red signal), 1 = open.
                   Interface j sits between cell j-1 and cell j.
    inflow_state : (4,) state for the left ghost cell.

    Returns
    -------
    U_new : (4, N) updated conservative state
    """
    N = U.shape[1]

    if inflow_state is None:
        inflow_state = U[:, 0]

    # Extended domain: indices 0..N+1, original cells at 1..N
    U_ext = _apply_boundary_conditions(U, inflow_state)

    # Fluxes and wave speeds on extended domain
    F_ext = physical_flux(U_ext)
    a_max = max_wavespeed(U_ext)

    # Numerical flux at N+1 interfaces (between ext cells i and i+1)
    # Interface j (for j=0..N) sits between ext cells j and j+1.
    U_L = U_ext[:, :-1]   # (4, N+1) — left states
    U_R = U_ext[:, 1:]    # (4, N+1) — right states
    F_L = F_ext[:, :-1]
    F_R = F_ext[:, 1:]

    F_num = 0.5 * (F_L + F_R) - 0.5 * a_max * (U_R - U_L)

    # Apply signal mask (block flux at red interfaces)
    if signal_mask is not None:
        F_num *= signal_mask[np.newaxis, :]

    # Conservative update for original cells (indices 0..N-1 in U,
    # which correspond to ext indices 1..N, so interfaces 0..N)
    U_new = U - (dt / dx) * (F_num[:, 1:N + 1] - F_num[:, 0:N])

    # Clip densities to non-negative
    U_new[0] = np.maximum(U_new[0], 0.0)
    U_new[2] = np.maximum(U_new[2], 0.0)

    return U_new


def apply_relaxation(U: np.ndarray, dt: float) -> np.ndarray:
    """
    Relaxation source term (operator splitting):
        ∂(ρ_k w_k)/∂t = (ρ_k / τ_k) · (v_k^e - v_k)

    Drives velocity toward equilibrium.
    """
    rho_m, v_m, rho_c, v_c = cons_to_prim(U)
    rho_tot = total_pce_density(rho_m, rho_c)

    ve_m = equilibrium_speed(rho_tot, MOTO)
    ve_c = equilibrium_speed(rho_tot, CAR)

    # Source terms (implicit-ish: limit the relaxation to avoid overshoot)
    dv_m = np.clip((ve_m - v_m), -V_FREE[MOTO], V_FREE[MOTO])
    dv_c = np.clip((ve_c - v_c), -V_FREE[CAR], V_FREE[CAR])

    # Apply with relaxation time scale
    factor_m = np.minimum(dt / TAU[MOTO], 1.0)  # cap at 1 to avoid overshoot
    factor_c = np.minimum(dt / TAU[CAR], 1.0)

    # Update momentum: Δ(ρ_k w_k) = ρ_k · Δv_k  (since dp/dt = 0 at fixed ρ)
    U_new = U.copy()
    U_new[1] += rho_m * dv_m * factor_m
    U_new[3] += rho_c * dv_c * factor_c

    return U_new


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def make_inflow_state(scenario) -> np.ndarray:
    """Build inflow ghost-cell state from a scenario's inflow densities."""
    rho_m = np.array([scenario.inflow_moto])
    rho_c = np.array([scenario.inflow_car])
    rho_tot = total_pce_density(rho_m, rho_c)
    v_m = equilibrium_speed(rho_tot, MOTO)
    v_c = equilibrium_speed(rho_tot, CAR)
    return prim_to_cons(rho_m, v_m, rho_c, v_c).flatten()


def make_signal_mask(phases: np.ndarray, signal_cells: np.ndarray,
                     n_cells: int) -> np.ndarray:
    """
    Create flux-blocking mask from signal phases.

    phases       : (N_signals,) array of 0 (main-green) or 1 (side-green).
                   When side-green (phase=1), main road is blocked at that cell.
    signal_cells : (N_signals,) cell indices where signals are located.
    n_cells      : total number of cells.

    Returns
    -------
    mask : (n_cells + 1,) array of 0/1 for each interface.
           Interface j is between cell j-1 and cell j.
    """
    mask = np.ones(n_cells + 1, dtype=np.float64)
    for i, cell_idx in enumerate(signal_cells):
        if phases[i] == 1:  # side-green → block main road
            # Block the interface just downstream of the signal cell
            iface = cell_idx + 1
            if 0 <= iface <= n_cells:
                mask[iface] = 0.0
    return mask


def simulate(U: np.ndarray, duration: float, signal_phases: np.ndarray,
             signal_cells: np.ndarray, inflow_state: np.ndarray,
             side_injections: dict = None) -> np.ndarray:
    """
    Advance the ARZ system by `duration` seconds.

    Parameters
    ----------
    U              : (4, N) conservative state
    duration       : simulation time (s)
    signal_phases  : (N_signals,) current phases (0=main-green, 1=side-green)
    signal_cells   : (N_signals,) cell indices of signals
    inflow_state   : (4,) ghost cell state for left boundary
    side_injections: dict with keys 'cells', 'rho_moto_rate', 'rho_car_rate'
                     Injection rates (veh/m/s) at each signal cell when side is green.

    Returns
    -------
    U : updated (4, N) conservative state
    """
    N = U.shape[1]
    dx = DX
    t = 0.0

    mask = make_signal_mask(signal_phases, signal_cells, N)

    while t < duration:
        # Adaptive Δt from CFL
        a_max = max_wavespeed(U)
        dt = CFL * dx / max(a_max, EPS_RHO)
        dt = min(dt, duration - t)  # don't overshoot

        # 1. Hyperbolic step (LxF)
        U = lax_friedrichs_step(U, dt, dx, signal_mask=mask, inflow_state=inflow_state)

        # 2. Relaxation source
        U = apply_relaxation(U, dt)

        # 3. Side-street injection (at signal cells with side-green)
        if side_injections is not None:
            for i, cell_idx in enumerate(signal_cells):
                if signal_phases[i] == 1:  # side-green → inject
                    # Add density at this cell
                    drho_m = side_injections['rho_moto_rate'][i] * dt
                    drho_c = side_injections['rho_car_rate'][i] * dt
                    U[0, cell_idx] += drho_m
                    U[2, cell_idx] += drho_c

                    # Set momentum for injected vehicles at local equilibrium
                    rho_m_local = np.maximum(U[0, cell_idx], EPS_RHO)
                    rho_c_local = np.maximum(U[2, cell_idx], EPS_RHO)
                    rho_m_arr = np.array([rho_m_local])
                    rho_c_arr = np.array([rho_c_local])
                    rho_tot_local = total_pce_density(rho_m_arr, rho_c_arr)
                    ve_m = equilibrium_speed(rho_tot_local, MOTO)[0]
                    ve_c = equilibrium_speed(rho_tot_local, CAR)[0]
                    p_m = pressure(rho_m_arr, rho_c_arr, MOTO)[0]
                    p_c = pressure(rho_m_arr, rho_c_arr, CAR)[0]
                    U[1, cell_idx] = rho_m_local * (ve_m + p_m)
                    U[3, cell_idx] = rho_c_local * (ve_c + p_c)

        t += dt

    # Final clip
    U[0] = np.maximum(U[0], 0.0)
    U[2] = np.maximum(U[2], 0.0)

    return U


def init_uniform(scenario) -> np.ndarray:
    """Initialize uniform state from a scenario."""
    rho_m = np.full(N_CELLS, scenario.rho_moto_init)
    rho_c = np.full(N_CELLS, scenario.rho_car_init)
    rho_tot = total_pce_density(rho_m, rho_c)
    v_m = equilibrium_speed(rho_tot, MOTO)
    v_c = equilibrium_speed(rho_tot, CAR)
    return prim_to_cons(rho_m, v_m, rho_c, v_c)
