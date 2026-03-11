"""
validate_riemann.py — Riemann problem validation for the multi-class ARZ solver.

Runs three canonical Riemann problems and generates a publication-quality figure:
  Test 1: Shock wave (single-class motorcycle)
  Test 2: Rarefaction wave (single-class car)
  Test 3: Multi-class interaction (motorcycle-car coupling)

Each test verifies that the LxF solver produces physically correct wave structures.
Output: images/chapter3/fig_riemann_validation.png  (and .pdf for vector)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from solver import (
    prim_to_cons, cons_to_prim, lax_friedrichs_step, apply_relaxation,
    total_pce_density, equilibrium_speed, max_wavespeed, pressure,
)
from params import (
    MOTO, CAR, DX, CFL, EPS_RHO, V_FREE, RHO_MAX_PHYS, PHI, RHO_MAX_PCE,
)


def run_riemann(rho_m_L, rho_m_R, rho_c_L, rho_c_R, N=200, T=30.0, dx=15.0):
    """
    Solve a Riemann problem on [0, N*dx] with initial discontinuity at midpoint.

    Returns (x, rho_m, v_m, rho_c, v_c) at time T.
    """
    x = np.linspace(dx / 2, N * dx - dx / 2, N)
    mid = N // 2

    # Initial densities
    rho_m = np.where(np.arange(N) < mid, rho_m_L, rho_m_R)
    rho_c = np.where(np.arange(N) < mid, rho_c_L, rho_c_R)

    # Equilibrium velocities
    rho_tot = total_pce_density(rho_m, rho_c)
    v_m = equilibrium_speed(rho_tot, MOTO)
    v_c = equilibrium_speed(rho_tot, CAR)

    # Conservative state
    U = prim_to_cons(rho_m, v_m, rho_c, v_c)

    # Inflow = left state
    inflow = prim_to_cons(
        np.array([rho_m_L]), np.array([equilibrium_speed(total_pce_density(
            np.array([rho_m_L]), np.array([rho_c_L])), MOTO)[0]]),
        np.array([rho_c_L]), np.array([equilibrium_speed(total_pce_density(
            np.array([rho_m_L]), np.array([rho_c_L])), CAR)[0]])
    ).flatten()

    # Time integration
    t = 0.0
    while t < T:
        a_max = max_wavespeed(U)
        dt = CFL * dx / max(a_max, EPS_RHO)
        dt = min(dt, T - t)
        U = lax_friedrichs_step(U, dt, dx, inflow_state=inflow)
        U = apply_relaxation(U, dt)
        t += dt

    rho_m_out, v_m_out, rho_c_out, v_c_out = cons_to_prim(U)
    return x, rho_m_out, v_m_out, rho_c_out, v_c_out


def main():
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, hspace=0.45, wspace=0.35)

    # ─── Test 1: Shock wave — Motorcycles ───────────────────────────────
    # High moto density on left, low on right → rightward-propagating shock
    rho_m_L, rho_m_R = 0.40, 0.05   # veh/m (i.e., 400 vs 50 veh/km)
    rho_c_L, rho_c_R = 0.01, 0.01   # negligible cars
    x, rm, vm, rc, vc = run_riemann(rho_m_L, rho_m_R, rho_c_L, rho_c_R,
                                     N=200, T=30.0)

    ax1a = fig.add_subplot(gs[0, 0])
    ax1a.plot(x, rm * 1000, 'b-', lw=1.5, label='Motos')
    ax1a.plot(x, rc * 1000, 'r--', lw=1.0, alpha=0.5, label='Cars')
    ax1a.axvline(x=1500, color='gray', ls=':', lw=0.8)
    ax1a.set_ylabel('Density (veh/km)')
    ax1a.set_title('Test 1: Shock Wave', fontweight='bold', fontsize=10)
    ax1a.legend(fontsize=8)
    ax1a.set_xlim(0, 3000)

    ax1b = fig.add_subplot(gs[0, 1])
    ax1b.plot(x, vm * 3.6, 'b-', lw=1.5)
    ax1b.plot(x, vc * 3.6, 'r--', lw=1.0, alpha=0.5)
    ax1b.set_ylabel('Speed (km/h)')
    ax1b.set_title('Test 1: Speed Profile', fontsize=10)
    ax1b.set_xlim(0, 3000)

    ax1c = fig.add_subplot(gs[0, 2])
    # Flow = rho * v
    ax1c.plot(x, rm * vm * 3600, 'b-', lw=1.5, label='Motos')
    ax1c.plot(x, rc * vc * 3600, 'r--', lw=1.0, alpha=0.5, label='Cars')
    ax1c.set_ylabel('Flow (veh/h)')
    ax1c.set_title('Test 1: Flow Profile', fontsize=10)
    ax1c.set_xlim(0, 3000)

    # ─── Test 2: Rarefaction wave — Cars ────────────────────────────────
    # Low car density on left, moderate on right → rarefaction fan
    rho_m_L2, rho_m_R2 = 0.01, 0.01   # negligible motos
    rho_c_L2, rho_c_R2 = 0.01, 0.10   # 10 vs 100 veh/km
    x2, rm2, vm2, rc2, vc2 = run_riemann(rho_m_L2, rho_m_R2, rho_c_L2, rho_c_R2,
                                          N=200, T=30.0)

    ax2a = fig.add_subplot(gs[1, 0])
    ax2a.plot(x2, rc2 * 1000, 'r-', lw=1.5, label='Cars')
    ax2a.plot(x2, rm2 * 1000, 'b--', lw=1.0, alpha=0.5, label='Motos')
    ax2a.axvline(x=1500, color='gray', ls=':', lw=0.8)
    ax2a.set_ylabel('Density (veh/km)')
    ax2a.set_title('Test 2: Rarefaction Wave', fontweight='bold', fontsize=10)
    ax2a.legend(fontsize=8)
    ax2a.set_xlim(0, 3000)

    ax2b = fig.add_subplot(gs[1, 1])
    ax2b.plot(x2, vc2 * 3.6, 'r-', lw=1.5)
    ax2b.plot(x2, vm2 * 3.6, 'b--', lw=1.0, alpha=0.5)
    ax2b.set_ylabel('Speed (km/h)')
    ax2b.set_title('Test 2: Speed Profile', fontsize=10)
    ax2b.set_xlim(0, 3000)

    ax2c = fig.add_subplot(gs[1, 2])
    ax2c.plot(x2, rc2 * vc2 * 3600, 'r-', lw=1.5, label='Cars')
    ax2c.plot(x2, rm2 * vm2 * 3600, 'b--', lw=1.0, alpha=0.5, label='Motos')
    ax2c.set_ylabel('Flow (veh/h)')
    ax2c.set_title('Test 2: Flow Profile', fontsize=10)
    ax2c.set_xlim(0, 3000)

    # ─── Test 3: Multi-class interaction ────────────────────────────────
    # Dense motorcycle platoon meets dense car platoon → cross-class coupling
    rho_m_L3, rho_m_R3 = 0.35, 0.02   # dense motos left, sparse right
    rho_c_L3, rho_c_R3 = 0.02, 0.08   # sparse cars left, dense right
    x3, rm3, vm3, rc3, vc3 = run_riemann(rho_m_L3, rho_m_R3, rho_c_L3, rho_c_R3,
                                          N=200, T=25.0)

    ax3a = fig.add_subplot(gs[2, 0])
    ax3a.plot(x3, rm3 * 1000, 'b-', lw=1.5, label='Motos')
    ax3a.plot(x3, rc3 * 1000, 'r-', lw=1.5, label='Cars')
    ax3a.axvline(x=1500, color='gray', ls=':', lw=0.8)
    ax3a.set_ylabel('Density (veh/km)')
    ax3a.set_xlabel('Position (m)')
    ax3a.set_title('Test 3: Multi-Class Interaction', fontweight='bold', fontsize=10)
    ax3a.legend(fontsize=8)
    ax3a.set_xlim(0, 3000)

    ax3b = fig.add_subplot(gs[2, 1])
    ax3b.plot(x3, vm3 * 3.6, 'b-', lw=1.5, label='Motos')
    ax3b.plot(x3, vc3 * 3.6, 'r-', lw=1.5, label='Cars')
    ax3b.set_ylabel('Speed (km/h)')
    ax3b.set_xlabel('Position (m)')
    ax3b.set_title('Test 3: Speed Profile', fontsize=10)
    ax3b.legend(fontsize=8)
    ax3b.set_xlim(0, 3000)

    ax3c = fig.add_subplot(gs[2, 2])
    # PCE density shows the combined impact
    rho_pce3 = PHI[MOTO] * rm3 + PHI[CAR] * rc3
    ax3c.plot(x3, rho_pce3 * 1000, 'k-', lw=1.8, label='Total PCE')
    ax3c.plot(x3, PHI[MOTO] * rm3 * 1000, 'b--', lw=1.0, alpha=0.7, label='Moto PCE')
    ax3c.plot(x3, PHI[CAR] * rc3 * 1000, 'r--', lw=1.0, alpha=0.7, label='Car PCE')
    ax3c.set_ylabel('PCE Density (PCE/km)')
    ax3c.set_xlabel('Position (m)')
    ax3c.set_title('Test 3: PCE Density', fontsize=10)
    ax3c.legend(fontsize=8)
    ax3c.set_xlim(0, 3000)

    # Global labels
    fig.suptitle('Riemann Problem Validation — Multi-Class ARZ Solver',
                 fontsize=13, fontweight='bold', y=0.98)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'images', 'chapter3')
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, 'fig_riemann_validation.png')
    pdf_path = os.path.join(out_dir, 'fig_riemann_validation.pdf')
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Convergence analysis: L1 error vs grid refinement
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Grid Convergence Analysis (Test 1: Shock) ===")
    Ns = [50, 100, 200, 400, 800]
    ref_N = 1600  # Reference solution on very fine grid
    dx_ref = 3000.0 / ref_N
    _, rm_ref, _, _, _ = run_riemann(0.40, 0.05, 0.01, 0.01,
                                     N=ref_N, T=30.0, dx=dx_ref)

    errors = []
    dxs = []
    for N in Ns:
        dx_test = 3000.0 / N
        _, rm_test, _, _, _ = run_riemann(0.40, 0.05, 0.01, 0.01,
                                           N=N, T=30.0, dx=dx_test)
        # Interpolate reference to test grid
        x_test = np.linspace(dx_test / 2, 3000 - dx_test / 2, N)
        x_ref = np.linspace(dx_ref / 2, 3000 - dx_ref / 2, ref_N)
        rm_ref_interp = np.interp(x_test, x_ref, rm_ref)
        err = np.mean(np.abs(rm_test - rm_ref_interp))
        errors.append(err)
        dxs.append(dx_test)
        print(f"  N={N:4d}, dx={dx_test:6.1f}m, L1 error={err:.6f}")

    # Compute convergence rate
    dxs = np.array(dxs)
    errors = np.array(errors)
    rates = np.diff(np.log(errors)) / np.diff(np.log(dxs))
    print(f"  Convergence rates: {rates}")
    avg_rate = np.mean(rates)
    print(f"  Average convergence rate: {avg_rate:.2f} (expected ~1.0 for LxF)")

    # Plot convergence
    fig2, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(dxs, errors, 'ko-', lw=2, ms=8, label='LxF solver')
    # Reference slope: first-order
    ref_line = errors[0] * (dxs / dxs[0]) ** 1.0
    ax.loglog(dxs, ref_line, 'r--', lw=1.5, alpha=0.7, label='$O(\\Delta x)$ reference')
    ax.set_xlabel('$\\Delta x$ (m)', fontsize=12)
    ax.set_ylabel('$L^1$ error', fontsize=12)
    ax.set_title(f'Grid Convergence (avg. rate = {avg_rate:.2f})', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    conv_png = os.path.join(out_dir, 'fig_convergence.png')
    conv_pdf = os.path.join(out_dir, 'fig_convergence.pdf')
    fig2.savefig(conv_png, dpi=600, bbox_inches='tight')
    fig2.savefig(conv_pdf, bbox_inches='tight')
    print(f"\nSaved: {conv_png}")
    print(f"Saved: {conv_pdf}")
    plt.close()


if __name__ == '__main__':
    main()
