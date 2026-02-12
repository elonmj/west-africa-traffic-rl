"""
generate_all_figures.py — Publication-quality Case Study figures.

Produces 5 figures for the thesis / JITS article:
  Fig 1: Corridor schematic (topology + signal positions + scenario table)
  Fig 2: Multi-class fundamental diagrams (flow-density + speed-density)
  Fig 3: Demand scenario profiles (visual comparison)
  Fig 4: DQN training curve with baseline reference
  Fig 5: Scenario performance comparison (grouped bars + metrics)

Output: images/chapter3/

Usage:
    python generate_all_figures.py              # from full results
    python generate_all_figures.py --proof      # from proof results
    python generate_all_figures.py --show       # also display interactively
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import (
    V_FREE, RHO_MAX_PCE, BETA, PHI, ALPHA, TAU,
    L_CORRIDOR, N_CELLS, DX, N_SIGNALS, SIGNAL_CELLS,
    SCENARIOS, SCENARIO_NAMES,
    DT_DECISION, N_STEPS_EPISODE,
    BASELINE_MAIN_STEPS, BASELINE_SIDE_STEPS,
    DQN_CONFIG, FULL_TIMESTEPS,
)

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # -> New/
DATA_DIR = Path(__file__).resolve().parent / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "images" / "chapter3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════
STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLOR_BL = "#777777"       # baseline gray
COLOR_RL = "#2171B5"       # DQN blue
COLOR_MOTO = "#2171B5"     # motorcycle blue
COLOR_CAR = "#CB181D"      # car red
COLOR_SIGNAL = "#E6550D"   # signal orange
COLOR_SIDE = "#31A354"     # side-street green
COLOR_ROAD = "#F0EDE4"     # road surface cream
COLOR_DARK = "#333333"     # annotations

SCENARIO_COLORS = {
    "light": "#74C476",
    "moderate": "#FDD835",
    "heavy": "#FB8C00",
    "saturated": "#E53935",
}


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1: URBAN ARTERIAL MAP — Bird's-eye view
# ═══════════════════════════════════════════════════════════════════════════

def _draw_building_block(ax, x0, y0, w, h, color="#E8E0D4", label=None):
    """Draw a city block with slight texture."""
    block = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=2",
                           facecolor=color, edgecolor="#C8C0B4",
                           linewidth=0.6, zorder=1)
    ax.add_patch(block)
    # Subtle hatching for "built" texture
    for dx in np.arange(x0 + 8, x0 + w - 5, 18):
        for dy in np.arange(y0 + 8, y0 + h - 5, 14):
            ax.add_patch(Rectangle((dx, dy), 10, 8,
                                   facecolor="#D8D0C4", edgecolor="none",
                                   alpha=0.5, zorder=1.5))
    if label:
        ax.text(x0 + w / 2, y0 + h / 2, label, ha="center", va="center",
                fontsize=5, color="#999999", style="italic", zorder=2)


def generate_corridor_schematic(dpi=300, show=False):
    """Bird's-eye urban arterial map with 3 signalized intersections."""
    plt.rcParams.update(STYLE)

    fig = plt.figure(figsize=(7.16, 6.8))

    # =====================================================================
    # TOP PANEL — Urban map (bird's-eye view)
    # =====================================================================
    ax = fig.add_axes([0.02, 0.40, 0.96, 0.58])
    ax.set_xlim(-120, 1650)
    ax.set_ylim(-170, 200)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Coordinate system ---
    road_hw = 22  # half-width of main road
    side_hw = 16  # half-width of side streets
    signal_positions_m = SIGNAL_CELLS * DX  # [375, 750, 1125]

    # Title
    ax.text(750, 195, "Case Study: Urban Arterial with Mixed Traffic",
            ha="center", va="top", fontsize=11.5, fontweight="bold",
            color=COLOR_DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC",
                      lw=0.5, alpha=0.9))

    # ── MAIN ARTERIAL ROAD ──────────────────────────────────────────────
    # Asphalt surface
    road_rect = FancyBboxPatch((-30, -road_hw), L_CORRIDOR + 60, 2 * road_hw,
                               boxstyle="round,pad=2",
                               facecolor="#D6D2CA", edgecolor="#B0ADA6",
                               linewidth=0.8, zorder=2)
    ax.add_patch(road_rect)

    # Sidewalks (thin strips on each side)
    for sign in [1, -1]:
        sw_y = sign * road_hw
        sw_h = 5 * sign
        sw = Rectangle((-30, sw_y), L_CORRIDOR + 60, sw_h,
                        facecolor="#C8C4BB", edgecolor="none", zorder=2.1)
        ax.add_patch(sw)

    # Road edge lines (solid white)
    ax.plot([-30, L_CORRIDOR + 30], [road_hw, road_hw],
            color="white", lw=1.0, zorder=3)
    ax.plot([-30, L_CORRIDOR + 30], [-road_hw, -road_hw],
            color="white", lw=1.0, zorder=3)

    # Center dashed line
    seg_starts = []
    seg_ends = []
    x_cursor = 10
    while x_cursor < L_CORRIDOR - 10:
        # Skip the intersection zones
        in_intersection = False
        for sp in signal_positions_m:
            if sp - side_hw - 5 < x_cursor < sp + side_hw + 5:
                in_intersection = True
                break
        if not in_intersection:
            seg_starts.append(x_cursor)
            seg_ends.append(min(x_cursor + 18, L_CORRIDOR - 10))
        x_cursor += 28

    for s, e in zip(seg_starts, seg_ends):
        ax.plot([s, e], [0, 0], color="#FFFFFF", lw=1.2, zorder=3, alpha=0.7)

    # Direction arrows (chevrons on road surface)
    arrow_style = dict(arrowstyle="-|>", color="#FFFFFF", lw=1.8,
                       mutation_scale=12)
    for xa in [100, 260, 550, 900, 1250, 1420]:
        ax.annotate("", xy=(xa + 30, 8), xytext=(xa, 8),
                    arrowprops=arrow_style, zorder=3.5)
        # Motorcycles weave in the other lane too
        if xa % 300 < 200:
            ax.annotate("", xy=(xa + 25, -7), xytext=(xa, -7),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR_MOTO,
                                        lw=1.0, mutation_scale=8, alpha=0.5),
                        zorder=3.5)

    # Small vehicle markers as shapes instead of emoji (font-safe)
    for vx, vy, vw, vh, vc, vl in [
        (155, 7, 16, 8, COLOR_CAR, "V"),     # car
        (125, -9, 10, 5, COLOR_MOTO, "M"),   # moto
        (525, 7, 16, 8, COLOR_CAR, "V"),
        (505, -7, 10, 5, COLOR_MOTO, "M"),
        (485, -10, 10, 5, COLOR_MOTO, "M"),
        (1295, 7, 16, 8, COLOR_CAR, "V"),
        (1265, -7, 10, 5, COLOR_MOTO, "M"),
        (950, 8, 16, 8, COLOR_CAR, "V"),
        (930, -6, 10, 5, COLOR_MOTO, "M"),
    ]:
        veh = FancyBboxPatch((vx - vw/2, vy - vh/2), vw, vh,
                             boxstyle="round,pad=1",
                             facecolor=vc, edgecolor="white",
                             linewidth=0.4, alpha=0.7, zorder=4)
        ax.add_patch(veh)
        ax.text(vx, vy, vl, ha="center", va="center",
                fontsize=4, color="white", fontweight="bold", zorder=4.5)

    # ── SIDE STREETS ────────────────────────────────────────────────────
    side_length = 80  # how far the side streets extend

    for i, xpos in enumerate(signal_positions_m):
        # Side street N (going up)
        side_n = Rectangle((xpos - side_hw, road_hw),
                            2 * side_hw, side_length,
                            facecolor="#D6D2CA", edgecolor="#B0ADA6",
                            linewidth=0.6, zorder=2)
        ax.add_patch(side_n)
        # Sidewalks
        ax.add_patch(Rectangle((xpos - side_hw - 4, road_hw),
                                4, side_length, facecolor="#C8C4BB",
                                edgecolor="none", zorder=2.1))
        ax.add_patch(Rectangle((xpos + side_hw, road_hw),
                                4, side_length, facecolor="#C8C4BB",
                                edgecolor="none", zorder=2.1))

        # Side street S (going down)
        side_s = Rectangle((xpos - side_hw, -road_hw - side_length),
                            2 * side_hw, side_length,
                            facecolor="#D6D2CA", edgecolor="#B0ADA6",
                            linewidth=0.6, zorder=2)
        ax.add_patch(side_s)
        ax.add_patch(Rectangle((xpos - side_hw - 4, -road_hw - side_length),
                                4, side_length, facecolor="#C8C4BB",
                                edgecolor="none", zorder=2.1))
        ax.add_patch(Rectangle((xpos + side_hw, -road_hw - side_length),
                                4, side_length, facecolor="#C8C4BB",
                                edgecolor="none", zorder=2.1))

        # Center line on side streets
        ax.plot([xpos, xpos], [road_hw + 10, road_hw + side_length - 5],
                color="white", lw=0.6, ls="--", alpha=0.5, zorder=3)
        ax.plot([xpos, xpos], [-road_hw - 10, -road_hw - side_length + 5],
                color="white", lw=0.6, ls="--", alpha=0.5, zorder=3)

        # Side-street demand arrows (vehicles entering from side)
        ax.annotate("", xy=(xpos + 5, road_hw + 5),
                    xytext=(xpos + 5, road_hw + side_length - 10),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_SIDE,
                                    lw=2.0), zorder=5)
        ax.annotate("", xy=(xpos - 5, -road_hw - 5),
                    xytext=(xpos - 5, -road_hw - side_length + 10),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_SIDE,
                                    lw=2.0), zorder=5)

        # Side street labels
        ax.text(xpos, road_hw + side_length + 5,
                f"Rue latérale {i + 1}",
                ha="center", va="bottom", fontsize=6.5, color=COLOR_SIDE,
                fontweight="bold")

        # ── Traffic Light ───────────────────────────────────────────────
        # Light on NE corner of intersection
        lx = xpos + side_hw + 6
        ly = road_hw + 4
        light_bg = FancyBboxPatch((lx - 5, ly - 2), 10, 18,
                                  boxstyle="round,pad=1.5",
                                  facecolor="#2C2C2C", edgecolor="#111111",
                                  linewidth=0.5, zorder=6)
        ax.add_patch(light_bg)
        ax.plot(lx, ly + 2, "o", color="#FF3333", ms=3.5, zorder=7)
        ax.plot(lx, ly + 7, "o", color="#FFCC00", ms=3.5, zorder=7)
        ax.plot(lx, ly + 12, "o", color="#33CC33", ms=3.5, zorder=7)

        # Signal label
        ax.text(lx + 10, ly + 7, f"$S_{{{i + 1}}}$",
                ha="left", va="center", fontsize=10,
                fontweight="bold", color=COLOR_SIGNAL,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=COLOR_SIGNAL, lw=0.6, alpha=0.9))

        # Intersection zone highlight
        inter_rect = Rectangle((xpos - side_hw - 2, -road_hw - 2),
                                2 * side_hw + 4, 2 * road_hw + 4,
                                facecolor="none", edgecolor=COLOR_SIGNAL,
                                linewidth=1.2, ls="--", zorder=4, alpha=0.5)
        ax.add_patch(inter_rect)

    # ── CITY BLOCKS ─────────────────────────────────────────────────────
    block_h = 55
    block_gap = 6

    # Blocks between intersections (above and below the road)
    block_regions = [
        (30, signal_positions_m[0] - side_hw - block_gap - 30),
        (signal_positions_m[0] + side_hw + block_gap,
         signal_positions_m[1] - side_hw - block_gap - signal_positions_m[0] - side_hw - block_gap),
        (signal_positions_m[1] + side_hw + block_gap,
         signal_positions_m[2] - side_hw - block_gap - signal_positions_m[1] - side_hw - block_gap),
        (signal_positions_m[2] + side_hw + block_gap,
         L_CORRIDOR - 30 - signal_positions_m[2] - side_hw - block_gap),
    ]

    block_labels_n = ["Résidentiel", "Commercial", "Marché", "Résidentiel"]
    block_labels_s = ["Atelier", "École", "Banque", "Résidentiel"]

    for idx, (bx, bw) in enumerate(block_regions):
        if bw > 20:
            # North blocks
            _draw_building_block(ax, bx, road_hw + block_gap + 5, bw, block_h,
                                 color="#E8E0D4",
                                 label=block_labels_n[idx] if bw > 80 else None)
            # South blocks
            _draw_building_block(ax, bx, -road_hw - block_gap - 5 - block_h, bw, block_h,
                                 color="#DED6CA",
                                 label=block_labels_s[idx] if bw > 80 else None)

    # ── INFLOW / OUTFLOW LABELS ─────────────────────────────────────────
    # Inflow (left)
    ax.annotate("", xy=(-15, 0), xytext=(-80, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.4",
                                color=COLOR_MOTO, lw=3.0), zorder=6)
    ax.text(-90, 0,
            "Flux entrant\n(motos + voitures)",
            ha="right", va="center", fontsize=7, color=COLOR_DARK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB",
                      ec=COLOR_MOTO, lw=0.6, alpha=0.9))

    # Outflow (right)
    ax.annotate("", xy=(L_CORRIDOR + 80, 0),
                xytext=(L_CORRIDOR + 15, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.4",
                                color="#666666", lw=3.0), zorder=6)
    ax.text(L_CORRIDOR + 90, 0,
            "Sortie",
            ha="left", va="center", fontsize=7, color=COLOR_DARK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5",
                      ec="#999999", lw=0.6, alpha=0.9))

    # ── DISTANCE ANNOTATIONS (below everything) ────────────────────────
    ann_y = -140
    positions = [0] + list(signal_positions_m) + [L_CORRIDOR]
    labels_between = ["375 m", "375 m", "375 m", "375 m"]

    for j in range(len(positions) - 1):
        mid = (positions[j] + positions[j + 1]) / 2
        dist = positions[j + 1] - positions[j]
        ax.annotate("", xy=(positions[j + 1], ann_y),
                    xytext=(positions[j], ann_y),
                    arrowprops=dict(arrowstyle="<->", color="#888888",
                                    lw=0.7))
        ax.text(mid, ann_y - 8, f"{dist:.0f} m",
                ha="center", va="top", fontsize=6, color="#888888")

    # Total
    ax.annotate("", xy=(L_CORRIDOR, ann_y - 22),
                xytext=(0, ann_y - 22),
                arrowprops=dict(arrowstyle="<->", color=COLOR_DARK, lw=1.0))
    ax.text(750, ann_y - 28,
            f"L = {L_CORRIDOR:.0f} m  •  {N_CELLS} cellules  •  Δx = {DX:.0f} m",
            ha="center", va="top", fontsize=7, color=COLOR_DARK,
            fontweight="bold")

    # ── NORTH ARROW + CONTEXT ───────────────────────────────────────────
    ax.annotate("N", xy=(1580, 170), fontsize=8, ha="center", va="bottom",
                fontweight="bold", color="#555555")
    ax.annotate("", xy=(1580, 165), xytext=(1580, 135),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5))

    # =====================================================================
    # BOTTOM PANEL — Parameter tables
    # =====================================================================
    ax_tab = fig.add_axes([0.04, 0.02, 0.92, 0.36])
    ax_tab.axis("off")

    col_titles = [
        "Paramètres des classes de véhicules",
        "Configuration de l'environnement RL",
    ]

    left_data = [
        ["Paramètre", "Motos (k=0)", "Voitures (k=1)"],
        ["Facteur UVP φ", f"{PHI[0]:.2f}", f"{PHI[1]:.1f}"],
        ["Vitesse libre", f"{V_FREE[0]*3.6:.0f} km/h", f"{V_FREE[1]*3.6:.0f} km/h"],
        ["Densité max ρ_max", f"{RHO_MAX_PCE[0]*1000:.0f} UVP/km", f"{RHO_MAX_PCE[1]*1000:.0f} UVP/km"],
        ["Exposant β", f"{BETA[0]:.1f}", f"{BETA[1]:.1f}"],
        ["Relaxation τ", f"{TAU[0]:.0f} s", f"{TAU[1]:.0f} s"],
        ["Interaction α_self", f"{ALPHA[0,0]:.1f}", f"{ALPHA[1,1]:.1f}"],
        ["Interaction α_cross", f"α₀₁ = {ALPHA[0,1]:.1f}", f"α₁₀ = {ALPHA[1,0]:.1f}"],
    ]

    right_data = [
        ["Paramètre", "Valeur"],
        ["Intervalle de décision", f"{DT_DECISION:.0f} s"],
        ["Pas / épisode", f"{N_STEPS_EPISODE}  ({N_STEPS_EPISODE * DT_DECISION:.0f} s)"],
        ["Espace d'actions", f"Discret(8) — {N_SIGNALS} signaux binaires"],
        ["Dim. observation", f"18  (6 × {N_SIGNALS} intersections)"],
        ["Algorithme", "DQN (Stable-Baselines3)"],
        ["Réseau", f"{DQN_CONFIG['policy_kwargs']['net_arch']}"],
        ["Pas d'entraînement", f"{FULL_TIMESTEPS:,}"],
        ["Taux d'apprentissage", f"{DQN_CONFIG['learning_rate']}"],
        ["Baseline", f"Fixe {BASELINE_MAIN_STEPS}:{BASELINE_SIDE_STEPS} (vert:latéral)"],
    ]

    table_left = ax_tab.table(
        cellText=left_data[1:],
        colLabels=left_data[0],
        loc="center left",
        bbox=[0.0, 0.05, 0.48, 0.85],
        cellLoc="center",
    )
    table_left.auto_set_font_size(False)
    table_left.set_fontsize(7)

    for j in range(len(left_data[0])):
        cell = table_left[0, j]
        cell.set_facecolor("#2171B5")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    for i in range(1, len(left_data)):
        for j in range(len(left_data[0])):
            cell = table_left[i, j]
            cell.set_facecolor("#F8F8F8" if i % 2 == 0 else "white")
            cell.set_edgecolor("#DDDDDD")

    ax_tab.text(0.24, 0.95, col_titles[0], ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COLOR_DARK,
                transform=ax_tab.transAxes)

    # Draw right table
    table_right = ax_tab.table(
        cellText=right_data[1:],
        colLabels=right_data[0],
        loc="center right",
        bbox=[0.52, 0.05, 0.48, 0.85],
        cellLoc="center",
    )
    table_right.auto_set_font_size(False)
    table_right.set_fontsize(7)

    for j in range(len(right_data[0])):
        cell = table_right[0, j]
        cell.set_facecolor(COLOR_SIGNAL)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    for i in range(1, len(right_data)):
        for j in range(len(right_data[0])):
            cell = table_right[i, j]
            cell.set_facecolor("#F8F8F8" if i % 2 == 0 else "white")
            cell.set_edgecolor("#DDDDDD")

    ax_tab.text(0.76, 0.95, col_titles[1], ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COLOR_DARK,
                transform=ax_tab.transAxes)

    out = OUTPUT_DIR / "fig_corridor_schematic.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2: FUNDAMENTAL DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════

def generate_fundamental_diagrams(dpi=300, show=False):
    """Two-panel figure: flow-density and speed-density."""
    plt.rcParams.update(STYLE)

    rho_max_display = max(RHO_MAX_PCE) * 1000 * 1.02
    rho_pce_km = np.linspace(0, rho_max_display, 1000)
    rho_pce_m = rho_pce_km / 1000.0

    classes = {
        "Motorcycles": {"idx": 0, "color": COLOR_MOTO, "ls": "-", "marker": "o"},
        "Cars": {"idx": 1, "color": COLOR_CAR, "ls": "--", "marker": "s"},
    }

    speeds = {}
    flows = {}
    for name, c in classes.items():
        k = c["idx"]
        ratio = np.clip(1.0 - rho_pce_m / RHO_MAX_PCE[k], 0.0, 1.0)
        v = V_FREE[k] * ratio ** BETA[k]
        speeds[name] = v * 3.6
        flows[name] = rho_pce_m * v * 3600

    fig, (ax_flow, ax_speed) = plt.subplots(1, 2, figsize=(7.16, 3.2),
                                             constrained_layout=True)

    # ── Flow-Density ────────────────────────────────────────────────────
    for name, c in classes.items():
        ax_flow.plot(rho_pce_km, flows[name],
                     color=c["color"], ls=c["ls"], lw=1.8, label=name, zorder=3)

        k = c["idx"]
        rho_cr_pce_m = RHO_MAX_PCE[k] / (1.0 + BETA[k])
        rho_cr_pce_km = rho_cr_pce_m * 1000
        ratio_cr = np.clip(1.0 - rho_cr_pce_m / RHO_MAX_PCE[k], 0.0, 1.0)
        v_cr = V_FREE[k] * ratio_cr ** BETA[k]
        q_cr = rho_cr_pce_m * v_cr * 3600

        ax_flow.plot(rho_cr_pce_km, q_cr, c["marker"],
                     color=c["color"], ms=6, zorder=4,
                     markeredgecolor="white", markeredgewidth=0.6)

        offset_y = 120 if name == "Cars" else -200
        ax_flow.annotate(
            f"$q_{{\\max}}$={q_cr:.0f}\n$\\rho_{{cr}}$={rho_cr_pce_km:.0f}",
            xy=(rho_cr_pce_km, q_cr),
            xytext=(rho_cr_pce_km + 15, q_cr + offset_y),
            fontsize=7, color=c["color"],
            arrowprops=dict(arrowstyle="-|>", color=c["color"],
                            lw=0.7, connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c["color"],
                      lw=0.5, alpha=0.85),
        )

    # Shade scenario density ranges on flow-density plot
    for sname, sc in SCENARIOS.items():
        rho_init_pce = sc.rho_moto_init * PHI[0] + sc.rho_car_init * PHI[1]
        rho_init_km = rho_init_pce * 1000
        ax_flow.axvline(rho_init_km, color=SCENARIO_COLORS[sname],
                        ls=":", lw=0.9, alpha=0.7)
        ax_flow.text(rho_init_km, ax_flow.get_ylim()[1] * 0.02,
                     sname[0].upper(), fontsize=6, ha="center",
                     color=SCENARIO_COLORS[sname], fontweight="bold")

    ax_flow.set_xlabel(r"PCE Density $\rho_{\mathrm{tot}}$ (PCE/km)")
    ax_flow.set_ylabel(r"PCE Flow $q$ (PCE/h)")
    ax_flow.set_xlim(0, rho_max_display)
    ax_flow.set_ylim(bottom=0)
    ax_flow.legend(loc="upper right", framealpha=0.9)
    ax_flow.set_title("(a) Flow\u2013Density", fontweight="bold")
    ax_flow.xaxis.set_major_locator(ticker.MultipleLocator(50))

    # ── Speed-Density ───────────────────────────────────────────────────
    for name, c in classes.items():
        ax_speed.plot(rho_pce_km, speeds[name],
                      color=c["color"], ls=c["ls"], lw=1.8, label=name, zorder=3)

    # Crossover
    diff = np.array(speeds["Motorcycles"]) - np.array(speeds["Cars"])
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        d0, d1 = diff[idx], diff[idx + 1]
        frac = -d0 / (d1 - d0)
        rho_cross = rho_pce_km[idx] + frac * (rho_pce_km[idx + 1] - rho_pce_km[idx])
        v_cross = speeds["Motorcycles"][idx] + frac * (speeds["Motorcycles"][idx + 1] - speeds["Motorcycles"][idx])

        ax_speed.axvline(rho_cross, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax_speed.plot(rho_cross, v_cross, "D", color=COLOR_DARK, ms=5, zorder=5,
                      markeredgecolor="white", markeredgewidth=0.5)
        ax_speed.annotate(
            f"Crossover\n$\\rho$={rho_cross:.0f} PCE/km",
            xy=(rho_cross, v_cross),
            xytext=(rho_cross - 55, v_cross + 14),
            fontsize=7, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=COLOR_DARK, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666666",
                      lw=0.5, alpha=0.85),
        )

        mask = rho_pce_km >= rho_cross
        ax_speed.fill_between(
            rho_pce_km[mask],
            np.array(speeds["Cars"])[mask],
            np.array(speeds["Motorcycles"])[mask],
            alpha=0.08, color=COLOR_MOTO,
            label="Gap-filling regime",
        )

    ax_speed.set_xlabel(r"PCE Density $\rho_{\mathrm{tot}}$ (PCE/km)")
    ax_speed.set_ylabel(r"Equilibrium Speed $v_k^e$ (km/h)")
    ax_speed.set_xlim(0, rho_max_display)
    ax_speed.set_ylim(bottom=0)
    ax_speed.legend(loc="upper right", framealpha=0.9)
    ax_speed.set_title("(b) Speed\u2013Density", fontweight="bold")
    ax_speed.xaxis.set_major_locator(ticker.MultipleLocator(50))

    out = OUTPUT_DIR / "fig_fundamental_diagram.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3: DEMAND SCENARIO PROFILES
# ═══════════════════════════════════════════════════════════════════════════

def generate_scenario_profiles(dpi=300, show=False):
    """Visual comparison of the 4 demand scenarios with stacked bars."""
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0), constrained_layout=True)

    names = [s.capitalize() for s in SCENARIO_NAMES]
    x = np.arange(len(names))
    width = 0.55

    # --- Panel (a): Initial PCE densities (stacked motos + cars) ---
    ax = axes[0]
    moto_pce = [SCENARIOS[s].rho_moto_init * PHI[0] * 1000 for s in SCENARIO_NAMES]
    car_pce = [SCENARIOS[s].rho_car_init * PHI[1] * 1000 for s in SCENARIO_NAMES]

    bars_m = ax.bar(x, moto_pce, width, label="Motos (PCE)", color=COLOR_MOTO,
                    edgecolor="white", linewidth=0.5, zorder=3)
    bars_c = ax.bar(x, car_pce, width, bottom=moto_pce, label="Cars (PCE)",
                    color=COLOR_CAR, edgecolor="white", linewidth=0.5, zorder=3)

    # Total labels
    for i in range(len(names)):
        total = moto_pce[i] + car_pce[i]
        ax.text(i, total + 1.5, f"{total:.0f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=COLOR_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Init. Density (PCE/km)")
    ax.set_title("(a) Main-Road Demand", fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left")

    # Color bars by scenario
    for i, bar in enumerate(bars_m):
        bar.set_alpha(0.85)
    for i, bar in enumerate(bars_c):
        bar.set_alpha(0.85)

    # --- Panel (b): Side-street demand ---
    ax = axes[1]
    side_demands = [SCENARIOS[s].side_demand for s in SCENARIO_NAMES]
    moto_fracs = [SCENARIOS[s].moto_fraction for s in SCENARIO_NAMES]

    colors = [SCENARIO_COLORS[s] for s in SCENARIO_NAMES]
    bars = ax.bar(x, side_demands, width, color=colors,
                  edgecolor="white", linewidth=0.5, zorder=3)

    for i, (bar, mf) in enumerate(zip(bars, moto_fracs)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{mf:.0%} motos",
                ha="center", va="bottom", fontsize=5.5, color=COLOR_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Side Demand (PCE/s)")
    ax.set_title("(b) Side-Street Demand", fontweight="bold")

    # --- Panel (c): Scenario tension diagram ---
    ax = axes[2]
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("Main-Road Saturation", fontsize=8)
    ax.set_ylabel("Side-Street Pressure", fontsize=8)
    ax.set_title("(c) Control Tension", fontweight="bold")

    # Normalize to [0,1]
    max_main_pce = max(SCENARIOS[s].rho_moto_init * PHI[0] + SCENARIOS[s].rho_car_init * PHI[1]
                       for s in SCENARIO_NAMES)
    max_side = max(SCENARIOS[s].side_demand for s in SCENARIO_NAMES)

    for sname in SCENARIO_NAMES:
        sc = SCENARIOS[sname]
        main_sat = (sc.rho_moto_init * PHI[0] + sc.rho_car_init * PHI[1]) / max_main_pce
        side_press = sc.side_demand / max_side

        ax.scatter(main_sat, side_press, s=120, color=SCENARIO_COLORS[sname],
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(sname.capitalize(),
                    xy=(main_sat, side_press),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, fontweight="bold",
                    color=SCENARIO_COLORS[sname])

    # Quadrant labels
    ax.text(0.25, 0.75, "Easy\n(low main,\nhigh side)", ha="center",
            va="center", fontsize=5.5, color="#AAAAAA", style="italic")
    ax.text(0.85, 0.25, "Critical\n(high main,\nlow side)", ha="center",
            va="center", fontsize=5.5, color="#AAAAAA", style="italic")
    ax.text(0.85, 0.85, "Maximum\ntension", ha="center",
            va="center", fontsize=5.5, color="#AAAAAA", style="italic")

    # Diagonal tension arrow
    ax.annotate("", xy=(1.15, 0.15), xytext=(0.15, 1.15),
                arrowprops=dict(arrowstyle="-|>", color="#CCCCCC",
                                lw=1.5, ls="--"))
    ax.text(0.72, 0.58, "RL advantage →", ha="center", va="center",
            fontsize=6, color="#AAAAAA", rotation=-45, style="italic")

    out = OUTPUT_DIR / "fig_scenario_profiles.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4: TRAINING CURVE
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_curve(results_file: Path, dpi=300, show=False):
    """Training reward over episodes with baseline reference and phases."""
    plt.rcParams.update(STYLE)

    with open(results_file) as f:
        results = json.load(f)

    rewards = results.get("training", {}).get("episode_rewards", [])
    if not rewards:
        print("  WARNING: No training rewards found, skipping training curve.")
        return

    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards)

    # Smoothing
    window = max(len(rewards) // 40, 5)
    kernel = np.ones(window) / window
    smooth = np.convolve(rewards, kernel, mode='same')

    bl_reward = results["overall"]["baseline_mean"]
    rl_reward = results["overall"]["dqn_mean"]
    total_steps = results.get("training", {}).get("total_timesteps", "?")

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)

    # Raw rewards (transparent)
    ax.plot(episodes, rewards, color="#B0C4DE", lw=0.3, alpha=0.4, zorder=1)

    # Smoothed
    ax.plot(episodes, smooth, color=COLOR_RL, lw=1.8,
            label="DQN (smoothed)", zorder=3)

    # Baseline reference
    ax.axhline(bl_reward, color=COLOR_CAR, ls="--", lw=1.3,
               label=f"Fixed-Timing Baseline ({bl_reward:.1f})", zorder=2)

    # Final DQN level
    ax.axhline(rl_reward, color=COLOR_RL, ls=":", lw=0.9, alpha=0.7,
               label=f"DQN Final ({rl_reward:.1f})", zorder=2)

    # Exploration phase shading
    n_explore = int(len(rewards) * 0.15)  # exploration_fraction = 0.15
    ax.axvspan(1, n_explore, alpha=0.06, color=COLOR_SIGNAL,
               label="Exploration phase")

    # Improvement annotation with box
    delta = results["overall"]["improvement_pct"]
    mid_y = (rl_reward + bl_reward) / 2
    ax.annotate(
        f"Improvement\n$\\Delta$ = {delta:+.1f}%",
        xy=(len(rewards) * 0.65, mid_y),
        xytext=(len(rewards) * 0.82, bl_reward - 1.5),
        fontsize=8, ha="center", fontweight="bold", color=COLOR_RL,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_RL, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB",
                  ec=COLOR_RL, lw=0.8, alpha=0.95),
    )

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Reward (higher = better)")
    ax.set_title(f"DQN Learning Curve ({total_steps:,} steps, {len(rewards)} episodes)",
                 fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax.set_xlim(1, len(rewards))

    out = OUTPUT_DIR / "fig_training_curve.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5: SCENARIO COMPARISON (COMPREHENSIVE)
# ═══════════════════════════════════════════════════════════════════════════

def generate_scenario_comparison(results_file: Path, dpi=300, show=False):
    """Comprehensive comparison: reward bars + throughput + density reduction."""
    plt.rcParams.update(STYLE)

    with open(results_file) as f:
        results = json.load(f)

    scenarios = results.get("scenarios", {})
    if not scenarios:
        breakdown_file = results_file.parent / "scenario_breakdown.json"
        if breakdown_file.exists():
            with open(breakdown_file) as f:
                scenarios = json.load(f)["scenarios"]
        else:
            print("  WARNING: No scenario data found.")
            return

    # Extract data
    names = []
    bl_rewards = []
    rl_rewards = []
    improvements = []
    bl_densities = []
    rl_densities = []
    bl_throughputs = []
    rl_throughputs = []

    for sname in SCENARIO_NAMES:
        if sname not in scenarios:
            continue
        s = scenarios[sname]
        names.append(sname.capitalize())

        if "baseline" in s:
            bl_rewards.append(s["baseline"]["mean_reward"])
            rl_rewards.append(s["dqn"]["mean_reward"])
            bl_densities.append(s["baseline"].get("mean_density", 0))
            rl_densities.append(s["dqn"].get("mean_density", 0))
            bl_throughputs.append(s["baseline"].get("mean_throughput", 0))
            rl_throughputs.append(s["dqn"].get("mean_throughput", 0))
            improvements.append(s["improvement_pct"])
        else:
            bl_rewards.append(s.get("baseline_reward", 0))
            rl_rewards.append(s.get("dqn_reward", 0))
            improvements.append(s.get("improvement_pct", 0))

    # Add overall
    overall = results.get("overall", {})
    names.append("Overall")
    bl_rewards.append(overall.get("baseline_mean", np.mean(bl_rewards)))
    rl_rewards.append(overall.get("dqn_mean", np.mean(rl_rewards)))
    improvements.append(overall.get("improvement_pct", 0))

    x = np.arange(len(names))
    width = 0.30

    fig, (ax_reward, ax_imp) = plt.subplots(1, 2, figsize=(7.16, 3.5),
                                             gridspec_kw={"width_ratios": [2, 1]},
                                             constrained_layout=True)

    # --- Left: Grouped bar chart (rewards) ---
    bars_bl = ax_reward.bar(x - width / 2, [-r for r in bl_rewards], width,
                            label="Fixed-Timing", color=COLOR_BL,
                            edgecolor="white", linewidth=0.5, zorder=3)
    bars_rl = ax_reward.bar(x + width / 2, [-r for r in rl_rewards], width,
                            label="DQN", color=COLOR_RL,
                            edgecolor="white", linewidth=0.5, zorder=3)

    # Values on bars
    for i, (bl_bar, rl_bar) in enumerate(zip(bars_bl, bars_rl)):
        ax_reward.text(bl_bar.get_x() + bl_bar.get_width() / 2,
                       bl_bar.get_height() + 0.2,
                       f"{-bl_rewards[i]:.1f}", ha="center", va="bottom",
                       fontsize=5.5, color=COLOR_BL)
        ax_reward.text(rl_bar.get_x() + rl_bar.get_width() / 2,
                       rl_bar.get_height() + 0.2,
                       f"{-rl_rewards[i]:.1f}", ha="center", va="bottom",
                       fontsize=5.5, color=COLOR_RL, fontweight="bold")

    # Separator before "Overall"
    ax_reward.axvline(len(names) - 1.5, color="#DDDDDD", ls="-", lw=0.8)

    ax_reward.set_xticks(x)
    ax_reward.set_xticklabels(names, fontsize=7.5)
    ax_reward.set_ylabel("Cost (−Reward, lower = better)")
    ax_reward.set_title("(a) Performance by Scenario", fontweight="bold")
    ax_reward.legend(loc="upper left", framealpha=0.9)
    ax_reward.set_ylim(bottom=0)

    # --- Right: Improvement percentage (horizontal bars) ---
    colors_imp = [COLOR_RL if imp > 0 else COLOR_CAR for imp in improvements]

    bars_imp = ax_imp.barh(x, improvements, height=0.5,
                           color=colors_imp, edgecolor="white",
                           linewidth=0.5, zorder=3)

    for i, (bar, imp) in enumerate(zip(bars_imp, improvements)):
        xpos = bar.get_width() + 0.5 if imp > 0 else bar.get_width() - 0.5
        ha = "left" if imp > 0 else "right"
        ax_imp.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{imp:+.1f}%", ha=ha, va="center",
                    fontsize=7.5, fontweight="bold",
                    color=colors_imp[i])

    # Highlight overall
    ax_imp.axhline(len(names) - 1.5, color="#DDDDDD", ls="-", lw=0.8)

    ax_imp.axvline(0, color=COLOR_DARK, lw=0.5, zorder=2)
    ax_imp.set_yticks(x)
    ax_imp.set_yticklabels(names, fontsize=7.5)
    ax_imp.set_xlabel("Improvement (%)")
    ax_imp.set_title("(b) DQN vs. Baseline", fontweight="bold")
    ax_imp.invert_yaxis()

    # Shade positive improvement zone
    xlim = ax_imp.get_xlim()
    ax_imp.axvspan(0, xlim[1], alpha=0.03, color=COLOR_RL)
    ax_imp.axvspan(xlim[0], 0, alpha=0.03, color=COLOR_CAR)

    out = OUTPUT_DIR / "fig_rl_comparison.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate all case study figures")
    parser.add_argument("--proof", action="store_true",
                        help="Use proof results instead of full")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively")
    args = parser.parse_args()

    results_file = DATA_DIR / ("proof_results.json" if args.proof else "full_results.json")

    print(f"\nGenerating figures from: {results_file}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Fig 1: Corridor schematic + case study
    print("Fig 1: Corridor Schematic & Case Study")
    generate_corridor_schematic(show=args.show)

    # Fig 2: Fundamental diagrams
    print("\nFig 2: Fundamental Diagrams")
    generate_fundamental_diagrams(show=args.show)

    # Fig 3: Scenario profiles
    print("\nFig 3: Demand Scenario Profiles")
    generate_scenario_profiles(show=args.show)

    # Fig 4 & 5: Need training results
    if results_file.exists():
        print("\nFig 4: Training Curve")
        generate_training_curve(results_file, show=args.show)

        print("\nFig 5: Scenario Comparison")
        generate_scenario_comparison(results_file, show=args.show)
    else:
        print(f"\nResults file not found: {results_file}")
        print("Run train.py first. Only Figs 1-3 were generated.")

    print("\nDone — 5 figures generated.")


if __name__ == "__main__":
    main()
