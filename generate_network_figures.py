"""
generate_network_figures.py — Publication-quality figures for the network case study.

Produces 3 figures:
  Fig 6: Network schematic (2×2 grid topology with street names)
  Fig 7: Network DQN training curve with baseline reference
  Fig 8: Network scenario performance comparison (grouped bars)

Output: images/chapter3/

Usage:
    python generate_network_figures.py              # from full results
    python generate_network_figures.py --proof      # from proof results
    python generate_network_figures.py --show       # also display interactively
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from network_params import (
    LINKS, JUNCTIONS, JUNCTION_IDS, N_JUNCTIONS, TOTAL_CELLS,
    NETWORK_SCENARIOS, NETWORK_SCENARIO_NAMES,
    NET_DT_DECISION, NET_N_STEPS_EPISODE,
    NET_FULL_TIMESTEPS,
)

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "images" / "chapter3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STYLE (matches generate_all_figures.py)
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

COLOR_BL = "#777777"
COLOR_RL = "#2171B5"
COLOR_MOTO = "#2171B5"
COLOR_CAR = "#CB181D"
COLOR_SIGNAL = "#E6550D"
COLOR_SIDE = "#31A354"
COLOR_ROAD = "#F0EDE4"
COLOR_DARK = "#333333"

SCENARIO_COLORS = {
    "light": "#74C476",
    "moderate": "#FDD835",
    "heavy": "#FB8C00",
    "saturated": "#E53935",
}


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6: NETWORK SCHEMATIC
# ═══════════════════════════════════════════════════════════════════════════

def generate_network_schematic(dpi=300, show=False):
    """Bird's-eye view of the 2×2 grid network inspired by Quartier Ganhi."""
    plt.rcParams.update(STYLE)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    ax.set_xlim(-60, 520)
    ax.set_ylim(-80, 430)
    ax.set_aspect('equal')
    ax.axis('off')

    # Junction positions
    jpos = {
        'I1': (0, 300), 'I2': (400, 300),
        'I3': (0, 0),   'I4': (400, 0),
    }

    # Draw city blocks (background rectangles)
    blocks = [
        (-50, 320, 100, 90, "#E8E0D4"),   # NW
        (70, 320, 260, 90, "#DDD5C8"),     # N center
        (420, 320, 90, 90, "#E0D8CC"),     # NE
        (-50, 20, 100, 260, "#E5DDD0"),    # W
        (70, 20, 260, 260, "#D8D0C4"),     # Center block
        (420, 20, 90, 260, "#E2DAD0"),     # E
        (-50, -70, 100, 70, "#E8E0D4"),    # SW
        (70, -70, 260, 70, "#DDD5C8"),     # S center
        (420, -70, 90, 70, "#E0D8CC"),     # SE
    ]
    for bx, by, bw, bh, bc in blocks:
        block = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=3",
                               facecolor=bc, edgecolor="#C8C0B4",
                               linewidth=0.5, zorder=0)
        ax.add_patch(block)

    # Draw roads (wide bands)
    road_width = 18
    hw = road_width / 2

    # Horizontal roads (E-W)
    ax.add_patch(Rectangle((-50, 300 - hw), 560, road_width,
                           facecolor=COLOR_ROAD, edgecolor='#999',
                           linewidth=0.4, zorder=1))
    ax.add_patch(Rectangle((-50, -hw), 560, road_width,
                           facecolor=COLOR_ROAD, edgecolor='#999',
                           linewidth=0.4, zorder=1))

    # Vertical roads (N-S)
    ax.add_patch(Rectangle((-hw, -70), road_width, 490,
                           facecolor=COLOR_ROAD, edgecolor='#999',
                           linewidth=0.4, zorder=1))
    ax.add_patch(Rectangle((400 - hw, -70), road_width, 490,
                           facecolor=COLOR_ROAD, edgecolor='#999',
                           linewidth=0.4, zorder=1))

    # Road center lines (dashed)
    ax.plot([-50, 510], [300, 300], '--', color='#FFA726', lw=0.7, zorder=2)
    ax.plot([-50, 510], [0, 0], '--', color='#FFA726', lw=0.7, zorder=2)
    ax.plot([0, 0], [-70, 420], '--', color='#FFA726', lw=0.7, zorder=2)
    ax.plot([400, 400], [-70, 420], '--', color='#FFA726', lw=0.7, zorder=2)

    # Flow direction arrows on roads
    arrow_style = dict(arrowstyle='->', color=COLOR_RL, lw=1.8, mutation_scale=14)

    # L1: I1→I2 (East, 299.8m, Rue Gén. Félix Éboué)
    ax.annotate('', xy=(340, 306), xytext=(60, 306),
                arrowprops=arrow_style, zorder=5)
    ax.text(200, 316, 'L1: Rue Gén. F. Éboué (300 m)',
            ha='center', va='bottom', fontsize=7, color=COLOR_RL,
            fontweight='bold', zorder=5)

    # L2: I3→I4 (East, 298.8m, Rue José Firmin Santos)
    ax.annotate('', xy=(340, 6), xytext=(60, 6),
                arrowprops=arrow_style, zorder=5)
    ax.text(200, 16, 'L2: Rue J.F. Santos (299 m)',
            ha='center', va='bottom', fontsize=7, color=COLOR_RL,
            fontweight='bold', zorder=5)

    # L3: I1→I3 (South, 279.3m, Av. Capitaine Adjovi)
    ax.annotate('', xy=(-6, 60), xytext=(-6, 240),
                arrowprops=arrow_style, zorder=5)
    ax.text(-25, 150, 'L3: Av. Cap.\nAdjovi\n(279 m)',
            ha='right', va='center', fontsize=7, color=COLOR_RL,
            fontweight='bold', rotation=90, zorder=5)

    # L4: I2→I4 (South, 281.5m, Av. Augustin Nikoué)
    ax.annotate('', xy=(406, 60), xytext=(406, 240),
                arrowprops=arrow_style, zorder=5)
    ax.text(425, 150, 'L4: Av. Aug.\nNikoué\n(282 m)',
            ha='left', va='center', fontsize=7, color=COLOR_RL,
            fontweight='bold', rotation=90, zorder=5)

    # Draw junctions as traffic-light circles
    for jid, (jx, jy) in jpos.items():
        circle = Circle((jx, jy), 12, facecolor=COLOR_SIGNAL,
                        edgecolor='white', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(jx, jy, jid, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=11)

    # External inflows (green arrows)
    ext_style = dict(arrowstyle='->', color=COLOR_SIDE, lw=1.5, mutation_scale=12)

    # ext_W_I1, ext_W_I3
    ax.annotate('', xy=(-15, 294), xytext=(-50, 294),
                arrowprops=ext_style, zorder=5)
    ax.text(-50, 288, 'ext W', fontsize=6, color=COLOR_SIDE, ha='left')

    ax.annotate('', xy=(-15, -6), xytext=(-50, -6),
                arrowprops=ext_style, zorder=5)
    ax.text(-50, -12, 'ext W', fontsize=6, color=COLOR_SIDE, ha='left')

    # ext_N_I1, ext_N_I2
    ax.annotate('', xy=(6, 315), xytext=(6, 395),
                arrowprops=ext_style, zorder=5)
    ax.text(12, 390, 'ext N', fontsize=6, color=COLOR_SIDE, ha='left')

    ax.annotate('', xy=(394, 315), xytext=(394, 395),
                arrowprops=ext_style, zorder=5)
    ax.text(380, 390, 'ext N', fontsize=6, color=COLOR_SIDE, ha='right')

    # External outflows (gray arrows)
    out_style = dict(arrowstyle='->', color=COLOR_BL, lw=1.3, mutation_scale=10)

    # exit_E at I2
    ax.annotate('', xy=(460, 294), xytext=(415, 294),
                arrowprops=out_style, zorder=5)
    ax.text(465, 294, 'exit E', fontsize=6, color=COLOR_BL, ha='left', va='center')

    # exit_E + exit_S at I4
    ax.annotate('', xy=(460, -6), xytext=(415, -6),
                arrowprops=out_style, zorder=5)
    ax.text(465, -6, 'exit E', fontsize=6, color=COLOR_BL, ha='left', va='center')

    ax.annotate('', xy=(394, -50), xytext=(394, -15),
                arrowprops=out_style, zorder=5)
    ax.text(380, -55, 'exit S', fontsize=6, color=COLOR_BL, ha='right')

    # exit_S at I3
    ax.annotate('', xy=(6, -50), xytext=(6, -15),
                arrowprops=out_style, zorder=5)
    ax.text(12, -55, 'exit S', fontsize=6, color=COLOR_BL, ha='left')

    # Title
    ax.set_title('Quartier Ganhi Network — 2×2 Grid (Cotonou, Benin)',
                 fontweight='bold', fontsize=11, pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_SIGNAL, edgecolor='white', label='Signalized junction'),
        mpatches.FancyArrow(0, 0, 1, 0, width=0.3, color=COLOR_RL, label='Modeled link'),
        mpatches.FancyArrow(0, 0, 1, 0, width=0.3, color=COLOR_SIDE, label='External inflow'),
        mpatches.FancyArrow(0, 0, 1, 0, width=0.3, color=COLOR_BL, label='Network exit'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=7, framealpha=0.9, ncol=2)

    # Info box
    info_text = (f"4 junctions · 4 links · {TOTAL_CELLS} cells\n"
                 f"Action: Discrete(16) = 2⁴\n"
                 f"Obs: 24-dim (6 × 4 junctions)")
    ax.text(510, 420, info_text, fontsize=6.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#CCC', alpha=0.9))

    # Save
    for ext in ['png', 'pdf']:
        out = OUTPUT_DIR / f"fig_network_schematic.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7: NETWORK TRAINING CURVE
# ═══════════════════════════════════════════════════════════════════════════

def generate_network_training_curve(results_file: Path, dpi=300, show=False):
    """DQN training reward curve for the network case study."""
    plt.rcParams.update(STYLE)

    with open(results_file) as f:
        results = json.load(f)

    rewards = results.get("training", {}).get("episode_rewards", [])
    if not rewards:
        print("  WARNING: No training rewards found, skipping training curve.")
        return

    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards)

    window = max(len(rewards) // 40, 5)
    kernel = np.ones(window) / window
    smooth = np.convolve(rewards, kernel, mode='same')

    bl_reward = results["overall"]["baseline_mean"]
    rl_reward = results["overall"]["dqn_mean"]
    total_steps = results.get("training", {}).get("total_timesteps", "?")

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)

    ax.plot(episodes, rewards, color="#B0C4DE", lw=0.3, alpha=0.4, zorder=1)
    ax.plot(episodes, smooth, color=COLOR_RL, lw=1.8,
            label="DQN (smoothed)", zorder=3)
    ax.axhline(bl_reward, color=COLOR_CAR, ls="--", lw=1.3,
               label=f"Fixed-Timing Baseline ({bl_reward:.1f})", zorder=2)
    ax.axhline(rl_reward, color=COLOR_RL, ls=":", lw=0.9, alpha=0.7,
               label=f"DQN Final ({rl_reward:.1f})", zorder=2)

    n_explore = int(len(rewards) * 0.15)
    ax.axvspan(1, n_explore, alpha=0.06, color=COLOR_SIGNAL,
               label="Exploration phase")

    delta = results["overall"]["improvement_pct"]
    mid_y = (rl_reward + bl_reward) / 2
    ax.annotate(
        f"Improvement\n$\\Delta$ = {delta:+.1f}%",
        xy=(len(rewards) * 0.65, mid_y),
        xytext=(len(rewards) * 0.82, bl_reward - abs(bl_reward) * 0.08),
        fontsize=8, ha="center", fontweight="bold", color=COLOR_RL,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_RL, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB",
                  ec=COLOR_RL, lw=0.8, alpha=0.95),
    )

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Reward (higher = better)")
    ax.set_title(f"Network DQN Learning Curve "
                 f"({total_steps:,} steps, {len(rewards)} episodes)",
                 fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax.set_xlim(1, len(rewards))

    for ext in ['png', 'pdf']:
        out = OUTPUT_DIR / f"fig_network_training_curve.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 8: NETWORK SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def generate_network_scenario_comparison(results_file: Path, dpi=300, show=False):
    """Comparison of DQN vs fixed-timing across all scenarios."""
    plt.rcParams.update(STYLE)

    with open(results_file) as f:
        results = json.load(f)

    scenarios = results.get("scenarios", {})
    if not scenarios:
        bfile = results_file.parent / "net_scenario_breakdown.json"
        if bfile.exists():
            with open(bfile) as f:
                scenarios = json.load(f)["scenarios"]
        else:
            print("  WARNING: No scenario data found.")
            return

    names = []
    bl_rewards = []
    rl_rewards = []
    improvements = []
    bl_densities = []
    rl_densities = []

    for sname in NETWORK_SCENARIO_NAMES:
        if sname not in scenarios:
            continue
        s = scenarios[sname]
        names.append(sname.capitalize())

        if "baseline" in s:
            bl_rewards.append(s["baseline"]["mean_reward"])
            rl_rewards.append(s["dqn"]["mean_reward"])
            bl_densities.append(s["baseline"].get("mean_density", 0))
            rl_densities.append(s["dqn"].get("mean_density", 0))
        else:
            bl_rewards.append(s.get("baseline_reward", 0))
            rl_rewards.append(s.get("dqn_reward", 0))
            bl_densities.append(s.get("baseline_density", 0))
            rl_densities.append(s.get("dqn_density", 0))

        improvements.append(s.get("improvement_pct", 0))

    if not names:
        print("  No scenario results to plot.")
        return

    x = np.arange(len(names))
    w = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)

    # ── Panel A: Reward Comparison ──
    bars_bl = ax1.bar(x - w / 2, bl_rewards, w, label="Fixed-Timing",
                      color=COLOR_BL, edgecolor="white", linewidth=0.5)
    bars_rl = ax1.bar(x + w / 2, rl_rewards, w, label="DQN",
                      color=COLOR_RL, edgecolor="white", linewidth=0.5)

    for i, imp in enumerate(improvements):
        y_top = max(rl_rewards[i], bl_rewards[i])
        ax1.annotate(
            f"+{imp:.1f}%", xy=(x[i] + w / 2, rl_rewards[i]),
            xytext=(x[i] + w / 2, y_top + abs(y_top) * 0.08),
            fontsize=7, ha="center", fontweight="bold", color=COLOR_RL,
        )

    overall = results.get("overall", {})
    overall_delta = overall.get("improvement_pct", 0)
    ax1.text(0.98, 0.02, f"Overall: {overall_delta:+.1f}%",
             transform=ax1.transAxes, fontsize=9, fontweight="bold",
             ha="right", va="bottom", color=COLOR_RL,
             bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB",
                       ec=COLOR_RL, lw=0.8, alpha=0.95))

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Episode Reward (higher = better)")
    ax1.set_title("(a) Reward by Scenario", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper left")

    # ── Panel B: Density Reduction ──
    if any(d > 0 for d in bl_densities):
        density_reduction = [
            ((bl - rl) / bl * 100 if bl > 0 else 0)
            for bl, rl in zip(bl_densities, rl_densities)
        ]

        colors = [SCENARIO_COLORS.get(n.lower(), COLOR_RL) for n in names]
        bars = ax2.bar(x, density_reduction, 0.55, color=colors,
                       edgecolor="white", linewidth=0.5)

        for i, val in enumerate(density_reduction):
            ax2.text(x[i], val + 0.5, f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_ylabel("PCE Density Reduction (%)")
        ax2.set_title("(b) Density Reduction", fontweight="bold")
        ax2.axhline(0, color="black", lw=0.5)
    else:
        ax2.text(0.5, 0.5, "Density data\nnot available",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=10, color="#999")
        ax2.set_title("(b) Density Reduction", fontweight="bold")

    fig.suptitle("Network Case Study: DQN vs Fixed-Timing Signal Control",
                 fontweight="bold", fontsize=11, y=1.02)

    for ext in ['png', 'pdf']:
        out = OUTPUT_DIR / f"fig_network_comparison.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate network case study figures")
    parser.add_argument("--proof", action="store_true",
                        help="Use proof results instead of full")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Resolution (default 600)")
    args = parser.parse_args()

    prefix = "net_proof" if args.proof else "net_full"
    results_file = DATA_DIR / f"{prefix}_results.json"

    print(f"\n{'=' * 60}")
    print("GENERATING NETWORK CASE STUDY FIGURES")
    print(f"{'=' * 60}")
    print(f"  Results file: {results_file}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print(f"  DPI:          {args.dpi}")

    # Fig 6: Network schematic (always generated)
    print("\n[Fig 6] Network schematic...")
    generate_network_schematic(dpi=args.dpi, show=args.show)

    if not results_file.exists():
        print(f"\n  WARNING: {results_file} not found. "
              f"Run network_train.py first.")
        print("  Only the schematic was generated.")
        return

    # Fig 7: Training curve
    print("\n[Fig 7] Network training curve...")
    generate_network_training_curve(results_file, dpi=args.dpi, show=args.show)

    # Fig 8: Scenario comparison
    print("\n[Fig 8] Network scenario comparison...")
    generate_network_scenario_comparison(results_file, dpi=args.dpi,
                                         show=args.show)

    print(f"\n  All network figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
