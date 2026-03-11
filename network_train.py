"""
network_train.py — Training pipeline for the 2×2 grid network case study.

Phase 1 (proof):  10k steps → sanity check
Phase 2 (full):   60k steps → comprehensive evaluation

Usage:
    python network_train.py                # Full pipeline (proof + full)
    python network_train.py --proof-only   # Proof of concept only
    python network_train.py --full-only    # Skip proof, go straight to 60k
"""

import os
import sys
import json
import csv
import time
import argparse
import numpy as np
from pathlib import Path

# Ensure python/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import PHI, MOTO, CAR
from network_params import (
    NETWORK_SCENARIOS, NETWORK_SCENARIO_NAMES, N_JUNCTIONS, JUNCTION_IDS,
    LINKS, TOTAL_CELLS,
    NET_DQN_CONFIG, NET_N_STEPS_EPISODE,
    NET_PROOF_TIMESTEPS, NET_FULL_TIMESTEPS,
    NET_PROOF_EVAL_EPISODES, NET_FULL_EVAL_EPISODES,
    NET_BASELINE_MAIN_STEPS, NET_BASELINE_SIDE_STEPS,
)
from network_env import TrafficNetworkEnv, NetworkFixedTimingController
from network_solver import get_network_pce_density
from solver import cons_to_prim, total_pce_density

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def sanity_check():
    """Verify the network environment produces sensible dynamics."""
    print("\n" + "=" * 60)
    print("SANITY CHECK: Network Environment")
    print("=" * 60)

    env = TrafficNetworkEnv(scenario_name="moderate")
    obs, info = env.reset(seed=42)
    print(f"  Scenario : {info['scenario']}")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Links    : {list(LINKS.keys())} ({TOTAL_CELLS} cells total)")
    print(f"  Junctions: {JUNCTION_IDS} ({N_JUNCTIONS})")

    # Run several actions and check reward varies
    rewards_by_action = {}
    for test_action in [0, 5, 10, 15]:
        env.reset(seed=42)
        total_reward = 0.0
        for _ in range(10):
            obs, r, term, trunc, info = env.step(test_action)
            total_reward += r
        rewards_by_action[test_action] = total_reward

    print(f"\n  Rewards after 10 steps with constant action:")
    for a, r in rewards_by_action.items():
        bits = f"{a:04b}"
        print(f"    Action {a:2d} ({bits}): reward = {r:.4f}")

    unique_rewards = len(set(f"{r:.6f}" for r in rewards_by_action.values()))
    if unique_rewards < 2:
        print("  WARNING: Rewards do not vary across actions!")
        return False
    print(f"  OK: {unique_rewards} distinct reward values across 4 actions")

    # Check observation evolves
    env.reset(seed=42)
    obs_init = env._get_obs()
    for _ in range(15):
        env.step(15)  # All NS-green → blocks EW traffic
    obs_after = env._get_obs()
    change = np.abs(obs_after - obs_init).max()
    print(f"\n  Max obs change after 15 NS-green steps: {change:.4f}")
    if change < 0.001:
        print("  WARNING: Observations barely change!")
        return False
    print("  OK: Observations evolve meaningfully")

    # Check side queues
    queues = env.side_queues
    q_total = sum(queues.values())
    print(f"  Side queues: {dict(queues)}")
    print(f"  Total queue: {q_total:.1f} PCE")

    env.close()
    print("\n  SANITY CHECK PASSED")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_controller(controller, scenario_name, n_episodes,
                        seed=0, is_baseline=False):
    """Evaluate a controller on a specific scenario."""
    env = TrafficNetworkEnv(scenario_name=scenario_name)
    episode_rewards = []
    episode_densities = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if is_baseline:
            controller.reset()

        total_reward = 0.0
        densities = []
        done = False

        while not done:
            if is_baseline:
                action, _ = controller.predict(obs, deterministic=True)
            else:
                action, _ = controller.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Track network-wide PCE density
            mean_pce = get_network_pce_density(env.link_states)
            densities.append(float(mean_pce))

        episode_rewards.append(total_reward)
        episode_densities.append(np.mean(densities))

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_density": float(np.mean(episode_densities)),
        "std_density": float(np.std(episode_densities)),
        "all_rewards": [float(r) for r in episode_rewards],
    }


def run_evaluation(model, n_episodes, label):
    """Evaluate model + baseline across all network scenarios."""
    baseline = NetworkFixedTimingController(NET_BASELINE_MAIN_STEPS,
                                           NET_BASELINE_SIDE_STEPS)

    results = {"scenarios": {}, "overall": {}}
    all_bl = []
    all_rl = []

    print(f"\n{'─' * 60}")
    print(f"EVALUATION: {label} ({n_episodes} episodes/scenario)")
    print(f"{'─' * 60}")
    print(f"{'Scenario':<12} {'Baseline':>10} {'DQN':>10} {'Δ%':>8}")
    print(f"{'─' * 42}")

    for sname in NETWORK_SCENARIO_NAMES:
        bl = evaluate_controller(baseline, sname, n_episodes, seed=100,
                                 is_baseline=True)
        rl = evaluate_controller(model, sname, n_episodes, seed=100,
                                 is_baseline=False)

        delta = 0.0
        if abs(bl["mean_reward"]) > 1e-10:
            delta = (rl["mean_reward"] - bl["mean_reward"]) / abs(bl["mean_reward"]) * 100

        results["scenarios"][sname] = {
            "baseline": bl,
            "dqn": rl,
            "improvement_pct": delta,
        }
        all_bl.extend(bl["all_rewards"])
        all_rl.extend(rl["all_rewards"])

        print(f"{sname:<12} {bl['mean_reward']:>10.3f} "
              f"{rl['mean_reward']:>10.3f} {delta:>+7.2f}%")

    overall_bl = float(np.mean(all_bl))
    overall_rl = float(np.mean(all_rl))
    overall_delta = (overall_rl - overall_bl) / abs(overall_bl) * 100 \
        if abs(overall_bl) > 1e-10 else 0.0

    results["overall"] = {
        "baseline_mean": overall_bl,
        "dqn_mean": overall_rl,
        "improvement_pct": overall_delta,
    }

    print(f"{'─' * 42}")
    print(f"{'OVERALL':<12} {overall_bl:>10.3f} {overall_rl:>10.3f} "
          f"{overall_delta:>+7.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_dqn(total_timesteps, label):
    """Train DQN on the network environment."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {label} ({total_timesteps:,} steps)")
    print(f"{'=' * 60}")

    train_env = Monitor(TrafficNetworkEnv())
    eval_env = Monitor(TrafficNetworkEnv(scenario_name="moderate"))

    config = NET_DQN_CONFIG.copy()
    policy = config.pop("policy")

    model = DQN(policy, train_env, verbose=1, **config)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(DATA_DIR / "net_best_model"),
        log_path=str(DATA_DIR / "net_eval_logs"),
        eval_freq=max(total_timesteps // 20, 500),
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Collect episode rewards
    training_rewards = []
    try:
        training_rewards = [float(r) for r in train_env.get_episode_rewards()]
    except Exception:
        pass
    if not training_rewards:
        try:
            training_rewards = [float(r) for r in train_env.return_queue]
        except Exception:
            pass

    train_env.close()
    eval_env.close()

    training_info = {
        "total_timesteps": total_timesteps,
        "training_time_s": elapsed,
        "n_episodes": len(training_rewards),
        "episode_rewards": training_rewards,
        "mean_reward": float(np.mean(training_rewards)) if training_rewards else None,
    }
    return model, training_info


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train DQN for multi-class ARZ network traffic control")
    parser.add_argument("--proof-only", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()

    # ── Sanity check ────────────────────────────────────────────────────
    if not args.full_only:
        ok = sanity_check()
        if not ok:
            print("\nSanity check failed. Aborting.")
            sys.exit(1)

    # ── Phase 1: Proof ──────────────────────────────────────────────────
    if not args.full_only:
        model, train_info = train_dqn(NET_PROOF_TIMESTEPS, "Network Proof")

        with open(DATA_DIR / "net_proof_training.json", "w") as f:
            json.dump(train_info, f, indent=2)

        eval_results = run_evaluation(model, NET_PROOF_EVAL_EPISODES, "Net Proof")
        eval_results["training"] = train_info

        with open(DATA_DIR / "net_proof_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        print(f"\n  Proof results saved to {DATA_DIR / 'net_proof_results.json'}")

        if args.proof_only:
            print("\n  --proof-only: Stopping here.")
            return

    # ── Phase 2: Full training ──────────────────────────────────────────
    model, train_info = train_dqn(NET_FULL_TIMESTEPS, "Network Full")

    with open(DATA_DIR / "net_full_training.json", "w") as f:
        json.dump(train_info, f, indent=2)

    if train_info["episode_rewards"]:
        with open(DATA_DIR / "net_training_monitor.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
            for i, r in enumerate(train_info["episode_rewards"]):
                writer.writerow([i + 1, r])

    eval_results = run_evaluation(model, NET_FULL_EVAL_EPISODES, "Network Full")
    eval_results["training"] = train_info

    with open(DATA_DIR / "net_full_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Scenario breakdown for figure generation
    breakdown = {"scenarios": {}, "overall": eval_results["overall"]}
    for sname in NETWORK_SCENARIO_NAMES:
        s = eval_results["scenarios"][sname]
        breakdown["scenarios"][sname] = {
            "baseline_reward": s["baseline"]["mean_reward"],
            "baseline_reward_std": s["baseline"]["std_reward"],
            "dqn_reward": s["dqn"]["mean_reward"],
            "dqn_reward_std": s["dqn"]["std_reward"],
            "improvement_pct": s["improvement_pct"],
            "baseline_density": s["baseline"]["mean_density"],
            "dqn_density": s["dqn"]["mean_density"],
        }

    with open(DATA_DIR / "net_scenario_breakdown.json", "w") as f:
        json.dump(breakdown, f, indent=2)

    print(f"\n  Full results saved to {DATA_DIR}")

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("NETWORK CASE STUDY — FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Topology : 2×2 grid, {N_JUNCTIONS} junctions, "
          f"{len(LINKS)} links, {TOTAL_CELLS} cells")
    print(f"  Training : {NET_FULL_TIMESTEPS:,} steps, "
          f"{train_info['n_episodes']} episodes, "
          f"{train_info['training_time_s']:.0f}s")
    print(f"  Baseline : {eval_results['overall']['baseline_mean']:.3f}")
    print(f"  DQN      : {eval_results['overall']['dqn_mean']:.3f}")
    print(f"  Improvement: {eval_results['overall']['improvement_pct']:+.2f}%")


if __name__ == "__main__":
    main()
