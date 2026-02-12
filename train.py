"""
train.py — Two-phase training pipeline for the multi-class ARZ RL environment.

Phase 1 (proof):  10k steps → quick sanity check + brief evaluation
Phase 2 (full):   60k steps → comprehensive multi-scenario evaluation

Usage:
    python train.py                # Full pipeline (proof + full)
    python train.py --proof-only   # Proof of concept only
    python train.py --full-only    # Skip proof, go straight to 60k
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Ensure python/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import (
    SCENARIOS, SCENARIO_NAMES,
    DQN_CONFIG, N_STEPS_EPISODE,
    PROOF_TIMESTEPS, FULL_TIMESTEPS,
    PROOF_EVAL_EPISODES, FULL_EVAL_EPISODES,
    BASELINE_MAIN_STEPS, BASELINE_SIDE_STEPS,
)
from environment import TrafficCorridorEnv, FixedTimingController

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def sanity_check_environment():
    """Verify the environment produces sensible dynamics."""
    print("\n" + "=" * 60)
    print("SANITY CHECK: Environment Dynamics")
    print("=" * 60)

    env = TrafficCorridorEnv(scenario_name="moderate")
    obs, info = env.reset(seed=42)
    print(f"  Scenario: {info['scenario']}")
    print(f"  Initial obs shape: {obs.shape}")
    print(f"  Initial obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Run a few steps with different actions, check reward varies
    rewards_by_action = {}
    for test_action in [0, 7, 3, 5]:
        env.reset(seed=42)
        total_reward = 0.0
        for _ in range(10):
            obs, r, term, trunc, info = env.step(test_action)
            total_reward += r
        rewards_by_action[test_action] = total_reward

    print(f"\n  Rewards after 10 steps with constant action:")
    for a, r in rewards_by_action.items():
        binary = f"{a:03b}"
        print(f"    Action {a} ({binary}): reward = {r:.4f}")

    unique_rewards = len(set(f"{r:.6f}" for r in rewards_by_action.values()))
    if unique_rewards < 2:
        print("  WARNING: Rewards do not vary across actions!")
        return False
    else:
        print(f"  OK: {unique_rewards} distinct reward values across 4 actions")

    # Check that densities change
    env.reset(seed=42)
    obs_init = env._get_obs()
    for _ in range(15):
        env.step(7)  # all side-green → should build queues
    obs_after = env._get_obs()
    density_change = np.abs(obs_after - obs_init).max()
    print(f"\n  Max observation change after 15 side-green steps: {density_change:.4f}")
    if density_change < 0.001:
        print("  WARNING: Observations barely change!")
        return False
    else:
        print("  OK: Observations evolve meaningfully")

    # Check side queues
    print(f"  Side queues after 15 side-green steps: {env.side_queues}")
    queue_sum = env.side_queues.sum()
    print(f"  Total side queue: {queue_sum:.1f} PCE")

    env.close()
    print("\n  SANITY CHECK PASSED")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_controller(controller, scenario_name: str, n_episodes: int,
                        seed: int = 0, is_baseline: bool = False):
    """
    Evaluate a controller on a specific scenario.

    Returns dict with mean_reward, std_reward, mean_density, mean_throughput.
    """
    env = TrafficCorridorEnv(scenario_name=scenario_name)
    episode_rewards = []
    episode_densities = []
    episode_throughputs = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if is_baseline:
            controller.reset()

        total_reward = 0.0
        densities = []
        throughputs = []

        done = False
        while not done:
            if is_baseline:
                action, _ = controller.predict(obs, deterministic=True)
            else:
                action, _ = controller.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Track metrics
            from solver import cons_to_prim, total_pce_density
            from params import PHI as _PHI
            rho_m, v_m, rho_c, v_c = cons_to_prim(env.U)
            rho_pce = total_pce_density(rho_m, rho_c)
            densities.append(float(np.mean(rho_pce)))

            # Throughput: PCE flow at right boundary (PCE/h)
            q_pce = float(_PHI[0] * rho_m[-1] * v_m[-1] + _PHI[1] * rho_c[-1] * v_c[-1])
            throughputs.append(q_pce * 3600)  # convert to PCE/h

        episode_rewards.append(total_reward)
        episode_densities.append(np.mean(densities))
        episode_throughputs.append(np.mean(throughputs))

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_density": float(np.mean(episode_densities)),
        "std_density": float(np.std(episode_densities)),
        "mean_throughput": float(np.mean(episode_throughputs)),
        "std_throughput": float(np.std(episode_throughputs)),
        "all_rewards": [float(r) for r in episode_rewards],
    }


def run_evaluation(model, n_episodes: int, label: str):
    """Evaluate model + baseline across all scenarios."""
    baseline = FixedTimingController(BASELINE_MAIN_STEPS, BASELINE_SIDE_STEPS)

    results = {"scenarios": {}, "overall": {}}
    all_bl_rewards = []
    all_rl_rewards = []

    print(f"\n{'─' * 60}")
    print(f"EVALUATION: {label} ({n_episodes} episodes/scenario)")
    print(f"{'─' * 60}")
    print(f"{'Scenario':<12} {'Baseline':>10} {'DQN':>10} {'Δ%':>8}")
    print(f"{'─' * 42}")

    for sname in SCENARIO_NAMES:
        bl_res = evaluate_controller(baseline, sname, n_episodes, seed=100,
                                     is_baseline=True)
        rl_res = evaluate_controller(model, sname, n_episodes, seed=100,
                                     is_baseline=False)

        delta = 0.0
        if abs(bl_res["mean_reward"]) > 1e-10:
            # Improvement: since rewards are negative, higher is better
            delta = (rl_res["mean_reward"] - bl_res["mean_reward"]) / abs(bl_res["mean_reward"]) * 100

        results["scenarios"][sname] = {
            "baseline": bl_res,
            "dqn": rl_res,
            "improvement_pct": delta,
        }

        all_bl_rewards.extend(bl_res["all_rewards"])
        all_rl_rewards.extend(rl_res["all_rewards"])

        print(f"{sname:<12} {bl_res['mean_reward']:>10.3f} "
              f"{rl_res['mean_reward']:>10.3f} {delta:>+7.2f}%")

    # Overall
    overall_bl = float(np.mean(all_bl_rewards))
    overall_rl = float(np.mean(all_rl_rewards))
    overall_delta = (overall_rl - overall_bl) / abs(overall_bl) * 100 if abs(overall_bl) > 1e-10 else 0

    results["overall"] = {
        "baseline_mean": overall_bl,
        "dqn_mean": overall_rl,
        "improvement_pct": overall_delta,
    }

    print(f"{'─' * 42}")
    print(f"{'OVERALL':<12} {overall_bl:>10.3f} {overall_rl:>10.3f} {overall_delta:>+7.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_dqn(total_timesteps: int, label: str):
    """Train a DQN agent and return the model + training metrics."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {label} ({total_timesteps:,} steps)")
    print(f"{'=' * 60}")

    # Training env (random scenarios)
    train_env = Monitor(TrafficCorridorEnv())

    # Eval env (moderate scenario for stable eval signal)
    eval_env = Monitor(TrafficCorridorEnv(scenario_name="moderate"))

    # Build DQN config
    config = DQN_CONFIG.copy()
    policy = config.pop("policy")

    model = DQN(
        policy,
        train_env,
        verbose=1,
        **config,
    )

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(DATA_DIR / "best_model"),
        log_path=str(DATA_DIR / "eval_logs"),
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

    print(f"\n  Training completed in {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")

    # Collect training metrics from monitor
    monitor_df = None
    try:
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        monitor_path = str(Path(train_env.get_wrapper_attr("filename")).parent
                           if hasattr(train_env, "get_wrapper_attr") else DATA_DIR)
    except Exception:
        pass

    # Save the episode rewards from the monitor wrapper
    training_rewards = []
    if hasattr(train_env, 'get_episode_rewards'):
        training_rewards = list(train_env.get_episode_rewards())
    elif hasattr(train_env, 'episode_returns'):
        training_rewards = list(train_env.episode_returns)

    # Try getting rewards from the wrapper
    try:
        ep_rewards = train_env.get_episode_rewards()
        training_rewards = [float(r) for r in ep_rewards]
    except Exception:
        pass

    # Also try return_queue
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
        "std_reward": float(np.std(training_rewards)) if training_rewards else None,
    }

    return model, training_info


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train DQN for multi-class ARZ traffic control")
    parser.add_argument("--proof-only", action="store_true", help="Run proof of concept only (10k steps)")
    parser.add_argument("--full-only", action="store_true", help="Skip proof, go straight to 60k steps")
    args = parser.parse_args()

    # ── Sanity check ────────────────────────────────────────────────────
    if not args.full_only:
        ok = sanity_check_environment()
        if not ok:
            print("\nSanity check failed. Aborting.")
            sys.exit(1)

    # ── Phase 1: Proof of concept ───────────────────────────────────────
    if not args.full_only:
        model, train_info = train_dqn(PROOF_TIMESTEPS, "Proof of Concept")

        # Save training info
        with open(DATA_DIR / "proof_training.json", "w") as f:
            json.dump(train_info, f, indent=2)

        # Brief evaluation
        eval_results = run_evaluation(model, PROOF_EVAL_EPISODES, "Proof")
        eval_results["training"] = train_info

        with open(DATA_DIR / "proof_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        print(f"\n  Proof results saved to {DATA_DIR / 'proof_results.json'}")

        if args.proof_only:
            print("\n  --proof-only: Stopping here.")
            return

    # ── Phase 2: Full training ──────────────────────────────────────────
    model, train_info = train_dqn(FULL_TIMESTEPS, "Full Training")

    # Save training info
    with open(DATA_DIR / "full_training.json", "w") as f:
        json.dump(train_info, f, indent=2)

    # Save episode rewards as CSV for figure generation
    if train_info["episode_rewards"]:
        import csv
        with open(DATA_DIR / "training_monitor.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
            for i, r in enumerate(train_info["episode_rewards"]):
                writer.writerow([i + 1, r])

    # Comprehensive evaluation
    eval_results = run_evaluation(model, FULL_EVAL_EPISODES, "Full")
    eval_results["training"] = train_info

    with open(DATA_DIR / "full_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Save scenario breakdown separately for figure generation
    scenario_data = {
        "scenarios": {},
        "overall": eval_results["overall"],
    }
    for sname in SCENARIO_NAMES:
        s = eval_results["scenarios"][sname]
        scenario_data["scenarios"][sname] = {
            "baseline_reward": s["baseline"]["mean_reward"],
            "baseline_reward_std": s["baseline"]["std_reward"],
            "dqn_reward": s["dqn"]["mean_reward"],
            "dqn_reward_std": s["dqn"]["std_reward"],
            "improvement_pct": s["improvement_pct"],
            "baseline_density": s["baseline"]["mean_density"],
            "dqn_density": s["dqn"]["mean_density"],
            "baseline_throughput": s["baseline"]["mean_throughput"],
            "dqn_throughput": s["dqn"]["mean_throughput"],
        }

    with open(DATA_DIR / "scenario_breakdown.json", "w") as f:
        json.dump(scenario_data, f, indent=2)

    print(f"\n  Full results saved to {DATA_DIR}")
    print(f"  Files: full_results.json, training_monitor.csv, scenario_breakdown.json")

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Training steps: {FULL_TIMESTEPS:,}")
    print(f"  Episodes trained: {train_info['n_episodes']}")
    print(f"  Overall baseline reward: {eval_results['overall']['baseline_mean']:.3f}")
    print(f"  Overall DQN reward:      {eval_results['overall']['dqn_mean']:.3f}")
    print(f"  Overall improvement:     {eval_results['overall']['improvement_pct']:+.2f}%")


if __name__ == "__main__":
    main()
