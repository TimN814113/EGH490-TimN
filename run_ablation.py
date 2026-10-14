"""
run_ablation.py
EGH490-1 — Tim Nguyen N11505222

Isolates which parameter error in OLLAMA_CASUAL is responsible for its real
DDPG collisions (0.097/trial pooled — the worst rate found anywhere in the
report-replication study). Tests four configs — Expert, the original
Ollama-Casual, and two ablation variants that each revert a different subset
of Ollama-Casual's errors (see sim/configs.py for exactly what each variant
changes) — across the DDPG-Approx car-following model at N=25, 110 km/h,
AV in {0%, 25%, 50%} (the exact condition where the original collisions were
observed).

Deliberately narrower than run_report_replication.py: DDPG only (that's where
the collisions are), no LSTM/IDM, no density/speed/interaction sweeps — this
is a targeted diagnostic, not a full replication.

Usage:
    python run_ablation.py
    python run_ablation.py --trials 30 --t_sim 300
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sim.configs import EXPERT, OLLAMA_CASUAL, OLLAMA_CASUAL_FIX_ACC, OLLAMA_CASUAL_FIX_ALL_V0
from sim.runner import run_simulation


N_TRIALS_DEFAULT = 30   # more than the main study's 20 — collisions are a rare-ish
                        # count metric and benefit from extra trials for resolution
T_SIM_DEFAULT = 300.0


def main(n_trials=N_TRIALS_DEFAULT, t_sim=T_SIM_DEFAULT):
    configs = [
        EXPERT,
        OLLAMA_CASUAL,
        OLLAMA_CASUAL_FIX_ACC,
        OLLAMA_CASUAL_FIX_ALL_V0,
    ]
    av_pcts = [0.0, 0.25, 0.50]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("results") / f"ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    total = len(configs) * len(av_pcts) * n_trials
    print(f"\n{'='*80}")
    print("  Collision Ablation — EGH490-1 Tim Nguyen")
    print(f"  {total} total runs ({n_trials} trials/condition, t_sim={t_sim}s, DDPG only)")
    print(f"{'='*80}\n")

    for config in configs:
        for av_pct in av_pcts:
            sim_cfg = {"t_sim": t_sim, "n_veh": 25, "road_len": 800.0, "v_limit_kmh": 110.0}
            trial_summaries = []
            for trial in range(1, n_trials + 1):
                t0 = time.time()
                print(f"  {config['name']:32.32s} | AV={int(av_pct*100):3d}% | "
                      f"trial {trial}/{n_trials} ...", end=" ", flush=True)
                summary, _ = run_simulation(
                    config=config, cf_model="DDPG-Approx", av_pct=av_pct, sim_cfg=sim_cfg,
                )
                elapsed = time.time() - t0
                print(f"done ({elapsed:.2f}s)  collisions={summary['collision_count']:.2f}  "
                      f"jerk={summary['jerk_mean_ms3']:.2f}  ttc={summary['ttc_exposure_pct']:.3f}%")
                trial_summaries.append(summary)

            all_summaries.append({
                "config": config["name"],
                "av_pct": int(av_pct * 100),
                "n_trials": n_trials,
                "collision_mean": round(np.mean([s["collision_count"] for s in trial_summaries]), 4),
                "collision_max": round(max(s["collision_count"] for s in trial_summaries), 4),
                "collision_nonzero_trials": sum(1 for s in trial_summaries if s["collision_count"] > 0),
                "jerk_mean": round(np.mean([s["jerk_mean_ms3"] for s in trial_summaries]), 6),
                "ttc_mean": round(np.mean([s["ttc_exposure_pct"] for s in trial_summaries]), 4),
                "spacing_mse_mean": round(np.mean([s["spacing_mse_m2"] for s in trial_summaries]), 4),
            })

    summary_path = out_dir / "ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*90}")
    print(f"  {'Config':32s}  AV%   Coll.mean  Coll.max  NonzeroTrials  Jerk    TTC%")
    print(f"{'-'*90}")
    for r in all_summaries:
        print(f"  {r['config']:32s}  {r['av_pct']:3d}%  {r['collision_mean']:9.3f}  "
              f"{r['collision_max']:8.2f}  {r['collision_nonzero_trials']:13d}  "
              f"{r['jerk_mean']:6.2f}  {r['ttc_mean']:6.3f}")
    print(f"{'='*90}\n")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision-cause ablation study")
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT, help="Trials per condition")
    parser.add_argument("--t_sim", type=float, default=T_SIM_DEFAULT, help="Simulation duration (s)")
    args = parser.parse_args()
    main(n_trials=args.trials, t_sim=args.t_sim)
