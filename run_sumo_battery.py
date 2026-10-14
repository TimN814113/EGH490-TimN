"""
run_sumo_battery.py
EGH490-1 — Tim Nguyen N11505222

Runs the real SUMO engine (via chatsumo.sumo_runner.run_sumo_full_metrics)
across a focused set of configs, computing all 4 FollowNet metrics through
actual SUMO/TraCI rather than the Python approximation. Answers two open
items from the study:
  1. Does SUMO's real IDM implementation confirm the SPORTY.a_max finding
     (LLM-Generated dangerous, Claude-Casual-Run2 safe) that the Python
     engine and its DDPG-Approx approximation showed? IDM is the only
     car-following model SUMO can run here.
  2. How does this project's own hand-calibrated EXPERT compare against a
     genuinely independent second reference (SUMO's own built-in IDM
     defaults, SUMO_DEFAULT_IDM in sim/configs.py), with zero LLM involvement
     in either config?

Runs SEQUENTIALLY, not in parallel — traci's global default connection
collides across threads/processes without explicit per-run connection labels
(confirmed 2026-09-04: concurrent runs threw KeyError/peer-shutdown errors).
At ~17s/trial (t_sim=90s, n_veh=25), this is slow but reliable.

KNOWN CAVEAT: SUMO's fleet-average speed decays over a run in a way the
Python engine does not (see run_sumo_full_metrics docstring) — t_sim is kept
moderate (90s) here to limit how much this affects results, but it is not
resolved, and any comparison against Python-engine numbers should account
for it rather than treating the two as perfectly equivalent physics.

Usage:
    python run_sumo_battery.py
    python run_sumo_battery.py --trials 8 --t_sim 90
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from chatsumo.sumo_runner import run_sumo_full_metrics
from sim.configs import EXPERT, LLM, CLAUDE_CASUAL_RUN2, SUMO_DEFAULT_IDM

CONFIGS = [
    ("Expert-Calibrated", EXPERT, False),
    ("LLM-Generated", LLM, False),
    ("Claude-Casual-Run2", CLAUDE_CASUAL_RUN2, False),
    ("SUMO-Default-IDM", SUMO_DEFAULT_IDM, True),
]


def main(n_trials: int, t_sim: float, n_veh: int, av_pct: float):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("results") / f"sumo_battery_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    total = len(CONFIGS) * n_trials
    print(f"\n{'='*80}\n  SUMO Battery — EGH490-1 Tim Nguyen\n"
          f"  {total} total runs ({n_trials} trials/config, t_sim={t_sim}s, "
          f"n_veh={n_veh}, av_pct={av_pct})\n{'='*80}\n")

    for label, preset, use_defaults in CONFIGS:
        trial_summaries = []
        for trial in range(1, n_trials + 1):
            t0 = time.time()
            print(f"  {label:20s} | trial {trial}/{n_trials} ...", end=" ", flush=True)
            summary, _ = run_sumo_full_metrics(
                preset, n_veh=n_veh, av_pct=av_pct, road_len=800.0,
                t_sim=t_sim, v_limit_kmh=110.0, use_sumo_defaults=use_defaults,
            )
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)  MSE={summary['spacing_mse_m2']:.1f}  "
                  f"Jerk={summary['jerk_mean_ms3']:.4f}  TTC={summary['ttc_exposure_pct']:.3f}%  "
                  f"Coll={summary['collision_count']}")
            summary["trial"] = trial
            trial_summaries.append(summary)

        import numpy as np
        agg = {
            "config": label,
            "n_trials": n_trials,
            "t_sim": t_sim,
            "n_veh": n_veh,
            "av_pct": int(av_pct * 100),
            "spacing_mse_mean": round(np.mean([s["spacing_mse_m2"] for s in trial_summaries]), 4),
            "spacing_mse_std": round(np.std([s["spacing_mse_m2"] for s in trial_summaries]), 4),
            "collision_mean": round(np.mean([s["collision_count"] for s in trial_summaries]), 4),
            "jerk_mean": round(np.mean([s["jerk_mean_ms3"] for s in trial_summaries]), 6),
            "jerk_std": round(np.std([s["jerk_mean_ms3"] for s in trial_summaries]), 6),
            "ttc_mean": round(np.mean([s["ttc_exposure_pct"] for s in trial_summaries]), 4),
            "ttc_std": round(np.std([s["ttc_exposure_pct"] for s in trial_summaries]), 4),
        }
        all_summaries.append(agg)

    summary_path = out_dir / "sumo_battery_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*90}\n  {'Config':20s}  MSE      Jerk     TTC%     Collisions\n{'-'*90}")
    for r in all_summaries:
        print(f"  {r['config']:20s}  {r['spacing_mse_mean']:7.1f}  {r['jerk_mean']:6.3f}  "
              f"{r['ttc_mean']:6.3f}  {r['collision_mean']:.3f}")
    print(f"{'='*90}\n")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--t_sim", type=float, default=90.0)
    parser.add_argument("--n-veh", type=int, default=25)
    parser.add_argument("--av-pct", type=float, default=0.25)
    args = parser.parse_args()
    main(args.trials, args.t_sim, args.n_veh, args.av_pct)
