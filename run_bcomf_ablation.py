"""
run_bcomf_ablation.py
EGH490-1 -- Tim Nguyen N11505222

Causal ablation for Ledger Finding 09 (NORMAL.b_comf runs +36.7% high in 23/25
configs vs Expert's 1.7 -- a near-universal elicitation bias, but so far only
demonstrated as a correlation across configs, not an isolated cause-and-effect
test). Reverts ONLY NORMAL.b_comf to Expert's value on the two most-studied
dangerous configs (LLM-Generated, Ollama-Casual -- see sim/configs.py's
LLM_FIX_NORMAL_BCOMF / OLLAMA_CASUAL_FIX_NORMAL_BCOMF), holding every other
parameter fixed, and compares against the unmodified originals under BOTH
DDPG-Approx (where all prior danger differentiation concentrated) and IDM
(to check whether, unlike SPORTY.a_max, this parameter's effect is engine-
specific or not).

Usage:
    python run_bcomf_ablation.py
    python run_bcomf_ablation.py --trials 20 --t_sim 300
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sim.configs import (
    EXPERT, LLM, LLM_FIX_NORMAL_BCOMF,
    OLLAMA_CASUAL, OLLAMA_CASUAL_FIX_NORMAL_BCOMF,
)
from sim.runner import run_simulation

N_TRIALS_DEFAULT = 20
T_SIM_DEFAULT = 300.0


def main(n_trials=N_TRIALS_DEFAULT, t_sim=T_SIM_DEFAULT):
    configs = [EXPERT, LLM, LLM_FIX_NORMAL_BCOMF, OLLAMA_CASUAL, OLLAMA_CASUAL_FIX_NORMAL_BCOMF]
    cf_models = ["DDPG-Approx", "IDM"]
    av_pcts = [0.0, 0.25, 0.50]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("results") / f"bcomf_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    total = len(configs) * len(cf_models) * len(av_pcts) * n_trials
    print(f"\n{'='*80}")
    print("  NORMAL.b_comf Causal Ablation -- EGH490-1 Tim Nguyen")
    print(f"  {total} total runs ({n_trials} trials/condition, t_sim={t_sim}s)")
    print(f"{'='*80}\n")

    for config in configs:
        for cf_model in cf_models:
            for av_pct in av_pcts:
                sim_cfg = {"t_sim": t_sim, "n_veh": 25, "road_len": 800.0, "v_limit_kmh": 110.0}
                trial_summaries = []
                for trial in range(1, n_trials + 1):
                    t0 = time.time()
                    print(f"  {config['name']:38.38s} | {cf_model:11s} | AV={int(av_pct*100):3d}% | "
                          f"trial {trial}/{n_trials} ...", end=" ", flush=True)
                    summary, _ = run_simulation(
                        config=config, cf_model=cf_model, av_pct=av_pct, sim_cfg=sim_cfg,
                    )
                    elapsed = time.time() - t0
                    print(f"done ({elapsed:.2f}s)  collisions={summary['collision_count']:.2f}  "
                          f"jerk={summary['jerk_mean_ms3']:.2f}  ttc={summary['ttc_exposure_pct']:.3f}%")
                    trial_summaries.append(summary)

                all_summaries.append({
                    "config": config["name"],
                    "cf_model": cf_model,
                    "av_pct": int(av_pct * 100),
                    "n_trials": n_trials,
                    "spacing_mse_mean": round(np.mean([s["spacing_mse_m2"] for s in trial_summaries]), 4),
                    "spacing_mse_std": round(np.std([s["spacing_mse_m2"] for s in trial_summaries], ddof=1), 4),
                    "collision_mean": round(np.mean([s["collision_count"] for s in trial_summaries]), 4),
                    "jerk_mean": round(np.mean([s["jerk_mean_ms3"] for s in trial_summaries]), 6),
                    "jerk_std": round(np.std([s["jerk_mean_ms3"] for s in trial_summaries], ddof=1), 6),
                    "ttc_mean": round(np.mean([s["ttc_exposure_pct"] for s in trial_summaries]), 4),
                    "ttc_std": round(np.std([s["ttc_exposure_pct"] for s in trial_summaries], ddof=1), 4),
                })

    summary_path = out_dir / "bcomf_ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*100}")
    print(f"  {'Config':38s}  {'CF':11s} AV%   Coll.mean   Jerk     TTC%")
    print(f"{'-'*100}")
    for r in all_summaries:
        print(f"  {r['config']:38s}  {r['cf_model']:11s} {r['av_pct']:3d}%  {r['collision_mean']:9.3f}  "
              f"{r['jerk_mean']:7.2f}  {r['ttc_mean']:6.3f}")
    print(f"{'='*100}\n")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal ablation: NORMAL.b_comf reverted to Expert's value")
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT, help="Trials per condition")
    parser.add_argument("--t_sim", type=float, default=T_SIM_DEFAULT, help="Simulation duration (s)")
    args = parser.parse_args()
    main(n_trials=args.trials, t_sim=args.t_sim)
