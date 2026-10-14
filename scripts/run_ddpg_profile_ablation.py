"""
run_ddpg_profile_ablation.py
EGH490-1 — Tim Nguyen N11505222

Isolates why LLM-Generated (the report's real Table 2 config) is significantly
MORE dangerous than Expert under DDPG-Approx (TTC +106% to +221% worse, p<0.005
at all 3 AV%; results/significance_report.csv, 2026-09-03) while
CLAUDE_CASUAL_RUN2 — an independent re-elicitation of the EXACT SAME casual
prompt — is instead SAFER than Expert under DDPG (-82% to -88% TTC, no
significant collision increase). Since both came from the identical prompt,
the divergence must trace to specific parameter values that happened to differ
between the two elicitations.

One-shot design (originally planned as two phases — profile-level isolation,
then parameter-level refinement — but folded into a single run since DDPG-only
diagnostics are cheap, ~2.4s/trial):
  - LLM_FIX_ACC / _CAUTIOUS / _NORMAL / _SPORTY: LLM-Generated with exactly ONE
    profile swapped to CLAUDE_CASUAL_RUN2's version of that profile, everything
    else left as LLM. Whichever swap moves the metrics from "LLM-like" (worse
    than Expert) toward "Run2-like" (safer than Expert) identifies the
    responsible profile.
  - LLM_FIX_SPORTY_AMAX / _BCOMF: if SPORTY is implicated, these narrow it
    further — SPORTY.a_max (5.5 vs 2.0, a 2.75x gap, by far the largest
    single-parameter divergence anywhere between the two elicitations) and
    SPORTY.b_comf (4.0 vs 3.0) each changed alone. See sim/configs.py for the
    physics rationale (ddpg_approx_accel uses a_max in three places via a
    threshold-branching policy, not just as a smooth cap like IDM).

Usage:
    python scripts/run_ddpg_profile_ablation.py
    python scripts/run_ddpg_profile_ablation.py --trials 30 --t_sim 300
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for sim/chatsumo imports

from sim.configs import (
    EXPERT, LLM, CLAUDE_CASUAL_RUN2,
    LLM_FIX_ACC, LLM_FIX_CAUTIOUS, LLM_FIX_NORMAL, LLM_FIX_SPORTY,
    LLM_FIX_SPORTY_AMAX, LLM_FIX_SPORTY_BCOMF,
)
from sim.runner import run_simulation

N_TRIALS_DEFAULT = 20
T_SIM_DEFAULT = 300.0


def main(n_trials=N_TRIALS_DEFAULT, t_sim=T_SIM_DEFAULT):
    configs = [
        EXPERT, LLM, CLAUDE_CASUAL_RUN2,
        LLM_FIX_ACC, LLM_FIX_CAUTIOUS, LLM_FIX_NORMAL, LLM_FIX_SPORTY,
        LLM_FIX_SPORTY_AMAX, LLM_FIX_SPORTY_BCOMF,
    ]
    av_pcts = [0.0, 0.25, 0.50]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("results") / f"ddpg_profile_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    total = len(configs) * len(av_pcts) * n_trials
    print(f"\n{'='*80}")
    print("  DDPG Profile Ablation — EGH490-1 Tim Nguyen")
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

    summary_path = out_dir / "ddpg_profile_ablation_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*95}")
    print(f"  {'Config':32s}  AV%   Coll.mean  Coll.max  NonzeroTrials  Jerk     TTC%")
    print(f"{'-'*95}")
    for r in all_summaries:
        print(f"  {r['config']:32s}  {r['av_pct']:3d}%  {r['collision_mean']:9.3f}  "
              f"{r['collision_max']:8.2f}  {r['collision_nonzero_trials']:13d}  "
              f"{r['jerk_mean']:7.2f}  {r['ttc_mean']:6.3f}")
    print(f"{'='*95}\n")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG profile/parameter ablation: LLM-Generated vs Claude-Casual-Run2")
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT, help="Trials per condition")
    parser.add_argument("--t_sim", type=float, default=T_SIM_DEFAULT, help="Simulation duration (s)")
    args = parser.parse_args()
    main(n_trials=args.trials, t_sim=args.t_sim)
