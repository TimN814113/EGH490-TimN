"""
run_report_replication.py
EGH490-1 — Tim Nguyen N11505222

Reproduces the experimental design from the Week 12/13 progress report
(EGH490-1_ProgressReport_TimNguyen_N11505222) using the Python simulator, for
direct comparison against the MATLAB-derived results in that report
(Table 3, Figures 3-8). See report Sections 2.3/2.4 for the original protocol.

NOTE ON WHAT THIS COMPARISON ACTUALLY SHOWS: the report's numbers came from
GenerateurTraficV47_FollowNet.m in MATLAB. This script runs the independent
Python reimplementation of the same equations. So a match between the two is
evidence the six Findings are a real consequence of the Expert vs LLM
parameter differences (cross-implementation validation) — not a "before vs
after" comparison of the same tool. Exact point values will differ (different
RNG streams per run); compare distributions/direction/magnitude, not single
numbers.

Five experiment groups:
  1. Baseline grid   — N configs x 3 CF models x 3 AV% (0/25/50), N=25, 110 km/h
                        (report Figures 3, 4, 5; Table 3)
  2. Density sweep   — IDM only, N in {10, 25, 40}, AV%/speed held at baseline
                        (report Figure 6). The original report did not test LLM
                        at N=10 — this replication fills that cell.
  3. Speed sweep     — IDM only, v_limit in {80, 110, 129} km/h, N/AV% held at
                        baseline (report Figure 7)
  4. density_x_av    — IDM only, every (N, AV%) combination not already covered
                        by groups 1-2
  5. speed_x_av      — IDM only, every (v_limit, AV%) combination not already
                        covered by groups 1-3

Groups 1-3 are the report's original one-factor-at-a-time (OFAT) design: each
sweep holds every other factor at a "baseline" point (N=25, 110 km/h, 25% AV —
the report's tool defaults / sweep midpoints, not independently justified
beyond that) while varying one factor. OFAT can miss interaction effects — an
LLM-vs-Expert gap that only appears when two factors shift together (e.g.
low AV% AND high density at once). Groups 4-5 cover those combinations. Skip
them with --skip-interaction to reproduce the report's exact OFAT design only.

--n-veh, --speeds, --av-pcts let you replace the swept values entirely (not
just N=25/110km/h/25% AV) — e.g. --n-veh 5,15,25,35,50 for a finer density
sweep, or --av-pcts 0,10,20,30,40,50 for a finer AV-penetration sweep.

--variability overrides IDM personality-noise for EVERY config in the run
(including EXPERT) to the same value — e.g. to test whether giving a casual/
technical LLM config (normally 0% noise, since neither prompt requested any)
the same ±5% noise as EXPERT changes the comparison, or to match MATLAB's
single global-slider default (1%) applied uniformly to whichever preset is
active. Omit it to keep each config's own defined variability (current
default behaviour, unchanged).

With --include-ollama, a third config is added: a config live-elicited from a
locally-running Ollama model, using the exact prompt documented in the
report's Appendix B (originally sent to Claude for the LLM config). Ollama is
called ONCE at the start of the run (not once per trial) — the resulting
preset is then held fixed across all trials/conditions, exactly like EXPERT
and LLM already are. Adding it makes every group ~50% larger (3 configs
instead of 2), so budget time accordingly.

Usage (from the repository root):
    python scripts/run_report_replication.py
    python scripts/run_report_replication.py --trials 8 --t_sim 300
    python scripts/run_report_replication.py --trials 8 --t_sim 300 --include-ollama
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for sim/chatsumo imports

from sim.configs import EXPERT
# from sim.configs import LLM  # Claude-sonnet-4-6 config — commented out, testing Ollama only for now
from sim.runner import run_simulation


N_TRIALS_DEFAULT = 8      # report's clean baseline used 5-7 runs per condition
T_SIM_DEFAULT    = 300.0  # report's own Figure 1 runs extend to ~250-450s

# Original casual prompt, verbatim from the report's Appendix B (sent to Claude).
# Kept for reference / reproducibility but NOT invoked below — get_ollama_config()
# now uses TECHNICAL_ELICITATION_PROMPT instead. Swap which constant is passed to
# elicit_idm_params() in get_ollama_config() to switch back.
REPORT_ELICITATION_PROMPT = (
    "I need to configure an IDM car-following simulation with four driver profiles: "
    "(1) an autonomous vehicle in ACC mode that keeps very safe gaps; (2) a cautious HDV; "
    "(3) a normal HDV following the speed limit; (4) an aggressive sporty HDV. The road is "
    "a 3-lane motorway with a speed limit of 110 km/h. Give me the IDM parameters v0, T, "
    "a_max, b_comf, s0, and delta for each profile."
)

# Technical variant: explicit units, literature-typical ranges (matching
# chatsumo/llm_backend.py's SYSTEM_PROMPT), and explicit cross-profile ordering
# constraints — directly targeting the failure modes seen in the casual-prompt
# Ollama run (ACC/NORMAL/SPORTY v0 below the speed limit, CAUTIOUS with a
# shorter headway than baseline NORMAL). Tests whether Ollama's errors are a
# capability gap or a prompt-specification gap.
TECHNICAL_ELICITATION_PROMPT = (
    "Configure the Intelligent Driver Model (IDM, Treiber et al. 2000) for a "
    "microscopic traffic simulation on a 3-lane motorway with a posted speed limit "
    "of 110 km/h. Provide numeric parameter values for four distinct driver/vehicle "
    "profiles: ACC (an autonomous vehicle in adaptive cruise control mode), CAUTIOUS "
    "(a conservative human driver), NORMAL (a typical human driver complying with "
    "the posted speed limit), and SPORTY (an aggressive human driver exceeding the "
    "posted speed limit).\n\n"
    "For each profile, specify these six IDM parameters, each within the stated "
    "literature-typical range:\n"
    "  - v0: desired speed, as a fraction of the 110 km/h speed limit. NORMAL and "
    "ACC must be >= 1.00 (at or above the limit); SPORTY must be > 1.00 (faster "
    "than the limit); CAUTIOUS must be the ONLY profile below 1.00.\n"
    "  - T: desired time headway to the leading vehicle, in seconds. Range 0.6-2.5 s. "
    "SPORTY must have the SHORTEST T of the four profiles; CAUTIOUS must have the "
    "LONGEST.\n"
    "  - a_max: maximum acceleration, in m/s^2. Range 1.0-5.0 m/s^2. SPORTY must have "
    "the HIGHEST a_max; CAUTIOUS must have the LOWEST.\n"
    "  - b_comf: comfortable (non-emergency) deceleration, in m/s^2. Range 0.8-4.0 "
    "m/s^2. SPORTY must have the HIGHEST b_comf; CAUTIOUS must have the LOWEST.\n"
    "  - s0: minimum standstill gap to the leading vehicle, in metres. Range 1.5-6.0 m. "
    "SPORTY must have the SMALLEST s0; CAUTIOUS must have the LARGEST.\n"
    "  - delta: IDM acceleration exponent (dimensionless). Range 2-6. Higher values "
    "produce more aggressive, near-constant acceleration until close to v0.\n\n"
    "Every profile's parameters must be internally consistent with its driving style, "
    "and all four profiles must be clearly differentiated from each other on every "
    "parameter — no two profiles may share an identical value for the same parameter. "
    "Return numeric values only, within the stated ranges, for all six parameters for "
    "all four profiles."
)


def get_ollama_config(out_dir: Path) -> dict:
    """
    Live-elicit an IDM config from a locally-running Ollama model, using
    TECHNICAL_ELICITATION_PROMPT (explicit units, ranges, and cross-profile
    ordering constraints — see comment above). Called once, at the start of the
    run — the resulting preset is then held fixed across every trial/condition,
    exactly like EXPERT.

    Saves the raw response + resulting preset to JSON in out_dir, mirroring the
    verbatim-transcript record your report already keeps for the Claude prompt.
    """
    from chatsumo.llm_backend import elicit_idm_params, build_preset_from_llm

    prompt = TECHNICAL_ELICITATION_PROMPT

    print(f"\n{'='*80}")
    print("  Eliciting IDM config from Ollama (first call can be slow — model load)...")
    print(f"{'='*80}")

    response = elicit_idm_params(prompt, backend="ollama")
    preset = build_preset_from_llm(response)
    preset["name"] = "Ollama-Generated (llama3.1, technical prompt)"

    print(f"  Source: {response['source']}")
    for mode, p in preset["profiles"].items():
        print(f"    {mode:10s}: {p}")

    transcript_path = out_dir / "ollama_elicitation_transcript.json"
    with open(transcript_path, "w") as f:
        json.dump({"prompt": prompt, "response": response, "preset": preset}, f, indent=2)
    print(f"  Transcript saved: {transcript_path}\n")

    return preset


def run_condition(group, condition_label, config, cf_model, av_pct,
                  n_trials, sim_cfg, all_summaries, all_timeseries):
    trial_summaries = []
    for trial in range(1, n_trials + 1):
        t0 = time.time()
        print(f"  [{group:8s}] {condition_label:38s} | {config['name']:12.12s} | "
              f"{cf_model:12s} | trial {trial}/{n_trials} ...", end=" ", flush=True)

        summary, ts = run_simulation(
            config=config, cf_model=cf_model, av_pct=av_pct, sim_cfg=sim_cfg,
        )
        elapsed = time.time() - t0
        print(f"done ({elapsed:.2f}s)  MSE={summary['spacing_mse_m2']:.1f}  "
              f"Jerk={summary['jerk_mean_ms3']:.4f}  TTC={summary['ttc_exposure_pct']:.3f}%")

        summary["trial"] = trial
        trial_summaries.append(summary)
        all_timeseries.extend(ts)

    agg = {
        "group":            group,
        "condition":        condition_label,
        "config":           config["name"],
        "cf_model":         cf_model,
        "av_pct":           int(av_pct * 100),
        "n_veh":            sim_cfg.get("n_veh", 25),
        "road_len":         sim_cfg.get("road_len", 800),
        "v_limit_kmh":      sim_cfg.get("v_limit_kmh", 110),
        "t_sim":            sim_cfg.get("t_sim", T_SIM_DEFAULT),
        "variability":      config.get("variability", 0.0),
        "n_trials":         n_trials,
        "spacing_mse_mean": round(np.mean([s["spacing_mse_m2"] for s in trial_summaries]), 4),
        "spacing_mse_std":  round(np.std([s["spacing_mse_m2"] for s in trial_summaries]), 4),
        "collision_mean":   round(np.mean([s["collision_count"] for s in trial_summaries]), 3),
        "jerk_mean":        round(np.mean([s["jerk_mean_ms3"] for s in trial_summaries]), 6),
        "jerk_std":         round(np.std([s["jerk_mean_ms3"] for s in trial_summaries]), 6),
        "ttc_mean":         round(np.mean([s["ttc_exposure_pct"] for s in trial_summaries]), 4),
        "ttc_std":          round(np.std([s["ttc_exposure_pct"] for s in trial_summaries]), 4),
    }
    all_summaries.append(agg)


def with_variability(config: dict, variability: float) -> dict:
    """
    Shallow copy of `config` with variability overridden. Used to test IDM
    personality noise as its own experimental factor — e.g. giving a
    casual/technical-prompt config (normally 0% variability, since neither
    prompt asked for noise) the same ±5% per-vehicle noise as EXPERT, or
    matching MATLAB's single global default of 1% across every config.
    Never mutates the original preset dict (profiles dict is reused, only
    the top-level "variability" key differs).
    """
    return {**config, "variability": variability}


def main(n_trials=N_TRIALS_DEFAULT, t_sim=T_SIM_DEFAULT, include_ollama=False, claude_variant=None,
        ollama_variant=None, chatsumo_claude_variant=None,
        n_veh_values=(10, 25, 40), speed_values=(80.0, 110.0, 129.0), av_values=(0.0, 0.25, 0.50),
        variability=None, skip_interaction=False):
    configs = [EXPERT]
    # configs = [EXPERT, LLM]  # Claude-sonnet-4-6 config — commented out, testing Ollama only for now
    cf_models_full = ["IDM", "LSTM-Approx", "DDPG-Approx"]
    av_pcts = list(av_values)
    n_veh_values = list(n_veh_values)
    speed_values = list(speed_values)
    # "Baseline" point held fixed while sweeping one other factor — the middle
    # of each range by convention (matches the report), not independently justified.
    baseline_n_veh = 25 if 25 in n_veh_values else n_veh_values[len(n_veh_values) // 2]
    baseline_speed = 110.0 if 110.0 in speed_values else speed_values[len(speed_values) // 2]
    baseline_av = 0.25 if 0.25 in av_pcts else av_pcts[len(av_pcts) // 2]

    # Second-resolution timestamps collide when multiple instances are launched
    # concurrently (observed 2026-09-04: 8 parallel processes silently clobbered
    # each other's output folders, losing 2+ full runs with no error raised —
    # mkdir(exist_ok=True) doesn't error on collision, and the last writer wins).
    # microseconds + PID make collision effectively impossible even at high concurrency.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    out_dir = Path("results") / f"report_replication_{timestamp}_pid{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if include_ollama:
        configs.append(get_ollama_config(out_dir))

    if ollama_variant == "casual":
        from sim.configs import OLLAMA_CASUAL
        configs.append(OLLAMA_CASUAL)
    elif ollama_variant == "casual2":
        from sim.configs import OLLAMA_CASUAL_RUN2
        configs.append(OLLAMA_CASUAL_RUN2)
    elif ollama_variant == "technical":
        from sim.configs import OLLAMA_TECHNICAL
        configs.append(OLLAMA_TECHNICAL)
    elif ollama_variant == "technical2":
        from sim.configs import OLLAMA_TECHNICAL_RUN2
        configs.append(OLLAMA_TECHNICAL_RUN2)
    elif ollama_variant == "casual-rush1":
        from sim.configs import OLLAMA_CASUAL_RUSH_1
        configs.append(OLLAMA_CASUAL_RUSH_1)
    elif ollama_variant == "casual-rush2":
        from sim.configs import OLLAMA_CASUAL_RUSH_2
        configs.append(OLLAMA_CASUAL_RUSH_2)
    elif ollama_variant == "technical-rush1":
        from sim.configs import OLLAMA_TECHNICAL_RUSH_1
        configs.append(OLLAMA_TECHNICAL_RUSH_1)
    elif ollama_variant == "technical-rush2":
        from sim.configs import OLLAMA_TECHNICAL_RUSH_2
        configs.append(OLLAMA_TECHNICAL_RUSH_2)
    elif ollama_variant == "quiet":
        from sim.configs import OLLAMA_QUIET
        configs.append(OLLAMA_QUIET)

    if claude_variant == "casual":
        # LLM (imported at module level, currently commented out above) IS the
        # casual-prompt Claude config — re-running the report's exact Appendix B
        # prompt against real claude-sonnet-4-6 reproduced it value-for-value.
        from sim.configs import LLM as CLAUDE_CASUAL_CONFIG
        configs.append(CLAUDE_CASUAL_CONFIG)
    elif claude_variant == "technical":
        from sim.configs import CLAUDE_TECHNICAL
        configs.append(CLAUDE_TECHNICAL)
    elif claude_variant == "casual2":
        from sim.configs import CLAUDE_CASUAL_RUN2
        configs.append(CLAUDE_CASUAL_RUN2)
    elif claude_variant == "technical2":
        from sim.configs import CLAUDE_TECHNICAL_RUN2
        configs.append(CLAUDE_TECHNICAL_RUN2)
    elif claude_variant == "technical3":
        from sim.configs import CLAUDE_TECHNICAL_RUN3
        configs.append(CLAUDE_TECHNICAL_RUN3)
    elif claude_variant == "casual-rush1":
        from sim.configs import CLAUDE_PLAIN_CASUAL_RUSH_1
        configs.append(CLAUDE_PLAIN_CASUAL_RUSH_1)
    elif claude_variant == "casual-rush2":
        from sim.configs import CLAUDE_PLAIN_CASUAL_RUSH_2
        configs.append(CLAUDE_PLAIN_CASUAL_RUSH_2)
    elif claude_variant == "technical-rush1":
        from sim.configs import CLAUDE_PLAIN_TECHNICAL_RUSH_1
        configs.append(CLAUDE_PLAIN_TECHNICAL_RUSH_1)
    elif claude_variant == "technical-rush2":
        from sim.configs import CLAUDE_PLAIN_TECHNICAL_RUSH_2
        configs.append(CLAUDE_PLAIN_TECHNICAL_RUSH_2)

    if chatsumo_claude_variant == "casual":
        from sim.configs import CHATSUMO_CLAUDE_CASUAL
        configs.append(CHATSUMO_CLAUDE_CASUAL)
    elif chatsumo_claude_variant == "casual2":
        from sim.configs import CHATSUMO_CLAUDE_CASUAL_RUN2
        configs.append(CHATSUMO_CLAUDE_CASUAL_RUN2)
    elif chatsumo_claude_variant == "technical":
        from sim.configs import CHATSUMO_CLAUDE_TECHNICAL
        configs.append(CHATSUMO_CLAUDE_TECHNICAL)
    elif chatsumo_claude_variant == "technical2":
        from sim.configs import CHATSUMO_CLAUDE_TECHNICAL_RUN2
        configs.append(CHATSUMO_CLAUDE_TECHNICAL_RUN2)
    elif chatsumo_claude_variant == "casual-rush1":
        from sim.configs import CHATSUMO_CLAUDE_CASUAL_RUSH_1
        configs.append(CHATSUMO_CLAUDE_CASUAL_RUSH_1)
    elif chatsumo_claude_variant == "casual-rush2":
        from sim.configs import CHATSUMO_CLAUDE_CASUAL_RUSH_2
        configs.append(CHATSUMO_CLAUDE_CASUAL_RUSH_2)
    elif chatsumo_claude_variant == "technical-rush1":
        from sim.configs import CHATSUMO_CLAUDE_TECHNICAL_RUSH_1
        configs.append(CHATSUMO_CLAUDE_TECHNICAL_RUSH_1)
    elif chatsumo_claude_variant == "technical-rush2":
        from sim.configs import CHATSUMO_CLAUDE_TECHNICAL_RUSH_2
        configs.append(CHATSUMO_CLAUDE_TECHNICAL_RUSH_2)
    elif chatsumo_claude_variant == "quiet":
        from sim.configs import CHATSUMO_CLAUDE_QUIET
        configs.append(CHATSUMO_CLAUDE_QUIET)

    if variability is not None:
        configs = [with_variability(c, variability) for c in configs]

    all_summaries, all_timeseries = [], []

    # -2 because the (baseline_n_veh, baseline_av) and (baseline_speed, baseline_av)
    # combos are skipped in the interaction loops below (already run in groups 2-3).
    n_interaction = 0 if skip_interaction else (len(n_veh_values) * len(av_pcts) + len(speed_values) * len(av_pcts) - 2)
    total = (len(configs) * len(av_pcts) * len(cf_models_full)   # baseline grid
             + len(configs) * len(n_veh_values)                  # density sweep
             + len(configs) * len(speed_values)                  # speed sweep
             + len(configs) * n_interaction) * n_trials           # interaction sweeps

    print(f"\n{'='*80}")
    print("  Report Replication — EGH490-1 Tim Nguyen")
    print(f"  {total} total runs ({n_trials} trials/condition, t_sim={t_sim}s)")
    if variability is not None:
        print(f"  Variability overridden to {variability*100:.1f}% for ALL configs")
    print(f"{'='*80}\n")

    # --- Group 1: baseline grid (report Figures 3, 4, 5; Table 3) ---
    for config in configs:
        for av_pct in av_pcts:
            for cf_model in cf_models_full:
                sim_cfg = {"t_sim": t_sim, "n_veh": baseline_n_veh, "road_len": 800.0, "v_limit_kmh": baseline_speed}
                run_condition("baseline", f"N={baseline_n_veh},{int(baseline_speed)}kmh,AV={int(av_pct*100)}%",
                              config, cf_model, av_pct, n_trials, sim_cfg,
                              all_summaries, all_timeseries)

    # --- Group 2: density sweep, IDM only, AV held at baseline (report Figure 6) ---
    for config in configs:
        for n_veh in n_veh_values:
            sim_cfg = {"t_sim": t_sim, "n_veh": n_veh, "road_len": 800.0, "v_limit_kmh": baseline_speed}
            run_condition("density", f"N={n_veh},{int(baseline_speed)}kmh,AV={int(baseline_av*100)}%",
                          config, "IDM", baseline_av, n_trials, sim_cfg,
                          all_summaries, all_timeseries)

    # --- Group 3: speed-limit sweep, IDM only, AV held at baseline (report Figure 7) ---
    for config in configs:
        for v_limit in speed_values:
            sim_cfg = {"t_sim": t_sim, "n_veh": baseline_n_veh, "road_len": 800.0, "v_limit_kmh": v_limit}
            run_condition("speed", f"N={baseline_n_veh},{int(v_limit)}kmh,AV={int(baseline_av*100)}%",
                          config, "IDM", baseline_av, n_trials, sim_cfg,
                          all_summaries, all_timeseries)

    # --- Groups 4-5: interaction sweeps, IDM only — density x AV%, speed x AV%.
    # A one-factor-at-a-time design (groups 1-3, holding AV at baseline_av while
    # sweeping N or speed) can miss cases where the LLM-vs-Expert gap only shows
    # up when TWO factors shift together — e.g. an error that's invisible at
    # 25% AV but large at 0% AV under high density. These groups sweep N and
    # speed at every AV% level instead of just the baseline one.
    if not skip_interaction:
        for config in configs:
            for av_pct in av_pcts:
                for n_veh in n_veh_values:
                    if n_veh == baseline_n_veh and av_pct == baseline_av:
                        continue  # already covered by the density sweep above
                    sim_cfg = {"t_sim": t_sim, "n_veh": n_veh, "road_len": 800.0, "v_limit_kmh": baseline_speed}
                    run_condition("density_x_av", f"N={n_veh},{int(baseline_speed)}kmh,AV={int(av_pct*100)}%",
                                  config, "IDM", av_pct, n_trials, sim_cfg,
                                  all_summaries, all_timeseries)

        for config in configs:
            for av_pct in av_pcts:
                for v_limit in speed_values:
                    if v_limit == baseline_speed and av_pct == baseline_av:
                        continue  # already covered by the speed sweep above
                    sim_cfg = {"t_sim": t_sim, "n_veh": baseline_n_veh, "road_len": 800.0, "v_limit_kmh": v_limit}
                    run_condition("speed_x_av", f"N={baseline_n_veh},{int(v_limit)}kmh,AV={int(av_pct*100)}%",
                                  config, "IDM", av_pct, n_trials, sim_cfg,
                                  all_summaries, all_timeseries)

    # --- Write CSVs ---
    summary_path = out_dir / "report_replication_summary.csv"
    ts_path = out_dir / "report_replication_timeseries.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\nSummary saved: {summary_path}")

    with open(ts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_timeseries[0].keys())
        writer.writeheader()
        writer.writerows(all_timeseries)
    print(f"Timeseries saved: {ts_path}")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report replication experiment (Python vs MATLAB report)")
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT, help="Trials per condition")
    parser.add_argument("--t_sim", type=float, default=T_SIM_DEFAULT, help="Simulation duration (s)")
    parser.add_argument("--include-ollama", action="store_true",
                        help="Add a config FRESHLY live-elicited from a locally-running Ollama "
                             "model right now (adds ~50%% more conditions; requires Ollama running "
                             "and introduces new run-to-run randomness). Prefer --ollama-config "
                             "for a reproducible, parameter-fixed comparison.")
    parser.add_argument("--ollama-config",
                        choices=["casual", "casual2", "technical", "technical2",
                                 "casual-rush1", "casual-rush2", "technical-rush1", "technical-rush2",
                                 "quiet"],
                        default=None,
                        help="Add a hardcoded Ollama/llama3.1 config (adds ~50%% more conditions), "
                             "fixed from a real elicitation already on record (OLLAMA_CASUAL / "
                             "OLLAMA_TECHNICAL in sim/configs.py) — no live Ollama call, no new "
                             "randomness. Use this instead of --include-ollama for anything you "
                             "want to compare against another run (e.g. with vs without noise).")
    parser.add_argument("--claude-config",
                        choices=["casual", "technical", "casual2", "technical2", "technical3",
                                 "casual-rush1", "casual-rush2", "technical-rush1", "technical-rush2"],
                        default=None,
                        help="Add a real claude-sonnet-4-6 'plain conversational' config (no "
                             "system prompt, no live API call — hardcoded from a real elicitation "
                             "in sim/configs.py). 'casual'/'technical' = first run (LLM / "
                             "CLAUDE_TECHNICAL, the latter CONTAMINATED). 'casual2'/'technical2' = "
                             "second draw. 'technical3' = third draw, via live API "
                             "(elicit_idm_params_claude_plain_full), pairs with technical2 since "
                             "technical (run1) is excluded.")
    parser.add_argument("--chatsumo-claude-config",
                        choices=["casual", "casual2", "technical", "technical2",
                                 "casual-rush1", "casual-rush2", "technical-rush1", "technical-rush2",
                                 "quiet"],
                        default=None,
                        help="Add a Claude config elicited through chatsumo/llm_backend.py's "
                             "structured JSON pipeline (SYSTEM_PROMPT) — the SAME pipeline used "
                             "for every Ollama config, run against Claude instead. A DIFFERENT "
                             "elicitation channel from --claude-config (which is a plain "
                             "conversational ask, no system prompt). Hardcoded from real "
                             "elicitations in sim/configs.py (CHATSUMO_CLAUDE_*).")
    parser.add_argument("--variability", type=float, default=None,
                        help="Override IDM personality-noise variability [0-0.20] for EVERY "
                             "config in this run (including EXPERT). Default: each config keeps "
                             "its own defined variability (Expert=5%%, casual/technical LLM "
                             "configs=0%% since neither prompt asked for noise). E.g. --variability "
                             "0.05 tests whether giving an LLM config the same noise as Expert "
                             "changes the comparison; --variability 0.01 matches MATLAB's single "
                             "global-slider default applied to whichever preset is active.")
    parser.add_argument("--n-veh", type=str, default="10,25,40",
                        help="Comma-separated fleet sizes for the density sweep + interaction "
                             "sweeps. Default matches the report's Figure 6 (10,25,40). The "
                             "middle value (or 25 if present) is used as the fixed N for the "
                             "baseline grid and speed sweep.")
    parser.add_argument("--speeds", type=str, default="80,110,129",
                        help="Comma-separated speed limits [km/h] for the speed sweep + "
                             "interaction sweeps. Default matches the report's Figure 7 "
                             "(80,110,129). The middle value (or 110 if present) is used as the "
                             "fixed speed for the baseline grid and density sweep.")
    parser.add_argument("--av-pcts", type=str, default="0,25,50",
                        help="Comma-separated AV penetration rates [%%] for the baseline grid + "
                             "interaction sweeps. Default matches the report (0,25,50). The "
                             "middle value (or 25 if present) is used as the fixed AV%% for the "
                             "density/speed sweeps.")
    parser.add_argument("--skip-interaction", action="store_true",
                        help="Skip the density x AV%% and speed x AV%% interaction sweeps "
                             "(groups 4-5) and only run the original one-factor-at-a-time design "
                             "(baseline grid + density sweep + speed sweep), matching the report's "
                             "original methodology exactly. Roughly halves the run count.")
    args = parser.parse_args()
    main(n_trials=args.trials, t_sim=args.t_sim, include_ollama=args.include_ollama,
         claude_variant=args.claude_config, ollama_variant=args.ollama_config,
         chatsumo_claude_variant=args.chatsumo_claude_config,
         n_veh_values=[int(x) for x in args.n_veh.split(",")],
         speed_values=[float(x) for x in args.speeds.split(",")],
         av_values=[int(x) / 100 for x in args.av_pcts.split(",")],
         variability=args.variability,
         skip_interaction=args.skip_interaction)
