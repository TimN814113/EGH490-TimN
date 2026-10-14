"""
run_claude_scenario_elicitation.py
EGH490-1 -- Tim Nguyen N11505222

Power-up rerun of the Claude quiet-vs-rush-hour scenario-conditioning check
(egh490_power_up_todo Priority 1 item 4). The original n=4/scenario dataset
was manually pasted from separate fresh Claude conversations (never run
programmatically). Now that ANTHROPIC_API_KEY works, this calls
elicit_idm_params_claude() -- the SAME ChatSUMO-structured channel the
original n=4 draws used (system-prompt-forced JSON), per
results/claude_scenario_elicitations.json's documented provenance -- live,
16 fresh times per scenario, and pools with the existing 4 for a combined
n=20/scenario. Then re-runs the Welch's t-test + BH-FDR significance check
(reusing welch_test/bh_fdr from analyze_significance.py) on the pooled data.
"""

import json
import time
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from analyze_significance import welch_test, bh_fdr
from chatsumo.llm_backend import elicit_idm_params_claude

PROMPTS = {
    "quiet": (
        "I need to configure an IDM car-following simulation with four driver "
        "profiles: (1) an autonomous vehicle in ACC mode that keeps very safe "
        "gaps; (2) a cautious HDV; (3) a normal HDV following the speed limit; "
        "(4) an aggressive sporty HDV. The road is a 3-lane motorway with a "
        "speed limit of 110 km/h, and traffic is very light — it is the "
        "middle of the day, roads are nearly empty, and everyone can cruise "
        "comfortably at their preferred speed with no congestion. Give me the "
        "IDM parameters v0, T, a_max, b_comf, s0, and delta for each profile."
    ),
    "rush_hour": (
        "I need to configure an IDM car-following simulation with four driver "
        "profiles: (1) an autonomous vehicle in ACC mode that keeps very safe "
        "gaps; (2) a cautious HDV; (3) a normal HDV following the speed limit; "
        "(4) an aggressive sporty HDV. The road is a 3-lane motorway with a "
        "speed limit of 110 km/h, and it is rush hour — traffic is heavy "
        "and congested, everyone is running late for work, drivers are "
        "impatient and aggressive, tailgating and jockeying for position. "
        "Give me the IDM parameters v0, T, a_max, b_comf, s0, and delta for "
        "each profile."
    ),
}

PROFILES = ["ACC", "CAUTIOUS", "NORMAL", "SPORTY"]
PARAMS = ["v0_frac", "T", "a_max", "b_comf", "s0", "delta", "threshold"]
N_NEW_DRAWS = 16


def elicit_fresh(scenario, prompt, n_draws):
    draws = []
    for i in range(n_draws):
        for attempt in range(3):
            try:
                result = elicit_idm_params_claude(prompt)
                draws.append(result["profiles"])
                print(f"[{scenario}] draw {i + 1}/{n_draws} ok")
                break
            except Exception as e:
                print(f"[{scenario}] draw {i + 1}/{n_draws} attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        else:
            print(f"[{scenario}] draw {i + 1}/{n_draws} FAILED after 3 attempts, skipping")
        time.sleep(0.5)
    return draws


def main():
    with open("results/claude_scenario_elicitations.json") as f:
        existing = json.load(f)

    combined = {}
    fresh_only = {}
    for scenario, prompt in PROMPTS.items():
        new_draws = elicit_fresh(scenario, prompt, N_NEW_DRAWS)
        fresh_only[scenario] = new_draws
        old_draws = existing[scenario]
        combined[scenario] = old_draws + new_draws
        print(f"[{scenario}] combined n={len(combined[scenario])} "
              f"({len(old_draws)} original + {len(new_draws)} fresh)")

    combined["_provenance"] = (
        "Power-up rerun 2026-09-04: original 4 pasted draws/scenario (see "
        "claude_scenario_elicitations.json) pooled with 16 fresh live "
        "elicit_idm_params_claude() calls/scenario (same ChatSUMO-structured "
        "channel), via run_claude_scenario_elicitation.py."
    )
    with open("results/claude_scenario_elicitations_v2.json", "w") as f:
        json.dump(combined, f, indent=2)
    with open("results/claude_scenario_fresh_draws_only.json", "w") as f:
        json.dump(fresh_only, f, indent=2)

    # Long-form dataframe for testing.
    rows = []
    for scenario in ["quiet", "rush_hour"]:
        for draw in combined[scenario]:
            for profile in PROFILES:
                for param in PARAMS:
                    if profile in draw and param in draw[profile]:
                        rows.append(dict(scenario=scenario, profile=profile,
                                          param=param, value=draw[profile][param]))
    df = pd.DataFrame(rows)

    results = []
    for profile in PROFILES:
        for param in PARAMS:
            sub = df[(df.profile == profile) & (df.param == param)]
            q = sub[sub.scenario == "quiet"].value
            r = sub[sub.scenario == "rush_hour"].value
            if len(q) < 2 or len(r) < 2:
                continue
            if q.std() == 0 and r.std() == 0 and q.mean() == r.mean():
                results.append(dict(profile=profile, param=param,
                                     mean_quiet=q.mean(), mean_rush=r.mean(),
                                     n_quiet=len(q), n_rush=len(r),
                                     p_value=np.nan, note="invariant, excluded"))
                continue
            test = welch_test(q.mean(), q.std(ddof=1), len(q),
                               r.mean(), r.std(ddof=1), len(r))
            results.append(dict(profile=profile, param=param,
                                 mean_quiet=q.mean(), mean_rush=r.mean(),
                                 n_quiet=len(q), n_rush=len(r),
                                 diff=test["diff"], p_value=test["p_value"],
                                 cohens_d=test["cohens_d"]))

    res_df = pd.DataFrame(results)
    testable = res_df["p_value"].notna()
    res_df.loc[testable, "q_value"] = bh_fdr(res_df.loc[testable, "p_value"])
    res_df = res_df.sort_values("q_value", na_position="last")
    res_df.to_csv("results/claude_scenario_significance_v2.csv", index=False)

    print("\n=== Significant at q<0.05 (pooled n=20/scenario) ===")
    sig = res_df[res_df["q_value"] < 0.05]
    print(sig[["profile", "param", "mean_quiet", "mean_rush", "p_value", "q_value"]]
          .to_string(index=False))
    print(f"\nFull results: results/claude_scenario_significance_v2.csv")
    print(f"Combined transcripts: results/claude_scenario_elicitations_v2.json")


if __name__ == "__main__":
    main()
