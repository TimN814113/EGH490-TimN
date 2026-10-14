"""
build_signed_deviation_tables.py
EGH490-1 — Tim Nguyen N11505222

Systematic signed-deviation tables: for every config in sim/configs.py (except
EXPERT itself and SUMO_DEFAULT_IDM, which is a parameters-mostly-ignored
placeholder — see its own docstring), per-profile per-parameter signed %
deviation from EXPERT-Calibrated. Generalizes chatsumo/llm_backend.py's
compare_to_expert() (built for one config at a time, used ad hoc in the
ChatSUMO tab) into one comprehensive table covering the whole study — this
was flagged repeatedly through the project as "not yet rebuilt through the
new pipeline" and never actually existed as a single artifact.

Signed deviation = (llm_value - expert_value) / expert_value * 100, so a
positive value means the config's parameter is HIGHER than Expert's, negative
means LOWER — sign is preserved deliberately (unlike an unsigned |deviation|)
so directional patterns (e.g. "casual prompting inflates SPORTY.a_max") are
visible directly in the table rather than needing per-config manual inspection.

Usage:
    python analysis/build_signed_deviation_tables.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for sim/chatsumo imports

from sim.configs import EXPERT, ALL_CONFIGS

PARAMS = ["v0_frac", "T", "a_max", "b_comf", "s0", "delta", "threshold"]
PROFILES = ["ACC", "CAUTIOUS", "NORMAL", "SPORTY"]

# SUMO_DEFAULT_IDM only actually uses v0_frac (see its docstring in
# sim/configs.py) — every other field is a placeholder never sent to SUMO, so
# including it here would misrepresent SUMO's real (TraCI-queried) parameters
# as if they were these hardcoded numbers. Excluded for that reason, not
# oversight.
EXCLUDE = {"Expert-Calibrated", "SUMO-Default-IDM"}


def main():
    rows = []
    for label, cfg in ALL_CONFIGS.items():
        if label in EXCLUDE:
            continue
        for profile in PROFILES:
            if profile not in cfg["profiles"] or profile not in EXPERT["profiles"]:
                continue
            ep = EXPERT["profiles"][profile]
            lp = cfg["profiles"][profile]
            for param in PARAMS:
                if param not in ep or param not in lp:
                    continue
                ev, lv = ep[param], lp[param]
                pct = (lv - ev) / ev * 100 if ev != 0 else float("nan")
                rows.append({
                    "config": label,
                    "profile": profile,
                    "parameter": param,
                    "expert_value": ev,
                    "config_value": lv,
                    "signed_deviation_pct": round(pct, 2) if pct == pct else "",
                })

    out_path = Path("results") / "signed_deviation_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {out_path}")

    # --- Summary: which (profile, parameter) cells show the largest and most
    # consistent (same-sign-across-configs) deviations — the fastest way to
    # spot a structural elicitation pattern vs one-off noise.
    import pandas as pd
    df = pd.DataFrame(rows)
    df["signed_deviation_pct"] = pd.to_numeric(df["signed_deviation_pct"], errors="coerce")

    print(f"\n{'='*90}\n  Largest |mean signed deviation|, averaged across all {df['config'].nunique()} configs\n{'='*90}")
    summary = (df.groupby(["profile", "parameter"])["signed_deviation_pct"]
                 .agg(mean_pct="mean", std_pct="std", n="count")
                 .reset_index()
                 .sort_values("mean_pct", key=abs, ascending=False))
    print(summary.head(20).to_string(index=False))

    print(f"\n{'='*90}\n  Consistency: fraction of configs with the SAME SIGN as the mean (1.00 = fully consistent)\n{'='*90}")
    def sign_consistency(g):
        s = (g > 0).mean() if g.mean() > 0 else (g < 0).mean() if g.mean() < 0 else float("nan")
        return s
    consistency = (df.groupby(["profile", "parameter"])["signed_deviation_pct"]
                     .apply(sign_consistency)
                     .reset_index(name="sign_consistency")
                     .merge(summary[["profile", "parameter", "mean_pct", "n"]], on=["profile", "parameter"])
                     .sort_values("sign_consistency", ascending=False))
    print(consistency[consistency["n"] >= 10].head(20).to_string(index=False))

    return out_path


if __name__ == "__main__":
    main()
