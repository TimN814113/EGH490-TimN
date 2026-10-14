"""
aggregate_results.py
EGH490-1 — Tim Nguyen N11505222

Merges every results/report_replication_*/report_replication_summary.csv into
one master dataset. Each run of run_report_replication.py only ever compares
EXPERT against at most one other config, and re-runs EXPERT every time — so
the "5 configs x 2 noise levels" picture described in analysis notes only
exists by hand-stitching ~29 separate CSVs. This script does that stitching
mechanically and reproducibly.

Rows for the SAME (config, group, condition, cf_model, av_pct, n_veh,
road_len, v_limit_kmh, t_sim, variability) that appear in more than one run
folder are pooled into a single higher-power row rather than averaged
naively: n_trials adds up, and mean/std are recombined with the exact
population-variance pooling formula (accounts for between-batch mean shifts,
not just a mean of means). t_sim and variability are both part of the
pooling key deliberately — runs used t_sim in {180, 300, 600}s, and
--variability overrides personality noise for every config in a run
(including configs whose own default is 0%) without changing that config's
name, so two runs of the literal same config name can be a 0%-noise run and
a 5%-noise run. Pooling across either axis silently blends non-comparable
conditions into one number (caught 2026-09-03: a "LLM-Generated" pooled row
turned out to be ~65% the real 0%-noise Table 2 config blended with ~35% a
5%-noise robustness variant, which materially shifted its jerk mean/std).

report_replication_summary.csv did not record a variability column until
this fix — CSVs from before then need their variability backfilled via
--backfill-map (a JSON file of {run_dir: variability}), since that
information only exists in whoever's shell history launched each run, not in
the CSV itself. Any run_dir with neither a variability column nor a
backfill-map entry is left as NaN variability and reported as unresolved
rather than assumed to be 0% — an unresolved row still won't silently pool
with a resolved 0% row (NaN != 0.0), but it also won't pool correctly with
other unresolved runs of the same true (unknown) setting, so treat any
"unresolved" warning as blocking for headline numbers until backfilled.

Usage:
    python aggregate_results.py
    python aggregate_results.py --backfill-map results/variability_backfill.json
    python aggregate_results.py --pattern "results/report_replication_*/report_replication_summary.csv"
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATTERN = "results/report_replication_*/report_replication_summary.csv"
OUT_PATH = Path("results") / "master_summary.csv"

# Expert-Calibrated's own baked-in default (sim/configs.py) is 5% IDM
# personality noise. with_variability() overrides EVERY config in a run
# uniformly when --variability is passed, but 0.05 is a no-op for Expert
# specifically — so Expert is 0.05 in every historical run regardless of
# whether the flag was passed, and a backfill map's value (which records
# whether the flag was passed) must never be applied to Expert rows.
EXPERT_CONFIG_NAME = "Expert-Calibrated"
EXPERT_VARIABILITY = 0.05

KEY_COLS = ["config", "group", "condition", "cf_model", "av_pct",
            "n_veh", "road_len", "v_limit_kmh", "t_sim", "variability"]

# (mean_col, std_col) pairs that get proper variance-aware pooling.
# collision_mean has no recorded std in report_replication_summary.csv, so it
# only gets a weighted-mean pool (flagged separately downstream for testing).
STAT_PAIRS = [("spacing_mse_mean", "spacing_mse_std"),
              ("jerk_mean", "jerk_std"),
              ("ttc_mean", "ttc_std")]


def pool_group(g: pd.DataFrame) -> pd.Series:
    """Combine multiple (mean, std, n) batches for one key into one pooled row."""
    n_total = g["n_trials"].sum()
    out = {
        "n_trials": int(n_total),
        "n_runs_pooled": len(g),
        "run_dirs": ";".join(sorted(g["run_dir"].unique())),
        "collision_mean": float(np.average(g["collision_mean"], weights=g["n_trials"])),
    }
    for mean_col, std_col in STAT_PAIRS:
        n = g["n_trials"].to_numpy(dtype=float)
        m = g[mean_col].to_numpy(dtype=float)
        s = g[std_col].to_numpy(dtype=float)
        pooled_mean = float(np.sum(n * m) / n_total)
        # E[x^2] per batch = var + mean^2 (population variance, ddof=0 — matches
        # np.std's default used when these summary CSVs were originally written).
        sum_sq = float(np.sum(n * (s ** 2 + m ** 2)))
        pooled_var = max(sum_sq / n_total - pooled_mean ** 2, 0.0)
        out[mean_col] = round(pooled_mean, 6)
        out[std_col] = round(pooled_var ** 0.5, 6)
    return pd.Series(out)


def main(pattern: str, backfill_map_path: str | None):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {pattern}")

    backfill_map = {}
    if backfill_map_path:
        with open(backfill_map_path) as f:
            backfill_map = json.load(f)

    frames = []
    unresolved_run_dirs = set()
    for f in files:
        df = pd.read_csv(f)
        run_dir = Path(f).parent.name
        df["run_dir"] = run_dir
        if "variability" not in df.columns:
            is_expert = df["config"] == EXPERT_CONFIG_NAME
            if run_dir in backfill_map:
                override_val = float(backfill_map[run_dir])
                df["variability"] = np.where(is_expert, EXPERT_VARIABILITY, override_val)
            else:
                df["variability"] = np.where(is_expert, EXPERT_VARIABILITY, np.nan)
                if not is_expert.all():
                    unresolved_run_dirs.add(run_dir)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(files)} run folders, {len(raw)} raw condition-rows total.")
    if unresolved_run_dirs:
        print(f"\n*** WARNING: {len(unresolved_run_dirs)} run folder(s) have no 'variability' column "
              f"and no --backfill-map entry — their noise setting is UNKNOWN and left as NaN. ***")
        print("*** Any pooled row touching these run_dirs may silently blend a 0%-noise run with a "
              "***\n*** --variability-overridden run of the same config name. Do not cite headline ***\n"
              "*** numbers built from these until backfilled: ***")
        for rd in sorted(unresolved_run_dirs):
            print(f"    - {rd}")

    pooled = raw.groupby(KEY_COLS, as_index=True, dropna=False).apply(
        pool_group, include_groups=False
    ).reset_index()

    multi = pooled[pooled["n_runs_pooled"] > 1]
    print(f"Merged into {len(pooled)} unique conditions "
          f"({len(multi)} of them pooled from 2+ run folders).")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(OUT_PATH, index=False)
    print(f"Master dataset saved: {OUT_PATH}")

    print("\nCoverage (rows per config x cf_model, all av_pct/n_veh/speed/t_sim conditions summed):")
    coverage = pooled.pivot_table(index="config", columns="cf_model", values="n_trials",
                                   aggfunc="sum", fill_value=0)
    print(coverage.to_string())

    print("\nMax n_trials reached per config (any single condition):")
    print(pooled.groupby("config")["n_trials"].max().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                         help="Glob pattern for run summary CSVs to merge")
    parser.add_argument("--backfill-map", default=None,
                         help="JSON file of {run_dir: variability} for CSVs written before "
                              "run_report_replication.py recorded a variability column")
    args = parser.parse_args()
    main(args.pattern, args.backfill_map)
