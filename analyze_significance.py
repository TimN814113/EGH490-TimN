"""
analyze_significance.py
EGH490-1 — Tim Nguyen N11505222

Statistical significance testing on top of results/master_summary.csv (built
by aggregate_results.py). Every claim in the report and in the follow-up
analysis so far ("+92% more TTC exposure", "+277% TTC deviation under DDPG",
etc.) has been a raw percentage difference between condition means, with no
test of whether that difference is distinguishable from trial-to-trial noise.
This script adds that layer for every (LLM config) x (condition) pair against
Expert-Calibrated, for all four FollowNet metrics.

Method:
  - spacing_mse, jerk, ttc: Welch's t-test (unequal variances, unequal n) on
    the pooled per-condition (mean, std, n) from master_summary.csv, since raw
    per-trial values are not retained in that CSV — only summary statistics
    are, so a from-statistics Welch test is the correct tool rather than a
    t-test requiring raw samples. 95% CI on the mean difference and Cohen's d
    (average-variance version, appropriate for unequal n/variance) are also
    reported.
  - collision: report_replication_summary.csv never recorded a std for
    collision counts (only the mean), so a normal Welch t-test isn't available
    for it. Collisions are rare-event counts, so instead this uses a Poisson
    rate z-test (normal approximation to the Poisson, valid once total event
    counts aren't tiny) on the total collision counts implied by
    collision_mean * n_trials.
  - Benjamini-Hochberg FDR correction is applied within each (config, metric)
    family — i.e. across all conditions tested for one config on one metric —
    since many conditions are compared per config and an uncorrected p<0.05
    threshold would be expected to produce false positives by chance alone.

Usage:
    python analyze_significance.py
    python analyze_significance.py --baseline "Expert-Calibrated" --alpha 0.05
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sim.configs import ALL_CONFIGS

MASTER_PATH = Path("results") / "master_summary.csv"
OUT_PATH = Path("results") / "significance_report.csv"

# Each config's own baked-in default noise level (sim/configs.py) — Expert is
# 0.05, every LLM-elicited config is 0.0 unless a prompt explicitly asked for
# variability. Comparing "each config at its own default" is what the report
# actually did (nobody passed --variability for the headline comparison) and
# is NOT the same as requiring identical variability on both sides — Expert
# never appears at 0.0 noise in this dataset at all, so an identical-value
# match would silently drop every Expert comparison.
DEFAULT_VARIABILITY = {cfg["name"]: cfg.get("variability", 0.0) for cfg in ALL_CONFIGS.values()}

JOIN_KEYS = ["group", "condition", "cf_model", "av_pct",
             "n_veh", "road_len", "v_limit_kmh", "t_sim"]

METRICS = [
    ("spacing_mse_mean", "spacing_mse_std", "spacing_mse"),
    ("jerk_mean", "jerk_std", "jerk"),
    ("ttc_mean", "ttc_std", "ttc"),
]


def welch_test(m1, s1, n1, m2, s2, n2, alpha=0.05):
    """Welch's t-test + Cohen's d + 95% CI from summary statistics only."""
    diff = m1 - m2
    v1, v2 = s1 ** 2, s2 ** 2
    se_sq = v1 / n1 + v2 / n2
    if se_sq <= 0:
        # Both sides have zero variance (e.g. identical deterministic runs).
        return dict(diff=diff, pct_diff=np.nan if m2 == 0 else 100 * diff / abs(m2),
                    t_stat=np.nan, df=np.nan, p_value=1.0 if diff == 0 else 0.0,
                    ci_lo=diff, ci_hi=diff, cohens_d=np.nan, note="zero variance both sides")
    se = se_sq ** 0.5
    t_stat = diff / se
    df = se_sq ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lo, ci_hi = diff - t_crit * se, diff + t_crit * se
    pooled_sd = ((v1 + v2) / 2) ** 0.5
    cohens_d = diff / pooled_sd if pooled_sd > 0 else np.nan
    pct_diff = 100 * diff / abs(m2) if m2 != 0 else np.nan
    return dict(diff=diff, pct_diff=pct_diff, t_stat=t_stat, df=df, p_value=p_value,
                ci_lo=ci_lo, ci_hi=ci_hi, cohens_d=cohens_d, note="")


def poisson_rate_ztest(mean1, n1, mean2, n2):
    """
    Normal-approximation two-sample Poisson rate test on collision counts.
    rate_i = mean_i (collisions/trial); Var(count_i) ~= count_i under Poisson,
    so Var(rate_i) ~= count_i / n_i^2 = mean_i / n_i.
    """
    k1, k2 = mean1 * n1, mean2 * n2
    var1, var2 = (k1 / n1 ** 2) if n1 > 0 else 0, (k2 / n2 ** 2) if n2 > 0 else 0
    se_sq = var1 + var2
    diff = mean1 - mean2
    if se_sq <= 0:
        return dict(diff=diff, z_stat=np.nan, p_value=1.0 if diff == 0 else np.nan,
                     note="no collision variance (both configs collision-free)")
    z = diff / se_sq ** 0.5
    p = 2 * stats.norm.sf(abs(z))
    return dict(diff=diff, z_stat=z, p_value=p, note="")


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = pvalues.to_numpy(dtype=float)
    n = len(p)
    valid = ~np.isnan(p)
    q = np.full(n, np.nan)
    if valid.sum() == 0:
        return pd.Series(q, index=pvalues.index)
    order = np.argsort(p[valid])
    ranked_p = p[valid][order]
    ranks = np.arange(1, valid.sum() + 1)
    raw_q = ranked_p * valid.sum() / ranks
    # enforce monotonicity (standard BH step-up correction)
    adj_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_valid = np.empty(valid.sum())
    q_valid[order] = np.clip(adj_q, 0, 1)
    q[valid] = q_valid
    return pd.Series(q, index=pvalues.index)


def main(baseline: str, alpha: float, variability_mode: str):
    if not MASTER_PATH.exists():
        raise SystemExit(f"{MASTER_PATH} not found — run aggregate_results.py first.")
    df = pd.read_csv(MASTER_PATH)

    if "variability" not in df.columns:
        raise SystemExit("master_summary.csv has no 'variability' column — re-run "
                          "aggregate_results.py (with --backfill-map if needed) first.")

    if variability_mode == "default":
        default_variab = df["config"].map(DEFAULT_VARIABILITY).fillna(0.0)
        df = df[np.isclose(df["variability"], default_variab)]
        print("Variability filter: each config's own baked-in default "
              f"({', '.join(f'{k}={v}' for k, v in DEFAULT_VARIABILITY.items())}).")
    else:
        target = float(variability_mode)
        df = df[np.isclose(df["variability"], target)]
        print(f"Variability filter: exactly {target} for every config.")
    print(f"{len(df)} rows remain after the variability filter.\n")

    base = df[df["config"] == baseline].set_index(JOIN_KEYS)
    if base.empty:
        raise SystemExit(f"No rows found for baseline config '{baseline}'. "
                          f"Available configs: {sorted(df['config'].unique())}")
    if base.index.duplicated().any():
        raise SystemExit(f"Baseline '{baseline}' has duplicate condition keys after "
                          f"pooling — aggregate_results.py should have prevented this.")

    rows = []
    other_configs = [c for c in df["config"].unique() if c != baseline]
    for cfg in other_configs:
        cfg_df = df[df["config"] == cfg].set_index(JOIN_KEYS)
        shared = cfg_df.index.intersection(base.index)
        for key in shared:
            b, l = base.loc[key], cfg_df.loc[key]
            row = {k: v for k, v in zip(JOIN_KEYS, key)}
            row["config"] = cfg
            row["n_expert"], row["n_config"] = int(b["n_trials"]), int(l["n_trials"])

            for mean_col, std_col, label in METRICS:
                res = welch_test(l[mean_col], l[std_col], l["n_trials"],
                                  b[mean_col], b[std_col], b["n_trials"], alpha)
                for k, v in res.items():
                    row[f"{label}_{k}"] = v

            crow = poisson_rate_ztest(l["collision_mean"], l["n_trials"],
                                       b["collision_mean"], b["n_trials"])
            for k, v in crow.items():
                row[f"collision_{k}"] = v

            rows.append(row)

    report = pd.DataFrame(rows)

    for label in [m[2] for m in METRICS] + ["collision"]:
        for cfg in other_configs:
            mask = report["config"] == cfg
            report.loc[mask, f"{label}_q_value"] = bh_fdr(report.loc[mask, f"{label}_p_value"])
        report[f"{label}_significant"] = report[f"{label}_q_value"] < alpha

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_PATH, index=False)
    print(f"Significance report saved: {OUT_PATH} ({len(report)} config-vs-{baseline} comparisons)\n")

    print(f"{'='*80}\nSummary: significant differences vs {baseline} after BH-FDR correction (q<{alpha})\n{'='*80}")
    for cfg in other_configs:
        sub = report[report["config"] == cfg]
        if sub.empty:
            continue
        print(f"\n{cfg}  ({len(sub)} shared conditions)")
        for label in [m[2] for m in METRICS] + ["collision"]:
            n_sig = int(sub[f"{label}_significant"].sum())
            n_testable = int(sub[f"{label}_p_value"].notna().sum())
            print(f"  {label:12s}: {n_sig}/{n_testable} conditions significant")

    print(f"\n{'='*80}\nReport Findings 2 & 3 recheck: IDM, N=25, 110km/h, AV=25% (the report's own baseline point)\n{'='*80}")
    focus = report[(report["cf_model"] == "IDM") & (report["av_pct"] == 25)
                    & (report["n_veh"] == 25) & (report["v_limit_kmh"] == 110)]
    if focus.empty:
        print("  No matching pooled condition found for this exact point.")
    else:
        cols = ["config", "group", "t_sim", "n_expert", "n_config",
                "jerk_pct_diff", "jerk_p_value", "jerk_q_value", "jerk_significant",
                "ttc_pct_diff", "ttc_p_value", "ttc_q_value", "ttc_significant"]
        focus = focus.sort_values(["config", "t_sim"])
        print(focus[cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="Expert-Calibrated")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--variability", dest="variability_mode", default="default",
                         help="'default' (recommended) = each config at its own baked-in "
                              "noise setting, matching what the report actually compared "
                              "(Expert=0.05, LLM configs=0.0 unless noted). Or a float "
                              "(e.g. 0.05) to instead compare all configs at that exact "
                              "noise level — only meaningful for configs that were actually "
                              "run with --variability set to that value.")
    args = parser.parse_args()
    main(args.baseline, args.alpha, args.variability_mode)
