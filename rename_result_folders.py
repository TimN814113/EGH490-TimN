"""
rename_result_folders.py
EGH490-1 -- Tim Nguyen N11505222

One-off cleanup: results/report_replication_*/ folders are named only by
launch timestamp + PID (e.g. report_replication_2026-09-04_19-37-07-836546_
pid928), which was necessary to make parallel launches collision-proof (see
[[egh490-current-analysis]]) but is unreadable when browsing results/ by
hand. This script reads each folder's own report_replication_summary.csv to
derive a human-readable descriptor (which non-Expert config it ran, n_trials,
t_sim, variability setting) and renames the folder to
report_replication_<descriptor>_<original timestamp+pid suffix>/ -- keeping
the "report_replication_" prefix (aggregate_results.py's DEFAULT_PATTERN glob
depends on it) and the original suffix (guarantees uniqueness + preserves
chronological sort).

variability_backfill.json is keyed by exact OLD folder name for pre-fix runs
that never got a variability column of their own -- renaming would silently
break that lookup, so this script rewrites the backfill map's keys in lockstep.

Usage:
    python rename_result_folders.py --dry-run   # print the plan, rename nothing
    python rename_result_folders.py             # apply it
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
BACKFILL_PATH = RESULTS_DIR / "variability_backfill.json"

# Matches the collision-proof suffix (2026-09-04_19-37-07-836546_pid928) or
# the older plain-timestamp suffix (2026-08-29_20-45-37).
SUFFIX_RE = re.compile(
    r"^report_replication_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d+_pid\d+)?)$"
)


def sanitize(name: str) -> str:
    # Drop the "(model name)" parenthetical -- implied by the config, just noise here.
    name = re.sub(r"\s*\([^)]*\)", "", name)
    return re.sub(r"[^A-Za-z0-9+_-]", "", name.replace(" ", "-"))


def variability_slug(cfg_group: pd.DataFrame, old_name: str, backfill: dict) -> str:
    if "variability" in cfg_group.columns:
        vals = sorted(cfg_group["variability"].dropna().unique())
        if len(vals) == 1:
            return f"var{int(round(vals[0] * 100))}"
        if len(vals) > 1:
            return "var" + "-".join(str(int(round(v * 100))) for v in vals)
    if old_name in backfill:
        return f"var{int(round(backfill[old_name] * 100))}"
    return "varUnk"


def build_descriptor(csv_path: Path, old_name: str, backfill: dict) -> str:
    df = pd.read_csv(csv_path)
    non_expert = df[df["config"] != "Expert-Calibrated"]
    configs = sorted(non_expert["config"].unique()) if len(non_expert) else sorted(df["config"].unique())
    config_slug = "+".join(sanitize(c) for c in configs) or "Expert"

    t_sims = sorted(df["t_sim"].dropna().unique())
    t_sim_slug = "t" + "-".join(str(int(v)) for v in t_sims)

    trials = sorted(df["n_trials"].dropna().unique())
    n_slug = "n" + "-".join(str(int(v)) for v in trials)

    var_slug = variability_slug(non_expert if len(non_expert) else df, old_name, backfill)

    return f"{config_slug}_{n_slug}_{t_sim_slug}_{var_slug}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    backfill = {}
    if BACKFILL_PATH.exists():
        with open(BACKFILL_PATH) as f:
            backfill = json.load(f)

    folders = sorted(RESULTS_DIR.glob("report_replication_*"))
    plan = []  # (old_path, new_path, old_name)
    seen_new_names = {}

    for folder in folders:
        if not folder.is_dir():
            continue
        m = SUFFIX_RE.match(folder.name)
        if not m:
            print(f"SKIP (already renamed or unexpected name): {folder.name}")
            continue
        suffix = m.group(1)
        csv_path = folder / "report_replication_summary.csv"
        if not csv_path.exists():
            print(f"SKIP (no summary csv): {folder.name}")
            continue
        try:
            descriptor = build_descriptor(csv_path, folder.name, backfill)
        except Exception as e:
            print(f"SKIP (error reading {csv_path}): {e}")
            continue
        new_name = f"report_replication_{descriptor}_{suffix}"
        count = seen_new_names.get(new_name, 0)
        seen_new_names[new_name] = count + 1
        if count > 0:
            new_name = f"{new_name}-{count}"
        plan.append((folder, RESULTS_DIR / new_name, folder.name))

    print(f"\n{len(plan)} folders to rename:\n")
    for old, new, _ in plan:
        print(f"  {old.name}\n    -> {new.name}\n")

    if args.dry_run:
        print("DRY RUN -- nothing renamed.")
        return

    backfill_updates = {}
    for old, new, old_name in plan:
        old.rename(new)
        if old_name in backfill:
            backfill_updates[new.name] = backfill.pop(old_name)
    backfill.update(backfill_updates)
    with open(BACKFILL_PATH, "w") as f:
        json.dump(backfill, f, indent=2, sort_keys=True)

    print(f"\nRenamed {len(plan)} folders. variability_backfill.json keys updated for "
          f"{len(backfill_updates)} of them.")


if __name__ == "__main__":
    main()
