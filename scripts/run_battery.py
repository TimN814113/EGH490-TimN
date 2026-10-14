"""
run_battery.py
EGH490-1 — Tim Nguyen N11505222

Launches the full casual/technical x 3-source x 2-draw comparison battery
(Claude plain-conversational, ChatSUMO x Claude, Ollama) against Expert, at
n=10 trials, t_sim=180s, baseline+density+speed sweeps (--skip-interaction —
groups 1-3 already vary N/speed/AV%, which is what "modify simulation
parameters often" in the request maps onto; interaction sweeps can be added
as a follow-up if this first pass shows something worth digging into further).

Each of the 12 configs is run twice: once at its own default noise setting,
once with --variability 0.05 forced (the "noise" phase) — 24 total
run_report_replication.py invocations, run with bounded concurrency (not all
24 at once) since this machine has 20 logical cores and each run is a
single-threaded numpy-heavy process.

Usage:
    python scripts/run_battery.py
    python scripts/run_battery.py --max-workers 8 --trials 10 --t_sim 180
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PYTHON = str(Path(".venv/Scripts/python.exe").resolve())

CONFIGS = [
    ("claude", "casual"), ("claude", "casual2"),
    ("claude", "technical2"), ("claude", "technical3"),
    ("chatsumo-claude", "casual"), ("chatsumo-claude", "casual2"),
    ("chatsumo-claude", "technical"), ("chatsumo-claude", "technical2"),
    ("ollama", "casual"), ("ollama", "casual2"),
    ("ollama", "technical"), ("ollama", "technical2"),
]

RUSH_CONFIGS = [
    ("claude", "casual-rush1"), ("claude", "casual-rush2"),
    ("claude", "technical-rush1"), ("claude", "technical-rush2"),
    ("chatsumo-claude", "casual-rush1"), ("chatsumo-claude", "casual-rush2"),
    ("chatsumo-claude", "technical-rush1"), ("chatsumo-claude", "technical-rush2"),
    ("ollama", "casual-rush1"), ("ollama", "casual-rush2"),
    ("ollama", "technical-rush1"), ("ollama", "technical-rush2"),
]

FLAG = {
    "claude": "--claude-config",
    "chatsumo-claude": "--chatsumo-claude-config",
    "ollama": "--ollama-config",
}


def run_one(source: str, variant: str, noise: bool, trials: int, t_sim: float, log_dir: Path) -> str:
    label = f"{source}_{variant}_{'noise' if noise else 'base'}"
    cmd = [PYTHON, str(Path(__file__).resolve().parent / "run_report_replication.py"),
           "--trials", str(trials), "--t_sim", str(t_sim), "--skip-interaction",
           FLAG[source], variant]
    if noise:
        cmd += ["--variability", "0.05"]

    log_path = log_dir / f"{label}.log"
    t0 = time.time()
    # First run crashed 13/24 processes with a fatal interpreter error (matching
    # numpy/BLAS thread-oversubscription: each of 8 concurrent processes' BLAS
    # backend spawning ~20 threads = ~160 threads fighting for 20 cores). Pin
    # each subprocess to single-threaded BLAS — these simulations don't do
    # large matrix ops, so this costs nothing and eliminates the contention.
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    with open(log_path, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return f"{label}: {status} ({elapsed:.0f}s) -> {log_path}"


def main(max_workers: int, trials: int, t_sim: float, only: list[str] | None, rush: bool):
    log_dir = Path("results") / "battery_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for source, variant in (RUSH_CONFIGS if rush else CONFIGS):
        tasks.append((source, variant, False))
        tasks.append((source, variant, True))

    if only:
        wanted = set(only)
        tasks = [t for t in tasks if f"{t[0]}_{t[1]}_{'noise' if t[2] else 'base'}" in wanted]
        missing = wanted - {f"{s}_{v}_{'noise' if n else 'base'}" for s, v, n in tasks}
        if missing:
            raise SystemExit(f"Unknown --only labels: {missing}")

    print(f"Launching {len(tasks)} runs, max {max_workers} concurrent, "
          f"trials={trials}, t_sim={t_sim}s, skip-interaction\n", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, s, v, n, trials, t_sim, log_dir): (s, v, n) for s, v, n in tasks}
        for fut in as_completed(futures):
            print(fut.result(), flush=True)

    print("\nBattery complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--t_sim", type=float, default=180.0)
    parser.add_argument("--only", type=str, default=None,
                         help="Comma-separated task labels (e.g. ollama_casual_base) to run "
                              "instead of the full 24 — for retrying failures without redoing "
                              "successful runs.")
    parser.add_argument("--rush", action="store_true",
                         help="Run the rush-hour-elicited config set (RUSH_CONFIGS) instead of "
                              "the baseline set (CONFIGS).")
    args = parser.parse_args()
    main(args.max_workers, args.trials, args.t_sim, args.only.split(",") if args.only else None, args.rush)
