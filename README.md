# EGH490-1 — LLM-Based Decision-Making for an Automated Vehicle

**Tim Nguyen (N11505222) — Queensland University of Technology**
**Supervisor: Dr Sebastien Glaser**

Can LLM tools (Claude, Ollama/Llama 3.1) be trusted to configure IDM
car-following parameters for AV traffic simulation? This repository re-tests
the project's original Week 12/13 report at far higher statistical power —
up to 88× the original sample size — across three car-following models
(IDM, LSTM-Approx, DDPG-Approx) with a real SUMO cross-validation, and finds
that the report's own headline safety claim does not replicate.

The full write-up of every finding, exact test conditions, and statistical
evidence is "The Elicitation Ledger" (linked separately — ask Tim if you
need the URL). This repository is the code and data behind it.

## ⚠️ Omitted base files

**This repository does not include the base simulation engine.** Several
files are a direct Python restructuring of the project's original MATLAB
codebase (`GenerateurTraficV47_FollowNet.m`, supplied by Dr Glaser), and are
withheld from git pending his approval to publish:

| Omitted file | What it is |
|---|---|
| `sim/physics.py` | IDM / LSTM-Approx / DDPG-Approx car-following physics — translated from the `.m` file's physics/decision functions |
| `sim/fleet.py` | Fleet initialisation and per-step physics update |
| `sim/metrics.py` | FollowNet metric accumulators (spacing MSE, jerk, TTC, collisions) |
| `sim/runner.py` | Headless + live simulation runner |
| `sim/telemetry.py` | Per-vehicle telemetry (RSS risk matrix, fuel/energy consumption) |
| `sim/configs_base.py` | The `EXPERT` and `LLM` baseline configs — the literature calibration and the original report's own Table 2 values |
| `sim_ui_base.py` | The Streamlit Simulation tab — a direct GUI recreation of the MATLAB app (road view, RSS grid, live graphs) |
| `run_experiment.py` | Headless batch runner — Python equivalent of `FollowNet_Experiment('batch')` |

These files exist locally and are used at runtime; they are simply not
tracked by or pushed through this git repository. **A fresh clone of this
repository will not run end-to-end** — `sim/configs.py` and `sim_ui_base.py`
imports will fail, and any script that calls into `sim.runner` (everything
under "What depends on the omitted base", below) won't execute until those
files are restored locally. That's intentional: it keeps the derivative
MATLAB-port code out of a public repository while still letting every
*original* contribution — the LLM elicitation pipeline, the statistical
testing, the ablation studies, the SUMO cross-validation — be read, reviewed,
and built on in full.

Everything **not** in the table above is this project's own new work with
no MATLAB equivalent, and is committed normally.

### What depends on the omitted base

These files are fully committed (their *code* is all original work) but
import `sim.runner` and therefore won't execute standalone without the
omitted files: `chatsumo/sumo_runner.py`, `run_ablation.py`,
`run_bcomf_ablation.py`, `run_ddpg_profile_ablation.py`,
`run_report_replication.py`, `run_battery.py`, `run_sumo_battery.py`,
`run_claude_scenario_elicitation.py`, and `app.py`'s Simulation tab
(it degrades to a warning message instead of crashing — see `app.py`).

`chatsumo/llm_backend.py`, `aggregate_results.py`, `analyze_significance.py`,
`build_signed_deviation_tables.py`, `rename_result_folders.py`, and
`build_summary_pptx.py` have no dependency on the omitted base and run as-is.

## Repository layout

```
sim/
  configs.py          # 25+ LLM-elicited & ablation configs (new work)
  configs_base.py      # OMITTED — Expert/LLM baseline dicts
  physics.py            # OMITTED — car-following physics
  fleet.py, metrics.py, # OMITTED — engine internals
  runner.py, telemetry.py
chatsumo/
  llm_backend.py       # Claude API + Ollama elicitation (new)
  sumo_runner.py       # Real SUMO/TraCI integration (new)
app.py                  # Streamlit dashboard (ChatSUMO/SUMO tabs new;
                         # Simulation tab calls the omitted sim_ui_base.py)
run_report_replication.py   # Main experiment harness (new)
run_ablation.py              # ACC role-inversion ablation (new)
run_ddpg_profile_ablation.py # SPORTY.a_max mechanistic ablation (new)
run_bcomf_ablation.py        # NORMAL.b_comf causal ablation (new)
run_sumo_battery.py          # Real SUMO validation battery (new)
run_claude_scenario_elicitation.py  # Quiet/rush-hour scenario testing (new)
run_battery.py                # Parallel battery launcher (new)
aggregate_results.py          # Pools raw runs into master_summary.csv (new)
analyze_significance.py       # Welch's t-test + Poisson z-test + BH-FDR (new)
build_signed_deviation_tables.py  # Per-parameter bias table (new)
rename_result_folders.py      # Results-folder housekeeping (new)
build_summary_pptx.py         # Generates a findings-summary deck (new)
results_summary/              # Curated aggregate CSVs/JSONs — the actual
                               # data backing every finding (raw per-trial
                               # data, ~4GB, is not included — see below)
```

## Setup

See `SETUP.md` for full install steps. Short version:

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY if using the ChatSUMO tab
```

`SETUP.md` was written before the base-file split above and still describes
the complete local project layout (useful once the omitted files are
restored) — treat this README as the authoritative note on what's actually
in the public repository.

## Data

`results_summary/` holds the curated CSV/JSON files that back every finding
cited in the project write-up: `master_summary.csv` (every simulation run,
pooled), `significance_report.csv` (every statistical test), the ablation
study summaries, the SUMO battery summaries, and the LLM scenario-elicitation
transcripts. The raw per-trial timeseries data behind these (~4GB across the
full experimental history) is not included.
