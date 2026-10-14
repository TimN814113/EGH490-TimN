# FollowNet + ChatSUMO — Setup Guide
**EGH490-1 | Tim Nguyen N11505222**

---

## Project structure

```
follownet_sumo/
├── sim/
│   ├── configs.py      # Expert + LLM IDM presets
│   ├── physics.py      # IDM, LSTM-approx, DDPG-approx engines
│   ├── fleet.py        # Fleet initialisation and physics update
│   ├── metrics.py      # FollowNet metric accumulators
│   └── runner.py       # Headless simulation runner
├── chatsumo/
│   ├── llm_backend.py  # Claude API + Ollama LLM integration
│   └── sumo_runner.py  # SUMO network generation + TraCI runner
├── app.py              # Streamlit dashboard (main entry point)
├── run_experiment.py   # Headless batch runner (no GUI)
└── requirements.txt
```

---

## Step 1 — Install Python dependencies

In VS Code terminal (inside the `follownet_sumo/` folder):
```bash
pip install -r requirements.txt
```

---

## Step 2 — Install SUMO

1. Download from: https://sumo.dlr.de/docs/Downloads.php
   - Windows: use the `.msi` installer
   - Mac: `brew install sumo`
   - Linux: `sudo apt install sumo sumo-tools`

2. Set the `SUMO_HOME` environment variable:
   - **Windows**: Search "Edit environment variables" → System variables → New
     - Name: `SUMO_HOME`
     - Value: `C:\Program Files (x86)\Eclipse\Sumo`
   - **Mac/Linux**: Add to `~/.bashrc` or `~/.zshrc`:
     ```bash
     export SUMO_HOME=/usr/share/sumo
     ```

3. Restart VS Code after setting the variable.

4. Verify:
   ```bash
   python -c "import traci; print('TraCI OK')"
   ```

---

## Step 3 — Set up Anthropic API key

For the ChatSUMO tab to work with Claude:

**Option A — Environment variable (recommended):**
```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Mac/Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Option B — Enter in the app sidebar** (not saved between sessions).

---

## Step 4 — Run the Streamlit dashboard

```bash
streamlit run app.py
```

Opens in your browser at http://localhost:8501

---

## Step 5 (optional) — Run headless batch experiment

No browser needed. Runs all 30 conditions (2 configs × 3 AV rates × 3 CF models × 5 trials):
```bash
python run_experiment.py
# Or with custom settings:
python run_experiment.py --trials 10 --t_sim 300
```

Results saved to `results/<timestamp>/` as CSV files.

---

## Using the Streamlit dashboard

### Simulation tab
- Select config (Expert / LLM / Custom from ChatSUMO) in sidebar
- Select car-following model (IDM / LSTM-Approx / DDPG-Approx)
- Adjust AV%, fleet size, duration
- Click **▶ Run simulation**
- Metrics update live; results accumulate in the Results tab

### ChatSUMO tab
- Type a plain English description of your traffic scenario
- Click **🤖 Generate IDM config**
- The LLM produces IDM parameters for all 4 driver profiles
- A deviation table shows how it differs from the Expert baseline
- Click **✅ Apply as active config** then switch to "Custom" in sidebar

### SUMO Runner tab
- Requires SUMO installed
- Generates a circular motorway network from scratch
- Configures SUMO vehicle types using the active IDM preset
- Runs via TraCI and extracts trajectory data
- Downloads as CSV

### Results tab
- Bar charts comparing all runs by metric
- Time-series overlay for all completed runs

### Data tab
- Download summary CSV (one row per run)
- Download timeseries CSV (one row per second per run)

---

## Relationship to MATLAB simulator

| Component | MATLAB | Python |
|---|---|---|
| IDM physics | `compute_physics_accel_v47` | `sim/physics.py::idm_accel` |
| MOBIL lane-change | `decide_lane_v47` | `sim/physics.py::mobil_lane_change` |
| Fleet init | `init_fleet_v47` | `sim/fleet.py::init_fleet` |
| FollowNet metrics | `sim.fn.*` accumulators | `sim/metrics.py::FollowNetAccumulator` |
| Config presets | `sim.idm_presets.*` | `sim/configs.py::EXPERT, LLM` |
| GUI | MATLAB uicontrol | Streamlit (browser) |
| LLM elicitation | `apply_config_and_reset` | `chatsumo/llm_backend.py` |
| SUMO integration | N/A (MATLAB only) | `chatsumo/sumo_runner.py` |

---

## Troubleshooting

**"SUMO_HOME not set"** → Set the environment variable and restart VS Code.

**"traci module not found"** → `pip install traci sumolib`

**"Anthropic API key not found"** → Set `ANTHROPIC_API_KEY` or enter it in the sidebar.

**"LLM did not return valid JSON"** → The LLM response was malformed. Try rephrasing the prompt more specifically.

**Streamlit app not opening** → Check that port 8501 is not in use: `streamlit run app.py --server.port 8502`
