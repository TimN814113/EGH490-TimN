"""
app.py
EGH490-1 — Tim Nguyen N11505222
Supervisor: Sebastien Glaser

Streamlit dashboard for FollowNet + ChatSUMO experiment.
Equivalent to GenerateurTraficV47_FollowNet.m but in Python with a browser GUI.

Run with:
    streamlit run app.py

Tabs:
  1. Simulation  — run IDM/LSTM/DDPG simulation, live metric charts
  2. ChatSUMO    — natural language → IDM parameters via LLM
  3. SUMO Runner — generate SUMO network and run via TraCI
  4. Results     — compare all runs, FollowNet summary table
  5. Data        — download CSVs
"""

import os
import json
import tempfile
import pandas as pd
import plotly.express as px
import streamlit as st

from sim.configs import EXPERT, LLM, ALL_CONFIGS, VEHICLE_COLORS
from chatsumo.llm_backend import (
    elicit_idm_params, build_preset_from_llm, compare_to_expert
)
# sim.runner (run_simulation, run_simulation_live, DEFAULT_CFG) is only needed
# by the Simulation tab, imported lazily from sim_ui_base.py below — that
# module (and sim.runner itself) is intentionally omitted from this git
# repository, see README.md "Omitted base files".


st.set_page_config(
    page_title="FollowNet + ChatSUMO — EGH490-1",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ─────────────────────────────────────────────
if "results"    not in st.session_state: st.session_state.results    = []
if "timeseries" not in st.session_state: st.session_state.timeseries = []
if "llm_preset" not in st.session_state: st.session_state.llm_preset = None
if "llm_deviations" not in st.session_state: st.session_state.llm_deviations = []


# ── Sidebar — global simulation settings ─────────────────────────────────────
with st.sidebar:
    st.markdown("### EGH490-1 FollowNet")
    st.caption("Tim Nguyen · N11505222")
    st.divider()

    st.markdown("**Simulation settings**")
    n_veh    = st.slider("Fleet size (vehicles)", 10, 50, 25, 5)
    av_pct   = st.slider("AV penetration rate (%)", 0, 100, 25, 5) / 100
    t_sim    = st.slider("Simulation duration (s)", 30, 300, 120, 30)
    v_limit  = st.slider("Speed limit (km/h)", 60, 130, 110, 10)
    road_len = st.number_input("Road loop length (m)", 400, 2000, 800, 100)
    sensor_range = st.slider(
        "Perception range (m)", 20, 500, 120, 10,
        help="Radar/sensing distance for neighbour detection. Matches MATLAB's "
             "'Perception Range' slider (GenerateurTraficV47_FollowNet.m)."
    )

    st.divider()

    config_choice = st.selectbox(
        "IDM configuration",
        ["Expert-Calibrated", "LLM-Generated", "Custom (from ChatSUMO)"]
    )
    cf_model = st.selectbox(
        "Car-following model",
        ["IDM", "LSTM-Approx", "DDPG-Approx"],
        help="IDM = classical. LSTM-Approx and DDPG-Approx reproduce benchmark "
             "behaviours from Chen et al. (2023) Table 3."
    )

    if config_choice == "Expert-Calibrated":
        active_config = EXPERT
    elif config_choice == "LLM-Generated":
        active_config = LLM
    else:
        active_config = st.session_state.llm_preset or LLM

    override_variability = st.checkbox(
        "Override IDM personality variability", value=False,
        help="MATLAB drives per-vehicle IDM noise from a single global slider "
             "(0-20%, default 1%) applied to whichever config is active. Python "
             "instead gives each config its own fixed variability by default "
             "(Expert=5%, LLM=0%) as part of the Expert-vs-LLM deviation study. "
             "Off by default so existing results are unaffected — turn on to "
             "match MATLAB's single global-slider behaviour instead."
    )
    variability_pct = st.slider(
        "Variability override (%)", 0, 20, 1, 1,
        disabled=not override_variability,
        help="Only applied when the override above is enabled."
    )
    if override_variability:
        active_config = {**active_config, "variability": variability_pct / 100}

    st.divider()
    st.markdown("**API settings**")
    api_key = st.text_input("Anthropic API key", type="password",
                             value=os.environ.get("ANTHROPIC_API_KEY", ""),
                             help="Required for the ChatSUMO tab.")
    llm_backend = st.selectbox("LLM backend", ["Claude API", "Ollama (Llama 3.1)"])
    if llm_backend == "Ollama (Llama 3.1)":
        ollama_host = st.text_input("Ollama host", "http://localhost:11434")
    else:
        ollama_host = "http://localhost:11434"


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_sim, tab_chatsumo, tab_sumo, tab_results, tab_data = st.tabs([
    "🚗 Simulation", "💬 ChatSUMO", "🌐 SUMO Runner", "📊 Results", "💾 Data"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATION
#
# Rendering logic lives in sim_ui_base.py, which is intentionally omitted from
# this git repository (direct recreation of GenerateurTraficV47_FollowNet.m's
# GUI) — see README.md "Omitted base files". Falls back to a notice when that
# module isn't present, e.g. in a fresh clone of the public repository.
# ════════════════════════════════════════════════════════════════════════════
with tab_sim:
    try:
        from sim_ui_base import render_simulation_tab
    except ImportError:
        render_simulation_tab = None

    if render_simulation_tab is None:
        st.warning(
            "This tab requires the base simulation engine (ported from the "
            "original MATLAB FollowNet code), which is withheld from this "
            "public repository pending supervisor approval — see README.md "
            "'Omitted base files'. The ChatSUMO and SUMO Runner tabs are "
            "fully functional without it."
        )
    else:
        render_simulation_tab(
            active_config, cf_model, av_pct, n_veh, t_sim, road_len,
            v_limit, sensor_range, override_variability, VEHICLE_COLORS,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHATSUMO
# ════════════════════════════════════════════════════════════════════════════
with tab_chatsumo:
    st.header("ChatSUMO — natural language → IDM configuration")
    st.caption(
        "Inspired by Li et al. (2024). Describe your traffic scenario in plain English "
        "and the LLM generates IDM parameters for each driver profile. "
        "The generated config can then be used in the Simulation tab."
    )

    example_prompts = [
        "A congested 3-lane motorway in Brisbane at peak hour, 100 km/h speed limit. "
        "Mix of cautious commuters, aggressive young drivers, and self-driving rideshare vehicles.",
        "A rural dual carriageway, 110 km/h limit, light traffic. "
        "Mostly trucks and family cars, few overtakes.",
        "Urban arterial road, 60 km/h limit, high pedestrian activity. "
        "Cautious driving expected with frequent stops at signals.",
    ]

    st.markdown("**Example prompts** — click to load:")
    eg_cols = st.columns(3)
    for col, ex in zip(eg_cols, example_prompts):
        if col.button(ex[:50] + "...", use_container_width=True):
            st.session_state["prompt_val"] = ex

    prompt = st.text_area(
        "Describe your traffic scenario",
        value=st.session_state.get("prompt_val", ""),
        height=120,
        placeholder="e.g. A 3-lane motorway at 110 km/h with 30% autonomous vehicles…"
    )

    col_gen, col_apply = st.columns([2, 1])
    generate_btn = col_gen.button("🤖 Generate IDM config", type="primary", use_container_width=True)
    apply_btn    = col_apply.button("✅ Apply as active config", use_container_width=True,
                                    disabled=st.session_state.llm_preset is None)

    if generate_btn and prompt:
        with st.spinner("Calling LLM…"):
            try:
                backend = "ollama" if "Ollama" in llm_backend else "claude"
                response = elicit_idm_params(
                    prompt,
                    backend=backend,
                    api_key=api_key or None,
                    ollama_host=ollama_host,
                )
                llm_preset = build_preset_from_llm(response)
                st.session_state.llm_preset = llm_preset
                deviations = compare_to_expert(llm_preset, EXPERT)
                st.session_state.llm_deviations = deviations
                st.success(f"✅ Config generated: **{llm_preset['name']}** via {response['source']}")
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

    if apply_btn and st.session_state.llm_preset:
        st.info("✅ LLM config is now active. Switch to 'Custom (from ChatSUMO)' in the sidebar to use it.")

    # Display generated config
    if st.session_state.llm_preset:
        st.divider()
        preset = st.session_state.llm_preset

        st.markdown(f"**Generated config:** {preset['name']}")
        st.caption(preset.get("description", "")[:300])

        rows = []
        for mode, p in preset["profiles"].items():
            rows.append({"Mode": mode, "v0 frac": p["v0_frac"], "T (s)": p["T"],
                         "a_max": p["a_max"], "b_comf": p["b_comf"],
                         "s0 (m)": p["s0"], "δ": p["delta"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Deviation table
        if st.session_state.llm_deviations:
            st.markdown("**Deviations from Expert-Calibrated baseline:**")
            df_dev = pd.DataFrame(st.session_state.llm_deviations)
            df_dev["diff_pct"] = df_dev["diff_pct"].apply(lambda x: f"{x:+.1f}%")

            def color_diff(val):
                if "▲" in str(val): return "color: #e63946"
                if "▼" in str(val): return "color: #2dc653"
                return ""

            st.dataframe(
                df_dev.style.map(color_diff, subset=["direction"]),
                use_container_width=True, hide_index=True
            )

        # JSON export
        st.download_button(
            "⬇ Download generated config (JSON)",
            json.dumps(preset, indent=2),
            file_name="chatsumo_config.json",
            mime="application/json",
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SUMO RUNNER
# ════════════════════════════════════════════════════════════════════════════
with tab_sumo:
    st.header("SUMO network runner (ChatSUMO pipeline)")
    st.caption(
        "Generates a circular motorway network in SUMO, configures vehicles "
        "with the active IDM preset, runs via TraCI, and extracts trajectory data. "
        "Requires SUMO to be installed and SUMO_HOME set."
    )

    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        st.success(f"SUMO found: `{sumo_home}`")
    else:
        st.warning(
            "⚠ SUMO_HOME not set. Install SUMO from https://sumo.dlr.de/docs/Downloads.php "
            "and set the SUMO_HOME environment variable, then restart the app."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        sumo_t_sim  = st.slider("SUMO simulation duration (s)", 30, 300, 120, 30, key="sumo_t")
        sumo_av_pct = st.slider("SUMO AV penetration (%)", 0, 100, 25, 5, key="sumo_av") / 100
        sumo_n_veh  = st.slider("SUMO fleet size", 5, 50, 25, 5, key="sumo_n")
    with col_b:
        sumo_config_choice = st.selectbox(
            "IDM config for SUMO", ["Expert-Calibrated", "LLM-Generated", "Custom (from ChatSUMO)"],
            key="sumo_cfg"
        )
        if sumo_config_choice == "Expert-Calibrated": sumo_preset = EXPERT
        elif sumo_config_choice == "LLM-Generated":   sumo_preset = LLM
        else: sumo_preset = st.session_state.llm_preset or LLM

    run_sumo_btn = st.button(
        "🌐 Generate network + run SUMO",
        type="primary",
        disabled=not sumo_home,
    )

    if run_sumo_btn:
        from chatsumo.sumo_runner import (
            generate_network, generate_vehicles,
            generate_sumo_config, run_sumo_traci,
            sumo_trajectories_to_follownet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prog = st.progress(0.0, "Generating network…")
            try:
                net_path = generate_network(
                    tmpdir, road_len=road_len,
                    n_lanes=3, speed_limit_ms=v_limit/3.6
                )
                prog.progress(0.2, "Generating vehicles…")

                vtypes_path, routes_path = generate_vehicles(
                    tmpdir, sumo_preset, n_veh=sumo_n_veh,
                    av_pct=sumo_av_pct, road_len=road_len,
                    t_sim=sumo_t_sim, speed_limit_ms=v_limit/3.6
                )
                prog.progress(0.4, "Writing SUMO config…")

                cfg_path = generate_sumo_config(
                    tmpdir, net_path, vtypes_path, routes_path, sumo_t_sim
                )
                prog.progress(0.5, "Running SUMO via TraCI…")

                trajectories = run_sumo_traci(
                    cfg_path, sumo_t_sim,
                    progress_cb=lambda f: prog.progress(0.5 + f*0.45, f"SUMO running… {int(f*100)}%")
                )
                prog.progress(1.0, "Done!")

                metrics = sumo_trajectories_to_follownet(trajectories)

                st.success("✅ SUMO simulation complete")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Simulation time (s)", metrics.get("sim_time_s"))
                col_m2.metric("Mean fleet speed (km/h)", metrics.get("mean_speed_kmh"))
                col_m3.metric("Mean jerk (m/s³)", f"{metrics.get('jerk_mean_ms3', 0):.4f}")

                df_traj = pd.DataFrame(trajectories)
                fig = px.line(
                    df_traj.groupby("t")["speed"].mean().reset_index(),
                    x="t", y="speed",
                    title="SUMO: Mean fleet speed over time (m/s)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇ Download SUMO trajectories (CSV)",
                    df_traj.to_csv(index=False),
                    file_name="sumo_trajectories.csv",
                    mime="text/csv",
                )

            except Exception as e:
                prog.empty()
                st.error(f"SUMO error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.header("FollowNet benchmark results")

    results = st.session_state.results
    if not results:
        st.info("No runs yet. Go to the Simulation tab and run at least one experiment.")
    else:
        df_sum = pd.DataFrame(results)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

        if len(results) > 1:
            st.divider()
            metric_cols_plot = st.columns(2)

            with metric_cols_plot[0]:
                fig_mse = px.bar(
                    df_sum,
                    x=[f"{r['config']} · {r['cf_model']} · {r['av_pct']}%" for r in results],
                    y="spacing_mse_m2",
                    title="Spacing MSE (m²) — lower is better",
                    color="config",
                    color_discrete_map={"Expert-Calibrated": "#2E75B6", "LLM-Generated (Claude claude-sonnet-4-6)": "#C55A11"},
                )
                fig_mse.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_mse, use_container_width=True)

            with metric_cols_plot[1]:
                fig_jerk = px.bar(
                    df_sum,
                    x=[f"{r['config']} · {r['cf_model']} · {r['av_pct']}%" for r in results],
                    y="jerk_mean_ms3",
                    title="Mean jerk (m/s³) — lower is better",
                    color="config",
                    color_discrete_map={"Expert-Calibrated": "#2E75B6", "LLM-Generated (Claude claude-sonnet-4-6)": "#C55A11"},
                )
                fig_jerk.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_jerk, use_container_width=True)

            if st.session_state.timeseries:
                df_ts = pd.DataFrame(st.session_state.timeseries)
                df_ts["run_label"] = df_ts["config"] + " · " + df_ts["cf_model"]

                fig_ts = px.line(
                    df_ts, x="sim_time_s", y="fleet_speed_kmh",
                    color="run_label",
                    title="Fleet speed over time — all runs",
                    labels={"sim_time_s": "Time (s)", "fleet_speed_kmh": "Speed (km/h)", "run_label": "Run"},
                )
                fig_ts.add_hline(y=v_limit, line_dash="dot", line_color="gray")
                st.plotly_chart(fig_ts, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("Export data")

    if st.session_state.results:
        df_sum = pd.DataFrame(st.session_state.results)
        st.download_button(
            "⬇ Download summary CSV",
            df_sum.to_csv(index=False),
            file_name="FollowNet_MASTER_summary.csv",
            mime="text/csv",
        )

    if st.session_state.timeseries:
        df_ts = pd.DataFrame(st.session_state.timeseries)
        st.download_button(
            "⬇ Download time-series CSV",
            df_ts.to_csv(index=False),
            file_name="FollowNet_MASTER_timeseries.csv",
            mime="text/csv",
        )

    if not st.session_state.results:
        st.info("No data yet. Run simulations first.")
    else:
        st.subheader("Current session results")
        st.dataframe(pd.DataFrame(st.session_state.results), use_container_width=True)

    st.divider()
    if st.button("🗑 Clear all session data", type="secondary"):
        st.session_state.results    = []
        st.session_state.timeseries = []
        st.rerun()
