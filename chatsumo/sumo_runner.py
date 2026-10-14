"""
chatsumo/sumo_runner.py
EGH490-1 — Tim Nguyen N11505222

Generates SUMO network files (equivalent to ChatSUMO's Simulation Generation Module)
and runs SUMO via TraCI, extracting vehicle trajectory data for FollowNet metric
computation. This bridges the ChatSUMO pipeline with our FollowNet analysis.

Workflow:
  1. generate_network()  — write XML files for a circular motorway network
  2. generate_vehicles() — write vehicle type + route XML with IDM parameters
  3. run_sumo_traci()    — execute SUMO via TraCI, collect trajectory data
  4. compute_metrics()   — apply FollowNet metrics to SUMO trajectory output

References:
  Li et al. (2024) ChatSUMO — network generation approach
  Chen et al. (2023) FollowNet — evaluation metrics
"""

import os
import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


SUMO_HOME = os.environ.get("SUMO_HOME", "")

# Number of straight segments approximating the circular motorway loop. Using
# too few (the original value was 8) makes each corner a sharp ~45-degree turn,
# and SUMO computes a geometric cornering speed limit at each junction from the
# turning angle — at 8 segments this capped speed far below any of the
# elicited configs' desired speed (observed 2026-09-04: ~36 km/h realized vs a
# 110 km/h speed limit), which then dominates jerk/TTC/spacing-MSE results
# with a network artifact rather than genuine car-following behaviour. 72
# segments (5 degrees/corner) keeps the cornering speed limit comfortably
# above any realistic desired speed used in this study.
N_ROAD_SEGMENTS = 72


def _check_sumo():
    if not SUMO_HOME:
        raise EnvironmentError(
            "SUMO_HOME environment variable is not set.\n"
            "Install SUMO from https://sumo.dlr.de/docs/Downloads.php "
            "and set SUMO_HOME to the installation directory "
            "(e.g. C:\\Program Files (x86)\\Eclipse\\Sumo)."
        )
    sumo_bin = Path(SUMO_HOME) / "bin" / "sumo"
    if not sumo_bin.exists() and not Path(str(sumo_bin) + ".exe").exists():
        raise FileNotFoundError(
            f"SUMO binary not found at {sumo_bin}.\n"
            f"Check that SUMO_HOME is set correctly: {SUMO_HOME}"
        )


def generate_network(output_dir: str, road_len: float = 800.0,
                     n_lanes: int = 3, speed_limit_ms: float = 30.56) -> str:
    """
    Generate a circular motorway network (SUMO .net.xml equivalent).
    Uses a simple straight road with endpoints connected (loop route).

    This replicates ChatSUMO's Simulation Generation Module but for a
    controlled benchmark scenario rather than an OSM-based real network.

    Returns path to the generated network directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- nodes.xml ---
    nodes = ET.Element("nodes")
    n_segments = N_ROAD_SEGMENTS
    radius = road_len / (2 * math.pi)
    for i in range(n_segments):
        angle = 2 * math.pi * i / n_segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        ET.SubElement(nodes, "node", id=f"n{i}", x=f"{x:.2f}", y=f"{y:.2f}", type="priority")
    nodes_path = os.path.join(output_dir, "nodes.xml")
    ET.ElementTree(nodes).write(nodes_path)

    # --- edges.xml ---
    edges = ET.Element("edges")
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # "from" is a Python keyword, so it must be passed via `attrib`, not as a
        # kwarg (ET.SubElement does not rename from_= to "from" — it writes the
        # attribute literally as "from_", which netconvert then never sees).
        e = ET.SubElement(edges, "edge", attrib={
            "id": f"e{i}", "from": f"n{i}", "to": f"n{j}",
            "numLanes": str(n_lanes), "speed": f"{speed_limit_ms:.2f}",
            "priority": "1",
        })
    edges_path = os.path.join(output_dir, "edges.xml")
    ET.ElementTree(edges).write(edges_path)

    # Run netconvert to build .net.xml
    net_path = os.path.join(output_dir, "motorway.net.xml")
    netconvert = Path(SUMO_HOME) / "bin" / "netconvert"
    result = subprocess.run([
        str(netconvert),
        "--node-files", nodes_path,
        "--edge-files", edges_path,
        "--output-file", net_path,
        "--no-warnings",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed:\n{result.stderr}")

    return net_path


def generate_vehicles(output_dir: str, preset: dict,
                      n_veh: int = 25, av_pct: float = 0.25,
                      road_len: float = 800.0, t_sim: float = 120.0,
                      speed_limit_ms: float = 30.56,
                      use_sumo_defaults: bool = False) -> tuple[str, str]:
    """
    Generate SUMO vehicle types and routes XML files.
    Maps IDM presets to SUMO's IDM car-following model parameters.

    This is the ChatSUMO Customisation Module equivalent.

    use_sumo_defaults: if True, omit accel/decel/tau/minGap/delta entirely so
    SUMO applies its OWN built-in IDM defaults — a genuine second "expert"
    reference independent of this project's own hand-calibrated EXPERT preset
    and with zero LLM involvement. Only maxSpeed (v0) still varies per profile,
    preserving the 4-driver-type behavioural framing used throughout this
    study. The actual default values SUMO applies are NOT hardcoded here —
    run_sumo_full_metrics() queries them back via TraCI after the vTypes are
    loaded, so the spacing-MSE/TTC formulas always use SUMO's real applied
    parameters rather than an assumption about what its defaults are.

    Returns (vtypes_path, routes_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    profiles = preset["profiles"]

    # --- vtypes.xml ---
    vtypes = ET.Element("additional")
    profile_map = {
        "ACC":      {"color": "0,180,216",  "length": "4.50", "accel": None, "decel": None},
        "CAUTIOUS": {"color": "45,198,83",  "length": "4.50", "accel": None, "decel": None},
        "NORMAL":   {"color": "72,149,239", "length": "4.50", "accel": None, "decel": None},
        "SPORTY":   {"color": "230,57,70",  "length": "4.50", "accel": None, "decel": None},
        "TRUCK":    {"color": "200,150,50", "length": "14.0", "accel": None, "decel": None},
    }

    for mode, pm in profile_map.items():
        p = profiles.get(mode, profiles["NORMAL"])
        v0 = speed_limit_ms * p["v0_frac"]
        attrs = dict(
            id=mode,
            color=pm["color"],
            length=pm["length"],
            maxSpeed=f"{v0:.2f}",
            emergencyDecel="9.0",
            carFollowModel="IDM",
            stepping="0.25",
        )
        if use_sumo_defaults:
            # Deliberately omit accel/decel/tau/minGap/delta/lc* — SUMO fills
            # in its own built-in IDM defaults, queried back via TraCI later.
            pass
        else:
            attrs.update(
                accel=f"{p['a_max']:.2f}",
                decel=f"{p['b_comf']:.2f}",
                tau=f"{p['T']:.3f}",
                delta=f"{p['delta']:.2f}",
                minGap=f"{p['s0']:.2f}",
                lcStrategic="1.0",
                lcCooperative=f"{1.0 - p['threshold']:.2f}",
            )
        ET.SubElement(vtypes, "vType", **attrs)

    vtypes_path = os.path.join(output_dir, "vtypes.xml")
    ET.ElementTree(vtypes).write(vtypes_path)

    # --- routes.xml ---
    routes_xml = ET.Element("routes")
    # A SUMO <route> is a ONE-SHOT edge sequence — a vehicle that reaches the
    # last edge is removed from the simulation as "arrived," it does NOT loop
    # back to the first edge just because the edges are geometrically
    # connected (observed 2026-09-04: with the correct network-artifact fix
    # above, vehicles hit ~110 km/h, completed one ~800m lap in ~27s, and were
    # then silently removed — collapsing n_active mid-run and corrupting every
    # metric downstream). Fix: repeat the edge sequence enough times that no
    # vehicle can exhaust its route within t_sim, even at 1.5x the speed limit
    # (covers SPORTY's >100% v0_frac) — this is the standard SUMO technique
    # for a circular test track (a single <route> can legally repeat edges).
    n_seg = N_ROAD_SEGMENTS
    max_plausible_speed = speed_limit_ms * 1.5
    n_laps = max(4, math.ceil((max_plausible_speed * t_sim) / road_len) + 2)
    edge_list = " ".join(f"e{i % n_seg}" for i in range(n_seg * n_laps))
    ET.SubElement(routes_xml, "route", id="circle", edges=edge_list)

    # Vehicle mode assignment
    n_av   = round(n_veh * av_pct)
    n_hdv  = n_veh - n_av
    n_each = max(1, n_hdv // 3)
    modes  = (["ACC"] * n_av + ["CAUTIOUS"] * n_each +
              ["NORMAL"] * n_each + ["SPORTY"] * max(0, n_hdv - 2*n_each))
    while len(modes) < n_veh:
        modes.append("NORMAL")

    spacing = road_len / n_veh
    for i, mode in enumerate(modes[:n_veh]):
        depart_pos = i * spacing
        ET.SubElement(routes_xml, "vehicle",
            id=f"v{i}",
            type=mode,
            route="circle",
            depart="0",
            departPos=f"{depart_pos:.1f}",
            departLane="random",
            departSpeed="max",
        )

    routes_path = os.path.join(output_dir, "routes.xml")
    ET.ElementTree(routes_xml).write(routes_path)

    return vtypes_path, routes_path


def generate_sumo_config(output_dir: str, net_path: str,
                         vtypes_path: str, routes_path: str,
                         t_sim: float = 120.0) -> str:
    """Write the SUMO .sumocfg file."""
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file",     value=net_path)
    ET.SubElement(inp, "additional-files", value=vtypes_path)
    ET.SubElement(inp, "route-files",  value=routes_path)
    tim = ET.SubElement(cfg, "time")
    ET.SubElement(tim, "begin",  value="0")
    ET.SubElement(tim, "end",    value=str(int(t_sim)))
    ET.SubElement(tim, "step-length", value="0.05")

    cfg_path = os.path.join(output_dir, "simulation.sumocfg")
    ET.ElementTree(cfg).write(cfg_path)
    return cfg_path


def run_sumo_traci(sumocfg_path: str, t_sim: float = 120.0,
                   progress_cb=None) -> list[dict]:
    """
    Run SUMO via TraCI and collect per-timestep vehicle trajectory data.

    Returns a list of trajectory records:
        [{"step": int, "t": float, "vid": str, "x": float, "speed": float,
          "lane": int, "accel": float}, ...]
    """
    _check_sumo()
    try:
        import traci
    except ImportError:
        raise ImportError(
            "traci is not installed. Run: pip install traci"
        )

    # Defensively drop any stale connection left over from a previous run that
    # errored (or was interrupted by a Streamlit rerun) before reaching
    # traci.close() below — otherwise traci.start() fails with
    # "Connection 'default' is already active."
    try:
        traci.close()
    except Exception:
        pass

    sumo_bin = Path(SUMO_HOME) / "bin" / "sumo"
    if not sumo_bin.exists():
        sumo_bin = Path(SUMO_HOME) / "bin" / "sumo.exe"

    traci.start([str(sumo_bin), "-c", sumocfg_path,
                 "--no-warnings", "--no-step-log"])

    trajectories = []
    step = 0
    dt   = 0.05
    prev_speeds = {}

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            t = step * dt
            if t > t_sim:
                break

            for vid in traci.vehicle.getIDList():
                x     = traci.vehicle.getDistance(vid)
                speed = traci.vehicle.getSpeed(vid)
                lane  = traci.vehicle.getLaneIndex(vid)
                prev  = prev_speeds.get(vid, speed)
                accel = (speed - prev) / dt
                prev_speeds[vid] = speed

                trajectories.append({
                    "step":  step,
                    "t":     round(t, 3),
                    "vid":   vid,
                    "x":     round(x, 2),
                    "speed": round(speed, 3),
                    "lane":  lane + 1,
                    "accel": round(accel, 4),
                })

            step += 1
            if progress_cb and step % 200 == 0:
                progress_cb(t / t_sim)
    finally:
        # Always close, even if the loop raises, so the next run doesn't
        # inherit a dangling connection.
        traci.close()

    return trajectories


def run_sumo_full_metrics(
    preset: dict,
    n_veh: int = 25,
    av_pct: float = 0.25,
    road_len: float = 800.0,
    t_sim: float = 120.0,
    v_limit_kmh: float = 110.0,
    sensor_range: float = 120.0,
    use_sumo_defaults: bool = False,
) -> tuple[dict, list]:
    """
    Full FollowNet metric pipeline through REAL SUMO (not the Python
    approximation in sim/runner.py) — generates the network/vehicles/config,
    runs TraCI collecting leader/gap/collision data every step, and computes
    the exact same 4 metrics using the exact same formulas as
    sim/metrics.py's FollowNetAccumulator, so results are directly comparable
    to run_simulation()'s output (same summary dict shape, same units).

    Only IDM is meaningful here (SUMO's own carFollowModel="IDM", set in
    generate_vehicles()) — there is no SUMO analogue of LSTM-Approx/DDPG-Approx,
    so this is inherently an IDM-only comparison: "our Python port of IDM" vs
    "SUMO's own IDM implementation," given identical elicited parameters.

    KNOWN DIVERGENCE (found 2026-09-04, not resolved — disclose in any writeup
    using this function): SUMO's fleet-average speed progressively decays over
    a run (observed 110->77 km/h over 105s for Expert-Calibrated, N=15,
    AV=25%) in a way the Python engine's own IDM+MOBIL implementation does NOT
    (Python stabilises ~106-109 km/h under identical parameters over the same
    duration). Likely cause: SUMO's own lane-changing model (LC2013 default)
    interacting differently with the lcCooperative/lcStrategic mapping in
    generate_vehicles() than the Python engine's MOBIL port does, causing more
    persistent blocking behind slow leaders. Not chased further given time
    constraints — comparisons using this function should either use short/
    moderate t_sim (<=90s) where the divergence is milder, or explicitly note
    it as a real SUMO-vs-Python-engine fidelity difference rather than treating
    long-duration SUMO metrics as directly comparable to the Python engine's.

    Spacing MSE / TTC formulas (matching sim/metrics.py exactly):
        s_star = s0 + max(0, v*T + (v*dv) / (2*sqrt(a_max*b_comf)))
        spacing_mse = mean((s_actual - s_star)^2)
        ttc = s_actual / dv if dv > 0.1 (closing) else 999 (no danger)
        ttc_exposure_pct = % of vehicle-timesteps with ttc < TTC_THRESHOLD (3.0s)
    s_actual comes directly from TraCI's vehicle.getLeader() gap, which SUMO
    already reports bumper-to-bumper (not centre-to-centre) — no length
    correction needed, unlike the Python engine's raw centre-to-centre distance.

    Collision counting matches sim/metrics.py's convention (+0.5 per vehicle
    involved, so a 2-vehicle collision pair = 1.0 event), via TraCI's
    getCollidingVehiclesIDList() with explicit collision.action=remove so
    collisions are flagged rather than silently teleported past.

    Returns (summary, timeseries) — same shape as sim.runner.run_simulation(),
    with "source": "SUMO/TraCI" added to summary for provenance.
    """
    import tempfile as _tempfile
    _check_sumo()
    try:
        import traci
    except ImportError:
        raise ImportError("traci is not installed. Run: pip install traci")

    from sim.metrics import TTC_THRESHOLD

    v_limit_ms = v_limit_kmh / 3.6

    work_dir = _tempfile.mkdtemp(prefix="sumo_full_")
    net_path = generate_network(work_dir, road_len=road_len, speed_limit_ms=v_limit_ms)
    vtypes_path, routes_path = generate_vehicles(
        work_dir, preset, n_veh=n_veh, av_pct=av_pct,
        road_len=road_len, t_sim=t_sim, speed_limit_ms=v_limit_ms,
        use_sumo_defaults=use_sumo_defaults,
    )
    sumocfg_path = generate_sumo_config(work_dir, net_path, vtypes_path, routes_path, t_sim=t_sim)

    try:
        traci.close()
    except Exception:
        pass

    sumo_bin = Path(SUMO_HOME) / "bin" / "sumo"
    if not sumo_bin.exists():
        sumo_bin = Path(SUMO_HOME) / "bin" / "sumo.exe"

    # SUMO defaults to a FIXED internal seed when --seed/--random isn't given
    # — every "trial" was silently deterministic and byte-identical (caught
    # 2026-09-04: 8/8 trials of the same config returned literally identical
    # metrics to 4 decimal places). Passing --random makes SUMO seed from
    # system time/entropy each call, matching how every other trial-based
    # simulation in this study gets real between-trial variance.
    traci.start([str(sumo_bin), "-c", sumocfg_path,
                 "--no-warnings", "--no-step-log",
                 "--collision.action", "remove",
                 "--collision.check-junctions", "true",
                 "--random"])

    # Query the parameters SUMO ACTUALLY applied per vType, rather than
    # trusting what generate_vehicles() intended to set — correct regardless
    # of use_sumo_defaults, and self-correcting if SUMO clamps/adjusts a
    # requested value. delta isn't queried: it doesn't appear in the s_star/
    # TTC formulas below (only in the acceleration equation, which SUMO
    # computes internally — not needed here).
    profiles = {}
    for mode in ("ACC", "CAUTIOUS", "NORMAL", "SPORTY"):
        profiles[mode] = {
            "a_max":  traci.vehicletype.getAccel(mode),
            "b_comf": traci.vehicletype.getDecel(mode),
            "T":      traci.vehicletype.getTau(mode),
            "s0":     traci.vehicletype.getMinGap(mode),
        }

    dt = 0.05
    spacing_sq_sum, spacing_count = 0.0, 0
    jerk_sum, jerk_count = 0.0, 0
    ttc_danger, ttc_total = 0, 0
    collision_count = 0.0
    already_crashed: set[str] = set()
    prev_speed: dict[str, float] = {}
    prev_accel: dict[str, float] = {}
    timeseries: list[dict] = []
    win = {"sq": 0.0, "sq_n": 0, "jk": 0.0, "jk_n": 0, "ttcd": 0, "ttct": 0}
    snap_every_steps = int(1.0 / dt)  # matches sim_cfg's snapshot_every=1.0s default
    step = 0
    t = 0.0
    n_veh_seen: set[str] = set()

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            t = step * dt
            if t > t_sim:
                break

            for vid in traci.simulation.getCollidingVehiclesIDList():
                if vid not in already_crashed:
                    collision_count += 0.5
                    already_crashed.add(vid)

            speeds_now = []
            for vid in traci.vehicle.getIDList():
                n_veh_seen.add(vid)
                speed = traci.vehicle.getSpeed(vid)
                speeds_now.append(speed)
                prev_v = prev_speed.get(vid, speed)
                accel = (speed - prev_v) / dt
                prev_speed[vid] = speed

                mode = traci.vehicle.getTypeID(vid)
                p = profiles.get(mode, profiles["NORMAL"])

                leader = traci.vehicle.getLeader(vid, sensor_range)
                if leader is not None and leader[0]:
                    leader_id, s_actual = leader
                    if s_actual >= 0:
                        leader_speed = traci.vehicle.getSpeed(leader_id)
                        dv = speed - leader_speed
                        ab = max(p["a_max"] * p["b_comf"], 0.01)
                        s_star = p["s0"] + max(0.0, speed * p["T"] + (speed * dv) / (2.0 * math.sqrt(ab)))
                        err_sq = (s_actual - s_star) ** 2
                        spacing_sq_sum += err_sq
                        spacing_count += 1
                        win["sq"] += err_sq
                        win["sq_n"] += 1

                        ttc = s_actual / dv if dv > 0.1 else 999.0
                        ttc_total += 1
                        win["ttct"] += 1
                        if ttc < TTC_THRESHOLD:
                            ttc_danger += 1
                            win["ttcd"] += 1

                prev_a = prev_accel.get(vid, accel)
                jk = abs(accel - prev_a) / dt
                jerk_sum += jk
                jerk_count += 1
                win["jk"] += jk
                win["jk_n"] += 1
                prev_accel[vid] = accel

            if step % snap_every_steps == 0:
                fleet_speed_kmh = (sum(speeds_now) / len(speeds_now) * 3.6) if speeds_now else 0.0
                timeseries.append({
                    "sim_time_s": round(t, 2),
                    "config": preset["name"],
                    "cf_model": "IDM",
                    "av_pct": int(av_pct * 100),
                    "n_active": len(speeds_now),
                    "fleet_speed_kmh": round(fleet_speed_kmh, 2),
                    "spacing_mse_window": round(win["sq"] / win["sq_n"], 4) if win["sq_n"] else 0.0,
                    "collision_count": int(round(collision_count)),
                    "jerk_window": round(win["jk"] / win["jk_n"], 6) if win["jk_n"] else 0.0,
                    "ttc_pct_window": round(win["ttcd"] / win["ttct"] * 100, 4) if win["ttct"] else 0.0,
                })
                win = {"sq": 0.0, "sq_n": 0, "jk": 0.0, "jk_n": 0, "ttcd": 0, "ttct": 0}

            step += 1
    finally:
        traci.close()

    summary = {
        "source":           "SUMO/TraCI",
        "config":           preset["name"],
        "cf_model":         "IDM",
        "av_pct":           int(av_pct * 100),
        "sim_time_s":       round(t, 1),
        "n_vehicles":       len(n_veh_seen),
        "n_active":         len(n_veh_seen) - len(already_crashed),
        "n_crashed":        len(already_crashed),
        "spacing_mse_m2":   round(spacing_sq_sum / spacing_count, 4) if spacing_count else 0.0,
        "collision_count":  int(round(collision_count)),
        "jerk_mean_ms3":    round(jerk_sum / jerk_count, 6) if jerk_count else 0.0,
        "ttc_exposure_pct": round(ttc_danger / ttc_total * 100, 4) if ttc_total else 0.0,
    }
    return summary, timeseries


def sumo_trajectories_to_follownet(trajectories: list[dict], dt: float = 0.05) -> dict:
    """
    Compute FollowNet metrics from SUMO trajectory data.
    Returns a summary dict compatible with sim/metrics.py output format.
    Note: spacing MSE from SUMO uses SUMO's own gap reporting.
    """
    if not trajectories:
        return {}

    speeds   = [r["speed"] for r in trajectories]
    accels   = [r["accel"] for r in trajectories]
    t_max    = max(r["t"]   for r in trajectories)
    n_veh    = len(set(r["vid"] for r in trajectories))

    # Jerk = |da/dt|
    vid_accels = {}
    for r in trajectories:
        vid_accels.setdefault(r["vid"], []).append(r["accel"])

    jerks = []
    for vid, acc_seq in vid_accels.items():
        for i in range(1, len(acc_seq)):
            jerks.append(abs(acc_seq[i] - acc_seq[i-1]) / dt)

    mean_jerk = sum(jerks) / len(jerks) if jerks else 0.0

    return {
        "source":         "SUMO/TraCI",
        "sim_time_s":     round(t_max, 1),
        "n_vehicles":     n_veh,
        "mean_speed_kmh": round(sum(speeds) / len(speeds) * 3.6, 2),
        "jerk_mean_ms3":  round(mean_jerk, 6),
        "note":           "Spacing MSE and TTC require leader-follower pairing from TraCI (see run_sumo_full_metrics).",
    }
