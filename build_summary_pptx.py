"""
build_summary_pptx.py
EGH490-1 -- Tim Nguyen N11505222

One-off script to build a simple summary PowerPoint of the whole project:
baseline measurements, models used, how they were configured, how the
codebase is set up, and the key findings/analysis. Plain bullet-point and
table slides, no design effort requested. Every analytical slide carries a
"Source:" line pointing at the exact file(s) the numbers came from.

Usage: python build_summary_pptx.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

TITLE_LAYOUT = prs.slide_layouts[0]
BULLET_LAYOUT = prs.slide_layouts[1]
BLANK_LAYOUT = prs.slide_layouts[6]


def add_bullet_slide(title, bullets, source=None, font_size=18):
    slide = prs.slides.add_slide(BULLET_LAYOUT)
    slide.shapes.title.text = title
    body = slide.placeholders[1].text_frame
    body.word_wrap = True
    items = list(bullets)
    if source:
        items.append(("Source: " + source, 0))
    first = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        if first:
            p = body.paragraphs[0]
            first = False
        else:
            p = body.add_paragraph()
        p.text = text
        p.level = level
        is_source_line = source and i == len(items) - 1
        p.font.size = Pt(13 if is_source_line else (font_size if level == 0 else font_size - 2))
        if is_source_line:
            p.font.italic = True
    return slide


def add_table_slide(title, headers, rows, source=None, col_widths=None, note=None):
    slide = prs.slides.add_slide(BLANK_LAYOUT)
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.8))
    tf = tb.text_frame
    tf.text = title
    tf.paragraphs[0].font.size = Pt(28)
    tf.paragraphs[0].font.bold = True

    n_rows = len(rows) + 1
    n_cols = len(headers)
    top = Inches(1.15)
    height = Inches(min(5.3, 0.5 * n_rows))
    left = Inches(0.5)
    width = Inches(12.3)
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table

    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(12.3 * w / total)

    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = str(h)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(14)

    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            cell.text_frame.paragraphs[0].font.size = Pt(13)

    y = top + height + Inches(0.15)
    if note:
        nb = slide.shapes.add_textbox(Inches(0.5), y, Inches(12.3), Inches(0.6))
        p = nb.text_frame.paragraphs[0]
        p.text = note
        p.font.size = Pt(14)
        y = y + Inches(0.5)
    if source:
        sb = slide.shapes.add_textbox(Inches(0.5), y, Inches(12.3), Inches(0.5))
        p = sb.text_frame.paragraphs[0]
        p.text = "Source: " + source
        p.font.size = Pt(13)
        p.font.italic = True
    return slide


# 1. Title
slide = prs.slides.add_slide(TITLE_LAYOUT)
slide.shapes.title.text = "LLM-Based Decision-Making for an Automated Vehicle"
slide.placeholders[1].text = (
    "EGH490-1 \u2014 Tim Nguyen N11505222\n"
    "Supervisor: Dr Sebastien Glaser, QUT\n"
    "Project summary: findings, methodology, and analysis"
)

# 2. Research question
add_bullet_slide("Research Question", [
    "To what extent can LLM-based simulation tools accurately model car-following behaviour in mixed AV/HDV traffic?",
    "Does reduced technical complexity for non-experts (casual prompting vs. technical prompting vs. an expert's own calibration) cost simulation accuracy or safety?",
    "Original Week 12/13 report answered this with a single MATLAB run at 5\u20138 trials/condition, one car-following model (IDM), and one LLM config.",
    "This project re-built the simulator in Python, fixed real translation bugs, and re-ran the whole study at up to 88\u00d7 the original sample size, across 3 car-following models, 2 LLM sources, and a real SUMO validation.",
], source="Original EGH490-1 Week 12/13 report (MATLAB study, provided project brief).")

# 3. Codebase setup
add_bullet_slide("Codebase Setup", [
    "sim/ package: the core simulation engine",
    ("physics.py \u2014 IDM, LSTM-Approx, and DDPG-Approx car-following models", 1),
    ("configs.py \u2014 all IDM parameter presets (Expert, LLM, Ollama, ablation variants \u2014 25+ configs)", 1),
    ("runner.py \u2014 runs one simulation trial, returns spacing MSE / jerk / TTC / collisions", 1),
    "chatsumo/llm_backend.py \u2014 LLM parameter-elicitation calls (Claude API, Ollama local)",
    "run_report_replication.py \u2014 main experiment harness (baseline/density/speed/interaction sweeps)",
    "run_ablation.py, run_ddpg_profile_ablation.py, run_bcomf_ablation.py \u2014 targeted causal ablations",
    "run_sumo_battery.py \u2014 real SUMO/TraCI validation harness",
    "aggregate_results.py + analyze_significance.py \u2014 statistical pipeline (pooling, Welch's t-test, BH-FDR correction)",
    "app.py \u2014 Streamlit UI for interactively running/visualising simulations",
], source="Project codebase, C:\\QUT\\EGH490\\Python 490\\ (file/folder names as listed).")

# 4. Models / engines used
add_bullet_slide("Models & Engines Used", [
    "Car-following models (all implemented in the Python engine):",
    ("IDM \u2014 Intelligent Driver Model, the classical continuous car-following formula", 1),
    ("LSTM-Approx \u2014 a learned-behaviour approximation", 1),
    ("DDPG-Approx \u2014 a reinforcement-learning-style approximation with discontinuous, threshold-branching acceleration policy", 1),
    "Real SUMO (via TraCI) \u2014 industry-standard traffic simulator, IDM only (no DDPG analogue exists)",
    "LLM sources used to elicit driving parameters:",
    ("Claude (claude-sonnet-4-6) \u2014 via direct API calls", 1),
    ("Ollama / Llama 3.1 \u2014 run locally, matching the original ChatSUMO paper's backend", 1),
], source="sim/physics.py (car-following models); chatsumo/sumo_runner.py (SUMO/TraCI integration); chatsumo/llm_backend.py (LLM backends).")

# 5. How LLM configs were elicited/configured
add_bullet_slide("How Driver Configs Were Elicited", [
    "Every config defines 4 driver profiles: ACC (autonomous/ACC-controlled), CAUTIOUS, NORMAL, SPORTY",
    "Each profile has 7 IDM parameters: v0_frac (desired speed), T (headway), a_max, b_comf (comfortable braking), s0 (min gap), delta, threshold (lane-change incentive)",
    "Three distinct elicitation channels tested:",
    ("Plain conversational \u2014 no system prompt, matches how the original report's config was actually made (a human just chatting with Claude)", 1),
    ("ChatSUMO structured \u2014 a JSON-forcing system prompt, the same pipeline used for every Ollama call", 1),
    ("Prompt style: casual (\"configure a motorway simulation...\") vs. technical (explicit ranges/constraints)", 1),
    "Scenario framing tested on top of prompt style: generic motorway vs. \"quiet\" (light traffic) vs. \"rush hour\" (heavy, aggressive, tailgating)",
    "Every LLM config used AS ELICITED, with no expert correction \u2014 the whole point is testing what a non-expert gets by just asking",
], source="chatsumo/llm_backend.py (SYSTEM_PROMPT, elicit_idm_params_claude / _plain / _ollama); sim/configs.py (stored elicitation results + descriptions).")

# 6. Baseline measurements -- TABLE
add_table_slide(
    "Baseline: Expert-Calibrated Parameters",
    ["Profile", "v0_frac", "T (s)", "a_max", "b_comf", "s0 (m)", "delta", "threshold"],
    [
        ["ACC",      "1.00", "1.6", "2.0", "2.0", "3.0", "4", "0.50"],
        ["CAUTIOUS", "0.70", "2.2", "1.2", "1.0", "4.0", "2", "0.80"],
        ["NORMAL",   "1.00", "1.5", "2.5", "1.7", "3.0", "4", "0.40"],
        ["SPORTY",   "1.20", "0.8", "4.0", "3.0", "2.0", "6", "0.10"],
    ],
    note=("\u00b15% per-vehicle IDM noise applied by default. Standard test conditions: N=25 vehicles, "
          "110 km/h speed limit, 3-lane motorway, AV penetration swept at 0% / 25% / 50%. A second "
          "expert baseline (SUMO's own built-in default IDM parameters) was also tested \u2014 this "
          "project's own calibration significantly outperformed it on spacing accuracy."),
    source="sim/configs.py (EXPERT dict) \u2014 calibrated against Treiber et al. 2000 (IDM) and the FollowNet benchmark table (Chen et al. 2023, Table 2).",
)

# 6b. Baseline RESULTS -- Expert-Calibrated's own simulated performance -- TABLE
add_table_slide(
    "Baseline Result: Expert-Calibrated Performance",
    ["Car-Following Model", "AV%", "Spacing MSE", "Jerk (m/s\u00b3)", "TTC Exposure (%)", "Collisions/trial"],
    [
        ["IDM",         "0%",  "1743.8", "0.634", "0.0291", "0.001"],
        ["IDM",         "25%", "1578.9", "0.551", "0.0155", "0.000"],
        ["IDM",         "50%", "1419.9", "0.330", "0.0050", "0.000"],
        ["LSTM-Approx", "0%",  "1124.2", "1.754", "0.0237", "0.000"],
        ["LSTM-Approx", "25%", "1114.5", "1.447", "0.0118", "0.000"],
        ["LSTM-Approx", "50%", "1245.5", "1.121", "0.0044", "0.000"],
        ["DDPG-Approx", "0%",  "433.0",  "24.59", "0.0661", "0.081"],
        ["DDPG-Approx", "25%", "331.0",  "22.31", "0.0397", "0.030"],
        ["DDPG-Approx", "50%", "195.8",  "17.41", "0.0174", "0.004"],
    ],
    note=("This is the anchor every LLM config is compared against. N=25 vehicles, 110 km/h, "
          "n=936 trials pooled per row (across every run at this condition, t_sim=600s), \u00b15% IDM "
          "noise. Note DDPG-Approx's much lower spacing MSE and higher jerk are inherent to that "
          "model's own driving style, not a defect \u2014 comparisons are always LLM-config-vs-Expert "
          "WITHIN the same car-following model, never across models."),
    source="results/master_summary.csv (config=Expert-Calibrated, group=baseline, n_veh=25, v_limit_kmh=110, t_sim=600).",
)

# 7. Statistical methodology
add_bullet_slide("Statistical Methodology", [
    "4 outcome metrics per trial: spacing MSE, jerk (m/s\u00b3), TTC exposure (%), collision count",
    "Welch's t-test (unequal variance) for spacing MSE / jerk / TTC, comparing each LLM config against Expert-Calibrated at matched conditions",
    "Poisson rate z-test for collisions (a rare-count metric, not well-suited to a t-test)",
    "Benjamini\u2013Hochberg (BH) false-discovery-rate correction applied within each config\u00d7metric family \u2014 a finding is only called significant at corrected q < 0.05",
    "Sample sizes ranged from n=8 trials (earliest, low-power checks) up to n=40\u201380 trials/condition and t_sim up to 600s in the final \"power-up\" rounds",
    "Every headline finding was independently re-derived from the raw CSVs at least once, not just trusted from a summary",
], source="analyze_significance.py (welch_test, poisson_rate_ztest, bh_fdr functions); aggregate_results.py (pooling across repeated runs).")

# 8. Finding: report reversal -- TABLE
add_table_slide(
    "Finding: The Original Report's Headline Result Reverses",
    ["Metric", "Report's Claim", "Re-run Result (n=436 Expert / n=76 LLM)", "p-value"],
    [
        ["Jerk vs. Expert", "+14% (worse)", "\u221221.9% (better)", "1.0e\u221214"],
        ["TTC exposure vs. Expert", "+92% (worse)", "\u221230.4% (better)", "3.9e\u221211"],
    ],
    note=("Report's own config (LLM-Generated) re-simulated at the report's exact baseline point: "
          "N=25, 110 km/h, AV=25%, IDM, t_sim=600s. Both results are the OPPOSITE sign from the "
          "report's claim, and the reversal held at every sample size tested on the way up. Likely "
          "cause: the report's own Figure 2 prose contradicts its stated numbers, and Table 3 carries "
          "an anomalous preliminary row superseded by the report's own later dataset."),
    source="results/significance_report.csv (config=LLM-Generated, cf_model=IDM, group=baseline, av_pct=25, t_sim=600) vs. the original Week 12/13 report, Figure 2 / Table 3.",
)

# 9. Finding: DDPG concentrates danger
add_bullet_slide("Finding: Danger Concentrates in the DDPG Model \u2014 Not IDM", [
    "The report only ever tested IDM. Under IDM, LLM configs generally look SAFER than Expert \u2014 which is why the reversal above happens.",
    "But under DDPG-Approx, real danger appears for SPECIFIC configs:",
    ("LLM-Generated: TTC +106% to +221% worse than Expert, collisions significantly higher (all AV%)", 1),
    ("Ollama-Casual: up to +764% worse TTC, collisions significantly higher at AV\u226525%", 1),
    ("Ollama-Technical: up to +145% worse TTC, collision spike at AV=50% (and later found: also at AV=0% for longer simulations \u2014 see later slide)", 1),
    "BUT two independently re-elicited configs from the IDENTICAL prompts (Claude-Casual-Run2, Claude-Technical-Run2) are SAFER than Expert under DDPG too",
    "Correct conclusion: DDPG concentrates danger for specific unlucky elicitation draws \u2014 not a general property of LLM-generated configs",
], source="results/significance_report.csv (cf_model=DDPG-Approx, group=baseline, t_sim=600, all 5 configs vs Expert-Calibrated).")

# 10. Finding: mechanistic cause isolated
add_bullet_slide("Finding: The Exact Cause \u2014 One Outlier Parameter", [
    "Controlled ablation: swapped ONE profile at a time between the dangerous LLM-Generated and the safe Claude-Casual-Run2 (same prompt, different draw)",
    "Swapping the whole SPORTY profile fully flips the config from dangerous to safe; swapping any other profile makes it WORSE",
    "Narrowed further: SPORTY.a_max alone (5.5 vs. 2.0, the single largest parameter gap anywhere in the study) reproduces almost the entire effect",
    "Why DDPG-specific: the DDPG approximation uses a_max in 3 places inside a discontinuous, threshold-branching acceleration policy \u2014 not as a smooth cap the way IDM does",
    "All 4 metrics corroborate: fixing a_max alone brings spacing MSE to within 0.5% of Expert's own value",
], source="run_ddpg_profile_ablation.py; results/ddpg_profile_ablation_SPORTY-a_max_LLM-Generated-vs-Run2_n20_2026-09-03_15-18-02/ddpg_profile_ablation_summary.csv; sim/physics.py (ddpg_approx_accel).")

# 11. Finding: MSE can mask danger -- TABLE
add_table_slide(
    "Finding: A Good-Looking Metric Can Hide Real Danger",
    ["Metric", "\u0394 vs. Expert", "p-value", "Reads as"],
    [
        ["Spacing MSE", "\u221247.9%", "4.9e\u221213", "\"More accurate\""],
        ["TTC exposure", "+201.7%", "3.4e\u22124", "Dramatically worse"],
        ["Collisions", "Significantly higher", "6.1e\u221210", "Dramatically worse"],
    ],
    note=("Ollama-Casual vs. Expert, DDPG-Approx, AV=25%, t_sim=600s. Same condition, same config: the "
          "metric that looks best is measuring the exact point where real safety is worst. Confirms \u2014 "
          "with an actual p-value \u2014 a caveat the original report only asserted qualitatively."),
    source="results/significance_report.csv (config=Ollama-Casual, cf_model=DDPG-Approx, av_pct=25, t_sim=600).",
)

# 12. Finding: Claude vs Ollama collision patterns
add_bullet_slide("Finding: Claude Drafts Are Mostly Collision-Free; Ollama Is Prompt-Dependent", [
    "8 independent Claude elicitations tested (2 prompt styles \u00d7 2 elicitation channels \u00d7 2 draws): 7 of 8 show ZERO collisions under DDPG, at every AV% \u2014 confirmed and strengthened at n=40",
    "Only the report's own original draw (with its outlier SPORTY.a_max) shows real collisions",
    "Ollama splits cleanly by prompt style at AV\u226525%: casual prompts \u2192 real collisions (0.30\u20130.40/trial), technical prompts \u2192 clean \u2014 replicates across independent draws",
    "IMPORTANT CAVEAT found in later re-testing: Ollama-Technical-Run2 at AV=0% shows collisions in HALF of all 40 trials when tested at t_sim=600s \u2014 completely invisible at the original shorter t_sim=180s",
    "This means collision risk can be time-cumulative \u2014 a short simulation window can systematically miss real danger. A similar (weaker, not-yet-significant) signal also showed up for ChatSUMO-Claude-Technical-Rush2 at longer duration.",
], source="results/master_summary.csv + results/significance_report.csv (configs: LLM-Generated, Claude-Casual/Technical-Run2/Run3, ChatSUMO-Claude-Casual/Technical, Ollama-Casual/Technical + Run2, cf_model=DDPG-Approx).")

# 13. Finding: ACC role-inversion ablation
add_bullet_slide("Finding: One Specific Bug Explains Most of Ollama-Casual's Danger", [
    "Ollama-Casual's ACC profile (the autonomous vehicle, described as keeping \"very safe gaps\") was elicited with v0_frac=0.5 \u2014 cruising at HALF the speed limit, a role-inversion bug",
    "Ablation: fixing ONLY this ACC error cuts collisions from 0.567\u21920.033/trial at AV=25% (p=0.00016)",
    "Surprising twist: fixing ALL FOUR profiles' speed-role errors together makes AV=0% SIGNIFICANTLY WORSE (0\u21920.800 collisions/trial, p<0.0001) \u2014 a genuine non-additive interaction: fixing one error exposes a second, previously-masked problem",
    "Lesson: LLM elicitation bugs don't add up simply \u2014 partial fixes can behave unpredictably",
], source="run_ablation.py; results/ablation_OllamaCasual-FixACC-FixAllV0_n30_2026-09-04_16-13-01/ablation_summary.csv.")

# 14. Finding: rush-hour / scenario conditioning
add_bullet_slide("Finding: Scenario Framing Changes Elicited Parameters \u2014 and Behaviour", [
    "Rush-hour framing (\"heavy, aggressive, tailgating\") made Ollama-Casual SAFER, not more dangerous \u2014 confirmed significant, both draws (TTC \u22120.10 to \u22120.13, p<0.005; collisions \u22120.53 to \u22120.65/trial, p<0.006)",
    "Mechanism: \"everyone's late and aggressive\" framing pushed desired-speed values UP across every profile, including the buggy low ones \u2014 accidentally masking the role-inversion bug",
    "Confirmed even at scenario-matched HIGH density (N=40) \u2014 not just an artifact of testing too gently",
    "Claude vs. Ollama contextual adaptation, quiet vs. rush-hour prompts (n=20/scenario): Claude adapts NEARLY EVERY parameter (20 of 27 tested) in the sensible direction; Ollama moves almost none (7 of 24) \u2014 including headway (T), the single most literally-cued parameter in the prompt",
], source="results/significance_report.csv (Ollama-Casual-Rush1/2 vs Ollama-Casual); results/claude_scenario_elicitations_v2.json + results/claude_scenario_significance_v2.csv (scenario-adaptation test); results/scenario_elicitation_transcripts.json (Ollama scenario data).")

# 15. Finding: elicitation bias + causal test -- TABLE
add_table_slide(
    "Finding: A Near-Universal Elicitation Bias (and a Surprising Causal Test)",
    ["Config", "AV%", "Collisions: Before \u2192 After Fix", "Raw p", "q (BH-FDR)"],
    [
        ["LLM-Generated",  "0%",  "0.25 \u2192 0.85 (worse)", "0.011", "0.126"],
        ["Ollama-Casual",  "0%",  "0.00 \u2192 0.20 (worse)", "0.046", "0.273"],
        ["LLM-Generated",  "50%", "0.15 \u2192 0.00 (better)", "0.083", "0.333"],
    ],
    note=("Across 25 non-Expert configs, NORMAL.b_comf (comfortable braking) runs +36.7% higher than "
          "Expert's value in 23 of 25 \u2014 a structural, model-agnostic bias. \"Fixing\" it (reverting to "
          "Expert's 1.7) on 2 independent dangerous configs made AV=0% WORSE both times, DDPG-only "
          "(zero effect under IDM). DIRECTIONAL and consistent, but does NOT survive BH-FDR correction "
          "at this sample size \u2014 disclosed as suggestive, not proven. Possible interpretation: the "
          "\"biased\" parameter may be accidentally compensating for other elicitation errors."),
    source="results/signed_deviation_table.csv (the 23/25-config bias); run_bcomf_ablation.py + results/bcomf_ablation_2026-09-08_16-54-47/bcomf_ablation_summary.csv (the causal test).",
)

# 16. Finding: density/interaction effects
add_bullet_slide("Finding: Density and AV-Penetration Interact in Non-Obvious Ways", [
    "For the report's own config, higher traffic density does NOT amplify the jerk gap \u2014 it REVERSES it: safer than Expert at N=10 (\u221224 to \u221231%), worse at N=40 (+18 to +22%), all significant",
    "Cause: jerk scales with a_max, and higher density means more acceleration/deceleration events \u2014 compounds the same SPORTY.a_max outlier",
    "The AV-penetration-widens-the-gap effect is real, but specifically a high-density (N=40) phenomenon \u2014 does not appear at N=10 or N=25",
    "Ollama-Casual (whose a_max is much closer to Expert's) shows only a narrowing, not a reversal, over the same density range \u2014 confirms this is config-specific, not a general density effect",
], source="results/significance_report.csv (group=density and group=density_x_av, config=LLM-Generated / Ollama-Casual-Rush1/2, cf_model=IDM).")

# 17. Finding: real SUMO validation
add_bullet_slide("Validation: Real SUMO Engine + a Second Expert Baseline", [
    "Built full metrics (spacing MSE, TTC, collisions) into real SUMO via TraCI \u2014 previously only jerk was computed",
    "3 real bugs found and fixed to get a trustworthy reading: geometric cornering-speed cap, non-looping routes emptying the fleet, and a FIXED random seed (silently returning byte-identical \"trials\")",
    "Real SUMO/IDM shows no significant danger differentiation for the pair central to the mechanistic finding \u2014 consistent with the Python engine's own IDM result (the danger is DDPG-specific; SUMO has no DDPG analogue)",
    "This project's own Expert-Calibrated significantly outperforms SUMO's own built-in default IDM parameters on spacing accuracy (+27% worse MSE for SUMO's defaults, p=0.006) \u2014 a genuine second expert baseline, zero LLM involved",
    "Open limitation: SUMO's fleet speed decays over a run (110\u219277 km/h) in a way the Python engine doesn't \u2014 not fully resolved, comparisons kept to moderate duration to limit its effect",
], source="chatsumo/sumo_runner.py (run_sumo_full_metrics); run_sumo_battery.py; results/sumo_battery_Expert-LLMGenerated-ClaudeCasualRun2-SUMODefaultIDM_n20_t90_2026-09-04_19-37-09/sumo_battery_summary.csv.")

# 18. Limitations
add_bullet_slide("Limitations (Disclosed, Not Swept Under the Rug)", [
    "SUMO's fleet-speed decay vs. the Python engine's stability \u2014 unresolved, likely a lane-changing model difference",
    "Concurrent SUMO runs are unsafe (TraCI connection collides) \u2014 all SUMO batteries run sequentially",
    "The NORMAL.b_comf bias is confirmed as a correlation across 25 configs; the causal ablation is directional but not statistically proven at current sample size",
    "Several rush-hour/quiet-scenario configs remain at lower power (n=15\u201320) than the headline configs (n=40\u201380) \u2014 results for these are consistent in direction but have wider confidence intervals",
    "Collision rate is duration-dependent \u2014 a config that looks \"clean\" at a short simulation time is not guaranteed to stay clean at a longer one",
])

# 19. Conclusion
add_bullet_slide("Conclusions", [
    "The original report's headline safety claim does not replicate under IDM at high power \u2014 the true effect runs the opposite direction",
    "But the underlying concern (LLM-elicited configs can be dangerous) is real \u2014 it's just concentrated in a different car-following model (DDPG) and traceable to specific, identifiable parameter errors, not a general property of LLM elicitation",
    "Model identity (Claude vs. Ollama) is not the main driver of accuracy \u2014 prompt style, elicitation channel, and plain run-to-run variance matter more",
    "Claude shows much broader, more reliable contextual adaptation to scenario framing than Ollama/Llama 3.1",
    "A single outlier parameter (SPORTY.a_max) can flip an entire config from safe to dangerous \u2014 LLM elicitation for safety-critical parameters is fragile in a way that's hard to detect without dedicated testing",
    "Real SUMO validation and a second independent expert baseline both support the Python engine's conclusions where they can be compared",
])

# 20. Data & code sources -- consolidated reference
add_bullet_slide("Data & Code Sources (Consolidated Reference)", [
    "Raw results: results/master_summary.csv (every simulation run, pooled), results/significance_report.csv (all significance tests), results/signed_deviation_table.csv (per-parameter bias table)",
    "Ablation results: results/ablation_OllamaCasual-*/ablation_summary.csv, results/ddpg_profile_ablation_*/ddpg_profile_ablation_summary.csv, results/bcomf_ablation_*/bcomf_ablation_summary.csv",
    "SUMO validation: results/sumo_battery_*/sumo_battery_summary.csv",
    "LLM elicitation transcripts: results/claude_scenario_elicitations_v2.json, results/scenario_elicitation_transcripts.json, results/claude_battery_elicitations.json, results/rush_hour_battery_elicitations.json",
    "Key scripts: sim/physics.py, sim/configs.py, sim/runner.py, chatsumo/llm_backend.py, chatsumo/sumo_runner.py, run_report_replication.py, run_ablation.py, run_ddpg_profile_ablation.py, run_bcomf_ablation.py, run_sumo_battery.py, aggregate_results.py, analyze_significance.py, build_signed_deviation_tables.py",
    "Full narrative write-up (all findings, every run's exact conditions): \"The Elicitation Ledger\" \u2014 https://claude.ai/code/artifact/f623ac49-14a6-4a97-9fb9-c579f3e92634",
    "Original baseline: EGH490-1 Week 12/13 report (MATLAB/FollowNet methodology, provided project brief)",
])

out_path = "EGH490_Findings_Summary.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
