"""
chatsumo/llm_backend.py
EGH490-1 — Tim Nguyen N11505222

LLM backend for ChatSUMO-inspired parameter elicitation.
Supports:
  1. Claude API (claude-sonnet-4-6) — primary backend
  2. Ollama / Llama 3.1          — local fallback (matches original ChatSUMO)

Takes a natural language scenario description and returns IDM parameters
for each driver profile (ACC, CAUTIOUS, NORMAL, SPORTY).
"""

import json
import re
import time
import requests
from typing import Optional


SYSTEM_PROMPT = """You are an expert traffic engineering assistant specialising in
the Intelligent Driver Model (IDM) for microscopic traffic simulation.

When given a natural language description of a traffic scenario, you extract
IDM car-following parameters for the requested driver profiles.

The IDM parameters are:
  v0_frac   : desired speed as fraction of the speed limit (e.g. 0.70 = 70% of limit)
  T         : safe time headway in seconds (typical: 0.6–2.5 s)
  a_max     : maximum acceleration in m/s² (typical: 1.0–5.0 m/s²)
  b_comf    : comfortable braking deceleration in m/s² (typical: 0.8–4.0 m/s²)
  s0        : minimum standstill gap in metres (typical: 1.5–6.0 m)
  delta     : acceleration exponent (typical: 2–6; higher = more aggressive)
  threshold : MOBIL lane-change incentive threshold in m/s² (typical: 0.1–1.0)

Always respond with ONLY a JSON object in this exact format, no other text:
{
  "config_name": "short name for this configuration",
  "speed_limit_kmh": <number>,
  "variability": <0.0–0.10>,
  "profiles": {
    "ACC":      {"v0_frac": <num>, "T": <num>, "a_max": <num>, "b_comf": <num>, "s0": <num>, "delta": <num>, "threshold": <num>},
    "CAUTIOUS": {"v0_frac": <num>, "T": <num>, "a_max": <num>, "b_comf": <num>, "s0": <num>, "delta": <num>, "threshold": <num>},
    "NORMAL":   {"v0_frac": <num>, "T": <num>, "a_max": <num>, "b_comf": <num>, "s0": <num>, "delta": <num>, "threshold": <num>},
    "SPORTY":   {"v0_frac": <num>, "T": <num>, "a_max": <num>, "b_comf": <num>, "s0": <num>, "delta": <num>, "threshold": <num>}
  }
}"""


def elicit_idm_params_claude(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Call Claude API to elicit IDM parameters from a natural language prompt.

    Parameters
    ----------
    prompt  : natural language scenario description
    api_key : Anthropic API key (reads ANTHROPIC_API_KEY env var if None)
    model   : Claude model string

    Returns
    -------
    dict with keys: config_name, speed_limit_kmh, variability, profiles
    """
    import os
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment "
            "variable or pass api_key= to this function."
        )

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 800,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip()
    return _parse_json_response(text, prompt, "Claude")


def elicit_idm_params_claude_plain(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    Call Claude with NO system prompt and no JSON-format instruction — a plain
    natural-language ask, matching how LLM-Generated/CLAUDE_CASUAL_RUN2/
    CLAUDE_TECHNICAL_RUN2 in sim/configs.py were actually elicited (a human
    chatting with Claude and manually transcribing a prose/table response).
    elicit_idm_params_claude() always sends the ChatSUMO JSON-forcing system
    prompt, which is a DIFFERENT elicitation channel — this function exists to
    reproduce the "plain" channel programmatically instead of requiring manual
    copy-paste from a separate conversation.

    Returns the raw response TEXT (unstructured — use extract_idm_json_from_text
    separately to parse it, so the actual elicited values come from this
    single-shot, no-system-prompt call and nothing else).
    """
    import os
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY or pass api_key=.")

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        # Unlike elicit_idm_params_claude() (JSON-only, fits in 800 tokens),
        # a plain no-system-prompt ask tends to draw much longer responses
        # (justification prose, formulas, per-profile tables) — 800 truncated
        # mid-response before profiles 2-4's tables appeared (observed 2026-09-04).
        "max_tokens": 3000,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=payload, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def extract_idm_json_from_text(
    raw_text: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Mechanical extraction, NOT re-elicitation: given an already-generated
    natural-language response (from elicit_idm_params_claude_plain), transcribe
    the IDM values it already contains into the same structured JSON schema
    SYSTEM_PROMPT defines. This call is not asked to judge or generate new
    parameter values — only to transcribe numbers already present in raw_text —
    so it doesn't change what was actually elicited.
    """
    import os
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY or pass api_key=.")

    extraction_system_prompt = (
        "You are a data-transcription tool. You will be given a block of text "
        "containing IDM car-following parameters for driver profiles ACC, "
        "CAUTIOUS, NORMAL, and SPORTY. Transcribe the EXACT numeric values "
        "already present in the text into this JSON schema — do not invent, "
        "round, or adjust any value; if a threshold/delta value is not present "
        "in the text for a profile, use 0.5 for threshold and 4 for delta as "
        "literal placeholders (do not guess a 'sensible' value):\n" + SYSTEM_PROMPT.split("\n\n", 1)[1]
    )
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 800,
        "system": extraction_system_prompt,
        "messages": [{"role": "user", "content": raw_text}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=payload, timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip()
    return _parse_json_response(text, raw_text, "Claude-plain-extraction")


def elicit_idm_params_claude_plain_full(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Full pipeline for the "plain conversational Claude" elicitation channel:
    elicit (no system prompt) -> extract (mechanical transcription only).
    Returns a dict like elicit_idm_params()'s output, PLUS raw_response so the
    original unstructured text is preserved for the transcript record.
    """
    raw_text = elicit_idm_params_claude_plain(prompt, api_key, model)
    data = extract_idm_json_from_text(raw_text, api_key, model)
    data["raw_response"] = raw_text
    data["original_prompt"] = prompt
    data["source"] = "Claude-plain"
    data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data.setdefault("config_name", "LLM-Generated (Claude-plain)")
    return data


def elicit_idm_params_ollama(
    prompt: str,
    model: str = "llama3.1",
    host: str = "http://localhost:11434",
) -> dict:
    """
    Call a locally-running Ollama model (e.g. Llama 3.1) to elicit IDM parameters.
    This replicates the original ChatSUMO backend (Li et al. 2024).

    Parameters
    ----------
    prompt : natural language scenario description
    model  : Ollama model name (default: llama3.1)
    host   : Ollama server URL

    Returns
    -------
    dict with keys: config_name, speed_limit_kmh, variability, profiles
    """
    payload = {
        "model": model,
        "prompt": SYSTEM_PROMPT + "\n\nUser: " + prompt + "\n\nAssistant:",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 800},
    }
    # Local CPU inference (esp. the first call, which also has to load the
    # model into memory) can take well over a minute — Claude's API timeout
    # of a few seconds doesn't apply here.
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    return _parse_json_response(text, prompt, f"Ollama/{model}")


def elicit_idm_params(
    prompt: str,
    backend: str = "claude",
    api_key: Optional[str] = None,
    ollama_model: str = "llama3.1",
    ollama_host: str = "http://localhost:11434",
) -> dict:
    """
    Unified entry point. backend = 'claude' or 'ollama'.
    """
    if backend == "ollama":
        return elicit_idm_params_ollama(prompt, ollama_model, ollama_host)
    else:
        return elicit_idm_params_claude(prompt, api_key)


def _parse_json_response(text: str, original_prompt: str, source: str) -> dict:
    """Extract and validate the JSON config from LLM response text."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract just the JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
        else:
            raise ValueError(f"LLM ({source}) did not return valid JSON.\nResponse:\n{text}")

    # Validate structure
    required_profiles = {"ACC", "CAUTIOUS", "NORMAL", "SPORTY"}
    required_keys     = {"v0_frac", "T", "a_max", "b_comf", "s0", "delta", "threshold"}

    if "profiles" not in data:
        raise ValueError(f"LLM response missing 'profiles' key.\nResponse:\n{text}")

    for profile in required_profiles:
        if profile not in data["profiles"]:
            raise ValueError(f"LLM response missing profile '{profile}'.")
        missing = required_keys - set(data["profiles"][profile].keys())
        if missing:
            raise ValueError(f"Profile '{profile}' missing keys: {missing}")

    # Add metadata
    data["source"]          = source
    data["original_prompt"] = original_prompt
    data["timestamp"]       = time.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure required top-level fields have defaults
    data.setdefault("config_name", f"LLM-Generated ({source})")
    data.setdefault("speed_limit_kmh", 110.0)
    data.setdefault("variability", 0.0)

    return data


def build_preset_from_llm(llm_response: dict) -> dict:
    """
    Convert LLM elicitation result into a preset dict compatible with sim/configs.py.
    """
    return {
        "name":        llm_response.get("config_name", "LLM-Generated"),
        "description": (
            f"Generated by {llm_response.get('source', 'LLM')} "
            f"at {llm_response.get('timestamp', '')}.\n"
            f"Prompt: {llm_response.get('original_prompt', '')[:200]}"
        ),
        "variability": float(llm_response.get("variability", 0.0)),
        "profiles":    llm_response["profiles"],
    }


def compare_to_expert(llm_preset: dict, expert_preset: dict) -> list[dict]:
    """
    Compute per-profile, per-parameter deviation of LLM preset vs expert.
    Returns a list of deviation records for display.
    """
    deviations = []
    for profile in ["ACC", "CAUTIOUS", "NORMAL", "SPORTY"]:
        ep = expert_preset["profiles"][profile]
        lp = llm_preset["profiles"][profile]
        for param in ["v0_frac", "T", "a_max", "b_comf", "s0", "delta"]:
            ev = ep[param]
            lv = lp[param]
            pct = (lv - ev) / ev * 100 if ev != 0 else 0
            deviations.append({
                "profile": profile,
                "parameter": param,
                "expert": round(ev, 3),
                "llm":    round(lv, 3),
                "diff_pct": round(pct, 1),
                "direction": "▲ higher" if pct > 5 else ("▼ lower" if pct < -5 else "≈ similar"),
            })
    return deviations
