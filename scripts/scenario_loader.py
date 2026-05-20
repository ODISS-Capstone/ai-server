"""Load scenario notes without importing the full validation harness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def extract_json_from_markdown(raw: str) -> dict[str, Any] | list[Any]:
    match = re.search(r"```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", raw, re.IGNORECASE)
    if not match:
        raise ValueError("Markdown scenario notes must include a fenced ```json block.")
    return json.loads(match.group(1))


def normalize_scenario(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario #{index} must be an object.")
    scenario_id = scenario.get("id") or f"custom_scenario_{index}"
    steps = scenario.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Scenario '{scenario_id}' must define a non-empty steps list.")

    normalized_steps = []
    for step_index, step in enumerate(steps, start=1):
        if isinstance(step, str):
            step = {"text": step}
        if not isinstance(step, dict):
            raise ValueError(f"Scenario '{scenario_id}' step #{step_index} must be a string or object.")
        if "text" not in step:
            raise ValueError(f"Scenario '{scenario_id}' step #{step_index} is missing 'text'.")
        normalized_steps.append(
            {
                "id": step.get("id", f"step_{step_index}"),
                "text": str(step.get("text", "")),
                "expected_mode": step.get("expected_mode", "memory_only"),
                "expected_intent": step.get("expected_intent"),
                "expected_terms": list(step.get("expected_terms") or []),
            }
        )

    return {
        "id": str(scenario_id),
        "speaker_id": str(scenario.get("speaker_id") or f"custom_{scenario_id}"),
        "runner": scenario.get("runner", "orchestrator"),
        "seed_medications": list(scenario.get("seed_medications") or []),
        "steps": normalized_steps,
    }


def load_scenarios_from_file(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if path.suffix.lower() == ".json" else extract_json_from_markdown(raw)
    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    elif isinstance(data, dict):
        scenarios = [data]
    elif isinstance(data, list):
        scenarios = data
    else:
        raise ValueError("Scenario file must contain a scenario object, a list, or {'scenarios': [...]}.")
    return [normalize_scenario(scenario, index) for index, scenario in enumerate(scenarios, start=1)]
