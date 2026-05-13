"""Selective DUR endpoint routing tests."""
from __future__ import annotations

import asyncio

from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.schemas.engine_contracts import ReasoningTask
from app.tools import dur_api


def run(coro):
    return asyncio.run(coro)


def test_select_dur_endpoint_keys_uses_only_relevant_checks():
    combo = dur_api.select_dur_endpoint_keys(
        query_text="와파린이랑 아스피린 같이 먹어도 돼?",
        patient_age=52,
        medication_count=2,
    )
    assert combo == ["dur_product_info", "combination_contraindication"]

    overdose = dur_api.select_dur_endpoint_keys(
        query_text="혈압약 두 번 먹으면 더 빨리 좋아져?",
        patient_age=52,
        medication_count=1,
    )
    assert overdose == ["dur_product_info", "dosage_caution"]

    older_user = dur_api.select_dur_endpoint_keys(
        query_text="이 약 주의할 점 알려줘",
        patient_age=72,
        medication_count=1,
    )
    assert older_user == ["dur_product_info", "elderly_caution"]


def test_check_dur_for_prescription_defaults_to_basic_item_info(monkeypatch):
    called: list[str] = []

    async def fake_call(endpoint_key, **kwargs):
        called.append(endpoint_key)
        return {"success": True, "items": [], "endpoint": endpoint_key}

    monkeypatch.setattr(dur_api, "call_dur_api", fake_call)

    result = run(dur_api.check_dur_for_prescription([{"name": "혈압약A"}, {"name": "당뇨약B"}]))

    assert [row["medication"] for row in result] == ["혈압약A", "당뇨약B"]
    assert called == ["dur_product_info", "dur_product_info"]


def test_reasoning_dur_check_passes_selected_endpoint_keys(monkeypatch):
    captured: list[tuple[str, tuple[str, ...]]] = []

    async def fake_check_dur_for_prescription(medications, endpoint_keys=None):
        keys = tuple(endpoint_keys or ())
        rows = []
        for medication in medications:
            name = medication["name"]
            captured.append((name, keys))
            rows.append({
                "medication": name,
                "dur": {key: {"success": True, "items": []} for key in keys},
            })
        return rows

    monkeypatch.setattr(dur_api, "check_dur_for_prescription", fake_check_dur_for_prescription)
    engine = ReasoningEngine(MemoryEngine(), LLMJudgeEngine())

    result = run(
        engine.execute_tasks(
            text="혈압약 두 번 먹으면 더 빨리 좋아져?",
            intent="medication_query",
            context={"user_profile": {"age": "52"}},
            tasks=[ReasoningTask(type="dur_check", priority=1)],
        )
    )

    assert captured
    assert captured[0][1] == ("dur_product_info", "dosage_caution")
    assert result["task_results"]["dur_endpoint_keys"] == ["dur_product_info", "dosage_caution"]
