from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.services.assistant_intent_classifier import AssistantIntentClassifier


FIXTURE = Path("tests/fixtures/odiss_utterance_dataset.csv")
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "emergency": 3}


def _fixture_rows() -> list[dict[str, str]]:
    with FIXTURE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for raw in reader:
            rows.append(
                {
                    "category": raw[0],
                    "utterance": raw[1],
                    "intent": raw[2],
                    "route_label_expected": raw[header.index("route_label_expected")],
                    "engine_scope_expected": raw[header.index("engine_scope_expected")],
                    "risk_level_expected": raw[header.index("risk_level_expected")],
                    "fast_path_expected": raw[header.index("fast_path_expected")],
                    "ui_action_expected": raw[header.index("ui_action_expected")],
                    "db_write_expected": raw[header.index("db_write_expected")],
                    "active_flow_required": raw[header.index("active_flow_required")],
                }
            )
        return rows


@pytest.mark.parametrize("row", _fixture_rows(), ids=lambda row: row["utterance"][:40])
def test_utterance_route_contract_fixture(row: dict[str, str]) -> None:
    classifier = AssistantIntentClassifier()
    decision = classifier.classify(
        row["utterance"],
        active_flow=row["active_flow_required"] or "none",
        active_session=True,
        odiss_directed=True,
    )

    assert decision.engine_scope != "ignored"
    assert decision.route_label == row["route_label_expected"]
    assert decision.engine_scope == row["engine_scope_expected"]
    assert decision.risk_level == row["risk_level_expected"]
    assert decision.fast_path == row["fast_path_expected"]
    assert decision.ui_action == row["ui_action_expected"]
    assert ("true" if decision.db_write_expected else "false") == row["db_write_expected"]
    if decision.route_label == "unclear":
        assert not row["active_flow_required"]


@pytest.mark.parametrize(
    ("text", "route_label", "risk_level"),
    [
        ("가슴이 꽉 막힌 것 같아", "emergency", "emergency"),
        ("숨이 잘 안 쉬어져", "emergency", "emergency"),
        ("혀가 붓고 말이 잘 안 돼", "emergency", "emergency"),
        ("타이레놀 4개 먹어도 돼", "medication_safety", "high"),
        ("혈압약 또 먹어도 돼", "medication_safety", "high"),
        ("남편 약을 내가 먹어도 되나", "third_party_medication", "high"),
    ],
)
def test_global_safety_precheck_beats_active_flow(text: str, route_label: str, risk_level: str) -> None:
    classifier = AssistantIntentClassifier()
    for active_flow in ("ocr_confirm", "ocr_camera", "reminder", "identity", "medication_guidance"):
        decision = classifier.classify(
            text,
            active_flow=active_flow,
            active_session=True,
            odiss_directed=True,
        )
        assert decision.route_label == route_label
        assert decision.engine_scope == "safety"
        assert RISK_ORDER[decision.risk_level] >= RISK_ORDER[risk_level]
        assert decision.db_write_expected is False


def test_inactive_background_without_wake_is_ignored() -> None:
    classifier = AssistantIntentClassifier()
    decision = classifier.classify(
        "티비 소리야 약 광고 나온 거야",
        active_flow="none",
        active_session=False,
        client_context={"active_session": False, "voice_armed": False},
    )

    assert decision.route_label == "ignored_background"
    assert decision.engine_scope == "ignored"


def test_wakeless_emergency_requires_high_confidence_setting() -> None:
    off = AssistantIntentClassifier(allow_wakeless_emergency=False)
    on = AssistantIntentClassifier(allow_wakeless_emergency=True)

    off_decision = off.classify("숨이 안 쉬어", active_session=False)
    on_decision = on.classify("숨이 안 쉬어", active_session=False)

    assert off_decision.engine_scope == "ignored"
    assert on_decision.route_label == "emergency"
    assert on_decision.risk_level == "emergency"
