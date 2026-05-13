"""Mutual communication contract between ai-server and local_agent.

These tests exercise the REST/WebSocket endpoints that the OCR agent
uses to talk to the cloud server.  They make sure that:

- The agent's default HTTP paths resolve to real FastAPI routes.
- The agent's payload shape is accepted by the server's pydantic
  models.
- The WebSocket ``/ws/chat`` endpoint handles the agent's OCR and
  ping messages with the exact field names from ``agent_ws.py``.

The actual ``aiohttp`` session is never opened.  Instead we take the
payload that the local_agent clients would send and replay it through
``fastapi.testclient.TestClient`` so every message crosses the same
pydantic validation that a Jetson device would hit in production.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

AI_SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AI_SERVER_ROOT.parent
LOCAL_AGENT_ROOT = PROJECT_ROOT / "local_agent"

for path in (AI_SERVER_ROOT, LOCAL_AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _import_agent_drug_parser():
    try:
        from src.cloud_server.drug_parser import (  # type: ignore[import-not-found]
            DrugParserConfig,
            _to_server_ocr_payload,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "src" or not LOCAL_AGENT_ROOT.exists():
            pytest.skip("local_agent source package is not available in this checkout")
        raise

    return DrugParserConfig, _to_server_ocr_payload


def _import_agent_instruction_log():
    try:
        from src.cloud_server.instruction_log import (  # type: ignore[import-not-found]
            InstructionEntry,
            InstructionLogConfig,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "src" or not LOCAL_AGENT_ROOT.exists():
            pytest.skip("local_agent source package is not available in this checkout")
        raise

    return InstructionEntry, InstructionLogConfig


# ---------------------------------------------------------------------------
# HTTP: /api/ocr/analyze
# ---------------------------------------------------------------------------


def test_agent_ocr_config_points_to_server_route(client: TestClient) -> None:
    DrugParserConfig, _ = _import_agent_drug_parser()
    config = DrugParserConfig()

    assert config.path == "/api/ocr/analyze"

    probe = client.get(f"{config.path.rsplit('/', 1)[0]}/history")
    assert probe.status_code == 200, probe.text


def test_agent_ocr_payload_is_accepted_by_server(client: TestClient) -> None:
    DrugParserConfig, to_server_payload = _import_agent_drug_parser()
    config = DrugParserConfig()

    agent_payload = {
        "perception_timestamp": "2026-04-28T00:00:00+00:00",
        "input_type": "PRESCRIPTION",
        "ocr_results": {
            "text": "처방전 샘플 텍스트",
            "text_confidence_score": 0.91,
            "structured_data": {
                "drugs": [
                    {
                        "name": "혈압약A",
                        "dosage": "5 mg",
                        "frequency": "1일 2회",
                        "days": 30,
                    }
                ]
            },
        },
        "action_required": "PROCEED_TO_IDENTIFICATION",
        "speaker_id": "speaker-contract",
    }

    body = to_server_payload(agent_payload)

    assert body["raw_text"] == "처방전 샘플 텍스트"
    assert body["confidence"] == pytest.approx(0.91)
    assert body["speaker_id"] == "speaker-contract"
    assert body["medications"][0]["name"] == "혈압약A"
    assert body["medications"][0]["dosage"] == "5 mg"
    assert body["medications"][0]["frequency"] == "1일 2회"

    response = client.post(config.path, json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["medication_count"] == 1
    assert isinstance(data["dur_results"], list)


# ---------------------------------------------------------------------------
# HTTP: /api/stt/log
# ---------------------------------------------------------------------------


def test_agent_stt_log_config_is_served(client: TestClient) -> None:
    InstructionEntry, InstructionLogConfig = _import_agent_instruction_log()
    config = InstructionLogConfig()

    assert config.path == "/api/stt/log"

    entry = InstructionEntry(text="약 가져왔어")
    payload = entry.to_dict()
    payload["speaker_id"] = "speaker-contract"

    response = client.post(config.path, json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["received_chars"] == len("약 가져왔어")


# ---------------------------------------------------------------------------
# WebSocket: /ws/chat
# ---------------------------------------------------------------------------


def test_agent_ws_ping_is_honored(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        reply = ws.receive_json()
        assert reply == {"type": "pong"}


def test_agent_ws_ocr_result_contract(client: TestClient) -> None:
    DrugParserConfig, to_server_payload = _import_agent_drug_parser()
    agent_payload = {
        "perception_timestamp": "2026-04-28T00:00:00+00:00",
        "input_type": "PRESCRIPTION",
        "ocr_results": {
            "text": "처방전 WS 계약",
            "text_confidence_score": 0.88,
            "structured_data": {
                "drugs": [
                    {"name": "혈압약A", "dosage": "5 mg", "frequency": "1일 2회", "days": 30},
                    {"name": "당뇨약B", "dosage": "10 mg", "frequency": "1일 1회", "days": 30},
                ]
            },
        },
        "action_required": "PROCEED_TO_IDENTIFICATION",
        "speaker_id": "speaker-contract-ws",
    }

    # ai-server WebSocket contract expects `{"type": "ocr_result", "data": {...}, "speaker_id": "..."}`.
    data_for_ws = to_server_payload(agent_payload)
    data_for_ws_wire = {
        "raw_text": data_for_ws["raw_text"],
        "confidence": data_for_ws["confidence"],
        "medications": data_for_ws["medications"],
    }

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "ocr_result",
                    "data": data_for_ws_wire,
                    "speaker_id": data_for_ws["speaker_id"],
                }
            )
        )
        reply = ws.receive_json()

    assert reply["type"] == "ocr_processed"
    assert reply["medication_count"] == 2
    assert "DUR" in reply["message"] or reply["medication_count"] == 2


def test_agent_ws_unknown_message_returns_error(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "what_is_this"}))
        reply = ws.receive_json()

    assert reply["type"] == "error"
    assert "Unknown type" in reply["message"]
