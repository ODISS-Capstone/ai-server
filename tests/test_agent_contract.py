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


@pytest.fixture(autouse=True)
def clear_agent_ws_state():
    from app.api.routes import agent_ws

    agent_ws._pending_ocr_by_speaker.clear()
    agent_ws._queued_ocr_request_by_speaker.clear()
    yield
    agent_ws._pending_ocr_by_speaker.clear()
    agent_ws._queued_ocr_request_by_speaker.clear()


def _receive_ocr_filler_then_payload(ws):
    filler = ws.receive_json()
    assert filler["type"] == "filler"
    assert filler["stage"] == "ocr_processing"
    assert filler["requires_tts"] is True
    assert "사진" in filler["text"]
    return filler, ws.receive_json()


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
        _, reply = _receive_ocr_filler_then_payload(ws)

    assert reply["type"] == "ocr_processed"
    assert reply["medication_count"] == 2
    assert "DUR" in reply["message"] or reply["medication_count"] == 2


def test_agent_ws_uncertain_ocr_requests_recapture(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "ocr_result",
                    "speaker_id": "speaker-contract-ws",
                    "data": {
                        "raw_text": "흐림 [불명확] 약 이름 확인 불가",
                        "confidence": 0.3,
                        "medications": [{"name": "불명확"}],
                    },
                }
            )
        )
        _, first = _receive_ocr_filler_then_payload(ws)
        second = ws.receive_json()

    assert first["type"] == "ocr_processed"
    assert first["needs_recapture"] is True
    assert first["pending_confirmation"] is False
    assert second["type"] == "ocr_request"
    assert second["reason"] == "uncertain_ocr_result"
    assert second["requires_tts"] is False


def test_agent_ws_ocr_does_not_regex_guess_when_frontier_llm_finds_no_medication(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.routes import agent_ws

    async def fake_extract(raw_text: str):
        return {"medications": [], "clarification_question": ""}

    monkeypatch.setattr(agent_ws, "extract_ocr_medication_candidates_with_llm", fake_extract)
    raw_text = (
        "### 2. 처방 의약품 목록 | 처방 의약품의 명칭 | 1회 투약량 | 1일 투여횟수 | "
        "총 투약일수 | 용법 | | 664704210 무브록정40mg | 1 | 1 | 30 | [불명확] |"
    )
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "ocr_result",
                    "speaker_id": "speaker-contract-ws",
                    "data": {
                        "raw_text": raw_text,
                        "confidence": 0.95,
                        "medications": [],
                    },
                }
            )
        )
        _, first = _receive_ocr_filler_then_payload(ws)
        second = ws.receive_json()

    assert first["type"] == "ocr_processed"
    assert first["medication_count"] == 0
    assert first["needs_recapture"] is True
    assert second["type"] == "ocr_request"


def test_agent_ws_ocr_uses_llm_candidates_and_question(client: TestClient, monkeypatch) -> None:
    from app.api.routes import agent_ws

    async def fake_extract(raw_text: str):
        assert "처방 의약품" in raw_text
        return {
            "medications": [
                {
                    "name": "무브록정40mg",
                    "dosage": "1",
                    "frequency": "1일 1회",
                    "timing": "",
                    "purpose_or_symptom": "",
                }
            ],
            "clarification_question": "통증이나 염증 때문에 처방받으신 약인가요?",
        }

    monkeypatch.setattr(agent_ws, "extract_ocr_medication_candidates_with_llm", fake_extract)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "ocr_result",
                    "speaker_id": "speaker-contract-ws",
                    "data": {
                        "raw_text": "### 처방 의약품 목록 | 664704210 무브록정40mg | 1 | 1 | 30 | [불명확] |",
                        "confidence": 0.95,
                        "medications": [],
                    },
                }
            )
        )
        _, reply = _receive_ocr_filler_then_payload(ws)

    assert reply["type"] == "ocr_processed"
    assert reply["medication_count"] == 1
    assert "무브록정40mg" in reply["message"]
    assert "통증이나 염증 때문에 처방받으신 약인가요?" in reply["message"]
    assert reply["pending_confirmation"] is True


def test_agent_ws_unknown_message_returns_error(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "what_is_this"}))
        reply = ws.receive_json()

    assert reply["type"] == "error"
    assert "Unknown type" in reply["message"]


def test_agent_ws_accepts_identity_confirmed_compat_event(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "identity_confirmed",
                    "speaker_id": "speaker-contract-ws",
                    "text": "어 난 김영수가 맞아",
                }
            )
        )
        reply = ws.receive_json()

    assert reply["type"] != "error"
    assert "Unknown type" not in json.dumps(reply, ensure_ascii=False)


def test_agent_ws_ocr_capture_request_waits_for_identity_gate(monkeypatch, client: TestClient) -> None:
    from app.api.routes import agent_ws

    async def fake_identity_gate(**kwargs):
        from app.services.identity_guard import IdentityGateResult

        return IdentityGateResult(
            allowed=False,
            reason="needs_registration",
            response_text="먼저 신원 확인이 필요합니다.",
        )

    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "stt_result",
                    "speaker_id": "speaker-contract-ws",
                    "text": "처방전을 찍어야 되는데 켜 줄래",
                }
            )
        )
        reply = ws.receive_json()

    assert reply["type"] == "response"
    assert reply["identity_gate"]["reason"] == "needs_registration"
    assert agent_ws._queued_ocr_request_by_speaker[agent_ws._pending_ocr_key("speaker-contract-ws")]["reason"] == (
        "direct_ocr_capture_request"
    )


def test_agent_ws_camera_ocr_request_is_detected(monkeypatch, client: TestClient) -> None:
    from app.api.routes import agent_ws

    async def fake_identity_gate(**kwargs):
        from app.services.identity_guard import IdentityGateResult

        return IdentityGateResult(allowed=True, reason="identity_verified")

    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "stt_result",
                    "speaker_id": "speaker-contract-ws",
                    "text": "처방전 등록을 해야 되는데 한번 사진 찍게 카메라 좀 켜 줄래?",
                }
            )
        )
        reply = ws.receive_json()

    assert reply["type"] == "ocr_request"
    assert reply["action"] == "request_ocr"


def test_agent_ws_queued_ocr_request_runs_after_identity_passes(monkeypatch, client: TestClient) -> None:
    from app.api.routes import agent_ws

    key = agent_ws._pending_ocr_key("speaker-contract-ws")
    agent_ws._queued_ocr_request_by_speaker[key] = {
        "reason": "direct_ocr_capture_request",
        "created_at": "test",
    }

    async def fake_identity_gate(**kwargs):
        from app.services.identity_guard import IdentityGateResult

        return IdentityGateResult(allowed=True, reason="identity_verified")

    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "stt_result",
                    "speaker_id": "speaker-contract-ws",
                    "text": "김영수 72세 남성",
                }
            )
        )
        reply = ws.receive_json()

    assert reply["type"] == "ocr_request"
    assert reply["reason"] == "direct_ocr_capture_request"
    assert key not in agent_ws._queued_ocr_request_by_speaker


def test_agent_ws_recapture_reply_bypasses_identity_gate(monkeypatch, client: TestClient) -> None:
    from app.api.routes import agent_ws

    async def fake_identity_gate(**kwargs):
        from app.services.identity_guard import IdentityGateResult

        return IdentityGateResult(allowed=False, reason="needs_registration", response_text="신원 확인 필요")

    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "stt_result",
                    "speaker_id": "speaker-contract-ws",
                    "text": "다시 찍자",
                }
            )
        )
        reply = ws.receive_json()

    assert reply["type"] == "response"
    assert reply["identity_gate"]["reason"] == "needs_registration"
    assert agent_ws._queued_ocr_request_by_speaker[agent_ws._pending_ocr_key("speaker-contract-ws")]["reason"] == (
        "user_requested_recapture"
    )
