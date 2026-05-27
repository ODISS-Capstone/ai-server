"""Tests for the deployable assistant web API surface."""
from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import settings
from app.database.md_store import MDStore
from app.main import app
from app.schemas.ocr import OcrResponse


def _turn_payload() -> dict:
    return {
        "session_id": "session-1",
        "speaker_id": "speaker-1",
        "turn_id": "turn-1",
        "rating": "down",
        "tags": ["latency"],
        "comment": "응답이 느렸습니다.",
        "user_text": "오디스",
        "response_text": "네, 말씀하세요.",
        "response_type": "wake_word_ack",
        "fast_path": "wake_word",
        "latency": {"first_message_ms": 12, "final_response_ms": 12},
        "raw": {"type": "response"},
        "user_agent": "pytest",
    }


def test_feedback_requires_token_in_production(tmp_path, monkeypatch):
    from app.api.routes import feedback_api

    store = MDStore(base_path=str(tmp_path / "md_database"))
    monkeypatch.setattr(feedback_api, "md_store", store)
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "assistant_web_token", "assistant-token")

    client = TestClient(app)
    missing = client.post("/api/feedback/turn", json=_turn_payload())
    assert missing.status_code == 401

    saved = client.post(
        "/api/feedback/turn",
        json=_turn_payload(),
        headers={"Authorization": "Bearer assistant-token"},
    )
    assert saved.status_code == 200
    data = saved.json()
    assert data["success"] is True
    assert data["path"].startswith("permanent/feedback/")
    assert (store.base / data["path"]).exists()


def test_feedback_allows_local_without_token(tmp_path, monkeypatch):
    from app.api.routes import feedback_api

    store = MDStore(base_path=str(tmp_path / "md_database"))
    monkeypatch.setattr(feedback_api, "md_store", store)
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "assistant_web_token", "")

    client = TestClient(app)
    response = client.post("/api/feedback/session", json={
        "session_id": "session-1",
        "speaker_id": "speaker-1",
        "satisfaction": 4,
        "comment": "괜찮았습니다.",
        "problem_tags": ["OCR"],
        "turn_count": 3,
        "user_agent": "pytest",
    })

    assert response.status_code == 200
    assert (store.base / response.json()["path"]).exists()


def test_upload_image_uses_assistant_token_in_production(monkeypatch):
    from app.api import upload

    async def fake_ocr(_contents: bytes, *, content_type: str) -> OcrResponse:
        return OcrResponse(
            raw_text="디오반정 80mg",
            medications=[{"name": "디오반정", "strength": "80mg"}],
            success=True,
        )

    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "assistant_web_token", "assistant-token")
    monkeypatch.setattr(upload.ocr_service, "run_ocr_image", fake_ocr)

    client = TestClient(app)
    files = {"file": ("rx.png", b"fake-image", "image/png")}
    missing = client.post("/upload/image", files=files)
    assert missing.status_code == 401

    files = {"file": ("rx.png", b"fake-image", "image/png")}
    response = client.post(
        "/upload/image",
        files=files,
        headers={"Authorization": "Bearer assistant-token"},
    )
    assert response.status_code == 200
    assert response.json()["medications"][0]["name"] == "디오반정"


def test_websocket_requires_query_token_in_production(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "assistant_web_token", "assistant-token")

    client = TestClient(app)
    try:
        with client.websocket_connect("/ws/chat"):
            pass
    except WebSocketDisconnect as exc:
        assert exc.code == 1008
    else:  # pragma: no cover - defensive branch for unexpected auth bypass
        raise AssertionError("WebSocket accepted without token")

    with client.websocket_connect("/ws/chat?token=assistant-token") as websocket:
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json() == {"type": "pong"}
