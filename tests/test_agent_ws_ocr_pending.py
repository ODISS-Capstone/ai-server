"""Regression coverage for WebSocket OCR confirmation state."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from app.api.routes import agent_ws
from app.tools import dur_api


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def run(coro):
    return asyncio.run(coro)


def setup_function() -> None:
    agent_ws._pending_ocr_by_speaker.clear()


def test_pending_ocr_rejection_clears_payload() -> None:
    key = agent_ws._pending_ocr_key("speaker-ocr")
    agent_ws._pending_ocr_by_speaker[key] = {
        "data": {"medications": [{"name": "DrugA"}]},
        "created_at": datetime.now(),
    }
    ws = FakeWebSocket()

    handled = run(
        agent_ws._handle_pending_ocr_confirmation(
            ws,
            "아니 저장하지 마",
            "speaker-ocr",
        )
    )

    assert handled is True
    assert key not in agent_ws._pending_ocr_by_speaker
    assert ws.sent[-1]["response_type"] == "ocr_cancelled"
    assert "저장하지 않았습니다" in ws.sent[-1]["response_text"]


def test_pending_ocr_expiry_does_not_save_stale_payload() -> None:
    key = agent_ws._pending_ocr_key("speaker-ocr")
    agent_ws._pending_ocr_by_speaker[key] = {
        "data": {"medications": [{"name": "DrugA"}]},
        "created_at": datetime.now() - agent_ws.OCR_PENDING_TTL - timedelta(seconds=1),
    }
    ws = FakeWebSocket()

    handled = run(
        agent_ws._handle_pending_ocr_confirmation(
            ws,
            "네 저장해",
            "speaker-ocr",
        )
    )

    assert handled is True
    assert key not in agent_ws._pending_ocr_by_speaker
    assert ws.sent[-1]["response_type"] == "ocr_expired"


def test_pending_ocr_unclear_reply_keeps_confirmation_state() -> None:
    key = agent_ws._pending_ocr_key("speaker-ocr")
    agent_ws._pending_ocr_by_speaker[key] = {
        "data": {"medications": [{"name": "DrugA"}]},
        "created_at": datetime.now(),
    }
    ws = FakeWebSocket()

    handled = run(
        agent_ws._handle_pending_ocr_confirmation(
            ws,
            "오늘 저녁 약 알려줘",
            "speaker-ocr",
        )
    )

    assert handled is True
    assert key in agent_ws._pending_ocr_by_speaker
    assert ws.sent[-1]["response_type"] == "ocr_confirmation_required"


def test_ocr_dur_background_keeps_full_result_rows(monkeypatch) -> None:
    captured: dict = {}

    async def fake_check_dur_for_prescription(medications):
        return [
            {
                "medication": "DrugA",
                "dur": {"dur_product_info": {"success": True, "items": [{"name": "DrugA"}]}},
            }
        ]

    async def fake_sync_ocr_dur(ocr_data, dur_results, speaker_id=None):
        captured["dur_results"] = dur_results
        captured["speaker_id"] = speaker_id

    monkeypatch.setattr(dur_api, "check_dur_for_prescription", fake_check_dur_for_prescription)
    monkeypatch.setattr(agent_ws.memory_engine, "sync_ocr_dur", fake_sync_ocr_dur)

    run(
        agent_ws._sync_ocr_dur_background(
            {"medications": [{"name": "DrugA"}]},
            [{"name": "DrugA"}],
            "speaker-ocr",
        )
    )

    assert captured["speaker_id"] == "speaker-ocr"
    assert captured["dur_results"][0]["medication"] == "DrugA"
