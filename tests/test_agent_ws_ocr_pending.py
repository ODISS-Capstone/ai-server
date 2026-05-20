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


def test_pending_ocr_recapture_sends_new_ocr_request() -> None:
    key = agent_ws._pending_ocr_key("speaker-ocr")
    agent_ws._pending_ocr_by_speaker[key] = {
        "data": {"medications": [{"name": "DrugA"}]},
        "created_at": datetime.now(),
    }
    ws = FakeWebSocket()

    handled = run(
        agent_ws._handle_pending_ocr_confirmation(
            ws,
            "다시 찍자",
            "speaker-ocr",
        )
    )

    assert handled is True
    assert key not in agent_ws._pending_ocr_by_speaker
    assert ws.sent[0]["response_type"] == "ocr_cancelled"
    assert ws.sent[1]["type"] == "ocr_request"
    assert ws.sent[1]["reason"] == "user_requested_recapture"


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


def test_pending_ocr_symptom_answer_refines_medication_candidate(monkeypatch) -> None:
    key = agent_ws._pending_ocr_key("speaker-ocr")
    agent_ws._pending_ocr_by_speaker[key] = {
        "data": {
            "raw_text": "처방 의약품 | 페니라민정 | 록소나정 60mg",
            "medications": [{"name": "페니라민정"}, {"name": "록소나정 60mg"}],
        },
        "created_at": datetime.now(),
    }
    ws = FakeWebSocket()

    async def fake_refine_ocr_medication_candidates_with_context(**kwargs):
        assert "통풍" in kwargs["user_text"]
        return {
            "source": "frontier_openai_context_refine",
            "medications": [
                {
                    "name": "페브릭정",
                    "purpose_or_symptom": "통풍",
                    "correction_reason": "사용자가 통풍 처방이라고 설명함",
                }
            ],
            "clarification_question": "",
        }

    monkeypatch.setattr(
        agent_ws,
        "refine_ocr_medication_candidates_with_context",
        fake_refine_ocr_medication_candidates_with_context,
    )

    handled = run(
        agent_ws._handle_pending_ocr_confirmation(
            ws,
            "통풍 때문에 처방받아서 통풍약이야",
            "speaker-ocr",
        )
    )

    pending = agent_ws._pending_ocr_by_speaker[key]
    assert handled is True
    assert pending["data"]["medications"][0]["name"] == "페브릭정"
    assert pending["data"]["symptom_context"] == "통풍 때문에 처방받아서 통풍약이야"
    assert ws.sent[-1]["response_type"] == "ocr_refined_confirmation"
    assert "페브릭정" in ws.sent[-1]["response_text"]
    assert "저장할까요" in ws.sent[-1]["response_text"]


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


def test_ocr_background_enrichment_redacts_sensitive_context(monkeypatch) -> None:
    captured: dict = {}

    async def fake_llm_search(query, context=None):
        captured["query"] = query
        captured["context"] = context
        return {"success": True, "answer": "무브록정40mg 약물 후보 요약"}

    async def fake_check_dur_for_prescription(medications):
        captured["dur_meds"] = medications
        return []

    async def fake_sync_ocr_dur(ocr_data, dur_results, speaker_id=None):
        captured["sync_speaker"] = speaker_id

    saved: list[tuple[str, str]] = []
    flashed: dict[str, str] = {}

    async def fake_save(category, content):
        saved.append((category, content))

    async def fake_write_flash(key, content):
        flashed[key] = content

    monkeypatch.setattr(agent_ws.llm_search, "llm_search", fake_llm_search)
    monkeypatch.setattr(dur_api, "check_dur_for_prescription", fake_check_dur_for_prescription)
    monkeypatch.setattr(agent_ws.memory_engine, "sync_ocr_dur", fake_sync_ocr_dur)
    monkeypatch.setattr(agent_ws.memory_engine.store, "save", fake_save)
    monkeypatch.setattr(agent_ws.memory_engine.store, "write_flash", fake_write_flash)

    run(
        agent_ws._enrich_ocr_medication_background(
            ocr_data={
                "raw_text": "성명 | 이종석 | 주민등록번호 900101-1234567 | 전화번호 032-670-0001 | 무브록정40mg",
                "medications": [{"name": "무브록정40mg"}],
            },
            medications=[{"name": "무브록정40mg"}],
            speaker_id="speaker-ocr",
            user_text="처방전 저장해줘",
        )
    )

    assert "무브록정40mg" in captured["query"]
    assert "이종석" not in captured["context"]
    assert "900101-1234567" not in captured["context"]
    assert "032-670-0001" not in captured["context"]
    assert captured["dur_meds"] == [{"name": "무브록정40mg"}]
    assert captured["sync_speaker"] == "speaker-ocr"
    assert saved[-1][0] == "medication_log"
    assert "무브록정40mg" in flashed["context_memory"]
