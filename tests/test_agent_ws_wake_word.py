"""Regression coverage for wake-word-only WebSocket turns."""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.api.routes import agent_ws
from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory import StructuredMemoryService
from app.services.identity_guard import IdentityGateResult
from app.services.reminders import ReminderService


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class FakeMemoryEngine:
    def __init__(self) -> None:
        self.seen: list[tuple[str, bool]] = []

    async def bootstrap_flash_from_permanent(self, speaker_id: str) -> None:
        return None

    async def load_context(self, speaker_id: str) -> dict:
        return {
            "user_profile": {
                "name": "김영수",
                "age": "72",
                "gender": "남성",
            },
            "prescription_log": "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        }

    async def load_identity_state(self, speaker_id: str) -> dict:
        return {
            "exists": True,
            "profile": {
                "name": "김영수",
                "age": "72",
                "gender": "남성",
            },
        }

    async def mark_identity_seen(self, speaker_id: str, *, verified: bool = False, now=None) -> dict:
        self.seen.append((speaker_id, verified))
        return {}

    async def update_and_compress(self, response_data: dict, speaker_id: str | None = None) -> None:
        return None


class FakeReminderService:
    def register_connection(self, speaker_id: str, callback) -> None:
        return None

    async def restore_for_speaker(self, memory_engine, speaker_id: str) -> None:
        return None

    async def handle_user_text(self, **kwargs):
        return None


def run(coro):
    return asyncio.run(coro)


def setup_function() -> None:
    agent_ws._bootstrapped_speakers.clear()
    agent_ws._pending_ocr_by_speaker.clear()
    agent_ws._queued_ocr_request_by_speaker.clear()
    agent_ws._wake_profile_cache_by_speaker.clear()
    agent_ws._identity_pending_action_cache_by_speaker.clear()


def make_real_memory(tmp_path) -> MemoryEngine:
    memory = MemoryEngine()
    memory.store = MDStore(str(tmp_path / "md_database"))
    memory.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(memory.initialize())
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    return memory


def test_registered_wake_word_uses_profile_and_skips_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("wake-word-only turn should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("wake-word-only turn should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "wake_word_ack"
    assert websocket.sent[-1]["requires_tts"] is True
    assert websocket.sent[-1]["response_text"] == "네, 김영수님. 말씀하세요."
    assert "어르신" not in websocket.sent[-1]["response_text"]
    assert "약" not in websocket.sent[-1]["response_text"]
    assert fake_memory.seen == []


def test_registered_smalltalk_greeting_uses_fast_path_and_skips_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("smalltalk fast path should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("smalltalk fast path should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "안녕",
            },
            set(),
        )
    )

    payload = websocket.sent[-1]
    assert payload["type"] == "response"
    assert payload["response_type"] == "smalltalk"
    assert payload["requires_tts"] is True
    assert payload["fast_path"] == "smalltalk"
    assert payload["server_elapsed_ms"] <= 300
    assert payload["response_text"].count("김영수님") == 1
    assert "안녕하세요" in payload["response_text"]
    assert "확인된 정보가 제한적" not in payload["response_text"]


def test_anonymous_smalltalk_thanks_uses_fast_path_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("anonymous smalltalk should not ask for registration")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("anonymous smalltalk should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", FakeMemoryEngine())
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "text": "고마워",
            },
            set(),
        )
    )

    payload = websocket.sent[-1]
    assert payload["response_type"] == "smalltalk"
    assert payload["fast_path"] == "smalltalk"
    assert "사용자님" in payload["response_text"]
    assert "이름, 나이, 성별" not in payload["response_text"]


def test_registered_smalltalk_acknowledgement_uses_fast_path_without_filler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("acknowledgement smalltalk should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("acknowledgement smalltalk should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "그래 잘했어",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    payload = websocket.sent[-1]
    assert payload["response_type"] == "smalltalk"
    assert payload["fast_path"] == "smalltalk"
    assert "필요하시면" in payload["response_text"]
    assert "out_of_scope" not in payload["response_text"]


def test_symptom_utterance_does_not_use_smalltalk_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()
    calls: list[str] = []

    async def fake_identity_gate(**kwargs):
        calls.append("identity_gate")
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    async def fake_run_turn(**kwargs):
        calls.append("orchestrator")
        return SimpleNamespace(
            filler_text="",
            conversation=SimpleNamespace(
                requires_tts=True,
                response_text="김영수님, 가슴이 답답하면 즉시 119나 응급실에 연락하세요.",
                response_type="medical_response",
            ),
            execution_results={"task_results": {}},
            decision=SimpleNamespace(rationale="emergency_policy_first", intent="emergency"),
            core_message="",
            judge_review={},
        )

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fake_run_turn)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "가슴이 답답해",
            },
            set(),
        )
    )

    assert calls == ["identity_gate", "orchestrator"]
    assert websocket.sent[-1]["response_type"] == "medical_response"
    assert websocket.sent[-1].get("fast_path") != "smalltalk"
    assert "119" in websocket.sent[-1]["response_text"]


def test_reminder_setup_sends_filler_before_reminder_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    class SlowReminderService(FakeReminderService):
        async def handle_user_text(self, **kwargs):
            assert websocket.sent
            assert websocket.sent[0]["type"] == "filler"
            await asyncio.sleep(0)
            return "네, 김영수님. 현재 저장된 복약 정보 기준으로 식후 복용 알림을 설정할 수 있습니다."

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", SlowReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스, 내가 밥 먹고 나서 약 먹어야 한다는 걸 알림 추가해줘.",
            },
            set(),
        )
    )

    assert websocket.sent[0]["type"] == "filler"
    assert websocket.sent[0]["stage"] == "reminder"
    assert websocket.sent[0]["requires_tts"] is True
    assert "알림" in websocket.sent[0]["text"]
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "reminder"
    assert "김영수님" in websocket.sent[-1]["response_text"]


def test_agent_ws_reminder_setup_confirm_and_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    current = datetime(2026, 5, 18, 11, 55)

    def now_provider() -> datetime:
        return current

    fake_memory = make_real_memory(tmp_path)
    websocket = FakeWebSocket()
    reminder_service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스, 약 먹을 때 깨워줘.",
            },
            set(),
        )
    )
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "점심은 12시로 해줘.",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "reminder"
    assert "점심 약 알림은 오후 12시" in websocket.sent[-1]["response_text"]

    current = datetime(2026, 5, 18, 12, 0)
    dispatched = run(reminder_service.dispatch_due_reminders())

    assert dispatched
    assert websocket.sent[-1]["type"] == "reminder"
    assert "점심 혈압약" in websocket.sent[-1]["text"]


def test_agent_ws_relative_one_shot_reminder_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    current = datetime(2026, 5, 26, 19, 47)

    def now_provider() -> datetime:
        return current

    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("relative one-shot alarm should not call the identity gate")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "30초 뒤에 혈압 약 먹으라고 알람 설정해 줄 수 있어",
            },
            set(),
        )
    )

    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["response_type"] == "reminder"
    assert "30초 뒤" in response_payloads[-1]["response_text"]
    assert "혈압약" in response_payloads[-1]["response_text"]
    assert "아침은 오전 8시" not in response_payloads[-1]["response_text"]

    current = datetime(2026, 5, 26, 19, 47, 30)
    dispatched = run(reminder_service.dispatch_due_reminders())

    assert dispatched
    assert websocket.sent[-1]["type"] == "reminder"
    assert websocket.sent[-1]["reminder_kind"] == "one_shot"
    assert "정현기님" in websocket.sent[-1]["text"]
    assert "혈압약" in websocket.sent[-1]["text"]


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("10초 뒤에 혈압 약 먹으라고 알려 줘", "혈압약"),
        ("10초 뒤에 타이레놀 먹으라고 알려 줘", "타이레놀"),
    ],
)
def test_agent_ws_exact_short_medication_alarm_skips_llm_and_filler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    text: str,
    expected_label: str,
) -> None:
    current = datetime(2026, 5, 26, 20, 36, 19)

    def now_provider() -> datetime:
        return current

    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("short one-shot medication alarm should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("short one-shot medication alarm should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": text,
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["response_type"] == "reminder"
    assert response_payloads[-1]["fast_path"] == "relative_alarm"
    assert response_payloads[-1]["delay_seconds"] == 10
    assert expected_label in response_payloads[-1]["response_text"]
    assert "확인된 정보가 제한적" not in response_payloads[-1]["response_text"]
    assert "의사·약사" not in response_payloads[-1]["response_text"]

    current = datetime(2026, 5, 26, 20, 36, 29)
    dispatched = run(reminder_service.dispatch_due_reminders())

    assert dispatched
    assert websocket.sent[-1]["type"] == "reminder"
    assert websocket.sent[-1]["reminder_kind"] == "one_shot"
    assert expected_label in websocket.sent[-1]["text"]


def test_agent_ws_generic_relative_alarm_skips_orchestrator_and_filler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    current = datetime(2026, 5, 26, 20, 14, 57)

    def now_provider() -> datetime:
        return current

    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("relative one-shot alarm should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("relative one-shot alarm should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "30초 뒤에 알람 설정해 줘",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["response_type"] == "reminder"
    assert response_payloads[-1]["fast_path"] == "relative_alarm"
    assert response_payloads[-1]["reminder_kind"] == "one_shot"
    assert response_payloads[-1]["delay_seconds"] == 30
    assert "30초 뒤" in response_payloads[-1]["response_text"]
    assert "아침은 오전 8시" not in response_payloads[-1]["response_text"]
    assert "의사·약사" not in response_payloads[-1]["response_text"]


def test_agent_ws_missed_one_shot_check_dispatches_due_alarm_without_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    current = datetime(2026, 5, 26, 20, 14, 57)

    def now_provider() -> datetime:
        return current

    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("missed one-shot check should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("missed one-shot check should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    active_speakers: set[str] = set()
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "30초 뒤에 알람 설정해 줘",
            },
            active_speakers,
        )
    )
    current = datetime(2026, 5, 26, 20, 15, 27)
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "30초 지났음",
            },
            active_speakers,
        )
    )

    assert websocket.sent[-1]["type"] == "reminder"
    assert websocket.sent[-1]["reminder_kind"] == "one_shot"
    assert "정현기님" in websocket.sent[-1]["text"]
    assert not any(
        payload.get("response_text") == "방금 설정된 알림을 찾지 못했습니다. 다시 설정해 주세요."
        for payload in websocket.sent
    )


def test_agent_ws_medication_taken_time_recall_skips_filler_and_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.save_user_file(
            "speaker-hyun",
            "medication_taken.md",
            '- {"taken_at": "2026-05-26T20:52:39+09:00", "meal": "식후", "medication_label": "타이레놀", "source_text": "먹었어"}\n',
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("medication taken recall should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "몇 시에 먹었지",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["response_type"] == "reminder"
    assert response_payloads[-1]["fast_path"] == "medication_taken_recall"
    assert "오후 8시 52분" in response_payloads[-1]["response_text"]
    assert "타이레놀" in response_payloads[-1]["response_text"]


def test_agent_ws_spoken_medication_registration_skips_filler_and_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("spoken medication registration should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "나 디오반정 가지고 있거든 한번 확인해 줘",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "spoken_medication_registration"
    assert "디오반정" in response_payloads[-1]["response_text"]
    prescription_log = run(fake_memory.store.read_flash("prescription_log"))
    assert "디오반정" in prescription_log


def test_agent_ws_medication_safety_question_skips_filler_and_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-kim",
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("medication safety question should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스 내가 디오반정을 내게 동시에 먹어도 될까",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "medication_safety_fast_path"
    assert response_payloads[-1]["response_type"] == "medical_response"
    answer = response_payloads[-1]["response_text"]
    assert "디오반정" in answer
    assert "한 번에" in answer
    assert "119" in answer
    assert "디곡신" not in answer
    assert "方才" not in answer
    assert "主治" not in answer
    assert len(answer) < 260


def test_agent_ws_stored_medication_guidance_handles_vague_that_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("stored medication guidance should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "오늘 그거 먹어야 돼",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "stored_medication_guidance"
    assert "디오반정" in response_payloads[-1]["response_text"]
    assert "한 번 더 드시지 마세요" in response_payloads[-1]["response_text"]


def test_agent_ws_explicit_stored_medication_need_uses_fast_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("explicit stored medication guidance should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "나 디오반정 먹어야 되는데",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "stored_medication_guidance"
    assert "디오반정" in response_payloads[-1]["response_text"]


def test_agent_ws_medication_taken_confirmation_uses_stored_medication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("taken confirmation should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "어 먹었어",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "medication_taken_record"
    assert "식후 디오반정" in response_payloads[-1]["response_text"]
    assert "식후 식후 약" not in response_payloads[-1]["response_text"]


def test_agent_ws_colloquial_taken_record_command_uses_fast_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("taken record command should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "어 먹었어 기록해 줘",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "medication_taken_record"
    assert "식후 디오반정" in response_payloads[-1]["response_text"]
    saved = run(fake_memory.store.read_user_file("speaker-hyun", "medication_taken.md"))
    assert "디오반정" in saved


def test_agent_ws_medication_intent_to_take_is_not_smalltalk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("intent-to-take should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "어 알았어 지금 먹을게",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "medication_intent_to_take"
    assert response_payloads[-1]["response_type"] == "medical_response"
    assert "디오반정" in response_payloads[-1]["response_text"]
    assert "먹었어" in response_payloads[-1]["response_text"]


def test_agent_ws_current_medication_list_recall_uses_fast_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("current medication list recall should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "그거 말고 저장된게 있나",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "stored_medication_list_recall"
    assert "디오반정" in response_payloads[-1]["response_text"]
    assert "추가로 저장된 약은 없습니다" in response_payloads[-1]["response_text"]


def test_agent_ws_now_take_time_record_and_correction_use_fast_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    current = datetime(2026, 5, 26, 22, 30, 56)

    def now_provider() -> datetime:
        return current

    fake_memory = make_real_memory(tmp_path)
    run(
        fake_memory.save_identity_profile(
            "speaker-kim",
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        fake_memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("time record/correction should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    active_speakers: set[str] = set()
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "알았어 나 지금 먹을테니까 기록해 줘",
            },
            active_speakers,
        )
    )
    first_response = [payload for payload in websocket.sent if payload.get("type") == "response"][-1]
    assert first_response["fast_path"] == "medication_taken_record"
    assert "오후 10시 30분" in first_response["response_text"]
    assert "식후 디오반정" in first_response["response_text"]
    assert "나지금" not in first_response["response_text"]
    assert not any(payload.get("type") == "filler" for payload in websocket.sent)

    websocket.sent.clear()
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "내가 언제 뭘 먹었다고",
            },
            active_speakers,
        )
    )
    recall_response = [payload for payload in websocket.sent if payload.get("type") == "response"][-1]
    assert recall_response["fast_path"] == "medication_taken_recall"
    assert "오후 10시 30분" in recall_response["response_text"]
    assert not any(payload.get("type") == "filler" for payload in websocket.sent)

    current = datetime(2026, 5, 26, 22, 31, 25)
    websocket.sent.clear()
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "지금은 31분",
            },
            active_speakers,
        )
    )
    correction_response = [payload for payload in websocket.sent if payload.get("type") == "response"][-1]
    assert correction_response["fast_path"] == "medication_taken_time_correction"
    assert "오후 10시 31분" in correction_response["response_text"]

    websocket.sent.clear()
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "언제 먹었다고",
            },
            active_speakers,
        )
    )
    final_recall = [payload for payload in websocket.sent if payload.get("type") == "response"][-1]
    assert final_recall["fast_path"] == "medication_taken_recall"
    assert "오후 10시 31분" in final_recall["response_text"]


def test_agent_ws_profile_memory_ack_does_not_use_medication_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("profile memory acknowledgement should not call identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("profile memory acknowledgement should not call orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "알았어 나 잘 기억해 줘",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    assert websocket.sent[-1]["fast_path"] == "profile_memory_ack"
    assert websocket.sent[-1]["response_type"] == "profile_memory_ack"
    assert "앞으로 김영수님 정보로 잘 기억하겠습니다" in websocket.sent[-1]["response_text"]
    assert "남성, 72세" not in websocket.sent[-1]["response_text"]
    assert "타이레놀" not in websocket.sent[-1]["response_text"]
    assert "의사·약사" not in websocket.sent[-1]["response_text"]


def test_agent_ws_current_profile_negation_prompts_registration_without_filler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    speaker_id = "speaker-kim"
    run(
        fake_memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("profile negation should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)
    agent_ws._wake_profile_cache_by_speaker[speaker_id] = {
        "name": "김영수",
        "age": "72",
        "gender": "남성",
    }

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": speaker_id,
                "text": "아니야 나 김영수 아니야",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["response_type"] == "identity_check"
    assert response_payloads[-1]["identity_gate"]["reason"] == "identity_rejected_needs_registration"
    assert "김영수님으로 보지 않겠습니다" in response_payloads[-1]["response_text"]
    assert "새로 등록할 이름, 나이, 성별" in response_payloads[-1]["response_text"]
    assert speaker_id not in agent_ws._wake_profile_cache_by_speaker

    websocket.sent.clear()
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": speaker_id,
                "text": "오디스",
            },
            set(),
        )
    )

    assert "김영수님" not in websocket.sent[-1]["response_text"]


def test_agent_ws_pending_registration_greeting_keeps_registration_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    speaker_id = "speaker-pending-registration"
    run(
        fake_memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(fake_memory.mark_identity_pending(speaker_id, "registration"))
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("pending registration greeting should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": speaker_id,
                "text": "안녕",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["response_type"] == "identity_check"
    assert response_payloads[-1]["identity_gate"]["reason"] == "needs_registration"
    assert "이름, 나이, 성별" in response_payloads[-1]["response_text"]
    assert response_payloads[-1].get("fast_path") != "smalltalk"

    websocket.sent.clear()
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": speaker_id,
                "text": "나 누군지 알아",
            },
            set(),
        )
    )

    recall_response = [payload for payload in websocket.sent if payload.get("type") == "response"][-1]
    assert recall_response["identity_gate"]["reason"] == "needs_registration"
    assert "아직" in recall_response["response_text"]
    assert "누구신지 모릅니다" in recall_response["response_text"]


def test_agent_ws_short_rejection_does_not_send_general_filler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "정현기", "age": "23", "gender": "남성"}},
        )

    async def fake_run_turn(**kwargs):
        return SimpleNamespace(
            filler_text="",
            conversation=SimpleNamespace(
                requires_tts=False,
                response_text="",
                response_type="ignored",
            ),
            execution_results={"task_results": {}},
            decision=SimpleNamespace(rationale="unknown", intent="unknown"),
            core_message="",
            judge_review={},
        )

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fake_run_turn)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-hyun",
                "text": "아니",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)


def test_medication_reasoning_sends_filler_before_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    async def fake_run_turn(**kwargs):
        assert websocket.sent
        assert websocket.sent[0]["type"] == "filler"
        return SimpleNamespace(
            filler_text="",
            conversation=SimpleNamespace(
                requires_tts=True,
                response_text="김영수님, 현재 기록 기준으로 혈압약을 확인하고 답변드릴게요.",
                response_type="medical_response",
            ),
            execution_results={"task_results": {}},
            decision=SimpleNamespace(rationale="test", intent="medication_query"),
            core_message="",
            judge_review={},
        )

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fake_run_turn)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스, 혈압약 두 번 먹으면 더 빨리 좋아져?",
            },
            set(),
        )
    )

    assert websocket.sent[0]["type"] == "filler"
    assert websocket.sent[0]["stage"] == "dur"
    assert websocket.sent[0]["requires_tts"] is True
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "medical_response"


def test_progress_fillers_send_only_one_late_update(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = FakeWebSocket()
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(agent_ws.asyncio, "sleep", fake_sleep)

    run(
        agent_ws._send_progress_fillers(
            websocket,
            "오디스, 혈압약 두 번 먹으면 더 빨리 좋아져?",
            initial_sent=True,
        )
    )

    assert delays == [6.0]
    assert len(websocket.sent) == 1
    assert {payload["stage"] for payload in websocket.sent} == {"dur"}
    assert all(payload["type"] == "filler" for payload in websocket.sent)
    assert "잠시만 기다려주세요" in websocket.sent[0]["text"]


def test_ocr_progress_fillers_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = FakeWebSocket()
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(agent_ws.asyncio, "sleep", fake_sleep)

    run(agent_ws._send_ocr_progress_fillers(websocket))

    assert delays == [5.0, 8.0]
    assert len(websocket.sent) == 2
    assert {payload["stage"] for payload in websocket.sent} == {"ocr_processing"}
    assert all(payload["type"] == "filler" for payload in websocket.sent)
    assert all("잠시만 기다려주세요" in payload["text"] for payload in websocket.sent)
