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
    agent_ws._dialog_state_by_speaker.clear()


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


STATE_SHORT_UTTERANCE_CASES: list[dict] = []


def _add_short_utterance_cases(
    *,
    flow: str,
    expected: str,
    utterances: list[str],
    client_context: dict | None = None,
    last_assistant_text: str = "제가 방금 안내드렸습니다.",
) -> None:
    for text in utterances:
        STATE_SHORT_UTTERANCE_CASES.append(
            {
                "id": f"{flow}:{expected}:{text}",
                "flow": flow,
                "text": text,
                "expected": expected,
                "client_context": client_context or {},
                "last_assistant_text": last_assistant_text,
            }
        )


_add_short_utterance_cases(
    flow="none",
    expected="wake_word",
    utterances=["오디스", "오디세", "오티스", "오티즈", "오지스", "보리스", "보디스", "오 디 스", "야", "들려?"],
)
_add_short_utterance_cases(
    flow="assistant_social",
    expected="assistant_acknowledgement",
    utterances=["응", "어", "그래", "맞아", "어 맞아", "응 맞아", "네", "예", "아니", "아냐", "아니야", "아니요"],
)
_add_short_utterance_cases(
    flow="assistant_social",
    expected="assistant_repeat",
    utterances=["다시 말해줘", "한 번 더 말해줘", "못 들었어", "방금 뭐라고", "다시 알려줘"],
)
_add_short_utterance_cases(
    flow="assistant_social",
    expected="assistant_stop",
    utterances=["그만", "됐어", "잠깐만", "기다려", "멈춰"],
)
_add_short_utterance_cases(
    flow="ocr_camera",
    expected="camera_cancel",
    utterances=[
        "사진 안 찍는다고",
        "아니야 사진 안 찍어",
        "안 찍는다고",
        "안 찍어",
        "찍지 마",
        "찍지 말라고",
        "카메라 꺼",
        "카메라 꺼줘",
        "카메라 닫아",
        "사진 취소",
        "촬영 취소",
        "사진 필요 없어",
        "사진 그만",
        "됐어",
        "아니",
        "아니야",
        "그만",
        "필요없어",
    ],
    client_context={"camera_mode": "ready"},
)
_add_short_utterance_cases(
    flow="none",
    expected="camera_cancel",
    utterances=["사진 안 찍는다고", "카메라 꺼줘", "사진 취소", "촬영 그만", "약봉투 사진 안 찍어"],
)
_add_short_utterance_cases(
    flow="ocr_confirm",
    expected="ocr_save_confirm",
    utterances=["네", "응", "예", "그래", "좋아", "맞아", "확인", "저장해", "네 저장해", "그대로 저장"],
)
_add_short_utterance_cases(
    flow="ocr_confirm",
    expected="ocr_save_reject",
    utterances=["아니", "아냐", "싫어", "취소", "저장하지 마", "저장 안 해", "하지 마", "틀렸어", "다시 찍어", "재촬영", "삭제해", "다시"],
)
_add_short_utterance_cases(
    flow="ocr_confirm",
    expected="ocr_confirmation_followup",
    utterances=["감기 때문에 받은 약이야", "두통 때문에 먹는 약", "통증약이야", "처방받은 거야", "증상은 기침이야"],
)
_add_short_utterance_cases(
    flow="identity",
    expected="identity_followup",
    utterances=["응", "어", "네", "예", "맞아", "어 맞아", "맞습니다", "아니", "아니야", "아냐"],
)
_add_short_utterance_cases(
    flow="medication_guidance",
    expected="medication_taken_record",
    utterances=["먹었어", "어 먹었어", "약 먹었어", "방금 먹었어", "지금 먹었어", "복용했어", "먹었습니다", "다 먹었어", "먹음", "먹었어요"],
)
_add_short_utterance_cases(
    flow="medication_guidance",
    expected="medication_taken_recall",
    utterances=["언제 먹었지", "몇 시에 먹었지", "아까 언제 먹었어", "내가 약 먹었나", "먹은 기록 있어", "먹었다고 했지", "몇시에 먹었다고"],
)
_add_short_utterance_cases(
    flow="medication_guidance",
    expected="medication_guidance",
    utterances=["오늘 그거 먹어야 돼", "그거 먹어야 하나", "밥 먹었어", "저녁 먹었어", "식사 끝났어", "식후 약 뭐야"],
)
_add_short_utterance_cases(
    flow="reminder",
    expected="reminder_control",
    utterances=["아니", "아니야", "됐어", "그만", "잠깐만", "멈춰"],
)
_add_short_utterance_cases(
    flow="reminder",
    expected="missed_reminder_check",
    utterances=["왜 안 울려", "알람 안 왔어", "알림 안 왔어", "시간 지났어", "30초 지났음"],
)
_add_short_utterance_cases(
    flow="none",
    expected="assistant_suggestion",
    utterances=["뭐 하지", "뭐 하면 좋을까", "나 뭐 해야 돼", "도와줘", "뭘 하면 돼"],
)
_add_short_utterance_cases(
    flow="none",
    expected="assistant_capability",
    utterances=["너 뭐 할 수 있어", "뭐 도와줄 수 있어", "무엇을 도와줄 수 있어", "너는 뭐 해줘"],
)
_add_short_utterance_cases(
    flow="none",
    expected="presence",
    utterances=["어딨어", "어디 있어", "거기 있어"],
)
_add_short_utterance_cases(
    flow="none",
    expected="thanks",
    utterances=["고마워", "감사해", "땡큐"],
)
_add_short_utterance_cases(
    flow="none",
    expected="assistant_companion",
    utterances=["심심해", "말동무 해줘", "이야기하자"],
)


assert len(STATE_SHORT_UTTERANCE_CASES) >= 100


@pytest.mark.parametrize(
    "case",
    STATE_SHORT_UTTERANCE_CASES,
    ids=[case["id"] for case in STATE_SHORT_UTTERANCE_CASES],
)
def test_assistant_turn_router_state_short_utterance_table(case: dict) -> None:
    speaker_id = "short-utterance-table"
    state = agent_ws._speaker_state(speaker_id)
    state.active_flow = case["flow"]
    state.last_assistant_text = case["last_assistant_text"]
    state.last_response_type = "smalltalk"

    router = agent_ws.AssistantTurnRouter(
        text=case["text"],
        speaker_id=speaker_id,
        client_context=case["client_context"],
    )

    assert router.short_utterance_route() == case["expected"]


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


def test_agent_ws_presence_smalltalk_is_answered_without_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("assistant social turn should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("assistant social turn should not call the orchestrator")

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
                "text": "어딨어",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "smalltalk"
    assert websocket.sent[-1]["requires_tts"] is True
    assert websocket.sent[-1]["route_reason"] == "presence"
    assert "여기 있어요" in websocket.sent[-1]["response_text"]


def test_agent_ws_unsupported_request_is_answered_without_ignore(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("unsupported assistant turn should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("unsupported assistant turn should not call the orchestrator")

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
                "text": "오늘 뉴스 알려줘",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "smalltalk"
    assert websocket.sent[-1]["requires_tts"] is True
    assert "복약" in websocket.sent[-1]["response_text"]


def test_agent_ws_assistant_suggestion_uses_fast_path_without_identity_or_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("assistant suggestion should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("assistant suggestion should not call the orchestrator")

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
                "text": "아니 뭐 하면 좋을까",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "smalltalk"
    assert websocket.sent[-1]["requires_tts"] is True
    assert websocket.sent[-1]["fast_path"] == "smalltalk"
    assert websocket.sent[-1]["active_flow"] == "assistant_social"
    assert websocket.sent[-1]["route_reason"] == "assistant_suggestion"
    assert (
        "드실 약 확인" in websocket.sent[-1]["response_text"]
        or "식후 약 확인" in websocket.sent[-1]["response_text"]
    )
    assert "복약 알림" in websocket.sent[-1]["response_text"]
    assert "약봉투 사진" in websocket.sent[-1]["response_text"]


def test_agent_ws_anonymous_assistant_suggestion_does_not_force_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("anonymous assistant suggestion should not ask for registration")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("anonymous assistant suggestion should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": None,
                "text": "뭐 하지",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "smalltalk"
    assert "이름, 나이, 성별" not in websocket.sent[-1]["response_text"]
    assert "약봉투 사진" in websocket.sent[-1]["response_text"]


def test_agent_ws_pending_identity_suggestion_keeps_registration_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = make_real_memory(tmp_path)
    speaker_id = "speaker-pending-suggestion"
    run(
        fake_memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(fake_memory.mark_identity_pending(speaker_id, "registration"))
    websocket = FakeWebSocket()

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("pending identity assistant suggestion should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", ReminderService(start_background_tasks=False))
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": speaker_id,
                "text": "뭐 하면 좋을까",
            },
            set(),
        )
    )

    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["response_type"] == "identity_check"
    assert response_payloads[-1]["identity_gate"]["reason"] == "needs_registration"
    assert "먼저 이름, 나이, 성별" in response_payloads[-1]["response_text"]
    assert "복약 확인" in response_payloads[-1]["response_text"]
    assert response_payloads[-1].get("fast_path") != "smalltalk"


def test_agent_ws_repeat_control_repeats_last_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()
    state = agent_ws._speaker_state("speaker-repeat")
    state.last_assistant_text = "김영수님, 오늘 아침 식후 약은 혈압약입니다."
    state.last_response_type = "medical_response"
    state.last_response_can_repeat = True

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("repeat control should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("repeat control should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-repeat",
                "text": "다시 말해줘",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["fast_path"] == "assistant_repeat"
    assert websocket.sent[-1]["response_text"] == "김영수님, 오늘 아침 식후 약은 혈압약입니다."


def test_agent_ws_colloquial_affirmative_ack_does_not_fallback_to_unclear(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()
    state = agent_ws._speaker_state("speaker-ack")
    state.active_flow = "assistant_social"
    state.last_assistant_text = "제가 도와드릴 수 있는 건 세 가지예요."
    state.last_response_type = "smalltalk"

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("colloquial acknowledgement should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("colloquial acknowledgement should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-ack",
                "text": "어 맞아",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["fast_path"] == "assistant_acknowledgement"
    assert websocket.sent[-1]["route_reason"] == "affirmative"
    assert "짧게" not in websocket.sent[-1]["response_text"]
    assert "다시" not in websocket.sent[-1]["response_text"]


def test_agent_ws_camera_cancel_understands_photo_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()
    state = agent_ws._speaker_state("speaker-camera")
    state.active_flow = "ocr"
    state.last_assistant_text = "약봉투를 화면에 맞춰주세요."
    state.last_response_type = "ocr_request"

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("camera cancel should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("camera cancel should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-camera",
                "text": "사진 안 찍는다고",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["fast_path"] == "assistant_camera_cancel"
    assert websocket.sent[-1]["route_reason"] == "camera_cancel"
    assert websocket.sent[-1]["ui_action"] == "close_camera"
    assert websocket.sent[-1]["active_flow"] == "none"
    assert websocket.sent[-1]["response_text"] == "네, 사진 확인을 중단할게요."


def test_agent_ws_camera_cancel_uses_client_camera_state_without_server_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("camera cancel should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("camera cancel should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-camera-client",
                "text": "아니야 사진 안 찍는다고",
                "client_context": {"camera_mode": "ready"},
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    assert websocket.sent[-1]["fast_path"] == "assistant_camera_cancel"
    assert websocket.sent[-1]["ui_action"] == "close_camera"


def test_agent_ws_camera_cancel_handles_omitted_photo_word_when_camera_is_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("camera cancel should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("camera cancel should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-camera-omitted",
                "text": "안 찍는다고",
                "client_context": {"camera_mode": "ready"},
            },
            set(),
        )
    )

    assert websocket.sent[-1]["fast_path"] == "assistant_camera_cancel"
    assert websocket.sent[-1]["ui_action"] == "close_camera"


def test_agent_ws_medication_guidance_followup_records_short_taken_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()
    state = agent_ws._speaker_state("speaker-guided-med")
    state.active_flow = "medication_guidance"
    state.last_guided_medication = "혈압약"
    state.last_medication_candidates = ["혈압약"]
    state.last_assistant_text = "김영수님, 저장된 약은 혈압약입니다."

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("guided medication taken confirmation should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-guided-med",
                "text": "어 먹었어",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["fast_path"] == "medication_taken_record"
    assert "기록" in websocket.sent[-1]["response_text"]


@pytest.mark.parametrize(
    "wake_text",
    ["오디세", "오디", "오티스", "오티즈", "오지스", "보리스", "보디스", "오 디 스", "야", "들려?"],
)
def test_agent_ws_wake_word_stt_variant_uses_wake_fast_path(
    monkeypatch: pytest.MonkeyPatch,
    wake_text: str,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("STT wake-word variant should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("STT wake-word variant should not call the orchestrator")

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
                "text": wake_text,
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "wake_word_ack"
    assert websocket.sent[-1]["response_text"] == "네, 김영수님. 말씀하세요."


def test_agent_ws_profile_recall_skips_medication_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("profile recall should not wait for identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("profile recall should not call the orchestrator")

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
                "text": "지금 나 누군지 알아",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "profile_recall"
    assert websocket.sent[-1]["fast_path"] == "profile_recall"
    assert "김영수님" in websocket.sent[-1]["response_text"]
    assert "남성" in websocket.sent[-1]["response_text"]
    assert "72세" in websocket.sent[-1]["response_text"]
    assert "임의로" not in websocket.sent[-1]["response_text"]
    assert "약봉투" not in websocket.sent[-1]["response_text"]


def test_agent_ws_turn_id_is_echoed_and_diagnostic_log_is_saved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_memory = FakeMemoryEngine()
    diagnostic_store = MDStore(str(tmp_path / "md_database"))
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("wake-word turn should not call the identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("wake-word turn should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "md_store", diagnostic_store)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "session_id": "session-web",
                "turn_id": "turn-web-1",
                "text": "오디스",
                "client_context": {"source": "speech", "speaking": False},
            },
            set(),
        )
    )

    payload = websocket.sent[-1]
    assert payload["turn_id"] == "turn-web-1"
    assert payload["session_id"] == "session-web"
    assert isinstance(payload["ws_elapsed_ms"], int)

    entries = run(diagnostic_store.list_entries("assistant_diagnostics"))
    assert len(entries) == 1
    content = run(diagnostic_store.read_entry(entries[0]))
    assert "turn-web-1" in content
    assert "오디스" in content
    assert "wake_word_ack" in content


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


@pytest.mark.parametrize("text", ["가슴이 답답해", "어 나 갑자기 가슴이 아파 어떡하지"])
def test_symptom_utterance_does_not_use_smalltalk_fast_path(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fake_identity_gate(**kwargs):
        raise AssertionError("emergency precheck should not wait for identity gate")

    async def fake_run_turn(**kwargs):
        raise AssertionError("emergency precheck should not wait for orchestrator")

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
                "text": text,
            },
            set(),
        )
    )

    assert websocket.sent[-1]["response_type"] == "medical_response"
    assert websocket.sent[-1]["route_label"] == "emergency"
    assert websocket.sent[-1]["engine_scope"] == "safety"
    assert websocket.sent[-1]["fast_path"] == "global_safety_precheck"
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
            # 지연 필러 계약: 처리 시간이 INITIAL_FILLER_DELAY_SEC을 넘기면 필러가 발화된다.
            await asyncio.sleep(0.05)
            assert websocket.sent
            assert websocket.sent[0]["type"] == "filler"
            return "네, 김영수님. 현재 저장된 복약 정보 기준으로 식후 복용 알림을 설정할 수 있습니다."

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", SlowReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws, "INITIAL_FILLER_DELAY_SEC", 0.01)

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


def test_agent_ws_simple_tylenol_suitability_is_short_not_overdose(
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
    websocket = FakeWebSocket()
    reminder_service = ReminderService(start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("simple medication suitability should use the fast path")

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
                "text": "내가 지금 타이레놀 먹어도 될까",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "medication_safety_fast_path"
    answer = response_payloads[-1]["response_text"]
    assert "타이레놀" in answer
    assert "용량" in answer
    assert "시간" in answer
    assert "여러 알" not in answer
    assert "저혈압" not in answer
    assert "119" not in answer
    assert len(answer) < 150


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


@pytest.mark.parametrize(
    "user_text",
    [
        "나 밥 먹고 나서 타이레놀 먹어야 되는데 알림 해 줄 수 있어",
        "어 나 밥 먹고 오면은 타이레놀 먹으라고 알려 줘",
    ],
)
def test_agent_ws_named_after_meal_medication_guidance_infers_breakfast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    user_text: str,
) -> None:
    memory = MemoryEngine()
    memory.store = MDStore(str(tmp_path / "md_database"))
    memory.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(memory.initialize())
    run(
        memory.save_identity_profile(
            "speaker-kim",
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
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
        raise AssertionError("named after-meal medication guidance should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)
    monkeypatch.setattr(agent_ws, "_meal_hint_from_current_time", lambda now=None: "아침")
    monkeypatch.setattr(agent_ws, "_current_time_phrase", lambda now=None: "오전 8시 9분")

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": user_text,
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "named_meal_medication_guidance"
    answer = response_payloads[-1]["response_text"]
    assert "오전 8시 9분" in answer
    assert "아침 식사 후" in answer
    assert "타이레놀" in answer
    assert "밥 먹었어" in answer
    assert "먹었어" in answer
    prescription_log = run(memory.store.read_flash("prescription_log"))
    assert "타이레놀" in prescription_log


def test_agent_ws_after_meal_completion_gives_direct_take_instruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    memory = MemoryEngine()
    memory.store = MDStore(str(tmp_path / "md_database"))
    memory.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(memory.initialize())
    run(
        memory.save_identity_profile(
            "speaker-kim",
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 타이레놀\n",
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
        raise AssertionError("after-meal completion guidance should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)
    monkeypatch.setattr(agent_ws, "_meal_hint_from_current_time", lambda now=None: "아침")

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "너 나 밥 먹었어",
            },
            set(),
        )
    )

    assert not any(payload.get("type") == "filler" for payload in websocket.sent)
    response_payloads = [payload for payload in websocket.sent if payload.get("type") == "response"]
    assert response_payloads[-1]["fast_path"] == "stored_medication_guidance"
    answer = response_payloads[-1]["response_text"]
    assert "아침 식사" in answer
    assert "타이레놀" in answer
    assert "드시면 됩니다" in answer
    assert "먹었어" in answer


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
        # 지연 필러 계약: 오케스트레이터가 INITIAL_FILLER_DELAY_SEC보다 오래 걸리면 필러가 발화된다.
        await asyncio.sleep(0.05)
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
    monkeypatch.setattr(agent_ws, "INITIAL_FILLER_DELAY_SEC", 0.01)

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
