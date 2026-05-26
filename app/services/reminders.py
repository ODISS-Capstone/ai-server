"""Medication reminder scheduling for ODISS WebSocket sessions."""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.config import settings
from app.engines.memory import MemoryEngine

SendCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

DEFAULT_REMINDER_TIMES = {
    "아침": "08:00",
    "점심": "13:00",
    "저녁": "19:00",
}
REMINDER_PENDING_TTL = timedelta(minutes=5)


@dataclass
class ReminderSchedule:
    speaker_id: str
    times: dict[str, str]
    display_name: str = ""
    medication_label: str = "식후 약"
    next_runs: dict[str, str] | None = None
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "times": self.times,
            "display_name": self.display_name,
            "medication_label": self.medication_label,
            "next_runs": self.next_runs or {},
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReminderSchedule":
        return cls(
            speaker_id=str(payload.get("speaker_id") or ""),
            times={
                meal: str(value)
                for meal, value in (payload.get("times") or {}).items()
                if meal in DEFAULT_REMINDER_TIMES and value
            },
            display_name=str(payload.get("display_name") or ""),
            medication_label=str(payload.get("medication_label") or "식후 약"),
            next_runs={
                meal: str(value)
                for meal, value in (payload.get("next_runs") or {}).items()
                if meal in DEFAULT_REMINDER_TIMES and value
            },
            active=bool(payload.get("active", True)),
        )


class ReminderService:
    """Small in-process scheduler for demo medication reminders."""

    def __init__(
        self,
        *,
        now_provider: Optional[Callable[[], datetime]] = None,
        start_background_tasks: bool = True,
    ) -> None:
        self._timezone = self._load_timezone()
        self._now = now_provider or (lambda: datetime.now(self._timezone))
        self._start_background_tasks = start_background_tasks
        self._callbacks: dict[str, SendCallback] = {}
        self._tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._pending: dict[str, ReminderSchedule] = {}
        self._pending_started_at: dict[str, datetime] = {}
        self._active: dict[str, ReminderSchedule] = {}
        self._memory: dict[str, MemoryEngine] = {}
        self._restored: set[str] = set()
        self._last_prompt: dict[str, dict[str, str]] = {}

    def register_connection(self, speaker_id: Optional[str], callback: SendCallback) -> None:
        if speaker_id:
            self._callbacks[speaker_id] = callback

    def unregister_connection(self, speaker_id: Optional[str]) -> None:
        if not speaker_id:
            return
        self._callbacks.pop(speaker_id, None)
        self._pending.pop(speaker_id, None)
        self._pending_started_at.pop(speaker_id, None)
        for key in [key for key in self._tasks if key[0] == speaker_id]:
            self._tasks.pop(key).cancel()
        self._restored.discard(speaker_id)

    async def restore_for_speaker(
        self,
        memory_engine: MemoryEngine,
        speaker_id: Optional[str],
    ) -> None:
        if not speaker_id or speaker_id in self._restored:
            return
        self._restored.add(speaker_id)
        self._memory[speaker_id] = memory_engine
        schedule = await self.load_schedule(memory_engine, speaker_id)
        if schedule and schedule.active:
            self._active[speaker_id] = schedule
            if self._start_background_tasks:
                self._start_tasks(schedule)

    async def handle_user_text(
        self,
        *,
        memory_engine: MemoryEngine,
        speaker_id: Optional[str],
        text: str,
        user_profile: Optional[dict[str, Any]] = None,
        prescription_log: str = "",
    ) -> Optional[str]:
        if not speaker_id:
            return None
        self._memory[speaker_id] = memory_engine
        stripped = (text or "").strip()

        if speaker_id in self._pending:
            if self._is_pending_expired(speaker_id):
                self._pending.pop(speaker_id, None)
                self._pending_started_at.pop(speaker_id, None)
                if self.extract_time_overrides(stripped) or self._is_affirmative(stripped) or self._is_rejection(stripped):
                    return "이전 알림 설정 대기 시간이 지나 취소했습니다. 알림을 다시 설정하려면 새로 말씀해 주세요."
                return None
            if self._is_rejection(stripped):
                self._pending.pop(speaker_id, None)
                self._pending_started_at.pop(speaker_id, None)
                return "알겠습니다. 방금 시작한 복약 알림 설정은 취소했습니다."
            if self.extract_time_overrides(stripped) or self._is_affirmative(stripped):
                return await self.finalize_pending(
                    memory_engine=memory_engine,
                    speaker_id=speaker_id,
                    text=stripped,
                    user_profile=user_profile,
                )
            if self._is_wait_ack(stripped):
                return None
            return "복약 알림 설정을 계속할까요? 계속하려면 '네', 취소하려면 '아니'라고 말씀해 주세요."

        if self.is_taken_recall(stripped):
            return await self.recall_last_taken(
                memory_engine=memory_engine,
                speaker_id=speaker_id,
                user_profile=user_profile,
            )

        if self.is_taken_confirmation(stripped):
            return await self.record_taken(
                memory_engine=memory_engine,
                speaker_id=speaker_id,
                text=stripped,
                user_profile=user_profile,
            )

        if self.is_setup_request(stripped, prescription_log=prescription_log):
            return self.start_setup(
                speaker_id=speaker_id,
                user_profile=user_profile,
                prescription_log=prescription_log,
            )

        return None

    def start_setup(
        self,
        *,
        speaker_id: str,
        user_profile: Optional[dict[str, Any]] = None,
        prescription_log: str = "",
    ) -> str:
        medication_label = self._medication_label_from_context(prescription_log)
        name = self._name(user_profile)
        self._pending[speaker_id] = ReminderSchedule(
            speaker_id=speaker_id,
            times=dict(DEFAULT_REMINDER_TIMES),
            display_name=name,
            medication_label=medication_label,
        )
        self._pending_started_at[speaker_id] = self._now()
        return (
            f"네, {name}. 현재 저장된 복약 정보 기준으로 식후 복용 알림을 설정할 수 있습니다. "
            "기본 알림 시간은 아침은 오전 8시, 점심은 오후 1시, 저녁은 오후 7시로 설정하려고 하는데 괜찮으신가요?"
        )

    async def finalize_pending(
        self,
        *,
        memory_engine: MemoryEngine,
        speaker_id: str,
        text: str,
        user_profile: Optional[dict[str, Any]] = None,
        start_tasks: Optional[bool] = None,
    ) -> str:
        schedule = self._pending.pop(
            speaker_id,
            ReminderSchedule(speaker_id=speaker_id, times=dict(DEFAULT_REMINDER_TIMES)),
        )
        self._pending_started_at.pop(speaker_id, None)
        schedule.times.update(self.extract_time_overrides(text))
        name = self._name(user_profile)
        schedule.display_name = name
        schedule.next_runs = {
            meal: self.compute_next_run(time_text).isoformat(timespec="seconds")
            for meal, time_text in schedule.times.items()
        }
        schedule.active = True
        self._active[speaker_id] = schedule
        await self.save_schedule(memory_engine, schedule)
        if self._start_background_tasks if start_tasks is None else start_tasks:
            self._start_tasks(schedule)
        return (
            f"알겠습니다. {name}. 아침 약 알림은 {self._display_time(schedule.times['아침'])}, "
            f"점심 약 알림은 {self._display_time(schedule.times['점심'])}, "
            f"저녁 약 알림은 {self._display_time(schedule.times['저녁'])}로 설정하겠습니다. "
            "해당 시간이 되면 약 복용을 알려드리고, 복용하셨다고 말씀해주시면 기록해두겠습니다."
        )

    async def save_schedule(
        self,
        memory_engine: MemoryEngine,
        schedule: ReminderSchedule,
    ) -> None:
        content = (
            "# 복약 알림\n\n"
            "```json\n"
            + json.dumps(schedule.to_dict(), ensure_ascii=False, indent=2)
            + "\n```\n"
        )
        await memory_engine.store.save_user_file(schedule.speaker_id, "reminders.md", content)

    async def load_schedule(
        self,
        memory_engine: MemoryEngine,
        speaker_id: str,
    ) -> Optional[ReminderSchedule]:
        content = await memory_engine.store.read_user_file(speaker_id, "reminders.md")
        match = re.search(r"```json\s*(.*?)\s*```", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        schedule = ReminderSchedule.from_dict(payload)
        return schedule if schedule.speaker_id else None

    async def dispatch_due_reminders(self) -> list[dict[str, Any]]:
        """Dispatch due reminders immediately; useful for tests with fake clocks."""
        sent: list[dict[str, Any]] = []
        now = self._now()
        for schedule in list(self._active.values()):
            for meal, next_run_text in list((schedule.next_runs or {}).items()):
                next_run = self._parse_datetime(next_run_text, now=now)
                if next_run and next_run <= now:
                    payload = await self._send_reminder(schedule, meal)
                    if payload:
                        sent.append(payload)
                    schedule.next_runs[meal] = (next_run + timedelta(days=1)).isoformat(timespec="seconds")
                    memory_engine = self._memory.get(schedule.speaker_id)
                    if memory_engine:
                        await self.save_schedule(memory_engine, schedule)
        return sent

    async def record_taken(
        self,
        *,
        memory_engine: MemoryEngine,
        speaker_id: str,
        text: str,
        user_profile: Optional[dict[str, Any]] = None,
    ) -> str:
        now = self._now()
        prompt = self._last_prompt.get(speaker_id, {})
        explicit_medication = self._medication_from_text(text)
        bare_confirmation = self._is_bare_taken_confirmation(text)
        meal = self._meal_from_text(text) or (prompt.get("meal") if bare_confirmation else "") or "식후"
        medication_label = explicit_medication or (prompt.get("medication_label") if bare_confirmation else "")
        if not medication_label:
            medication_label = self._active.get(speaker_id, ReminderSchedule(speaker_id, {})).medication_label
        if not medication_label:
            medication_label = "식후 약"

        record = {
            "taken_at": now.isoformat(timespec="seconds"),
            "meal": meal,
            "medication_label": medication_label,
            "source_text": text,
        }
        existing = await memory_engine.store.read_user_file(speaker_id, "medication_taken.md")
        await memory_engine.store.save_user_file(
            speaker_id,
            "medication_taken.md",
            existing + "\n- " + json.dumps(record, ensure_ascii=False),
        )
        await memory_engine.store.save(
            "medication_log",
            (
                "# 복용 기록\n"
                f"> 기록 시각: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"- 사용자: {speaker_id}\n"
                f"- 식사 구분: {meal}\n"
                f"- 약: {medication_label}\n"
            ),
        )
        name = self._name(user_profile)
        return (
            f"알겠습니다. {name}이 오늘 {self._display_now(now)}에 "
            f"{meal} {medication_label}을 복용한 것으로 기록해두겠습니다."
        )

    async def recall_last_taken(
        self,
        *,
        memory_engine: MemoryEngine,
        speaker_id: str,
        user_profile: Optional[dict[str, Any]] = None,
    ) -> str:
        records = await self._load_taken_records(memory_engine, speaker_id)
        name = self._name(user_profile)
        if not records:
            return f"{name}, 아직 오늘 복용했다고 기록된 내용은 없습니다. 헷갈리시면 약봉투나 약통을 한 번 더 확인해 주세요."
        last = records[-1]
        taken_at = self._parse_datetime(last.get("taken_at", ""), now=self._now()) or self._now()
        meal = last.get("meal") or "식후"
        medication_label = last.get("medication_label") or "약"
        return (
            f"확인해보겠습니다. {name}은 오늘 {meal} {medication_label}을 복용했다고 말씀하셨습니다. "
            "다만 실제 복용 여부가 헷갈리시면 약봉투나 약통을 한 번 더 확인해 주세요."
        )

    @staticmethod
    def is_setup_request(text: str, *, prescription_log: str = "") -> bool:
        lowered = text.lower()
        compact = re.sub(r"[\s.?!,，。~]+", "", lowered)
        medication_context = any(token in lowered for token in ("약", "복용", "식후", "밥")) or bool(
            prescription_log.strip()
        )
        reminder_signal = any(
            token in lowered
            for token in (
                "알림",
                "알람",
                "예약",
                "깨워",
                "챙겨",
                "까먹",
                "잊어",
                "잊어버",
            )
        ) or any(token in compact for token in ("시간되면", "때되면", "먹을때", "먹을시간"))
        setup_signal = any(
            token in lowered
            for token in (
                "추가",
                "설정",
                "맞춰",
                "해줘",
                "해 줘",
                "해야",
                "알려",
                "말해",
                "깨워",
                "챙겨",
            )
        )
        if medication_context and reminder_signal and setup_signal:
            return True
        return (
            "알림" in lowered
            and any(token in lowered for token in ("약", "복용", "식후", "밥"))
            and any(token in lowered for token in ("추가", "설정", "해줘", "해 줘", "해야"))
        )

    @staticmethod
    def is_taken_confirmation(text: str) -> bool:
        stripped = text.strip().lower()
        if any(
            token in stripped
            for token in (
                "?",
                "먹어도",
                "괜찮",
                "되나",
                "돼",
                "문제",
                "위험",
                "못 먹",
                "깜빡",
                "헷갈",
                "기억",
                "어떡",
                "어쩌",
                "숨",
                "어지",
                "아파",
                "두 번",
                "한 번 더",
                "한번 더",
            )
        ):
            return False
        normalized = ReminderService._normalize_short_reply(stripped)
        if normalized in {"먹었어", "먹었어요", "먹었습니다", "복용했어", "복용했어요"}:
            return True
        return any(
            token in stripped
            for token in (
                "약 먹었어",
                "약 먹었어요",
                "복용했어",
                "복용했어요",
                "혈압약 먹었어",
                "혈압약 먹었어요",
            )
        )

    @staticmethod
    def is_taken_recall(text: str) -> bool:
        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "한 번 더",
                "한번 더",
                "먹을까",
                "먹어도",
                "괜찮",
                "되나",
                "돼",
                "문제",
                "위험",
            )
        ):
            return False
        return any(token in lowered for token in ("약 먹었나", "복용했나", "먹었는지", "먹었나?", "아까 약", "아까 먹"))

    @staticmethod
    def extract_time_overrides(text: str) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for meal in DEFAULT_REMINDER_TIMES:
            if meal not in text:
                continue
            match = re.search(
                rf"{meal}.*?(오전|오후)?\s*(\d{{1,2}})\s*시(?:\s*(\d{{1,2}})\s*분)?",
                text,
            )
            if match:
                overrides[meal] = ReminderService._normalize_time(
                    int(match.group(2)),
                    int(match.group(3) or 0),
                    match.group(1) or "",
                )
        return overrides

    def compute_next_run(self, time_text: str) -> datetime:
        hour, minute = [int(part) for part in time_text.split(":", 1)]
        now = self._now()
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    def _start_tasks(self, schedule: ReminderSchedule) -> None:
        for meal in schedule.times:
            key = (schedule.speaker_id, meal)
            if key in self._tasks:
                self._tasks.pop(key).cancel()
            self._tasks[key] = asyncio.create_task(self._run_loop(schedule.speaker_id, meal))

    async def _run_loop(self, speaker_id: str, meal: str) -> None:
        while True:
            schedule = self._active.get(speaker_id)
            if not schedule or not schedule.active or meal not in schedule.times:
                return
            now = self._now()
            next_run_text = (schedule.next_runs or {}).get(meal)
            next_run = self._parse_datetime(next_run_text, now=now) if next_run_text else self.compute_next_run(schedule.times[meal])
            delay = max(0.0, (next_run - now).total_seconds())
            await asyncio.sleep(delay)
            await self._send_reminder(schedule, meal)
            if schedule.next_runs is None:
                schedule.next_runs = {}
            schedule.next_runs[meal] = (next_run + timedelta(days=1)).isoformat(timespec="seconds")
            memory_engine = self._memory.get(speaker_id)
            if memory_engine:
                await self.save_schedule(memory_engine, schedule)

    async def _send_reminder(
        self,
        schedule: ReminderSchedule,
        meal: str,
    ) -> Optional[dict[str, Any]]:
        callback = self._callbacks.get(schedule.speaker_id)
        if not callback:
            return None
        recipient = schedule.display_name or "사용자님"
        text = (
            f"{recipient}, {self._display_time(schedule.times[meal])}가 되었습니다. "
            f"{meal} {schedule.medication_label}을 복용하실 시간입니다. "
            "약을 드신 뒤에는 \"먹었어\"라고 말씀해 주세요."
        )
        payload = {
            "type": "reminder",
            "text": text,
            "requires_tts": True,
            "speaker_id": schedule.speaker_id,
            "meal": meal,
        }
        self._last_prompt[schedule.speaker_id] = {
            "meal": meal,
            "medication_label": schedule.medication_label,
        }
        result = callback(payload)
        if inspect.isawaitable(result):
            await result
        return payload

    async def _load_taken_records(
        self,
        memory_engine: MemoryEngine,
        speaker_id: str,
    ) -> list[dict[str, Any]]:
        content = await memory_engine.store.read_user_file(speaker_id, "medication_taken.md")
        records: list[dict[str, Any]] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            try:
                payload = json.loads(stripped[2:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    @staticmethod
    def _normalize_time(hour: int, minute: int, meridiem: str = "") -> str:
        if meridiem == "오후" and hour < 12:
            hour += 12
        if meridiem == "오전" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _display_time(time_text: str) -> str:
        hour, minute = [int(part) for part in time_text.split(":", 1)]
        meridiem = "오전" if hour < 12 else "오후"
        display_hour = hour if 1 <= hour <= 12 else hour - 12 if hour > 12 else 12
        if minute:
            return f"{meridiem} {display_hour}시 {minute}분"
        return f"{meridiem} {display_hour}시"

    @staticmethod
    def _display_now(now: datetime) -> str:
        meridiem = "오전" if now.hour < 12 else "오후"
        display_hour = now.hour if 1 <= now.hour <= 12 else now.hour - 12 if now.hour > 12 else 12
        if now.minute:
            return f"{meridiem} {display_hour}시 {now.minute}분"
        return f"{meridiem} {display_hour}시"

    def _parse_datetime(self, value: Any, *, now: Optional[datetime] = None) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
        reference = now or self._now()
        if reference.tzinfo is not None and parsed.tzinfo is None:
            return parsed.replace(tzinfo=self._timezone)
        if reference.tzinfo is None and parsed.tzinfo is not None:
            return parsed.replace(tzinfo=None)
        return parsed

    @staticmethod
    def _load_timezone() -> tzinfo:
        try:
            return ZoneInfo(settings.app_timezone or "Asia/Seoul")
        except ZoneInfoNotFoundError:
            return timezone(timedelta(hours=9), "Asia/Seoul")

    @staticmethod
    def _name(user_profile: Optional[dict[str, Any]]) -> str:
        name = str((user_profile or {}).get("name") or "").strip()
        return f"{name}님" if name else "사용자님"

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        return any(token in text.strip().lower() for token in ("네", "예", "응", "그래", "괜찮", "좋아", "맞아"))

    @staticmethod
    def _is_wait_ack(text: str) -> bool:
        lowered = text.strip().lower()
        return any(token in lowered for token in ("알았어", "알겠습니다", "기다려", "잠시", "나중에"))

    def _is_pending_expired(self, speaker_id: str) -> bool:
        started_at = self._pending_started_at.get(speaker_id)
        if not started_at:
            return False
        return self._now() - started_at > REMINDER_PENDING_TTL

    @staticmethod
    def _is_rejection(text: str) -> bool:
        lowered = text.strip().lower()
        return any(token in lowered for token in ("아니", "아냐", "취소", "하지 마", "하지마", "싫", "필요 없어", "안 해"))

    @staticmethod
    def _is_bare_taken_confirmation(text: str) -> bool:
        return ReminderService._normalize_short_reply(text) in {
            "먹었어",
            "먹었어요",
            "먹었습니다",
            "복용했어",
            "복용했어요",
        }

    @staticmethod
    def _normalize_short_reply(text: str) -> str:
        return re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())

    @staticmethod
    def _medication_label_from_context(prescription_log: str) -> str:
        medication_names: list[str] = []
        for line in str(prescription_log or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            name = stripped[2:].strip()
            if name and name not in medication_names:
                medication_names.append(name)
        if any("혈압" in name for name in medication_names) or "혈압" in prescription_log:
            return "혈압약"
        if medication_names:
            if len(medication_names) <= 2:
                return ", ".join(medication_names)
            return ", ".join(medication_names[:2]) + f" 외 {len(medication_names) - 2}개 약"
        if "약품 목록" in prescription_log:
            return "식후 약"
        return "식후 약"

    @staticmethod
    def _meal_from_text(text: str) -> str:
        for meal in DEFAULT_REMINDER_TIMES:
            if meal in text:
                return meal
        return ""

    @staticmethod
    def _medication_from_text(text: str) -> str:
        if "혈압약" in text:
            return "혈압약"
        match = re.search(r"([가-힣A-Za-z0-9]+(?:정|장용정|캡슐|시럽))", text)
        return match.group(1) if match else ""
