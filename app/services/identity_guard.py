"""Identity gate for multi-speaker WebSocket conversations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from app.engines.memory import MemoryEngine
from app.services.llm import extract_identity_profile_with_llm, judge_identity_conflict


IDENTITY_REVERIFY_WINDOW_SECONDS = 5 * 60


@dataclass(frozen=True)
class IdentityGateResult:
    allowed: bool
    reason: str
    response_text: str = ""
    response_type: str = "identity_check"
    metadata: dict[str, Any] | None = None


async def evaluate_identity_gate(
    *,
    memory_engine: MemoryEngine,
    text: str,
    speaker_id: Optional[str],
    now: Optional[datetime] = None,
) -> IdentityGateResult:
    """Return whether a turn may proceed to the main engine pipeline."""
    if not speaker_id:
        return IdentityGateResult(
            allowed=True,
            reason="no_speaker_id",
            metadata={"speaker_id": None},
        )

    current_time = now or datetime.now()
    state = await memory_engine.load_identity_state(speaker_id)
    profile = state.get("profile") or {}
    heuristic_identity = memory_engine.extract_identity_from_text(text)
    pending_action = state.get("pending_identity_action") or ""
    identity_extract: dict[str, Any] = {
        "profile": {},
        "source": "heuristic_fast_path",
        "raw": "",
    }
    if _should_call_identity_extract_llm(
        text=text,
        state=state,
        heuristic_identity=heuristic_identity,
        pending_action=pending_action,
    ):
        identity_extract = await extract_identity_profile_with_llm(current_text=text)
    llm_identity = identity_extract.get("profile") or {}
    identity_update = _merge_identity_updates(heuristic_identity, llm_identity)

    if not state.get("exists"):
        if _has_identity_core(identity_update):
            saved = await memory_engine.save_identity_profile(
                speaker_id,
                identity_update,
                mark_verified=True,
            )
            await memory_engine.update_flash_profile(speaker_id, saved)
            return IdentityGateResult(
                allowed=False,
                reason="identity_registered",
                response_text=_registration_completed(saved),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": saved,
                    "identity_extract": identity_extract,
                    "saved_state": saved,
                },
            )
        await memory_engine.mark_identity_pending(speaker_id, "registration")
        return IdentityGateResult(
            allowed=False,
            reason="needs_registration",
            response_text=_registration_question(),
            metadata={
                "speaker_id": speaker_id,
                "profile": profile,
                "identity_update": identity_update,
                "identity_extract": identity_extract,
            },
        )

    if pending_action == "confirm_new_identity":
        candidate = state.get("pending_identity_candidate") or {}
        if _is_affirmative(text):
            saved = await memory_engine.confirm_identity_candidate(speaker_id)
            return IdentityGateResult(
                allowed=False,
                reason="identity_candidate_registered",
                response_text=_registration_completed(saved),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": saved,
                    "identity_candidate": candidate,
                },
            )
        if _has_identity_core(identity_update):
            saved = await memory_engine.mark_identity_candidate(
                speaker_id,
                identity_update,
                action="confirm_new_identity",
            )
            return IdentityGateResult(
                allowed=False,
                reason="confirm_new_identity",
                response_text=_confirm_new_identity_question(identity_update),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": profile,
                    "identity_candidate": identity_update,
                    "identity_extract": identity_extract,
                    "saved_state": saved,
                },
            )
        await memory_engine.mark_identity_pending(speaker_id, "registration")
        return IdentityGateResult(
            allowed=False,
            reason="needs_registration",
            response_text=_registration_question(),
            metadata={"speaker_id": speaker_id, "profile": profile, "identity_candidate": candidate},
        )

    if pending_action == "registration":
        if _has_identity_core(identity_update):
            saved = await memory_engine.save_identity_profile(
                speaker_id,
                identity_update,
                mark_verified=True,
            )
            await memory_engine.update_flash_profile(speaker_id, saved)
            return IdentityGateResult(
                allowed=False,
                reason="identity_registered",
                response_text=_registration_completed(saved),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": saved,
                    "identity_extract": identity_extract,
                    "saved_state": saved,
                },
            )
        return IdentityGateResult(
            allowed=False,
            reason="needs_registration",
            response_text=_registration_question(),
            metadata={"speaker_id": speaker_id, "profile": profile},
        )

    if pending_action in {"reverification", "identity_conflict"}:
        if _is_affirmative(text):
            if pending_action == "identity_conflict" and state.get("pending_identity_candidate"):
                saved = await memory_engine.confirm_identity_candidate(speaker_id)
                reason = "identity_candidate_registered"
                response_text = _registration_completed(saved)
            else:
                saved = await memory_engine.mark_identity_seen(speaker_id, verified=True)
                reason = "identity_reverified"
                response_text = _reverified_message(saved)
            return IdentityGateResult(
                allowed=False,
                reason=reason,
                response_text=response_text,
                metadata={"speaker_id": speaker_id, "profile": saved},
            )
        if _has_identity_core(identity_update):
            saved = await memory_engine.mark_identity_candidate(
                speaker_id,
                identity_update,
                action="confirm_new_identity",
            )
            return IdentityGateResult(
                allowed=False,
                reason="confirm_new_identity",
                response_text=_confirm_new_identity_question(identity_update),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": profile,
                    "identity_candidate": identity_update,
                    "identity_extract": identity_extract,
                    "saved_state": saved,
                },
            )
        return IdentityGateResult(
            allowed=False,
            reason=pending_action,
            response_text=_reverify_question(profile, conflict=pending_action == "identity_conflict"),
            metadata={"speaker_id": speaker_id, "profile": profile},
        )

    if not _has_profile_identity(profile):
        if _has_identity_core(identity_update):
            saved = await memory_engine.save_identity_profile(
                speaker_id,
                identity_update,
                mark_verified=True,
            )
            await memory_engine.update_flash_profile(speaker_id, saved)
            return IdentityGateResult(
                allowed=False,
                reason="identity_registered",
                response_text=_registration_completed(saved),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": saved,
                    "identity_extract": identity_extract,
                    "saved_state": saved,
                },
            )
        await memory_engine.mark_identity_pending(speaker_id, "registration")
        return IdentityGateResult(
            allowed=False,
            reason="needs_registration",
            response_text=_registration_question(),
            metadata={"speaker_id": speaker_id, "profile": profile},
        )

    last_seen = _parse_datetime(state.get("last_seen_at"))
    if last_seen and (current_time - last_seen).total_seconds() > IDENTITY_REVERIFY_WINDOW_SECONDS:
        await memory_engine.mark_identity_pending(speaker_id, "reverification")
        return IdentityGateResult(
            allowed=False,
            reason="needs_reverification",
            response_text=_reverify_question(profile, conflict=False),
            metadata={
                "speaker_id": speaker_id,
                "profile": profile,
                "last_seen_at": state.get("last_seen_at"),
                "current_time": current_time.isoformat(timespec="seconds"),
            },
        )

    judge: dict[str, Any] = {
        "conflict": _heuristic_profile_conflict(identity_update, profile),
        "source": "heuristic_fast_path",
    }
    if not judge["conflict"] and _should_call_identity_conflict_llm(text, identity_update):
        history = await memory_engine.store.read_user_file(speaker_id, "history.md")
        judge = await judge_identity_conflict(
            current_text=text,
            patient_profile=profile,
            recent_history=history[:1200],
            current_time=current_time.isoformat(timespec="seconds"),
        )
    if judge.get("conflict"):
        if _has_identity_core(identity_update):
            saved = await memory_engine.mark_identity_candidate(
                speaker_id,
                identity_update,
                action="identity_conflict",
            )
            response_text = _confirm_new_identity_question(identity_update, existing_profile=profile)
            metadata: dict[str, Any] = {
                "speaker_id": speaker_id,
                "profile": profile,
                "identity_candidate": identity_update,
                "identity_extract": identity_extract,
                "judge": judge,
                "saved_state": saved,
            }
        else:
            await memory_engine.mark_identity_pending(speaker_id, "identity_conflict")
            response_text = _reverify_question(profile, conflict=True)
            metadata = {"speaker_id": speaker_id, "profile": profile, "judge": judge}
        return IdentityGateResult(
            allowed=False,
            reason="identity_conflict",
            response_text=response_text,
            metadata=metadata,
        )

    return IdentityGateResult(
        allowed=True,
        reason="identity_verified",
        metadata={"speaker_id": speaker_id, "profile": profile, "judge": judge},
    )


def _has_profile_identity(profile: dict[str, Any]) -> bool:
    return bool(profile.get("name") and (profile.get("age") or profile.get("gender")))


def _has_identity_core(profile: dict[str, Any]) -> bool:
    return bool(profile.get("name") and (profile.get("age") or profile.get("gender")))


def _merge_identity_updates(
    heuristic_identity: dict[str, Any],
    llm_identity: dict[str, Any],
) -> dict[str, Any]:
    merged = {**heuristic_identity, **{key: value for key, value in llm_identity.items() if value}}
    conditions: list[Any] = []
    for source in (heuristic_identity, llm_identity):
        raw_conditions = source.get("conditions") or []
        if isinstance(raw_conditions, list):
            for condition in raw_conditions:
                if condition and condition not in conditions:
                    conditions.append(condition)
    if conditions:
        merged["conditions"] = conditions
    return merged


def _should_call_identity_extract_llm(
    *,
    text: str,
    state: dict[str, Any],
    heuristic_identity: dict[str, Any],
    pending_action: str,
) -> bool:
    """Avoid LLM identity extraction on ordinary conversation turns."""
    if _has_identity_core(heuristic_identity):
        return False
    if not _has_identity_text_signal(text):
        return False
    if pending_action in {"registration", "confirm_new_identity", "identity_conflict"}:
        return True
    if not state.get("exists") or not _has_profile_identity(state.get("profile") or {}):
        return True
    return bool(heuristic_identity)


def _should_call_identity_conflict_llm(
    text: str,
    identity_update: dict[str, Any],
) -> bool:
    if identity_update:
        return True
    return any(token in text for token in ("보호자", "대신", "아버지", "어머니", "다른 사람"))


def _has_identity_text_signal(text: str) -> bool:
    import re

    return bool(
        re.search(r"\d{1,3}\s*(?:살|세)", text)
        or any(
            token in text
            for token in (
                "제 이름",
                "저는",
                "나는",
                "남자",
                "남성",
                "여자",
                "여성",
                "고혈압",
                "당뇨",
                "천식",
                "신장질환",
                "간질환",
                "심장질환",
            )
        )
    )


def _heuristic_profile_conflict(
    identity_update: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    if not identity_update or not profile:
        return False
    for key in ("name", "age", "gender"):
        incoming = str(identity_update.get(key) or "").strip()
        existing = str(profile.get(key) or "").strip()
        if incoming and existing and incoming != existing:
            return True
    return False


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _is_affirmative(text: str) -> bool:
    stripped = text.strip().lower()
    return any(token in stripped for token in ("네", "예", "맞아", "맞습니다", "응", "그래", "본인"))


def _registration_question() -> str:
    return (
        "안녕하세요. 처음 뵙는 분인 것 같아요. "
        "복약 안내를 도와드리기 위해 이름, 성별, 나이를 말씀해 주세요."
    )


def _confirm_new_identity_question(
    candidate: dict[str, Any],
    *,
    existing_profile: dict[str, Any] | None = None,
) -> str:
    name = candidate.get("name") or "새 대상자"
    details = []
    if candidate.get("age"):
        details.append(f"{candidate['age']}세")
    if candidate.get("gender"):
        details.append(str(candidate["gender"]))
    conditions = candidate.get("conditions") or []
    if conditions:
        details.append("기저질환 " + ", ".join(str(item) for item in conditions))
    detail_text = f" ({', '.join(details)})" if details else ""
    if existing_profile and existing_profile.get("name"):
        return (
            f"현재 저장된 분은 {existing_profile['name']}님인데, "
            f"새로 {name}님{detail_text} 정보가 들렸어요. "
            "이 신원으로 새로 등록하거나 현재 화자 정보로 바꿔도 될까요?"
        )
    return f"{name}님{detail_text}으로 들렸어요. 이 신원으로 등록해도 될까요?"


def _reverify_question(profile: dict[str, Any], *, conflict: bool) -> str:
    name = profile.get("name") or "등록된 분"
    if conflict:
        return (
            f"지금 말씀하신 내용이 저장된 {name}님 정보와 조금 달라 보여요. "
            "본인이 맞으신지, 아니면 다른 분이 말씀 중인지 확인해 주세요."
        )
    return (
        f"마지막 대화 후 시간이 지나서 다시 확인할게요. "
        f"지금 말씀하시는 분이 {name}님 본인이 맞으신가요?"
    )


def _registration_completed(profile: dict[str, Any]) -> str:
    name = profile.get("name") or "사용자"
    details = []
    if profile.get("gender"):
        details.append(str(profile["gender"]))
    if profile.get("age"):
        details.append(f"{profile['age']}세")
    detail_text = ", ".join(details)
    if detail_text:
        return (
            f"알겠습니다. {name}님, {detail_text}로 기억하겠습니다. "
            f"앞으로 복약 정보와 상담 내용을 {name}님 기준으로 안내드릴게요."
        )
    return f"알겠습니다. {name}님으로 기억하겠습니다. 앞으로 {name}님 기준으로 안내드릴게요."


def _reverified_message(profile: dict[str, Any]) -> str:
    name = profile.get("name") or "사용자"
    return f"{name}님으로 확인했습니다. 이어서 말씀해 주세요."
