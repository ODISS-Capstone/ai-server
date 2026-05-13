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
    identity_extract = await extract_identity_profile_with_llm(current_text=text)
    identity_update = identity_extract.get("profile") or memory_engine.extract_identity_from_text(text)
    pending_action = state.get("pending_identity_action") or ""

    if not state.get("exists"):
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
        "처음 등록하는 분이라 신원 확인이 필요해요. "
        "성함, 나이, 성별, 그리고 주요 기저질환이 있으면 함께 말씀해 주세요."
    )


def _confirm_new_identity_question(
    candidate: dict[str, Any],
    *,
    existing_profile: dict[str, Any] | None = None,
) -> str:
    name = candidate.get("name") or "새 환자"
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
    name = profile.get("name") or "등록된 환자"
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
    name = profile.get("name") or "어르신"
    return f"{name}님 신원 정보를 등록했어요. 이제 복약 상담을 이어가셔도 됩니다."


def _reverified_message(profile: dict[str, Any]) -> str:
    name = profile.get("name") or "어르신"
    return f"{name}님으로 확인했습니다. 이어서 말씀해 주세요."
