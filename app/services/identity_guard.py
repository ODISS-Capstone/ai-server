"""Identity gate for multi-speaker WebSocket conversations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Optional

from app.engines.memory import MemoryEngine
from app.services.llm import (
    extract_identity_profile_with_llm,
    judge_identity_conflict,
    judge_pending_identity_reply_with_llm,
    judge_prior_conversation_turn,
)
from app.services.medication_extraction import is_wake_word_only


IDENTITY_REVERIFY_WINDOW_SECONDS = 5 * 60
IDENTITY_PENDING_TIMEOUT_SECONDS = 5 * 60

PROFILE_RECALL_TOKENS = (
    "내 이름",
    "내 프로필",
    "기저질환",
    "내가 누구",
    "누구인지",
)


def is_profile_recall_query(text: str) -> bool:
    raw = text or ""
    compact = re.sub(r"\s+", "", raw)
    return any(token in raw for token in PROFILE_RECALL_TOKENS) or any(
        token in compact
        for token in (
            "내이름",
            "내프로필",
            "내가누구",
            "누군지",
            "누구인지",
            "나누구",
            "저누구",
        )
    )


def has_profile_identity(profile: dict[str, Any]) -> bool:
    return _has_profile_identity(profile)


def has_identity_core(profile: dict[str, Any]) -> bool:
    return _has_identity_core(profile)


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
    pending_since = _parse_datetime(state.get("pending_identity_since"))
    if (
        pending_action
        and pending_since
        and (current_time - pending_since).total_seconds() > IDENTITY_PENDING_TIMEOUT_SECONDS
    ):
        cleared = await memory_engine.mark_identity_seen(speaker_id, verified=False, now=current_time)
        state = await memory_engine.load_identity_state(speaker_id)
        profile = state.get("profile") or cleared
        pending_action = ""
    if is_wake_word_only(text):
        heuristic_identity = {}
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
    if is_wake_word_only(text) or _is_ambiguous_identity_reply(text):
        identity_update = {}

    if is_profile_recall_query(text) and _has_profile_identity(profile):
        if pending_action in {"identity_conflict", "reverification"}:
            await memory_engine.mark_identity_seen(speaker_id, verified=True)
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"speaker_id": speaker_id, "profile": profile},
        )

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
        flash_profile = state.get("flash_profile") or {}
        if _has_identity_core(flash_profile):
            saved = await memory_engine.mark_identity_candidate(
                speaker_id,
                flash_profile,
                action="confirm_new_identity",
            )
            return IdentityGateResult(
                allowed=False,
                reason="confirm_flash_identity",
                response_text=_confirm_flash_identity_question(flash_profile),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": profile,
                    "flash_profile": flash_profile,
                    "identity_candidate": flash_profile,
                    "saved_state": saved,
                },
            )
        await memory_engine.mark_identity_pending(speaker_id, "prior_conversation_check")
        return IdentityGateResult(
            allowed=False,
            reason="prior_conversation_check",
            response_text=_prior_conversation_question(),
            metadata={
                "speaker_id": speaker_id,
                "profile": profile,
                "identity_update": identity_update,
                "identity_extract": identity_extract,
            },
        )

    if pending_action == "prior_conversation_check":
        return await _handle_prior_conversation_check(
            memory_engine=memory_engine,
            speaker_id=speaker_id,
            text=text,
            state=state,
            profile=profile,
            identity_update=identity_update,
            identity_extract=identity_extract,
        )

    if pending_action == "confirm_new_identity":
        candidate = state.get("pending_identity_candidate") or {}
        if _is_affirmative(text):
            saved = await memory_engine.confirm_identity_candidate(speaker_id)
            await memory_engine.update_flash_profile(speaker_id, saved)
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
        if is_profile_recall_query(text) and _has_profile_identity(profile):
            saved = await memory_engine.mark_identity_seen(speaker_id, verified=True)
            return IdentityGateResult(
                allowed=True,
                reason="identity_verified",
                metadata={"speaker_id": speaker_id, "profile": saved},
            )
        pending_reply = await judge_pending_identity_reply_with_llm(
            current_text=text,
            patient_profile=profile,
            pending_action=pending_action,
            extracted_profile=identity_update,
        )
        pending_decision = str(pending_reply.get("decision") or "unclear")
        judged_identity = _merge_identity_updates(identity_update, pending_reply.get("profile") or {})
        if pending_decision == "noise":
            return IdentityGateResult(
                allowed=False,
                reason="identity_pending_noise",
                response_text="",
                response_type="ignored",
                metadata={"speaker_id": speaker_id, "profile": profile, "pending_identity_judge": pending_reply},
            )
        if pending_decision == "same_person":
            if pending_action == "identity_conflict" and state.get("pending_identity_candidate"):
                saved = await memory_engine.confirm_identity_candidate(speaker_id)
                await memory_engine.update_flash_profile(speaker_id, saved)
                reason = "identity_candidate_registered"
                response_text = _registration_completed(saved)
            else:
                saved = await memory_engine.mark_identity_seen(speaker_id, verified=True)
                await memory_engine.update_flash_profile(speaker_id, saved)
                reason = "identity_reverified"
                response_text = _reverified_message(saved)
            return IdentityGateResult(
                allowed=False,
                reason=reason,
                response_text=response_text,
                metadata={"speaker_id": speaker_id, "profile": saved, "pending_identity_judge": pending_reply},
            )
        if pending_decision in {"rejected", "different_person"}:
            saved = await memory_engine.mark_identity_pending(speaker_id, "registration")
            return IdentityGateResult(
                allowed=False,
                reason="identity_rejected_needs_registration",
                response_text=_identity_rejected_registration_question(profile),
                metadata={"speaker_id": speaker_id, "profile": saved, "pending_identity_judge": pending_reply},
            )
        if pending_decision == "provided_identity" and _has_identity_core(judged_identity):
            saved = await memory_engine.mark_identity_candidate(
                speaker_id,
                judged_identity,
                action="confirm_new_identity",
            )
            return IdentityGateResult(
                allowed=False,
                reason="confirm_new_identity",
                response_text=_confirm_new_identity_question(judged_identity),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": profile,
                    "identity_candidate": judged_identity,
                    "identity_extract": identity_extract,
                    "pending_identity_judge": pending_reply,
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
        flash_profile = state.get("flash_profile") or {}
        if _has_identity_core(flash_profile):
            saved = await memory_engine.mark_identity_candidate(
                speaker_id,
                flash_profile,
                action="confirm_new_identity",
            )
            return IdentityGateResult(
                allowed=False,
                reason="confirm_flash_identity",
                response_text=_confirm_flash_identity_question(flash_profile),
                metadata={
                    "speaker_id": speaker_id,
                    "profile": profile,
                    "flash_profile": flash_profile,
                    "identity_candidate": flash_profile,
                    "saved_state": saved,
                },
            )
        await memory_engine.mark_identity_pending(speaker_id, "prior_conversation_check")
        return IdentityGateResult(
            allowed=False,
            reason="prior_conversation_check",
            response_text=_prior_conversation_question(),
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

    if not _should_check_identity_conflict(
        text=text,
        profile=profile,
        identity_update=identity_update,
        pending_action=pending_action,
    ):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"speaker_id": speaker_id, "profile": profile},
        )

    judge: dict[str, Any] = {"conflict": False, "source": "not_checked"}
    if _should_call_identity_conflict_llm(
        text, identity_update, pending_action=pending_action, profile=profile
    ):
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

    updated_profile = profile
    if identity_update.get("conditions"):
        merged_conditions = list(profile.get("conditions") or [])
        changed = False
        for condition in identity_update.get("conditions") or []:
            condition_text = str(condition).strip()
            if condition_text and condition_text not in merged_conditions:
                merged_conditions.append(condition_text)
                changed = True
        if changed:
            updated_profile = await memory_engine.save_identity_profile(
                speaker_id,
                {"conditions": merged_conditions},
                mark_verified=False,
                mark_seen=True,
                now=current_time,
            )
            await memory_engine.update_flash_profile(speaker_id, updated_profile)

    return IdentityGateResult(
        allowed=True,
        reason="identity_verified",
        metadata={"speaker_id": speaker_id, "profile": updated_profile, "judge": judge},
    )


def _has_profile_identity(profile: dict[str, Any]) -> bool:
    normalized = _normalize_identity_profile(profile)
    return bool(normalized.get("name") and (normalized.get("age") or normalized.get("gender")))


def _has_identity_core(profile: dict[str, Any]) -> bool:
    normalized = _normalize_identity_profile(profile)
    return bool(normalized.get("name") and (normalized.get("age") or normalized.get("gender")))


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
    return _normalize_identity_profile(merged)


def _normalize_identity_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    name = _normalize_korean_person_name(str(profile.get("name") or "").strip())
    if re.fullmatch(r"[가-힣]{2,5}", name) and not _looks_like_invalid_identity_name(name):
        normalized["name"] = name
    age_raw = str(profile.get("age") or "").strip()
    if re.fullmatch(r"\d{1,3}", age_raw):
        age = int(age_raw)
        if 1 <= age <= 120:
            normalized["age"] = str(age)
    gender = str(profile.get("gender") or "").strip()
    if gender in {"남성", "여성"}:
        normalized["gender"] = gender
    conditions = profile.get("conditions") or []
    if isinstance(conditions, list):
        clean_conditions = [str(item).strip() for item in conditions if str(item).strip()]
        if clean_conditions:
            normalized["conditions"] = clean_conditions
    return normalized


def _normalize_korean_person_name(name: str) -> str:
    if len(name) >= 3 and name.endswith(("이가", "이는", "이야")):
        return name[:-2]
    if len(name) >= 4 and name[-1:] in {"가", "은", "는", "야"}:
        return name[:-1]
    return name


def _looks_like_invalid_identity_name(name: str) -> bool:
    return name in {
        "남자고",
        "여자고",
        "남성이고",
        "여성이고",
        "우쭈우쭈",
    }


def _should_call_identity_extract_llm(
    *,
    text: str,
    state: dict[str, Any],
    heuristic_identity: dict[str, Any],
    pending_action: str,
) -> bool:
    """Avoid LLM identity extraction on ordinary conversation turns."""
    if is_wake_word_only(text) or _is_ambiguous_identity_reply(text):
        return False
    if _has_identity_core(heuristic_identity):
        return False
    if pending_action == "prior_conversation_check":
        return True
    if not _has_identity_text_signal(text):
        return False
    if pending_action in {
        "registration",
        "confirm_new_identity",
        "identity_conflict",
    }:
        return True
    if not state.get("exists") or not _has_profile_identity(state.get("profile") or {}):
        return True
    return bool(heuristic_identity)


def _should_call_identity_conflict_llm(
    text: str,
    identity_update: dict[str, Any],
    *,
    pending_action: str = "",
    profile: dict[str, Any] | None = None,
) -> bool:
    if is_profile_recall_query(text):
        return False
    if pending_action in {
        "prior_conversation_check",
        "registration",
        "confirm_new_identity",
        "reverification",
        "identity_conflict",
    }:
        if identity_update or any(
            token in text for token in ("보호자", "대신", "아버지", "어머니", "다른 사람")
        ):
            return True
    if profile and _has_profile_identity(profile):
        return _has_identity_core(identity_update) or _should_ask_identity_conflict_judge(text)
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
                "통풍",
                "신장질환",
                "간질환",
                "심장질환",
            )
        )
    )


def _is_ambiguous_identity_reply(text: str) -> bool:
    stripped = re.sub(r"\s+", "", (text or "").strip().lower())
    if not stripped:
        return True
    ambiguous_tokens = (
        "음뭐라고요",
        "뭐라고요",
        "잘안들렸어요",
        "다시말해",
        "다시말씀",
        "몰라",
        "모르겠",
    )
    if any(token in stripped for token in ambiguous_tokens):
        return True
    return len(stripped) <= 2 and not re.search(r"\d{1,3}(?:살|세)", stripped)


def _is_name_only_profile_mismatch(
    identity_update: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    if _has_identity_core(identity_update):
        return False
    incoming_name = str(identity_update.get("name") or "").strip()
    existing_name = str(profile.get("name") or "").strip()
    return bool(incoming_name and existing_name and incoming_name != existing_name)


def _should_check_identity_conflict(
    *,
    text: str,
    profile: dict[str, Any],
    identity_update: dict[str, Any],
    pending_action: str,
) -> bool:
    if is_profile_recall_query(text):
        return False
    if pending_action in {
        "prior_conversation_check",
        "registration",
        "confirm_new_identity",
        "reverification",
        "identity_conflict",
    }:
        return True
    if not _has_profile_identity(profile):
        return bool(identity_update)
    return bool(identity_update) or _should_ask_identity_conflict_judge(text)


def _should_ask_identity_conflict_judge(text: str) -> bool:
    """Only decides whether to ask the LLM judge; it does not decide conflict."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if is_wake_word_only(stripped) or _is_ambiguous_identity_reply(stripped):
        return False
    return _has_identity_text_signal(stripped) or any(
        token in stripped for token in ("보호자", "대신", "아버지", "어머니", "다른 사람")
    )


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
    stripped = re.sub(r"[\s.?!,，。~]+", "", text.strip().lower())
    return stripped in {
        "네",
        "예",
        "맞아",
        "맞아요",
        "맞습니다",
        "응",
        "그래",
        "그렇습니다",
        "본인",
        "본인맞아",
        "본인맞습니다",
    }


def _is_negative_identity_reply(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text.strip().lower())
    if not stripped:
        return False
    return any(
        token in stripped
        for token in (
            "아니",
            "아니야",
            "아닙니다",
            "다른사람",
            "사람달라",
            "틀렸",
            "김양수아니",
            "김영수아니",
            "내가아니",
            "본인아니",
        )
    )


async def _handle_prior_conversation_check(
    *,
    memory_engine: MemoryEngine,
    speaker_id: str,
    text: str,
    state: dict[str, Any],
    profile: dict[str, Any],
    identity_update: dict[str, Any],
    identity_extract: dict[str, Any],
) -> IdentityGateResult:
    """Ask whether we've met before; match an existing profile or start registration."""
    metadata_base = {
        "speaker_id": speaker_id,
        "profile": profile,
        "identity_update": identity_update,
        "identity_extract": identity_extract,
    }

    if _has_identity_core(identity_update):
        return await _complete_prior_conversation_registration(
            memory_engine=memory_engine,
            speaker_id=speaker_id,
            profile=profile,
            identity_update=identity_update,
            metadata_base=metadata_base,
            identity_extract=identity_extract,
            current_text=text,
        )

    extracted_name = str(identity_update.get("name") or "").strip()
    if _has_profile_identity(profile) and _name_matches_profile(extracted_name, profile):
        saved = await memory_engine.mark_identity_seen(speaker_id, verified=True)
        await memory_engine.update_flash_profile(speaker_id, saved)
        return IdentityGateResult(
            allowed=False,
            reason="identity_recognized",
            response_text=_recognized_message(saved),
            metadata={**metadata_base, "saved_state": saved},
        )

    judge = await judge_prior_conversation_turn(
        current_text=text,
        stored_profile=profile,
        extracted_profile=identity_update,
    )
    decision = str(judge.get("decision") or "unclear").strip().lower()
    judged_profile = _merge_identity_updates(
        identity_update,
        judge.get("profile") or {},
    )
    metadata_base = {
        **metadata_base,
        "prior_conversation_judge": judge,
        "prior_decision": decision,
    }

    if _has_identity_core(judged_profile):
        return await _complete_prior_conversation_registration(
            memory_engine=memory_engine,
            speaker_id=speaker_id,
            profile=profile,
            identity_update=judged_profile,
            metadata_base=metadata_base,
            identity_extract=identity_extract,
            current_text=text,
        )

    if decision == "returning_match" and _has_profile_identity(profile):
        saved = await memory_engine.mark_identity_seen(speaker_id, verified=True)
        await memory_engine.update_flash_profile(speaker_id, saved)
        return IdentityGateResult(
            allowed=False,
            reason="identity_recognized",
            response_text=_recognized_message(saved),
            metadata={**metadata_base, "saved_state": saved},
        )

    if decision == "new_user":
        await memory_engine.mark_identity_pending(speaker_id, "registration")
        return IdentityGateResult(
            allowed=False,
            reason="needs_registration",
            response_text=_registration_question(),
            metadata=metadata_base,
        )

    if decision == "returning_match":
        await memory_engine.mark_identity_pending(speaker_id, "registration")
        return IdentityGateResult(
            allowed=False,
            reason="needs_registration",
            response_text=_ask_identity_after_prior_yes(),
            metadata=metadata_base,
        )

    # unclear: 같은 질문 반복 금지 — 등록 단계로 진행
    await memory_engine.mark_identity_pending(speaker_id, "registration")
    return IdentityGateResult(
        allowed=False,
        reason="needs_registration",
        response_text=_registration_question(),
        metadata=metadata_base,
    )


async def _complete_prior_conversation_registration(
    *,
    memory_engine: MemoryEngine,
    speaker_id: str,
    profile: dict[str, Any],
    identity_update: dict[str, Any],
    metadata_base: dict[str, Any],
    identity_extract: dict[str, Any],
    current_text: str = "",
) -> IdentityGateResult:
    if _has_profile_identity(profile) and _identity_matches_profile(identity_update, profile):
        saved = await memory_engine.mark_identity_seen(speaker_id, verified=True)
        await memory_engine.update_flash_profile(speaker_id, saved)
        return IdentityGateResult(
            allowed=False,
            reason="identity_recognized",
            response_text=_recognized_message(saved),
            metadata={**metadata_base, "saved_state": saved},
        )
    if _has_profile_identity(profile):
        history = await memory_engine.store.read_user_file(speaker_id, "history.md")
        judge = await judge_identity_conflict(
            current_text=current_text or str(metadata_base.get("text") or ""),
            patient_profile=profile,
            recent_history=history[:1200],
            current_time=datetime.now().isoformat(timespec="seconds"),
        )
        if judge.get("conflict"):
            saved = await memory_engine.mark_identity_candidate(
                speaker_id,
                identity_update,
                action="identity_conflict",
            )
            return IdentityGateResult(
                allowed=False,
                reason="identity_conflict",
                response_text=_confirm_new_identity_question(identity_update, existing_profile=profile),
                metadata={
                    **metadata_base,
                    "identity_candidate": identity_update,
                    "judge": judge,
                    "saved_state": saved,
                },
            )
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
        metadata={**metadata_base, "saved_state": saved},
    )


def _identity_matches_profile(identity_update: dict[str, Any], profile: dict[str, Any]) -> bool:
    incoming_name = str(identity_update.get("name") or "").strip()
    existing_name = str(profile.get("name") or "").strip()
    if incoming_name and existing_name and incoming_name != existing_name:
        return False
    for key in ("age", "gender"):
        incoming = str(identity_update.get(key) or "").strip()
        existing = str(profile.get(key) or "").strip()
        if incoming and existing and incoming != existing:
            return False
    return bool(incoming_name and existing_name)


def _name_matches_profile(name: str, profile: dict[str, Any]) -> bool:
    existing_name = str(profile.get("name") or "").strip()
    return bool(name and existing_name and name == existing_name)


def _prior_conversation_question() -> str:
    return "안녕하세요. 저희가 일전에 대화한 적 있나요?"


def _ask_identity_after_prior_yes() -> str:
    return "그렇다면 어떤 분이신지 이름, 성별, 나이를 말씀해 주세요."


def _confirm_flash_identity_question(profile: dict[str, Any]) -> str:
    name = profile.get("name") or "저장된 분"
    details = []
    if profile.get("age"):
        details.append(f"{profile['age']}세")
    if profile.get("gender"):
        details.append(str(profile["gender"]))
    detail_text = f" ({', '.join(details)})" if details else ""
    return f"방금 전 대화에 {name}님{detail_text} 정보가 남아 있어요. 지금 말씀하시는 분이 {name}님 맞으신가요?"


def _registration_question() -> str:
    return "이름, 나이, 성별을 한 번에 말씀해 주세요."


def _recognized_message(profile: dict[str, Any]) -> str:
    name = profile.get("name") or "사용자"
    details = []
    if profile.get("gender"):
        details.append(str(profile["gender"]))
    if profile.get("age"):
        details.append(f"{profile['age']}세")
    detail_text = ", ".join(details)
    if detail_text:
        return f"{name}님, {detail_text}로 등록됐습니다. 이제 무엇을 도와드릴까요?"
    return f"{name}님으로 확인했습니다. 이제 무엇을 도와드릴까요?"


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


def _identity_rejected_registration_question(profile: dict[str, Any]) -> str:
    name = profile.get("name") or "저장된 분"
    return (
        f"알겠습니다. {name}님으로 보지 않겠습니다. "
        "새로 등록할 이름, 나이, 성별을 말씀해 주세요."
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
