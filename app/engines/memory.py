"""메모리 엔진 (Memory Engine) — 지식 창고 및 사서.

server.mermaid 매핑:
  OCR_Logging         → log_ocr_result()
  OCR_DUR_Interaction → sync_ocr_dur()
  ME_Context          → load_context()
  ME_Parse            → parse_patient_info()
  ME_RAG              → search_history()
  ME_Update           → update_and_compress()

메모리 조회 계층은 Claude Code 방식의 `structured_memory`
(`MEMORY.md` 인덱스 + frontmatter topic 파일)도 함께 사용한다.
"""
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from app.core.config import settings
from app.database.md_store import md_store
from app.memory import StructuredMemoryService
from app.schemas.engine_contracts import (
    MemoryArtifactRef,
    MemoryEvidenceBundle,
    MemoryEvidenceRequest,
)
from app.services.dur_summary import summarize_dur_result
from app.services.medication_extraction import (
    extract_medication_suffix_tokens,
    filter_drug_name_candidates,
    is_non_medication_token,
    strip_wake_words,
)
from app.services.patient_safety import classify_patient_safety_situation
from app.tools import llm_search

logger = logging.getLogger(__name__)

OCR_TYPO_MAP = {
    "타이레롤": "타이레놀",
    "와파린정정": "와파린정",
    "아스피린장용정정": "아스피린장용정",
}
COMMON_MEDICATION_NAMES = {
    "와파린",
    "아스피린",
    "로사르탄",
    "오메프라졸",
    "인슐린",
    "메트포르민",
    "암로디핀",
}
SPOKEN_MEDICATION_ALIASES = {
    "디오반": "디오반정",
    "타이레놀": "타이레놀",
    "로사르탄": "로사르탄정",
    "아스피린": "아스피린",
    "와파린": "와파린",
}


class MemoryEngine:
    """메모리 엔진: 사용자 식별, 컨텍스트 관리, RAG, 데이터 압축."""

    def __init__(self):
        self.store = md_store
        self.structured_memory = StructuredMemoryService()

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.structured_memory.initialize()

    async def bootstrap_flash_from_permanent(
        self,
        speaker_id: Optional[str] = None,
    ) -> None:
        """Rebuild volatile flash memory from permanent memory when no dialogue is active.

        Runtime policy:
        - Server/session startup may rebuild flash from permanent memory.
        - During an active user dialogue, use current flash and only refresh after
          explicit writes in that same turn.
        """
        await self.initialize()
        if speaker_id and await self.store.user_exists(speaker_id):
            profile = self._parse_profile(
                await self.store.read_user_file(speaker_id, "profile.md")
            )
            await self.update_flash_profile(speaker_id, profile)
            history = await self.store.read_user_file(speaker_id, "history.md")
            medication_events = await self.store.read_user_file(speaker_id, "medication_events.md")
            await self.store.write_flash(
                "current_manual",
                self._format_patient_special_notes(profile, history, medication_events),
            )

        latest_prescriptions = await self.store.read_latest("prescriptions", n=1)
        if latest_prescriptions:
            med_names = self._extract_medications_from_markdown(
                latest_prescriptions[0].get("content", "")
            )
            if med_names:
                await self.store.write_flash(
                    "prescription_log",
                    self._format_prescription_log(
                        med_names,
                        recorded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )

        latest_logs = await self.store.read_latest("medication_log", n=1)
        if latest_logs:
            content = latest_logs[0].get("content", "")
            await self.store.write_flash(
                "context_memory",
                self._format_context_memory_from_latest_log(content),
            )

    # ── OCR_Logging: 처방전 OCR 로깅 ──

    async def log_ocr_result(self, ocr_data: dict, confidence: float = 0.0) -> None:
        """OCR 결과를 ocr_history/{날짜}/NNN.md 에 개별 파일로 저장."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meds = ocr_data.get("medications", [])
        med_lines = "\n".join(
            f"- {m.get('name', '?')}" for m in meds
        ) or "- (인식된 약품 없음)"

        content = (
            f"# OCR 결과\n"
            f"> 기록 시각: {now}\n"
            f"> 신뢰도: {confidence:.2f}\n\n"
            f"## 인식된 약품\n{med_lines}\n\n"
            f"## 원본 데이터\n"
            f"```json\n{json.dumps(ocr_data, ensure_ascii=False, default=str)[:1000]}\n```\n"
        )
        await self.store.save("ocr_history", content)

    # ── OCR_DUR_Interaction: OCR 처방전 DUR 동기화 ──

    async def sync_ocr_dur(
        self,
        ocr_data: dict,
        dur_results: list[dict],
        speaker_id: Optional[str] = None,
    ) -> None:
        """처방전 + DUR 결과를 prescriptions/{날짜}/NNN.md 에 저장."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meds = ocr_data.get("medications", [])
        med_names = [m.get("name", "알 수 없음") for m in meds]

        dur_lines: list[str] = []
        for dur in dur_results:
            summary = summarize_dur_result(dur)
            dur_lines.append(
                f"- {summary['name']}: 정보 {summary['info']}건 / "
                f"금기 {summary['contraindications']}건 / 주의 {summary['precautions']}건"
            )

        content = (
            f"# 처방전 DUR 동기화\n"
            f"> 기록 시각: {now}\n\n"
            f"## 약품 목록\n"
            + "\n".join(f"- {n}" for n in med_names) + "\n\n"
            f"## DUR 검증 결과 ({len(dur_results)}건)\n"
            + ("\n".join(dur_lines) if dur_lines else "- DUR 결과 없음") + "\n"
        )
        await self.store.save("prescriptions", content)

        prescription_log = (
            f"# 현재 복용 약 요약\n"
            f"> 최종 갱신: {now}\n\n"
            f"## 약품 목록\n"
            + ("\n".join(f"- {name}" for name in med_names) if med_names else "- 확인된 약품 없음")
            + "\n"
        )
        await self.store.write_flash("prescription_log", prescription_log)
        await self.structured_memory.sync_medication_context(
            med_names=med_names,
            dur_results=dur_results,
            recorded_at=now,
            speaker_id=speaker_id,
        )

    async def store_ocr_text_result(
        self,
        text: str,
        *,
        speaker_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> list[str]:
        """Normalize STT text that reports OCR results into OCR/prescription memory."""
        med_names = self.extract_ocr_medications_from_text(text)
        if not med_names:
            return []

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ocr_data = {
            "source": "stt_text",
            "raw_text": text,
            "medications": [{"name": name} for name in med_names],
        }
        await self.log_ocr_result(ocr_data, confidence=confidence)

        prescription = (
            f"# 처방전 OCR 기록\n"
            f"> 기록 시각: {now}\n"
            f"> 입력 경로: STT 텍스트 OCR 결과\n\n"
            "## 약품 목록\n"
            + "\n".join(f"- {name}" for name in med_names)
            + "\n\n## 원문\n"
            + text[:1000]
            + "\n"
        )
        await self.store.save("prescriptions", prescription)
        await self.store.write_flash(
            "prescription_log",
            self._format_prescription_log(med_names, recorded_at=now),
        )
        await self.structured_memory.sync_medication_context(
            med_names=med_names,
            dur_results=[],
            recorded_at=now,
            speaker_id=speaker_id,
        )
        return med_names

    async def store_spoken_medication_result(
        self,
        text: str,
        med_names: list[str],
        *,
        speaker_id: Optional[str] = None,
    ) -> list[str]:
        """Store medication names the user provided verbally as current meds."""
        normalized: list[str] = []
        for med in med_names:
            name = self._normalize_spoken_medication_candidate(med)
            if name and name not in normalized:
                normalized.append(name)
        if not normalized:
            return []

        existing_log = await self.store.read_flash("prescription_log")
        merged = self._extract_medications_from_markdown(existing_log)
        for name in normalized:
            if name not in merged:
                merged.append(name)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prescription = (
            f"# 음성 복약 등록\n"
            f"> 기록 시각: {now}\n"
            f"> 입력 경로: STT 음성 약 이름 등록\n\n"
            "## 약품 목록\n"
            + "\n".join(f"- {name}" for name in normalized)
            + "\n\n## 원문\n"
            + text[:1000]
            + "\n"
        )
        await self.store.save("prescriptions", prescription)
        await self.store.write_flash(
            "prescription_log",
            self._format_prescription_log(merged, recorded_at=now),
        )
        await self.structured_memory.sync_medication_context(
            med_names=merged,
            dur_results=[],
            recorded_at=now,
            speaker_id=speaker_id,
        )
        return merged

    # ── ME_Context: 사용자 식별 및 컨텍스트 로드 ──

    async def load_context(self, speaker_id: Optional[str] = None) -> dict[str, Any]:
        """현재 대화자의 컨텍스트를 로드. 새 사용자이면 프로필 생성."""
        context: dict[str, Any] = {
            "speaker_id": speaker_id,
            "is_new_user": False,
            "user_profile": {},
            "current_requirement": "",
            "current_manual": "",
            "context_memory": "",
            "prescription_log": "",
            "memory_prompt": "",
            "memory_index": "",
            "relevant_memories": [],
            "memory_briefs": [],
        }

        if speaker_id:
            exists = await self.store.user_exists(speaker_id)
            if not exists:
                context["is_new_user"] = True
                await self._register_new_user(speaker_id)
            profile_md = await self.store.read_user_file(speaker_id, "profile.md")
            context["user_profile"] = self._parse_profile(profile_md)

        context["current_requirement"] = await self.store.read_flash("current_requirement")
        context["current_manual"] = await self.store.read_flash("current_manual")
        context["context_memory"] = await self.store.read_flash("context_memory")
        context["prescription_log"] = await self.store.read_flash("prescription_log")

        structured_context = await self.structured_memory.build_context(
            "",
            speaker_id=speaker_id,
        )
        context.update(structured_context)
        return context

    async def load_identity_state(self, speaker_id: str) -> dict[str, Any]:
        """Load patient identity metadata used before multi-speaker turns."""
        exists = await self.store.user_exists(speaker_id)
        if not exists:
            await self._register_new_user(speaker_id)
        profile_md = await self.store.read_user_file(speaker_id, "profile.md")
        profile = self._parse_profile(profile_md)
        flash_profile = self._parse_flash_current_user_profile(
            await self.store.read_flash("current_user_profile")
        )
        return {
            "speaker_id": speaker_id,
            "exists": exists,
            "profile": profile,
            "flash_profile": flash_profile,
            "profile_text": profile_md,
            "pending_identity_action": profile.get("pending_identity_action", ""),
            "pending_identity_candidate": profile.get("pending_identity_candidate", {}),
            "pending_identity_since": profile.get("pending_identity_since", ""),
            "last_seen_at": profile.get("last_seen_at", ""),
            "verified_at": profile.get("verified_at", ""),
        }

    async def save_identity_profile(
        self,
        speaker_id: str,
        profile_update: dict[str, Any],
        *,
        pending_identity_action: str = "",
        mark_verified: bool = False,
        mark_seen: bool = True,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Merge identity fields into the patient profile markdown."""
        current = self._parse_profile(await self.store.read_user_file(speaker_id, "profile.md"))
        profile_update = self._normalize_profile_update(profile_update)
        merged = {**current, **{key: value for key, value in profile_update.items() if value}}
        timestamp = (now or datetime.now()).isoformat(timespec="seconds")
        if mark_seen:
            merged["last_seen_at"] = timestamp
        if mark_verified:
            merged["verified_at"] = timestamp
        previous_pending = current.get("pending_identity_action", "")
        merged["pending_identity_action"] = pending_identity_action
        if pending_identity_action:
            if pending_identity_action != previous_pending or not current.get("pending_identity_since"):
                merged["pending_identity_since"] = timestamp
            else:
                merged["pending_identity_since"] = current.get("pending_identity_since", "")
        else:
            merged.pop("pending_identity_since", None)
        if "pending_identity_candidate" in profile_update:
            merged["pending_identity_candidate"] = profile_update["pending_identity_candidate"]
        elif not pending_identity_action:
            merged.pop("pending_identity_candidate", None)
        content = self._format_profile_markdown(speaker_id, merged, registered_at=timestamp)
        await self.store.save_user_file(speaker_id, "profile.md", content)
        await self.structured_memory.sync_patient_profile(
            speaker_id,
            {
                "name": merged.get("name", ""),
                "age": merged.get("age", ""),
                "gender": merged.get("gender", ""),
                "conditions": self._normalize_conditions(merged.get("conditions")),
            },
        )
        return merged

    def _normalize_profile_update(self, profile_update: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(profile_update or {})
        if "name" in normalized:
            name = self._normalize_korean_person_name(str(normalized.get("name") or "").strip())
            if not name or self._looks_like_non_name_identity_candidate(name):
                normalized.pop("name", None)
            else:
                normalized["name"] = name
        return normalized

    @staticmethod
    def _normalize_korean_person_name(name: str) -> str:
        if len(name) >= 3 and name.endswith(("이가", "이는", "이야")):
            return name[:-2]
        if len(name) >= 4 and name[-1:] in {"가", "은", "는", "야"}:
            return name[:-1]
        return name

    async def mark_identity_pending(self, speaker_id: str, action: str) -> dict[str, Any]:
        return await self.save_identity_profile(
            speaker_id,
            {},
            pending_identity_action=action,
            mark_seen=False,
        )

    async def mark_identity_candidate(
        self,
        speaker_id: str,
        candidate: dict[str, Any],
        *,
        action: str = "confirm_new_identity",
    ) -> dict[str, Any]:
        return await self.save_identity_profile(
            speaker_id,
            {"pending_identity_candidate": candidate},
            pending_identity_action=action,
            mark_seen=False,
        )

    async def confirm_identity_candidate(self, speaker_id: str) -> dict[str, Any]:
        state = await self.load_identity_state(speaker_id)
        candidate = state.get("pending_identity_candidate") or {}
        if not isinstance(candidate, dict):
            candidate = {}
        return await self.save_identity_profile(
            speaker_id,
            candidate,
            pending_identity_action="",
            mark_verified=True,
        )

    async def mark_identity_seen(
        self,
        speaker_id: str,
        *,
        verified: bool = False,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        return await self.save_identity_profile(
            speaker_id,
            {},
            pending_identity_action="",
            mark_verified=verified,
            mark_seen=True,
            now=now,
        )

    async def force_identity_last_seen_minutes_ago(
        self,
        speaker_id: str,
        minutes: int,
    ) -> dict[str, Any]:
        forced_time = datetime.now() - timedelta(minutes=minutes)
        return await self.save_identity_profile(
            speaker_id,
            {},
            pending_identity_action="",
            mark_seen=True,
            now=forced_time,
        )

    def extract_identity_from_text(self, text: str) -> dict[str, Any]:
        return self._extract_profile_from_text(text)

    async def build_query_memory_context(
        self,
        query: str,
        speaker_id: Optional[str] = None,
    ) -> str:
        structured_context = await self.structured_memory.build_context(
            query,
            speaker_id=speaker_id,
        )
        return structured_context.get("memory_prompt", "")

    async def prepare_evidence_bundle(
        self,
        request: MemoryEvidenceRequest,
    ) -> MemoryEvidenceBundle:
        """Prepare normalized memory evidence for reasoning/conversation.

        Ownership:
        - Memory engine normalizes OCR entities and picks minimal artifact refs.
        - It may use frontier search only when DUR-search is not feasible.
        """
        normalized_query = " ".join(request.query.strip().split())
        history = await self.search_history(
            normalized_query,
            speaker_id=request.speaker_id,
        )
        normalized_meds = self.normalize_ocr_medications(request.ocr_payload)
        if not normalized_meds:
            normalized_meds = self._extract_query_medications(normalized_query)
        dur_searchable = bool(normalized_meds) and all(
            self._is_dur_search_supported(name) for name in normalized_meds
        )

        artifact_refs = self._select_artifacts(history)
        structured_memory = history.get("structured_memory", {})
        memory_prompt = (
            structured_memory.get("prompt", "")
            if isinstance(structured_memory, dict)
            else ""
        )
        summary = self._summarize_artifacts(history)

        used_frontier_fallback = False
        frontier_answer_preview = ""
        if request.allow_frontier_fallback and not dur_searchable and normalized_query:
            fallback = await llm_search.llm_search(normalized_query, context=memory_prompt)
            if fallback.get("success") and fallback.get("answer"):
                used_frontier_fallback = True
                frontier_answer_preview = fallback["answer"][:500].strip()
                if frontier_answer_preview:
                    summary = (
                        f"{summary}\n\n[fallback]\n{frontier_answer_preview}"
                        if summary
                        else frontier_answer_preview
                    )

        return MemoryEvidenceBundle(
            normalized_query=normalized_query,
            normalized_medications=normalized_meds,
            dur_searchable=dur_searchable,
            used_frontier_fallback=used_frontier_fallback,
            frontier_answer_preview=frontier_answer_preview,
            artifact_refs=artifact_refs,
            summary=summary,
            memory_prompt=memory_prompt,
        )

    # ── ME_Parse: 사용자/복약 대상자 개인정보 구분 및 주요 정보 로그 파싱 ──

    async def parse_patient_info(self, raw_data: dict) -> dict:
        return {
            "name": raw_data.get("name", ""),
            "age": raw_data.get("age", ""),
            "gender": raw_data.get("gender", ""),
            "conditions": raw_data.get("conditions", []),
            "allergies": raw_data.get("allergies", []),
        }

    # ── ME_RAG: 사용자/복약 대상자 관리 및 관련 이력 검색 ──

    async def search_history(
        self, query: str, speaker_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """질문 키워드로 Permanent Memory 전체와 structured memory를 함께 탐색."""
        if not query:
            return {}

        results: dict[str, Any] = {}

        for category in ["medication_log", "prescriptions", "dur_linkage", "health_supplement"]:
            hits = await self.store.search(category, query, limit=5)
            if hits:
                results[category] = hits

        if speaker_id:
            user_history = await self.store.read_user_file(speaker_id, "history.md")
            if user_history and self._text_relevant(user_history, query):
                results["user_history"] = user_history

            medication_events = await self.store.read_user_file(speaker_id, "medication_events.md")
            if medication_events and self._medication_events_relevant(medication_events, query):
                results["medication_events"] = medication_events

        structured_context = await self.structured_memory.build_context(
            query,
            speaker_id=speaker_id,
        )
        if structured_context.get("relevant_memories"):
            results["structured_memory"] = {
                "items": structured_context["relevant_memories"],
                "briefs": structured_context["memory_briefs"],
                "prompt": structured_context["memory_prompt"],
            }

        return results

    # ── ME_Update: 데이터 업데이트 및 압축 저장 ──

    async def update_and_compress(
        self, response_data: dict, speaker_id: Optional[str] = None,
    ) -> None:
        """응답 결과를 개별 MD 파일로 저장하고 Flash Memory를 압축 갱신."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query_text = response_data.get("query", "")
        answer_text = response_data.get("answer", "")
        resp_type = response_data.get("type", "unknown")

        log_content = (
            f"# 상담 기록\n"
            f"> 기록 시각: {now}\n"
            f"> 유형: {resp_type}\n\n"
            f"## 질문\n{query_text[:500]}\n\n"
            f"## 응답\n{answer_text[:1000]}\n"
        )
        await self.store.save("medication_log", log_content)

        dur_results = response_data.get("dur_results")
        if dur_results:
            dur_content = (
                f"# DUR 호출 기록\n"
                f"> 기록 시각: {now}\n\n"
                f"## 대상 약품\n{response_data.get('medications', 'N/A')}\n\n"
                f"## 결과\n```json\n{json.dumps(dur_results, ensure_ascii=False, default=str)[:2000]}\n```\n"
            )
            await self.store.save("dur_linkage", dur_content)

        context_summary = (
            f"# 대화 컨텍스트 메모리\n"
            f"> 최종 갱신: {now}\n\n"
            f"## 최근 대화 요약\n"
            f"- 질문: {query_text[:100]}\n"
            f"- 핵심 응답: {answer_text[:200]}\n"
        )
        await self.store.write_flash("context_memory", context_summary)

        prev = await self.store.read_flash("current_requirement")
        lines = [l for l in prev.strip().split("\n") if l.startswith("- [")]
        lines.append(f"- [{now}] {query_text[:100]}")
        if len(lines) > 5:
            lines = lines[-5:]
        req_content = (
            f"# 최근 요구사항 (최대 5회)\n> 최종 갱신: {now}\n\n"
            + "\n".join(lines) + "\n"
        )
        await self.store.write_flash("current_requirement", req_content)

        if speaker_id:
            await self.record_safety_incident_from_text(
                query_text,
                answer_text=answer_text,
                speaker_id=speaker_id,
            )
            await self.record_medication_event_from_text(
                query_text,
                speaker_id=speaker_id,
            )

            existing = await self.store.read_user_file(speaker_id, "history.md")
            entry = (
                f"\n---\n### {now} ({resp_type})\n"
                f"- Q: {query_text[:200]}\n"
                f"- A: {answer_text[:300]}\n"
            )
            await self.store.save_user_file(
                speaker_id, "history.md", existing + entry,
            )

    async def record_safety_incident_from_text(
        self,
        text: str,
        *,
        answer_text: str = "",
        speaker_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Optional[dict[str, Any]]:
        """Append common medication-use mistakes to a speaker safety log."""
        if not speaker_id:
            return None
        situation = classify_patient_safety_situation(text)
        if not situation or not situation.should_record_incident:
            return None

        timestamp = (now or datetime.now()).isoformat(timespec="seconds")
        incident = {
            "recorded_at": timestamp,
            "speaker_id": speaker_id,
            "situation": situation.key,
            "severity": situation.severity,
            "source_text": text,
            "response": answer_text[:500],
        }
        existing = await self.store.read_user_file(speaker_id, "safety_incidents.md")
        content = existing.rstrip()
        if content:
            content += "\n"
        else:
            content = "# 복약 안전 사건\n\n"
        content += "- " + json.dumps(incident, ensure_ascii=False, sort_keys=True) + "\n"
        await self.store.save_user_file(speaker_id, "safety_incidents.md", content)
        return incident

    async def record_medication_event_from_text(
        self,
        text: str,
        *,
        speaker_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Optional[dict[str, Any]]:
        """Extract and persist a typed medication event from a user utterance."""
        if not speaker_id:
            return None
        event = self.extract_medication_event_from_text(text, now=now)
        if not event:
            return None

        event["speaker_id"] = speaker_id
        event["recorded_at"] = (now or datetime.now()).isoformat(timespec="seconds")
        event["source_text"] = text

        existing = await self.store.read_user_file(speaker_id, "medication_events.md")
        events = self._parse_medication_events(existing)
        dedupe_key = (
            event.get("date"),
            event.get("time"),
            event.get("medication"),
            event.get("action"),
            event.get("source_text"),
        )
        if not any(
            (
                item.get("date"),
                item.get("time"),
                item.get("medication"),
                item.get("action"),
                item.get("source_text"),
            )
            == dedupe_key
            for item in events
        ):
            events.append(event)

        content = "# 복약 이벤트\n\n" + "\n".join(
            "- " + json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in events[-50:]
        )
        if events:
            content += "\n"
        await self.store.save_user_file(speaker_id, "medication_events.md", content)
        await self.structured_memory.sync_medication_events(
            speaker_id=speaker_id,
            events=events[-50:],
        )
        return event

    def extract_medication_event_from_text(
        self,
        text: str,
        *,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Return a typed medication event when text clearly records a taken dose."""
        lowered = (text or "").lower()
        if not any(token in lowered for token in ("복용", "먹었", "먹었다", "드셨")):
            return {}
        medication = self._first_medication_from_text(text)
        if not medication or medication in {"약", "식후약", "처방약"}:
            return {}

        base = now or datetime.now()
        return {
            "date": self._extract_event_date(text, base),
            "time": self._extract_event_time(text),
            "medication": medication,
            "action": "taken",
        }

    async def update_flash_profile(self, speaker_id: str, profile: dict) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = (
            f"# 현재 복약 관리 프로필\n> 최종 갱신: {now}\n\n"
            f"| 항목 | 값 |\n|------|----|\n"
            f"| 이름 | {profile.get('name', '-')} |\n"
            f"| ID | {speaker_id} |\n"
            f"| 연령 | {profile.get('age', '-')} |\n"
            f"| 성별 | {profile.get('gender', '-')} |\n"
            f"| 기저질환 | {', '.join(profile.get('conditions', []))} |\n"
        )
        await self.store.write_flash("current_user_profile", content)
        history = await self.store.read_user_file(speaker_id, "history.md")
        medication_events = await self.store.read_user_file(speaker_id, "medication_events.md")
        await self.store.write_flash(
            "current_manual",
            self._format_patient_special_notes(profile, history, medication_events),
        )
        await self.structured_memory.sync_patient_profile(speaker_id, profile)

    # ── 내부 유틸 ──

    async def _save_user_profile(self, speaker_id: str, profile: dict[str, Any]) -> None:
        await self.save_identity_profile(
            speaker_id,
            profile,
            pending_identity_action="",
            mark_verified=True,
        )

    def _extract_profile_from_text(self, text: str) -> dict[str, Any]:
        profile: dict[str, Any] = {}
        name_patterns = [
            r"(?:제\s*이름은|이름은)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
            r"(?:저는|나는|난)\s*([가-힣]{2,5})(?:이야|야)(?:\.|$|\s)",
            r"(?:저는|나는|난)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
            r"(?:^|\s)([가-힣]{2,5})\s*(?:남자|남성|여자|여성)\s*,?\s*\d{1,3}\s*(?:살|세)?",
            r"(?:^|\s)([가-힣]{2,5})\s*,?\s*(?:\d{1,3}|[가-힣]{2,8})\s*(?:살|세)\s*(?:남자|남성|여자|여성)?",
            r"^\s*([가-힣]{2,5})(?:야|요|입니다|이에요|예요)(?:\.|$|\s)",
            r"^\s*([가-힣]{2,5})\s*(?:남자|남성|여자|여성)",
            r"^\s*([가-힣]{2,5})\s*,?\s*(?:\d{1,3}|[가-힣]{2,8})\s*(?:살|세)",
            (
                r"(?:환자|대상자|아버지|어머니|엄마|아빠|남편|아내|배우자)"
                r"(?:\s*이름은|\s*성함은|\s*는|\s*가|\s*께서는)?\s*"
                r"([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s+\d{1,3}\s*(?:살|세)|\s|$)"
            ),
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            candidate_name = match.group(1)
            if self._looks_like_non_name_identity_candidate(candidate_name):
                continue
            profile["name"] = candidate_name
            break
        age_match = re.search(r"(\d{1,3})\s*(?:살|세)", text)
        if age_match:
            profile["age"] = age_match.group(1)
        else:
            korean_age_match = re.search(r"([가-힣]{2,8})\s*(?:살|세)", text)
            if korean_age_match:
                age = self._parse_korean_age(korean_age_match.group(1))
                if age:
                    profile["age"] = str(age)
        if "남자" in text or "남성" in text:
            profile["gender"] = "남성"
        elif "여자" in text or "여성" in text:
            profile["gender"] = "여성"
        conditions = []
        for token in ("고혈압", "당뇨", "천식", "통풍", "신장질환", "간질환", "심장질환", "임신 가능성", "임신"):
            if token in text:
                conditions.append(token)
        if conditions:
            profile["conditions"] = conditions
        return profile

    @staticmethod
    def _parse_korean_age(text: str) -> int:
        compact = re.sub(r"\s+", "", text or "")
        direct = {
            "스무": 20,
            "스물": 20,
            "서른": 30,
            "마흔": 40,
            "쉰": 50,
            "예순": 60,
            "일흔": 70,
            "여든": 80,
            "아흔": 90,
        }
        if compact in direct:
            return direct[compact]
        tens = {
            "스물": 20,
            "서른": 30,
            "마흔": 40,
            "쉰": 50,
            "예순": 60,
            "일흔": 70,
            "여든": 80,
            "아흔": 90,
        }
        ones = {
            "한": 1,
            "하나": 1,
            "두": 2,
            "둘": 2,
            "세": 3,
            "셋": 3,
            "네": 4,
            "넷": 4,
            "다섯": 5,
            "여섯": 6,
            "일곱": 7,
            "여덟": 8,
            "아홉": 9,
        }
        for ten_text, ten_value in tens.items():
            if compact.startswith(ten_text):
                rest = compact[len(ten_text):]
                if not rest:
                    return ten_value
                if rest in ones:
                    return ten_value + ones[rest]
        return 0

    @staticmethod
    def _looks_like_non_name_identity_candidate(value: str) -> bool:
        return value in {
            "고혈압",
            "당뇨",
            "천식",
            "신장질환",
            "간질환",
            "심장질환",
            "임신",
            "남자",
            "여자",
            "남성",
            "여성",
            "남자고",
            "여자고",
            "남성이고",
            "여성이고",
            "딸",
            "딸이",
            "아들",
            "아들이",
            "보호자",
            "가족",
            "엄마",
            "아빠",
            "아버지",
            "어머니",
            "먹어",
            "먹고",
            "먹었",
            "알려",
            "나중에",
        }

    async def _register_new_user(self, speaker_id: str) -> None:
        now = datetime.now()
        profile = self._format_profile_markdown(
            speaker_id,
            {
                "name": "",
                "gender": "",
                "age": "",
                "conditions": [],
                "pending_identity_action": "prior_conversation_check",
                "pending_identity_candidate": {},
                "pending_identity_since": now.isoformat(timespec="seconds"),
                "last_seen_at": "",
                "verified_at": "",
            },
            registered_at=now.isoformat(timespec="seconds"),
        )
        await self.store.save_user_file(speaker_id, "profile.md", profile)
        await self.structured_memory.sync_patient_profile(
            speaker_id,
            {
                "name": "",
                "age": "",
                "gender": "",
                "conditions": [],
            },
        )

        registration = (
            f"# 신규 복약 관리 대상자 등록\n"
            f"> 등록 시각: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"> ID: {speaker_id}\n"
        )
        await self.store.save("patients", registration)

    def _parse_profile(self, profile_md: str) -> dict:
        result: dict[str, Any] = {}
        key_map = {
            "ID": "speaker_id",
            "이름": "name",
            "성별": "gender",
            "연령": "age",
            "기저질환": "conditions",
            "최종대화": "last_seen_at",
            "최종확인": "verified_at",
            "신원확인상태": "pending_identity_action",
            "신원확인시작": "pending_identity_since",
            "신원후보": "pending_identity_candidate",
        }
        for line in profile_md.split("\n"):
            if "|" in line and "---" not in line and "항목" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) == 2:
                    key, value = parts
                    if value != "-":
                        canonical_key = key_map.get(key, key)
                        if canonical_key == "conditions":
                            result[canonical_key] = [
                                item.strip()
                                for item in value.split(",")
                                if item.strip() and item.strip() != "-"
                            ]
                        else:
                            result[canonical_key] = value
                        if canonical_key == "pending_identity_candidate":
                            try:
                                parsed_candidate = json.loads(value)
                            except json.JSONDecodeError:
                                parsed_candidate = {}
                            result[canonical_key] = parsed_candidate if isinstance(parsed_candidate, dict) else {}
        return result

    def _parse_flash_current_user_profile(self, profile_md: str) -> dict[str, Any]:
        profile = self._parse_profile(profile_md or "")
        if not profile:
            return {}
        normalized = self._normalize_profile_update(profile)
        conditions = profile.get("conditions")
        if isinstance(conditions, list):
            normalized["conditions"] = conditions
        if normalized.get("name") and (normalized.get("age") or normalized.get("gender")):
            return normalized
        return {}

    def _text_relevant(self, text: str, query: str) -> bool:
        if not text or not query:
            return False
        keywords = [w for w in query.split() if len(w) > 1]
        return any(kw in text for kw in keywords)

    def normalize_ocr_medications(self, ocr_payload: Optional[dict[str, Any]]) -> list[str]:
        """Normalize OCR medication names and apply typo fixes."""
        if not ocr_payload:
            return []

        names: list[str] = []
        meds = ocr_payload.get("medications", [])
        if isinstance(meds, list):
            for med in meds:
                if isinstance(med, dict):
                    raw = str(med.get("name", "")).strip()
                else:
                    raw = str(med).strip()
                if raw:
                    names.append(raw)

        ocr_results = ocr_payload.get("ocr_results", {})
        if isinstance(ocr_results, dict):
            structured_data = ocr_results.get("structured_data", {})
            if isinstance(structured_data, dict):
                drugs = structured_data.get("drugs", [])
                if isinstance(drugs, list):
                    for drug in drugs:
                        if isinstance(drug, dict) and drug.get("name"):
                            names.append(str(drug["name"]).strip())

        normalized: list[str] = []
        for name in names:
            canon = self._normalize_medication_name(name)
            if canon and canon not in normalized:
                normalized.append(canon)
        return normalized

    def _normalize_medication_name(self, raw: str) -> str:
        cleaned = raw.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"[\s\t\r\n]+", "", cleaned)
        cleaned = re.sub(r"[()\\[\\]{}.,;:\"'`~!@#$%^&*_+=|<>?/\\\\-]", "", cleaned)
        if cleaned in OCR_TYPO_MAP:
            cleaned = OCR_TYPO_MAP[cleaned]
        return cleaned

    def _is_dur_search_supported(self, medication_name: str) -> bool:
        """Heuristic DUR-searchability gate used before deterministic tool calls."""
        if not medication_name or len(medication_name) < 2:
            return False
        if not (settings.data_go_kr_service_key or settings.kpic_dur_api_key):
            return False
        # At least one Korean/alpha/num token should exist.
        return bool(re.search(r"[가-힣A-Za-z0-9]", medication_name))

    def _select_artifacts(self, history: dict[str, Any]) -> list[MemoryArtifactRef]:
        """Pick minimal artifacts likely needed for this turn."""
        refs: list[MemoryArtifactRef] = []

        for category, payload in history.items():
            if category == "structured_memory" and isinstance(payload, dict):
                items = payload.get("items", [])
                for item in items[:3]:
                    if isinstance(item, dict):
                        refs.append(
                            MemoryArtifactRef(
                                category=category,
                                path=item.get("path"),
                                reason="semantic_match",
                                score=0.9,
                            )
                        )
                continue

            if isinstance(payload, list):
                for entry in payload[:2]:
                    if isinstance(entry, dict):
                        refs.append(
                            MemoryArtifactRef(
                                category=category,
                                path=entry.get("path"),
                                reason="keyword_match",
                                score=0.7,
                            )
                        )
            elif isinstance(payload, str) and payload.strip():
                refs.append(
                    MemoryArtifactRef(
                        category=category,
                        path=None,
                        reason="direct_text_hit",
                        score=0.6,
                    )
                )
        return refs[:8]

    def _summarize_artifacts(self, history: dict[str, Any]) -> str:
        parts: list[str] = []
        structured = history.get("structured_memory")
        if isinstance(structured, dict):
            briefs = structured.get("briefs", [])
            if isinstance(briefs, list) and briefs:
                parts.extend(str(brief).strip() for brief in briefs[:3] if str(brief).strip())

        if not parts:
            for category, payload in history.items():
                if category == "structured_memory":
                    continue
                if isinstance(payload, list) and payload:
                    parts.append(f"{category}: {len(payload)}건")
                elif isinstance(payload, str) and payload.strip():
                    parts.append(f"{category}: 텍스트 히트")
        return " | ".join(parts[:5])

    def _extract_query_medications(self, query: str) -> list[str]:
        cleaned_query = strip_wake_words(query)
        if not cleaned_query:
            return []
        meds: list[str] = list(extract_medication_suffix_tokens(cleaned_query))
        for known_name in COMMON_MEDICATION_NAMES:
            if known_name in cleaned_query:
                meds.append(known_name)
        for token in cleaned_query.split():
            token = self._normalize_medication_name(token)
            if not token or is_non_medication_token(token):
                continue
            if "정" in token or "캡슐" in token or "시럽" in token:
                meds.append(token)
        return filter_drug_name_candidates(meds)[:5]

    def extract_ocr_medications_from_text(self, text: str) -> list[str]:
        """Extract medication names from STT text that explicitly reports OCR output."""
        lowered = (text or "").lower()
        if "ocr" not in lowered and "처방전" not in text:
            return []
        if not any(token in text for token in ("결과", "나왔", "읽힌", "인식")):
            return []

        candidates = re.findall(
            r"([가-힣A-Za-z0-9]+(?:장용정|정|캡슐|시럽))",
            text,
        )
        normalized: list[str] = []
        for item in candidates:
            name = self._normalize_medication_name(item)
            if name and name not in normalized:
                normalized.append(name)
        return normalized[:8]

    def extract_spoken_medications_from_text(self, text: str) -> list[str]:
        """Extract medication names from verbal current-medication registration."""
        raw = text or ""
        lowered = raw.lower().strip()
        compact = re.sub(r"[\s\t\r\n.,;:!?~'\"`]+", "", lowered)
        if not compact:
            return []

        explicit_registration = any(
            token in lowered
            for token in (
                "가지고",
                "갖고",
                "먹고 있어",
                "먹고있",
                "복용 중",
                "복용중",
                "처방받",
                "처방 받",
                "받아왔",
                "추가",
                "등록",
                "저장",
                "내 약",
            )
        ) or any(token in compact for token in ("가지고", "갖고", "먹고있", "복용중", "처방받", "받아왔", "추가", "등록", "저장", "내약"))
        blood_pressure_naming = "혈압" in compact and any(alias in compact for alias in SPOKEN_MEDICATION_ALIASES)
        if not explicit_registration and not blood_pressure_naming:
            return []

        safety_question = any(
            token in compact
            for token in (
                "먹어도돼",
                "먹어도되",
                "괜찮",
                "문제",
                "위험",
                "부작용",
                "같이먹",
                "동시에",
                "중단",
                "끊어도",
            )
        )
        strong_registration = any(token in compact for token in ("추가", "등록", "저장", "내약"))
        if safety_question and not strong_registration:
            return []

        candidates: list[str] = list(extract_medication_suffix_tokens(raw))
        for alias, canonical in SPOKEN_MEDICATION_ALIASES.items():
            if alias in compact:
                candidates.append(canonical)
        for match in re.finditer(r"혈압\s*(?:약|양)?\s*([가-힣A-Za-z0-9]+)", raw):
            candidates.append(match.group(1))
        for match in re.finditer(r"(?:약\s*이름은|약은|먹는\s*약은|복용\s*약은)\s*([가-힣A-Za-z0-9]+)", raw):
            candidates.append(match.group(1))

        normalized: list[str] = []
        for candidate in candidates:
            name = self._normalize_spoken_medication_candidate(candidate)
            if name and name not in normalized:
                normalized.append(name)
        return normalized[:8]

    def _normalize_spoken_medication_candidate(self, raw: str) -> str:
        name = self._normalize_medication_name(raw)
        if not name or is_non_medication_token(name):
            return ""
        for alias, canonical in SPOKEN_MEDICATION_ALIASES.items():
            if name == alias or alias in name:
                return canonical
        blocked = {"혈압", "혈압약", "당뇨", "당뇨약", "약", "확인", "한번", "오늘", "저녁"}
        if name in blocked:
            return ""
        if any(name.endswith(suffix) for suffix in ("정", "장용정", "캡슐", "시럽")):
            return name
        if name in COMMON_MEDICATION_NAMES:
            return name
        return ""

    @staticmethod
    def _format_prescription_log(med_names: list[str], *, recorded_at: str) -> str:
        return (
            f"# 현재 복용 약 요약\n> 최종 갱신: {recorded_at}\n\n## 약품 목록\n"
            + "\n".join(f"- {name}" for name in med_names)
            + "\n"
        )

    def _format_patient_special_notes(
        self,
        profile: dict[str, Any],
        history: str,
        medication_events: str,
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = profile.get("name") or "등록 전"
        age = profile.get("age") or "-"
        gender = profile.get("gender") or "-"
        conditions = self._normalize_conditions(profile.get("conditions"))
        notes: list[str] = [
            f"- 대상자: {name}",
            f"- 연령/성별: {age} / {gender}",
            f"- 기저질환: {', '.join(conditions) if conditions else '-'}",
        ]
        recent_history = self._latest_history_excerpt(history)
        if recent_history:
            notes.append(f"- 최근 상담 특이사항: {recent_history}")
        latest_event = self._latest_medication_event_summary(medication_events)
        if latest_event:
            notes.append(f"- 최근 복약 이력: {latest_event}")
        return (
            f"# 현재 환자 특이 이력사항\n> 최종 갱신: {now}\n\n"
            + "\n".join(notes)
            + "\n"
        )

    @staticmethod
    def _format_context_memory_from_latest_log(content: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        question = ""
        answer = ""
        if "## 질문" in content:
            question = content.split("## 질문", 1)[1].split("## 응답", 1)[0].strip()
        if "## 응답" in content:
            answer = content.split("## 응답", 1)[1].strip()
        return (
            f"# 대화 컨텍스트 메모리\n> 최종 갱신: {now}\n\n"
            "## 최근 대화 요약\n"
            f"- 질문: {question[:100]}\n"
            f"- 핵심 응답: {answer[:200]}\n"
        )

    @staticmethod
    def _latest_history_excerpt(history: str) -> str:
        if not history.strip():
            return ""
        chunks = [chunk.strip() for chunk in history.split("---") if chunk.strip()]
        return re.sub(r"\s+", " ", chunks[-1])[:180] if chunks else ""

    def _latest_medication_event_summary(self, content: str) -> str:
        events = self._parse_medication_events(content)
        if not events:
            return ""
        event = events[-1]
        return " ".join(
            str(part)
            for part in (
                event.get("date"),
                event.get("time"),
                event.get("medication"),
                event.get("action"),
            )
            if part
        )

    def _extract_medications_from_markdown(self, content: str) -> list[str]:
        names: list[str] = []
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            candidate = stripped[2:].split(":", 1)[0].strip()
            candidate = self._normalize_medication_name(candidate)
            if candidate and candidate not in names and not is_non_medication_token(candidate):
                names.append(candidate)
        return names[:10]

    @staticmethod
    def _parse_medication_events(content: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            try:
                payload = json.loads(stripped[2:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _medication_events_relevant(self, content: str, query: str) -> bool:
        lowered = (query or "").lower()
        if any(token in lowered for token in ("어제", "오늘", "그제", "복용", "먹었", "먹었다", "기록", "약")):
            return True
        return self._text_relevant(content, query)

    @staticmethod
    def _extract_event_date(text: str, base: datetime) -> str:
        match = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"
        if "어제" in text:
            return (base - timedelta(days=1)).date().isoformat()
        if "그제" in text:
            return (base - timedelta(days=2)).date().isoformat()
        return base.date().isoformat()

    @staticmethod
    def _extract_event_time(text: str) -> str:
        match = re.search(
            r"(오전|오후|밤|저녁|아침|점심)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?",
            text,
        )
        if not match:
            return ""
        meridiem = match.group(1) or ""
        hour = int(match.group(2))
        minute = int(match.group(3) or 0)
        if meridiem in {"오후", "저녁", "밤"} and hour < 12:
            hour += 12
        if meridiem == "오전" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _first_medication_from_text(text: str) -> str:
        explicit = re.search(r"([가-힣A-Za-z0-9]+(?:장용정|정|캡슐|시럽|약))", text)
        return explicit.group(1) if explicit else ""

    def _format_profile_markdown(
        self,
        speaker_id: str,
        profile: dict[str, Any],
        *,
        registered_at: str,
    ) -> str:
        conditions = self._normalize_conditions(profile.get("conditions"))
        pending = profile.get("pending_identity_action", "")
        pending_since = profile.get("pending_identity_since", "")
        candidate = profile.get("pending_identity_candidate") or {}
        candidate_text = "-"
        if isinstance(candidate, dict) and candidate:
            candidate_text = json.dumps(candidate, ensure_ascii=False)
        return (
            "# 복약 관리 프로필\n"
            f"> 등록일: {registered_at}\n\n"
            "| 항목 | 값 |\n|------|----|\n"
            f"| ID | {speaker_id} |\n"
            f"| 이름 | {profile.get('name') or '-'} |\n"
            f"| 성별 | {profile.get('gender') or '-'} |\n"
            f"| 연령 | {profile.get('age') or '-'} |\n"
            f"| 기저질환 | {', '.join(conditions) if conditions else '-'} |\n"
            f"| 최종대화 | {profile.get('last_seen_at') or '-'} |\n"
            f"| 최종확인 | {profile.get('verified_at') or '-'} |\n"
            f"| 신원확인상태 | {pending or '-'} |\n"
            f"| 신원확인시작 | {pending_since or '-'} |\n"
            f"| 신원후보 | {candidate_text} |\n"
        )

    def _normalize_conditions(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            if not value or value == "-":
                return []
            return [item.strip() for item in value.split(",") if item.strip()]
        return []
