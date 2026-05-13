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
from app.tools import llm_search

logger = logging.getLogger(__name__)

OCR_TYPO_MAP = {
    "타이레롤": "타이레놀",
    "와파린정정": "와파린정",
    "아스피린장용정정": "아스피린장용정",
}


class MemoryEngine:
    """메모리 엔진: 사용자 식별, 컨텍스트 관리, RAG, 데이터 압축."""

    def __init__(self):
        self.store = md_store
        self.structured_memory = StructuredMemoryService()

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.structured_memory.initialize()

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
            name = dur.get("name", "?")
            contras = len(dur.get("contraindications", []))
            cautions = len(dur.get("precautions", []))
            dur_lines.append(f"- {name}: 금기 {contras}건 / 주의 {cautions}건")

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

    # ── ME_Context: 사용자 식별 및 컨텍스트 로드 ──

    async def load_context(self, speaker_id: Optional[str] = None) -> dict[str, Any]:
        """현재 대화자의 컨텍스트를 로드. 새 환자이면 프로필 생성."""
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
        return {
            "speaker_id": speaker_id,
            "exists": exists,
            "profile": profile,
            "profile_text": profile_md,
            "pending_identity_action": profile.get("pending_identity_action", ""),
            "pending_identity_candidate": profile.get("pending_identity_candidate", {}),
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
        merged = {**current, **{key: value for key, value in profile_update.items() if value}}
        timestamp = (now or datetime.now()).isoformat(timespec="seconds")
        if mark_seen:
            merged["last_seen_at"] = timestamp
        if mark_verified:
            merged["verified_at"] = timestamp
        merged["pending_identity_action"] = pending_identity_action
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

    # ── ME_Parse: 환자 개인정보 구분 및 주요 정보 로그 파싱 ──

    async def parse_patient_info(self, raw_data: dict) -> dict:
        return {
            "name": raw_data.get("name", ""),
            "age": raw_data.get("age", ""),
            "gender": raw_data.get("gender", ""),
            "conditions": raw_data.get("conditions", []),
            "allergies": raw_data.get("allergies", []),
        }

    # ── ME_RAG: 환자 관리 및 관련 이력 검색 ──

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
            profile_update = self._extract_profile_from_text(query_text)
            if profile_update:
                await self._save_user_profile(speaker_id, profile_update)
                await self.update_flash_profile(speaker_id, profile_update)

            existing = await self.store.read_user_file(speaker_id, "history.md")
            entry = (
                f"\n---\n### {now} ({resp_type})\n"
                f"- Q: {query_text[:200]}\n"
                f"- A: {answer_text[:300]}\n"
            )
            await self.store.save_user_file(
                speaker_id, "history.md", existing + entry,
            )

    async def update_flash_profile(self, speaker_id: str, profile: dict) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = (
            f"# 현재 환자 프로필\n> 최종 갱신: {now}\n\n"
            f"| 항목 | 값 |\n|------|----|\n"
            f"| 이름 | {profile.get('name', '-')} |\n"
            f"| ID | {speaker_id} |\n"
            f"| 연령 | {profile.get('age', '-')} |\n"
            f"| 성별 | {profile.get('gender', '-')} |\n"
            f"| 기저질환 | {', '.join(profile.get('conditions', []))} |\n"
        )
        await self.store.write_flash("current_user_profile", content)
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
        name_match = re.search(
            r"(?:제\s*이름은|이름은)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
            text,
        ) or re.search(
            r"(?:저는|나는)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
            text,
        ) or re.search(
            r"^\s*([가-힣]{2,5})\s*(?:남자|남성|여자|여성)",
            text,
        ) or re.search(
            r"^\s*([가-힣]{2,5})\s*,?\s*\d{1,3}\s*(?:살|세)",
            text,
        )
        if name_match:
            profile["name"] = name_match.group(1)
        age_match = re.search(r"(\d{1,3})\s*(?:살|세)", text)
        if age_match:
            profile["age"] = age_match.group(1)
        if "남자" in text or "남성" in text:
            profile["gender"] = "남성"
        elif "여자" in text or "여성" in text:
            profile["gender"] = "여성"
        conditions = []
        for token in ("고혈압", "당뇨", "천식", "신장질환", "간질환", "심장질환", "임신 가능성", "임신"):
            if token in text:
                conditions.append(token)
        if conditions:
            profile["conditions"] = conditions
        return profile

    async def _register_new_user(self, speaker_id: str) -> None:
        now = datetime.now()
        profile = self._format_profile_markdown(
            speaker_id,
            {
                "name": "",
                "gender": "",
                "age": "",
                "conditions": [],
                "pending_identity_action": "registration",
                "pending_identity_candidate": {},
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
            f"# 신규 환자 등록\n"
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
        meds: list[str] = []
        for token in query.split():
            token = self._normalize_medication_name(token)
            if not token:
                continue
            if len(token) < 2:
                continue
            if "정" in token or "캡슐" in token or "시럽" in token:
                meds.append(token)
        # keep uniqueness
        unique: list[str] = []
        for med in meds:
            if med not in unique:
                unique.append(med)
        return unique[:5]

    def _format_profile_markdown(
        self,
        speaker_id: str,
        profile: dict[str, Any],
        *,
        registered_at: str,
    ) -> str:
        conditions = self._normalize_conditions(profile.get("conditions"))
        pending = profile.get("pending_identity_action", "")
        candidate = profile.get("pending_identity_candidate") or {}
        candidate_text = "-"
        if isinstance(candidate, dict) and candidate:
            candidate_text = json.dumps(candidate, ensure_ascii=False)
        return (
            "# 환자 프로필\n"
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
