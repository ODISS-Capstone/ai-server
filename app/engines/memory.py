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
from datetime import datetime
from typing import Any, Optional

from app.database.md_store import md_store
from app.memory import StructuredMemoryService

logger = logging.getLogger(__name__)


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

    async def _register_new_user(self, speaker_id: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profile = (
            f"# 환자 프로필\n"
            f"> 등록일: {now}\n\n"
            f"| 항목 | 값 |\n|------|----|\n"
            f"| ID | {speaker_id} |\n"
            f"| 이름 | - |\n| 성별 | - |\n| 연령 | - |\n| 기저질환 | - |\n"
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
            f"> 등록 시각: {now}\n"
            f"> ID: {speaker_id}\n"
        )
        await self.store.save("patients", registration)

    def _parse_profile(self, profile_md: str) -> dict:
        result: dict[str, Any] = {}
        for line in profile_md.split("\n"):
            if "|" in line and "---" not in line and "항목" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) == 2:
                    key, value = parts
                    if value != "-":
                        result[key] = value
        return result

    def _text_relevant(self, text: str, query: str) -> bool:
        if not text or not query:
            return False
        keywords = [w for w in query.split() if len(w) > 1]
        return any(kw in text for kw in keywords)
