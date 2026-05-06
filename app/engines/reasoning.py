"""추론 엔진 (Reasoning Engine) — 지휘통제실.

server.mermaid 매핑:
  RE_Intent            → classify_intent(), plan_tasks()
  CE_Prescription_OCR  → request_ocr()
  RE_Core_Msg          → synthesize_core_message()
"""
import logging
from typing import Any, Optional

from app.engines.memory import MemoryEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.schemas.engine_contracts import (
    ReasoningMode,
    ReasoningRouteDecision,
    ReasoningRouteInput,
    ReasoningTask,
)
from app.tools import dur_api, hira_api, health_supplement, llm_search

logger = logging.getLogger(__name__)


class IntentType:
    SMALLTALK = "smalltalk"
    MEDICATION_QUERY = "medication_query"
    PRESCRIPTION_CHECK = "prescription_check"
    SUPPLEMENT_QUERY = "supplement_query"
    DRUG_IDENTIFICATION = "drug_identification"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


MEDICATION_KEYWORDS = [
    "약", "복용", "먹어도", "드셔도", "먹고", "처방", "부작용",
    "금기", "주의", "효과", "효능", "용량", "언제",
]
SUPPLEMENT_KEYWORDS = [
    "비타민", "영양제", "건기식", "건강식품", "유산균", "오메가",
    "칼슘", "철분", "홍삼", "녹용",
]
EMERGENCY_KEYWORDS = [
    "쓰러", "의식", "호흡", "출혈", "경련", "응급", "119",
    "심장", "뇌졸중", "마비",
]
DRUG_ID_KEYWORDS = ["알약", "낱알", "이거 뭐", "무슨 약", "약 이름"]


class ReasoningEngine:
    """추론 엔진: 의도 파악, 태스크 설계, 도구 오케스트레이션, 핵심 답변 생성."""

    def __init__(
        self,
        memory_engine: MemoryEngine,
        llm_judge: LLMJudgeEngine,
    ):
        self.memory = memory_engine
        self.llm_judge = llm_judge

    # ── RE_Intent: 의도 파악 및 태스크 설계 ──

    def classify_intent(self, text: str) -> str:
        text_lower = text.lower().strip()

        if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
            return IntentType.EMERGENCY

        if any(kw in text_lower for kw in DRUG_ID_KEYWORDS):
            return IntentType.DRUG_IDENTIFICATION

        if any(kw in text_lower for kw in SUPPLEMENT_KEYWORDS):
            if self._contains_medication_signal(text_lower):
                return IntentType.MEDICATION_QUERY
            return IntentType.SUPPLEMENT_QUERY

        if self._contains_medication_signal(text_lower):
            return IntentType.MEDICATION_QUERY

        return IntentType.SMALLTALK

    def plan_tasks(self, intent: str, context: dict) -> list[dict[str, Any]]:
        """의도에 따른 태스크 목록 생성."""
        tasks: list[dict[str, Any]] = []

        if intent == IntentType.EMERGENCY:
            tasks.append({
                "type": "emergency_alert",
                "priority": 0,
                "description": "긴급 상황 감지 — 즉시 경고",
            })
            return tasks

        if intent == IntentType.DRUG_IDENTIFICATION:
            tasks.append({
                "type": "request_ocr",
                "priority": 1,
                "description": "처방전/약물 OCR 요청",
            })
            tasks.append({
                "type": "hira_lookup",
                "priority": 2,
                "description": "의약품 낱알식별 API 조회 (T1)",
            })

        if intent == IntentType.MEDICATION_QUERY:
            tasks.append({
                "type": "search_history",
                "priority": 1,
                "description": "과거 이력 조회 (ME_RAG)",
            })
            tasks.append({
                "type": "dur_check",
                "priority": 2,
                "description": "DUR 일괄 조회 (T2~T10)",
            })
            tasks.append({
                "type": "llm_judge_verify",
                "priority": 3,
                "description": "LLM as a Judge 팩트 체킹",
            })

        if intent == IntentType.SUPPLEMENT_QUERY:
            tasks.append({
                "type": "supplement_lookup",
                "priority": 1,
                "description": "건강기능식품 조회 (T11, T12)",
            })
            tasks.append({
                "type": "search_history",
                "priority": 2,
                "description": "관련 이력 검색",
            })

        if intent == IntentType.PRESCRIPTION_CHECK:
            tasks.append({
                "type": "search_history",
                "priority": 1,
                "description": "처방전 이력 확인",
            })
            tasks.append({
                "type": "dur_check",
                "priority": 2,
                "description": "처방 약물 DUR 확인",
            })

        return sorted(tasks, key=lambda t: t["priority"])

    def route_execution(self, route_input: ReasoningRouteInput) -> ReasoningRouteDecision:
        """Decide tool/frontier/memory route before task execution."""
        text = route_input.text.strip()
        context = route_input.context or {}

        if not text:
            return ReasoningRouteDecision(
                mode=ReasoningMode.ASK_USER_CLARIFY,
                intent=IntentType.UNKNOWN,
                rationale="empty_user_input",
                tasks=[],
            )

        intent = self.classify_intent(text)
        if route_input.is_smalltalk and intent == IntentType.SMALLTALK:
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=intent,
                rationale="smalltalk_detected",
                tasks=[],
            )

        if intent == IntentType.EMERGENCY:
            return ReasoningRouteDecision(
                mode=ReasoningMode.TOOL_FIRST,
                intent=intent,
                rationale="emergency_policy_first",
                tasks=[
                    ReasoningTask(
                        type="emergency_alert",
                        priority=0,
                        description="긴급 상황 감지 — 즉시 경고",
                    )
                ],
            )

        planned = self.plan_tasks(intent, context)
        if planned:
            return ReasoningRouteDecision(
                mode=ReasoningMode.TOOL_FIRST,
                intent=intent,
                rationale="deterministic_tools_available",
                tasks=[ReasoningTask(**task) for task in planned],
            )

        prescription_log = str(context.get("prescription_log", "")).strip()
        if prescription_log:
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=intent,
                rationale="memory_context_available_without_tool_plan",
                tasks=[],
            )

        return ReasoningRouteDecision(
            mode=ReasoningMode.FRONTIER_FIRST,
            intent=intent,
            rationale="no_deterministic_tool_plan_or_memory_context",
            tasks=[],
        )

    # ── 태스크 실행 오케스트레이션 ──

    async def execute_tasks(
        self,
        text: str,
        intent: str,
        context: dict,
        tasks: list[dict[str, Any] | ReasoningTask],
    ) -> dict[str, Any]:
        """태스크 목록을 순차 실행하고 결과를 수집."""
        results: dict[str, Any] = {
            "intent": intent,
            "query": text,
            "task_results": {},
            "emergency": False,
        }

        normalized_tasks: list[dict[str, Any]] = []
        for task in tasks:
            if isinstance(task, ReasoningTask):
                normalized_tasks.append(task.model_dump())
            elif isinstance(task, dict):
                normalized_tasks.append(task)

        for task in normalized_tasks:
            task_type = task.get("type", "")
            if not task_type:
                continue
            try:
                if task_type == "emergency_alert":
                    results["emergency"] = True
                    results["task_results"]["emergency"] = {
                        "message": "긴급 상황이 감지되었습니다. 119에 연락하세요.",
                    }

                elif task_type == "search_history":
                    history = await self.memory.search_history(
                        text, context.get("speaker_id")
                    )
                    results["task_results"]["history"] = history

                elif task_type == "dur_check":
                    drug_names = self._extract_drug_names(text, context)
                    dur_results: dict[str, Any] = {}
                    for name in drug_names:
                        dur_results[name] = await dur_api.check_all_dur(name)
                    results["task_results"]["dur"] = dur_results

                elif task_type == "hira_lookup":
                    drug_names = self._extract_drug_names(text, context)
                    hira_results = {}
                    for name in drug_names:
                        hira_results[name] = await hira_api.identify_medicine(
                            item_name=name
                        )
                    results["task_results"]["hira"] = hira_results

                elif task_type == "supplement_lookup":
                    supplement_names = self._extract_supplement_names(text)
                    supp_results = {}
                    for name in supplement_names:
                        supp_results[name] = (
                            await health_supplement.get_supplement_detail(
                                product_name=name
                            )
                        )
                    results["task_results"]["supplements"] = supp_results

                elif task_type == "llm_judge_verify":
                    pass  # synthesize 단계에서 실행

                elif task_type == "request_ocr":
                    results["task_results"]["ocr_requested"] = True

            except Exception as e:
                logger.error("Task execution error [%s]: %s", task_type, e)
                results["task_results"][task_type] = {"error": str(e)}

        return results

    # ── RE_Core_Msg: 핵심 답변 생성 ──

    async def synthesize_core_message(
        self,
        execution_results: dict[str, Any],
        *,
        verify_with_judge: bool = True,
    ) -> str:
        """실행 결과를 종합하여 순수 팩트 데이터 문자열 생성."""
        intent = execution_results.get("intent", "")
        query = execution_results.get("query", "")
        task_results = execution_results.get("task_results", {})

        if execution_results.get("emergency"):
            return (
                "긴급 상황이 감지되었습니다. "
                "즉시 119에 전화하시거나 가까운 응급실을 방문해 주세요. "
                "환자의 상태를 주시하면서 도움을 기다려 주세요."
            )

        parts: list[str] = []

        if task_results.get("ocr_requested"):
            parts.append(
                "약물 식별을 위해 처방전 또는 약 사진이 필요합니다."
            )

        dur_data = task_results.get("dur", {})
        if dur_data:
            for drug_name, dur_result in dur_data.items():
                dur_summary = self._summarize_dur(drug_name, dur_result)
                if dur_summary:
                    parts.append(dur_summary)

        supp_data = task_results.get("supplements", {})
        if supp_data:
            for name, result in supp_data.items():
                if result.get("success") and result.get("items"):
                    item = result["items"][0]
                    parts.append(
                        f"{name}: {item.get('RAWMTR_NM', '성분 정보 없음')}"
                    )

        history_data = task_results.get("history", {})
        if history_data:
            structured = history_data.get("structured_memory", {})
            briefs = structured.get("briefs", []) if isinstance(structured, dict) else []
            for brief in briefs[:2]:
                parts.append(brief)
            if not briefs:
                parts.append("과거 상담 이력을 참고하였습니다.")

        if not parts:
            search_result = await llm_search.llm_search(query)
            if search_result.get("success") and search_result.get("answer"):
                core_msg = search_result["answer"]
            else:
                core_msg = f"'{query}'에 대해 확인된 정보가 제한적입니다. 약사 또는 의사에게 직접 상담을 권장합니다."
        else:
            core_msg = " ".join(parts)

        if verify_with_judge:
            verified = await self.llm_judge.verify_fact(core_msg, query)
            if verified.get("needs_correction"):
                core_msg = verified.get("corrected", core_msg)

        return core_msg

    # ── CE_Prescription_OCR: 처방전 OCR 요청 ──

    def request_ocr(self) -> dict:
        """로컬 에이전트에 처방전 OCR 촬영 요청."""
        return {
            "action": "request_ocr",
            "message": "처방전 또는 약 사진을 찍어주세요.",
        }

    # ── 내부 유틸 ──

    def _extract_drug_names(
        self, text: str, context: dict
    ) -> list[str]:
        names: list[str] = []
        prescription_log = context.get("prescription_log", "")
        if prescription_log:
            for line in prescription_log.split("\n"):
                if line.strip().startswith("- ") and any(
                    c.isalpha() for c in line
                ):
                    name = line.strip().lstrip("- ").split(",")[0].strip()
                    if name and len(name) > 1:
                        names.append(name)

        words = text.split()
        for word in words:
            if (
                len(word) > 2
                and ("정" in word or "캡슐" in word or "시럽" in word)
            ):
                names.append(word)

        return names[:5] if names else [text[:20]]

    def _extract_supplement_names(self, text: str) -> list[str]:
        names: list[str] = []
        known = [
            "비타민", "오메가", "칼슘", "철분", "유산균",
            "홍삼", "녹용", "프로바이오틱스", "루테인", "마그네슘",
        ]
        for kw in known:
            if kw in text:
                names.append(kw)
        return names if names else [text[:20]]

    def _summarize_dur(
        self, drug_name: str, dur_result: dict
    ) -> str:
        """DUR 결과를 요약 문자열로 변환."""
        summaries: list[str] = []

        for check_type, result in dur_result.items():
            if not isinstance(result, dict):
                continue
            items = result.get("items", [])
            if not items:
                continue

            endpoint_desc = result.get("endpoint", check_type)
            for item in items[:3]:
                if isinstance(item, dict):
                    name = (
                        item.get("ITEM_NAME")
                        or item.get("itemName")
                        or drug_name
                    )
                    note = (
                        item.get("PROHBT_CONTENT")
                        or item.get("REMARK")
                        or item.get("NOTE")
                        or ""
                    )
                    if note:
                        summaries.append(f"{name} ({endpoint_desc}): {note[:100]}")

        if summaries:
            return f"{drug_name} DUR 확인 결과: " + "; ".join(summaries)
        return ""

    def _contains_medication_signal(self, text_lower: str) -> bool:
        if any(kw in text_lower for kw in MEDICATION_KEYWORDS if kw != "약"):
            return True
        # Prevent false positives like "요약" while keeping short utterances such as
        # "약 맞나요?" as medication queries.
        return "약" in text_lower and "요약" not in text_lower
