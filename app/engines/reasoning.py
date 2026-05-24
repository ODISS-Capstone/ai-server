"""추론 엔진 (Reasoning Engine) — 지휘통제실.

server.mermaid 매핑:
  RE_Intent            → classify_intent(), plan_tasks()
  CE_Prescription_OCR  → request_ocr()
  RE_Core_Msg          → synthesize_core_message()
"""
import logging
import asyncio
import re
from typing import Any, Optional

from app.engines.memory import MemoryEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.schemas.engine_contracts import (
    ReasoningMode,
    ReasoningRouteDecision,
    ReasoningRouteInput,
    ReasoningTask,
)
from app.services.medication_extraction import (
    extract_medication_suffix_tokens,
    filter_drug_name_candidates,
    is_ocr_capture_request_text,
    is_wake_word_only,
    strip_wake_words,
)
from app.services.patient_safety import classify_patient_safety_situation
from app.services.llm_queue import run_with_engine_queue
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
    "약", "복용", "먹어도", "먹으면", "같이 먹", "드셔도", "먹고", "처방", "부작용",
    "금기", "주의", "효과", "효능", "용량", "언제",
    "와파린", "아스피린", "혈압약", "당뇨약", "인슐린",
]
SUPPLEMENT_KEYWORDS = [
    "비타민", "영양제", "건기식", "건강식품", "유산균", "오메가",
    "칼슘", "철분", "홍삼", "녹용",
]
EMERGENCY_KEYWORDS = [
    "쓰러",
    "의식이 없",
    "의식 저하",
    "의식을 잃",
    "호흡곤란",
    "호흡 곤란",
    "숨이 차",
    "숨쉬기 힘",
    "피가 멈추지",
    "심한 출혈",
    "경련",
    "응급",
    "119",
    "가슴 통증",
    "가슴이 답답",
    "가슴 답답",
    "가슴 압박",
    "가슴이 조",
    "가슴이 눌",
    "흉통",
    "뇌졸중",
    "마비",
]
DRUG_ID_KEYWORDS = ["알약", "낱알", "이거 뭐", "무슨 약", "약 이름"]
PROFILE_OR_MEMORY_KEYWORDS = [
    "제 이름",
    "저는",
    "처음 왔",
    "기억해",
    "기억하고",
    "생활습관",
    "산책",
    "안부",
    "격려",
    "내가 누구",
    "누구인지",
    "내 프로필",
]
DIRECT_MEDICAL_QUERY_KEYWORDS = [
    "먹어도", "같이 먹", "드셔도", "부작용", "금기", "주의", "효능",
    "효과", "용량", "dur", "복용지도", "복용 계획",
]
MEMORY_RECALL_KEYWORDS = ["아까", "이전", "어제", "다시 말", "뭐였", "읽힌"]
COMMON_MEDICATION_NAMES = [
    "와파린",
    "아스피린",
    "로사르탄",
    "오메프라졸",
    "인슐린",
    "메트포르민",
    "암로디핀",
    "혈압약",
    "당뇨약",
]


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
        if is_wake_word_only(text):
            return IntentType.SMALLTALK

        text_lower = text.lower().strip()

        safety = classify_patient_safety_situation(text)
        if safety:
            return IntentType.EMERGENCY if safety.severity == "emergency" else IntentType.MEDICATION_QUERY

        if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
            return IntentType.EMERGENCY

        if self._is_missing_ocr_image_request(text_lower):
            return IntentType.MEDICATION_QUERY

        if self._is_nonmedical_smalltalk_request(text_lower):
            return IntentType.SMALLTALK

        if self._is_medication_record_request(text_lower):
            return IntentType.MEDICATION_QUERY

        if self._is_memory_recall_query(text_lower) and self._contains_medication_signal(text_lower):
            if any(kw in text_lower for kw in ("사진", "읽힌", "약 이름")):
                return IntentType.MEDICATION_QUERY
            if "약" in text_lower:
                return IntentType.MEDICATION_QUERY
            return IntentType.SMALLTALK

        if self._is_profile_or_lifestyle_context(text_lower):
            return IntentType.SMALLTALK

        if self._is_meal_medication_guidance_request(text):
            return IntentType.MEDICATION_QUERY

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
                "description": "질문 의도에 맞는 DUR 항목 선택 조회",
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

        profile = context.get("user_profile") or {}
        if self._is_profile_identity_recall(text, profile):
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.SMALLTALK,
                rationale="profile_identity_recall",
                tasks=[],
            )

        intent = self.classify_intent(text)
        safety = classify_patient_safety_situation(text)
        if safety and safety.severity != "emergency":
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.MEDICATION_QUERY,
                rationale=f"deterministic_patient_safety:{safety.key}",
                tasks=[],
            )
        if "ocr" in text.lower() and any(token in text for token in ("결과", "나왔")):
            return ReasoningRouteDecision(
                mode=ReasoningMode.TOOL_FIRST,
                intent=IntentType.MEDICATION_QUERY,
                rationale="ocr_result_requires_prescription_logging",
                tasks=[
                    ReasoningTask(
                        type="dur_product_info",
                        priority=1,
                        description="OCR 약물 DUR 품목정보 확인",
                    )
                ],
            )
        if (
            intent in {IntentType.MEDICATION_QUERY, IntentType.DRUG_IDENTIFICATION}
            and self._is_memory_recall_query(text.lower())
            and not self._is_meal_medication_guidance_request(text)
            and not any(kw in text.lower() for kw in DIRECT_MEDICAL_QUERY_KEYWORDS)
            and (
                str(context.get("prescription_log", "")).strip()
                or str(context.get("context_memory", "")).strip()
                or context.get("memory_prompt")
            )
        ):
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.MEDICATION_QUERY,
                rationale="medication_memory_recall_available",
                tasks=[],
            )

        if route_input.is_smalltalk and intent == IntentType.SMALLTALK:
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=intent,
                rationale="smalltalk_detected",
                tasks=[],
            )

        if intent == IntentType.SMALLTALK:
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.UNKNOWN,
                rationale="out_of_scope_smalltalk_suppressed",
                tasks=[],
            )

        if intent == IntentType.EMERGENCY:
            return ReasoningRouteDecision(
                mode=ReasoningMode.FRONTIER_FIRST,
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

        lowered = text.lower()
        if self._is_missing_ocr_image_request(lowered):
            return ReasoningRouteDecision(
                mode=ReasoningMode.TOOL_FIRST,
                intent=IntentType.MEDICATION_QUERY,
                rationale="ocr_capture_requested",
                tasks=[
                    ReasoningTask(
                        type="request_ocr",
                        priority=1,
                        description="약봉투/처방전 OCR 촬영 요청",
                    )
                ],
            )

        if intent in {IntentType.MEDICATION_QUERY, IntentType.DRUG_IDENTIFICATION} and self._is_meal_medication_guidance_request(text):
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.MEDICATION_QUERY,
                rationale=(
                    "stored_medication_meal_guidance"
                    if self._context_has_medication(context)
                    else "meal_guidance_missing_medication_context"
                ),
                tasks=[],
            )

        if self._is_medication_record_request(lowered):
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.MEDICATION_QUERY,
                rationale="medication_record_memory_write",
                tasks=[],
            )

        if self._is_generic_blood_pressure_medication_overview_request(text):
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.MEDICATION_QUERY,
                rationale="generic_blood_pressure_medication_overview",
                tasks=[],
            )

        if (
            intent in {IntentType.MEDICATION_QUERY, IntentType.DRUG_IDENTIFICATION}
            and self._is_current_medication_record_recall(text)
            and self._context_has_medication(context)
        ):
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent=IntentType.MEDICATION_QUERY,
                rationale="stored_medication_record_recall",
                tasks=[],
            )

        planned = self.plan_tasks(intent, context)
        if intent == IntentType.MEDICATION_QUERY and any(kw in lowered for kw in SUPPLEMENT_KEYWORDS):
            planned = [
                {"type": "supplement_lookup", "priority": 1, "description": "건강기능식품 조회 (T11, T12)"},
                {"type": "search_history", "priority": 2, "description": "관련 이력 검색"},
            ]
        if planned:
            return ReasoningRouteDecision(
                mode=ReasoningMode.TOOL_FIRST,
                intent=intent,
                rationale="deterministic_tools_available",
                tasks=[ReasoningTask(**task) for task in planned],
            )

        prescription_log = str(context.get("prescription_log", "")).strip()
        if prescription_log and intent != IntentType.SMALLTALK:
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
                    endpoint_keys = self._select_dur_endpoint_keys(
                        text=text,
                        context=context,
                        medication_count=len(drug_names),
                    )

                    async def _run_dur_check() -> list[dict[str, Any]]:
                        return await dur_api.check_dur_for_prescription(
                            [{"name": name} for name in drug_names],
                            endpoint_keys=endpoint_keys,
                        )

                    rows = await run_with_engine_queue("dur", _run_dur_check)
                    dur_results = {
                        row["medication"]: row.get("dur", {})
                        for row in rows
                    }
                    results["task_results"]["dur"] = dur_results
                    results["task_results"]["dur_endpoint_keys"] = endpoint_keys

                elif task_type == "dur_product_info":
                    drug_names = self._extract_drug_names(text, context)

                    async def _run_dur_product_info() -> list[dict[str, Any]]:
                        return await dur_api.check_dur_for_prescription(
                            [{"name": name} for name in drug_names],
                            endpoint_keys=("dur_product_info",),
                        )

                    rows = await run_with_engine_queue("dur", _run_dur_product_info)
                    dur_results = {
                        row["medication"]: row.get("dur", {})
                        for row in rows
                    }
                    results["task_results"]["dur"] = dur_results
                    results["task_results"]["dur_endpoint_keys"] = ["dur_product_info"]

                elif task_type == "hira_lookup":
                    drug_names = self._extract_drug_names(text, context)

                    async def _run_hira_lookup() -> list[dict[str, Any]]:
                        return await asyncio.gather(
                            *(hira_api.identify_medicine(item_name=name) for name in drug_names)
                        )

                    hira_rows = await run_with_engine_queue("tool", _run_hira_lookup)
                    hira_results = dict(zip(drug_names, hira_rows))
                    results["task_results"]["hira"] = hira_results

                elif task_type == "supplement_lookup":
                    supplement_names = self._extract_supplement_names(text)

                    async def _run_supplement_lookup() -> list[dict[str, Any]]:
                        return await asyncio.gather(
                            *(
                                health_supplement.get_supplement_detail(product_name=name)
                                for name in supplement_names
                            )
                        )

                    supplement_rows = await run_with_engine_queue("tool", _run_supplement_lookup)
                    supp_results = dict(zip(supplement_names, supplement_rows))
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
                "상태를 계속 살피면서 도움을 기다려 주세요."
            )

        parts: list[str] = []

        if task_results.get("ocr_requested"):
            parts.append(
                self.request_ocr()["message"]
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
                else:
                    parts.append(
                        f"{name}: 제품마다 성분과 함량이 달라 제품명이나 성분표 확인이 필요합니다."
                    )

        history_data = task_results.get("history", {})
        if history_data and intent not in {IntentType.SUPPLEMENT_QUERY, IntentType.SMALLTALK}:
            structured = history_data.get("structured_memory", {})
            briefs = structured.get("briefs", []) if isinstance(structured, dict) else []
            for brief in briefs[:2]:
                parts.append(brief)
            if not briefs and not self._is_profile_identity_recall(query, {"name": "_"}):
                parts.append("이전 기록에서 바로 답할 핵심 내용을 찾지 못했습니다.")

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
            "message": (
                "알겠습니다. 카메라 앞으로 약봉투를 잘 보이게 보여주세요. "
                "글자가 흔들리지 않도록 잠시만 멈춰주세요. "
                "5, 4, 3, 2, 1. 촬영하겠습니다."
            ),
        }

    # ── 내부 유틸 ──

    def _extract_drug_names(
        self, text: str, context: dict
    ) -> list[str]:
        query = strip_wake_words(text)
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

        names.extend(extract_medication_suffix_tokens(query))
        for known_name in COMMON_MEDICATION_NAMES:
            if known_name in query:
                names.append(known_name)

        filtered = filter_drug_name_candidates(names)
        return filtered[:5]

    def _extract_supplement_names(self, text: str) -> list[str]:
        names: list[str] = []
        known = [
            "비타민", "오메가", "칼슘", "철분", "유산균",
            "홍삼", "녹용", "프로바이오틱스", "루테인", "마그네슘",
        ]
        for kw in known:
            if kw in text:
                names.append(kw)
        return filter_drug_name_candidates(names)

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

            endpoint_desc = self._friendly_dur_endpoint_label(
                str(result.get("endpoint") or check_type)
            )
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
            return f"{drug_name} 안전 확인 결과: " + "; ".join(summaries)
        return ""

    @staticmethod
    def _friendly_dur_endpoint_label(label: str) -> str:
        cleaned = re.sub(r"\s*\(T\d+\)", "", label or "")
        replacements = {
            "병용 금기 정보조회": "함께 먹으면 안 되는 조합",
            "병용금기정보조회": "함께 먹으면 안 되는 조합",
            "노인주의 정보조회": "고령자 복용 주의",
            "DUR 품목정보 조회": "약 기본 정보",
            "특정연령대 금기 정보조회": "연령별 복용 금기",
            "용량주의 정보조회": "복용량 주의",
            "투여기간주의 정보조회": "복용 기간 주의",
            "효능군중복 정보조회": "비슷한 효과 약 중복 주의",
            "서방정분할주의 정보조회": "쪼개 먹으면 안 되는 약 주의",
            "임부금기 정보조회": "임신 중 복용 금기",
        }
        return replacements.get(cleaned, cleaned or "주의 정보")

    def _select_dur_endpoint_keys(
        self,
        *,
        text: str,
        context: dict,
        medication_count: int,
    ) -> list[str]:
        profile = context.get("user_profile") if isinstance(context, dict) else {}
        age = self._profile_age(profile or {})
        return dur_api.select_dur_endpoint_keys(
            query_text=text,
            patient_age=age,
            medication_count=medication_count,
            default_to_basic=True,
        )

    @staticmethod
    def _profile_age(profile: dict[str, Any]) -> Optional[int]:
        raw = str(profile.get("age") or "").strip()
        if not raw:
            return None
        match = re.search(r"\d{1,3}", raw)
        return int(match.group(0)) if match else None

    def _contains_medication_signal(self, text_lower: str) -> bool:
        if self._is_nonmedical_smalltalk_request(text_lower):
            return False
        if any(kw in text_lower for kw in MEDICATION_KEYWORDS if kw != "약"):
            return True
        # Prevent false positives like "요약" while keeping short utterances such as
        # "약 맞나요?" as medication queries.
        return "약" in text_lower and "요약" not in text_lower

    def _is_profile_or_lifestyle_context(self, text_lower: str) -> bool:
        if not any(kw in text_lower for kw in PROFILE_OR_MEMORY_KEYWORDS):
            return False
        return not any(kw in text_lower for kw in DIRECT_MEDICAL_QUERY_KEYWORDS)

    @staticmethod
    def _is_profile_identity_recall(text: str, profile: dict[str, Any]) -> bool:
        if not profile.get("name"):
            return False
        compact = re.sub(r"\s+", "", text)
        return any(token in text for token in ("내가 누구", "누구인지", "내 이름", "내 프로필")) or any(
            token in compact
            for token in ("내가누구", "누군지", "누구인지", "내이름", "내프로필", "나누구")
        )

    def _is_memory_recall_query(self, text_lower: str) -> bool:
        return any(kw in text_lower for kw in MEMORY_RECALL_KEYWORDS)

    @staticmethod
    def _is_meal_medication_guidance_request(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if ReasoningEngine._is_after_meal_completion_signal(text):
            return True
        return (
            any(token in text for token in ("밥", "식후", "식사"))
            and "약" in text
            and any(token in compact for token in ("무슨약", "어떤약", "뭐먹", "먹어야", "먹고난", "먹고나", "먹고왔", "먹었"))
        )

    @staticmethod
    def _is_current_medication_record_recall(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        return "기록" in text and any(token in compact for token in ("남아있", "있지않", "먹고있", "복용중"))

    @staticmethod
    def _is_after_meal_completion_signal(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "").lower()
        if not compact:
            return False
        if any(token in compact for token in ("약먹", "약복용", "복용했")):
            return False
        meal_signal = any(
            token in compact
            for token in (
                "밥",
                "식사",
                "아침",
                "점심",
                "저녁",
                "식후",
            )
        )
        done_signal = any(
            token in compact
            for token in (
                "먹었",
                "먹고왔",
                "먹고옴",
                "다먹",
                "먹음",
                "식사했",
                "식사끝",
                "식사마쳤",
                "먹고나",
            )
        )
        return meal_signal and done_signal

    @staticmethod
    def _is_generic_blood_pressure_medication_overview_request(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "").lower()
        if not any(token in text for token in ("혈압약", "고혈압약")):
            return False
        if any(token in compact for token in ("dur기준", "같이먹", "함께먹", "두번", "2번", "더빨리", "부작용", "위험", "먹어도")):
            return False
        return any(
            token in compact
            for token in (
                "어떤거",
                "어떤것",
                "무엇",
                "뭐가",
                "뭐있",
                "종류",
                "목록",
                "확인해",
                "알려줘",
            )
        )

    @staticmethod
    def _context_has_medication(context: dict[str, Any]) -> bool:
        haystack = "\n".join(
            str(context.get(key) or "")
            for key in ("prescription_log", "context_memory", "current_manual", "memory_prompt")
        )
        if any(token in haystack for token in ("혈압약", "고혈압약", "당뇨약", "인슐린", "와파린", "아스피린")):
            return True
        return bool(re.search(r"-\s*[가-힣A-Za-z0-9]+(?:장용정|정|캡슐|시럽)\b", haystack))

    def _is_nonmedical_smalltalk_request(self, text_lower: str) -> bool:
        return any(
            phrase in text_lower
            for phrase in (
                "약 얘기 말고",
                "약 이야기 말고",
                "긴 설명 말고",
                "짧게 응원",
                "안부만",
                "그냥 인사",
            )
        )

    def _is_medication_record_request(self, text_lower: str) -> bool:
        if self._is_after_meal_completion_signal(text_lower):
            return False
        if any(token in text_lower for token in ("먹었어", "먹었어요", "먹었습니다", "복용했어", "복용했어요")):
            return "약" in text_lower or len(text_lower.strip()) <= 8
        return (
            "기록" in text_lower
            and any(token in text_lower for token in ("복용", "먹었다", "먹었다고", "드셨"))
            and any(token in text_lower for token in ("약", "정", "캡슐", "시럽"))
        )

    def _is_missing_ocr_image_request(self, text_lower: str) -> bool:
        return is_ocr_capture_request_text(text_lower)

