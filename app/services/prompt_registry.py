"""JSON-backed prompt registry for LLM system/user message templates."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from string import Formatter
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS: dict[str, dict[str, str]] = {
    "main_answer": {
        "system": (
            "당신은 복약 상담을 보조하는 AI입니다.\n"
            "- 사용자의 질문에 대해 현재 복용 약물 정보와 DUR 주의사항을 바탕으로 답변합니다.\n"
            "- 사용 가능한 도구가 제공된 경우, 약물 안전성 확인이 필요할 때 관련 DUR/의약품 정보 도구를 사용해 근거를 보강합니다.\n"
            "- DUR 도구 호출 arguments는 반드시 제공된 tool schema의 snake_case 필드만 사용합니다. itemName, itemSeq, pageNo, numOfRows 같은 DUR API 원본 camelCase 키를 직접 출력하지 않습니다.\n"
            "- DUR 도구 호출 시 약 이름이 있으면 item_name을 우선 사용하고, 품목기준코드만 확인된 경우에만 item_seq를 사용합니다.\n"
            "- 비-DUR 도구 호출 시에도 tool schema에 정의된 필드만 사용합니다. 예: product_name, print_front, print_back, drug_shape, color_class1, page_no, num_of_rows.\n"
            "- 도구 결과는 참고용 안전 정보로만 사용하며, 이를 근거로 처방 변경, 복용 중단, 용량 조절, 질병 진단을 지시하지 않습니다.\n"
            "- 도구 결과가 없거나 불확실하면 추측하지 말고 의사·약사 확인이 필요하다고 안내합니다.\n"
            "- 처방을 대체하거나 진단을 내리지 않습니다.\n"
            "- 답변 끝에 \"정확한 판단은 의사·약사 상담이 필요합니다\"를 포함하세요.\n"
            "- 짧고 읽기 쉬운 문장으로, 고령 사용자도 이해하기 쉽게 작성하세요."
        ),
        "user": (
            "다음은 사용자 질문과 현재 복용 약물·주의사항 요약입니다.\n\n"
            "[복용 약물 및 주의사항]\n{llm_doc}\n\n"
            "[사용자 질문]\n{query_text}\n\n"
            "위 정보와 필요한 경우 도구 조회 결과를 참고해 친절하고 안전하게 답변해 주세요."
        ),
    },
    "external_review": {
        "system": (
            "당신은 복약 상담을 보조하는 AI입니다.\n"
            "- 사용자의 질문에 대해 현재 복용 약물 정보와 DUR 주의사항을 바탕으로 답변합니다.\n"
            "- 처방을 대체하거나 진단을 내리지 않습니다.\n"
            "- 답변 끝에 \"정확한 판단은 의사·약사 상담이 필요합니다\"를 포함하세요.\n"
            "- 짧고 읽기 쉬운 문장으로, 고령 사용자도 이해하기 쉽게 작성하세요."
        ),
        "user": "{censored_payload}",
    },
    "judge_verify": {
        "system": (
            "당신은 의약품 안전 정보를 검증하는 전문 판사(Judge) AI입니다.\n\n"
            "역할:\n"
            "1. 제공된 의약 정보가 사실에 부합하는지 검증합니다.\n"
            "2. 잘못된 정보나 위험한 조언이 포함되어 있으면 지적합니다.\n"
            "3. 누락된 중요 안전 정보가 있으면 보충합니다.\n\n"
            "출력 형식:\n"
            "- \"VERIFIED\": 정보가 정확하고 안전합니다\n"
            "- \"NEEDS_CORRECTION: [수정 사항]\": 수정이 필요합니다\n"
            "- \"DANGER: [위험 사항]\": 위험한 정보가 포함되어 있습니다\n\n"
            "항상 환자 안전을 최우선으로 판단합니다."
        ),
        "user": (
            "다음 의약 정보를 검증해 주세요.\n\n"
            "[원본 질문]\n{original_query}\n\n"
            "[검증 대상 정보]\n{statement}\n{additional_context_block}"
        ),
    },
    "judge_final_review": {
        "system": (
            "당신은 복약 안전 답변을 최종 검토하는 전문 Judge AI입니다.\n"
            "추론 엔진이 만든 핵심 팩트가 사용자에게 전달되기 전에 안전성, "
            "근거성, 표현 위험을 검토합니다.\n\n"
            "규칙:\n"
            "- 제공된 핵심 팩트와 추가 맥락 안에서만 검토합니다.\n"
            "- 처방 변경, 임의 중단, 임의 병용, 용량 조절을 지시하지 않습니다.\n"
            "- 위험하거나 단정적인 문장은 안전한 상담 권고 문장으로 수정합니다.\n"
            "- 불확실하면 확인이 필요하다고 말합니다.\n"
            "- <think>나 검토 과정은 출력하지 않습니다.\n"
            "- 출력은 로컬 발화 모델에 넘길 검토 완료 문장만 작성합니다."
        ),
        "user": (
            "[사용자 질문]\n{original_query}\n\n"
            "[추론 엔진 핵심 팩트]\n{core_message}\n"
            "{additional_context_block}\n\n"
            "위 내용을 최종 안전 검토한 뒤, 로컬 모델이 사용자 답변을 만들 수 있도록 "
            "검토 완료 문장만 작성해 주세요."
        ),
    },
    "local_delivery": {
        "system": (
            "당신은 ODISS 로컬 대화 모델입니다.\n"
            "GPT Judge가 검토한 복약 안전 문장을 어르신에게 말하듯 자연스럽고 "
            "짧은 한국어 답변으로 바꿉니다.\n\n"
            "규칙:\n"
            "- 검토 완료 문장에 없는 의학 정보를 새로 만들지 않습니다.\n"
            "- 약 이름, 위험 표현, 상담 권고는 유지합니다.\n"
            "- 2~4문장으로 짧게 말합니다.\n"
            "- 어려운 의학 용어는 쉬운 말로 바꿉니다.\n"
            "- 처방 변경, 임의 중단, 임의 병용을 지시하지 않습니다.\n"
            "- 마지막에는 필요한 경우 의사·약사 상담이 필요하다고 말합니다.\n"
            "- <think>나 내부 추론은 출력하지 않습니다."
        ),
        "user": (
            "[사용자 질문]\n{original_query}\n\n"
            "[사용자 프로필]\n{user_profile}\n\n"
            "[대화 맥락]\n{conversation_context}\n\n"
            "[GPT Judge 검토 완료 문장]\n{reviewed_message}\n\n"
            "위 검토 완료 문장을 어르신에게 들려줄 최종 답변으로 바꿔 주세요."
        ),
    },
    "judge_evaluate": {
        "system": (
            "다음 기준으로 응답을 1~10점으로 평가하세요.\n"
            "{criteria_text}\n\n"
            "형식: SCORE: N/10\nFEEDBACK: ..."
        ),
        "user": "{response_text}",
    },
    "identity_conflict_judge": {
        "system": (
            "당신은 ODISS의 신원 확인 판정기입니다.\n"
            "저장된 환자 프로필과 현재 발화가 같은 사람인지 판단합니다.\n\n"
            "출력은 반드시 첫 줄에 TRUE 또는 FALSE만 씁니다.\n"
            "- TRUE: 현재 발화자가 저장된 환자와 다른 이름, 나이, 성별, 관계자를 명시했거나 명백히 다른 사람입니다.\n"
            "- FALSE: 정보가 부족하거나, 단순 인사/약 질문/안부/보호자 도움 발화이거나, 저장 프로필과 충돌하지 않습니다.\n"
            "추론 과정은 출력하지 않습니다."
        ),
        "user": (
            "[현재 시각]\n{current_time}\n\n"
            "[저장된 환자 프로필]\n{patient_profile}\n\n"
            "[최근 대화 요약]\n{recent_history}\n\n"
            "[현재 발화]\n{current_text}\n\n"
            "현재 발화자가 저장된 환자와 엇갈리면 TRUE, 아니면 FALSE."
        ),
    },
    "identity_profile_extract": {
        "system": (
            "당신은 ODISS의 환자 신원정보 추출기입니다.\n"
            "현재 발화에서 환자 본인의 이름, 나이, 성별, 주요 기저질환만 추출합니다.\n"
            "추측하지 말고 명시된 정보만 사용합니다.\n"
            "출력은 반드시 JSON 한 개만 작성합니다.\n"
            "형식: {{\"name\":\"\", \"age\":\"\", \"gender\":\"\", \"conditions\":[]}}"
        ),
        "user": (
            "[현재 발화]\n{current_text}\n\n"
            "발화에 명시된 환자 신원정보만 JSON으로 추출하세요."
        ),
    },
    "search": {
        "system": (
            "당신은 의약품 및 건강 관련 정보를 검색하는 전문 에이전트입니다.\n"
            "사용자의 질문에 대해 정확하고 신뢰할 수 있는 의약 정보를 제공합니다.\n"
            "불확실한 정보는 '확인이 필요합니다'라고 명시합니다."
        ),
        "user": "{search_input}",
    },
}


class PromptRegistry:
    """Load prompt templates from JSON and render chat messages by key."""

    def __init__(
        self,
        path: str | Path | None = None,
        defaults: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.path = Path(path or settings.llm_prompts_path)
        self.defaults = defaults or DEFAULT_PROMPTS
        self.prompts = self._load_prompts()

    def render_messages(self, prompt_key: str, **variables: Any) -> list[dict[str, str]]:
        prompt = self._get_prompt(prompt_key)
        return [
            {"role": "system", "content": self._render(prompt["system"], variables)},
            {"role": "user", "content": self._render(prompt["user"], variables)},
        ]

    def render_system(self, prompt_key: str, **variables: Any) -> str:
        return self._render(self._get_prompt(prompt_key)["system"], variables)

    def render_user(self, prompt_key: str, **variables: Any) -> str:
        return self._render(self._get_prompt(prompt_key)["user"], variables)

    def _load_prompts(self) -> dict[str, dict[str, str]]:
        prompts = dict(self.defaults)
        if not self.path.exists():
            logger.info("Prompt registry file not found; using defaults: %s", self.path)
            return prompts

        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load prompt registry %s: %s", self.path, exc)
            return prompts

        configured = data.get("prompts", {})
        if not isinstance(configured, dict):
            logger.warning("Prompt registry %s has no object 'prompts' key", self.path)
            return prompts

        for key, prompt in configured.items():
            if self._is_valid_prompt(prompt):
                prompts[key] = {"system": prompt["system"], "user": prompt["user"]}
            else:
                logger.warning("Ignoring invalid prompt entry %s in %s", key, self.path)
        return prompts

    def _get_prompt(self, prompt_key: str) -> dict[str, str]:
        try:
            return self.prompts[prompt_key]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt key: {prompt_key}") from exc

    @staticmethod
    def _render(template: str, variables: dict[str, Any]) -> str:
        required = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name
        }
        missing = required - variables.keys()
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise KeyError(f"Missing prompt variables: {missing_list}")
        return template.format(**variables)

    @staticmethod
    def _is_valid_prompt(prompt: Any) -> bool:
        return (
            isinstance(prompt, dict)
            and isinstance(prompt.get("system"), str)
            and isinstance(prompt.get("user"), str)
        )


@lru_cache
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()
