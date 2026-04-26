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
            "- 처방을 대체하거나 진단을 내리지 않습니다.\n"
            "- 답변 끝에 \"정확한 판단은 의사·약사 상담이 필요합니다\"를 포함하세요.\n"
            "- 짧고 읽기 쉬운 문장으로, 고령 사용자도 이해하기 쉽게 작성하세요."
        ),
        "user": (
            "다음은 사용자 질문과 현재 복용 약물·주의사항 요약입니다.\n\n"
            "[복용 약물 및 주의사항]\n{llm_doc}\n\n"
            "[사용자 질문]\n{query_text}\n\n"
            "위 정보만 사용해 친절하고 안전하게 답변해 주세요."
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
    "judge_evaluate": {
        "system": (
            "다음 기준으로 응답을 1~10점으로 평가하세요.\n"
            "{criteria_text}\n\n"
            "형식: SCORE: N/10\nFEEDBACK: ..."
        ),
        "user": "{response_text}",
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
