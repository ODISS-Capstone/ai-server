"""LLM as a Judge Engine — 프론티어 LLM 검증 및 성능 증강.

server.mermaid 매핑:
  LLM_as_a_Judge → verify_fact(), evaluate_response()
  LLM_Search     → (app/tools/llm_search.py 에서 처리)
"""
import logging
from typing import Any, Optional

from app.services.frontier_llm import chat_completion, has_configured_frontier_provider
from app.services.prompt_registry import DEFAULT_PROMPTS, get_prompt_registry

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = DEFAULT_PROMPTS["judge_verify"]["system"]


class LLMJudgeEngine:
    """LLM as a Judge: 팩트 체킹 및 판단 검토."""

    async def verify_fact(
        self,
        statement: str,
        original_query: str,
        additional_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """중요 추론 항목의 팩트 체킹."""
        if not has_configured_frontier_provider():
            logger.warning("Frontier provider not configured — skipping LLM Judge verification")
            return {
                "verified": True,
                "needs_correction": False,
                "message": "LLM Judge 미설정 — 검증 생략",
            }

        additional_context_block = (
            f"\n[추가 맥락]\n{additional_context}\n"
            if additional_context
            else ""
        )
        messages = get_prompt_registry().render_messages(
            "judge_verify",
            original_query=original_query,
            statement=statement,
            additional_context_block=additional_context_block,
        )

        result = await chat_completion(
            task="judge",
            messages=messages,
            max_tokens=256,
            temperature=0.1,
        )
        if not result["success"]:
            return {
                "verified": True,
                "needs_correction": False,
                "message": result.get("message", "LLM Judge 오류"),
            }

        return self._parse_judge_response(result["content"], statement)

    async def review_final_answer(
        self,
        core_message: str,
        original_query: str,
        additional_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """GPT Judge로 사용자 전달 전 핵심 답변을 최종 안전 검토."""
        if not core_message or not core_message.strip():
            return {
                "reviewed": False,
                "reviewed_text": "",
                "message": "검토할 핵심 메시지가 없습니다.",
            }

        if not has_configured_frontier_provider():
            logger.warning("Frontier provider not configured — using core message as reviewed text")
            return {
                "reviewed": False,
                "reviewed_text": core_message,
                "message": "LLM Judge 미설정 — 최종 검토 생략",
            }

        additional_context_block = (
            f"\n[추가 맥락]\n{additional_context}\n"
            if additional_context
            else ""
        )
        messages = get_prompt_registry().render_messages(
            "judge_final_review",
            original_query=original_query,
            core_message=core_message,
            additional_context_block=additional_context_block,
        )

        result = await chat_completion(
            task="judge",
            messages=messages,
            max_tokens=400,
            temperature=0.1,
        )
        if not result["success"]:
            return {
                "reviewed": False,
                "reviewed_text": core_message,
                "message": result.get("message", "LLM Judge 오류"),
            }

        reviewed_text = (result.get("content") or "").strip()
        return {
            "reviewed": bool(reviewed_text),
            "reviewed_text": reviewed_text or core_message,
            "model": result.get("model"),
            "provider": result.get("provider"),
            "message": "최종 검토 완료" if reviewed_text else "빈 검토 결과 — 원문 사용",
        }

    async def evaluate_response(
        self,
        response_text: str,
        criteria: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """응답 품질 평가 (성능 증강)."""
        if not has_configured_frontier_provider():
            return {"score": 0.0, "feedback": "LLM Judge 미설정"}

        default_criteria = [
            "정확성: 의학적으로 올바른 정보인가",
            "안전성: 위험한 조언이 없는가",
            "이해도: 복약 정보가 익숙하지 않은 사용자도 이해하기 쉬운 표현인가",
            "완전성: 중요 정보가 누락되지 않았는가",
        ]
        criteria_text = "\n".join(f"- {c}" for c in (criteria or default_criteria))
        messages = get_prompt_registry().render_messages(
            "judge_evaluate",
            criteria_text=criteria_text,
            response_text=response_text,
        )

        result = await chat_completion(
            task="judge",
            messages=messages,
            max_tokens=160,
            temperature=0.1,
        )
        if not result["success"]:
            return {"score": 0.0, "feedback": result.get("message", "LLM Judge 오류")}

        return self._parse_evaluation(result["content"])

    def _parse_judge_response(
        self, response: str, original: str
    ) -> dict[str, Any]:
        stripped = response.strip()
        first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
        first_line_upper = first_line.upper()

        if first_line_upper.startswith("DANGER"):
            correction = response.split(":", 1)[-1].strip() if ":" in response else response
            return {
                "verified": False,
                "needs_correction": True,
                "danger": True,
                "corrected": correction,
                "message": response,
            }

        if first_line_upper.startswith("NEEDS_CORRECTION"):
            correction = response.split(":", 1)[-1].strip() if ":" in response else response
            return {
                "verified": False,
                "needs_correction": True,
                "danger": False,
                "corrected": correction,
                "message": response,
            }

        if first_line_upper.startswith("VERIFIED"):
            return {
                "verified": True,
                "needs_correction": False,
                "danger": False,
                "message": response,
            }

        return {
            "verified": False,
            "needs_correction": True,
            "danger": False,
            "corrected": original,
            "message": f"Unrecognized judge response format: {response}",
        }

    def _parse_evaluation(self, response: str) -> dict[str, Any]:
        score = 5.0
        feedback = response

        for line in response.split("\n"):
            if "SCORE" in line.upper():
                parts = line.split(":")
                if len(parts) > 1:
                    num_str = parts[-1].strip().split("/")[0].strip()
                    try:
                        score = float(num_str)
                    except ValueError:
                        pass
            if "FEEDBACK" in line.upper():
                feedback = line.split(":", 1)[-1].strip()

        return {"score": score, "feedback": feedback}
