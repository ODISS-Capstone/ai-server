"""LLM as a Judge Engine — 프론티어 LLM 검증 및 성능 증강.

server.mermaid 매핑:
  LLM_as_a_Judge → verify_fact(), evaluate_response()
  LLM_Search     → (app/tools/llm_search.py 에서 처리)
"""
import logging
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.services.llm_queue import run_with_engine_queue
from app.services.prompt_registry import DEFAULT_PROMPTS, get_prompt_registry

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = DEFAULT_PROMPTS["judge_verify"]["system"]


class LLMJudgeEngine:
    """LLM as a Judge: 팩트 체킹 및 판단 검토."""

    def __init__(self):
        self.model = settings.openai_model

    async def verify_fact(
        self,
        statement: str,
        original_query: str,
        additional_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """중요 추론 항목의 팩트 체킹."""
        api_key = settings.openai_api_key
        if not api_key:
            logger.warning("OpenAI API key not set — skipping LLM Judge verification")
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

        try:
            async def post_judge_verify() -> dict[str, Any]:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": 512,
                            "temperature": 0.1,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()

            data = await run_with_engine_queue("judge", post_judge_verify)

            answer = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_judge_response(answer, statement)

        except httpx.HTTPStatusError as e:
            logger.error("LLM Judge API error: %s", e.response.status_code)
            return {
                "verified": True,
                "needs_correction": False,
                "message": f"LLM Judge API 오류: {e.response.status_code}",
            }
        except Exception as e:
            logger.error("LLM Judge error: %s", e)
            return {
                "verified": True,
                "needs_correction": False,
                "message": f"LLM Judge 오류: {e}",
            }

    async def evaluate_response(
        self,
        response_text: str,
        criteria: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """응답 품질 평가 (성능 증강)."""
        api_key = settings.openai_api_key
        if not api_key:
            return {"score": 0.0, "feedback": "LLM Judge 미설정"}

        default_criteria = [
            "정확성: 의학적으로 올바른 정보인가",
            "안전성: 위험한 조언이 없는가",
            "이해도: 어르신이 이해하기 쉬운 표현인가",
            "완전성: 중요 정보가 누락되지 않았는가",
        ]
        criteria_text = "\n".join(f"- {c}" for c in (criteria or default_criteria))
        messages = get_prompt_registry().render_messages(
            "judge_evaluate",
            criteria_text=criteria_text,
            response_text=response_text,
        )

        try:
            async def post_judge_evaluate() -> dict[str, Any]:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": 256,
                            "temperature": 0.1,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()

            data = await run_with_engine_queue("judge", post_judge_evaluate)

            answer = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_evaluation(answer)

        except Exception as e:
            logger.error("LLM Judge evaluation error: %s", e)
            return {"score": 0.0, "feedback": str(e)}

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
