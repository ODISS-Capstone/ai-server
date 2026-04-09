"""LLM as a Judge Engine — 프론티어 LLM 검증 및 성능 증강.

server.mermaid 매핑:
  LLM_as_a_Judge → verify_fact(), evaluate_response()
  LLM_Search     → (app/tools/llm_search.py 에서 처리)
"""
import logging
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """당신은 의약품 안전 정보를 검증하는 전문 판사(Judge) AI입니다.

역할:
1. 제공된 의약 정보가 사실에 부합하는지 검증합니다.
2. 잘못된 정보나 위험한 조언이 포함되어 있으면 지적합니다.
3. 누락된 중요 안전 정보가 있으면 보충합니다.

출력 형식:
- "VERIFIED": 정보가 정확하고 안전합니다
- "NEEDS_CORRECTION: [수정 사항]": 수정이 필요합니다
- "DANGER: [위험 사항]": 위험한 정보가 포함되어 있습니다

항상 환자 안전을 최우선으로 판단합니다."""


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

        user_content = (
            f"다음 의약 정보를 검증해 주세요.\n\n"
            f"[원본 질문]\n{original_query}\n\n"
            f"[검증 대상 정보]\n{statement}\n"
        )
        if additional_context:
            user_content += f"\n[추가 맥락]\n{additional_context}\n"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "max_tokens": 512,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

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

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "다음 기준으로 응답을 1~10점으로 평가하세요.\n"
                                    f"{criteria_text}\n\n"
                                    "형식: SCORE: N/10\nFEEDBACK: ..."
                                ),
                            },
                            {"role": "user", "content": response_text},
                        ],
                        "max_tokens": 256,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

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
        response_upper = response.upper().strip()

        if response_upper.startswith("DANGER"):
            correction = response.split(":", 1)[-1].strip() if ":" in response else response
            return {
                "verified": False,
                "needs_correction": True,
                "danger": True,
                "corrected": correction,
                "message": response,
            }

        if response_upper.startswith("NEEDS_CORRECTION"):
            correction = response.split(":", 1)[-1].strip() if ":" in response else response
            return {
                "verified": False,
                "needs_correction": True,
                "danger": False,
                "corrected": correction,
                "message": response,
            }

        return {
            "verified": True,
            "needs_correction": False,
            "danger": False,
            "message": response,
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
