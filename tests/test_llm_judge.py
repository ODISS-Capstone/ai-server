"""LLM judge response parsing tests."""

from app.engines.llm_judge import LLMJudgeEngine


def test_parse_judge_response_accepts_verified_marker():
    parsed = LLMJudgeEngine()._parse_judge_response("VERIFIED", "원문")

    assert parsed["verified"] is True
    assert parsed["needs_correction"] is False


def test_parse_judge_response_treats_unrecognized_format_as_correction_needed():
    parsed = LLMJudgeEngine()._parse_judge_response("검토 결과: 안전합니다.", "원문")

    assert parsed["verified"] is False
    assert parsed["needs_correction"] is True
    assert parsed["corrected"] == "원문"

