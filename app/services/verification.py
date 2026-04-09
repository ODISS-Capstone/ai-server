"""팩트 대조 및 답변 수정: DUR·의약품 DB와 비교해 답변 검증."""
from typing import Optional

from app.schemas.dur import DurResponse


def verify_answer_against_dur(answer: str, dur_response: DurResponse) -> tuple[str, bool]:
    """
    LLM 답변과 DUR 결과를 대조. DUR에 명시된 금기/상호작용이 답변에 누락되면 보완 문구 추가.
    반환: (수정된 답변, 변경 여부)
    """
    modified = answer
    changed = False
    for item in dur_response.items:
        for warning in item.contraindications + item.interactions + item.precautions:
            if warning and warning.strip() not in modified:
                # 중요 주의사항이 답변에 없으면 끝에 추가
                if not modified.rstrip().endswith("."):
                    modified += "."
                modified += f" 참고로, {item.name} 관련하여: {warning[:100]} 등은 확인이 필요합니다."
                changed = True
    if not modified.rstrip().endswith("정확한 판단은 의사·약사 상담이 필요합니다."):
        if not modified.rstrip().endswith("."):
            modified += "."
        modified += " 정확한 판단은 의사·약사 상담이 필요합니다."
        changed = True
    return modified, changed
