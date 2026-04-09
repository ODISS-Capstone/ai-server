"""4단계: LLM용 문서 생성 — OCR·DUR 결과를 LLM이 참고할 문서 텍스트로 변환."""
from app.schemas.dur import DurItem, DurResponse
from app.schemas.ocr import MedicationItem, OcrResponse


def build_llm_doc(ocr_response: OcrResponse, dur_response: DurResponse) -> str:
    """
    OCR 추출 결과와 DUR 검증 결과를 합쳐 LLM용 한 문서 문자열로 만든다.
    """
    lines: list[str] = ["[현재 복용 중인 약]", ""]
    for m in ocr_response.medications:
        parts = [m.name]
        if m.strength:
            parts.append(m.strength)
        if m.dosage:
            parts.append(m.dosage)
        if m.frequency:
            parts.append(m.frequency)
        if m.timing:
            parts.append(m.timing)
        lines.append(" - " + ", ".join(parts))
    lines.append("")
    lines.append("[DUR 검증 및 주의사항]")
    for d in dur_response.items:
        lines.append(f" - {d.name}")
        if d.efficacy:
            lines.append(f"   효능: {d.efficacy}")
        for c in d.contraindications:
            lines.append(f"   금기: {c}")
        for i in d.interactions:
            lines.append(f"   병용 주의: {i}")
        for p in d.precautions:
            lines.append(f"   주의: {p}")
        lines.append("")
    return "\n".join(lines).strip()
