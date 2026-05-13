"""약학정보원(KPIC) DUR API: OCR로 추출한 약품명 검증, 금기·상호작용 조회."""
from typing import Optional

import httpx

from app.core.config import settings
from app.schemas.dur import DurItem, DurResponse
from app.schemas.ocr import MedicationItem


async def check_dur(medications: list[MedicationItem]) -> DurResponse:
    """
    OCR 결과 약품 목록을 KPIC DUR API에 보내 검증 및 금기·상호작용 정보를 반환.
    API 미설정 시 목록 그대로 verified=True 로 반환(목업).
    """
    api_url = settings.kpic_dur_api_url
    api_key = settings.kpic_dur_api_key

    if not api_url or not api_key:
        items = [
            DurItem(
                name=m.name,
                ingredient=m.name,
                efficacy="(DUR API 미설정)",
                contraindications=[],
                interactions=[],
                precautions=["실제 서비스에서는 약학정보원 DUR API를 연동해 주세요."],
                verified=True,
            )
            for m in medications
        ]
        return DurResponse(items=items, message="(DUR API 미설정, 목업 반환)")

    # KPIC 공공 API 형식에 맞게 요청 (실제 스펙은 약학정보원 문서 참고)
    drug_names = [m.name for m in medications]
    try:
        async with httpx.AsyncClient(timeout=settings.kpic_dur_api_timeout_seconds) as client:
            resp = await client.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}" or "ServiceKey {api_key}", "Content-Type": "application/json"},
                json={"itemSeqList": drug_names} if isinstance(drug_names, list) else {"drugName": drug_names},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001 - keep OCR pipeline responsive on DUR outage
        items = [
            DurItem(
                name=m.name,
                ingredient=m.name,
                efficacy=None,
                contraindications=[],
                interactions=[],
                precautions=[f"DUR API 응답 지연 또는 오류로 기본 정보만 저장했습니다: {exc}"],
                verified=False,
            )
            for m in medications
        ]
        return DurResponse(items=items, success=False, message="DUR API timeout or error")

    # 응답 구조에 따라 DurItem 리스트 구성 (실제 API 스펙에 맞게 수정 필요)
    items: list[DurItem] = []
    if isinstance(data, list):
        for row in data:
            items.append(
                DurItem(
                    name=row.get("itemName") or row.get("name", ""),
                    ingredient=row.get("ingredient"),
                    efficacy=row.get("efficacy"),
                    contraindications=row.get("contraindications") or [],
                    interactions=row.get("interactions") or [],
                    precautions=row.get("precautions") or [],
                    verified=row.get("verified", True),
                )
            )
    else:
        body = data if isinstance(data, dict) else {}
        for name in drug_names:
            items.append(
                DurItem(
                    name=name,
                    ingredient=body.get("ingredient"),
                    efficacy=body.get("efficacy"),
                    contraindications=body.get("contraindications") or [],
                    interactions=body.get("interactions") or [],
                    precautions=body.get("precautions") or [],
                    verified=body.get("verified", True),
                )
            )
    return DurResponse(items=items)
