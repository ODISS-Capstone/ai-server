"""식약처 DUR API 연동 (T2~T10).

T2: 병용 금기 정보조회
T3: 노인주의 정보 조회
T4: DUR 품목정보 조회
T5: 특정연령대 금기 정보조회
T6: 용량주의 정보 조회
T7: 투여기간주의 정보조회
T8: 효능군중복 정보 조회
T9: 서방정분할주의 정보조회
T10: 임부금기 정보조회
"""
import logging
import asyncio
from typing import Any, Optional
from urllib.parse import quote

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

DUR_ENDPOINTS = {
    "combination_contraindication": {
        "service": "DURPrdlstInfoService03",
        "operation": "getUsjntTabooInfoList03",
        "description": "병용 금기 정보조회 (T2)",
    },
    "elderly_caution": {
        "service": "DURPrdlstInfoService03",
        "operation": "getOdsnAtentInfoList03",
        "description": "노인주의 정보조회 (T3)",
    },
    "dur_product_info": {
        "service": "DURPrdlstInfoService03",
        "operation": "getDurPrdlstInfoList03",
        "description": "DUR 품목정보 조회 (T4)",
    },
    "age_contraindication": {
        "service": "DURPrdlstInfoService03",
        "operation": "getSpcifyAgrdeTabooInfoList03",
        "description": "특정연령대 금기 정보조회 (T5)",
    },
    "dosage_caution": {
        "service": "DURPrdlstInfoService03",
        "operation": "getCpctyAtentInfoList03",
        "description": "용량주의 정보조회 (T6)",
    },
    "period_caution": {
        "service": "DURPrdlstInfoService03",
        "operation": "getMdctnPdAtentInfoList03",
        "description": "투여기간주의 정보조회 (T7)",
    },
    "efficacy_overlap": {
        "service": "DURPrdlstInfoService03",
        "operation": "getEfcyDplctInfoList03",
        "description": "효능군중복 정보조회 (T8)",
    },
    "sr_tablet_caution": {
        "service": "DURPrdlstInfoService03",
        "operation": "getSeobangjeongDivisionInfoList03",
        "description": "서방정분할주의 정보조회 (T9)",
    },
    "pregnancy_contraindication": {
        "service": "DURPrdlstInfoService03",
        "operation": "getPwnmTabooInfoList03",
        "description": "임부금기 정보조회 (T10)",
    },
}

ALL_DUR_ENDPOINT_KEYS = tuple(DUR_ENDPOINTS.keys())
BASIC_DUR_ENDPOINT_KEYS = ("dur_product_info",)


async def call_dur_api(
    endpoint_key: str,
    item_name: Optional[str] = None,
    item_seq: Optional[str] = None,
    page_no: int = 1,
    num_of_rows: int = 10,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """식약처 DUR API 단일 호출."""
    service_key = settings.data_go_kr_service_key
    if not service_key:
        return {
            "success": False,
            "message": "data_go_kr_service_key 미설정",
            "items": [],
        }

    endpoint = DUR_ENDPOINTS.get(endpoint_key)
    if not endpoint:
        return {
            "success": False,
            "message": f"Unknown endpoint: {endpoint_key}",
            "items": [],
        }

    base_url = settings.dur_api_base_url
    url = f"{base_url}/{endpoint['service']}/{endpoint['operation']}"

    params: dict[str, Any] = {
        "serviceKey": service_key,
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        "type": "json",
    }
    if item_name:
        params["itemName"] = item_name
    if item_seq:
        params["itemSeq"] = item_seq

    try:
        if client is None:
            async with httpx.AsyncClient(timeout=settings.dur_api_timeout_seconds) as owned_client:
                resp = await owned_client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        else:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        body = data.get("body", {})
        items = body.get("items", [])
        if isinstance(items, dict):
            items = items.get("item", [])
        if isinstance(items, dict):
            items = [items]

        return {
            "success": True,
            "total_count": body.get("totalCount", len(items)),
            "items": items,
            "endpoint": endpoint["description"],
        }
    except httpx.HTTPStatusError as e:
        logger.error("DUR API HTTP error: %s %s", e.response.status_code, endpoint_key)
        return {"success": False, "message": str(e), "items": []}
    except Exception as e:
        logger.error("DUR API error: %s", e)
        return {"success": False, "message": str(e), "items": []}


async def check_all_dur(
    item_name: str,
    item_seq: Optional[str] = None,
    endpoint_keys: Optional[list[str] | tuple[str, ...]] = None,
) -> dict[str, Any]:
    """조회 대상 DUR 항목을 병렬로 확인한다.

    ``endpoint_keys``를 생략하면 하위 호환을 위해 T2~T10 전체를 조회한다.
    런타임 대화 경로에서는 질문 의도에 맞는 키만 넘겨 불필요한 외부 API
    왕복을 줄인다.
    """
    selected_keys = _normalize_endpoint_keys(endpoint_keys or ALL_DUR_ENDPOINT_KEYS)
    limit = max(1, int(settings.dur_api_max_concurrency))
    semaphore = asyncio.Semaphore(limit)

    async with httpx.AsyncClient(timeout=settings.dur_api_timeout_seconds) as client:
        async def call_one(key: str) -> tuple[str, dict[str, Any]]:
            async with semaphore:
                result = await call_dur_api(
                    key,
                    item_name=item_name,
                    item_seq=item_seq,
                    client=client,
                )
                return key, result

        pairs = await asyncio.gather(*(call_one(key) for key in selected_keys))
    results = {key: result for key, result in pairs}
    return results


async def check_dur_for_prescription(
    medications: list[dict],
    endpoint_keys: Optional[list[str] | tuple[str, ...]] = BASIC_DUR_ENDPOINT_KEYS,
) -> list[dict[str, Any]]:
    """처방전 내 약품 목록에 대한 DUR 조회.

    OCR 저장 경로의 기본값은 T4 품목 기본 정보만 확인한다. 병용/용량/기간
    같은 세부 DUR은 사용자의 실제 질문 의도에 맞춰 나중에 조회한다.
    """
    names = [str(med.get("name", "")).strip() for med in medications if med.get("name")]
    if not names:
        return []

    selected_keys = _normalize_endpoint_keys(endpoint_keys or BASIC_DUR_ENDPOINT_KEYS)
    limit = max(1, int(settings.dur_api_max_concurrency))
    semaphore = asyncio.Semaphore(limit)

    async with httpx.AsyncClient(timeout=settings.dur_api_timeout_seconds) as client:
        async def check_medication(name: str) -> dict[str, Any]:
            async def call_one(key: str) -> tuple[str, dict[str, Any]]:
                async with semaphore:
                    result = await call_dur_api(
                        key,
                        item_name=name,
                        client=client,
                    )
                    return key, result

            pairs = await asyncio.gather(*(call_one(key) for key in selected_keys))
            return {"medication": name, "dur": {key: result for key, result in pairs}}

        return await asyncio.gather(*(check_medication(name) for name in names))


def select_dur_endpoint_keys(
    *,
    query_text: str = "",
    patient_age: Optional[int] = None,
    medication_count: int = 1,
    default_to_basic: bool = True,
) -> list[str]:
    """Pick the smallest useful DUR endpoint set for a user question."""
    lowered = (query_text or "").lower()
    selected: list[str] = []

    def add(key: str) -> None:
        if key in DUR_ENDPOINTS and key not in selected:
            selected.append(key)

    if default_to_basic:
        add("dur_product_info")

    if any(token in lowered for token in ("모든 dur", "전체 dur", "dur 전체", "전부 확인", "다 확인")):
        return list(ALL_DUR_ENDPOINT_KEYS)

    if medication_count > 1 or any(token in lowered for token in ("같이", "함께", "병용", "동시에", "상호작용", "섞어")):
        add("combination_contraindication")
    if patient_age is not None and patient_age >= 65:
        add("elderly_caution")
    if any(token in lowered for token in ("노인", "고령", "65세", "65 살")):
        add("elderly_caution")
    if any(token in lowered for token in ("어린이", "아이", "소아", "영유아", "청소년", "나이", "연령")):
        add("age_contraindication")
    if any(token in lowered for token in ("두 번", "2번", "많이", "더 빨리", "용량", "복용량", "몇 알", "몇 정", "과다", "초과")):
        add("dosage_caution")
    if any(token in lowered for token in ("기간", "며칠", "몇 일", "오래", "장기", "계속")):
        add("period_caution")
    if any(token in lowered for token in ("중복", "같은 효과", "같은 효능", "같은 성분")):
        add("efficacy_overlap")
    if any(token in lowered for token in ("쪼개", "나눠", "분할", "부숴", "가루", "씹")):
        add("sr_tablet_caution")
    if any(token in lowered for token in ("임신", "임부", "수유", "임산부")):
        add("pregnancy_contraindication")

    return selected or list(BASIC_DUR_ENDPOINT_KEYS)


def _normalize_endpoint_keys(endpoint_keys: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for key in endpoint_keys:
        if key in DUR_ENDPOINTS and key not in normalized:
            normalized.append(key)
    return normalized or list(BASIC_DUR_ENDPOINT_KEYS)
