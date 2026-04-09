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


async def call_dur_api(
    endpoint_key: str,
    item_name: Optional[str] = None,
    item_seq: Optional[str] = None,
    page_no: int = 1,
    num_of_rows: int = 10,
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
        async with httpx.AsyncClient(timeout=30.0) as client:
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
) -> dict[str, Any]:
    """모든 DUR 항목을 일괄 조회 (T2~T10)."""
    results: dict[str, Any] = {}
    for key in DUR_ENDPOINTS:
        result = await call_dur_api(
            key, item_name=item_name, item_seq=item_seq
        )
        results[key] = result
    return results


async def check_dur_for_prescription(
    medications: list[dict],
) -> list[dict[str, Any]]:
    """처방전 내 약품 목록에 대한 DUR 일괄 조회."""
    all_results: list[dict[str, Any]] = []
    for med in medications:
        name = med.get("name", "")
        if not name:
            continue
        dur_result = await check_all_dur(name)
        all_results.append({"medication": name, "dur": dur_result})
    return all_results
