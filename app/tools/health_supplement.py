"""건강기능식품 API 연동 (T11, T12) — data.go.kr."""
import logging
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def get_supplement_detail(
    product_name: Optional[str] = None,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """건강기능식품 상세정보 조회 (T11)."""
    service_key = settings.data_go_kr_service_key
    if not service_key:
        return {
            "success": False,
            "message": "data_go_kr_service_key 미설정",
            "items": [],
        }

    url = f"{settings.health_supplement_api_base_url}/getHtfsSttusIdntfcInfoList01"

    params: dict[str, Any] = {
        "serviceKey": service_key,
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        "type": "json",
    }
    if product_name:
        params["prdlstNm"] = product_name

    try:
        async with httpx.AsyncClient(timeout=settings.health_supplement_api_timeout_seconds) as client:
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
        }
    except httpx.HTTPStatusError as e:
        logger.error("Health Supplement Detail API error: %s", e.response.status_code)
        return {"success": False, "message": str(e), "items": []}
    except Exception as e:
        logger.error("Health Supplement Detail API error: %s", e)
        return {"success": False, "message": str(e), "items": []}


async def list_supplements(
    product_name: Optional[str] = None,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """건강기능식품 목록 조회 및 제품명 검색 (T12)."""
    service_key = settings.data_go_kr_service_key
    if not service_key:
        return {
            "success": False,
            "message": "data_go_kr_service_key 미설정",
            "items": [],
        }

    url = f"{settings.health_supplement_api_base_url}/getHtfsSttusIdntfcInfoList01"

    params: dict[str, Any] = {
        "serviceKey": service_key,
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        "type": "json",
    }
    if product_name:
        params["prdlstNm"] = product_name

    try:
        async with httpx.AsyncClient(timeout=settings.health_supplement_api_timeout_seconds) as client:
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
        }
    except httpx.HTTPStatusError as e:
        logger.error("Health Supplement List API error: %s", e.response.status_code)
        return {"success": False, "message": str(e), "items": []}
    except Exception as e:
        logger.error("Health Supplement List API error: %s", e)
        return {"success": False, "message": str(e), "items": []}
