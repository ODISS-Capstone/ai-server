"""의약품 낱알식별 API 연동 (T1) — HIRA/data.go.kr."""
import logging
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def identify_medicine(
    item_name: Optional[str] = None,
    entp_name: Optional[str] = None,
    item_seq: Optional[str] = None,
    img_regist_ts: Optional[str] = None,
    print_front: Optional[str] = None,
    print_back: Optional[str] = None,
    drug_shape: Optional[str] = None,
    color_class1: Optional[str] = None,
    page_no: int = 1,
    num_of_rows: int = 10,
) -> dict[str, Any]:
    """의약품 낱알식별정보 조회 (T1)."""
    service_key = settings.data_go_kr_service_key
    if not service_key:
        return {
            "success": False,
            "message": "data_go_kr_service_key 미설정",
            "items": [],
        }

    url = f"{settings.hira_api_base_url}/getMdcinGrnIdntfcInfoList03"

    params: dict[str, Any] = {
        "serviceKey": service_key,
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        "type": "json",
    }
    if item_name:
        params["item_name"] = item_name
    if entp_name:
        params["entp_name"] = entp_name
    if item_seq:
        params["item_seq"] = item_seq
    if img_regist_ts:
        params["img_regist_ts"] = img_regist_ts
    if print_front:
        params["print_front"] = print_front
    if print_back:
        params["print_back"] = print_back
    if drug_shape:
        params["drug_shape"] = drug_shape
    if color_class1:
        params["color_class1"] = color_class1

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
        }
    except httpx.HTTPStatusError as e:
        logger.error("HIRA API HTTP error: %s", e.response.status_code)
        return {"success": False, "message": str(e), "items": []}
    except Exception as e:
        logger.error("HIRA API error: %s", e)
        return {"success": False, "message": str(e), "items": []}
