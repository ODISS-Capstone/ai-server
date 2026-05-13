"""Utilities for summarizing DUR result payloads."""
from __future__ import annotations

from typing import Any

INFO_ENDPOINTS = {"dur_product_info"}
CONTRAINDICATION_ENDPOINTS = {
    "combination_contraindication",
    "age_contraindication",
    "pregnancy_contraindication",
}


def summarize_dur_result(row: dict[str, Any]) -> dict[str, Any]:
    """Return medication name and warning counts from either legacy or endpoint DUR rows."""
    name = str(
        row.get("name")
        or row.get("medication")
        or row.get("drug_name")
        or row.get("item_name")
        or "이름 없음"
    ).strip()

    legacy_contra = row.get("contraindications")
    legacy_interactions = row.get("interactions")
    legacy_precautions = row.get("precautions")
    if legacy_contra is not None or legacy_interactions is not None or legacy_precautions is not None:
        return {
            "name": name or "이름 없음",
            "info": 0,
            "contraindications": _count_items(legacy_contra),
            "precautions": _count_items(legacy_interactions) + _count_items(legacy_precautions),
        }

    info = 0
    contraindications = 0
    precautions = 0
    dur_payload = row.get("dur") if isinstance(row.get("dur"), dict) else row
    for endpoint_key, payload in dur_payload.items():
        if endpoint_key in {"name", "medication", "drug_name", "item_name"}:
            continue
        count = _count_items(payload)
        if not count:
            continue
        if endpoint_key in INFO_ENDPOINTS:
            info += count
        elif endpoint_key in CONTRAINDICATION_ENDPOINTS or "contraindication" in endpoint_key:
            contraindications += count
        else:
            precautions += count

    return {
        "name": name or "이름 없음",
        "info": info,
        "contraindications": contraindications,
        "precautions": precautions,
    }


def _count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, tuple):
        return len(value)
    if isinstance(value, dict):
        items = value.get("items")
        if isinstance(items, list):
            return len(items)
        if "count" in value:
            try:
                return int(value["count"])
            except (TypeError, ValueError):
                return 0
        return 0
    return 1 if value else 0
