"""Helpers for reasoning about memory freshness."""
from __future__ import annotations

from datetime import datetime, timezone


def memory_age_days(mtime_ms: float) -> int:
    delta_ms = datetime.now(tz=timezone.utc).timestamp() * 1000 - mtime_ms
    if delta_ms <= 0:
        return 0
    return int(delta_ms // 86_400_000)


def memory_age_text(mtime_ms: float) -> str:
    days = memory_age_days(mtime_ms)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def memory_freshness_text(mtime_ms: float) -> str:
    days = memory_age_days(mtime_ms)
    if days <= 1:
        return ""
    return (
        f"This memory is {days} days old. Treat it as a point-in-time note, "
        "not live medical state. Verify with the latest OCR, DUR result, or "
        "patient confirmation before relying on it."
    )
