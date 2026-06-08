"""사용자(speaker) 식별 레지스트리 — 파일 기반 리스트.

신규/기존 핸드폰 앱 사용자와 웹 사용자를 영구 키(speaker_id)로 구분하고,
플랫폼/IP/최초·최근 접속/신규 여부를 정리한다. SQL 미사용 정책에 맞춰
``device_registry.json``과 동일한 경량 JSON 파일 레지스트리 패턴을 따른다.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class IdentityRecord:
    """식별된 사용자 1건."""

    speaker_id: str
    platform: str = "unknown"  # android | web | unknown
    first_seen: str = field(default_factory=_now_iso)
    last_seen: str = field(default_factory=_now_iso)
    first_ip: str = ""
    last_ip: str = ""
    app_version: str = ""
    display_name: str = ""
    seen_count: int = 0
    source: str = ""  # ws | device_register | auto_issued | touch_api

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityRecord":
        return cls(
            speaker_id=str(data.get("speaker_id", "")),
            platform=str(data.get("platform", "unknown")),
            first_seen=str(data.get("first_seen") or _now_iso()),
            last_seen=str(data.get("last_seen") or _now_iso()),
            first_ip=str(data.get("first_ip", "")),
            last_ip=str(data.get("last_ip", "")),
            app_version=str(data.get("app_version", "")),
            display_name=str(data.get("display_name", "")),
            seen_count=int(data.get("seen_count", 0) or 0),
            source=str(data.get("source", "")),
        )


class IdentityRegistry:
    """speaker_id를 PK로 하는 JSON 파일 기반 사용자 레지스트리."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or settings.identity_registry_path
        self._lock = threading.Lock()
        self._records: dict[str, IdentityRecord] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            self._records = {}
            return
        records: dict[str, IdentityRecord] = {}
        for item in raw.get("identities", []):
            record = IdentityRecord.from_dict(item)
            if record.speaker_id:
                records[record.speaker_id] = record
        self._records = records

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        payload = {"identities": [record.to_dict() for record in self._records.values()]}
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def touch(
        self,
        speaker_id: str,
        *,
        platform: str = "unknown",
        ip: str = "",
        app_version: str = "",
        source: str = "",
        display_name: str = "",
    ) -> tuple[IdentityRecord, bool]:
        """사용자 접속을 기록/갱신한다. (record, is_new)를 반환."""
        if not speaker_id:
            return IdentityRecord(speaker_id=""), False
        now = _now_iso()
        with self._lock:
            existing = self._records.get(speaker_id)
            is_new = existing is None
            if existing is None:
                record = IdentityRecord(
                    speaker_id=speaker_id,
                    platform=platform or "unknown",
                    first_seen=now,
                    last_seen=now,
                    first_ip=ip,
                    last_ip=ip,
                    app_version=app_version,
                    display_name=display_name,
                    seen_count=1,
                    source=source,
                )
            else:
                record = existing
                record.last_seen = now
                record.seen_count += 1
                if ip:
                    record.last_ip = ip
                    if not record.first_ip:
                        record.first_ip = ip
                # 더 구체적인 플랫폼 정보가 들어오면 갱신.
                if platform and platform != "unknown":
                    record.platform = platform
                if app_version:
                    record.app_version = app_version
                if display_name:
                    record.display_name = display_name
            self._records[speaker_id] = record
            self._flush()
        logger.info(
            "[IdentityRegistry] touch speaker_id=%s platform=%s is_new=%s source=%s seen=%d ip=%s",
            speaker_id,
            record.platform,
            is_new,
            source,
            record.seen_count,
            ip or "-",
        )
        return record, is_new

    def get(self, speaker_id: str) -> Optional[IdentityRecord]:
        with self._lock:
            return self._records.get(speaker_id)

    def list_all(self) -> list[IdentityRecord]:
        with self._lock:
            return sorted(
                self._records.values(),
                key=lambda r: r.last_seen,
                reverse=True,
            )


_registry = IdentityRegistry()


def touch_identity(
    speaker_id: str,
    *,
    platform: str = "unknown",
    ip: str = "",
    app_version: str = "",
    source: str = "",
    display_name: str = "",
) -> tuple[IdentityRecord, bool]:
    """모듈 레벨 헬퍼: 사용자 접속을 기록/갱신한다."""
    return _registry.touch(
        speaker_id,
        platform=platform,
        ip=ip,
        app_version=app_version,
        source=source,
        display_name=display_name,
    )


def get_identity(speaker_id: str) -> Optional[IdentityRecord]:
    return _registry.get(speaker_id)


def list_identities() -> list[IdentityRecord]:
    return _registry.list_all()
