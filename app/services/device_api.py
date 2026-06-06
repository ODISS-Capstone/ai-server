"""단말기 등록/조회 및 전송용 서버 API.

모바일 앱(Android assistant)이 FCM push token을 등록하고, 서버가 speaker별
디바이스를 조회하여 최종 답변/복약 알림을 전달할 수 있도록 한다.

저장소: JSON 파일 (``data/storage/device_registry.json``) — SQL 미사용 정책에 맞춰
가벼운 파일 기반 레지스트리를 사용한다.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "./data/storage/device_registry.json"


@dataclass
class DeviceRecord:
    """등록된 단말기 1건."""

    device_id: str
    speaker_id: str
    platform: str
    push_token: str
    app_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceRecord":
        return cls(
            device_id=str(data.get("device_id", "")),
            speaker_id=str(data.get("speaker_id", "")),
            platform=str(data.get("platform", "android")),
            push_token=str(data.get("push_token", "")),
            app_version=str(data.get("app_version", "")),
        )


class DeviceRegistry:
    """device_id를 PK로 하는 JSON 파일 기반 단말기 레지스트리."""

    def __init__(self, path: str = DEFAULT_REGISTRY_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records: dict[str, DeviceRecord] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            self._records = {}
            return
        records: dict[str, DeviceRecord] = {}
        for item in raw.get("devices", []):
            record = DeviceRecord.from_dict(item)
            if record.device_id:
                records[record.device_id] = record
        self._records = records

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        payload = {"devices": [record.to_dict() for record in self._records.values()]}
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def upsert(
        self,
        *,
        device_id: str,
        speaker_id: str,
        platform: str,
        push_token: str,
        app_version: str = "",
    ) -> DeviceRecord:
        record = DeviceRecord(
            device_id=device_id,
            speaker_id=speaker_id,
            platform=platform,
            push_token=push_token,
            app_version=app_version,
        )
        with self._lock:
            self._records[device_id] = record
            self._flush()
        logger.info(
            "[DeviceRegistry] upsert device_id=%s speaker_id=%s platform=%s",
            device_id,
            speaker_id,
            platform,
        )
        return record

    def list_for_speaker(self, speaker_id: str) -> list[DeviceRecord]:
        with self._lock:
            return [
                record
                for record in self._records.values()
                if record.speaker_id == speaker_id
            ]

    def all_records(self) -> list[DeviceRecord]:
        with self._lock:
            return list(self._records.values())


_registry = DeviceRegistry()


def register_device(
    *,
    device_id: str,
    speaker_id: str,
    platform: str,
    push_token: str,
    app_version: str = "",
) -> DeviceRecord:
    """단말기를 레지스트리에 등록(또는 갱신)한다."""
    return _registry.upsert(
        device_id=device_id,
        speaker_id=speaker_id,
        platform=platform,
        push_token=push_token,
        app_version=app_version,
    )


def list_devices_for_speaker(speaker_id: str) -> list[DeviceRecord]:
    """speaker_id에 등록된 단말기 목록을 반환한다."""
    return _registry.list_for_speaker(speaker_id)


async def send_to_device(
    device_id: str,
    text: str,
    tts_requested: bool = True,
    meta: Optional[dict[str, Any]] = None,
) -> bool:
    """최종 답변 텍스트를 지정 단말(스마트 스피커 등)로 전송.

    실제 구현 시 WebSocket 또는 디바이스별 Push API 호출.
    """
    _ = device_id, text, tts_requested, meta
    return True


async def send_to_speaker_devices(
    speaker_id: str,
    text: str,
    tts_requested: bool = True,
    meta: Optional[dict[str, Any]] = None,
) -> int:
    """speaker에 등록된 모든 단말기로 메시지를 전송하고 전송 건수를 반환한다.

    WebSocket 세션이 없을 때의 push fallback 경로에서 사용한다.
    """
    devices = list_devices_for_speaker(speaker_id)
    sent = 0
    for device in devices:
        ok = await send_to_device(
            device.device_id,
            text,
            tts_requested=tts_requested,
            meta=meta,
        )
        if ok:
            sent += 1
    return sent
