"""단말기 등록/조회 및 전송용 서버 API.

모바일 앱(Android assistant)이 FCM push token을 등록하고, 서버가 speaker별
디바이스를 조회하여 최종 답변/복약 알림을 전달할 수 있도록 한다.

저장소: JSON 파일 (``data/storage/device_registry.json``) — SQL 미사용 정책에 맞춰
가벼운 파일 기반 레지스트리를 사용한다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Optional

from app.core.config import settings

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

    def get(self, device_id: str) -> Optional[DeviceRecord]:
        with self._lock:
            return self._records.get(device_id)

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


# ── FCM (Firebase Cloud Messaging) ──

_fcm_lock = threading.Lock()
_fcm_app: Any = None
_fcm_unavailable_logged = False


def _get_fcm_messaging() -> Any:
    """firebase-admin messaging 모듈을 lazy 초기화. 미가용 시 None."""
    global _fcm_app, _fcm_unavailable_logged
    if not settings.fcm_enabled:
        return None
    with _fcm_lock:
        if _fcm_app is not None:
            try:
                from firebase_admin import messaging  # type: ignore

                return messaging
            except Exception:  # noqa: BLE001
                return None
        try:
            import firebase_admin  # type: ignore
            from firebase_admin import credentials, messaging  # type: ignore
        except ImportError:
            if not _fcm_unavailable_logged:
                logger.warning("[FCM] firebase-admin 미설치 — push 전송을 건너뜁니다.")
                _fcm_unavailable_logged = True
            return None

        cred_path = settings.fcm_credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        try:
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
            else:
                # ADC(Application Default Credentials) 시도.
                cred = credentials.ApplicationDefault()
            options = {"projectId": settings.fcm_project_id} if settings.fcm_project_id else None
            _fcm_app = firebase_admin.initialize_app(cred, options)
            logger.info("[FCM] initialized project_id=%s", settings.fcm_project_id or "(default)")
            return messaging
        except Exception as exc:  # noqa: BLE001 - 자격증명 없거나 잘못된 경우 안전하게 비활성
            if not _fcm_unavailable_logged:
                logger.warning("[FCM] 초기화 실패 — push 전송을 건너뜁니다. error=%s", exc)
                _fcm_unavailable_logged = True
            return None


def _send_fcm_message_sync(
    messaging: Any,
    *,
    push_token: str,
    data: dict[str, str],
    notification_title: Optional[str] = None,
    notification_body: Optional[str] = None,
) -> bool:
    try:
        notification = None
        if notification_title or notification_body:
            notification = messaging.Notification(
                title=notification_title or "",
                body=notification_body or "",
            )
        message = messaging.Message(
            token=push_token,
            data={str(k): str(v) for k, v in data.items()},
            notification=notification,
            android=messaging.AndroidConfig(priority="high"),
        )
        message_id = messaging.send(message)
        logger.info("[FCM] sent message_id=%s token=%s…", message_id, push_token[:12])
        return True
    except Exception as exc:  # noqa: BLE001 - 단건 실패가 다른 단말 전송을 막지 않도록
        logger.warning("[FCM] send_failed token=%s… error=%s", push_token[:12], exc)
        return False


async def _send_fcm(
    push_token: str,
    *,
    data: dict[str, Any],
    notification_title: Optional[str] = None,
    notification_body: Optional[str] = None,
) -> bool:
    messaging = _get_fcm_messaging()
    if messaging is None or not push_token:
        return False
    return await asyncio.to_thread(
        _send_fcm_message_sync,
        messaging,
        push_token=push_token,
        data={str(k): str(v) for k, v in data.items()},
        notification_title=notification_title,
        notification_body=notification_body,
    )


async def send_to_device(
    device_id: str,
    text: str,
    tts_requested: bool = True,
    meta: Optional[dict[str, Any]] = None,
) -> bool:
    """최종 답변 텍스트를 지정 단말로 FCM push 전송.

    활성 WebSocket 세션이 없을 때의 fallback 경로. 자격증명이 없으면 안전하게 no-op.
    """
    record = _registry.get(device_id)
    if record is None or not record.push_token:
        return False
    data: dict[str, Any] = {
        "type": str((meta or {}).get("type") or "response"),
        "text": text,
        "tts": "1" if tts_requested else "0",
    }
    if meta:
        for key, value in meta.items():
            if key not in data and isinstance(value, (str, int, float, bool)):
                data[key] = value
    return await _send_fcm(
        record.push_token,
        data=data,
        notification_body=text if tts_requested else None,
    )


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


async def send_data_to_speaker_devices(
    speaker_id: str,
    data: dict[str, Any],
    *,
    notification_title: Optional[str] = None,
    notification_body: Optional[str] = None,
) -> int:
    """speaker에 등록된 단말기로 데이터 푸시(예: ocr_request)를 전송하고 건수를 반환.

    앱이 종료된 상태에서도 촬영모드를 실행할 수 있도록 high-priority 데이터 메시지를 보낸다.
    자격증명/단말이 없으면 0을 반환한다.
    """
    devices = list_devices_for_speaker(speaker_id)
    sent = 0
    for device in devices:
        if not device.push_token:
            continue
        ok = await _send_fcm(
            device.push_token,
            data=data,
            notification_title=notification_title,
            notification_body=notification_body,
        )
        if ok:
            sent += 1
    if not sent:
        logger.info(
            "[FCM] no_device_delivery speaker_id=%s devices=%d (앱 미등록 또는 FCM 비활성)",
            speaker_id,
            len(devices),
        )
    return sent
