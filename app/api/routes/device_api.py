"""모바일 디바이스 등록/조회 API."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.services.device_api import DeviceRecord, list_devices_for_speaker, register_device
from app.services.identity_registry import touch_identity

router = APIRouter(prefix="/api/devices", tags=["devices"])


def _client_ip(request: Request) -> str:
    xff = str(request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return xff
    return request.client.host if request.client else ""


class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(..., description="고유 디바이스 ID")
    speaker_id: str = Field(..., description="복약 대상 speaker ID")
    platform: Literal["android"] = Field(..., description="디바이스 플랫폼")
    push_token: str = Field(..., description="FCM push token")
    app_version: str = Field("", description="앱 버전")


class DeviceRegisterResponse(BaseModel):
    success: bool = True
    device_id: str
    speaker_id: str
    platform: str
    is_new: bool = False


class DeviceListResponse(BaseModel):
    speaker_id: str
    devices: list[dict[str, str]]


def _to_public_payload(record: DeviceRecord) -> dict[str, str]:
    return {
        "device_id": record.device_id,
        "speaker_id": record.speaker_id,
        "platform": record.platform,
        "app_version": record.app_version,
    }


@router.post("/register", response_model=DeviceRegisterResponse)
async def register_device_endpoint(
    payload: DeviceRegisterRequest, request: Request
) -> DeviceRegisterResponse:
    record = register_device(
        device_id=payload.device_id,
        speaker_id=payload.speaker_id,
        platform=payload.platform,
        push_token=payload.push_token,
        app_version=payload.app_version,
    )
    _, is_new = touch_identity(
        payload.speaker_id,
        platform=payload.platform,
        ip=_client_ip(request),
        app_version=payload.app_version,
        source="device_register",
    )
    return DeviceRegisterResponse(
        success=True,
        device_id=record.device_id,
        speaker_id=record.speaker_id,
        platform=record.platform,
        is_new=is_new,
    )


@router.get("/speaker/{speaker_id}", response_model=DeviceListResponse)
async def list_speaker_devices_endpoint(speaker_id: str) -> DeviceListResponse:
    records = list_devices_for_speaker(speaker_id)
    return DeviceListResponse(
        speaker_id=speaker_id,
        devices=[_to_public_payload(record) for record in records],
    )
