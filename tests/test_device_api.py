"""Device registration and delivery behavior tests."""
from __future__ import annotations

from app.services import device_api


def test_register_and_list_devices(tmp_path, monkeypatch):
    registry = device_api.DeviceRegistry(str(tmp_path / "devices.json"))
    monkeypatch.setattr(device_api, "_registry", registry)

    record = device_api.register_device(
        device_id="android-demo-1",
        speaker_id="speaker-a",
        platform="android",
        push_token="token-abc",
        app_version="0.1.0",
    )
    assert record.device_id == "android-demo-1"

    listed = device_api.list_devices_for_speaker("speaker-a")
    assert len(listed) == 1
    assert listed[0].device_id == "android-demo-1"
    assert listed[0].push_token == "token-abc"

