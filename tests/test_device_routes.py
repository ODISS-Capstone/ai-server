"""Device registration endpoint tests."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_register_device_route():
    response = client.post(
        "/api/devices/register",
        json={
            "device_id": "android-r1",
            "speaker_id": "speaker-r1",
            "platform": "android",
            "push_token": "push-token-r1",
            "app_version": "1.0.0",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["device_id"] == "android-r1"
    assert payload["speaker_id"] == "speaker-r1"


def test_list_speaker_devices_route():
    response = client.get("/api/devices/speaker/speaker-r1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["speaker_id"] == "speaker-r1"
    assert any(item["device_id"] == "android-r1" for item in payload["devices"])

