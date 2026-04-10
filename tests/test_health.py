"""Health and app smoke tests."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("service") == "ODISS — 시니어 복약지도 AI 서버엔진"
    assert data.get("docs") == "/docs"
    assert data.get("health") == "/health"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
