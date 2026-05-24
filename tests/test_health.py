"""Health and app smoke tests."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("service") == "ODISS — 만성질환 복약관리 AI 서버엔진"
    assert data.get("docs") == "/docs"
    assert data.get("health") == "/health"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "odiss-medication-guidance"


def test_frontier_llm_health():
    r = client.get("/health/frontier-llm")
    assert r.status_code == 200
    data = r.json()
    assert "enabled_providers" in data
    assert "providers" in data
    assert "openai" in data["providers"]
    assert "together" in data["providers"]


def test_llm_health_includes_conversation_switch():
    r = client.get("/health/llm")
    assert r.status_code == 200
    data = r.json()
    assert "conversation" in data
    assert data["conversation"]["backend"] in {"local", "together", "auto"}
