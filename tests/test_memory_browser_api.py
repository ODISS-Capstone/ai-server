"""Tests for read-only patient memory browser API."""
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.main import app
from app.services.memory_browser import MemoryBrowserService


@pytest.fixture()
def memory_db(tmp_path, monkeypatch):
    db_path = tmp_path / "md_database"
    monkeypatch.setattr(settings, "md_database_path", str(db_path))
    monkeypatch.setattr(settings, "structured_memory_path", str(db_path / "structured_memory"))
    monkeypatch.setattr(settings, "memory_browser_token", "test-browser-token")
    monkeypatch.setattr(settings, "app_env", "development")

    from app.api.routes import memory_browser_api

    store = MDStore(base_path=str(db_path))
    engine = MemoryEngine()
    engine.store = store
    engine.structured_memory.base_path = db_path / "structured_memory"
    browser = MemoryBrowserService(memory_engine=engine)
    monkeypatch.setattr(memory_browser_api, "memory_browser", browser)
    return store, engine


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-browser-token"}


def _write_patient(store: MDStore, speaker_id: str, profile: dict) -> None:
    engine = MemoryEngine()
    engine.store = store
    content = engine._format_profile_markdown(
        speaker_id,
        profile,
        registered_at="2026-05-19T12:00:00",
    )
    user_dir = store.permanent / "patients" / speaker_id
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "profile.md").write_text(content, encoding="utf-8")
    (user_dir / "history.md").write_text(
        f"# history\n\n- {profile.get('name', 'unknown')}님과 복약 상담을 진행함.\n",
        encoding="utf-8",
    )


def test_search_patients_by_partial_name(memory_db):
    store, _ = memory_db
    _write_patient(
        store,
        "patient-kim",
        {"name": "김영수", "age": "72", "gender": "남성", "conditions": ["고혈압"]},
    )
    _write_patient(
        store,
        "patient-lee",
        {"name": "이재석", "age": "45", "gender": "남성", "conditions": []},
    )

    day_dir = store.permanent / "medication_log" / "2026-05-19"
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "001_120000.md").write_text(
        "# 상담 기록\n\n김영수님 녹용 복용 상담.\n",
        encoding="utf-8",
    )

    client = TestClient(app)
    response = client.get("/api/memory/patients?name=김영", headers=_auth_headers())
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["patients"][0]["speaker_id"] == "patient-kim"
    assert data["patients"][0]["name"] == "김영수"


def test_get_patient_records_filters_by_name(memory_db):
    store, _ = memory_db
    _write_patient(
        store,
        "patient-kim",
        {"name": "김영수", "age": "72", "gender": "남성", "conditions": ["고혈압"]},
    )

    ocr_dir = store.permanent / "ocr_history" / "2026-05-19"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "001_120000.md").write_text(
        "# OCR\n\n김영수님 처방전 OCR 결과: 혈압약.\n",
        encoding="utf-8",
    )
    unrelated_dir = store.permanent / "ocr_history" / "2026-05-18"
    unrelated_dir.mkdir(parents=True, exist_ok=True)
    (unrelated_dir / "001_120000.md").write_text(
        "# OCR\n\n다른 환자 처방전 OCR 결과.\n",
        encoding="utf-8",
    )

    client = TestClient(app)
    response = client.get(
        "/api/memory/patients/patient-kim/records",
        params={"categories": "ocr_history", "start": date(2026, 5, 19), "end": date(2026, 5, 19)},
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    records = response.json()["records"]
    assert len(records) == 1
    assert "김영수" in records[0]["preview"]


def test_read_entry_rejects_path_traversal(memory_db):
    store, _ = memory_db
    _write_patient(
        store,
        "patient-kim",
        {"name": "김영수", "age": "72", "gender": "남성", "conditions": []},
    )

    outside = Path(store.base).parent / "outside.md"
    outside.write_text("secret", encoding="utf-8")

    client = TestClient(app)
    response = client.get(
        "/api/memory/entry",
        params={"path": "../outside.md"},
        headers=_auth_headers(),
    )
    assert response.status_code == 400


def test_memory_browser_allows_unauthenticated_in_development_without_token(memory_db, monkeypatch):
    store, _ = memory_db
    _write_patient(
        store,
        "patient-kim",
        {"name": "김영수", "age": "72", "gender": "남성", "conditions": []},
    )
    monkeypatch.setattr(settings, "memory_browser_token", "")
    monkeypatch.setattr(settings, "app_env", "development")

    client = TestClient(app)
    response = client.get("/api/memory/patients?name=김영수")
    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_memory_browser_requires_token_when_configured(memory_db, monkeypatch):
    store, _ = memory_db
    _write_patient(
        store,
        "patient-kim",
        {"name": "김영수", "age": "72", "gender": "남성", "conditions": []},
    )
    monkeypatch.setattr(settings, "memory_browser_token", "required-token")

    client = TestClient(app)
    unauthorized = client.get("/api/memory/patients?name=김영수")
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/api/memory/patients?name=김영수",
        headers={"Authorization": "Bearer required-token"},
    )
    assert authorized.status_code == 200
