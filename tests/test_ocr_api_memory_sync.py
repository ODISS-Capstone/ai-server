"""OCR REST endpoint memory synchronization tests."""
from __future__ import annotations

import asyncio

from app.api.routes import ocr_api
from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory import StructuredMemoryService


def run(coro):
    return asyncio.run(coro)


def test_ocr_analyze_syncs_full_dur_rows_into_memory(tmp_path, monkeypatch):
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    async def fake_check_dur_for_prescription(medications):
        return [
            {
                "medication": medications[0]["name"],
                "dur": {
                    "dur_product_info": {"success": True, "items": [{"name": medications[0]["name"]}]},
                    "dosage_caution": {"success": True, "items": [{"dose": "high"}]},
                },
            }
        ]

    monkeypatch.setattr(ocr_api, "memory_engine", engine)
    monkeypatch.setattr(ocr_api, "check_dur_for_prescription", fake_check_dur_for_prescription)

    response = run(
        ocr_api.receive_ocr_result(
            ocr_api.OCRResultInput(
                raw_text="DrugA 5mg",
                medications=[ocr_api.MedicationItemInput(name="DrugA", dosage="1정")],
                confidence=0.91,
                speaker_id="speaker-ocr-rest",
            )
        )
    )

    latest = run(engine.store.read_latest("prescriptions", 1))[0]["content"]
    structured = (
        tmp_path
        / "structured_memory"
        / "speakers"
        / "speaker-ocr-rest"
        / "current_medication.md"
    ).read_text(encoding="utf-8")

    assert response.medication_count == 1
    assert response.dur_results[0]["medication"] == "DrugA"
    assert "DrugA" in latest
    assert "이름 없음" not in latest
    assert "DrugA" in structured
    assert "주의 1건" in structured
