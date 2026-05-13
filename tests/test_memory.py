"""structured_memory 동작 검증 테스트."""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory.service import StructuredMemoryService
from app.schemas.engine_contracts import MemoryEvidenceRequest
from app.core.config import settings


def run(coro):
    return asyncio.run(coro)


def test_upsert_memory_creates_topic_file_and_entrypoint(tmp_path):
    memory_dir = tmp_path / "structured_memory"
    service = StructuredMemoryService(base_path=str(memory_dir))

    run(
        service.upsert_memory(
            filename="blood_pressure.md",
            name="혈압약 복약 주의",
            description="혈압약A 복약 시 어지러움 확인",
            memory_type="project",
            body="# 혈압약 복약 주의\n\n- 혈압약A 복용 후 어지러움 여부를 확인한다.",
        )
    )

    topic = memory_dir / "global" / "blood_pressure.md"
    entrypoint = memory_dir / "global" / "MEMORY.md"

    assert topic.exists()
    assert "type: project" in topic.read_text(encoding="utf-8")
    assert "- [혈압약 복약 주의](blood_pressure.md) - 혈압약A 복약 시 어지러움 확인" in entrypoint.read_text(
        encoding="utf-8"
    )


def test_build_context_loads_entrypoint_and_relevant_topic(tmp_path):
    service = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(
        service.upsert_memory(
            filename="medication_context.md",
            name="최신 복약 맥락",
            description="혈압약A와 당뇨약B를 함께 복용 중",
            memory_type="project",
            body="# 최신 복약 맥락\n\n- 혈압약A와 당뇨약B를 함께 복용 중이다.",
            speaker_id="speaker-001",
        )
    )

    context = run(service.build_context("혈압약A 복용", speaker_id="speaker-001"))

    assert "Speaker speaker-001 MEMORY.md" in context["memory_prompt"]
    assert "Relevant memories" in context["memory_prompt"]
    assert context["relevant_memories"]
    assert any("혈압약A" in item["body"] for item in context["relevant_memories"])


def test_empty_query_does_not_surface_relevant_memories(tmp_path):
    service = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(
        service.upsert_memory(
            filename="manual.md",
            name="응답 매뉴얼",
            description="복약 안내 시 최신 처방을 먼저 확인",
            memory_type="reference",
            body="# 응답 매뉴얼\n\n- 최신 처방을 먼저 확인한다.",
        )
    )

    context = run(service.build_context("", speaker_id="speaker-001"))

    assert context["relevant_memories"] == []
    assert "Relevant memories" not in context["memory_prompt"]
    assert "Global MEMORY.md" in context["memory_prompt"]


def test_patient_profile_sync_writes_speaker_memory(tmp_path):
    service = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(
        service.sync_patient_profile(
            "patient-kor",
            {
                "name": "홍길동",
                "age": 75,
                "gender": "남",
                "conditions": ["고혈압"],
            },
        )
    )

    context = run(service.build_context("홍길동 고혈압", speaker_id="patient-kor"))

    assert any("홍길동" in item["body"] for item in context["relevant_memories"])
    assert "patient_profile.md" in context["memory_index"]


def test_memory_index_is_truncated_when_entrypoint_is_too_large(tmp_path):
    service = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    for idx in range(220):
        run(
            service.upsert_memory(
                filename=f"topic_{idx}.md",
                name=f"테스트 메모리 {idx}",
                description=f"긴 인덱스 테스트 {idx}",
                memory_type="project",
                body=f"# 테스트 메모리 {idx}\n\n- 인덱스 절단 테스트",
            )
        )

    context = run(service.build_context(""))

    assert "WARNING: MEMORY.md was truncated" in context["memory_prompt"]


def test_memory_engine_syncs_ocr_dur_into_structured_memory(tmp_path):
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(engine.initialize())
    run(
        engine.sync_ocr_dur(
            {"medications": [{"name": "혈압약A"}]},
            [{"name": "혈압약A", "contraindications": ["병용 금기"], "precautions": ["복용 주의"]}],
            speaker_id="speaker-001",
        )
    )

    topic = tmp_path / "structured_memory" / "speakers" / "speaker-001" / "current_medication.md"
    history = run(engine.search_history("혈압약A", speaker_id="speaker-001"))

    assert topic.exists()
    assert "structured_memory" in history
    assert history["structured_memory"]["items"]
    assert any("혈압약A" in brief for brief in history["structured_memory"]["briefs"])


def test_memory_engine_syncs_ocr_dur_endpoint_rows_without_losing_names(tmp_path):
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(engine.initialize())
    run(
        engine.sync_ocr_dur(
            {"medications": [{"name": "DrugA"}]},
            [
                {
                    "medication": "DrugA",
                    "dur": {
                        "dur_product_info": {"success": True, "items": [{"name": "DrugA"}]},
                        "combination_contraindication": {"success": True, "items": [{"pair": "DrugB"}]},
                        "dosage_caution": {"success": True, "items": [{"dose": "high"}]},
                    },
                }
            ],
            speaker_id="speaker-row",
        )
    )

    latest = run(engine.store.read_latest("prescriptions", 1))[0]["content"]
    structured = (
        tmp_path
        / "structured_memory"
        / "speakers"
        / "speaker-row"
        / "current_medication.md"
    ).read_text(encoding="utf-8")

    assert "DrugA" in latest
    assert "이름 없음" not in latest
    assert "DrugA" in structured
    assert "정보 1건" in structured
    assert "금기 1건" in structured
    assert "주의 1건" in structured


def test_memory_engine_syncs_legacy_dur_interactions_as_precautions(tmp_path):
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(engine.initialize())
    run(
        engine.sync_ocr_dur(
            {"medications": [{"name": "DrugA"}]},
            [
                {
                    "name": "DrugA",
                    "contraindications": ["금기"],
                    "interactions": ["상호작용"],
                    "precautions": ["주의"],
                }
            ],
            speaker_id="speaker-legacy-row",
        )
    )

    latest = run(engine.store.read_latest("prescriptions", 1))[0]["content"]

    assert "DrugA" in latest
    assert "금기 1건" in latest
    assert "주의 2건" in latest


def test_memory_evidence_extracts_common_medication_names_without_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_go_kr_service_key", "test-key")
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(engine.initialize())
    bundle = run(
        engine.prepare_evidence_bundle(
            MemoryEvidenceRequest(
                query="와파린이랑 아스피린 같이 먹으면 출혈 위험이 있는지 알려줘",
                speaker_id=None,
                ocr_payload=None,
            )
        )
    )

    assert set(bundle.normalized_medications) == {"와파린", "아스피린"}
    assert bundle.dur_searchable is True


def test_memory_engine_syncs_flash_profile_into_structured_memory(tmp_path):
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))

    run(engine.initialize())
    run(
        engine.update_flash_profile(
            "speaker-002",
            {
                "name": "김영희",
                "age": 80,
                "gender": "여",
                "conditions": ["당뇨"],
            },
        )
    )

    context = run(engine.structured_memory.build_context("김영희 당뇨", speaker_id="speaker-002"))

    assert any("김영희" in item["body"] for item in context["relevant_memories"])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
