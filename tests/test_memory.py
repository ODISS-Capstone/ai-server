"""Architecture-backed memory tests."""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory.service import StructuredMemoryService


def run(coro):
    return asyncio.run(coro)


def test_structured_memory_selects_relevant_architecture_backed_context(tmp_path):
    store = MDStore(base_path=str(tmp_path))
    service = StructuredMemoryService(base_path=str(tmp_path))

    run(store.initialize())
    run(
        store.save_user_file(
            "speaker-001",
            "profile.md",
            (
                "# 환자 프로필\n"
                "| 항목 | 값 |\n|------|----|\n"
                "| ID | speaker-001 |\n"
                "| 이름 | 홍길동 |\n"
                "| 성별 | 남 |\n"
                "| 연령 | 75 |\n"
                "| 기저질환 | 고혈압 |\n"
            ),
        )
    )
    run(
        store.write_flash(
            "current_manual",
            "# 응답 매뉴얼\n- 혈압약 상담 시 현재 복용 약과 주의사항을 먼저 설명한다.\n",
        )
    )

    context = run(service.build_context("홍길동 고혈압", speaker_id="speaker-001"))

    assert context["relevant_memories"]
    assert any("홍길동" in item["body"] for item in context["relevant_memories"])
    assert "Patient Memories" in context["memory_prompt"]


def test_structured_memory_does_not_surface_irrelevant_memories_for_empty_query(tmp_path):
    store = MDStore(base_path=str(tmp_path))
    service = StructuredMemoryService(base_path=str(tmp_path))

    run(store.initialize())
    run(store.write_flash("current_requirement", "# 최근 요구사항\n- 천천히 설명해 주세요.\n"))

    context = run(service.build_context("", speaker_id="patient-kor"))

    assert context["relevant_memories"] == []
    assert "Relevant memories" not in context["memory_prompt"]


def test_structured_memory_reads_korean_profile_file(tmp_path):
    store = MDStore(base_path=str(tmp_path))
    service = StructuredMemoryService(base_path=str(tmp_path))

    run(store.initialize())
    run(
        store.save_user_file(
            "patient-kor",
            "profile.md",
            (
                "# 환자 프로필\n"
                "| 항목 | 값 |\n|------|----|\n"
                "| ID | patient-kor |\n"
                "| 이름 | 홍길동 |\n"
                "| 성별 | 남 |\n"
                "| 연령 | 75 |\n"
                "| 기저질환 | 고혈압 |\n"
            ),
        )
    )

    context = run(service.build_context("홍길동 어르신 고혈압", speaker_id="patient-kor"))

    assert any("홍길동" in item["body"] for item in context["relevant_memories"])
    assert "홍길동" in context["memory_prompt"]


def test_memory_index_is_truncated_when_permanent_history_is_too_large(tmp_path):
    store = MDStore(base_path=str(tmp_path))
    service = StructuredMemoryService(base_path=str(tmp_path))

    run(store.initialize())
    long_query = "혈압약" + (" 아주길게설명" * 40)
    for idx in range(120):
        run(
            store.save(
                "medication_log",
                f"# 상담 기록\n## 질문\n{long_query} {idx}\n## 응답\n응답 {idx}\n",
            )
        )

    context = run(service.build_context("혈압약"))

    assert "WARNING: MEMORY.md was truncated" in context["memory_prompt"]
    assert "Permanent Memories" in context["memory_prompt"]


def test_memory_policy_matches_architecture_layout(tmp_path):
    service = StructuredMemoryService(base_path=str(tmp_path))

    context = run(service.build_context(""))

    assert "flash/" in context["memory_prompt"]
    assert "permanent/" in context["memory_prompt"]
    assert "server.mermaid" in context["memory_prompt"]
    assert "별도 저장소" not in context["memory_prompt"]


def test_memory_engine_search_history_returns_architecture_backed_memory(tmp_path):
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path))

    run(engine.initialize())
    run(
        engine.store.save_user_file(
            "speaker-001",
            "profile.md",
            (
                "# 환자 프로필\n"
                "| 항목 | 값 |\n|------|----|\n"
                "| ID | speaker-001 |\n"
                "| 이름 | 홍길동 |\n"
                "| 성별 | 남 |\n"
                "| 연령 | 75 |\n"
                "| 기저질환 | 고혈압 |\n"
            ),
        )
    )
    run(
        engine.sync_ocr_dur(
            {"medications": [{"name": "혈압약A"}]},
            [{"name": "혈압약A", "contraindications": ["병용 금기"], "precautions": ["복용 주의"]}],
            speaker_id="speaker-001",
        )
    )

    history = run(engine.search_history("혈압약A", speaker_id="speaker-001"))

    assert "structured_memory" in history
    assert history["structured_memory"]["items"]
    assert any("혈압약A" in brief for brief in history["structured_memory"]["briefs"])


def test_query_memory_context_uses_flash_and_permanent_layers(tmp_path):
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path))

    run(engine.initialize())
    run(engine.store.write_flash("current_manual", "# 현재 매뉴얼\n- 혈압약은 최신 복용 목록부터 확인.\n"))
    run(
        engine.store.save(
            "medication_log",
            "# 상담 기록\n## 질문\n혈압약 같이 먹어도 되나요\n## 응답\n기록된 상담 응답\n",
        )
    )

    prompt = run(engine.build_query_memory_context("혈압약", speaker_id=None))

    assert "Flash Memories" in prompt
    assert "Permanent Memories" in prompt
    assert "혈압약" in prompt


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
