"""Claude Code 방식 메모리 분류와 프롬프트 규칙."""
from __future__ import annotations

from pathlib import Path

from app.memory.models import MemoryType

MEMORY_TYPES: tuple[MemoryType, ...] = ("user", "feedback", "project", "reference")
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000


def parse_memory_type(raw: object) -> MemoryType | None:
    if not isinstance(raw, str):
        return None
    return raw if raw in MEMORY_TYPES else None


def build_memory_policy(memory_dir: Path) -> str:
    """structured_memory 사용 규칙을 메모리 프롬프트에 넣는다."""
    return "\n".join(
        [
            f"# ODISS structured memory ({memory_dir})",
            "",
            "Claude Code 방식의 파일 기반 장기 메모리입니다.",
            "",
            "## 저장 구조",
            "- global/: 전역 메모리. 공통 매뉴얼, 공통 참고 정보, 서버 전체에 적용되는 요구사항을 저장합니다.",
            "- speakers/{speaker_id}/: 화자별 메모리. 환자 프로필, 복약 맥락, 개인별 DUR 참고사항을 저장합니다.",
            "- 각 디렉터리는 MEMORY.md 인덱스와 topic .md 파일을 함께 가집니다.",
            "",
            "## topic 파일 형식",
            "- 각 topic 파일은 frontmatter에 name, description, type을 가집니다.",
            "- type은 user, feedback, project, reference 중 하나만 사용합니다.",
            "- MEMORY.md에는 본문을 직접 저장하지 않고 topic 파일 링크와 한 줄 설명만 둡니다.",
            "",
            "## 조회 규칙",
            "- MEMORY.md는 항상 먼저 읽는 인덱스입니다.",
            "- 질의와 명확히 관련 있는 topic 파일만 relevant memory로 본문을 불러옵니다.",
            "- 오래된 메모리는 당시의 스냅샷으로 보고 최신 OCR, DUR, 환자 확인으로 재검증합니다.",
        ]
    )
