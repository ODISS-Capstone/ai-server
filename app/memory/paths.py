"""Claude Code 방식 structured_memory 경로 관리."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

ENTRYPOINT_NAME = "MEMORY.md"


@dataclass(frozen=True, slots=True)
class MemoryDirectory:
    """MEMORY.md와 topic 파일을 함께 가지는 메모리 디렉터리."""

    path: Path
    scope: str
    label: str


class StructuredMemoryPaths:
    """전역 메모리와 화자별 메모리의 실제 저장 경로를 계산한다."""

    def __init__(self, base_path: str | None = None):
        self.base = Path(base_path or settings.structured_memory_path)
        self.global_dir = self.base / "global"
        self.speakers_dir = self.base / "speakers"

    def ensure_dirs(self, speaker_id: str | None = None) -> None:
        """structured_memory 기본 디렉터리와 필요한 MEMORY.md를 보장한다."""
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.speakers_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_entrypoint(self.global_dir)
        if speaker_id:
            speaker_dir = self.speaker_dir(speaker_id)
            speaker_dir.mkdir(parents=True, exist_ok=True)
            self.ensure_entrypoint(speaker_dir)

    def ensure_entrypoint(self, directory: Path) -> Path:
        entrypoint = directory / ENTRYPOINT_NAME
        if not entrypoint.exists():
            entrypoint.write_text("", encoding="utf-8")
        return entrypoint

    def speaker_dir(self, speaker_id: str) -> Path:
        return self.speakers_dir / self._safe_segment(speaker_id)

    def memory_dirs(self, speaker_id: str | None = None) -> list[MemoryDirectory]:
        dirs = [MemoryDirectory(self.global_dir, "global", "Global MEMORY.md")]
        if speaker_id:
            dirs.append(
                MemoryDirectory(
                    self.speaker_dir(speaker_id),
                    "speaker",
                    f"Speaker {speaker_id} MEMORY.md",
                )
            )
        return dirs

    def memory_file_path(self, filename: str, speaker_id: str | None = None) -> Path:
        directory = self.speaker_dir(speaker_id) if speaker_id else self.global_dir
        return directory / self._safe_filename(filename)

    @staticmethod
    def _safe_segment(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value.strip())
        return cleaned.strip("._") or "unknown"

    @classmethod
    def _safe_filename(cls, value: str) -> str:
        filename = cls._safe_segment(value)
        if not filename.endswith(".md"):
            filename += ".md"
        if filename == ENTRYPOINT_NAME:
            filename = "memory_topic.md"
        return filename
