"""ODISS MD 저장소 위에서 메모리 조회 문맥을 구성하는 서비스."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.memory.models import MemoryHeader, RelevantMemory
from app.memory.paths import StructuredMemoryPaths
from app.memory.scan import scan_memory_files
from app.memory.selector import select_relevant_memories
from app.memory.types import MAX_ENTRYPOINT_BYTES, MAX_ENTRYPOINT_LINES, build_memory_policy


class StructuredMemoryService:
    """`flash/`와 `permanent/`를 읽어 메모리 프롬프트를 만든다."""

    def __init__(self, base_path: str | None = None):
        self.paths = StructuredMemoryPaths(base_path)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """메모리 조회에 필요한 기본 디렉터리를 준비한다."""
        await asyncio.get_event_loop().run_in_executor(None, self._sync_initialize)

    async def build_context(
        self,
        query: str,
        *,
        speaker_id: Optional[str] = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """질의와 화자 정보를 바탕으로 메모리 프롬프트와 요약을 만든다."""
        await self.initialize()
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_build_context, query, speaker_id, limit,
        )

    def _sync_initialize(self) -> None:
        self.paths.ensure_dirs()

    def _sync_build_context(
        self,
        query: str,
        speaker_id: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        """flash, 환자별, 영구 이력을 합쳐 관련 메모리 문맥을 구성한다."""
        flash_headers = scan_memory_files(self.paths.flash_sources())
        patient_headers = scan_memory_files(self.paths.speaker_sources(speaker_id))
        permanent_headers = scan_memory_files(self.paths.permanent_sources())

        selected: list[RelevantMemory] = []
        selected.extend(
            select_relevant_memories(query, patient_headers, limit=limit, scope_boost=1.5)
        )
        selected.extend(
            select_relevant_memories(query, flash_headers, limit=limit, scope_boost=1.0)
        )
        selected.extend(
            select_relevant_memories(query, permanent_headers, limit=limit, scope_boost=0.25)
        )
        selected.sort(key=lambda item: item.score, reverse=True)
        selected = selected[:limit]

        relevant_blocks = []
        briefs = []
        for memory in selected:
            freshness = f"\nFreshness: {memory.freshness_note}" if memory.freshness_note else ""
            relevant_blocks.append(
                "\n".join(
                    [
                        f"### {memory.name} [{memory.memory_type or 'unknown'}]",
                        f"Path: {memory.path}",
                        f"Description: {memory.description or '(none)'}",
                        f"Updated: {memory.age_text}",
                        memory.body.strip(),
                        freshness.strip(),
                    ]
                ).strip()
            )
            briefs.append(self._build_brief(memory))

        index_sections = [
            self._build_virtual_index("Flash Memories", flash_headers),
            self._build_virtual_index("Patient Memories", patient_headers),
            self._build_virtual_index("Permanent Memories", permanent_headers),
        ]

        prompt_sections = [build_memory_policy(self.paths.base), *index_sections]
        prompt_sections = [section for section in prompt_sections if section.strip()]
        if relevant_blocks:
            prompt_sections.append("## Relevant memories\n\n" + "\n\n".join(relevant_blocks))

        return {
            "memory_prompt": "\n\n".join(prompt_sections).strip(),
            "memory_index": "\n\n".join(index_sections).strip(),
            "relevant_memories": [memory.to_dict() for memory in selected],
            "memory_briefs": briefs[:3],
        }

    def _build_virtual_index(self, title: str, headers: list[MemoryHeader]) -> str:
        """실제 `MEMORY.md`를 만들지 않고 가상 인덱스 텍스트를 생성한다."""
        if not headers:
            return f"## {title}\n\n(Empty)"

        lines = []
        for header in headers:
            hook = header.description or header.name
            lines.append(f"- [{header.name}]({header.filename}) - {hook}")

        content = self._truncate_index_content("\n".join(lines).strip())
        return f"## {title}\n\n{content}".strip()

    def _truncate_index_content(self, content: str) -> str:
        """인덱스가 너무 길어지면 줄 수와 바이트 수 기준으로 잘라낸다."""
        lines = content.splitlines()
        was_line_truncated = len(lines) > MAX_ENTRYPOINT_LINES
        truncated = "\n".join(lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else content

        encoded = truncated.encode("utf-8")
        was_byte_truncated = len(encoded) > MAX_ENTRYPOINT_BYTES
        if was_byte_truncated:
            encoded = encoded[:MAX_ENTRYPOINT_BYTES]
            while encoded and (encoded[-1] & 0b1100_0000) == 0b1000_0000:
                encoded = encoded[:-1]
            truncated = encoded.decode("utf-8", errors="ignore").rsplit("\n", 1)[0]

        if was_line_truncated or was_byte_truncated:
            truncated += (
                "\n\n> WARNING: MEMORY.md was truncated. Keep entries concise and rely on the original MD files for detail."
            )

        return truncated

    def _build_brief(self, memory: RelevantMemory) -> str:
        """선택된 메모리를 짧은 한 줄 브리프로 변환한다."""
        if memory.scope == "flash":
            prefix = "현재 상태"
        elif memory.memory_type == "user":
            prefix = "환자 맥락"
        elif memory.memory_type == "reference":
            prefix = "참고 기준"
        else:
            prefix = "참고 기록"
        return f"{prefix}: {memory.excerpt}"
