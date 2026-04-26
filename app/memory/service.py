"""structured_memory 저장소를 관리하고 조회 문맥을 구성하는 서비스."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.memory.frontmatter import build_frontmatter
from app.memory.models import MemoryHeader, MemoryType, RelevantMemory
from app.memory.paths import ENTRYPOINT_NAME, StructuredMemoryPaths
from app.memory.scan import scan_memory_files
from app.memory.selector import select_relevant_memories
from app.memory.types import MAX_ENTRYPOINT_BYTES, MAX_ENTRYPOINT_LINES, build_memory_policy


class StructuredMemoryService:
    """Claude Code의 memdir처럼 MEMORY.md 인덱스와 topic 파일을 관리한다."""

    def __init__(self, base_path: str | None = None):
        self.paths = StructuredMemoryPaths(base_path)
        self._lock = asyncio.Lock()

    async def initialize(self, speaker_id: Optional[str] = None) -> None:
        """structured_memory 기본 디렉터리를 준비한다."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._sync_initialize, speaker_id,
        )

    async def build_context(
        self,
        query: str,
        *,
        speaker_id: Optional[str] = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """MEMORY.md 인덱스와 관련 topic 파일을 묶어 프롬프트 문맥을 만든다."""
        await self.initialize(speaker_id)
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_build_context, query, speaker_id, limit,
        )

    async def upsert_memory(
        self,
        *,
        filename: str,
        name: str,
        description: str,
        memory_type: MemoryType,
        body: str,
        speaker_id: Optional[str] = None,
    ) -> None:
        """topic 파일을 쓰고 같은 디렉터리의 MEMORY.md 인덱스를 재생성한다."""
        async with self._lock:
            await self.initialize(speaker_id)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._sync_upsert_memory,
                filename,
                name,
                description,
                memory_type,
                body,
                speaker_id,
            )

    async def sync_patient_profile(self, speaker_id: str, profile: dict[str, Any]) -> None:
        """환자 프로필을 화자별 user memory로 동기화한다."""
        rows = [
            f"- 이름: {profile.get('name') or '-'}",
            f"- ID: {speaker_id}",
            f"- 연령: {profile.get('age') or '-'}",
            f"- 성별: {profile.get('gender') or '-'}",
            f"- 기저질환: {', '.join(profile.get('conditions') or []) or '-'}",
        ]
        await self.upsert_memory(
            filename="patient_profile.md",
            name="환자 프로필",
            description=f"{speaker_id} 환자 기본 프로필",
            memory_type="user",
            body="# 환자 프로필\n\n" + "\n".join(rows),
            speaker_id=speaker_id,
        )

    async def sync_medication_context(
        self,
        *,
        med_names: list[str],
        dur_results: list[dict[str, Any]],
        recorded_at: str,
        speaker_id: Optional[str] = None,
    ) -> None:
        """OCR+DUR 결과를 최신 복약 project memory로 동기화한다."""
        med_lines = "\n".join(f"- {name}" for name in med_names) or "- 확인된 약품 없음"
        dur_lines: list[str] = []
        for dur in dur_results:
            name = dur.get("name", "이름 없음")
            contras = len(dur.get("contraindications", []))
            cautions = len(dur.get("precautions", []))
            dur_lines.append(f"- {name}: 금기 {contras}건 / 주의 {cautions}건")

        body = (
            "# 최신 복약 및 DUR 요약\n\n"
            f"- 기록 시각: {recorded_at}\n"
            f"- 화자 ID: {speaker_id or 'global'}\n\n"
            "## 약품 목록\n"
            f"{med_lines}\n\n"
            "## DUR 결과\n"
            f"{chr(10).join(dur_lines) if dur_lines else '- DUR 결과 없음'}"
        )
        description = (
            "최근 OCR 처방 약품: " + ", ".join(med_names[:5])
            if med_names
            else "최근 OCR 처방 약품 없음"
        )
        await self.upsert_memory(
            filename="current_medication.md",
            name="최신 복약 및 DUR 요약",
            description=description,
            memory_type="project",
            body=body,
            speaker_id=speaker_id,
        )

    def _sync_initialize(self, speaker_id: Optional[str] = None) -> None:
        self.paths.ensure_dirs(speaker_id)

    def _sync_build_context(
        self,
        query: str,
        speaker_id: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        memory_dirs = self.paths.memory_dirs(speaker_id)
        headers_by_scope: dict[str, list[MemoryHeader]] = {
            memory_dir.scope: scan_memory_files(memory_dir.path, memory_dir.scope)
            for memory_dir in memory_dirs
        }

        selected: list[RelevantMemory] = []
        for memory_dir in memory_dirs:
            boost = 1.5 if memory_dir.scope == "speaker" else 0.5
            selected.extend(
                select_relevant_memories(
                    query,
                    headers_by_scope[memory_dir.scope],
                    limit=limit,
                    scope_boost=boost,
                )
            )
        selected.sort(key=lambda item: item.score, reverse=True)
        selected = selected[:limit]

        entrypoint_sections = [
            self._build_entrypoint_section(memory_dir.label, memory_dir.path)
            for memory_dir in memory_dirs
        ]
        relevant_blocks = [self._build_relevant_block(memory) for memory in selected]
        briefs = [self._build_brief(memory) for memory in selected]

        prompt_sections = [build_memory_policy(self.paths.base), *entrypoint_sections]
        if relevant_blocks:
            prompt_sections.append("## Relevant memories\n\n" + "\n\n".join(relevant_blocks))

        return {
            "memory_prompt": "\n\n".join(section for section in prompt_sections if section).strip(),
            "memory_index": "\n\n".join(entrypoint_sections).strip(),
            "relevant_memories": [memory.to_dict() for memory in selected],
            "memory_briefs": briefs[:3],
        }

    def _sync_upsert_memory(
        self,
        filename: str,
        name: str,
        description: str,
        memory_type: MemoryType,
        body: str,
        speaker_id: Optional[str],
    ) -> None:
        path = self.paths.memory_file_path(filename, speaker_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = build_frontmatter(
            {
                "name": name,
                "description": description,
                "type": memory_type,
            },
            body,
        )
        path.write_text(content, encoding="utf-8")
        self._sync_rebuild_entrypoint(path.parent, "speaker" if speaker_id else "global")

    def _sync_rebuild_entrypoint(self, directory, scope: str) -> None:
        headers = scan_memory_files(directory, scope, limit=None)
        lines = []
        for header in headers:
            hook = header.description or header.name
            lines.append(f"- [{header.name}]({header.filename}) - {hook}")
        (directory / ENTRYPOINT_NAME).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _build_entrypoint_section(self, title: str, directory) -> str:
        entrypoint = directory / ENTRYPOINT_NAME
        try:
            content = entrypoint.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        if not content:
            content = f"Your {ENTRYPOINT_NAME} is currently empty."
        return f"## {title}\n\n{self._truncate_entrypoint_content(content)}".strip()

    def _truncate_entrypoint_content(self, content: str) -> str:
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
                f"\n\n> WARNING: {ENTRYPOINT_NAME} was truncated. "
                "Keep index entries concise and move detail into topic files."
            )
        return truncated

    def _build_relevant_block(self, memory: RelevantMemory) -> str:
        freshness = f"\nFreshness: {memory.freshness_note}" if memory.freshness_note else ""
        return "\n".join(
            [
                f"### {memory.name} [{memory.memory_type or 'unknown'}]",
                f"Path: {memory.path}",
                f"Description: {memory.description or '(none)'}",
                f"Updated: {memory.age_text}",
                memory.body.strip(),
                freshness.strip(),
            ]
        ).strip()

    def _build_brief(self, memory: RelevantMemory) -> str:
        if memory.scope == "speaker":
            prefix = "화자 메모리"
        elif memory.memory_type == "user":
            prefix = "사용자 맥락"
        elif memory.memory_type == "reference":
            prefix = "참고 기준"
        else:
            prefix = "공통 메모리"
        return f"{prefix}: {memory.excerpt}"
