"""Read-only patient memory search for the memory browser web UI."""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from app.database.md_store import FLASH_FILES, md_store
from app.engines.memory import MemoryEngine

SEARCHABLE_CATEGORIES = [
    "ocr_history",
    "prescriptions",
    "medication_log",
    "dur_linkage",
    "health_supplement",
]

FLASH_CATEGORY_LABELS = {
    "current_user_profile": "현재 사용자 프로필",
    "current_manual": "현재 환자 메모",
    "current_requirement": "현재 요구사항",
    "context_memory": "대화 맥락",
    "prescription_log": "현재 복용 요약",
}


class MemoryBrowserService:
    """Search and aggregate patient memory records from the MD database."""

    def __init__(self, memory_engine: Optional[MemoryEngine] = None) -> None:
        self.memory = memory_engine or MemoryEngine()
        self.store = self.memory.store

    async def initialize(self) -> None:
        await self.memory.initialize()

    async def search_patients(self, name: str, *, limit: int = 20) -> list[dict[str, Any]]:
        query = (name or "").strip()
        if not query:
            return []

        results: list[dict[str, Any]] = []
        for speaker_id in await self.store.list_patient_ids():
            profile_md = await self.store.read_user_file(speaker_id, "profile.md")
            profile = self.memory._parse_profile(profile_md)
            patient_name = str(profile.get("name") or "").strip()
            if not patient_name or query not in patient_name:
                continue
            results.append(self._patient_summary(speaker_id, profile))
            if len(results) >= limit:
                break
        return results

    async def get_patient_detail(self, speaker_id: str) -> dict[str, Any]:
        profile_md = await self.store.read_user_file(speaker_id, "profile.md")
        profile = self.memory._parse_profile(profile_md)
        history = await self.store.read_user_file(speaker_id, "history.md")
        medication_events = await self.store.read_user_file(speaker_id, "medication_events.md")
        structured = await self.memory.structured_memory.build_context("", speaker_id=speaker_id)

        return {
            "speaker_id": speaker_id,
            "profile": profile,
            "profile_markdown": profile_md,
            "history_markdown": history,
            "medication_events_markdown": medication_events,
            "structured_memory": {
                "memory_index": structured.get("memory_index", ""),
                "memory_prompt": structured.get("memory_prompt", ""),
                "relevant_memories": structured.get("relevant_memories", []),
            },
        }

    async def search_patient_records(
        self,
        speaker_id: str,
        *,
        categories: Optional[list[str]] = None,
        query: str = "",
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        profile_md = await self.store.read_user_file(speaker_id, "profile.md")
        profile = self.memory._parse_profile(profile_md)
        keywords = self._build_keywords(speaker_id, profile, query)
        selected = categories or (SEARCHABLE_CATEGORIES + list(FLASH_FILES.keys()))

        records: list[dict[str, Any]] = []
        for category in selected:
            if category in FLASH_FILES:
                records.extend(await self._search_flash(category, keywords))
                continue
            if category not in SEARCHABLE_CATEGORIES:
                continue
            records.extend(
                await self._search_category(
                    category,
                    keywords=keywords,
                    start=start,
                    end=end,
                    limit=limit,
                )
            )

        records.sort(key=lambda item: (item.get("date") or "", item.get("path") or ""), reverse=True)
        return {
            "speaker_id": speaker_id,
            "profile": self._patient_summary(speaker_id, profile),
            "keywords": keywords,
            "records": records[:limit],
            "total": len(records[:limit]),
        }

    async def read_entry(self, relative_path: str) -> dict[str, Any]:
        content = await self.store.read_safe_entry(relative_path)
        return {
            "path": relative_path,
            "content": content,
        }

    async def _search_category(
        self,
        category: str,
        *,
        keywords: list[str],
        start: Optional[date],
        end: Optional[date],
        limit: int,
    ) -> list[dict[str, Any]]:
        entries = await self.store.list_entries(category, start=start, end=end)
        results: list[dict[str, Any]] = []
        for entry_path in reversed(entries):
            text = await self.store.read_entry(entry_path)
            if keywords and not any(keyword in text for keyword in keywords):
                continue
            relative = str(entry_path.relative_to(self.store.base))
            snippet_keyword = keywords[0] if keywords else ""
            results.append(
                {
                    "category": category,
                    "category_label": category,
                    "date": entry_path.parent.name,
                    "path": relative,
                    "snippet": self.store._extract_snippet(text, snippet_keyword, width=240),
                    "preview": text[:1200],
                }
            )
            if len(results) >= limit:
                break
        return results

    async def _search_flash(self, key: str, keywords: list[str]) -> list[dict[str, Any]]:
        text = await self.store.read_flash(key)
        if not text.strip():
            return []
        if keywords and not any(keyword in text for keyword in keywords):
            return []
        relative = f"flash/{FLASH_FILES[key]}"
        snippet_keyword = keywords[0] if keywords else ""
        return [
            {
                "category": key,
                "category_label": FLASH_CATEGORY_LABELS.get(key, key),
                "date": "flash",
                "path": relative,
                "snippet": self.store._extract_snippet(text, snippet_keyword, width=240),
                "preview": text[:1200],
            }
        ]

    @staticmethod
    def _build_keywords(
        speaker_id: str,
        profile: dict[str, Any],
        query: str,
    ) -> list[str]:
        keywords: list[str] = []
        for token in [speaker_id, str(profile.get("name") or ""), query]:
            token = token.strip()
            if token and token not in keywords:
                keywords.append(token)
        for condition in profile.get("conditions") or []:
            condition_text = str(condition).strip()
            if condition_text and condition_text not in keywords:
                keywords.append(condition_text)
        for part in query.split():
            part = part.strip()
            if len(part) > 1 and part not in keywords:
                keywords.append(part)
        return keywords

    @staticmethod
    def _patient_summary(speaker_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "speaker_id": speaker_id,
            "name": profile.get("name") or "",
            "age": profile.get("age") or "",
            "gender": profile.get("gender") or "",
            "conditions": profile.get("conditions") or [],
            "last_seen_at": profile.get("last_seen_at") or "",
            "verified_at": profile.get("verified_at") or "",
        }
