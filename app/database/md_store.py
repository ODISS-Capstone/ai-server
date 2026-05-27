"""MD Database Layer — 파일시스템 기반 Markdown 데이터베이스.

구조:
  permanent/
  ├── patients/{user_id}/profile.md     사용자별 프로필
  ├── ocr_history/2026-04-09/001.md     날짜별 OCR 기록
  ├── prescriptions/2026-04-09/001.md   날짜별 처방전
  ├── medication_log/2026-04-09/001.md  날짜별 상담 기록
  ├── dur_linkage/2026-04-09/001.md     날짜별 DUR 호출 내역
  └── health_supplement/2026-04-09/001.md
  flash/
  ├── prescription_log.md               현재 복용 요약 (덮어쓰기)
  ├── current_user_profile.md
  ├── current_requirement.md
  ├── current_manual.md
  └── context_memory.md
"""
import asyncio
import glob as glob_mod
import os
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

from app.core.config import settings

PERMANENT_CATEGORIES = [
    "patients",
    "ocr_history",
    "prescriptions",
    "medication_log",
    "dur_linkage",
    "health_supplement",
    "feedback",
    "assistant_diagnostics",
]

FLASH_FILES = {
    "prescription_log": "prescription_log.md",
    "current_user_profile": "current_user_profile.md",
    "current_requirement": "current_requirement.md",
    "current_manual": "current_manual.md",
    "context_memory": "context_memory.md",
}


class MDStore:
    """파일시스템 기반 Markdown 데이터베이스.

    Permanent 데이터는 ``category/YYYY-MM-DD/NNN.md`` 형태로 저장되어
    날짜 단위로 디렉토리가 생기고, 각 엔트리가 독립된 파일이 됩니다.
    Flash 데이터는 단일 파일을 덮어쓰는 휘발성 메모리입니다.
    """

    def __init__(self, base_path: Optional[str] = None):
        self.base = Path(base_path or settings.md_database_path)
        self.permanent = self.base / "permanent"
        self.flash = self.base / "flash"
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await asyncio.get_event_loop().run_in_executor(None, self._sync_init)

    # ────────────────────────────────────────────
    #  Permanent — 날짜별 개별 MD 파일로 저장
    # ────────────────────────────────────────────

    async def save(self, category: str, content: str, *, day: Optional[date] = None) -> Path:
        """새 엔트리를 ``category/YYYY-MM-DD/NNN.md`` 에 저장하고 경로 반환."""
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._sync_save, category, content, day,
            )

    async def read_entry(self, path: str | Path) -> str:
        """절대/상대 경로로 단일 엔트리 읽기."""
        p = Path(path) if Path(path).is_absolute() else self.base / path
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_read, p,
        )

    async def list_entries(
        self,
        category: str,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[Path]:
        """카테고리 내 엔트리 경로 목록 반환 (날짜 범위 필터 가능)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_list, category, start, end,
        )

    async def search(
        self,
        category: str,
        keyword: str,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 20,
    ) -> list[dict]:
        """카테고리 내 MD 파일들을 키워드로 탐색. ``[{path, snippet, date}]`` 반환."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_search, category, keyword, start, end, limit,
        )

    async def read_latest(self, category: str, n: int = 1) -> list[dict]:
        """카테고리에서 가장 최근 N개 엔트리를 읽어 반환."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_read_latest, category, n,
        )

    # ────────────────────────────────────────────
    #  사용자 (patients) — user_id별 디렉토리
    # ────────────────────────────────────────────

    async def save_user_file(self, user_id: str, filename: str, content: str) -> Path:
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._sync_save_user, user_id, filename, content,
            )

    async def read_user_file(self, user_id: str, filename: str) -> str:
        p = self.permanent / "patients" / user_id / filename
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_read, p,
        )

    async def user_exists(self, user_id: str) -> bool:
        p = self.permanent / "patients" / user_id / "profile.md"
        return p.exists()

    async def list_patient_ids(self) -> list[str]:
        """Return all speaker/user ids under permanent/patients."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_list_patient_ids,
        )

    async def list_user_files(self, user_id: str) -> list[str]:
        """Return filenames stored for a patient."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_list_user_files, user_id,
        )

    async def read_safe_entry(self, relative_path: str) -> str:
        """Read a markdown entry only if it stays inside the database base."""
        resolved = self.resolve_safe_path(relative_path)
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_read, resolved,
        )

    def resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve a relative path and reject traversal outside the base."""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.base / candidate).resolve()
        base = self.base.resolve()
        if base not in resolved.parents and resolved != base:
            raise ValueError("Path is outside the memory database root")
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(relative_path)
        return resolved

    # ────────────────────────────────────────────
    #  Flash — 단일 파일 덮어쓰기 / 읽기
    # ────────────────────────────────────────────

    async def read_flash(self, key: str) -> str:
        p = self._flash_path(key)
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_read, p,
        )

    async def write_flash(self, key: str, content: str) -> None:
        p = self._flash_path(key)
        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                None, self._sync_write, p, content,
            )

    async def clear_flash(self) -> None:
        async with self._lock:
            for key in FLASH_FILES:
                p = self._flash_path(key)
                await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_write, p, "",
                )

    # ────────────────────────────────────────────
    #  동기 구현 (executor 에서 실행)
    # ────────────────────────────────────────────

    def _sync_init(self) -> None:
        for cat in PERMANENT_CATEGORIES:
            (self.permanent / cat).mkdir(parents=True, exist_ok=True)
        self.flash.mkdir(parents=True, exist_ok=True)
        for key, fname in FLASH_FILES.items():
            p = self.flash / fname
            if not p.exists():
                p.write_text("", encoding="utf-8")

    def _sync_save(self, category: str, content: str, day: Optional[date]) -> Path:
        day = day or date.today()
        day_dir = self.permanent / category / day.isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(day_dir.glob("*.md"))
        seq = len(existing) + 1
        now_ts = datetime.now().strftime("%H%M%S")
        filename = f"{seq:03d}_{now_ts}.md"
        filepath = day_dir / filename
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def _sync_save_user(self, user_id: str, filename: str, content: str) -> Path:
        user_dir = self.permanent / "patients" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        p = user_dir / filename
        p.write_text(content, encoding="utf-8")
        return p

    def _sync_read(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _sync_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _sync_list(
        self, category: str, start: Optional[date], end: Optional[date],
    ) -> list[Path]:
        cat_dir = self.permanent / category
        if not cat_dir.exists():
            return []

        day_dirs = sorted(
            [d for d in cat_dir.iterdir() if d.is_dir() and self._is_date_dir(d.name)],
            key=lambda d: d.name,
        )

        if start:
            day_dirs = [d for d in day_dirs if d.name >= start.isoformat()]
        if end:
            day_dirs = [d for d in day_dirs if d.name <= end.isoformat()]

        results: list[Path] = []
        for d in day_dirs:
            results.extend(sorted(d.glob("*.md")))
        return results

    def _sync_search(
        self,
        category: str,
        keyword: str,
        start: Optional[date],
        end: Optional[date],
        limit: int,
    ) -> list[dict]:
        entries = self._sync_list(category, start, end)
        keywords = [w for w in keyword.split() if len(w) > 1]
        if not keywords:
            return []

        results: list[dict] = []
        for p in reversed(entries):
            text = self._sync_read(p)
            if any(kw in text for kw in keywords):
                snippet = self._extract_snippet(text, keywords[0], width=200)
                results.append({
                    "path": str(p.relative_to(self.base)),
                    "date": p.parent.name,
                    "snippet": snippet,
                })
                if len(results) >= limit:
                    break
        return results

    def _sync_read_latest(self, category: str, n: int) -> list[dict]:
        entries = self._sync_list(category, None, None)
        latest = entries[-n:] if len(entries) >= n else entries
        results: list[dict] = []
        for p in reversed(latest):
            results.append({
                "path": str(p.relative_to(self.base)),
                "date": p.parent.name,
                "content": self._sync_read(p),
            })
        return results

    def _sync_list_patient_ids(self) -> list[str]:
        patients_dir = self.permanent / "patients"
        if not patients_dir.exists():
            return []
        return sorted(
            item.name
            for item in patients_dir.iterdir()
            if item.is_dir() and (item / "profile.md").exists()
        )

    def _sync_list_user_files(self, user_id: str) -> list[str]:
        user_dir = self.permanent / "patients" / user_id
        if not user_dir.exists():
            return []
        return sorted(path.name for path in user_dir.glob("*.md"))

    def _flash_path(self, key: str) -> Path:
        fname = FLASH_FILES.get(key)
        if not fname:
            raise KeyError(f"Unknown flash key: {key}")
        return self.flash / fname

    @staticmethod
    def _is_date_dir(name: str) -> bool:
        try:
            date.fromisoformat(name)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_snippet(text: str, keyword: str, width: int = 200) -> str:
        idx = text.find(keyword)
        if idx == -1:
            return text[:width]
        start = max(0, idx - width // 2)
        end = min(len(text), idx + width // 2)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet


md_store = MDStore()
