"""아키텍처 문서 기준 메모리 경로를 계산하는 모듈."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.memory.models import MemoryType

ENTRYPOINT_NAME = "MEMORY.md"

FLASH_SPECS: dict[str, tuple[str, str, MemoryType | None]] = {
    "prescription_log": ("prescription_log.md", "Current prescription summary", "project"),
    "current_user_profile": ("current_user_profile.md", "Current user profile", "user"),
    "current_requirement": ("current_requirement.md", "Recent user requirements", "feedback"),
    "current_manual": ("current_manual.md", "Current response manual", "reference"),
    "context_memory": ("context_memory.md", "Conversation context memory", "project"),
}

PATIENT_FILE_SPECS: dict[str, tuple[str, MemoryType | None]] = {
    "profile.md": ("Patient profile", "user"),
    "history.md": ("Patient history", "project"),
}

PERMANENT_CATEGORY_SPECS: dict[str, tuple[str, MemoryType | None]] = {
    "ocr_history": ("OCR history", "project"),
    "prescriptions": ("Prescription history", "project"),
    "medication_log": ("Medication log", "project"),
    "dur_linkage": ("DUR linkage history", "project"),
    "health_supplement": ("Health supplement history", "project"),
}


@dataclass(frozen=True, slots=True)
class MemorySourceFile:
    """메모리 스캔 대상 파일의 경로와 기본 메타데이터."""

    path: Path
    relative_path: str
    scope: str
    default_name: str
    default_type: MemoryType | None


class StructuredMemoryPaths:
    """`flash/`와 `permanent/` 아래 메모리 소스 경로를 모아준다."""

    def __init__(self, base_path: str | None = None):
        self.base = Path(base_path or settings.md_database_path)
        self.permanent = self.base / "permanent"
        self.flash = self.base / "flash"

    def ensure_dirs(self) -> None:
        """아키텍처에서 사용하는 기본 MD 디렉터리를 보장한다."""
        self.base.mkdir(parents=True, exist_ok=True)
        self.permanent.mkdir(parents=True, exist_ok=True)
        self.flash.mkdir(parents=True, exist_ok=True)
        (self.permanent / "patients").mkdir(parents=True, exist_ok=True)
        for category in PERMANENT_CATEGORY_SPECS:
            (self.permanent / category).mkdir(parents=True, exist_ok=True)

    def flash_sources(self) -> list[MemorySourceFile]:
        """현재 상태를 담는 flash 메모리 파일 목록을 반환한다."""
        return [
            MemorySourceFile(
                path=self.flash / filename,
                relative_path=f"flash/{filename}",
                scope="flash",
                default_name=display_name,
                default_type=memory_type,
            )
            for filename, display_name, memory_type in FLASH_SPECS.values()
        ]

    def speaker_sources(self, speaker_id: str | None) -> list[MemorySourceFile]:
        """환자별 프로필과 이력 파일을 메모리 소스로 반환한다."""
        if not speaker_id:
            return []

        user_dir = self.permanent / "patients" / speaker_id
        sources: list[MemorySourceFile] = []
        seen: set[str] = set()

        for filename, (display_name, memory_type) in PATIENT_FILE_SPECS.items():
            sources.append(
                MemorySourceFile(
                    path=user_dir / filename,
                    relative_path=f"permanent/patients/{speaker_id}/{filename}",
                    scope="patient",
                    default_name=display_name,
                    default_type=memory_type,
                )
            )
            seen.add(filename)

        if user_dir.exists():
            for path in sorted(user_dir.glob("*.md")):
                if path.name in seen:
                    continue
                sources.append(
                    MemorySourceFile(
                        path=path,
                        relative_path=f"permanent/patients/{speaker_id}/{path.name}",
                        scope="patient",
                        default_name=path.stem.replace("_", " ").title(),
                        default_type="project",
                    )
                )

        return sources

    def permanent_sources(self, *, per_category_limit: int = 200) -> list[MemorySourceFile]:
        """영구 보관 카테고리별 이력 파일을 최신순으로 수집한다."""
        sources: list[MemorySourceFile] = []
        for category, (display_name, memory_type) in PERMANENT_CATEGORY_SPECS.items():
            category_dir = self.permanent / category
            if not category_dir.exists():
                continue

            files = sorted(
                category_dir.rglob("*.md"),
                key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
                reverse=True,
            )[:per_category_limit]

            for path in files:
                relative = path.relative_to(self.base).as_posix()
                date_hint = path.parent.name if path.parent != category_dir else ""
                default_name = f"{display_name} {date_hint}".strip()
                sources.append(
                    MemorySourceFile(
                        path=path,
                        relative_path=relative,
                        scope="permanent",
                        default_name=default_name,
                        default_type=memory_type,
                    )
                )

        return sources
