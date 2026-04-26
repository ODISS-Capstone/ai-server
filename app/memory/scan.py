"""structured_memory topic 파일을 얕게 스캔하는 도구."""
from __future__ import annotations

from pathlib import Path

from app.memory.frontmatter import parse_frontmatter
from app.memory.models import MemoryHeader
from app.memory.paths import ENTRYPOINT_NAME
from app.memory.types import parse_memory_type

MAX_MEMORY_FILES = 200
FRONTMATTER_MAX_LINES = 30


def _read_markdown_head(path: Path) -> str:
    """frontmatter 확인에 필요한 문서 앞부분만 읽는다."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for _, line in zip(range(FRONTMATTER_MAX_LINES), handle):
            lines.append(line)
    return "".join(lines)


def _fallback_description(head: str) -> str:
    for line in head.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("---") or stripped.startswith("#"):
            continue
        cleaned = stripped.lstrip("-").lstrip(">").strip()
        if cleaned and ":" not in cleaned[:20]:
            return cleaned[:240]
    return ""


def scan_memory_files(
    memory_dir: Path,
    scope: str,
    *,
    limit: int | None = MAX_MEMORY_FILES,
) -> list[MemoryHeader]:
    """메모리 디렉터리의 topic 파일 헤더를 최신순으로 반환한다."""
    if not memory_dir.exists():
        return []

    headers: list[MemoryHeader] = []
    for path in memory_dir.rglob("*.md"):
        if path.name == ENTRYPOINT_NAME:
            continue
        try:
            head = _read_markdown_head(path)
            metadata, _ = parse_frontmatter(head)
            stat = path.stat()
        except OSError:
            continue

        relative = path.relative_to(memory_dir).as_posix()
        name = metadata.get("name", "").strip() or path.stem.replace("_", " ").title()
        description = metadata.get("description", "").strip() or _fallback_description(head)
        headers.append(
            MemoryHeader(
                filename=relative,
                path=path,
                name=name,
                description=description,
                memory_type=parse_memory_type(metadata.get("type")),
                mtime_ms=stat.st_mtime * 1000,
                scope=scope,
            )
        )

    headers.sort(key=lambda item: item.mtime_ms, reverse=True)
    return headers[:limit] if limit is not None else headers
