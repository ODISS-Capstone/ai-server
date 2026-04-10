"""아키텍처 문서 기준 메모리 파일을 얕게 스캔하는 도구."""
from __future__ import annotations

from app.memory.frontmatter import parse_frontmatter
from app.memory.models import MemoryHeader
from app.memory.paths import MemorySourceFile

MAX_MEMORY_FILES = 200
FRONTMATTER_MAX_LINES = 30


def _read_markdown_head(path) -> str:
    """frontmatter와 제목을 확인할 수 있도록 문서 앞부분만 읽는다."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for _, line in zip(range(FRONTMATTER_MAX_LINES), handle):
            lines.append(line)
    return "".join(lines)


def _extract_name_and_description(head: str, default_name: str) -> tuple[str, str]:
    """frontmatter, 제목, 첫 본문 줄을 이용해 표시용 이름과 설명을 만든다."""
    metadata, _ = parse_frontmatter(head)
    name = metadata.get("name", "").strip()
    description = metadata.get("description", "").strip()

    lines = [line.strip() for line in head.splitlines() if line.strip()]

    if not name:
        for line in lines:
            if line.startswith("#"):
                name = line.lstrip("#").strip()
                break

    if not description:
        for line in lines:
            if line.startswith("#"):
                continue
            cleaned = line.lstrip("-").lstrip(">").strip()
            if not cleaned:
                continue
            if name and cleaned == name:
                continue
            description = cleaned[:240]
            break

    return name or default_name, description


def scan_memory_files(sources: list[MemorySourceFile]) -> list[MemoryHeader]:
    """메모리 후보 파일을 읽어 검색용 헤더 목록으로 정리한다."""
    if not sources:
        return []

    headers: list[MemoryHeader] = []
    for source in sources:
        path = source.path
        if not path.exists():
            continue
        try:
            head = _read_markdown_head(path)
            name, description = _extract_name_and_description(head, source.default_name)
            stat = path.stat()
        except OSError:
            continue

        headers.append(
            MemoryHeader(
                filename=source.relative_path,
                path=path,
                name=name,
                description=description,
                memory_type=source.default_type,
                mtime_ms=stat.st_mtime * 1000,
                scope=source.scope,
            )
        )

    headers.sort(key=lambda item: item.mtime_ms, reverse=True)
    return headers[:MAX_MEMORY_FILES]
