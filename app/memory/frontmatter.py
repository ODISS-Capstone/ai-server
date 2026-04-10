"""메모리용 마크다운 파일의 frontmatter를 읽고 쓰는 유틸리티."""
from __future__ import annotations

from typing import Any


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """문서 앞쪽 frontmatter를 분리해 메타데이터와 본문을 반환한다."""
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines:
        return {}, text

    metadata: dict[str, str] = {}
    end_index = None
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == "---":
            end_index = idx
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    if end_index is None:
        return {}, text

    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return metadata, body


def build_frontmatter(metadata: dict[str, Any], body: str) -> str:
    """메타데이터와 본문을 frontmatter 포함 마크다운 문자열로 조합한다."""
    lines = ["---"]
    for key, value in metadata.items():
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    return "\n".join(lines).strip() + "\n"
