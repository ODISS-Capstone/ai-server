"""메모리 조회용 관련도 점수를 계산하는 선택기."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from app.memory.freshness import memory_age_days, memory_age_text, memory_freshness_text
from app.memory.frontmatter import parse_frontmatter
from app.memory.models import MemoryHeader, RelevantMemory

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """영문, 숫자, 한글 토큰을 추출해 검색용 토큰 목록으로 만든다."""
    tokens = TOKEN_RE.findall(text.lower())
    return [token for token in tokens if len(token) >= 2 or re.search(r"[가-힣]", token)]


def _token_score(query_tokens: list[str], target_text: str) -> float:
    """질의 토큰이 대상 텍스트에 얼마나 겹치는지 점수화한다."""
    if not query_tokens:
        return 0.0
    target_counts = Counter(tokenize(target_text))
    if not target_counts:
        return 0.0

    score = 0.0
    for token in query_tokens:
        if token in target_counts:
            score += 1.0 + math.log1p(target_counts[token])
    return score / max(len(set(query_tokens)), 1)


def _read_memory_body(path: Path) -> str:
    """메모리 파일에서 frontmatter를 제외한 본문만 읽어온다."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    _, body = parse_frontmatter(text)
    return body.strip()


def _build_excerpt(body: str, fallback: str) -> str:
    """본문에서 요약 노출에 적합한 한 줄 발췌문을 만든다."""
    raw_lines = [line.strip() for line in body.splitlines() if line.strip()]

    bullet_lines = []
    for line in raw_lines:
        if not line.startswith("- "):
            continue
        candidate = line[2:].strip()
        if candidate:
            bullet_lines.append(candidate)

    for line in bullet_lines:
        if line.startswith("**Why:**") or line.startswith("**How to apply:**"):
            continue
        return line[:240]

    for line in raw_lines:
        if line.startswith("**Why:**") or line.startswith("**How to apply:**"):
            continue
        if line.endswith(":"):
            continue
        return line[:240]

    return fallback[:240]


def select_relevant_memories(
    query: str,
    headers: list[MemoryHeader],
    *,
    limit: int = 5,
    scope_boost: float = 0.0,
) -> list[RelevantMemory]:
    """질의와 실제로 맞는 메모리만 골라 점수순으로 반환한다."""
    query_tokens = tokenize(query)
    if not headers:
        return []

    ranked: list[RelevantMemory] = []
    query_lower = query.lower()

    for header in headers:
        body = _read_memory_body(header.path)
        header_text = " ".join(
            part for part in [header.name, header.description, header.filename] if part
        )
        header_score = _token_score(query_tokens, header_text) * 2.0
        body_score = _token_score(query_tokens, body)
        phrase_bonus = 1.5 if query_lower and query_lower in f"{header_text}\n{body}".lower() else 0.0
        has_match = header_score > 0 or body_score > 0 or phrase_bonus > 0
        if not has_match:
            continue
        recency_bonus = max(0.0, 1.0 - (memory_age_days(header.mtime_ms) / 30))
        score = header_score + body_score + phrase_bonus + scope_boost + recency_bonus
        if score <= 0:
            continue

        excerpt = _build_excerpt(body, header.description or header.name)
        ranked.append(
            RelevantMemory(
                path=str(header.path),
                name=header.name,
                description=header.description,
                memory_type=header.memory_type,
                body=body,
                excerpt=excerpt,
                age_text=memory_age_text(header.mtime_ms),
                freshness_note=memory_freshness_text(header.mtime_ms),
                score=score,
                scope=header.scope,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:limit]
