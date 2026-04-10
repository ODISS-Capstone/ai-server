"""Structured memory data models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

MemoryType = Literal["user", "feedback", "project", "reference"]


@dataclass(slots=True)
class MemoryHeader:
    filename: str
    path: Path
    name: str
    description: str
    memory_type: Optional[MemoryType]
    mtime_ms: float
    scope: str


@dataclass(slots=True)
class RelevantMemory:
    path: str
    name: str
    description: str
    memory_type: Optional[MemoryType]
    body: str
    excerpt: str
    age_text: str
    freshness_note: str
    score: float
    scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "type": self.memory_type,
            "body": self.body,
            "excerpt": self.excerpt,
            "age_text": self.age_text,
            "freshness_note": self.freshness_note,
            "score": round(self.score, 3),
            "scope": self.scope,
        }
