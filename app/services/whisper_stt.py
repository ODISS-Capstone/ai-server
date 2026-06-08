"""Local Whisper STT provider based on faster-whisper.

Mobile clients upload a short audio clip to ai-server; this module keeps the
model in-process and returns the Korean transcript without calling Gemini.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class WhisperSttError(RuntimeError):
    """Local Whisper STT failed or is not installed."""


def _resolve_device() -> str:
    if settings.whisper_device != "auto":
        return settings.whisper_device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(device: str) -> str:
    if settings.whisper_compute_type != "auto":
        return settings.whisper_compute_type
    return "float16" if device == "cuda" else "int8"


@lru_cache(maxsize=1)
def _load_model() -> Any:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on deployment env
        raise WhisperSttError(
            "faster-whisper is not installed. Install ai-server requirements first."
        ) from exc

    device = _resolve_device()
    compute_type = _resolve_compute_type(device)
    logger.info(
        "[WhisperSTT] loading model=%s device=%s compute_type=%s",
        settings.whisper_model,
        device,
        compute_type,
    )
    return WhisperModel(
        settings.whisper_model,
        device=device,
        compute_type=compute_type,
        cpu_threads=settings.whisper_cpu_threads,
    )


def _transcribe_file(path: Path, language: str) -> str:
    model = _load_model()
    segments, info = model.transcribe(
        str(path),
        language=(language.split("-", 1)[0] if language else "ko"),
        vad_filter=True,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    logger.info(
        "[WhisperSTT] transcribed chars=%d language=%s probability=%.3f",
        len(text),
        getattr(info, "language", ""),
        float(getattr(info, "language_probability", 0.0) or 0.0),
    )
    return text


async def transcribe_audio_with_whisper(audio_bytes: bytes, language: str = "ko-KR") -> str:
    if not audio_bytes:
        return ""

    suffix = ".m4a"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="odiss-whisper-", suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = Path(tmp.name)
        return await asyncio.to_thread(_transcribe_file, temp_path, language)
    except WhisperSttError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as provider-specific failure
        raise WhisperSttError(f"Whisper STT failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
