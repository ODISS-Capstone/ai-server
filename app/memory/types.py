"""메모리 분류 체계와 조회용 프롬프트 텍스트 정의."""
from __future__ import annotations

from pathlib import Path

MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000


def build_memory_policy(memory_dir: Path) -> str:
    """메모리 조회 프롬프트에 들어갈 저장소 규칙 설명을 만든다."""
    return "\n".join(
        [
            f"# ODISS Memory View ({memory_dir})",
            "",
            "This memory view follows the server.mermaid MD Database Layer.",
            "",
            "## Storage layout",
            "- flash/: current compressed state such as prescription_log.md, current_user_profile.md, current_requirement.md, current_manual.md, and context_memory.md.",
            "- permanent/: historical records such as patients/, ocr_history/, prescriptions/, medication_log/, dur_linkage/, and health_supplement/.",
            "",
            "## Memory interpretation",
            "- user: patient profile and patient-specific stable context.",
            "- feedback: recent user requirements or care workflow guidance.",
            "- project: OCR history, prescriptions, medication logs, DUR linkage, and conversation context.",
            "- reference: current manual or other guidance references.",
            "",
            "## Retrieval rules",
            "- Prefer flash files when the user asks about current state.",
            "- Use permanent files as historical evidence and prior context.",
            "- Treat older records as snapshots, not live truth. Re-verify current medication and safety state with the latest OCR, DUR, or direct confirmation.",
        ]
    )
