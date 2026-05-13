"""Typed contracts for Conversation / Memory / Reasoning engines.

These schemas define ownership boundaries between engines so runtime
orchestration and dataset design can target the same interfaces.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReasoningMode(str, Enum):
    """Top-level routing choice produced by ReasoningEngine."""

    TOOL_FIRST = "tool_first"
    FRONTIER_FIRST = "frontier_first"
    MEMORY_ONLY = "memory_only"
    ASK_USER_CLARIFY = "ask_user_clarify"


class ReasoningTask(BaseModel):
    """Deterministic task unit that the reasoning engine may execute."""

    type: str = Field(..., description="Task identifier such as dur_check")
    priority: int = Field(..., description="Smaller number runs first")
    description: str = Field("", description="Human readable rationale")
    owner: str = Field(
        "reasoning_engine",
        description="Logical owner of this task decision",
    )


class ReasoningRouteInput(BaseModel):
    """Inputs required to decide the top-level reasoning route."""

    text: str = Field(..., description="User utterance after STT")
    speaker_id: Optional[str] = Field(None, description="Current speaker identifier")
    is_smalltalk: bool = Field(False, description="Conversation-engine smalltalk hint")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Memory/context snapshot used for routing",
    )


class ReasoningRouteDecision(BaseModel):
    """ReasoningEngine output: route choice + optional deterministic tasks."""

    mode: ReasoningMode = Field(..., description="Primary execution strategy")
    intent: str = Field(..., description="Detected user intent")
    rationale: str = Field("", description="Why this route was chosen")
    tasks: list[ReasoningTask] = Field(
        default_factory=list,
        description="Tool-first deterministic tasks",
    )


class MemoryEvidenceRequest(BaseModel):
    """Input contract for memory evidence preparation."""

    query: str = Field(..., description="Current user question")
    speaker_id: Optional[str] = Field(None, description="Current speaker identifier")
    ocr_payload: Optional[dict[str, Any]] = Field(
        None,
        description="Optional OCR payload to normalize and cross-check",
    )
    allow_frontier_fallback: bool = Field(
        True,
        description="Whether memory engine may use frontier LLM search fallback",
    )


class MemoryArtifactRef(BaseModel):
    """Minimal pointer to a selected artifact in memory retrieval."""

    category: str = Field(..., description="Logical category or source")
    path: Optional[str] = Field(None, description="Optional file path in MD store")
    reason: str = Field("", description="Why this artifact was selected")
    score: float = Field(0.0, description="Heuristic relevance score")


class MemoryEvidenceBundle(BaseModel):
    """Normalized memory output consumed by reasoning/conversation engines."""

    normalized_query: str = Field(..., description="Canonical query for retrieval")
    normalized_medications: list[str] = Field(
        default_factory=list,
        description="OCR-normalized medication candidates",
    )
    dur_searchable: bool = Field(
        False,
        description="Whether deterministic DUR search is likely feasible",
    )
    used_frontier_fallback: bool = Field(
        False,
        description="Whether memory fallback delegated to frontier search",
    )
    frontier_answer_preview: str = Field(
        "",
        description="Optional frontier fallback answer snippet",
    )
    artifact_refs: list[MemoryArtifactRef] = Field(
        default_factory=list,
        description="Selected minimal artifacts to read/use downstream",
    )
    summary: str = Field("", description="Short memory summary for downstream use")
    memory_prompt: str = Field(
        "",
        description="Structured-memory prompt block for LLM stages",
    )


class ConversationComposeRequest(BaseModel):
    """Input contract for user-facing response composition."""

    input_text: str = Field(..., description="Original user text")
    user_profile: dict[str, Any] = Field(
        default_factory=dict,
        description="Profile fields from memory engine",
    )
    decision: ReasoningRouteDecision = Field(..., description="Reasoning route output")
    evidence: Optional[MemoryEvidenceBundle] = Field(
        None,
        description="Memory evidence prepared for this turn",
    )
    core_message: str = Field("", description="Core factual response candidate")
    reviewed_message: str = Field("", description="Post-judge reviewed candidate")
    delivery_message: str = Field("", description="Post-local-delivery candidate")


class ConversationComposeResponse(BaseModel):
    """Final conversation output passed to transport layer."""

    response_text: str = Field(..., description="Final user-facing response text")
    response_type: str = Field(..., description="smalltalk/medical_response/fallback")
    requires_tts: bool = Field(True, description="Whether TTS playback is needed")


class EngineTraceEvent(BaseModel):
    """Structured engine-stage call trace event."""

    stage: str = Field(..., description="Scenario-facing stage name such as CE_Input")
    component: str = Field(..., description="Owning component")
    action: str = Field(..., description="Concrete action or method")
    status: str = Field("observed", description="observed/skipped/error")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryTraceEvent(BaseModel):
    """Structured memory read/write trace event."""

    operation: str = Field(..., description="read/write/search/update")
    logical_file: str = Field(..., description="Scenario-facing logical file name")
    category: str = Field("", description="MD store category or flash key")
    path: Optional[str] = Field(None, description="Concrete path when known")
    status: str = Field("observed", description="observed/skipped/error")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolTraceEvent(BaseModel):
    """Structured external/deterministic tool call trace event."""

    tool_id: str = Field(..., description="Scenario-facing tool id such as T2")
    tool_name: str = Field(..., description="Concrete task/tool name")
    external_api: Optional[str] = Field(None, description="External API family")
    status: str = Field("observed", description="observed/skipped/error")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnginePipelineResult(BaseModel):
    """Unified per-turn trace used by WS/HTTP orchestrators."""

    input_data: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    identity_gate: dict[str, Any] = Field(default_factory=dict)
    decision: ReasoningRouteDecision
    evidence: MemoryEvidenceBundle
    execution_results: dict[str, Any] = Field(default_factory=dict)
    filler_text: str = ""
    core_message: str = ""
    judge_review: dict[str, Any] = Field(default_factory=dict)
    reviewed_message: str = ""
    delivery_message: str = ""
    conversation: ConversationComposeResponse
    engine_trace: list[EngineTraceEvent] = Field(default_factory=list)
    memory_trace: list[MemoryTraceEvent] = Field(default_factory=list)
    tool_trace: list[ToolTraceEvent] = Field(default_factory=list)
