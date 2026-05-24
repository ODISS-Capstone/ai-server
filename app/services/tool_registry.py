"""LLM tool-calling registry for data.go.kr-backed tools.

Maps the 12 tool names documented in 패치노트.md / 기능 사양서 to the async
Python functions already implemented under app.tools.*.

Tool schema source:
    [app/prompts/llm_tools.json](app/prompts/llm_tools.json)

References:
- 의약품 낱알식별: https://www.data.go.kr/data/15057639/openapi.do
- 식약처 DUR:      https://www.data.go.kr/data/15059486/openapi.do
- 건강기능식품:    https://www.data.go.kr/data/15056760/openapi.do
"""
from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.llm_queue import run_with_engine_queue
from app.tools import dur_api, health_supplement, hira_api

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


def _queue_engine_for_tool(tool_name: str) -> str:
    if tool_name.startswith(("Tool_Check_DUR_", "Tool_Get_DUR_")):
        return "dur"
    return "tool"


def _dur_handler(endpoint_key: str) -> ToolHandler:
    """Build a handler bound to a specific DUR endpoint key."""

    async def _call(**kwargs: Any) -> dict[str, Any]:
        return await dur_api.call_dur_api(endpoint_key, **kwargs)

    _call.__name__ = f"call_dur_api__{endpoint_key}"
    return _call


DEFAULT_TOOL_HANDLERS: dict[str, ToolHandler] = {
    "Tool_Get_Drug_Identification": hira_api.identify_medicine,
    "Tool_Check_DUR_Combination_Contraindication": _dur_handler(
        "combination_contraindication"
    ),
    "Tool_Check_DUR_Geriatric_Caution": _dur_handler("elderly_caution"),
    "Tool_Get_DUR_Basic_Item_Info": _dur_handler("dur_product_info"),
    "Tool_Check_DUR_Age_Specific_Contraindication": _dur_handler(
        "age_contraindication"
    ),
    "Tool_Check_DUR_Dosage_Caution": _dur_handler("dosage_caution"),
    "Tool_Check_DUR_Duration_Caution": _dur_handler("period_caution"),
    "Tool_Check_DUR_Duplicate_Therapeutic_Class": _dur_handler("efficacy_overlap"),
    "Tool_Check_DUR_Sustained_Release_Caution": _dur_handler("sr_tablet_caution"),
    "Tool_Check_DUR_Pregnancy_Contraindication": _dur_handler(
        "pregnancy_contraindication"
    ),
    "Tool_Get_Health_Supplement_Detail": health_supplement.get_supplement_detail,
    "Tool_Search_Health_Supplement_List": health_supplement.list_supplements,
}


class ToolRegistry:
    """Load tool schemas from JSON and dispatch calls to async handlers."""

    def __init__(
        self,
        schema_path: str | Path | None = None,
        handlers: dict[str, ToolHandler] | None = None,
    ) -> None:
        self.schema_path = Path(schema_path or self._default_schema_path())
        self.handlers: dict[str, ToolHandler] = dict(
            handlers if handlers is not None else DEFAULT_TOOL_HANDLERS
        )
        self.schemas: list[dict[str, Any]] = self._load_schemas()
        self._schema_properties_by_name = self._index_schema_properties()

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for the LLM `tools` field."""
        return [dict(schema) for schema in self.schemas]

    def get_tool_names(self) -> list[str]:
        return [
            schema["function"]["name"]
            for schema in self.schemas
            if "function" in schema and "name" in schema["function"]
        ]

    async def dispatch(
        self, tool_name: str, arguments: dict[str, Any] | str | None
    ) -> dict[str, Any]:
        """Execute a tool call and return its raw result.

        `arguments` may be a dict or the JSON string produced by OpenAI-style
        `tool_calls[i].function.arguments`.
        """
        handler = self.handlers.get(tool_name)
        if handler is None:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}",
                "items": [],
            }

        parsed = self._normalize_argument_aliases(self._coerce_arguments(arguments))
        parsed = self._filter_schema_properties(tool_name, parsed)
        filtered = self._filter_supported_kwargs(handler, parsed)
        engine = _queue_engine_for_tool(tool_name)

        async def _invoke() -> dict[str, Any]:
            return await handler(**filtered)

        try:
            return await run_with_engine_queue(engine, _invoke)
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc)
            return {
                "success": False,
                "message": f"Tool {tool_name} raised: {exc}",
                "items": [],
            }

    @staticmethod
    def _default_schema_path() -> Path:
        configured = getattr(settings, "llm_tools_path", None)
        if configured:
            return Path(configured)
        return Path(__file__).resolve().parents[1] / "prompts" / "llm_tools.json"

    def _load_schemas(self) -> list[dict[str, Any]]:
        if not self.schema_path.exists():
            logger.info("Tool schema file not found: %s", self.schema_path)
            return []

        try:
            with self.schema_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load tool schema %s: %s", self.schema_path, exc)
            return []

        tools = data.get("tools", [])
        if not isinstance(tools, list):
            logger.warning("Tool schema %s has no 'tools' list", self.schema_path)
            return []

        valid: list[dict[str, Any]] = []
        for tool in tools:
            if self._is_valid_tool_schema(tool):
                valid.append(tool)
            else:
                logger.warning("Ignoring invalid tool entry in %s", self.schema_path)
        return valid

    def _index_schema_properties(self) -> dict[str, set[str]]:
        indexed: dict[str, set[str]] = {}
        for schema in self.schemas:
            fn = schema.get("function", {}) or {}
            name = fn.get("name")
            parameters = fn.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            if isinstance(name, str) and isinstance(properties, dict):
                indexed[name] = {str(key) for key in properties}
        return indexed

    @staticmethod
    def _is_valid_tool_schema(tool: Any) -> bool:
        if not isinstance(tool, dict):
            return False
        if tool.get("type") != "function":
            return False
        fn = tool.get("function")
        return isinstance(fn, dict) and isinstance(fn.get("name"), str)

    @staticmethod
    def _coerce_arguments(
        arguments: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if not stripped:
                return {}
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Tool arguments must be JSON-decodable: {exc}"
                ) from exc
            if not isinstance(decoded, dict):
                raise ValueError("Tool arguments JSON must decode to an object")
            return decoded
        raise TypeError(f"Unsupported tool arguments type: {type(arguments)!r}")

    @staticmethod
    def _normalize_argument_aliases(arguments: dict[str, Any]) -> dict[str, Any]:
        aliases = {
            "itemName": "item_name",
            "itemSeq": "item_seq",
            "pageNo": "page_no",
            "numOfRows": "num_of_rows",
            "productName": "product_name",
            "entpName": "entp_name",
            "printFront": "print_front",
            "printBack": "print_back",
            "drugShape": "drug_shape",
            "colorClass1": "color_class1",
            "imgRegistTs": "img_regist_ts",
        }
        normalized = dict(arguments)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        return normalized

    def _filter_schema_properties(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = self._schema_properties_by_name.get(tool_name)
        if not allowed:
            return arguments
        return {key: value for key, value in arguments.items() if key in allowed}

    @staticmethod
    def _filter_supported_kwargs(
        handler: ToolHandler, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            return arguments

        accepts_var_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return arguments

        allowed = {
            name
            for name, param in signature.parameters.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return {key: value for key, value in arguments.items() if key in allowed}


@lru_cache
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()
