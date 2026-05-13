"""LLM tool registry tests."""
import asyncio
import json
from pathlib import Path

import pytest

from app.services import tool_registry as tool_registry_module
from app.services.tool_registry import DEFAULT_TOOL_HANDLERS, ToolRegistry


PROJECT_SCHEMA = (
    Path(__file__).resolve().parents[1] / "app" / "prompts" / "llm_tools.json"
)

EXPECTED_TOOL_NAMES = {
    "Tool_Get_Drug_Identification",
    "Tool_Check_DUR_Combination_Contraindication",
    "Tool_Check_DUR_Geriatric_Caution",
    "Tool_Get_DUR_Basic_Item_Info",
    "Tool_Check_DUR_Age_Specific_Contraindication",
    "Tool_Check_DUR_Dosage_Caution",
    "Tool_Check_DUR_Duration_Caution",
    "Tool_Check_DUR_Duplicate_Therapeutic_Class",
    "Tool_Check_DUR_Sustained_Release_Caution",
    "Tool_Check_DUR_Pregnancy_Contraindication",
    "Tool_Get_Health_Supplement_Detail",
    "Tool_Search_Health_Supplement_List",
}


def test_project_schema_exposes_twelve_tools():
    registry = ToolRegistry(schema_path=PROJECT_SCHEMA)
    assert set(registry.get_tool_names()) == EXPECTED_TOOL_NAMES
    assert set(DEFAULT_TOOL_HANDLERS.keys()) == EXPECTED_TOOL_NAMES


def test_health_supplement_search_schema_accepts_product_name():
    registry = ToolRegistry(schema_path=PROJECT_SCHEMA)
    schemas = {
        schema["function"]["name"]: schema
        for schema in registry.get_tool_schemas()
    }

    search_schema = schemas["Tool_Search_Health_Supplement_List"]["function"]["parameters"]
    assert "product_name" in search_schema["properties"]


def test_registry_falls_back_to_empty_when_schema_missing(tmp_path):
    registry = ToolRegistry(schema_path=tmp_path / "missing.json")
    assert registry.get_tool_schemas() == []


def test_dispatch_executes_handler_and_filters_unknown_kwargs(tmp_path):
    schema_file = tmp_path / "tools.json"
    schema_file.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Tool_Echo",
                            "description": "Echo tool",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    received: dict[str, object] = {}

    async def echo(value: str = "") -> dict[str, object]:
        received["value"] = value
        return {"success": True, "items": [value]}

    registry = ToolRegistry(
        schema_path=schema_file,
        handlers={"Tool_Echo": echo},
    )

    result = asyncio.run(
        registry.dispatch(
            "Tool_Echo",
            json.dumps({"value": "hello", "unknown_field": 1}),
        )
    )

    assert result == {"success": True, "items": ["hello"]}
    assert received == {"value": "hello"}


def test_dispatch_normalizes_llm_api_argument_aliases_and_schema_filters(tmp_path):
    schema_file = tmp_path / "tools.json"
    schema_file.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Tool_DUR_Test",
                            "description": "DUR test tool",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "item_name": {"type": "string"},
                                    "page_no": {"type": "integer"},
                                },
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    received: dict[str, object] = {}

    async def dur_like_handler(**kwargs) -> dict[str, object]:
        received.update(kwargs)
        return {"success": True, "items": []}

    registry = ToolRegistry(
        schema_path=schema_file,
        handlers={"Tool_DUR_Test": dur_like_handler},
    )

    result = asyncio.run(
        registry.dispatch(
            "Tool_DUR_Test",
            {"itemName": "아스피린", "pageNo": 2, "numOfRows": 100, "unknown": True},
        )
    )

    assert result["success"] is True
    assert received == {"item_name": "아스피린", "page_no": 2}


def test_dispatch_unknown_tool_returns_error_envelope():
    registry = ToolRegistry(schema_path=PROJECT_SCHEMA)
    result = asyncio.run(registry.dispatch("Tool_Does_Not_Exist", {}))
    assert result["success"] is False
    assert "Unknown tool" in result["message"]


def test_dispatch_rejects_non_object_argument_json(tmp_path):
    registry = ToolRegistry(schema_path=tmp_path / "missing.json", handlers={})

    async def handler() -> dict[str, object]:
        return {"success": True, "items": []}

    registry.handlers = {"Tool_Noop": handler}

    with pytest.raises(ValueError):
        asyncio.run(registry.dispatch("Tool_Noop", "[]"))


def test_run_chat_with_tools_loops_until_final_answer(monkeypatch, tmp_path):
    from app.services import llm as llm_module

    schema_file = tmp_path / "tools.json"
    schema_file.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Tool_Sum",
                            "description": "Sum two integers",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    async def sum_handler(a: int, b: int) -> dict[str, object]:
        return {"success": True, "result": a + b}

    registry = ToolRegistry(
        schema_path=schema_file,
        handlers={"Tool_Sum": sum_handler},
    )

    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "Tool_Sum",
                                    "arguments": json.dumps({"a": 2, "b": 3}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {"message": {"role": "assistant", "content": "합은 5입니다."}}
            ]
        },
    ]

    captured_payloads: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, headers=None, json=None):
            captured_payloads.append(json)
            return FakeResponse(responses.pop(0))

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)

    answer = asyncio.run(
        llm_module.run_chat_with_tools(
            messages=[{"role": "user", "content": "2와 3을 더해줘"}],
            api_url="http://fake",
            api_key="fake",
            tool_registry=registry,
            max_tool_rounds=2,
        )
    )

    assert answer == "합은 5입니다."
    assert len(captured_payloads) == 2
    tool_turn = captured_payloads[1]["messages"]
    assert tool_turn[-1]["role"] == "tool"
    assert tool_turn[-1]["tool_call_id"] == "call-1"
    assert json.loads(tool_turn[-1]["content"]) == {"success": True, "result": 5}


def test_get_tool_registry_is_cached():
    tool_registry_module.get_tool_registry.cache_clear()
    first = tool_registry_module.get_tool_registry()
    second = tool_registry_module.get_tool_registry()
    assert first is second
