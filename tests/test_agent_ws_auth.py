"""WebSocket auth guard tests."""
from __future__ import annotations

import pytest

from app.api.routes import agent_ws


class DummyWebSocket:
    def __init__(self, headers: dict[str, str] | None = None, query: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self.query_params = query or {}


def test_websocket_auth_disabled_allows_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_ws.settings, "websocket_auth_token", "")
    ws = DummyWebSocket()
    assert agent_ws._is_websocket_authorized(ws) is True


def test_websocket_auth_with_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_ws.settings, "websocket_auth_token", "demo-token")
    ws = DummyWebSocket(headers={"authorization": "Bearer demo-token"})
    assert agent_ws._is_websocket_authorized(ws) is True


def test_websocket_auth_with_query_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_ws.settings, "websocket_auth_token", "demo-token")
    ws = DummyWebSocket(query={"token": "demo-token"})
    assert agent_ws._is_websocket_authorized(ws) is True


def test_websocket_auth_rejects_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_ws.settings, "websocket_auth_token", "demo-token")
    ws = DummyWebSocket(headers={"authorization": "Bearer wrong-token"})
    assert agent_ws._is_websocket_authorized(ws) is False

