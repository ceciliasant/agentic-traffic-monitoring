"""Test setup for mcp_server.py.

mcp_server.py talks to MLflow, S3, Ditto and an auth service at import time or in tool calls.
The tests replace those with stubs so they run offline, in seconds, without Docker.
Run from the folder that contains mcp_server.py:   pytest -q
"""
import asyncio
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# environment read at import time
os.environ.setdefault("GOOSE_MODEL", "test-model")
os.environ.setdefault("VCI_HISTORIC_DITTO_URL", "http://ditto.test/")
os.environ.setdefault("VCI_DITTO_URL", "http://ditto-live.test/")
os.environ["ANOMALY_THRESHOLD"] = "0.20"
os.environ["TRACE_FILE"] = str(Path(os.environ.get("TMPDIR", "/tmp")) / "pytest_trace.jsonl")

# stub the modules that need a network, credentials or heavy dependencies
for name in ("mlflow", "boto3", "joblib", "xgboost"):
    sys.modules.setdefault(name, MagicMock())
auth = types.ModuleType("APItoken")
class AuthenticationService:                       
    def get_token(self):
        return "test-token"
auth.AuthenticationService = AuthenticationService
sys.modules["APItoken"] = auth

import mcp_server  
from fastmcp import Client  


@pytest.fixture
def srv():
    return mcp_server


@pytest.fixture
def call():
    """call("tool_name", arg=...) -> structured result, through the in-process MCP client (as classic_runner does)."""
    def _call(tool, **args):
        async def run():
            async with Client(mcp_server.mcp) as c:
                r = await c.call_tool(tool, args)
                sc = r.structured_content
                if isinstance(sc, dict) and set(sc) == {"result"}:      # plain return values are wrapped by FastMCP
                    return sc["result"]
                return sc if sc is not None else r.data
        return asyncio.run(run())
    return _call


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_http(monkeypatch):
    """Replace httpx.AsyncClient inside mcp_server. Set fake_http.get / fake_http.post to FakeResp or a function(url, params)."""
    state = types.SimpleNamespace(get=FakeResp([]), post=FakeResp({}), posted=[], gets=[])

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False
        async def get(self, url, headers=None, params=None):
            state.gets.append((url, params))
            return state.get(url, params) if callable(state.get) else state.get
        async def post(self, url, headers=None, json=None):
            state.posted.append((url, json))
            return state.post(url, json) if callable(state.post) else state.post

    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", FakeClient)
    return state
