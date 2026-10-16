"""Codex transport tests: synthetic executables only; no model/API calls."""
from __future__ import annotations

import copy
import json
import sys
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolArg, tool

from cli.prompts import _llm_provider_table, ensure_api_key
from tradingagents.agents import schemas
from tradingagents.agents.structured import invoke_structured
from tradingagents.default_config import DEFAULT_CONFIG, build_default_config
from tradingagents.llm_clients import create_tier_client
from tradingagents.llm_clients.codex_cli_client import (
    CodexCLIChatModel,
    CodexCLIError,
    _child_env,
    _json_loads,
    _strict_schema,
)
from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options

pytestmark = pytest.mark.unit


@tool
def quote(symbol: str) -> str:
    """Get a synthetic quote."""
    return f"{symbol}: 100"


def scripted(monkeypatch, response, usage=None):
    seen = []

    def run(self, prompt, schema):
        seen.append((prompt, schema))
        return response, usage

    monkeypatch.setattr(CodexCLIChatModel, "_run_codex", run)
    return CodexCLIChatModel(), seen


def fake_cli(tmp_path, body=None, login="ChatGPT", flags=True):
    path = tmp_path / "codex-test"
    default = '''
response = {"content": "Synthetic report", "tool_calls": []}
Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text(json.dumps(response))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":2}}))
'''
    source = f'''#!{sys.executable}
import json, os, sys, time
from pathlib import Path
from typing import Annotated
if "--help" in sys.argv:
    print({('--ignore-user-config --ephemeral --output-schema --output-last-message --json' if flags else '--help')!r})
    sys.exit(0)
if "login" in sys.argv:
    print({('Logged in using ' + login)!r})
    sys.exit(0)
{body or default}
'''
    path.write_text(source)
    path.chmod(0o700)
    return CodexCLIChatModel(command=str(path), timeout=2)


def test_factory_and_both_tiers_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(llm_provider="codex_cli", quick_think_llm="codex-default",
               deep_think_llm="codex-default")
    for tier in ("quick", "deep"):
        assert isinstance(create_tier_client(cfg, tier).get_llm(), CodexCLIChatModel)
    assert ensure_api_key("codex_cli") is None
    assert any(p == "codex_cli" and url is None for _, p, url in _llm_provider_table())
    assert get_model_options("codex_cli", "quick")[0][1] == "codex-default"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_CODEX_TIMEOUT", "42")
    monkeypatch.setenv("TRADINGAGENTS_CODEX_COMMAND", "/safe/codex")
    cfg = build_default_config()
    assert cfg["codex_timeout"] == 42
    assert cfg["codex_command"] == "/safe/codex"


@pytest.mark.parametrize("key", ["temperature", "max_tokens", "llm_max_retries"])
def test_reject_api_options(key):
    with pytest.raises(ValueError, match="does not support"):
        build_llm_kwargs({"llm_provider": "codex_cli", key: 1})


def test_reject_endpoint_and_key():
    with pytest.raises(ValueError, match="backend URL"):
        create_llm_client("codex_cli", "codex-default", base_url="http://example.invalid").get_llm()
    with pytest.raises(ValueError, match="API keys"):
        create_llm_client("codex_cli", "codex-default", api_key="synthetic").get_llm()


def test_environment_is_allowlisted(monkeypatch):
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CHATGPT_BASE_URL",
                "CUSTOMER_SECRET", "BASH_ENV", "NODE_OPTIONS"):
        monkeypatch.setenv(key, "must-not-reach-child")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("CODEX_HOME", "/login-location")
    env = _child_env()
    assert "must-not-reach-child" not in env.values()
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert env["CODEX_HOME"] == "/login-location"


def test_plain_prompt_and_usage(monkeypatch):
    model, seen = scripted(monkeypatch, {"content": "证据不足", "tool_calls": []},
                           {"input_tokens": 12, "output_tokens": 4})
    reply = (ChatPromptTemplate.from_messages([("human", "{question}")]) | model).invoke({"question": "测试"})
    assert reply.content == "证据不足"
    assert reply.usage_metadata["total_tokens"] == 16
    assert reply.response_metadata["authentication"] == "chatgpt"
    assert "测试" in seen[0][0]
    assert seen[0][1]["properties"]["tool_calls"]["maxItems"] == 0


def test_missing_usage_is_not_fake_zero(monkeypatch):
    model, _ = scripted(monkeypatch, {"content": "report", "tool_calls": []})
    assert model.invoke("test").usage_metadata is None


def test_host_tool_roundtrip(monkeypatch):
    model, seen = scripted(monkeypatch, {"content": "", "tool_calls": [
        {"name": "quote", "arguments": '{"symbol":"TEST"}'}]})
    reply = model.bind_tools([quote]).invoke("Quote TEST")
    call = reply.tool_calls[0]
    result = quote.invoke(call)
    assert result.content == "TEST: 100"
    assert result.tool_call_id == call["id"]
    messages = [HumanMessage(content="Quote TEST"), reply, result]
    model.bind_tools([quote]).invoke(messages)
    prompt = json.loads(seen[-1][0].split("\n\n", 1)[1])
    assert prompt["conversation"][-1]["tool_call_id"] == call["id"]
    assert prompt["conversation"][-2]["tool_calls"][0]["args"] == {"symbol": "TEST"}


@pytest.mark.parametrize("call", [
    {"name": "shell", "arguments": "{}"},
    {"name": "quote", "arguments": "[]"},
    {"name": "quote", "arguments": "{}"},
    {"name": "quote", "arguments": '{"symbol":"TEST","extra":"bad"}'},
    {"name": "quote", "arguments": '{"symbol":"TEST","symbol":"OTHER"}'},
])
def test_reject_invalid_tools(monkeypatch, call):
    model, _ = scripted(monkeypatch, {"content": "", "tool_calls": [call]})
    with pytest.raises(CodexCLIError, match="invalid_tool_response"):
        model.bind_tools([quote]).invoke("test")


def test_no_tools_means_no_tool_execution(monkeypatch):
    model, _ = scripted(monkeypatch, {"content": "", "tool_calls": [
        {"name": "quote", "arguments": '{"symbol":"TEST"}'}]})
    with pytest.raises(CodexCLIError):
        model.bind_tools([quote], tool_choice="none").invoke("test")


def test_structured_pydantic_and_include_raw(monkeypatch):
    data = {"action": "Hold", "reasoning": "Synthetic data only", "entry_price": None,
            "stop_loss": None, "position_sizing": None}
    model, seen = scripted(monkeypatch, data)
    decision = model.with_structured_output(schemas.TraderProposal).invoke("test")
    assert decision.action == schemas.TraderAction.HOLD
    assert "entry_price" in seen[0][1]["required"]
    raw = model.with_structured_output(schemas.TraderProposal, include_raw=True).invoke("test")
    assert raw["parsed"] == decision and raw["parsing_error"] is None
    assert isinstance(raw["raw"], AIMessage)


def test_invalid_decision_never_falls_back(monkeypatch):
    model, seen = scripted(monkeypatch, {"action": "Invented", "reasoning": "x"})
    structured = model.with_structured_output(schemas.TraderProposal)
    with pytest.raises(CodexCLIError, match="invalid_structured_output"):
        invoke_structured(structured, "test", "Trader")
    assert len(seen) == 1


def test_strict_schema_does_not_mutate_source():
    source = schemas.TraderProposal.model_json_schema()
    strict = _strict_schema(source)
    assert "additionalProperties" not in source
    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])
    assert "default" not in strict["properties"]["entry_price"]


def test_reject_multimodal_and_stop():
    model = CodexCLIChatModel()
    with pytest.raises(CodexCLIError, match="text messages only"):
        model.invoke([HumanMessage(content=[{"type": "image_url", "image_url": "invalid"}])])
    with pytest.raises(ValueError, match="stop sequences"):
        model.invoke("test", stop=["END"])


def test_preflight_does_not_generate(tmp_path):
    model = fake_cli(tmp_path, body='raise AssertionError("must not generate")')
    assert model.check_available()["authentication"] == "chatgpt"


@pytest.mark.parametrize("login", ["API key sk-fake-not-a-secret", "another method"])
def test_non_subscription_login_is_refused_and_redacted(tmp_path, login):
    model = fake_cli(tmp_path, body='raise AssertionError("must not generate")', login=login)
    with pytest.raises(CodexCLIError, match="chatgpt_login_required") as exc:
        model.invoke("test")
    assert "sk-fake" not in str(exc.value)


def test_old_cli_is_refused(tmp_path):
    model = fake_cli(tmp_path, flags=False)
    with pytest.raises(CodexCLIError, match="unsupported_cli"):
        model.invoke("test")


def test_missing_executable():
    with pytest.raises(CodexCLIError, match="command_not_found"):
        CodexCLIChatModel(command="/not/a/real/codex").invoke("test")


def test_real_subprocess_stdin_final_file_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward")
    body = '''
assert "OPENAI_API_KEY" not in os.environ
assert "--ignore-user-config" in sys.argv and "--ephemeral" in sys.argv
assert sys.argv[sys.argv.index("--sandbox")+1] == "read-only"
assert 'forced_login_method="chatgpt"' in sys.argv
assert 'features.shell_tool=false' in sys.argv
assert 'features.apps=false' in sys.argv
assert "sensitive prompt" not in " ".join(sys.argv)
assert "sensitive prompt" in sys.stdin.read()
assert Path.cwd() != Path(__file__).parent
out = Path(sys.argv[sys.argv.index("--output-last-message")+1])
out.write_text(json.dumps({"content":"file result", "tool_calls":[]}))
print(json.dumps({"type":"item.completed", "item":{"type":"agent_message", "text":"not the result"}}))
print(json.dumps({"type":"turn.completed"}))
'''
    assert fake_cli(tmp_path, body).invoke("sensitive prompt").content == "file result"


@pytest.mark.parametrize("body,code", [
    ('sys.exit(9)', "process_failed"),
    ('time.sleep(5)', "timeout"),
    ('print("x"*2000000)', "output_too_large"),
    ('print("not JSON")', "invalid_output"),
    ('print(json.dumps({"type":"turn.failed","error":{"message":"fake-secret"}}))', "turn_failed"),
    ('print(json.dumps({"type":"item.completed","item":{"type":"command_execution"}}))', "native_tool_detected"),
])
def test_subprocess_fail_closed(tmp_path, body, code):
    model = fake_cli(tmp_path, body)
    model.timeout = 1
    with pytest.raises(CodexCLIError, match=code) as exc:
        model.invoke("test")
    assert "fake-secret" not in str(exc.value)


def test_json_rejects_nonfinite():
    with pytest.raises(ValueError):
        _json_loads('{"value":NaN}')


def test_host_injected_argument_cannot_be_overridden(monkeypatch):
    @tool
    def scoped_quote(window: int, symbol: Annotated[str, InjectedToolArg]) -> str:
        """Read a quote for the host-selected symbol."""
        return f"{symbol}: {window}"

    model, _ = scripted(monkeypatch, {"content": "", "tool_calls": [
        {"name": "scoped_quote", "arguments": '{"window":1,"symbol":"OVERRIDE"}'}]})
    with pytest.raises(CodexCLIError, match="invalid_tool_response"):
        model.bind_tools([scoped_quote]).invoke("test")
