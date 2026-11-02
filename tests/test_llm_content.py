"""Model reply text across providers (Anthropic content blocks vs OpenAI strings).

ChatAnthropic returns ``AIMessage.content`` as a list of blocks when the reply
has a thinking block, a tool call or split text. Callers that treated
``.content`` as a string broke on Anthropic only, e.g. repository analysis
raising "the JSON object must be str, bytes or bytearray, not list".
"""

import json
import os
import sys

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plugin_implementation.llm_content import message_text, model_temperature  # noqa: E402

ANALYSIS = {"executive_summary": "Polyglot samples.", "tech_stack": ["Python", "TypeScript"], "capabilities": []}
ANALYSIS_JSON = json.dumps(ANALYSIS, indent=2)


def test_openai_string_content_is_returned_unchanged():
    message = AIMessage(content=ANALYSIS_JSON)
    assert message_text(message) == ANALYSIS_JSON
    assert json.loads(message_text(message)) == ANALYSIS


def test_anthropic_thinking_then_text_keeps_only_the_answer():
    message = AIMessage(content=[
        {"type": "thinking", "thinking": "Let me look at the files first.", "signature": "sig"},
        {"type": "text", "text": ANALYSIS_JSON},
    ])
    text = message_text(message)
    assert "Let me look" not in text
    assert json.loads(text) == ANALYSIS


def test_anthropic_json_split_across_text_blocks_is_joined_without_separators():
    half = len(ANALYSIS_JSON) // 2
    message = AIMessage(content=[
        {"type": "text", "text": ANALYSIS_JSON[:half]},
        {"type": "text", "text": ANALYSIS_JSON[half:]},
    ])
    assert json.loads(message_text(message)) == ANALYSIS


def test_redacted_thinking_and_tool_use_blocks_are_dropped():
    message = AIMessage(content=[
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "final answer"},
        {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
    ])
    assert message_text(message) == "final answer"


def test_tool_call_only_reply_has_no_text():
    message = AIMessage(content=[{"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}])
    assert message_text(message) == ""


@pytest.mark.parametrize("value, expected", [
    (None, ""),
    ("", ""),
    ([], ""),
    (["plain", " strings"], "plain strings"),
    ({"type": "text", "text": "single block"}, "single block"),
    ([{"text": "untyped block"}], "untyped block"),
])
def test_raw_content_values(value, expected):
    assert message_text(value) == expected


@pytest.mark.parametrize("structured", [True, False])
def test_repository_analysis_returns_text_for_anthropic_blocks(monkeypatch, structured):
    """The failing call site: _llm_analyze_repository validated the reply with json.loads."""
    from unittest.mock import MagicMock

    from plugin_implementation.agents import wiki_graph_optimized as module

    monkeypatch.setattr(module, "USE_STRUCTURED_REPO_ANALYSIS", structured)
    agent = module.OptimizedWikiGenerationAgent.__new__(module.OptimizedWikiGenerationAgent)
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = AIMessage(content=[
        {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
        {"type": "text", "text": ANALYSIS_JSON},
    ])
    agent.repository_url = "org/project/repo"
    agent.branch = "master"

    result = agent._llm_analyze_repository("tree", "readme", "samples", ["a.py"], config={})

    assert isinstance(result, str)
    assert "reasoning" not in result
    assert json.loads(result) == ANALYSIS


# ── Temperature: send only what the platform configured (Bedrock GPT-5 rejects it) ──

@pytest.mark.parametrize("settings, expected", [
    ({}, None),
    (None, None),
    ({"temperature": None}, None),
    ({"temperature": 0.3}, 0.3),
    ({"temperature": 0}, 0.0),
    ({"temperature": "0.7"}, 0.7),
    ({"temperature": "warm"}, None),
    ({"temperature": True}, None),
])
def test_model_temperature(settings, expected):
    assert model_temperature(settings) == expected


def _captured_body(build_llm):
    """Invoke a ChatOpenAI built by DeepWiki code and return the JSON body sent."""
    import httpx

    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured["body"] = body
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m",'
                    b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )
        return httpx.Response(200, json={
            "id": "1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
        })

    llm = build_llm()
    llm.client._client._client = httpx.Client(transport=httpx.MockTransport(handler))
    llm.invoke("hi")
    return captured["body"]


BEDROCK_SETTINGS = {
    "provider": "openai",
    "api_base": "http://platform/llm/v1",
    "api_key": "key",
    "model_name": "global.openai.gpt-5.6-luna",
    "organization": "406",
    "streaming": True,
}


@pytest.mark.parametrize("worker_module", [
    "plugin_implementation.wiki_subprocess_worker",
    "plugin_implementation.ask_subprocess_worker",
    "plugin_implementation.deep_research_subprocess_worker",
])
def test_workers_send_no_temperature_unless_configured(worker_module):
    import importlib

    module = importlib.import_module(worker_module)
    build = lambda settings: module._build_llm_and_embeddings(settings, None)[0]

    body = _captured_body(lambda: build(dict(BEDROCK_SETTINGS)))
    assert "temperature" not in body
    assert body["model"] == "global.openai.gpt-5.6-luna"

    configured = _captured_body(lambda: build({**BEDROCK_SETTINGS, "temperature": 0.2}))
    assert configured["temperature"] == 0.2


def test_resolve_wiki_llm_sends_no_temperature_unless_configured():
    from test_invoke_repo_config import _load_invoke_module

    invoke = _load_invoke_module()
    kwargs = dict(provider="openai", model_name="global.openai.gpt-5.6-luna", api_key="key",
                  api_base="http://platform/llm/v1", organization="406")

    body = _captured_body(lambda: invoke.create_llm(**kwargs))
    assert "temperature" not in body

    configured = _captured_body(lambda: invoke.create_llm(**kwargs, temperature=0.4))
    assert configured["temperature"] == 0.4
