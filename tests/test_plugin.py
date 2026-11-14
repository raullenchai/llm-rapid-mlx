import json

import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli

import llm_rapid_mlx as plugin


def completion(text="Hi", tool=None):
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [tool]
    return {
        "id": "reply",
        "object": "chat.completion",
        "model": "chat",
        "created": 0,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool else "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def stream(deltas, finish="stop"):
    return [
        {
            "id": "reply",
            "object": "chat.completion.chunk",
            "model": "chat",
            "created": 0,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ] + [
        {
            "id": "reply",
            "object": "chat.completion.chunk",
            "model": "chat",
            "created": 0,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        },
        {
            "id": "reply",
            "object": "chat.completion.chunk",
            "model": "chat",
            "created": 0,
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    ]


def test_discovery_registration_and_cli(server):
    server.catalogue += [
        {"id": "sounds", "capabilities": ["audio.speech"]},
        {"id": "legacy-embed-in-name"},
        {"id": "vision", "capabilities": ["text", "vision"]},
        {"id": "chat", "capabilities": ["text"]},
        {"id": "invalid", "capabilities": "embedding"},
    ]
    registered = []
    plugin.register_models(lambda *models: registered.append(models))
    assert [pair[0].model_id for pair in registered] == [
        "rapid-mlx/chat",
        "rapid-mlx/legacy-embed-in-name",
        "rapid-mlx/vision",
    ]
    assert isinstance(registered[0][1], plugin.RapidAsyncChat)
    assert registered[0][0].supports_tools
    assert not registered[1][0].supports_tools
    assert "image/png" in registered[2][0].attachment_types
    assert "application/pdf" not in registered[2][0].attachment_types
    embeddings = []
    plugin.register_embedding_models(embeddings.append)
    assert [model.model_id for model in embeddings] == ["rapid-mlx/embed"]
    result = CliRunner().invoke(cli, ["models", "-q", "rapid-mlx"])
    assert result.exit_code == 0, result.output
    assert "rapid-mlx/chat" in result.output
    assert "rapid-mlx/embed" not in result.output
    assert server.requests[0][0] == "/v1/models"
    assert server.requests[0][1]["authorization"] == "Bearer not-needed"


def test_no_embedding_without_advertisement(server):
    server.catalogue = [{"id": "embedding-looking-name", "capabilities": ["text"]}]
    registered = []
    plugin.register_embedding_models(registered.append)
    assert registered == []


@pytest.mark.parametrize("status", [401, 500])
def test_discovery_failure_is_nonfatal_and_diagnostic(server, status):
    server.status = status
    registered = []
    plugin.register_models(lambda *models: registered.extend(models))
    plugin.register_embedding_models(registered.append)
    assert registered == []
    result = CliRunner().invoke(cli, ["rapid-mlx", "models"])
    assert result.exit_code == 1
    assert f"HTTP {status}" in result.output
    assert "never-send-this" not in result.output


def test_unreachable_and_malformed_discovery(monkeypatch, server):
    monkeypatch.setenv("RAPID_MLX_BASE_URL", "http://127.0.0.1:0")
    assert plugin._catalogue() == (None, [])
    monkeypatch.setenv("RAPID_MLX_BASE_URL", server.url)
    server.catalogue = None
    assert plugin._catalogue() == (None, [])


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://localhost:8000", "http://localhost:8000/v1"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1"),
        ("https://example.org/proxy", "https://example.org/proxy/v1"),
    ],
)
def test_url_normalization(value, expected):
    assert plugin.normalize_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "ftp://host",
        "http://",
        "http://u:p@host",
        "http://host?key=x",
        "http://host#x",
        "http://host:no",
    ],
)
def test_invalid_urls(value):
    with pytest.raises(ValueError):
        plugin.normalize_base_url(value)


def test_key_and_endpoint_precedence(server, monkeypatch):
    user_dir = llm.user_dir()
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "keys.json").write_text(
        json.dumps({"rapid-mlx": "stored", "rapid-mlx-base-url": "http://stored:8000"})
    )
    monkeypatch.setenv("RAPID_MLX_API_KEY", "environment")
    assert plugin.resolve_key() == "stored"
    assert plugin.resolve_key("explicit") == "explicit"
    assert plugin.configured_base_url() == server.url
    monkeypatch.delenv("RAPID_MLX_BASE_URL")
    assert plugin.configured_base_url() == "http://stored:8000/v1"
    (user_dir / "keys.json").unlink()
    assert plugin.resolve_key() == "environment"
    assert plugin.configured_base_url() == plugin.DEFAULT_BASE_URL
    model = plugin.RapidChat("chat", server.url)
    model.key = "python"
    assert model.get_key("explicit") == "python"


def test_stream_conversation_system_and_usage(server):
    server.responses = [
        stream([{"content": "H"}, {"content": "i"}]),
        completion("pelican"),
    ]
    model = plugin.RapidChat("chat")
    conversation = model.conversation()
    first = conversation.prompt("Remember pelican", system="Be brief")
    assert list(first) == ["H", "i"]
    assert first.usage().input == 3
    assert conversation.prompt("What word?", stream=False).text() == "pelican"
    first_body = server.requests[0][2]
    assert first_body["stream"] is True
    assert first_body["model"] == "chat"
    assert first_body["stream_options"] == {"include_usage": True}
    assert server.requests[1][2]["messages"] == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Remember pelican"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "What word?"},
    ]


def test_schema_endpoint_option_and_key(server):
    server.responses = [completion('{"greeting":"hi"}')]
    model = plugin.RapidChat("chat", "http://127.0.0.1:1/v1")
    schema = {"type": "object", "properties": {"greeting": {"type": "string"}}}
    assert json.loads(
        model.prompt(
            "Hi",
            schema=schema,
            stream=False,
            base_url=server.url,
            temperature=0.2,
            key="secret",
        ).text()
    ) == {"greeting": "hi"}
    path, headers, body = server.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer secret"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["temperature"] == 0.2
    assert "base_url" not in body
    assert model.api_base == "http://127.0.0.1:1/v1"


@pytest.mark.parametrize("streaming", [False, True])
def test_tool_loop(server, streaming):
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    call = {
        "id": "call1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a":17,"b":25}'},
    }
    if streaming:
        server.responses = [
            stream(
                [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a":17,'},
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"b":25}'}}
                        ]
                    },
                ],
                "tool_calls",
            ),
            stream([{"content": "42"}]),
        ]
    else:
        server.responses = [completion(None, call), completion("42")]
    model = plugin.RapidChat("chat", capabilities=["text", "tools"])
    chain = model.chain("Add 17 and 25", tools=[add], stream=streaming)
    assert chain.text() == "42"
    body = server.requests[1][2]
    assert body["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call1",
        "content": "42",
    }
    assert body["messages"][-2]["tool_calls"][0]["id"] == "call1"
    assert server.requests[0][2]["tools"][0]["function"]["name"] == "add"


@pytest.mark.parametrize("streaming", [False, True])
async def test_async(server, streaming):
    server.responses = [
        stream([{"content": "Hello"}]) if streaming else completion("Hello")
    ]
    model = plugin.RapidAsyncChat("chat")
    assert await model.prompt("Hi", stream=streaming).text() == "Hello"


def test_embeddings_order_and_auth(server, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "embed-key")
    server.responses = [
        {
            "object": "list",
            "model": "embed",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [3, 4]},
                {"object": "embedding", "index": 0, "embedding": [1, 2]},
            ],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    ]
    model = plugin.RapidEmbedding("embed")
    assert list(model.embed_multi(iter(["one", "two"]))) == [[1.0, 2.0], [3.0, 4.0]]
    path, headers, body = server.requests[0]
    assert path == "/v1/embeddings"
    assert headers["authorization"] == "Bearer embed-key"
    assert body == {
        "model": "embed",
        "input": ["one", "two"],
        "encoding_format": "float",
    }
    with pytest.raises(ValueError, match="text strings"):
        list(model.embed_batch([b"binary"]))


def test_invalid_embedding_indexes(server):
    server.responses = [
        {
            "object": "list",
            "model": "embed",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [1, 2]},
            ],
        }
    ]
    with pytest.raises(llm.ModelError, match="indexes"):
        plugin.RapidEmbedding("embed").embed("one")


@pytest.mark.parametrize("kind", ["chat", "async", "embedding"])
async def test_api_errors_are_actionable_without_echoing_secrets(server, kind):
    server.post_status = 401
    with pytest.raises(llm.ModelError, match="HTTP 401") as error:
        if kind == "chat":
            plugin.RapidChat("chat").prompt(
                "secret-prompt", key="secret-key", stream=False
            ).text()
        elif kind == "async":
            await (
                plugin.RapidAsyncChat("chat")
                .prompt("secret-prompt", key="secret-key", stream=False)
                .text()
            )
        else:
            plugin.RapidEmbedding("embed").embed("secret-prompt", key="secret-key")
    assert "secret" not in str(error.value)
    assert len(server.requests) == 1  # no hidden SDK retries


def test_discovery_uses_stored_authentication(server):
    (llm.user_dir() / "keys.json").write_text(
        json.dumps({"rapid-mlx": "stored-secret"})
    )
    plugin.discover_models()
    assert server.requests[0][1]["authorization"] == "Bearer stored-secret"
