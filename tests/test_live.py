"""Opt-in smoke test; never launches or stops a server."""

import os

import pytest

import llm_rapid_mlx as plugin


@pytest.mark.live
def test_live_stream_and_conversation(monkeypatch):
    url = os.environ.get("RAPID_MLX_LIVE_URL")
    if not url:
        pytest.skip("set RAPID_MLX_LIVE_URL to use an existing server")
    monkeypatch.setenv("RAPID_MLX_BASE_URL", url)
    base_url, records = plugin.discover_models()
    server_id, caps = next((name, caps) for name, caps in records if "text" in caps)
    conversation = plugin.RapidChat(server_id, base_url, caps).conversation()
    response = conversation.prompt(
        "Remember the word pelican. Say hi.", system="Answer briefly.", max_tokens=128
    )
    chunks = list(response)
    assert chunks and response.text().strip()
    assert (
        "pelican"
        in conversation.prompt("What word did I ask you to remember?", max_tokens=128)
        .text()
        .lower()
    )


@pytest.mark.live
def test_live_embedding_if_advertised(monkeypatch):
    url = os.environ.get("RAPID_MLX_LIVE_URL")
    if not url:
        pytest.skip("set RAPID_MLX_LIVE_URL to use an existing server")
    monkeypatch.setenv("RAPID_MLX_BASE_URL", url)
    base_url, records = plugin.discover_models()
    embedding = next((name for name, caps in records if "embedding" in caps), None)
    if embedding is None:
        pytest.skip("server advertises no embedding model")
    vector = plugin.RapidEmbedding(embedding, base_url).embed("A pelican")
    assert vector and all(isinstance(value, float) for value in vector)
