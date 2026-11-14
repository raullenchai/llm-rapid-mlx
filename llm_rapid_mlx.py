"""LLM integration for the Rapid-MLX OpenAI-compatible API."""

from __future__ import annotations

import copy
import math
import os
from urllib.parse import urlsplit

import click
import llm
from llm.default_plugins.openai_models import AsyncChat, Chat
from openai import AsyncOpenAI, OpenAI, OpenAIError
from pydantic import Field, field_validator

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Rapid-MLX base URL has an invalid port") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port == 0
    ):
        raise ValueError(
            "Rapid-MLX base URL must be HTTP(S), without credentials, query or fragment"
        )
    value = value.rstrip("/")
    return value if parsed.path.rstrip("/").endswith("/v1") else value + "/v1"


def configured_base_url():
    # Key storage also gives CLI users a persistent endpoint setting.
    return normalize_base_url(
        os.environ.get("RAPID_MLX_BASE_URL")
        or llm.get_key(key_alias="rapid-mlx-base-url")
        or DEFAULT_BASE_URL
    )


def resolve_key(explicit_key=None):
    return llm.get_key(
        explicit_key=explicit_key, key_alias="rapid-mlx", env_var="RAPID_MLX_API_KEY"
    )


def client_kwargs(base_url, key=None, timeout=120):
    # Always provide a key so OPENAI_API_KEY cannot leak to a local endpoint.
    return {
        "base_url": base_url,
        "api_key": key or "not-needed",
        "timeout": timeout,
        "max_retries": 0,
    }


def model_error(error):
    status = getattr(error, "status_code", None)
    detail = f"HTTP {status}" if status else type(error).__name__
    return llm.ModelError(
        f"Rapid-MLX request failed ({detail}); check the server and configuration"
    )


def discover_models(base_url=None):
    """Fetch a fresh catalogue; never infer embeddings from a model's name."""
    base_url = normalize_base_url(base_url) if base_url else configured_base_url()
    with OpenAI(**client_kwargs(base_url, resolve_key(), timeout=2)) as client:
        records = client.models.list().data
    seen = set()
    result = []
    for record in records:
        if not isinstance(record.id, str) or not record.id or record.id in seen:
            continue
        caps = getattr(record, "capabilities", None)
        if caps is not None and (
            not isinstance(caps, list) or not all(isinstance(c, str) for c in caps)
        ):
            continue
        seen.add(record.id)
        # Older servers may omit capabilities: those entries are text only.
        caps = ["text"] if caps is None else caps
        result.append((record.id, caps))
    return base_url, result


def _catalogue():
    try:
        return discover_models()
    except (OpenAIError, ValueError, TypeError, AttributeError, OSError):
        return None, []


class Options(Chat.Options):
    base_url: str | None = Field(
        default=None, description="Override the Rapid-MLX API base URL for this prompt"
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value):
        return normalize_base_url(value) if value is not None else None


class _RapidMixin:
    needs_key = "rapid-mlx"
    key_env_var = "RAPID_MLX_API_KEY"
    Options = Options

    def __init__(self, server_id, base_url=None, capabilities=None):
        caps = ["text"] if capabilities is None else capabilities
        super().__init__(
            model_id="rapid-mlx/" + server_id,
            model_name=server_id,
            api_base=base_url or configured_base_url(),
            supports_schema=True,
            supports_tools="tools" in caps,
            vision="vision" in caps,
        )
        # The base also accepts PDFs for OpenAI; Rapid-MLX accepts images here.
        self.attachment_types.discard("application/pdf")

    def get_key(self, explicit_key=None):
        return self.key if self.key is not None else resolve_key(explicit_key)

    def get_client(self, key, *, async_=False):
        factory = AsyncOpenAI if async_ else OpenAI
        return factory(**client_kwargs(self.api_base, self.get_key(key)))

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("base_url", None)
        return kwargs

    def for_prompt(self, prompt):
        # An override must not change a shared registered model/conversation.
        model = copy.copy(self)
        model.api_base = prompt.options.base_url or self.api_base
        return model

    def __str__(self):
        return f"Rapid-MLX: {self.model_id}"


class RapidChat(_RapidMixin, Chat):
    def execute(self, prompt, stream, response, conversation=None, key=None):
        model = self.for_prompt(prompt)
        try:
            with model.get_client(key) as client:
                # Let the base adapter own message, stream, usage and tool handling.
                model.get_client = lambda *args, **kwargs: client
                yield from Chat.execute(
                    model, prompt, stream, response, conversation, key
                )
        except OpenAIError as ex:
            raise model_error(ex) from None


class RapidAsyncChat(_RapidMixin, AsyncChat):
    async def execute(self, prompt, stream, response, conversation=None, key=None):
        model = self.for_prompt(prompt)
        try:
            async with model.get_client(key, async_=True) as client:
                model.get_client = lambda *args, **kwargs: client
                async for chunk in AsyncChat.execute(
                    model, prompt, stream, response, conversation, key
                ):
                    yield chunk
        except OpenAIError as ex:
            raise model_error(ex) from None


class RapidEmbedding(llm.EmbeddingModel):
    needs_key = "rapid-mlx"
    key_env_var = "RAPID_MLX_API_KEY"
    supports_text = True
    supports_binary = False
    batch_size = 32

    def __init__(self, server_id, base_url=None):
        self.model_id = "rapid-mlx/" + server_id
        self.server_id = server_id
        self.base_url = base_url or configured_base_url()

    def get_key(self, explicit_key=None):
        return self.key if self.key is not None else resolve_key(explicit_key)

    def embed_batch(self, items, *, key=None):
        items = list(items)
        if any(not isinstance(item, str) for item in items):
            raise ValueError("Rapid-MLX embeddings accept text strings only")
        if not items:
            return
        try:
            with OpenAI(**client_kwargs(self.base_url, self.get_key(key))) as client:
                data = client.embeddings.create(
                    model=self.server_id, input=items, encoding_format="float"
                ).data
        except OpenAIError as ex:
            raise model_error(ex) from None
        ordered = sorted(data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(items))):
            raise llm.ModelError("Rapid-MLX returned invalid embedding indexes")
        vectors = [[float(value) for value in item.embedding] for item in ordered]
        if any(
            not vector or any(not math.isfinite(v) for v in vector)
            for vector in vectors
        ):
            raise llm.ModelError("Rapid-MLX returned invalid embedding vectors")
        yield from vectors


@llm.hookimpl
def register_models(register):
    base_url, records = _catalogue()
    for server_id, caps in records:
        if "embedding" not in caps and "text" in caps:
            register(
                RapidChat(server_id, base_url, caps),
                RapidAsyncChat(server_id, base_url, caps),
            )


@llm.hookimpl
def register_embedding_models(register):
    base_url, records = _catalogue()
    for server_id, caps in records:
        if "embedding" in caps:
            register(RapidEmbedding(server_id, base_url))


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="rapid-mlx")
    def rapid_mlx():
        "Commands for Rapid-MLX."

    @rapid_mlx.command(name="models")
    def models():
        "Discover server models, reporting connection or authentication errors."
        try:
            _, records = discover_models()
        except (OpenAIError, ValueError, TypeError, AttributeError, OSError) as ex:
            status = getattr(ex, "status_code", None)
            detail = f"HTTP {status}" if status else type(ex).__name__
            raise click.ClickException(
                f"Rapid-MLX discovery failed ({detail}); check RAPID_MLX_BASE_URL and the rapid-mlx key"
            ) from None
        for server_id, caps in records:
            click.echo(f"rapid-mlx/{server_id}\t{', '.join(caps)}")
