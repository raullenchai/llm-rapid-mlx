# llm-rapid-mlx

A plugin for [LLM](https://llm.datasette.io/) to use models served by
[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), a local inference server
for Apple Silicon built on MLX. The plugin is a Python HTTP client; only the
server requires Apple Silicon.

## Installation

Requires Python 3.10+ and LLM 0.37.2. From this checkout:

```sh
llm install -e .
```

Or install the locally built wheel with `llm install dist/llm_rapid_mlx-0.1.0-py3-none-any.whl`.
This package has not yet been published to PyPI.

Install and start Rapid-MLX separately (the server requires Python 3.12+ if
installed through pip):

```sh
brew install rapid-mlx
rapid-mlx serve qwen3.5-4b-4bit --port 8000
```

The plugin neither starts servers nor downloads model weights.

## Usage

```sh
llm rapid-mlx models
llm models -q rapid-mlx
llm -m rapid-mlx/mlx-community/Qwen3.5-4B-MLX-4bit 'say hi'
llm -m rapid-mlx/mlx-community/Qwen3.5-4B-MLX-4bit -s 'Answer briefly' 'Remember the word pelican'
llm -c 'What word did I ask you to remember?'
```

Use the IDs shown by discovery. They are always `rapid-mlx/<server-id>`;
the prefix is removed before sending the request. Streaming is on by default;
`--no-stream` uses a non-streaming request. Standard LLM conversation logging,
`llm chat`, options such as `-o temperature 0.5` and `-o max_tokens 128`,
and Python sync/async models work through LLM's OpenAI adapter.

```sh
llm -m rapid-mlx/qwen3.5-4b-4bit --functions 'def add(a: int, b: int) -> int: return a + b' 'Use add to calculate 17 + 25'
llm -m rapid-mlx/qwen3.5-4b-4bit --schema '{"type":"object","properties":{"greeting":{"type":"string"}},"required":["greeting"]}' 'Say hello in JSON'
```

Function tools are enabled when discovery advertises `tools`. JSON schemas are
passed through using OpenAI's `response_format`; successful generation and
schema enforcement depend on the model and server. Vision models advertising
`vision` accept PNG, JPEG, WebP and GIF attachments through `-a image.png`.
Audio, video and PDF attachments are outside this plugin's scope.

## Configuration

The default endpoint is `http://localhost:8000/v1`.

```sh
export RAPID_MLX_BASE_URL=http://127.0.0.1:8477/v1
# Persistent alternative (stores a URL, not a secret):
llm keys set rapid-mlx-base-url --value http://127.0.0.1:8477/v1
```

Endpoint precedence: `-o base_url URL` for an individual chat request, then
`RAPID_MLX_BASE_URL`, then the `rapid-mlx-base-url` entry in LLM's key store,
then the default. Discovery and embedding requests use the configured endpoint;
a chat option alone cannot discover models on a different server. Set the
environment variable or persistent URL before listing/selecting those models.
A URL without `/v1` is normalized to include it, including reverse proxy paths.

Authentication is optional. For an authenticated server:

```sh
llm keys set rapid-mlx
# Or: export RAPID_MLX_API_KEY=your-secret
llm -m rapid-mlx/qwen3.5-4b-4bit --key your-key 'Hello'
```

Keys follow LLM precedence: Python `model.key`, explicit `--key` (literal or key
alias), stored `rapid-mlx` key, then `RAPID_MLX_API_KEY`. Discovery uses the stored
key/environment variable, since model discovery runs before a prompt's `--key`.
Without a key the client sends the harmless placeholder `not-needed`, accepted
by Rapid-MLX's default unauthenticated mode. `OPENAI_API_KEY` is never used.

## Discovery and embeddings

Each registration reads `GET /v1/models` with a two-second timeout and no retries.
There is no persistent catalogue cache. Restart an interactive LLM session after
changing the server's model catalogue. Offline, unauthenticated or malformed
catalogues register no Rapid-MLX models, leaving other plugins usable.
`llm rapid-mlx models` reports discovery failures explicitly.

Models tagged `embedding` are registered only as embedding models. Names are
never used to guess embedding support. Entries with `text` are chat models;
older entries without `capabilities` are treated as text-only. Other modalities
are excluded from chat registration.

If the server advertises an embedding model:

```sh
llm embed-models
llm embed -m rapid-mlx/your-discovered-embedding-model -c 'A coastal bird'
```

Embeddings use `/v1/embeddings`, text input and batches of up to 32. No embedding
model is registered when the server only serves chat. Embedding vectors are
returned in input order, with indexes and finite numeric values checked.

## Development

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install llm '.[test]' build twine
.venv/bin/llm install -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

Offline tests use a mocked HTTP server and exercise the real LLM/OpenAI adapters.
Opt in to a small live smoke test against an existing server:

```sh
RAPID_MLX_LIVE_URL=http://127.0.0.1:8477/v1 .venv/bin/python -m pytest -m live -q
```

The optional test does not start or stop a server. Do not set that variable for
offline tests. The dependency on LLM's bundled OpenAI adapter is constrained to
0.37.x because that adapter is not a separately versioned public API.

## License

MIT. See [LICENSE](LICENSE).
