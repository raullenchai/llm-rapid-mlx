import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path / "llm"))
    for name in ("RAPID_MLX_API_KEY", "RAPID_MLX_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "never-send-this")


@pytest.fixture
def server(monkeypatch):
    state = SimpleNamespace(
        requests=[],
        responses=[],
        catalogue=[
            {"id": "chat", "capabilities": ["text", "tools"]},
            {"id": "embed", "capabilities": ["embedding"]},
        ],
        status=200,
        post_status=200,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state.requests.append(
                (self.path, {k.lower(): v for k, v in self.headers.items()}, None)
            )
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"object": "list", "data": state.catalogue}).encode()
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(
                (self.path, {k.lower(): v for k, v in self.headers.items()}, body)
            )
            if state.post_status != 200:
                self.send_response(state.post_status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"secret-prompt secret-key"}}')
                return
            if not state.responses:
                self.send_error(500, "No mocked response queued")
                return
            response = state.responses.pop(0)
            streaming = isinstance(response, list)
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.end_headers()
            if streaming:
                for chunk in response:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps(response).encode())

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_port}/v1"
    monkeypatch.setenv("RAPID_MLX_BASE_URL", state.url)
    try:
        yield state
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()
