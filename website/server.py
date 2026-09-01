"""Local PocketSQL playground server.

Serves the static website and exposes a small same-origin inference endpoint.
The checkpoint is loaded once per server process; requests only tokenize,
generate, and validate a query against the supplied schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pocketsql.inference import generate_sql, load_model_from_checkpoint  # noqa: E402
from pocketsql.model.tokenizer import load_tokenizer  # noqa: E402


DEFAULT_SCHEMA = """
CREATE TABLE customers (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT,
  created_at TEXT
);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  created_at TEXT,
  total_amount REAL,
  status TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  stock_quantity INTEGER,
  price REAL,
  category TEXT,
  created_at TEXT
);
""".strip()

CHECKPOINT = Path(os.environ.get(
    "POCKETSQL_CHECKPOINT",
    PROJECT_ROOT / "checkpoints/base-semantic-v14-factorized-best-execution",
))


class PocketSQLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/generate":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 32_000:
                raise ValueError("Request is too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
            schema = str(payload.get("schema", DEFAULT_SCHEMA)).strip() or DEFAULT_SCHEMA
            if not question:
                raise ValueError("Enter a question first")
            started = time.perf_counter()
            sql = generate_sql(MODEL, schema, question, TOKENIZER)
            latency_ms = round((time.perf_counter() - started) * 1000)
            self._json(200, {"sql": sql, "latencyMs": latency_ms, "model": CHECKPOINT.name})
        except Exception as error:  # return a useful UI error without a traceback
            self._json(422, {"error": str(error)})

    def log_message(self, format: str, *args) -> None:
        if self.path.startswith("/api/"):
            super().log_message(format, *args)


print(f"Loading PocketSQL checkpoint: {CHECKPOINT}", flush=True)
TOKENIZER = load_tokenizer(CHECKPOINT)
MODEL = load_model_from_checkpoint(str(CHECKPOINT), TOKENIZER)
print("PocketSQL model ready", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "4173"))
    # MLX binds its default stream to the thread that initializes the model.
    # A synchronous local server keeps inference on that same thread.
    server = HTTPServer(("127.0.0.1", port), PocketSQLHandler)
    print(f"PocketSQL playground: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
