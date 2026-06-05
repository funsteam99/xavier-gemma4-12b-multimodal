#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path
import json
import os
import time

ROOT = Path(__file__).resolve().parent / "public"
BACKEND = os.environ.get("GEMMA_BACKEND", "http://127.0.0.1:18084").rstrip("/")
PORT = int(os.environ.get("PORT", "18090"))
APP_TITLE = os.environ.get("APP_TITLE", "Gemma 4 E4B Test Console")
MODEL_HINT = os.environ.get("MODEL_HINT", "gemma-4-E4B-it-Q4_K_M.gguf")
MAX_BODY = 96 * 1024 * 1024

MIMES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

class Handler(BaseHTTPRequestHandler):
    server_version = "Gemma4Console/1.1"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), fmt % args), flush=True)

    def _headers(self, code, content_type="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._headers(code)
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _backend(self, method, path, body=None, timeout=240):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = Request(BACKEND + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        started = time.time()
        try:
            with urlopen(req, timeout=timeout) as res:
                text = res.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"text": text}
                return res.status, {"ok": 200 <= res.status < 300, "latency_ms": round((time.time() - started) * 1000), "backend": BACKEND, "data": payload}
        except HTTPError as err:
            text = err.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"text": text}
            return err.code, {"ok": False, "latency_ms": round((time.time() - started) * 1000), "backend": BACKEND, "data": payload}
        except URLError as err:
            return 502, {"ok": False, "latency_ms": round((time.time() - started) * 1000), "backend": BACKEND, "error": str(err)}

    def do_OPTIONS(self):
        self._headers(204)

    def do_POST(self):
        try:
            if self.path == "/api/health":
                code, payload = self._backend("GET", "/health", timeout=10)
                self._json(code, payload)
            elif self.path == "/api/config":
                self._json(200, {"ok": True, "app_title": APP_TITLE, "model_hint": MODEL_HINT})
            elif self.path == "/api/models":
                code, payload = self._backend("GET", "/v1/models", timeout=20)
                self._json(code, payload)
            elif self.path == "/api/chat":
                body = self._read_json()
                code, payload = self._backend("POST", "/v1/chat/completions", body.get("payload", {}), timeout=360)
                self._json(code, payload)
            else:
                self._json(404, {"ok": False, "error": "unknown endpoint"})
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc)})

    def do_GET(self):
        rel = self.path.split("?", 1)[0].strip("/") or "index.html"
        if rel.startswith("api/") or ".." in rel:
            self._json(404, {"ok": False, "error": "not found"})
            return
        path = ROOT / rel
        if not path.exists() or path.is_dir():
            path = ROOT / "index.html"
        ctype = MIMES.get(path.suffix.lower(), "application/octet-stream")
        data = path.read_bytes()
        self._headers(200, ctype)
        self.wfile.write(data)

if __name__ == "__main__":
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"{APP_TITLE} on 0.0.0.0:{PORT}, backend={BACKEND}", flush=True)
    httpd.serve_forever()
