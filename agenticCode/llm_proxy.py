#!/usr/bin/env python3
"""
llm_proxy.py - transparent logging proxy between goose and the OpenAI-compatible endpoint.

Writes one JSON line per LLM request to $LLM_LOG (default /app/output/llm_calls.jsonl):
tokens (from the server's `usage`), latency, time to first token, finish_reason,
request size, whether the request looks like a context-summarisation call, errors.

Streaming responses are passed through chunk by chunk. For streams, usage only
arrives if the request asks for it, so `stream_options.include_usage` is injected
(disable with LLM_INJECT_USAGE=0 if the server rejects it).

Env:
  LLM_UPSTREAM_URL   full chat-completions URL of the real endpoint (required)
  LLM_PROXY_PORT     default 8089
  LLM_LOG            default /app/output/llm_calls.jsonl
  LLM_HOUR_FILE      file holding the current replay hour (written by entrypoint.sh)
  LLM_INJECT_USAGE   default 1
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx

UPSTREAM = os.environ["LLM_UPSTREAM_URL"]
PORT = int(os.getenv("LLM_PROXY_PORT", "8089"))
LOG = os.getenv("LLM_LOG", "/app/output/llm_calls.jsonl")
HOUR_FILE = os.getenv("LLM_HOUR_FILE", "/app/output/current_hour.txt")
INJECT_USAGE = os.getenv("LLM_INJECT_USAGE", "1") != "0"
RETRY_STATUS = {429, 503}


def backoff(resp, attempt: int) -> float:
    try:
        ra = float(resp.headers.get("retry-after", ""))
    except ValueError:
        ra = 0.0
    return min(max(ra, 2.0 * (attempt + 1)), 30.0)


class _Held:
    """Keep a streaming response open for a `with` block and close it afterwards."""
    def __init__(self, cm, r):
        self.cm, self.r = cm, r

    def __enter__(self):
        return self.r

    def __exit__(self, *exc):
        return self.cm.__exit__(*exc)

MAX_RPM = int(os.getenv("LLM_MAX_RPM", "0"))              # 0 = no limit (requests per rolling minute)
MAX_CONC = int(os.getenv("LLM_MAX_CONCURRENT", "0"))      # 0 = no limit (requests in flight)
RETRY_429 = int(os.getenv("LLM_RETRY_429", "10"))         # retries on 429/503 (with backoff) before any output reaches goose
RETRIES = RETRY_429
_CONC = threading.Semaphore(MAX_CONC) if MAX_CONC > 0 else None
_RPM_LOCK = threading.Lock()
_RPM_STAMPS: list[float] = []


def throttle() -> float:
    """Block until a request may be sent. Returns seconds waited."""
    t0 = time.time()
    if _CONC is not None:
        _CONC.acquire()
    if MAX_RPM > 0:
        while True:
            with _RPM_LOCK:
                now = time.time()
                while _RPM_STAMPS and now - _RPM_STAMPS[0] >= 60:
                    _RPM_STAMPS.pop(0)
                if len(_RPM_STAMPS) < MAX_RPM:
                    _RPM_STAMPS.append(now)
                    break
                wait = 60 - (now - _RPM_STAMPS[0]) + 0.05
            time.sleep(max(wait, 0.05))
    return time.time() - t0


def release() -> None:
    if _CONC is not None:
        _CONC.release()


_u = urlsplit(UPSTREAM)
ORIGIN = f"{_u.scheme}://{_u.netloc}"
CLIENT = httpx.Client(timeout=httpx.Timeout(900, connect=30))
LOCK = threading.Lock()
SKIP_RESP_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}


def current_hour() -> str | None:
    try:
        return open(HOUR_FILE).read().strip() or None
    except OSError:
        return None


def write_log(rec: dict) -> None:
    line = json.dumps(rec, default=str)
    with LOCK:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def request_facts(body: dict) -> dict:
    msgs = body.get("messages") or []
    last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
    system = next((m for m in msgs if m.get("role") == "system"), {})
    probe = (_text(last_user.get("content")) + " " + _text(system.get("content"))).lower()
    has_tools = bool(body.get("tools"))
    return {
        "model": body.get("model"),
        "stream": bool(body.get("stream")),
        "n_messages": len(msgs),
        "req_chars": len(json.dumps(msgs)) + len(json.dumps(body.get("tools") or [])),
        "has_tools": has_tools,
        "max_tokens_requested": body.get("max_tokens") or body.get("max_completion_tokens"),
        # heuristic: goose's summarise strategy sends a request without tools asking for a summary
        "summarize_hint": (not has_tools) and ("summar" in probe),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # body ends when the connection closes

    def log_message(self, *a):  # silence default access log
        pass

    def _target(self) -> str:
        return UPSTREAM if self.path.split("?")[0].endswith("/chat/completions") else ORIGIN + self.path

    def _fwd_headers(self) -> dict:
        h = {k: v for k, v in self.headers.items()
             if k.lower() in ("authorization", "content-type", "accept", "x-api-key")}
        h.setdefault("Content-Type", "application/json")
        return h

    def do_GET(self):
        try:
            r = CLIENT.get(self._target(), headers=self._fwd_headers())
            self._send_full(r.status_code, r.headers.get("content-type", "application/json"), r.content)
        except Exception as e:  # noqa: BLE001
            self._send_full(502, "text/plain", str(e).encode())

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        t0 = time.time()
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "hour": current_hour(),
               "usage_source": "missing", "status": None, "error": None}
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            rec.update(request_facts(body))
            if body.get("stream") and INJECT_USAGE:
                body.setdefault("stream_options", {})["include_usage"] = True
            raw = json.dumps(body).encode()
        waited = throttle()
        rec["queue_wait_s"] = round(waited, 3)
        t0 = time.time()  # latency excludes the time spent waiting for a slot
        try:
            try:
                if rec.get("stream"):
                    self._stream(raw, rec, t0)
                else:
                    self._plain(raw, rec, t0)
            finally:
                release()   # frees the concurrency slot taken in throttle(), also on errors
        except Exception as e:  # noqa: BLE001  (timeouts, connection resets, broken pipe to goose)
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["latency_s"] = round(time.time() - t0, 3)
            if rec["status"] is None:
                try:
                    self._send_full(502, "text/plain", rec["error"].encode())
                except Exception:  # noqa: BLE001
                    pass
            write_log(rec)

    # ---- non-streaming ----
    def _plain(self, raw: bytes, rec: dict, t0: float) -> None:
        for attempt in range(RETRIES + 1):
            r = CLIENT.post(self._target(), content=raw, headers=self._fwd_headers())
            if r.status_code in RETRY_STATUS and attempt < RETRIES:
                rec["retries"] = attempt + 1
                time.sleep(backoff(r, attempt))
                continue
            break
        rec["status"] = r.status_code
        rec["latency_s"] = round(time.time() - t0, 3)
        try:
            data = r.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            self._take_usage(rec, data.get("usage"))
            ch = (data.get("choices") or [{}])[0]
            rec["finish_reason"] = ch.get("finish_reason")
            rec["n_tool_calls"] = len((ch.get("message") or {}).get("tool_calls") or [])
        if r.status_code >= 400:
            rec["error"] = r.text[:300]
        self._send_full(r.status_code, r.headers.get("content-type", "application/json"), r.content)
        write_log(rec)

    # ---- streaming ----
    def _stream(self, raw: bytes, rec: dict, t0: float) -> None:
        for attempt in range(RETRIES + 1):
            cm = CLIENT.stream("POST", self._target(), content=raw, headers=self._fwd_headers())
            r = cm.__enter__()
            if r.status_code in RETRY_STATUS and attempt < RETRIES:
                r.read()
                cm.__exit__(None, None, None)
                rec["retries"] = attempt + 1
                time.sleep(backoff(r, attempt))
                continue
            break
        with _Held(cm, r) as r:
            rec["status"] = r.status_code
            self.send_response(r.status_code)
            for k, v in r.headers.items():
                if k.lower() not in SKIP_RESP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            buf, tool_idx, err_body = b"", set(), b""
            for chunk in r.iter_raw():
                if "ttft_s" not in rec:
                    rec["ttft_s"] = round(time.time() - t0, 3)
                self.wfile.write(chunk)
                self.wfile.flush()
                if r.status_code >= 400:
                    err_body += chunk
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    self._take_usage(rec, ev.get("usage"))
                    for ch in ev.get("choices") or []:
                        if ch.get("finish_reason"):
                            rec["finish_reason"] = ch["finish_reason"]
                        for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                            tool_idx.add(tc.get("index", len(tool_idx)))
            rec["n_tool_calls"] = len(tool_idx)
            if r.status_code >= 400:
                rec["error"] = err_body[:300].decode(errors="ignore")
        rec["latency_s"] = round(time.time() - t0, 3)
        write_log(rec)

    @staticmethod
    def _take_usage(rec: dict, usage) -> None:
        if not isinstance(usage, dict) or usage.get("prompt_tokens") is None:
            return
        rec["usage_source"] = "server"
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["total_tokens"] = usage.get("total_tokens")
        det = usage.get("prompt_tokens_details") or {}
        if det.get("cached_tokens") is not None:
            rec["cached_tokens"] = det["cached_tokens"]

    def _send_full(self, status: int, ctype: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    print(f"[llm_proxy] 127.0.0.1:{PORT} -> {UPSTREAM}, log {LOG}", file=sys.stderr, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
