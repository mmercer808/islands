"""
AssistChannel — streaming LLM chat, Qt-free.
Assimilated from proc_streamer_v1_6.AssistChannel; uses callbacks so UI can wrap with Qt signals.
"""

from __future__ import annotations
import json
import threading
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import requests
except Exception:
    requests = None


# Callback types: UI can assign these to emit Qt signals
OnChunk = Callable[[str], None]
OnComplete = Callable[[str], None]
OnError = Callable[[str], None]
OnStatus = Callable[[bool, str], None]


class AssistChannel:
    """
    Streaming LLM channel. No Qt dependency.
    Set on_chunk, on_complete, on_error, on_status (callbacks) before calling query().
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._connected = False
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        # Callbacks (assign from UI to e.g. emit Qt signals)
        self.on_chunk: Optional[OnChunk] = None
        self.on_complete: Optional[OnComplete] = None
        self.on_error: Optional[OnError] = None
        self.on_status: Optional[OnStatus] = None

    def connect(self, provider: Optional[str] = None, url: Optional[str] = None,
                model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        if provider is not None:
            self.config["provider"] = provider
        if url is not None:
            self.config["url"] = url
        if model is not None:
            self.config["model"] = model
        if api_key is not None:
            self.config["api_key"] = api_key
        ok, msg = self._healthcheck()
        self._connected = ok
        if self.on_status:
            self.on_status(ok, msg)

    def disconnect(self) -> None:
        self._connected = False
        if self.on_status:
            self.on_status(False, "Disconnected")

    def query(self, prompt: str) -> None:
        if not self._connected:
            if self.on_error:
                self.on_error("Not connected")
            return
        if requests is None:
            if self.on_error:
                self.on_error("`requests` not installed")
            return
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run_query, args=(prompt,), daemon=True)
        self._thread.start()

    def _healthcheck(self) -> Tuple[bool, str]:
        if requests is None:
            return False, "Install `requests` to connect"
        provider = (self.config.get("provider") or "ollama").lower()
        if provider == "ollama":
            try:
                url = (self.config.get("url") or "").rstrip("/") + "/api/tags"
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json() or {}
                models = [m.get("name") for m in data.get("models", [])]
                m = self.config.get("model")
                if m and models:
                    if m not in models:
                        partial = any(
                            (model or "").startswith(m + ":") or model == m
                            for model in models
                        )
                        if not partial:
                            return False, (
                                f"Ollama OK, but model '{m}' not found. "
                                f"Installed: {', '.join(models) or 'none'}. Try: `ollama pull {m}`."
                            )
                return True, f"Connected to Ollama @ {self.config.get('url', '')} ({len(models)} models)"
            except Exception as e:
                return False, f"Ollama connect failed: {e}"
        else:
            try:
                url = (self.config.get("url") or "").rstrip("/") + "/v1/models"
                headers = {}
                if self.config.get("api_key"):
                    headers["Authorization"] = f"Bearer {self.config['api_key']}"
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                return True, f"Connected to OpenAI-compatible @ {self.config.get('url', '')}"
            except Exception as e:
                return False, f"OpenAI-compat connect failed: {e}"

    def _run_query(self, prompt: str) -> None:
        provider = (self.config.get("provider") or "ollama").lower()
        try:
            if provider == "ollama":
                self._query_ollama(prompt)
            else:
                self._query_openai_compat(prompt)
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))

    def _query_ollama(self, prompt: str) -> None:
        url = (self.config.get("url") or "").rstrip("/") + "/api/generate"
        model = self.config.get("model", "llama3:latest")
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        timeout = self.config.get("timeout", 60)
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            full: list[str] = []
            for line in r.iter_lines(decode_unicode=True):
                if self._stop_flag:
                    break
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        tok = obj["response"]
                        full.append(tok)
                        if self.on_chunk:
                            self.on_chunk(tok)
                    if obj.get("done"):
                        break
                except Exception:
                    full.append(line)
                    if self.on_chunk:
                        self.on_chunk(line)
            if self.on_complete:
                self.on_complete("".join(full))

    def _query_openai_compat(self, prompt: str) -> None:
        url = (self.config.get("url") or "").rstrip("/") + "/v1/chat/completions"
        model = self.config.get("model", "")
        headers = {}
        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        data: Dict[str, Any] = {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
        timeout = self.config.get("timeout", 60)
        with requests.post(url, headers=headers, json=data, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            full: list[str] = []
            for line in r.iter_lines(decode_unicode=True):
                if self._stop_flag:
                    break
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        full.append(delta)
                        if self.on_chunk:
                            self.on_chunk(delta)
                except Exception:
                    pass
            if self.on_complete:
                self.on_complete("".join(full))
