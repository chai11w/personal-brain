from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

from .config import ChatModelConfig, EmbeddingModelConfig


_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_MIN_REQUEST_INTERVAL_SECONDS = 1.5


class LLMClient:
    def __init__(self, config: ChatModelConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return self.config.enabled and bool(get_secret_env(self.config.api_key_env))

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str | None:
        if not self.available:
            return None
        if self.config.provider != "openai_compatible":
            raise ValueError(f"unsupported llm provider: {self.config.provider}")

        api_key = get_secret_env(self.config.api_key_env)
        if not api_key:
            return None
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = request_json_with_retries(
            request,
            timeout_seconds=self.config.timeout_seconds,
            error_prefix="model",
        )

        message = data["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part for part in parts if part).strip()
        return str(content or "").strip()


class EmbeddingClient:
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return self.config.enabled and bool(get_secret_env(self.config.api_key_env))

    def embed(self, text: str) -> list[float] | None:
        vectors = self.embed_many([text])
        if not vectors:
            return None
        return vectors[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.available:
            return []
        if self.config.provider != "openai_compatible":
            raise ValueError(f"unsupported embedding provider: {self.config.provider}")

        api_key = get_secret_env(self.config.api_key_env)
        if not api_key:
            return []
        endpoint = self.config.base_url.rstrip("/") + "/embeddings"
        payload = {
            "model": self.config.model,
            "input": texts,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = request_json_with_retries(
            request,
            timeout_seconds=self.config.timeout_seconds,
            error_prefix="embedding",
        )

        vectors_by_index: dict[int, list[float]] = {}
        for item in data.get("data", []):
            index = int(item.get("index", len(vectors_by_index)))
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError("embedding response item has no vector")
            vectors_by_index[index] = [float(value) for value in vector]
        return [vectors_by_index[index] for index in range(len(texts))]


def get_secret_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    text = str(value).strip()
    return text or None


def request_json_with_retries(
    request: urllib.request.Request,
    timeout_seconds: int,
    error_prefix: str,
    attempts: int = 3,
) -> dict:
    retry_delays = [5, 15]
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with _REQUEST_LOCK:
                wait_for_request_spacing()
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"{error_prefix} HTTP error {exc.code}: {detail}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = RuntimeError(f"{error_prefix} call failed: {exc}")
            if attempt == attempts:
                raise last_error from exc
        time.sleep(retry_delays[min(attempt - 1, len(retry_delays) - 1)])
    if last_error:
        raise last_error
    raise RuntimeError(f"{error_prefix} call failed")


def wait_for_request_spacing() -> None:
    global _LAST_REQUEST_AT
    now = time.monotonic()
    wait_seconds = _MIN_REQUEST_INTERVAL_SECONDS - (now - _LAST_REQUEST_AT)
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    _LAST_REQUEST_AT = time.monotonic()
