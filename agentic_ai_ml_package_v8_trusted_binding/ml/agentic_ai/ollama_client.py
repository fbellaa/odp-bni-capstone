from __future__ import annotations

import json
from typing import Any, Sequence, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .config import SETTINGS, Settings

T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Small HTTP client that keeps tool calls and structured outputs explicit."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or SETTINGS
        self.host = self.s.ollama_host.rstrip("/")
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def models(self) -> list[str]:
        try:
            r = self.session.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Tidak bisa menghubungi Ollama di {self.host}: {exc}") from exc
        return [m.get("name", "") for m in r.json().get("models", [])]

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        format_schema: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if self.s.ollama_keep_alive:
            payload["keep_alive"] = self.s.ollama_keep_alive
        if tools:
            payload["tools"] = list(tools)
        if format_schema is not None:
            payload["format"] = format_schema

        try:
            r = self.session.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.s.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise OllamaError(f"Permintaan ke Ollama gagal: {exc}") from exc

        if r.status_code == 404:
            raise OllamaError(f"Model {model!r} tidak ditemukan di Ollama.")
        if r.status_code >= 400:
            raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:1000]}")
        return r.json()["message"]

    def structured(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        temperature: float = 0.0,
        retries: int = 2,
    ) -> T:
        conversation = list(messages)
        last_error: Exception | None = None
        json_schema = schema.model_json_schema()

        for _ in range(max(1, retries)):
            msg = self.chat(
                model=model,
                messages=conversation,
                temperature=temperature,
                format_schema=json_schema,
            )
            raw = msg.get("content", "")
            try:
                return schema.model_validate_json(raw)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                conversation.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Output JSON tidak valid terhadap schema. Perbaiki tanpa "
                                f"menambah fakta baru. Validation error: {exc}"
                            ),
                        },
                    ]
                )

        raise OllamaError(
            f"Model {model} gagal menghasilkan {schema.__name__} yang valid: {last_error}"
        )
