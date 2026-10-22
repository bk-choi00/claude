"""Gemini API backend for Gemma 4 31B served by Google (``gemma-4-31b-it``).

``POST {base_url}/models/{model}:generateContent`` with the ``x-goog-api-key``
header. Images go first in each user turn as ``inline_data`` parts, followed by
the text. Features whose support for Gemma on the Gemini API is unverified are
tried first and dropped (and remembered) when the API answers HTTP 400 about
them: ``systemInstruction`` (fallback: prepend to the first user turn),
``responseJsonSchema`` (fallback: ``responseMimeType`` only, then prompt only)
and ``thinkingConfig``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional, Sequence

import httpx

from .base import LLMBackend, LLMError, LLMResponse, Message
from .openai_compat import LLMHTTPError, ensure_schema_in_prompt, image_captions, post_json

log = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_GEMMA_MODEL = "gemma-4-31b-it"
JSON_LEVELS = ("json_schema", "json_mime", "prompt")
_BLOCKED = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


class GeminiBackend(LLMBackend):
    """Gemma 4 through the Gemini API ``generateContent`` endpoint."""

    name = "gemini"
    supports_images = True
    supports_json_schema = True

    def __init__(
        self,
        model: str = GEMINI_GEMMA_MODEL,
        api_key: Optional[str] = None,
        base_url: str = GEMINI_BASE_URL,
        timeout: float = 600.0,
        max_retries: int = 2,
        transport: Optional[httpx.BaseTransport] = None,
        *,
        json_mode: str = "json_schema",
        thinking_default: Optional[bool] = None,
        backoff: float = 1.0,
    ):
        model = model[len("models/"):] if model.startswith("models/") else model
        super().__init__(model)
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError("Gemini API key missing: set SHIPPID_LLM_API_KEY (or GEMINI_API_KEY)")
        if json_mode not in JSON_LEVELS:
            json_mode = {"json_object": "json_mime"}.get(json_mode, "json_schema")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.thinking_default = thinking_default
        self.backoff = backoff
        # capability state, downgraded on HTTP 400 and remembered for later calls
        self.json_level = json_mode
        self.system_instruction_ok = True
        self.thinking_ok = True
        self._sleep: Callable[[float], None] = time.sleep
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 30.0)),
                                  transport=transport, headers={"x-goog-api-key": api_key})

    def close(self) -> None:
        self._http.close()

    @property
    def url(self) -> str:
        return f"{self.base_url}/models/{self.model}:generateContent"

    @property
    def structured_output(self) -> str:
        return {"json_schema": "json_schema", "json_mime": "json_object", "prompt": "prompt"}[self.json_level]

    # -- request / response ----------------------------------------------------------
    @staticmethod
    def _parts(msg: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"inline_data": {"mime_type": im.mime, "data": im.b64()}} for im in msg.images()]
        text = "\n".join(image_captions(msg) + ([msg.text()] if msg.text() else []))
        if text:
            parts.append({"text": text})
        return parts

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        thinking: Optional[bool] = None,
    ) -> dict[str, Any]:
        msgs = list(messages)
        if json_schema is not None and self.json_level != "json_schema":
            msgs = ensure_schema_in_prompt(msgs, json_schema)
        system = "\n\n".join(m.text() for m in msgs if m.role == "system" and m.text())
        contents: list[dict[str, Any]] = []
        for m in msgs:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            parts = self._parts(m)
            if not parts:
                continue
            if contents and contents[-1]["role"] == role:  # the API wants alternating turns
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        body: dict[str, Any] = {"contents": contents}
        if system:
            if self.system_instruction_ok:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            else:
                first = next((c for c in contents if c["role"] == "user"), None)
                if first is None:
                    first = {"role": "user", "parts": []}
                    contents.insert(0, first)
                n_img = sum(1 for p in first["parts"] if "inline_data" in p)  # keep images first
                first["parts"].insert(n_img, {"text": system})
        gen: dict[str, Any] = {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_tokens}
        if json_schema is not None and self.json_level != "prompt":
            gen["responseMimeType"] = "application/json"
            if self.json_level == "json_schema":
                gen["responseJsonSchema"] = json_schema
        thinking = self.thinking_default if thinking is None else thinking
        if thinking is not None and self.thinking_ok:
            gen["thinkingConfig"] = {"thinkingLevel": "high" if thinking else "minimal"}
        body["generationConfig"] = gen
        return body

    def parse_response(self, data: dict[str, Any], seconds: float = 0.0) -> LLMResponse:
        if data.get("error"):
            err = data["error"]
            raise LLMError(f"gemini: {err.get('message', err) if isinstance(err, dict) else err}")
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback") or {}
            raise LLMError(f"gemini[{self.model}]: no candidates (blockReason={feedback.get('blockReason')}): "
                           f"{json.dumps(data, ensure_ascii=False)[:500]}")
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        reasoning = "".join(p.get("text", "") for p in parts if p.get("thought"))
        finish = cand.get("finishReason")
        if not text and finish in _BLOCKED:
            raise LLMError(f"gemini[{self.model}]: answer blocked (finishReason={finish})")
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            reasoning=reasoning,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0),
            seconds=seconds,
            model=data.get("modelVersion") or self.model,
            finish_reason=finish,
            raw=data,
        )

    def _fallback(self, exc: LLMHTTPError, *, json_requested: bool, system_used: bool, thinking_used: bool) -> bool:
        """Drop the feature an HTTP 400 complains about; False when nothing can be dropped."""
        if exc.status_code != 400:
            return False
        msg = exc.body.lower()
        with self._lock:
            if system_used and self.system_instruction_ok and ("system" in msg or "developer instruction" in msg):
                self.system_instruction_ok = False
                what = "systemInstruction (prepending the system prompt to the first user turn)"
            elif thinking_used and self.thinking_ok and "thinking" in msg:
                self.thinking_ok = False
                what = "thinkingConfig"
            elif json_requested and self.json_level != "prompt" and any(k in msg for k in ("json", "schema", "mime")):
                old = self.json_level
                self.json_level = JSON_LEVELS[JSON_LEVELS.index(old) + 1]
                what = f"{old} structured output (now {self.json_level})"
            else:
                return False
        log.warning("Gemini API rejected a request feature (%s); dropping %s for this and later calls",
                    exc.body[:200], what)
        return True

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Optional[dict[str, Any]] = None,
        schema_name: str = "output",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        thinking: Optional[bool] = None,
    ) -> LLMResponse:
        has_system = any(m.role == "system" and m.text() for m in messages)
        while True:
            body = self.build_request(messages, json_schema=json_schema, temperature=temperature, top_p=top_p,
                                      max_tokens=max_tokens, thinking=thinking)
            t0 = time.time()
            try:
                data = post_json(self._http, self.url, body, max_retries=self.max_retries, backoff=self.backoff,
                                 sleep=self._sleep, what=f"gemini[{self.model}]")
            except LLMHTTPError as exc:
                if self._fallback(exc, json_requested=json_schema is not None,
                                  system_used="systemInstruction" in body and has_system,
                                  thinking_used="thinkingConfig" in body["generationConfig"]):
                    continue
                raise
            return self.parse_response(data, time.time() - t0)
