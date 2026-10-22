"""OpenAI-compatible ``/chat/completions`` backend for Gemma 4 31B.

Works with vLLM, SGLang, llama.cpp ``llama-server``, LM Studio and Ollama's
``/v1`` endpoint. Request conventions (see docs/RESEARCH.md):

* user content is a list of parts with the images FIRST, as base64 data URLs
  (``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``;
  Ollama accepts data URLs only), followed by one text part;
* structured output via ``response_format`` -- ``json_schema`` (guided decoding),
  downgraded to ``json_object`` and then to schema-in-prompt when the server
  answers HTTP 400/422 about the response format; the downgrade is remembered;
* Gemma 4 thinking is toggled per request with the top-level
  ``chat_template_kwargs: {"enable_thinking": bool}`` (what the OpenAI SDK's
  ``extra_body`` produces); thoughts come back in ``message.reasoning`` (vLLM) or
  ``message.reasoning_content`` (SGLang, llama.cpp);
* retries with exponential backoff on 408/429/5xx, timeouts and connection errors.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urlparse

import httpx

from .base import LLMBackend, LLMError, LLMResponse, Message
from .jsonutil import strip_thinking

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_BACKOFF = 60.0
RESPONSE_FORMAT_MODES = ("json_schema", "json_object", "prompt")
# Words that identify a 400/422 as a complaint about structured output (vLLM, SGLang,
# llama.cpp, LM Studio, Ollama error texts).
_RF_HINTS = ("response_format", "json_schema", "json_object", "guided", "grammar", "structured",
             "xgrammar", "outlines", "schema")
_THOUGHT = re.compile(r"<\|channel\|?>thought(.*?)<channel\|>", re.S)


class LLMHTTPError(LLMError):
    """Non-retryable (or retries exhausted) HTTP error; keeps status and body for fallbacks."""

    def __init__(self, message: str, status_code: int, body: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def error_text(resp: httpx.Response, limit: int = 1500) -> str:
    """The server's error message (JSON ``error.message`` / ``detail`` / raw text), truncated."""
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:limit].strip()
    msg: Any = data
    if isinstance(data, dict):
        err = data.get("error", data.get("detail", data.get("message")))
        if isinstance(err, dict):
            msg = err.get("message") or err
        elif err is not None:
            msg = err
    text = msg if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False)
    return text[:limit].strip()


def _retry_after(resp: httpx.Response) -> Optional[float]:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return min(MAX_BACKOFF, max(0.0, float(value)))
    except ValueError:
        return None


def post_json(
    http: httpx.Client,
    url: str,
    body: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    max_retries: int = 2,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "LLM",
) -> dict[str, Any]:
    """POST ``body`` as JSON and return the decoded response.

    Retries ``max_retries`` times with exponential backoff (honouring
    ``Retry-After``) on 408/429/5xx, timeouts and connection errors; raises
    :class:`LLMHTTPError` / :class:`LLMError` with the server's message otherwise.
    """
    attempt = 0
    while True:
        delay: Optional[float] = None
        try:
            resp = http.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            err: LLMError = LLMError(f"{what}: request to {url} timed out ({type(exc).__name__}); "
                                     "increase the timeout for long multimodal prompts")
        except httpx.TransportError as exc:
            err = LLMError(f"{what}: cannot reach {url}: {exc!r}")
        else:
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except ValueError:
                    raise LLMError(f"{what}: non-JSON response from {url} (HTTP {resp.status_code}): "
                                   f"{resp.text[:500]!r}") from None
                if not isinstance(data, dict):
                    raise LLMError(f"{what}: unexpected response from {url}: {str(data)[:500]}")
                return data
            text = error_text(resp)
            err = LLMHTTPError(f"{what}: HTTP {resp.status_code} from {url}: {text}", resp.status_code, text)
            if resp.status_code not in RETRY_STATUS:
                raise err
            delay = _retry_after(resp)
        if attempt >= max_retries:
            raise err
        delay = delay if delay is not None else min(MAX_BACKOFF, backoff * (2 ** attempt))
        log.warning("%s (attempt %d/%d), retrying in %.1fs", err, attempt + 1, max_retries + 1, delay)
        sleep(delay)
        attempt += 1


def normalize_base_url(base_url: str) -> str:
    """Strip a trailing ``/chat/completions`` and add ``/v1`` to a bare ``host:port``."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not urlparse(url).path:
        url += "/v1"
    return url


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".localhost")


def schema_note(schema: dict[str, Any]) -> str:
    return ("\n\nRespond with a single JSON object only (no prose, no markdown fences) that validates "
            "against this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False))


def ensure_schema_in_prompt(messages: Sequence[Message], schema: dict[str, Any]) -> list[Message]:
    """Append the schema to the last user turn unless some message already carries it."""
    dumped = json.dumps(schema, ensure_ascii=False)
    if any(dumped in m.text() for m in messages):
        return list(messages)
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == "user":
            out[i] = Message("user", out[i].parts() + [schema_note(schema)])
            return out
    return out + [Message("user", schema_note(schema).strip())]


def image_captions(msg: Message) -> list[str]:
    return [f"Image {i}: {im.label}" for i, im in enumerate(msg.images(), 1) if im.label]


class OpenAICompatBackend(LLMBackend):
    """Gemma 4 behind any OpenAI-compatible chat-completions server."""

    name = "openai"
    supports_images = True
    supports_json_schema = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        top_k: Optional[int] = None,
        extra_body: Optional[dict[str, Any]] = None,
        transport: Optional[httpx.BaseTransport] = None,
        *,
        name: Optional[str] = None,
        response_format: str = "json_schema",
        thinking_default: Optional[bool] = None,
        backoff: float = 1.0,
    ):
        super().__init__(model)
        if name:
            self.name = name
        if response_format not in RESPONSE_FORMAT_MODES:
            raise ValueError(f"response_format must be one of {RESPONSE_FORMAT_MODES}, got {response_format!r}")
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.top_k = top_k
        self.extra_body = dict(extra_body or {})
        self.thinking_default = thinking_default
        self.backoff = backoff
        self.response_format_mode = response_format  # downgraded on HTTP 400/422 about response_format
        self._sleep: Callable[[float], None] = time.sleep
        self._lock = threading.Lock()
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 30.0)),
            transport=transport,
            headers=headers,
            trust_env=not is_local_url(self.base_url),  # never send localhost traffic through a proxy
        )

    # -- lifecycle -----------------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "OpenAICompatBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def structured_output(self) -> str:
        """Current structured-output mode: json_schema | json_object | prompt."""
        return self.response_format_mode

    # -- request / response ----------------------------------------------------------
    @staticmethod
    def convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.role if m.role in ("system", "user", "assistant") else "user"
            images = m.images()
            text = "\n".join(image_captions(m) + ([m.text()] if m.text() else []))
            if not images:
                out.append({"role": role, "content": text})
                continue
            parts: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": im.data_url()}} for im in images]
            if text:
                parts.append({"type": "text", "text": text})
            out.append({"role": role, "content": parts})
        return out

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Optional[dict[str, Any]] = None,
        schema_name: str = "output",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        thinking: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """The JSON body for one chat completion (``mode`` overrides the remembered format)."""
        mode = mode or self.response_format_mode
        msgs = list(messages)
        if json_schema is not None and mode != "json_schema":
            msgs = ensure_schema_in_prompt(msgs, json_schema)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.convert_messages(msgs),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if json_schema is not None:
            if mode == "json_schema":
                name = re.sub(r"[^A-Za-z0-9_-]", "_", schema_name)[:64] or "output"
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": name, "schema": json_schema, "strict": True}}
            elif mode == "json_object":
                body["response_format"] = {"type": "json_object"}
        for key, value in self.extra_body.items():
            if isinstance(value, dict) and isinstance(body.get(key), dict):
                body[key] = {**body[key], **value}
            else:
                body[key] = value
        thinking = self.thinking_default if thinking is None else thinking
        if thinking is not None:
            body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": bool(thinking)}
        return body

    def parse_response(self, data: dict[str, Any], seconds: float = 0.0) -> LLMResponse:
        if data.get("error"):
            err = data["error"]
            raise LLMError(f"{self.name}: server error: {err.get('message', err) if isinstance(err, dict) else err}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.name}: response without choices: {json.dumps(data, ensure_ascii=False)[:800]}")
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = content or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if not reasoning and "<|channel" in text:  # no reasoning parser on the server: split inline thoughts
            reasoning = "\n".join(m.strip() for m in _THOUGHT.findall(text))
            text = strip_thinking(text)
        finish = choice.get("finish_reason")
        if not text.strip() and finish == "length":
            log.warning("%s: empty answer with finish_reason=length (thinking may have used the whole "
                        "max_tokens budget)", self.name)
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            reasoning=reasoning,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            seconds=seconds,
            model=data.get("model") or self.model,
            finish_reason=finish,
            raw=data,
        )

    def _downgrade(self, failed_mode: str, exc: LLMHTTPError) -> None:
        nxt = RESPONSE_FORMAT_MODES[RESPONSE_FORMAT_MODES.index(failed_mode) + 1]
        with self._lock:
            if self.response_format_mode == failed_mode:
                self.response_format_mode = nxt
        log.warning("%s rejected response_format=%s (HTTP %d: %s); falling back to %s for this and later calls",
                    self.base_url, failed_mode, exc.status_code, exc.body[:200], nxt)

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
        while True:
            mode = self.response_format_mode
            body = self.build_request(messages, json_schema=json_schema, schema_name=schema_name,
                                      temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                      thinking=thinking, mode=mode)
            t0 = time.time()
            try:
                data = post_json(self._http, self.url, body, max_retries=self.max_retries, backoff=self.backoff,
                                 sleep=self._sleep, what=f"{self.name}[{self.model}]")
            except LLMHTTPError as exc:
                rejected_format = (json_schema is not None and mode != "prompt" and exc.status_code in (400, 422)
                                   and any(h in exc.body.lower() for h in _RF_HINTS))
                if rejected_format:
                    self._downgrade(mode, exc)
                    continue
                raise
            return self.parse_response(data, time.time() - t0)
