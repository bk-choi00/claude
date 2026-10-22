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
  ``chat_template_kwargs: {"enable_thinking": bool}`` (vLLM, SGLang, llama.cpp; what
  the OpenAI SDK's ``extra_body`` produces) or, for Ollama ``/v1``, with
  ``reasoning_effort`` (``"none"`` = off, a level such as ``"high"`` = on; Ollama
  ignores ``chat_template_kwargs``); thoughts come back in ``message.reasoning``
  (vLLM, Ollama) or ``message.reasoning_content`` (SGLang, llama.cpp);
* retries with exponential backoff on 408/429/5xx, timeouts and connection errors;
* credentials never reach error texts or logs: the API key (stripped of stray whitespace when the
  backend is built; keys with control characters inside are rejected) and any user info / query
  of the URL are replaced by ``***`` (:func:`redact`, :func:`redact_url`), and raw ``repr``s of
  httpx protocol errors (which quote header values) are never interpolated.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Iterable, Optional, Sequence
from urllib.parse import parse_qsl, urlparse, urlsplit, urlunsplit

import httpx

from .base import LLMBackend, LLMError, LLMResponse, Message
from .jsonutil import strip_thinking

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_BACKOFF = 60.0
RESPONSE_FORMAT_MODES = ("json_schema", "json_object", "prompt")
# How the Gemma 4 thinking switch is sent: vLLM / SGLang / llama.cpp read the chat-template
# kwarg; Ollama's /v1 maps ``reasoning_effort`` to its ``think`` option instead.
THINKING_PARAMS = ("chat_template_kwargs", "reasoning_effort")
OLLAMA_DEFAULT_PORT = 11434
# Words that identify a 400/422 as a complaint about structured output (vLLM, SGLang,
# llama.cpp, LM Studio, Ollama error texts).
_RF_HINTS = ("response_format", "json_schema", "json_object", "guided", "grammar", "structured",
             "xgrammar", "outlines", "schema")
_THOUGHT = re.compile(r"<\|channel\|?>thought(.*?)<channel\|>", re.S)
REDACTED = "***"
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s?#]+@")
_MIN_SECRET = 4  # shorter values are not redacted (they would mangle ordinary words)


# --------------------------------------------------------------------------- credentials


def redact_url(url: Any) -> str:
    """``url`` without user info, query and fragment (where gateways put credentials)."""
    text = str(url)
    try:
        parts = urlsplit(text)
    except ValueError:  # malformed (e.g. a bad IPv6 host): strip what looks like credentials
        return _URL_USERINFO.sub(r"\1", text).split("?", 1)[0].split("#", 1)[0]
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def url_secrets(url: Any) -> list[str]:
    """Credential-like values inside a URL: the password (or a lone user name used as a token) and
    every query value."""
    out: list[str] = []
    try:
        parts = urlsplit(str(url))
        user, password = parts.username, parts.password
        query = parse_qsl(parts.query, keep_blank_values=False)
    except ValueError:
        return out
    if password:
        out.append(password)
    elif user:
        out.append(user)
    out.extend(v for _, v in query)
    return [v for v in out if len(v) >= _MIN_SECRET]


def collect_secrets(*values: Optional[str]) -> tuple[str, ...]:
    """Distinct secret strings (raw and stripped), longest first (so a key is replaced before its parts)."""
    seen: set[str] = set()
    for v in values:
        if not v:
            continue
        for form in (v, v.strip()):
            if len(form) >= _MIN_SECRET:
                seen.add(form)
    return tuple(sorted(seen, key=len, reverse=True))


def redact(text: Any, secrets: Iterable[str] = ()) -> str:
    """``text`` with every secret and every ``scheme://user:pass@`` credential replaced by ``***``."""
    out = str(text)
    for secret in sorted({s for s in secrets if s and len(s) >= _MIN_SECRET}, key=len, reverse=True):
        out = out.replace(secret, REDACTED)
        esc = secret.encode("unicode_escape").decode("ascii")  # as quoted inside a repr
        if esc != secret:
            out = out.replace(esc, REDACTED)
    return _URL_USERINFO.sub(r"\1" + REDACTED + "@", out)


def clean_api_key(api_key: Optional[str], what: str = "api_key") -> Optional[str]:
    """The API key without surrounding whitespace (a YAML block scalar or a CRLF env file adds a
    line break); ``None`` when empty. A key with control or non-ASCII characters inside cannot be
    sent as an HTTP header and is rejected with an error that does not quote it."""
    if api_key is None:
        return None
    key = str(api_key).strip()
    if not key:
        return None
    bad = [c for c in key if ord(c) < 32 or ord(c) == 127 or ord(c) > 126]
    if bad:
        kinds = sorted({"control character" if ord(c) < 32 or ord(c) == 127 else "non-ASCII character" for c in bad})
        raise LLMError(f"{what} contains {len(bad)} {' / '.join(kinds)}(s) inside the key (length {len(key)}); "
                       "it cannot be sent as an HTTP header - check the configured value (not shown)")
    return key


def join_url(base: str, suffix: str) -> str:
    """``base`` + ``suffix`` with any query string of ``base`` kept at the end."""
    if "?" not in base:
        return base + suffix
    head, query = base.split("?", 1)
    return f"{head.rstrip('/')}{suffix}?{query}"


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
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """POST ``body`` as JSON and return the decoded response.

    Retries ``max_retries`` times with exponential backoff (honouring
    ``Retry-After``) on 408/429/5xx, timeouts and connection errors; raises
    :class:`LLMHTTPError` / :class:`LLMError` with the server's message otherwise.
    Error texts show the URL without credentials and with ``secrets`` (API keys) and URL
    credentials replaced by ``***``.
    """
    secrets = tuple(secrets) + tuple(url_secrets(url))
    shown = redact_url(url)

    def clean(text: Any) -> str:
        return redact(text, secrets)

    attempt = 0
    while True:
        delay: Optional[float] = None
        try:
            resp = http.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            err: LLMError = LLMError(f"{what}: request to {shown} timed out ({type(exc).__name__}); "
                                     "increase the timeout for long multimodal prompts")
        except httpx.LocalProtocolError as exc:  # its text quotes the offending header value (the API key)
            raise LLMError(f"{what}: the request to {shown} could not be sent ({type(exc).__name__}: illegal "
                           "header value or URL); check api_key and base_url for stray characters") from None
        except httpx.TransportError as exc:
            err = LLMError(f"{what}: cannot reach {shown}: {type(exc).__name__}: {clean(exc)[:300]}")
        except (httpx.InvalidURL, UnicodeError) as exc:
            raise LLMError(f"{what}: invalid request to {shown} ({type(exc).__name__}); check base_url and "
                           "api_key") from None
        else:
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except ValueError:
                    raise LLMError(f"{what}: non-JSON response from {shown} (HTTP {resp.status_code}): "
                                   f"{clean(repr(resp.text[:500]))}") from None
                if not isinstance(data, dict):
                    raise LLMError(f"{what}: unexpected response from {shown}: {clean(str(data)[:500])}")
                return data
            text = clean(error_text(resp))
            err = LLMHTTPError(f"{what}: HTTP {resp.status_code} from {shown}: {text}", resp.status_code, text)
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
    """Strip a trailing ``/chat/completions`` and add ``/v1`` to a bare ``host:port`` (a query
    string, e.g. a gateway's ``?api-version=...``, is kept at the end)."""
    url = base_url.strip()
    query = ""
    if "?" in url:
        url, query = url.split("?", 1)
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not urlparse(url).path:
        url += "/v1"
    return f"{url}?{query}" if query else url


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


def default_thinking_param(name: str, base_url: str) -> str:
    """``reasoning_effort`` for Ollama (by backend name or its default port 11434), else the template kwarg."""
    try:
        port = urlparse(base_url).port
    except ValueError:  # malformed port: let the HTTP layer report it
        port = None
    if name == "ollama" or port == OLLAMA_DEFAULT_PORT:
        return "reasoning_effort"
    return "chat_template_kwargs"


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
        thinking_param: Optional[str] = None,
        reasoning_effort: str = "high",
    ):
        super().__init__(model)
        if name:
            self.name = name
        if response_format not in RESPONSE_FORMAT_MODES:
            raise ValueError(f"response_format must be one of {RESPONSE_FORMAT_MODES}, got {response_format!r}")
        self.base_url = normalize_base_url(base_url)
        thinking_param = thinking_param or default_thinking_param(self.name, self.base_url)
        if thinking_param not in THINKING_PARAMS:
            raise ValueError(f"thinking_param must be one of {THINKING_PARAMS}, got {thinking_param!r}")
        if reasoning_effort == "none":
            raise ValueError("reasoning_effort is the level used when thinking is ON; 'none' switches it off")
        self.thinking_param = thinking_param
        self.reasoning_effort = reasoning_effort
        self._secrets = collect_secrets(api_key, *url_secrets(self.base_url))
        api_key = clean_api_key(api_key)
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.top_k = top_k
        self.extra_body = dict(extra_body or {})
        self.thinking_default = thinking_default
        self.backoff = backoff
        self.response_format_mode = response_format  # downgraded on HTTP 400/422 about response_format
        self.configured_response_format = response_format
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
        return join_url(self.base_url, "/chat/completions")

    @property
    def display_url(self) -> str:
        """The endpoint without credentials (for reports and logs)."""
        return redact_url(self.url)

    @property
    def structured_output(self) -> str:
        """Current structured-output mode: json_schema | json_object | prompt."""
        return self.response_format_mode

    def redact(self, text: Any) -> str:
        """``text`` with this backend's API key and URL credentials replaced by ``***``."""
        return redact(text, self._secrets)

    def cache_fingerprint(self) -> dict[str, Any]:
        """Request settings beyond the client's that change the answer (replay-cache key). The
        configured structured-output mode is used, not the downgraded one, so a replay computes
        the same keys as the run that filled the cache."""
        return {"base_url": redact_url(self.base_url), "top_k": self.top_k, "extra_body": self.extra_body,
                "response_format": self.configured_response_format, "thinking_param": self.thinking_param,
                "reasoning_effort": self.reasoning_effort if self.thinking_param == "reasoning_effort" else None}

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
            if self.thinking_param == "reasoning_effort":  # Ollama /v1: "none" = off
                body["reasoning_effort"] = self.reasoning_effort if thinking else "none"
            else:
                body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}),
                                                "enable_thinking": bool(thinking)}
        return body

    def parse_response(self, data: dict[str, Any], seconds: float = 0.0) -> LLMResponse:
        if data.get("error"):
            err = data["error"]
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise LLMError(self.redact(f"{self.name}: server error: {msg}"))
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(self.redact(f"{self.name}: response without choices: "
                                       f"{json.dumps(data, ensure_ascii=False)[:800]}"))
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
                    redact_url(self.base_url), failed_mode, exc.status_code, exc.body[:200], nxt)

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
                                 sleep=self._sleep, what=f"{self.name}[{self.model}]", secrets=self._secrets)
            except LLMHTTPError as exc:
                rejected_format = (json_schema is not None and mode != "prompt" and exc.status_code in (400, 422)
                                   and any(h in exc.body.lower() for h in _RF_HINTS))
                if rejected_format:
                    self._downgrade(mode, exc)
                    continue
                raise
            return self.parse_response(data, time.time() - t0)
